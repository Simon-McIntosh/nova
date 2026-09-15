"""Fixed-shape parallel connected components on rectangular masks.

The kernel implements the hook-and-compress connected-components scheme.  A
cell begins as a one-vertex tree.  Each round finds the smallest grandparent
visible across its four-neighbour edges, hooks both the cell and its current
parent toward that representative, then pointer-jumps every parent to its
grandparent.  All arrays retain the input shape; only reductions and indexed
minimum updates cross cells, so the kernel is suitable for ``jit`` and
``vmap`` on an accelerator.

Why the fixed point is the canonical flood label follows from two invariants.
First, a hook crosses a real foreground edge and a pointer jump follows two
existing tree edges, so a parent never leaves its original connected
component.  Second, every update is a minimum, so parent indices decrease and
the minimum-index cell of a component remains its own root.  At a fixed point
the endpoints of every foreground edge have the same root; by induction along
paths, every cell in a component therefore has one root.  The unchanged
minimum-index root makes that common value exactly the one-based minimum
reachable index returned by the canonical fixed-point flood.

The schedule uses ``ceil(log2(cell_count)) + 2`` hook-and-compress trips.  A
hook joins adjacent trees, while the following pointer jump replaces every
remaining parent path by paths at most half as long.  Thus after ``k`` jumps a
parent spans at least ``2**k`` vertices of its pre-compression path; no forest
on ``cell_count`` vertices needs more than ``ceil(log2(cell_count))`` jumps.
One leading hook exposes cross-tree edges and one trailing trip confirms the
fixed point, giving the stated bound.  The returned ``settled`` bit makes that
proof obligation executable: callers measuring a new lattice must require it,
never interpret a capped, unsettled label field as a component result.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


__all__ = [
    "label_parallel_connected_components",
    "label_parallel_connected_components_with_steps",
]


def _neighbour_table(shape: tuple[int, int]) -> tuple[jax.Array, jax.Array]:
    """Return clipped four-neighbour indices and their geometric validity."""
    vertical_count, radial_count = shape
    index = jnp.arange(vertical_count * radial_count, dtype=jnp.int32).reshape(shape)
    row = jnp.arange(vertical_count, dtype=jnp.int32)[:, None]
    column = jnp.arange(radial_count, dtype=jnp.int32)[None, :]
    neighbours = jnp.stack(
        (
            jnp.maximum(row - 1, 0) * radial_count + column,
            jnp.minimum(row + 1, vertical_count - 1) * radial_count + column,
            row * radial_count + jnp.maximum(column - 1, 0),
            row * radial_count + jnp.minimum(column + 1, radial_count - 1),
        ),
        axis=-1,
    ).reshape((-1, 4))
    valid = jnp.stack(
        (
            jnp.broadcast_to(row > 0, shape),
            jnp.broadcast_to(row + 1 < vertical_count, shape),
            jnp.broadcast_to(column > 0, shape),
            jnp.broadcast_to(column + 1 < radial_count, shape),
        ),
        axis=-1,
    ).reshape((-1, 4))
    return neighbours, valid & (neighbours != index.reshape((-1, 1)))


@jax.jit
def label_parallel_connected_components_with_steps(
    confined: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return canonical labels, active trips, and a fixed-point confirmation.

    ``confined`` is a two-dimensional boolean mask.  Foreground labels are the
    one-based flat index of their component's minimum cell; background labels
    are zero.  ``steps`` counts changing hook-and-compress trips.  ``settled``
    is true only when one additional trip would leave every grandparent fixed.
    """
    if confined.ndim != 2:
        raise ValueError("confined must be a two-dimensional mask")

    shape = confined.shape
    cell_count = confined.size
    trip_limit = math.ceil(math.log2(max(cell_count, 1))) + 2
    neighbours, geometrically_valid = _neighbour_table(shape)
    foreground = confined.reshape(-1)
    edge_valid = geometrically_valid & foreground[:, None] & foreground[neighbours]
    vertex = jnp.arange(cell_count, dtype=jnp.int32)
    sentinel = jnp.asarray(cell_count, dtype=jnp.int32)

    def body(_trip, state):
        parents, grandparents, previous, steps = state
        active = jnp.any(grandparents != previous)
        adjacent = jnp.where(edge_valid, grandparents[neighbours], sentinel)
        neighbour_minimum = jnp.min(adjacent, axis=1)
        hook = foreground & (neighbour_minimum < grandparents)

        root_candidate = jnp.where(hook, neighbour_minimum, sentinel)
        hooked = parents.at[parents].min(root_candidate)
        hooked = jnp.where(hook, jnp.minimum(hooked, neighbour_minimum), hooked)
        hooked = jnp.minimum(hooked, grandparents)
        hooked = jnp.where(foreground, hooked, vertex)
        next_grandparents = hooked[hooked]
        return (
            jnp.where(active, hooked, parents),
            jnp.where(active, next_grandparents, grandparents),
            jnp.where(active, grandparents, previous),
            steps + active.astype(jnp.int32),
        )

    parents, grandparents, previous, steps = jax.lax.fori_loop(
        0,
        trip_limit,
        body,
        (
            vertex,
            vertex,
            jnp.full_like(vertex, -1),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    roots = parents

    def jump(_trip, current):
        return current[current]

    roots = jax.lax.fori_loop(0, math.ceil(math.log2(max(cell_count, 1))), jump, roots)
    roots = roots[roots]
    labels = jnp.where(foreground, roots + 1, 0).reshape(shape)
    settled = ~jnp.any(grandparents != previous)
    return labels, steps, settled


@jax.jit
def label_parallel_connected_components(confined: jax.Array) -> jax.Array:
    """Return canonical minimum-index labels for a rectangular confined mask."""
    labels, _steps, _settled = label_parallel_connected_components_with_steps(confined)
    return labels
