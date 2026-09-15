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

from functools import partial
import math

import jax
import jax.numpy as jnp


__all__ = [
    "label_parallel_connected_components",
    "label_parallel_connected_components_with_steps",
    "label_parallel_graph_components",
    "label_parallel_graph_components_with_steps",
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


@partial(jax.jit, static_argnums=(3,))
def label_parallel_graph_components_with_steps(
    confined: jax.Array,
    neighbours: jax.Array,
    neighbour_admissible: jax.Array,
    n_iter: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return exact labels over an explicit centre-first neighbour list.

    ``neighbours`` is a two-dimensional integer table whose first column is a
    centre vertex and whose remaining columns are its neighbours.
    ``neighbour_admissible`` has the same shape and masks graph links; column
    zero is padding and does not create an edge. The vertices themselves have
    the fixed shape of ``confined``.

    Each round hooks the current parent of every vertex toward the smallest
    grandparent visible on an admissible edge, then pointer-jumps every parent.
    A hook crosses a real foreground edge and a jump follows existing tree
    edges, so parents never cross connected components. Parent indices decrease,
    leaving the minimum-index vertex as the immutable root. At the fixed point
    every edge endpoint has that same root, making the result bit-identical to
    canonical minimum-label propagation.

    ``n_iter`` remains the caller's static upper bound, while the compiled pass
    needs at most ``ceil(log2(cell_count)) + 2`` hook-and-compress trips. The
    returned ``settled`` flag is false if a smaller caller cap prevents the
    confirming trip.
    """
    if neighbours.ndim != 2:
        raise ValueError("neighbours must be a two-dimensional table")
    if neighbours.shape != neighbour_admissible.shape:
        raise ValueError("neighbours and neighbour_admissible must have equal shape")

    cell_count = confined.size
    trip_limit = min(max(n_iter, 0), math.ceil(math.log2(max(cell_count, 1))) + 2)
    foreground = confined.reshape(-1)
    centre = neighbours[:, :1]
    adjacent = neighbours[:, 1:]
    edge_valid = neighbour_admissible[:, 1:] & foreground[centre] & foreground[adjacent]
    vertex = jnp.arange(cell_count, dtype=jnp.int32)
    sentinel = jnp.asarray(cell_count, dtype=jnp.int32)

    def advance(parents):
        grandparents = parents[parents]
        edge_minimum = jnp.where(
            edge_valid,
            jnp.minimum(grandparents[centre], grandparents[adjacent]),
            sentinel,
        )
        neighbour_minimum = jnp.full(cell_count, sentinel, dtype=jnp.int32)
        neighbour_minimum = neighbour_minimum.at[centre].min(
            jnp.min(edge_minimum, axis=1, keepdims=True)
        )
        neighbour_minimum = neighbour_minimum.at[adjacent].min(edge_minimum)
        hook = foreground & (neighbour_minimum < grandparents)

        root_candidate = jnp.where(hook, neighbour_minimum, sentinel)
        hooked = parents.at[parents].min(root_candidate)
        hooked = jnp.where(hook, jnp.minimum(hooked, neighbour_minimum), hooked)
        hooked = jnp.minimum(hooked, grandparents)
        hooked = jnp.where(foreground, hooked, vertex)
        return hooked[hooked]

    def body(_trip, state):
        parents, steps = state
        next_parents = advance(parents)
        changed = jnp.any(next_parents != parents)
        return next_parents, steps + changed.astype(jnp.int32)

    parents, steps = jax.lax.fori_loop(
        0,
        trip_limit,
        body,
        (
            vertex,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )

    def jump(_trip, current):
        return current[current]

    roots = jax.lax.fori_loop(
        0,
        math.ceil(math.log2(max(cell_count, 1))),
        jump,
        parents,
    )
    roots = roots[roots]
    labels = jnp.where(foreground, roots + 1, 0).reshape(confined.shape)
    settled = ~jnp.any(advance(parents) != parents)
    return labels, steps, settled


@partial(jax.jit, static_argnums=(3,))
def label_parallel_graph_components(
    confined: jax.Array,
    neighbours: jax.Array,
    neighbour_admissible: jax.Array,
    n_iter: int,
) -> jax.Array:
    """Return canonical labels over an explicit centre-first neighbour list."""
    labels, _steps, _settled = label_parallel_graph_components_with_steps(
        confined, neighbours, neighbour_admissible, n_iter
    )
    return labels


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
