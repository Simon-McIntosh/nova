"""Census stationary Bell roots from every cell and continue their identities."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import fixed_points, topology


class RootCensus(NamedTuple):
    roots: fixed_points.FixedPointResult
    candidates: jax.Array
    axes: jax.Array
    saddles: jax.Array
    distinct_saddles: jax.Array
    overflow: jax.Array
    axis_distance: jax.Array


def axis_identity(axis, expected, pitch):
    """An accepted Newton root must still be the intended magnetic axis."""
    return jnp.all(jnp.isfinite(axis)) & (jnp.linalg.norm(axis - expected) <= pitch)


def census(field, geometry, convention, policy, previous):
    """Use the fixed cell lattice plus previous roots; retain every distinct saddle.

    Cellwise quadratic stationary points only propose seeds. The final point,
    gradient and Hessian are evaluated from the complete Bell jet field.
    """
    coefficients = field.sampled_coefficient(geometry)
    hessian = jnp.stack(
        (
            jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=-1),
            jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=-1),
        ),
        axis=1,
    )
    conditioned, safe = jax.vmap(topology._conditioned_hessian, in_axes=(0, None))(
        hessian, policy.hessian_tolerance
    )
    gradient = coefficients[:, 1:3]
    offset = -jnp.linalg.solve(safe, gradient[..., None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * offset
    slot = jnp.arange(geometry.vertices.shape[1])
    # A cell fit may place its stationary point across a shared edge. Check
    # its own polygon and the polygons sharing one edge with it: the ring is
    # defined by the carrier's actual connectivity, including trimmed cells.
    ring = jnp.concatenate(
        (jnp.arange(len(seeds))[:, None], geometry.neighbour), axis=1
    )
    ring_live = ring >= 0
    ring = jnp.maximum(ring, 0)
    vertices = geometry.vertices[ring]
    counts = geometry.vertex_count[ring]
    following = jnp.where(slot[None, None] + 1 < counts[..., None], slot + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=2)
    edge = end - vertices
    relative = seeds[:, None, None] - vertices
    cross = topology._cross_plane(edge, relative)
    # A quadratic seed can land on a cell edge, where the cross product above
    # is zero to within its own rounding and its computed sign is set by
    # rounding: the cell then enters the candidate set in one state and leaves
    # it in the next, and a stationary point proposed only by that cell is
    # never polished. Admit a seed whose cross product is negative only beyond
    # the rounding the value actually carries. The bound composes the edge
    # cross product's forward error with uncertainty in the local Hessian
    # solve and its fitted coefficients, all computed from their operands.
    epsilon = jnp.finfo(seeds.dtype).eps
    inverse = jnp.linalg.inv(safe)
    hessian_scale = jnp.sum(jnp.abs(hessian), axis=(-2, -1))
    displacement_scale = jnp.sum(jnp.abs(inverse), axis=(-2, -1)) * (
        hessian_scale * jnp.sum(jnp.abs(offset), axis=-1)
        + jnp.sum(jnp.abs(gradient), axis=-1)
    )
    seed_uncertainty = epsilon * jnp.abs(geometry.pitch) * displacement_scale
    # The quadratic fit is a signed dot product, so cancellation can make its
    # coefficient error much larger than eps times the fitted coefficient.
    # Bound each row by the absolute dot-product terms, then propagate the
    # gradient and Hessian errors through the same local solve as the seed.
    samples = field.jets[field.sample_index, 0]
    centres = jax.vmap(field.cell_value)(jnp.arange(len(samples)), geometry.centre)
    samples = samples.at[:, 0].set(centres)
    operation_count = 2 * samples.shape[1]
    gamma = operation_count * epsilon / (1 - operation_count * epsilon)
    fit_error = gamma * jnp.einsum(
        "nij,nj->ni", jnp.abs(geometry.fit_inverse), jnp.abs(samples)
    )
    hessian_error = jnp.stack(
        (
            jnp.stack((2 * fit_error[:, 3], fit_error[:, 4]), axis=-1),
            jnp.stack((fit_error[:, 4], 2 * fit_error[:, 5]), axis=-1),
        ),
        axis=1,
    )
    fit_seed_uncertainty = (
        jnp.abs(geometry.pitch)
        * jnp.sum(jnp.abs(inverse), axis=(-2, -1))
        * (
            jnp.sum(fit_error[:, 1:3], axis=-1)
            + jnp.sum(jnp.abs(hessian_error), axis=(-2, -1))
            * jnp.sum(jnp.abs(offset), axis=-1)
        )
    )
    rounding = (
        4.0
        * epsilon
        * (jnp.abs(edge[..., 0]) + jnp.abs(edge[..., 1]))
        * (jnp.abs(relative[..., 0]) + jnp.abs(relative[..., 1]))
        + (jnp.abs(edge[..., 0]) + jnp.abs(edge[..., 1]))
        * (seed_uncertainty + fit_seed_uncertainty)[:, None, None]
    )
    inside_ring = (
        jnp.all(
            jnp.where(slot[None, None] < counts[..., None], cross >= -rounding, True),
            axis=2,
        )
        & ring_live
    )
    inside = jnp.any(inside_ring, axis=1)
    candidate = conditioned & inside & jnp.all(jnp.isfinite(seeds), axis=1)
    # A ring can contain one proposal from each of its incident fits. Reserve
    # that many proposal slots for every root slot, then prefer the fits whose
    # seeds are closest to stationary in the full field when a carrier exceeds
    # the reserve. The overflow flag still exposes a truncated proposal set.
    seed_capacity = min(
        len(seeds), max(policy.null_capacity * ring.shape[1], previous.shape[0])
    )
    safe_seeds = jnp.where(candidate[:, None], seeds, geometry.centre)
    residual = jax.vmap(lambda point: jnp.linalg.norm(field.evaluate(point).gradient))(
        safe_seeds
    )
    selected = jnp.argsort(jnp.where(candidate, residual, jnp.inf), stable=True)[
        :seed_capacity
    ]
    live_seed = jnp.arange(seed_capacity) < jnp.sum(candidate)
    all_seeds = jnp.concatenate((previous, seeds[selected]))
    active = jnp.concatenate((jnp.ones(previous.shape[0], bool), live_seed))
    polished = jax.vmap(topology.stationary_read, in_axes=(None, 0, None, None))(
        field, all_seeds, jnp.median(geometry.pitch), policy
    )
    inside_wall = jax.vmap(
        lambda point: jnp.any(topology._inside_cells(geometry, point))
    )(polished.position)
    live = active & polished.valid & inside_wall
    separation = jnp.linalg.norm(
        polished.position[:, None] - polished.position[None], axis=-1
    )
    earlier = jnp.arange(len(live))[None] < jnp.arange(len(live))[:, None]
    duplicate = jnp.any(
        earlier
        & live[None]
        & (separation < 8 * policy.position_tolerance * jnp.min(geometry.pitch)),
        axis=1,
    )
    live &= ~duplicate
    eigenvalues = jnp.linalg.eigvalsh(convention.sigma * polished.jet.hessian)
    axes = live & jnp.all(eigenvalues < 0, axis=1)
    saddles = live & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    distance = jnp.linalg.norm(polished.position - previous[0], axis=-1)
    axis_index = jnp.argmin(jnp.where(axes, distance, jnp.inf))
    saddle_count = jnp.sum(saddles, dtype=jnp.int32)
    slots = policy.null_capacity - 1
    if slots:
        saddle_slots = jnp.nonzero(saddles, size=slots, fill_value=0)[0]
        other_axes = axes & (jnp.arange(len(axes)) != axis_index)
        axis_slots = jnp.nonzero(other_axes, size=slots, fill_value=0)[0]
        remainder = jnp.arange(slots)
        gather = jnp.concatenate(
            (
                axis_index[None],
                jnp.where(
                    remainder < saddle_count,
                    saddle_slots,
                    axis_slots[jnp.clip(remainder - saddle_count, 0, slots - 1)],
                ),
            )
        )
        valid = jnp.concatenate(
            (
                jnp.any(axes)[None],
                remainder < saddle_count + jnp.maximum(jnp.sum(axes) - 1, 0),
            )
        )
    else:
        gather = axis_index[None]
        valid = jnp.any(axes)[None]
    roots = fixed_points.FixedPointResult(
        polished.position[gather],
        jax.tree.map(lambda leaf: leaf[gather], polished.jet),
        valid,
        jnp.where(valid, 0, 3),
        jnp.zeros(policy.null_capacity, jnp.int32),
        jnp.zeros(policy.null_capacity, bool),
    )
    return RootCensus(
        roots,
        jnp.sum(candidate),
        jnp.sum(axes),
        saddle_count,
        saddle_count,
        (jnp.sum(candidate) > seed_capacity)
        | (saddle_count + jnp.sum(axes) > policy.null_capacity),
        distance[axis_index],
    )
