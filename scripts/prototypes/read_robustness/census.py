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
    offset = -jnp.linalg.solve(safe, coefficients[:, 1:3, None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * offset
    slot = jnp.arange(geometry.vertices.shape[1])
    following = jnp.where(
        slot[None] + 1 < geometry.vertex_count[:, None], slot[None] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    cross = topology._cross_plane(
        end - geometry.vertices, seeds[:, None] - geometry.vertices
    )
    inside = jnp.all(
        jnp.where(slot[None] < geometry.vertex_count[:, None], cross >= 0, True),
        axis=1,
    )
    candidate = conditioned & inside & jnp.all(jnp.isfinite(seeds), axis=1)
    seed_capacity = max(4 * policy.null_capacity, previous.shape[0])
    selected = jnp.nonzero(candidate, size=seed_capacity, fill_value=0)[0]
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
    saddle_slots = jnp.nonzero(saddles, size=policy.null_capacity - 1, fill_value=0)[0]
    gather = jnp.concatenate((axis_index[None], saddle_slots))
    valid = jnp.concatenate(
        (
            jnp.any(axes)[None],
            jnp.arange(policy.null_capacity - 1) < saddle_count,
        )
    )
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
        | (saddle_count > policy.null_capacity - 1),
        distance[axis_index],
    )
