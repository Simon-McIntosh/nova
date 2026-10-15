"""Read a complete Bell jet map or gather a moving fixed-capacity window."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from nova.equilibrium import fixed_points


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MobileWindow:
    cache: fixed_points.RowCache
    required: jax.Array
    capacity: int
    overflow: jax.Array


def full_window(jets):
    """At coarse resolution every stored node is immediately available."""
    return fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(jets.nodes.shape[0], dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )


def mobile_window(jets, geometry, membership, roots, capacity, margin_pitches=4.0):
    """Gather current cut, wall and root rows with a shape-fixed device index.

    The full jet map remains the source. A caller checks ``overflow`` before
    using the gathered cache; missing rows cannot silently alias slot zero.
    """
    pitch = jnp.median(geometry.pitch)
    distances = jnp.linalg.norm(geometry.centre[:, None] - roots[None], axis=-1)
    near_root = jnp.any(distances <= margin_pitches * pitch, axis=1)
    edge_slot = jnp.arange(geometry.neighbour.shape[1])
    wall = jnp.any(
        (geometry.neighbour < 0) & (edge_slot[None] < geometry.vertex_count[:, None]),
        axis=1,
    )
    boundary = (membership > 0) & (
        (membership < 1 - 1e-14) | (geometry.vertex_count != 6)
    )
    active_cells = boundary | wall | near_root
    node_count = jets.nodes.shape[0]
    sample = jnp.ravel(jets.sample_index)
    active = jnp.repeat(active_cells, jets.sample_index.shape[1])
    required_mask = jnp.zeros(node_count, dtype=bool).at[sample].max(active)
    required = jnp.sum(required_mask, dtype=jnp.int32)
    indices = jnp.nonzero(required_mask, size=capacity, fill_value=0)[0]
    live = jnp.arange(capacity) < required
    lookup = jnp.full(node_count, -1, dtype=jnp.int32)
    lookup = lookup.at[indices].set(jnp.where(live, jnp.arange(capacity), -1))
    cache = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        lookup,
        jets.rows[indices],
        jets.exterior[indices],
    )
    return MobileWindow(cache, required, capacity, required > capacity)
