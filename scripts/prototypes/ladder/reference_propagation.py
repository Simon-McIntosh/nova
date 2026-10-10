"""One-edge-per-round component labels for a timing control."""

import jax
import jax.numpy as jnp


def one_hop_labels(area, intervals, geometry, tolerance):
    cells, capacity = area.shape
    edge_interval = jnp.transpose(intervals, (0, 2, 1, 3, 4))
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_interval = edge_interval[peer, geometry.neighbour_edge]
    peer_interval = 1.0 - peer_interval[..., ::-1]
    overlap = jnp.minimum(
        edge_interval[:, :, :, None, :, None, 1],
        peer_interval[:, :, None, :, None, :, 1],
    ) - jnp.maximum(
        edge_interval[:, :, :, None, :, None, 0],
        peer_interval[:, :, None, :, None, :, 0],
    )
    links = (
        jnp.any(overlap > tolerance, axis=(-2, -1))
        & (geometry.neighbour >= 0)[..., None, None]
    )
    sentinel = cells * capacity
    initial = jnp.where(
        area > 0.0, jnp.arange(sentinel).reshape(cells, capacity), sentinel
    )

    def condition(state):
        iteration, _labels, changed = state
        return changed & (iteration < sentinel)

    def propagate(state):
        iteration, labels, _changed = state
        incoming = labels[peer]
        next_label = jnp.minimum(
            labels,
            jnp.min(jnp.where(links, incoming[:, :, None, :], sentinel), axis=(1, 3)),
        )
        return iteration + 1, next_label, jnp.any(next_label != labels)

    iteration, labels, changed = jax.lax.while_loop(
        condition, propagate, (jnp.asarray(0), initial, jnp.asarray(True))
    )
    wall = jnp.any(
        (geometry.neighbour < 0)[:, :, None, None]
        & (edge_interval[..., 1] - edge_interval[..., 0] > tolerance),
        axis=(1, 3),
    )
    return labels, wall, ~changed, iteration
