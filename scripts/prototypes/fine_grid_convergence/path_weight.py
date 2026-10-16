"""Continuous edge-passage support for the prototype map."""

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from scripts.prototypes.fast_map_production import kernel
from scripts.prototypes.fast_map_production.kernel import _local_axis_sector


def _continuous_path_weight(
    geometry, coefficients, form, axis, seed, sigma, level, radius, saddle_live
):
    """Propagate the strongest smooth edge passage from the magnetic axis."""
    vertices = geometry.vertices
    count = geometry.vertex_count
    slot = jnp.arange(vertices.shape[1])
    following = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(vertices, following[..., None], axis=1)
    midpoint = (vertices + end) / 2
    local = (vertices - geometry.centre[:, None]) / geometry.pitch[:, None, None]
    local_middle = (midpoint - geometry.centre[:, None]) / geometry.pitch[:, None, None]
    local_end = (end - geometry.centre[:, None]) / geometry.pitch[:, None, None]

    def signed(position):
        x, y = position[..., 0], position[..., 1]
        return sigma * (
            coefficients[:, 0, None]
            + coefficients[:, 1, None] * x
            + coefficients[:, 2, None] * y
            + coefficients[:, 3, None] * x * x
            + coefficients[:, 4, None] * x * y
            + coefficients[:, 5, None] * y * y
        )

    first, middle, last = signed(local), signed(local_middle), signed(local_end)
    curvature = 2 * (first + last - 2 * middle)
    slope = last - first - curvature
    fraction = jnp.clip(-slope / jnp.where(curvature < 0, 2 * curvature, 1), 0, 1)
    peak = jnp.maximum(
        jnp.maximum(first, last),
        signed(local + fraction[..., None] * (local_end - local)),
    )
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_edge = jnp.maximum(geometry.neighbour_edge, 0)
    passage = jnp.minimum(peak, peak[peer, peer_edge])
    # The share of an edge above the boundary level vanishes as its passage
    # closes. Its measured flux excursion supplies the local scale.
    edge_low = jnp.minimum(
        jnp.minimum(first, last),
        jnp.minimum(first[peer, peer_edge], last[peer, peer_edge]),
    )
    flux_excursion = passage - edge_low
    flux_weight = jnp.clip((passage - level) / jnp.maximum(flux_excursion, 1e-14), 0, 1)

    axis_sector = _local_axis_sector(geometry, axis, form)
    offset = midpoint - form.position
    parameter = jnp.einsum("...i,ai->...a", offset, form.direction)
    residual = (
        offset[..., None, :]
        - parameter[..., None] ** 2 * form.curvature
        - parameter[..., None] ** 3 * form.cubic
    )
    side = (
        form.direction[:, 0] * residual[..., 1]
        - form.direction[:, 1] * residual[..., 0]
    )
    next_side = jnp.roll(side, -1, axis=-1)
    sector_margin = jnp.minimum(side[..., axis_sector], -next_side[..., axis_sector])
    length_width = jnp.maximum(1e-7, 1e-4 * jnp.median(geometry.pitch))
    sector_weight = jnp.clip(0.5 + sector_margin / length_width, 0, 1)
    distance = jnp.linalg.norm(geometry.centre - form.position, axis=1)
    near_weight = jnp.clip(0.5 + (radius - distance) / length_width, 0, 1)
    near_weight = jnp.maximum(
        near_weight,
        topology._inside_cells(geometry, form.position).astype(near_weight.dtype),
    )
    near_weight = jnp.where(saddle_live, near_weight, 0)
    frame_weight = 1 - near_weight[:, None] * (1 - sector_weight)
    frame_weight = jnp.minimum(frame_weight, frame_weight[peer, peer_edge])
    edge_weight = jnp.where(
        geometry.neighbour >= 0, jnp.minimum(flux_weight, frame_weight), 0
    )
    strength = (
        jnp.zeros((len(geometry.centre),), dtype=edge_weight.dtype).at[seed].set(1)
    )

    def relax(_, current):
        incoming = jnp.minimum(current[peer], edge_weight)
        return jnp.maximum(current, jnp.max(incoming, axis=1))

    return jax.lax.fori_loop(0, len(geometry.centre), relax, strength)


def install():
    """Replace the prototype kernel path strength without changing product code."""
    kernel._continuous_path_weight = _continuous_path_weight
