"""Memory-bounded quadrature reductions over clipped cell supports."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from nova.equilibrium.stencil_mesh import FluxFieldPolynomial

__all__ = [
    "ClippedCurrentMoments",
    "ClippedFieldIntegrals",
    "clipped_support_current_moments",
    "clipped_support_field_integrals",
    "clipped_support_quadrature",
    "cut_cell_bank_capacity",
]


_GAUSS_NODE, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(8)
_UNIT_NODE = 0.5 * (_GAUSS_NODE + 1.0)
_UNIT_WEIGHT = 0.5 * _GAUSS_WEIGHT
_WHOLE_CELL_VERTEX_CAPACITY = 24
_ROW_CROSSING_CAPACITY = 4


class ClippedFieldIntegrals(NamedTuple):
    """Per-cell pressure-volume and poloidal-field-volume integrals."""

    pressure_volume: jax.Array
    field_volume: jax.Array


class ClippedCurrentMoments(NamedTuple):
    """Per-cell current and centroid-relative first moments."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array


class _QuadratureSupport(NamedTuple):
    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array


def cut_cell_bank_capacity(coordinates: np.ndarray, ring_centres: np.ndarray) -> int:
    """Return the mesh-static bank bound for limited and single-null cuts.

    A limited or single-null separatrix intersects a horizontal carrier row at
    no more than four cells. Cells without a complete quadratic ring are added
    independently because the wall-clipped hull does not obey that row bound.
    Overflow remains fail-closed in the device reduction, so a topology outside
    this declared family cannot silently truncate its support.
    """
    point = np.asarray(coordinates, dtype=np.float64)
    if point.ndim != 2 or point.shape[1] != 2 or len(point) == 0:
        raise ValueError("coordinates must have shape (cells, 2)")
    centre = np.asarray(ring_centres, dtype=np.intp)
    out_of_bounds = centre.size and (centre.min() < 0 or centre.max() >= len(point))
    if centre.ndim != 1 or out_of_bounds:
        raise ValueError("ring centres must index the carrier cells")
    complete = np.zeros(len(point), dtype=bool)
    complete[centre] = True
    hull_count = int(np.count_nonzero(~complete))
    height = np.sort(point[:, 1])
    scale = max(float(np.max(np.abs(height))), float(np.ptp(height)), 1.0)
    tolerance = 1024.0 * np.finfo(np.float64).eps * scale
    row_count = 1 + int(np.count_nonzero(np.diff(height) > tolerance))
    return min(len(point), hull_count + _ROW_CROSSING_CAPACITY * row_count)


def _quadrature_from_arrays(vertices, count, centroids, selection):
    capacity = vertices.shape[1]
    triangle_slot = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:, :1], (len(vertices), capacity - 2, 2))
    second = vertices[:, triangle_slot]
    third = vertices[:, triangle_slot + 1]
    radial = jnp.asarray(_UNIT_NODE, dtype=vertices.dtype)
    vertical = jnp.asarray(_UNIT_NODE, dtype=vertices.dtype)
    radial_weight = jnp.asarray(_UNIT_WEIGHT, dtype=vertices.dtype)
    vertical_weight = jnp.asarray(_UNIT_WEIGHT, dtype=vertices.dtype)
    u, v = jnp.meshgrid(radial, vertical, indexing="ij")
    wu, wv = jnp.meshgrid(radial_weight, vertical_weight, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    rule_weight = (wu * wv).reshape(-1)
    edge_first = second - first
    edge_second = third - first
    points = (
        first[:, :, None, :]
        + u[None, None, :, None] * edge_first[:, :, None, :]
        + (1.0 - u)[None, None, :, None]
        * v[None, None, :, None]
        * edge_second[:, :, None, :]
    )
    cross = jnp.abs(
        edge_first[..., 0] * edge_second[..., 1]
        - edge_first[..., 1] * edge_second[..., 0]
    )
    live = (triangle_slot[None, :] + 1 < count[:, None]) & selection[:, None]
    weights = cross[:, :, None] * (1.0 - u)[None, None, :] * rule_weight[None, None, :]
    weights = jnp.where(live[:, :, None], weights, 0.0)
    points = points.reshape(len(vertices), -1, 2)
    weights = weights.reshape(len(vertices), -1)
    points = jnp.where((weights > 0.0)[..., None], points, centroids[:, None, :])
    return points, weights


def clipped_support_quadrature(support, selection):
    """Return fixed-shape degree-fifteen Duffy quadrature on each support."""
    vertices = jnp.asarray(support.support_vertices)
    count = jnp.asarray(support.vertex_count)
    selected = jnp.asarray(selection, dtype=bool)
    return _quadrature_from_arrays(
        vertices, count, jnp.asarray(support.centroids), selected
    )


def _integrate_points(
    points,
    weights,
    field: FluxFieldPolynomial,
    cell_index,
    pressure: Callable,
    boundary_pressure,
    flux_span,
) -> ClippedFieldIntegrals:
    psi_norm, radial_gradient, vertical_gradient = field.sample(points, cell_index)
    radius = points[..., 0]
    pressure_value = pressure(radius, psi_norm, boundary_pressure, flux_span)
    gradient_squared = flux_span**2 * (radial_gradient**2 + vertical_gradient**2)
    field_squared = gradient_squared / (2.0 * jnp.pi * radius) ** 2
    volume_weight = 2.0 * jnp.pi * radius * weights
    return ClippedFieldIntegrals(
        pressure_volume=jnp.sum(pressure_value * volume_weight, axis=1),
        field_volume=jnp.sum(field_squared * volume_weight, axis=1),
    )


def clipped_support_field_integrals(
    support,
    selection,
    field: FluxFieldPolynomial,
    pressure: Callable,
    boundary_pressure,
    flux_span,
    *,
    cut_cell_capacity: int,
) -> ClippedFieldIntegrals:
    """Reduce whole cells in one small bank and cut cells one at a time.

    Whole cells retain the authored 24-vertex, 64-node-per-triangle rule. Cut
    indices are compacted into the declared mesh-static bank, while a scan
    forms and immediately reduces one high-capacity polygon at a time. The
    expensive work is therefore proportional to live cut entries; dead bank
    entries execute no polygon branch and contribute exact zero.
    """
    vertices = jnp.asarray(support.support_vertices)
    count = jnp.asarray(support.vertex_count)
    centroids = jnp.asarray(support.centroids)
    selected = jnp.asarray(selection, dtype=bool)
    boundary = selected & jnp.asarray(support.boundary, dtype=bool)
    whole = selected & jnp.asarray(support.included, dtype=bool) & ~boundary

    whole_vertices = vertices[:, :_WHOLE_CELL_VERTEX_CAPACITY]
    whole_count = jnp.minimum(count, _WHOLE_CELL_VERTEX_CAPACITY)
    whole_points, whole_weights = _quadrature_from_arrays(
        whole_vertices, whole_count, centroids, whole
    )
    whole_integrals = _integrate_points(
        whole_points,
        whole_weights,
        field,
        jnp.arange(len(vertices), dtype=jnp.int32),
        pressure,
        boundary_pressure,
        flux_span,
    )

    capacity = int(cut_cell_capacity)
    if capacity < 1:
        raise ValueError("cut_cell_capacity must be positive")
    cut_count = jnp.sum(boundary, dtype=jnp.int32)
    cut_index = jnp.nonzero(boundary, size=capacity, fill_value=0)[0]
    active = jnp.arange(capacity, dtype=jnp.int32) < cut_count

    def one_cut(entry):
        cell, live = entry

        def integrate(index):
            point, weight = _quadrature_from_arrays(
                vertices[index][None, ...],
                count[index][None],
                centroids[index][None, ...],
                jnp.ones(1, dtype=bool),
            )
            value = _integrate_points(
                point,
                weight,
                field,
                jnp.asarray([index], dtype=jnp.int32),
                pressure,
                boundary_pressure,
                flux_span,
            )
            return value.pressure_volume[0], value.field_volume[0]

        return jax.lax.cond(
            live,
            jax.checkpoint(integrate),
            lambda _index: (
                jnp.asarray(0.0, dtype=vertices.dtype),
                jnp.asarray(0.0, dtype=vertices.dtype),
            ),
            cell,
        )

    cut_pressure, cut_field = jax.lax.map(one_cut, (cut_index, active))
    pressure_volume = whole_integrals.pressure_volume.at[cut_index].add(
        jnp.where(active, cut_pressure, 0.0)
    )
    field_volume = whole_integrals.field_volume.at[cut_index].add(
        jnp.where(active, cut_field, 0.0)
    )
    overflow = cut_count > capacity
    return ClippedFieldIntegrals(
        pressure_volume=jnp.where(overflow, jnp.nan, pressure_volume),
        field_volume=jnp.where(overflow, jnp.nan, field_volume),
    )


def _integrate_current_points(
    points,
    weights,
    field: FluxFieldPolynomial,
    cell_index,
    moment_centre,
    profile,
) -> ClippedCurrentMoments:
    psi_norm, _radial_gradient, _vertical_gradient = field.sample(points, cell_index)
    density = profile.current_density(points[..., 0], psi_norm)
    weighted = density * weights
    first = jnp.sum(
        weighted[..., None]
        * (points - jnp.asarray(moment_centre)[cell_index, None, :]),
        axis=1,
    )
    return ClippedCurrentMoments(
        cell_current=jnp.sum(weighted, axis=1),
        radial_moment=first[:, 0],
        vertical_moment=first[:, 1],
    )


def clipped_support_current_moments(
    support,
    selection,
    field: FluxFieldPolynomial,
    profile,
    *,
    cut_cell_capacity: int,
) -> ClippedCurrentMoments:
    """Reduce profile current moments with dense work confined to cut cells."""
    vertices = jnp.asarray(support.support_vertices)
    count = jnp.asarray(support.vertex_count)
    centroids = jnp.asarray(support.centroids)
    selected = jnp.asarray(selection, dtype=bool)
    boundary = selected & jnp.asarray(support.boundary, dtype=bool)
    whole = selected & jnp.asarray(support.included, dtype=bool) & ~boundary

    whole_points, whole_weights = _quadrature_from_arrays(
        vertices[:, :_WHOLE_CELL_VERTEX_CAPACITY],
        jnp.minimum(count, _WHOLE_CELL_VERTEX_CAPACITY),
        centroids,
        whole,
    )
    whole_moments = _integrate_current_points(
        whole_points,
        whole_weights,
        field,
        jnp.arange(len(vertices), dtype=jnp.int32),
        centroids,
        profile,
    )

    capacity = int(cut_cell_capacity)
    if capacity < 1:
        raise ValueError("cut_cell_capacity must be positive")
    cut_count = jnp.sum(boundary, dtype=jnp.int32)
    cut_index = jnp.nonzero(boundary, size=capacity, fill_value=0)[0]
    active = jnp.arange(capacity, dtype=jnp.int32) < cut_count

    def one_cut(entry):
        cell, live = entry

        def integrate(index):
            point, weight = _quadrature_from_arrays(
                vertices[index][None, ...],
                count[index][None],
                centroids[index][None, ...],
                jnp.ones(1, dtype=bool),
            )
            value = _integrate_current_points(
                point,
                weight,
                field,
                jnp.asarray([index], dtype=jnp.int32),
                centroids,
                profile,
            )
            return (
                value.cell_current[0],
                value.radial_moment[0],
                value.vertical_moment[0],
            )

        zero = jnp.asarray(0.0, dtype=vertices.dtype)
        return jax.lax.cond(
            live,
            jax.checkpoint(integrate),
            lambda _index: (zero, zero, zero),
            cell,
        )

    cut_current, cut_radial, cut_vertical = jax.lax.map(one_cut, (cut_index, active))

    def scatter(whole_value, cut_value):
        return whole_value.at[cut_index].add(jnp.where(active, cut_value, 0.0))

    overflow = cut_count > capacity
    return ClippedCurrentMoments(
        *(
            jnp.where(overflow, jnp.nan, value)
            for value in (
                scatter(whole_moments.cell_current, cut_current),
                scatter(whole_moments.radial_moment, cut_radial),
                scatter(whole_moments.vertical_moment, cut_vertical),
            )
        )
    )
