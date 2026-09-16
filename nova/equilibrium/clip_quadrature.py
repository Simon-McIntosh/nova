"""Memory-bounded quadrature reductions over clipped cell supports."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments

if TYPE_CHECKING:
    from nova.equilibrium.stencil_mesh import FluxFieldPolynomial

__all__ = [
    "ClippedCurrentMoments",
    "ClippedFieldIntegrals",
    "SaddleWedgeCurrentMoments",
    "clipped_support_current_moments",
    "clipped_support_field_integrals",
    "clipped_support_quadrature",
    "cut_cell_moment_evaluation_bound",
    "cut_cell_bank_capacity",
    "saddle_wedge_current_moments",
]


_GAUSS_NODE, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(8)
_UNIT_NODE = 0.5 * (_GAUSS_NODE + 1.0)
_UNIT_WEIGHT = 0.5 * _GAUSS_WEIGHT
_WHOLE_CELL_VERTEX_CAPACITY = 24
_ROW_CROSSING_CAPACITY = 4
_SPLINE_ARC_SEGMENTS = 128
_MAX_CURVED_ARCS = _WHOLE_CELL_VERTEX_CAPACITY // 2
_QUADRATIC_POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
_QUADRATIC_SAMPLE_LOCAL = np.asarray(
    ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.5, 0.5)),
    dtype=np.float64,
)
_QUADRATIC_SAMPLE_DESIGN = np.stack(
    [
        _QUADRATIC_SAMPLE_LOCAL[:, 0] ** radial
        * _QUADRATIC_SAMPLE_LOCAL[:, 1] ** vertical
        for radial, vertical in _QUADRATIC_POWERS
    ],
    axis=1,
)
_QUADRATIC_SAMPLE_INVERSE = np.linalg.inv(_QUADRATIC_SAMPLE_DESIGN)
_BOUNDARY_NODE, _BOUNDARY_WEIGHT = np.polynomial.legendre.leggauss(5)
_BOUNDARY_NODE = 0.5 * (_BOUNDARY_NODE + 1.0)
_BOUNDARY_WEIGHT = 0.5 * _BOUNDARY_WEIGHT


class ClippedFieldIntegrals(NamedTuple):
    """Per-cell pressure-volume and poloidal-field-volume integrals."""

    pressure_volume: jax.Array
    field_volume: jax.Array


class ClippedCurrentMoments(NamedTuple):
    """Per-cell current and centroid-relative first moments."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array


class SaddleWedgeCurrentMoments(NamedTuple):
    """Current and centroid-relative first moments for four saddle wedges."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array


class _QuadratureSupport(NamedTuple):
    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array


def cut_cell_moment_evaluation_bound() -> int:
    """Return the fixed live point bound for one curved cut-cell reduction."""
    return len(_QUADRATIC_SAMPLE_LOCAL) + len(_BOUNDARY_NODE) * _MAX_CURVED_ARCS


def _quadratic_sample_field(field, cell_index):
    """Sample one cell field on the unisolvent quadratic point set."""
    cell = jnp.asarray(cell_index, dtype=jnp.int32)
    centre = jnp.asarray(field.centre)[cell]
    scale = jnp.asarray(field.scale)[cell]
    local = jnp.asarray(_QUADRATIC_SAMPLE_LOCAL, dtype=centre.dtype)
    points = centre[:, None, :] + scale[:, None, :] * local[None, :, :]
    value, radial_gradient, vertical_gradient = field.sample(points, cell)
    return points, value, radial_gradient, vertical_gradient, centre, scale


def _quadratic_coefficients(values):
    """Interpolate sampled values onto the complete local quadratic basis."""
    inverse = jnp.asarray(_QUADRATIC_SAMPLE_INVERSE, dtype=jnp.asarray(values).dtype)
    return jnp.einsum("ij,nj->ni", inverse, jnp.asarray(values))


def _compact_chord_polygon(vertices, count):
    """Collapse every sampled level arc to its chord and retain its sagitta.

    The exact clip rotates a cut support to its leaving crossing, then packs the
    traced arc into the first 129 vertices: both endpoints and 127 interior
    samples. Slots 0 through 128 therefore carry the 128 arc edges, while every
    later live slot carries one straight cell-boundary vertex. The endpoints
    form the chord polygon and slot 64 supplies the sagitta. A layout that does
    not resolve to one arc plus a three-to-24-vertex base polygon is refused.
    """
    point = jnp.asarray(vertices)
    vertex_count = jnp.asarray(count)
    cell_count, capacity, _coordinate = point.shape
    slot = jnp.arange(capacity)
    valid = slot[None, :] < vertex_count[:, None]
    expanded = vertex_count > _WHOLE_CELL_VERTEX_CAPACITY
    base_count = vertex_count - (_SPLINE_ARC_SEGMENTS - 1)
    supported = jnp.where(
        expanded,
        (vertex_count >= _SPLINE_ARC_SEGMENTS + 2)
        & (base_count >= 3)
        & (base_count <= _WHOLE_CELL_VERTEX_CAPACITY),
        (vertex_count >= 3) & (vertex_count <= _WHOLE_CELL_VERTEX_CAPACITY),
    )
    arc_interior = (
        expanded[:, None] & (slot[None, :] > 0) & (slot[None, :] < _SPLINE_ARC_SEGMENTS)
    )
    keep = valid & ~arc_interior
    rank = jnp.cumsum(keep, axis=1) - 1
    safe_rank = jnp.where(keep, rank, 0)
    cell = jnp.broadcast_to(jnp.arange(cell_count)[:, None], safe_rank.shape)
    safe_destination = jnp.minimum(safe_rank, _WHOLE_CELL_VERTEX_CAPACITY - 1)
    chord = jnp.zeros((cell_count, _WHOLE_CELL_VERTEX_CAPACITY, 2), dtype=point.dtype)
    chord = chord.at[cell, safe_destination].add(
        jnp.where(
            keep[..., None] & (rank[..., None] < _WHOLE_CELL_VERTEX_CAPACITY),
            point,
            0.0,
        )
    )
    chord_count = jnp.sum(keep, axis=1)

    retained_arc_capacity = min(_MAX_CURVED_ARCS, capacity)
    arc_slot = jnp.arange(retained_arc_capacity)
    arc_active = expanded[:, None] & (arc_slot[None, :] == 0)
    arc_index = jnp.zeros((cell_count, retained_arc_capacity), dtype=jnp.int32)
    middle_index = jnp.full_like(
        arc_index, min(_SPLINE_ARC_SEGMENTS // 2, capacity - 1)
    )
    end_index = jnp.full_like(arc_index, min(_SPLINE_ARC_SEGMENTS, capacity - 1))
    arc_first = jnp.take_along_axis(point, arc_index[..., None], axis=1)
    arc_middle = jnp.take_along_axis(point, middle_index[..., None], axis=1)
    arc_last = jnp.take_along_axis(point, end_index[..., None], axis=1)
    supported = supported & (
        chord_count == jnp.where(expanded, base_count, vertex_count)
    )
    return (
        chord,
        jnp.minimum(chord_count, _WHOLE_CELL_VERTEX_CAPACITY),
        arc_first,
        arc_middle,
        arc_last,
        arc_active,
        supported,
    )


def _polynomial_antiderivative(local, coefficients, radial_shift, vertical_shift):
    """Evaluate the radial antiderivative of a shifted density monomial."""
    radial = local[..., 0]
    vertical = local[..., 1]
    value = jnp.zeros(radial.shape, dtype=radial.dtype)
    for column, (radial_power, vertical_power) in enumerate(_QUADRATIC_POWERS):
        exponent = radial_power + radial_shift + 1
        value = value + (
            coefficients[:, :, None, column]
            * radial**exponent
            * vertical ** (vertical_power + vertical_shift)
            / exponent
        )
    return value


def _quadratic_arc_correction(
    first,
    middle,
    last,
    active,
    polynomial_centre,
    coordinate_scale,
    coefficients,
):
    """Return density moments between each chord and its quadratic arc."""
    centre = jnp.asarray(polynomial_centre)
    scale = jnp.asarray(coordinate_scale)
    start = (jnp.asarray(first) - centre[:, None, :]) / scale[:, None, :]
    sagitta = (jnp.asarray(middle) - centre[:, None, :]) / scale[:, None, :]
    end = (jnp.asarray(last) - centre[:, None, :]) / scale[:, None, :]
    control = 2.0 * sagitta - 0.5 * (start + end)
    node = jnp.asarray(_BOUNDARY_NODE, dtype=start.dtype)[None, None, :, None]
    weight = jnp.asarray(_BOUNDARY_WEIGHT, dtype=start.dtype)[None, None, :]
    curve = (
        (1.0 - node) ** 2 * start[:, :, None, :]
        + 2.0 * (1.0 - node) * node * control[:, :, None, :]
        + node**2 * end[:, :, None, :]
    )
    curve_derivative = (
        2.0 * (1.0 - node) * (control - start)[:, :, None, :]
        + 2.0 * node * (end - control)[:, :, None, :]
    )
    chord = start[:, :, None, :] + node * (end - start)[:, :, None, :]
    chord_derivative = jnp.broadcast_to(
        (end - start)[:, :, None, :], curve_derivative.shape
    )
    coefficient = jnp.broadcast_to(
        jnp.asarray(coefficients)[:, None, :],
        (start.shape[0], start.shape[1], len(_QUADRATIC_POWERS)),
    )

    def correction(radial_shift, vertical_shift):
        curve_value = _polynomial_antiderivative(
            curve, coefficient, radial_shift, vertical_shift
        )
        chord_value = _polynomial_antiderivative(
            chord, coefficient, radial_shift, vertical_shift
        )
        line_integral = jnp.sum(
            weight
            * (
                curve_value * curve_derivative[..., 1]
                - chord_value * chord_derivative[..., 1]
            ),
            axis=2,
        )
        return jnp.sum(jnp.where(active, line_integral, 0.0), axis=1)

    area_scale = scale[:, 0] * scale[:, 1]
    return (
        area_scale * correction(0, 0),
        jnp.stack(
            (
                area_scale * scale[:, 0] * correction(1, 0),
                area_scale * scale[:, 1] * correction(0, 1),
            ),
            axis=1,
        ),
    )


def _boundary_polynomial_moments(
    vertices,
    count,
    polynomial_centre,
    coordinate_scale,
    coefficients,
    moment_centres,
) -> ClippedCurrentMoments:
    """Integrate a local quadratic over chord-plus-sagitta cut polygons."""
    (
        chord,
        chord_count,
        arc_first,
        arc_middle,
        arc_last,
        arc_active,
        supported,
    ) = _compact_chord_polygon(vertices, count)
    current, first = padded_polynomial_current_moments(
        chord,
        chord_count,
        polynomial_centre,
        coordinate_scale,
        coefficients,
        _QUADRATIC_POWERS,
    )
    slot = jnp.arange(chord.shape[1])
    following_slot = jnp.where(
        slot[None, :] + 1 < chord_count[:, None], slot[None, :] + 1, 0
    )
    following = jnp.take_along_axis(chord, following_slot[..., None], axis=1)
    local = chord - jnp.asarray(polynomial_centre)[:, None, :]
    following_local = following - jnp.asarray(polynomial_centre)[:, None, :]
    cross = (
        local[..., 0] * following_local[..., 1]
        - following_local[..., 0] * local[..., 1]
    )
    valid = slot[None, :] < chord_count[:, None]
    orientation = jnp.where(
        jnp.sum(jnp.where(valid, cross, 0.0), axis=1) < 0.0, -1.0, 1.0
    )
    arc_current, arc_first_moment = _quadratic_arc_correction(
        arc_first,
        arc_middle,
        arc_last,
        arc_active,
        polynomial_centre,
        coordinate_scale,
        coefficients,
    )
    current = current + orientation * arc_current
    first = first + orientation[:, None] * arc_first_moment
    first = first + current[:, None] * (
        jnp.asarray(polynomial_centre) - jnp.asarray(moment_centres)
    )
    current = jnp.where(supported, current, jnp.nan)
    first = jnp.where(supported[:, None], first, jnp.nan)
    return ClippedCurrentMoments(current, first[:, 0], first[:, 1])


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


def _integrate_field_polynomial(
    vertices,
    count,
    field: FluxFieldPolynomial,
    cell_index,
    centroids,
    pressure: Callable,
    boundary_pressure,
    flux_span,
) -> ClippedFieldIntegrals:
    """Integrate quadratic volume-density images over curved cut polygons."""
    (
        points,
        psi_norm,
        radial_gradient,
        vertical_gradient,
        polynomial_centre,
        coordinate_scale,
    ) = _quadratic_sample_field(field, cell_index)
    radius = points[..., 0]
    pressure_value = pressure(radius, psi_norm, boundary_pressure, flux_span)
    gradient_squared = flux_span**2 * (radial_gradient**2 + vertical_gradient**2)
    field_squared = gradient_squared / (2.0 * jnp.pi * radius) ** 2
    volume_weight = 2.0 * jnp.pi * radius
    pressure_coefficients = _quadratic_coefficients(pressure_value * volume_weight)
    field_coefficients = _quadratic_coefficients(field_squared * volume_weight)
    pressure_moments = _boundary_polynomial_moments(
        vertices,
        count,
        polynomial_centre,
        coordinate_scale,
        pressure_coefficients,
        centroids,
    )
    field_moments = _boundary_polynomial_moments(
        vertices,
        count,
        polynomial_centre,
        coordinate_scale,
        field_coefficients,
        centroids,
    )
    return ClippedFieldIntegrals(
        pressure_volume=pressure_moments.cell_current,
        field_volume=field_moments.cell_current,
    )


@jax.named_scope("compact_clipped_field_integrals")
def clipped_support_field_integrals(
    support,
    selection,
    field: FluxFieldPolynomial,
    pressure: Callable,
    boundary_pressure,
    flux_span,
    *,
    cut_cell_capacity: int,
    boundary_reduction: bool = False,
) -> ClippedFieldIntegrals:
    """Reduce field integrals with an opt-in polynomial boundary route.

    Whole cells retain the authored 24-vertex, 64-node-per-triangle rule. Cut
    indices are compacted into the declared mesh-static bank, while a scan
    forms and immediately reduces one high-capacity polygon at a time. The
    expensive work is therefore proportional to live cut entries; dead bank
    entries execute no polygon branch and contribute exact zero. Cut cells use
    the retained fan unless ``boundary_reduction`` is explicitly enabled.
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
    cut_vertices = vertices[cut_index]
    cut_vertex_count = count[cut_index]
    cut_centroids = centroids[cut_index]

    @jax.named_scope("compact_cut_field_scan_body")
    def scan_cut(_carry, entry):
        cell, polygon, polygon_count, centroid, live = entry

        def integrate(operand):
            index, carried_vertices, carried_count, carried_centroid = operand
            if boundary_reduction:
                value = _integrate_field_polynomial(
                    carried_vertices[None, ...],
                    carried_count[None],
                    field,
                    jnp.asarray([index], dtype=jnp.int32),
                    carried_centroid[None, ...],
                    pressure,
                    boundary_pressure,
                    flux_span,
                )
            else:
                point, weight = _quadrature_from_arrays(
                    carried_vertices[None, ...],
                    carried_count[None],
                    carried_centroid[None, ...],
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

        values = jax.lax.cond(
            live,
            jax.checkpoint(integrate),
            lambda _operand: (
                jnp.asarray(0.0, dtype=vertices.dtype),
                jnp.asarray(0.0, dtype=vertices.dtype),
            ),
            (cell, polygon, polygon_count, centroid),
        )
        return None, values

    _, (cut_pressure, cut_field) = jax.lax.scan(
        scan_cut,
        None,
        (cut_index, cut_vertices, cut_vertex_count, cut_centroids, active),
        unroll=1,
    )
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
    moment_centres,
    profile,
) -> ClippedCurrentMoments:
    psi_norm, _radial_gradient, _vertical_gradient = field.sample(points, cell_index)
    density = profile.current_density(points[..., 0], psi_norm)
    weighted = density * weights
    first = jnp.sum(
        weighted[..., None] * (points - jnp.asarray(moment_centres)[:, None, :]),
        axis=1,
    )
    return ClippedCurrentMoments(
        cell_current=jnp.sum(weighted, axis=1),
        radial_moment=first[:, 0],
        vertical_moment=first[:, 1],
    )


def _integrate_current_polynomial(
    vertices,
    count,
    field: FluxFieldPolynomial,
    cell_index,
    moment_centres,
    profile,
) -> ClippedCurrentMoments:
    """Integrate a profile's quadratic density image over curved cut polygons."""
    points, psi_norm, _radial, _vertical, polynomial_centre, coordinate_scale = (
        _quadratic_sample_field(field, cell_index)
    )
    density = profile.current_density(points[..., 0], psi_norm)
    coefficients = _quadratic_coefficients(density)
    return _boundary_polynomial_moments(
        vertices,
        count,
        polynomial_centre,
        coordinate_scale,
        coefficients,
        moment_centres,
    )


@jax.named_scope("saddle_wedge_current_moments")
def saddle_wedge_current_moments(wedges, field, profiles, selection=None):
    """Integrate one statically declared profile over each saddle wedge.

    ``profiles`` is a four-item tuple in core, private-flux, and common-SOL
    order. The tuple length and all polygon capacities are static; changing the
    crossing geometry therefore changes values but never compiled shapes.
    """
    profiles = tuple(profiles)
    if len(profiles) != 4:
        raise ValueError("profiles must contain exactly four wedge profiles")
    vertices = jnp.asarray(wedges.support_vertices)
    count = jnp.asarray(wedges.vertex_count)
    centroids = jnp.asarray(wedges.centroids)
    if vertices.ndim != 4 or vertices.shape[1] != 4 or vertices.shape[-1] != 2:
        raise ValueError("wedge vertices must have shape (cells, 4, capacity, 2)")
    cell_count = vertices.shape[0]
    if count.shape != (cell_count, 4):
        raise ValueError("wedge counts must have shape (cells, 4)")
    if selection is None:
        selected = jnp.asarray(wedges.saddle, dtype=bool)
    else:
        selected = jnp.asarray(selection, dtype=bool) & jnp.asarray(
            wedges.saddle, dtype=bool
        )
        if selected.shape != (cell_count,):
            raise ValueError("selection must carry one flag per cell")

    cell_index = jnp.arange(cell_count, dtype=jnp.int32)
    current = []
    radial = []
    vertical = []
    for wedge, profile in enumerate(profiles):
        points, weights = _quadrature_from_arrays(
            vertices[:, wedge], count[:, wedge], centroids, selected
        )
        moments = _integrate_current_points(
            points,
            weights,
            field,
            cell_index,
            centroids,
            profile,
        )
        current.append(moments.cell_current)
        radial.append(moments.radial_moment)
        vertical.append(moments.vertical_moment)
    return SaddleWedgeCurrentMoments(
        cell_current=jnp.stack(current, axis=1),
        radial_moment=jnp.stack(radial, axis=1),
        vertical_moment=jnp.stack(vertical, axis=1),
    )


@jax.named_scope("compact_clipped_current_moments")
def clipped_support_current_moments(
    support,
    selection,
    field: FluxFieldPolynomial,
    profile,
    *,
    cut_cell_capacity: int,
    boundary_reduction: bool = False,
) -> ClippedCurrentMoments:
    """Reduce current moments with an opt-in polynomial boundary route."""
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
    cut_vertices = vertices[cut_index]
    cut_vertex_count = count[cut_index]
    cut_centroids = centroids[cut_index]

    @jax.named_scope("compact_cut_current_scan_body")
    def scan_cut(_carry, entry):
        cell, polygon, polygon_count, centroid, live = entry

        def integrate(operand):
            index, carried_vertices, carried_count, carried_centroid = operand
            if boundary_reduction:
                value = _integrate_current_polynomial(
                    carried_vertices[None, ...],
                    carried_count[None],
                    field,
                    jnp.asarray([index], dtype=jnp.int32),
                    carried_centroid[None, ...],
                    profile,
                )
            else:
                point, weight = _quadrature_from_arrays(
                    carried_vertices[None, ...],
                    carried_count[None],
                    carried_centroid[None, ...],
                    jnp.ones(1, dtype=bool),
                )
                value = _integrate_current_points(
                    point,
                    weight,
                    field,
                    jnp.asarray([index], dtype=jnp.int32),
                    carried_centroid[None, ...],
                    profile,
                )
            return (
                value.cell_current[0],
                value.radial_moment[0],
                value.vertical_moment[0],
            )

        zero = jnp.asarray(0.0, dtype=vertices.dtype)
        values = jax.lax.cond(
            live,
            jax.checkpoint(integrate),
            lambda _operand: (zero, zero, zero),
            (cell, polygon, polygon_count, centroid),
        )
        return None, values

    _, (cut_current, cut_radial, cut_vertical) = jax.lax.scan(
        scan_cut,
        None,
        (cut_index, cut_vertices, cut_vertex_count, cut_centroids, active),
        unroll=1,
    )

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
