"""Traced structured-grid extraction of flux-surface-averaged geometry.

The composition and reuse boundaries follow
``scripts/geometry_service_reuse/report.md``: axis-connected flood fills
establish topology at every surface level, a global tensor spline supplies every
cell polynomial, fixed-capacity clips carry Green-theorem moments, and masked
reductions select the constrained extrema.  The returned dictionary is the
fixed-shape record consumed by the transport geometry adapter.
"""

from functools import partial

import numpy as np

from nova.biot.arcbandedcoupling import _arc_rule
from nova.biot.greens import MU0
from nova.equilibrium.flux_surface_connectivity import flood_fill_core
from nova.equilibrium.separatrix_clip import _traced_clip
from nova.linalg.interpolant import Bernstein
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp


_TWO_PI = 2.0 * np.pi
_16PI3 = 16.0 * np.pi**3
_16PI2 = 16.0 * np.pi**2
_BICUBIC_ARC_POINT, _BICUBIC_ARC_WEIGHT = _arc_rule(12)
_BICUBIC_EDGE_ROOT_CAPACITY = 3
_BICUBIC_ARC_CAPACITY = 2
_SUPPORT_CAPACITY = 8
NONNEGATIVE_EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _traced_profile_shapes(psi_n, n_terms: int, *, nonnegative: bool):
    """Profile-ladder shapes on a normalised-flux sample, as JAX arrays."""
    normalised = jnp.clip(psi_n, 0.0, 1.0)
    if n_terms == 0:
        return jnp.empty((*normalised.shape, 0), dtype=normalised.dtype)
    if nonnegative:
        exponents = jnp.asarray(NONNEGATIVE_EXPONENTS[:n_terms], dtype=normalised.dtype)
        return (1.0 - normalised)[..., jnp.newaxis] ** exponents

    coordinate = 2.0 * normalised - 1.0
    terms = [jnp.ones_like(normalised)]
    if n_terms > 1:
        terms.append(coordinate)
    for degree in range(2, n_terms):
        terms.append(
            ((2 * degree - 1) * coordinate * terms[-1] - (degree - 1) * terms[-2])
            / degree
        )
    return jnp.stack(terms, axis=-1) * (1.0 - normalised)[..., jnp.newaxis]


def _cumulative_trapezoid_from_axis(coordinate, values):
    """Cumulative trapezoid with the finite axis value extended to zero."""
    increments = 0.5 * (values[1:] + values[:-1]) * (coordinate[1:] - coordinate[:-1])
    return jnp.concatenate(
        [
            values[:1] * coordinate[:1],
            values[:1] * coordinate[:1] + jnp.cumsum(increments),
        ]
    )


def _integrate_diamagnetic_drive(
    psi_n, diamagnetic_drive, *, boundary_f, poloidal_flux_span
):
    """Integrate ``F F'`` inward from the boundary and return ``F(psi_n)``."""
    increments = (
        0.5 * (diamagnetic_drive[1:] + diamagnetic_drive[:-1]) * jnp.diff(psi_n)
    )
    integral_from_boundary = jnp.concatenate(
        [-jnp.cumsum(increments[::-1])[::-1], jnp.zeros(1, psi_n.dtype)]
    )
    f_squared = (
        boundary_f**2 + 2.0 * poloidal_flux_span / _TWO_PI * integral_from_boundary
    )
    return (
        jnp.sign(boundary_f) * jnp.sqrt(jnp.clip(f_squared, min=0.0)),
        jnp.all(f_squared > 0.0),
    )


def _connectivity_core(psi_n, inside_limiter):
    """Return the axis-connected confined cells using Nova's flood-fill kernel."""
    from nova.equilibrium.flux_surface_connectivity import flood_fill_core

    confined = (psi_n < 1.0) & inside_limiter
    seed_position = jnp.argmin(jnp.where(confined, psi_n, jnp.inf).reshape(-1))
    seed = (
        jnp.zeros(psi_n.size, dtype=bool)
        .at[seed_position]
        .set(True)
        .reshape(psi_n.shape)
    )
    return flood_fill_core(confined, seed, psi_n.shape[0] + psi_n.shape[1])


def _surface_interpolation(rho, rho_samples, values, axis_value, edge_value):
    """Interpolate surface samples onto ``rho`` with explicit axis and edge limits."""
    interpolated = jnp.interp(rho, rho_samples, values, right=edge_value)
    inner = rho < rho_samples[0]
    inner_value = axis_value + (values[0] - axis_value) * (
        rho / jnp.maximum(rho_samples[0], 1e-12)
    )
    return jnp.where(inner, inner_value, interpolated)


def _tensor_bicubic_coefficients(values):
    """Return C1 tensor-bicubic Bernstein coefficients for every grid cell."""
    nz, nr = values.shape
    row = jnp.clip(jnp.arange(nz - 1)[:, None] + jnp.arange(-1, 3)[None, :], 0, nz - 1)
    column = jnp.clip(
        jnp.arange(nr - 1)[:, None] + jnp.arange(-1, 3)[None, :], 0, nr - 1
    )
    stencil = values[row[:, None, :, None], column[None, :, None, :]]
    transform = jnp.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (-1.0 / 6.0, 1.0, 1.0 / 6.0, 0.0),
            (0.0, 1.0 / 6.0, 1.0, -1.0 / 6.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        dtype=values.dtype,
    )
    coefficient = jnp.einsum("ij,...jk,lk->...il", transform, stencil, transform)
    return coefficient.reshape(-1, 4, 4)


def _bernstein_matrix(coordinate, order):
    """Evaluate Nova's traced Bernstein basis without changing leading shape."""
    coordinate = jnp.asarray(coordinate)
    return (
        Bernstein(order=order)
        .coefficent_matrix(coordinate.reshape(-1))
        .reshape(coordinate.shape + (order + 1,))
    )


def _tensor_bernstein(coefficient, radial, vertical, radial_order, vertical_order):
    """Evaluate a tensor Bernstein polynomial at paired coordinates."""
    radial_basis = _bernstein_matrix(radial, radial_order)
    vertical_basis = _bernstein_matrix(vertical, vertical_order)
    return jnp.einsum("...i,...ij,...j->...", vertical_basis, coefficient, radial_basis)


def _bicubic_derivatives(coefficient, radial, vertical):
    """Evaluate a bicubic and its first and second local derivatives."""
    radial_coefficient = 3.0 * jnp.diff(coefficient, axis=-1)
    vertical_coefficient = 3.0 * jnp.diff(coefficient, axis=-2)
    radial_second = 2.0 * jnp.diff(radial_coefficient, axis=-1)
    vertical_second = 2.0 * jnp.diff(vertical_coefficient, axis=-2)
    mixed = 3.0 * jnp.diff(vertical_coefficient, axis=-1)
    values = (
        _tensor_bernstein(coefficient, radial, vertical, 3, 3),
        _tensor_bernstein(radial_coefficient, radial, vertical, 2, 3),
        _tensor_bernstein(vertical_coefficient, radial, vertical, 3, 2),
        _tensor_bernstein(radial_second, radial, vertical, 1, 3),
        _tensor_bernstein(mixed, radial, vertical, 2, 2),
        _tensor_bernstein(vertical_second, radial, vertical, 3, 1),
    )
    return tuple(jnp.asarray(value, dtype=coefficient.dtype) for value in values)


def _bicubic_edge_coordinates(edge, fraction):
    """Map counter-clockwise unit-cell edge coordinates to the cell interior."""
    zero = jnp.zeros((), dtype=fraction.dtype)
    one = jnp.ones((), dtype=fraction.dtype)
    radial = jnp.where(
        edge == 0,
        fraction,
        jnp.where(edge == 1, one, jnp.where(edge == 2, one - fraction, zero)),
    )
    vertical = jnp.where(
        edge == 0,
        zero,
        jnp.where(edge == 1, fraction, jnp.where(edge == 2, one, one - fraction)),
    )
    return radial, vertical


def _bicubic_edge_crossings(level, coefficient, corner_flux, *, detect_even_roots=True):
    """Locate every simple bicubic edge root in derivative-bounded intervals."""
    level = jnp.asarray(level, dtype=corner_flux.dtype)
    half = jnp.asarray(0.5, dtype=corner_flux.dtype)
    zero = jnp.zeros((), dtype=corner_flux.dtype)
    one = jnp.ones((), dtype=corner_flux.dtype)
    derivative_floor = jnp.asarray(1e-14, dtype=corner_flux.dtype)
    inside = corner_flux < level

    edge_control = jnp.stack(
        (
            coefficient[:, 0, :],
            coefficient[:, :, -1],
            coefficient[:, -1, ::-1],
            coefficient[:, ::-1, 0],
        ),
        axis=1,
    )
    derivative_control = 3.0 * jnp.diff(edge_control, axis=-1)
    quadratic = (
        derivative_control[..., 0]
        - 2.0 * derivative_control[..., 1]
        + derivative_control[..., 2]
    )
    linear = 2.0 * (derivative_control[..., 1] - derivative_control[..., 0])
    constant = derivative_control[..., 0]
    discriminant = jnp.maximum(linear**2 - 4.0 * quadratic * constant, zero)
    square_root = jnp.sqrt(discriminant)
    safe_quadratic = jnp.where(jnp.abs(quadratic) > derivative_floor, quadratic, one)
    quadratic_roots = jnp.stack(
        (
            (-linear - square_root) / (2.0 * safe_quadratic),
            (-linear + square_root) / (2.0 * safe_quadratic),
        ),
        axis=-1,
    )
    quadratic_valid = (
        (jnp.abs(quadratic) > derivative_floor)[..., None]
        & (discriminant[..., None] > zero)
        & (quadratic_roots > zero)
        & (quadratic_roots < one)
    )
    safe_linear = jnp.where(jnp.abs(linear) > derivative_floor, linear, one)
    linear_root = -constant / safe_linear
    linear_valid = (
        (jnp.abs(quadratic) <= derivative_floor)
        & (jnp.abs(linear) > derivative_floor)
        & (linear_root > zero)
        & (linear_root < one)
    )
    derivative_roots = jnp.concatenate(
        (quadratic_roots, linear_root[..., None]), axis=-1
    )
    derivative_valid = jnp.concatenate(
        (quadratic_valid, linear_valid[..., None]), axis=-1
    )
    derivative_roots = jnp.sort(
        jnp.where(derivative_valid, derivative_roots, one), axis=-1
    )[..., :2]

    bounds = jnp.concatenate(
        (
            jnp.zeros((*corner_flux.shape, 1), dtype=corner_flux.dtype),
            derivative_roots,
            jnp.ones((*corner_flux.shape, 1), dtype=corner_flux.dtype),
        ),
        axis=-1,
    )
    low = bounds[..., :-1]
    high = bounds[..., 1:]
    ordinary_crossing = inside != jnp.roll(inside, -1, axis=1)
    use_full_edge = (
        ordinary_crossing if detect_even_roots else jnp.ones_like(ordinary_crossing)
    )
    ordinary_slot = jnp.arange(_BICUBIC_EDGE_ROOT_CAPACITY) == 0
    low = jnp.where(
        use_full_edge[..., None],
        jnp.where(ordinary_slot, zero, one),
        low,
    )
    high = jnp.where(use_full_edge[..., None], one, high)
    edge = jnp.broadcast_to(jnp.arange(4)[None, :, None], low.shape)
    low_value = (
        jnp.einsum(
            "...i,...i->...", _bernstein_matrix(low, 3), edge_control[:, :, None]
        )
        - level
    )
    high_value = (
        jnp.einsum(
            "...i,...i->...", _bernstein_matrix(high, 3), edge_control[:, :, None]
        )
        - level
    )
    crossing = (high > low) & (jnp.signbit(low_value) != jnp.signbit(high_value))

    def bisect(_, state):
        lower, upper, lower_value = state
        middle = half * (lower + upper)
        radial, vertical = _bicubic_edge_coordinates(edge, middle)
        value = (
            _bicubic_derivatives(coefficient[:, None, None], radial, vertical)[0]
            - level
        )
        same_side = jnp.signbit(value) == jnp.signbit(lower_value)
        next_lower = jnp.where(same_side, middle, lower)
        next_upper = jnp.where(same_side, upper, middle)
        next_value = jnp.where(same_side, value, lower_value)
        return next_lower, next_upper, next_value

    low, high, _ = jax.lax.fori_loop(0, 28, bisect, (low, high, low_value))
    fraction = half * (low + high)
    radial, vertical = _bicubic_edge_coordinates(edge, fraction)
    points = jnp.stack((radial, vertical), axis=-1)
    return (
        points.reshape(points.shape[0], 4 * _BICUBIC_EDGE_ROOT_CAPACITY, 2),
        crossing.reshape(crossing.shape[0], 4 * _BICUBIC_EDGE_ROOT_CAPACITY),
        inside,
    )


def _solve_bicubic_ordinate(
    coefficient, level, independent, initial, *, solve_vertical
):
    """Follow one bicubic level-set branch at fixed traced coordinates."""
    dtype = initial.dtype
    level = jnp.asarray(level, dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    derivative_floor = jnp.asarray(1e-12, dtype=dtype)
    maximum_step = jnp.asarray(0.2, dtype=dtype)

    def update(_, ordinate):
        radial = jnp.where(solve_vertical, independent, ordinate)
        vertical = jnp.where(solve_vertical, ordinate, independent)
        value, radial_gradient, vertical_gradient, *_ = _bicubic_derivatives(
            coefficient, radial, vertical
        )
        derivative = jnp.where(solve_vertical, vertical_gradient, radial_gradient)
        safe_derivative = jnp.where(
            jnp.abs(derivative) > derivative_floor,
            derivative,
            jnp.where(derivative < zero, -derivative_floor, derivative_floor),
        )
        step = jnp.clip((value - level) / safe_derivative, -maximum_step, maximum_step)
        return jnp.clip(ordinate - step, zero, one)

    return jax.lax.fori_loop(0, 10, update, initial)


def _bicubic_arc_moment_correction(
    level,
    coefficient,
    corner_flux,
    dr,
    dz,
    base_moments,
    *,
    detect_even_roots=True,
):
    """Return exact clipped moments and monotone bicubic arc quadrature."""
    dtype = corner_flux.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    point = jnp.asarray(_BICUBIC_ARC_POINT, dtype=dtype)
    weight = jnp.asarray(_BICUBIC_ARC_WEIGHT, dtype=dtype)
    tolerance = jnp.asarray(2.0e-8, dtype=dtype) * jnp.maximum(one, jnp.abs(level))
    derivative_floor = jnp.asarray(1e-14, dtype=dtype)
    crossing_points, crossing, _ = _bicubic_edge_crossings(
        level, coefficient, corner_flux, detect_even_roots=detect_even_roots
    )
    edge = jnp.repeat(jnp.arange(4), _BICUBIC_EDGE_ROOT_CAPACITY)
    _, gradient_r, gradient_z, *_ = _bicubic_derivatives(
        coefficient[:, None], crossing_points[..., 0], crossing_points[..., 1]
    )
    tangent_gradient = jnp.where((edge == 0) | (edge == 2), gradient_r, gradient_z)
    tangent_gradient = jnp.where(
        (edge == 2) | (edge == 3), -tangent_gradient, tangent_gradient
    )
    leaving = crossing & (tangent_gradient > zero)
    entering = crossing & (tangent_gradient < zero)
    event = jnp.arange(4 * _BICUBIC_EDGE_ROOT_CAPACITY)
    event_grid = jnp.broadcast_to(event, crossing.shape)
    leaving_order = jnp.argsort(jnp.where(leaving, event_grid, event.size), axis=1)[
        :, :_BICUBIC_ARC_CAPACITY
    ]
    start = jnp.take_along_axis(crossing_points, leaving_order[..., None], axis=1)
    cyclic_distance = (event_grid[:, None, :] - leaving_order[:, :, None]) % event.size
    entering_distance = jnp.where(
        entering[:, None] & (cyclic_distance > 0), cyclic_distance, event.size + 1
    )
    end_order = jnp.argmin(entering_distance, axis=2)
    end = jnp.take_along_axis(crossing_points, end_order[..., None], axis=1)
    leaving_count = jnp.sum(leaving, axis=1)
    entering_count = jnp.sum(entering, axis=1)
    arc_slot = jnp.arange(_BICUBIC_ARC_CAPACITY)[None]
    arc_mask = (arc_slot < leaving_count[:, None]) & (
        jnp.min(entering_distance, axis=2) <= event.size
    )

    whole_span = end - start
    radial_primary = jnp.abs(whole_span[..., 0]) >= jnp.abs(whole_span[..., 1])
    seed = 0.5 * (start + end)
    turning_r, turning_z, turning_valid = _bicubic_stationary_point(
        coefficient[:, None],
        level,
        seed[..., 0],
        seed[..., 1],
        radial_extremum=radial_primary,
    )
    turning = jnp.stack((turning_r, turning_z), axis=-1)
    endpoint_distance = jnp.minimum(
        jnp.linalg.norm(turning - start, axis=-1),
        jnp.linalg.norm(turning - end, axis=-1),
    )
    turning_valid &= endpoint_distance > jnp.asarray(1e-8, dtype=dtype)

    def interval_quadrature(interval_start, interval_end, interval_mask):
        span = interval_end - interval_start
        use_radial = jnp.abs(span[..., 0]) >= jnp.abs(span[..., 1])
        radial_parameter = interval_start[..., 0, None] + span[..., 0, None] * point
        vertical_seed = interval_start[..., 1, None] + span[..., 1, None] * point
        vertical_arc = _solve_bicubic_ordinate(
            coefficient[:, None, None],
            level,
            radial_parameter,
            vertical_seed,
            solve_vertical=True,
        )
        vertical_parameter = interval_start[..., 1, None] + span[..., 1, None] * point
        radial_seed = interval_start[..., 0, None] + span[..., 0, None] * point
        radial_arc = _solve_bicubic_ordinate(
            coefficient[:, None, None],
            level,
            vertical_parameter,
            radial_seed,
            solve_vertical=False,
        )
        arc_r = jnp.where(use_radial[..., None], radial_parameter, radial_arc)
        arc_z = jnp.where(use_radial[..., None], vertical_arc, vertical_parameter)
        value, gradient_r, gradient_z, *_ = _bicubic_derivatives(
            coefficient[:, None, None], arc_r, arc_z
        )
        ordinate_gradient = jnp.where(use_radial[..., None], gradient_z, gradient_r)
        parameter_span = jnp.where(
            use_radial, jnp.abs(span[..., 0]), jnp.abs(span[..., 1])
        )
        coarea_weight = (
            dr
            * dz
            * parameter_span[..., None]
            * weight
            / jnp.maximum(jnp.abs(ordinate_gradient), derivative_floor)
        )
        local_r = dr * (arc_r - 0.5)
        local_z = dz * (arc_z - 0.5)
        safe_gradient_r = jnp.where(
            jnp.abs(gradient_r) > derivative_floor,
            gradient_r,
            jnp.where(gradient_r < zero, -derivative_floor, derivative_floor),
        )
        radial_measure = jnp.where(
            use_radial[..., None],
            dr * span[..., 0, None] * weight,
            -dr * gradient_z / safe_gradient_r * span[..., 1, None] * weight,
        )
        moments = jnp.stack(
            (
                -jnp.sum(local_z * radial_measure, axis=-1),
                -jnp.sum(local_r * local_z * radial_measure, axis=-1),
                -0.5 * jnp.sum(local_z**2 * radial_measure, axis=-1),
                -0.5 * jnp.sum(local_r * local_z**2 * radial_measure, axis=-1),
            ),
            axis=-1,
        )
        sign = jnp.sign(
            jnp.where(use_radial[..., None], ordinate_gradient, ordinate_gradient)
        )
        monotone = jnp.all(sign == sign[..., :1], axis=-1)
        valid = (
            interval_mask
            & monotone
            & jnp.all(jnp.isfinite(moments), axis=-1)
            & jnp.all(jnp.isfinite(coarea_weight), axis=-1)
            & (jnp.max(jnp.abs(value - level), axis=-1) < tolerance)
        )
        return arc_r, arc_z, coarea_weight, moments, valid, jnp.abs(ordinate_gradient)

    independent_start = jnp.where(radial_primary, start[..., 0], start[..., 1])
    independent_end = jnp.where(radial_primary, end[..., 0], end[..., 1])
    independent_turning = jnp.where(radial_primary, turning[..., 0], turning[..., 1])
    coordinate_tolerance = jnp.asarray(1e-8, dtype=dtype)
    interior_turning = (
        independent_turning
        < jnp.minimum(independent_start, independent_end) - coordinate_tolerance
    ) | (
        independent_turning
        > jnp.maximum(independent_start, independent_end) + coordinate_tolerance
    )
    split_candidate = (
        (leaving_count == 1) & turning_valid[:, 0] & interior_turning[:, 0]
    )
    candidate_start = jnp.where(
        split_candidate[:, None, None],
        jnp.stack((start[:, 0], turning[:, 0]), axis=1),
        start,
    )
    candidate_end = jnp.where(
        split_candidate[:, None, None],
        jnp.stack((turning[:, 0], end[:, 0]), axis=1),
        end,
    )
    candidate_mask = jnp.where(
        split_candidate[:, None],
        jnp.broadcast_to(arc_mask[:, :1], arc_mask.shape),
        arc_mask,
    )
    candidate = interval_quadrature(candidate_start, candidate_end, candidate_mask)
    split_connected = split_candidate & jnp.all(
        (~candidate_mask) | candidate[4], axis=1
    )
    interval_start = jnp.where(split_connected[:, None, None], candidate_start, start)
    interval_end = jnp.where(split_connected[:, None, None], candidate_end, end)
    interval_mask = jnp.where(split_connected[:, None], candidate_mask, arc_mask)
    selected = interval_quadrature(interval_start, interval_end, interval_mask)
    interval_valid = selected[4]
    ordinate_gradient = selected[5]
    sample_mask = jnp.broadcast_to(interval_valid[..., None], selected[0].shape)

    root_points = crossing_points.reshape(
        crossing_points.shape[0], 4, _BICUBIC_EDGE_ROOT_CAPACITY, 2
    )
    root_mask = crossing.reshape(crossing.shape[0], 4, _BICUBIC_EDGE_ROOT_CAPACITY)
    root_fraction = jnp.stack(
        (
            root_points[:, 0, :, 0],
            root_points[:, 1, :, 1],
            one - root_points[:, 2, :, 0],
            one - root_points[:, 3, :, 1],
        ),
        axis=1,
    )
    root_fraction = jnp.sort(jnp.where(root_mask, root_fraction, one), axis=-1)
    boundary_bounds = jnp.concatenate(
        (
            jnp.zeros((*root_fraction.shape[:2], 1), dtype=dtype),
            root_fraction,
            jnp.ones((*root_fraction.shape[:2], 1), dtype=dtype),
        ),
        axis=-1,
    )
    boundary_low = boundary_bounds[..., :-1]
    boundary_high = boundary_bounds[..., 1:]
    boundary_mid = 0.5 * (boundary_low + boundary_high)
    boundary_edge = jnp.broadcast_to(jnp.arange(4)[None, :, None], boundary_mid.shape)
    mid_r, mid_z = _bicubic_edge_coordinates(boundary_edge, boundary_mid)
    mid_value = _bicubic_derivatives(coefficient[:, None, None], mid_r, mid_z)[0]
    boundary_mask = (boundary_high > boundary_low) & (mid_value < level)
    boundary_parameter = (
        boundary_low[..., None] + (boundary_high - boundary_low)[..., None] * point
    )
    boundary_r, boundary_z = _bicubic_edge_coordinates(
        boundary_edge[..., None], boundary_parameter
    )
    local_r = dr * (boundary_r - 0.5)
    local_z = dz * (boundary_z - 0.5)
    parameter_measure = (boundary_high - boundary_low)[..., None] * weight
    radial_measure = jnp.where(
        (boundary_edge == 0)[..., None],
        dr * parameter_measure,
        jnp.where((boundary_edge == 2)[..., None], -dr * parameter_measure, zero),
    )
    boundary_moments = jnp.stack(
        (
            jnp.sum(-local_z * radial_measure, axis=-1),
            jnp.sum(
                -local_r * local_z * radial_measure,
                axis=-1,
            ),
            jnp.sum(
                -0.5 * local_z**2 * radial_measure,
                axis=-1,
            ),
            jnp.sum(
                -0.5 * local_r * local_z**2 * radial_measure,
                axis=-1,
            ),
        ),
        axis=-1,
    )
    boundary_moments = jnp.sum(
        jnp.where(boundary_mask[..., None], boundary_moments, zero), axis=(1, 2)
    )
    arc_moments = jnp.sum(
        jnp.where(interval_valid[..., None], selected[3], zero), axis=1
    )
    chord_r = start[..., 0, None] + (end[..., 0] - start[..., 0])[..., None] * point
    chord_z = start[..., 1, None] + (end[..., 1] - start[..., 1])[..., None] * point
    local_chord_r = dr * (chord_r - 0.5)
    local_chord_z = dz * (chord_z - 0.5)
    chord_measure = dr * (end[..., 0] - start[..., 0])[..., None] * weight
    chord_moments = jnp.stack(
        (
            -jnp.sum(local_chord_z * chord_measure, axis=-1),
            -jnp.sum(local_chord_r * local_chord_z * chord_measure, axis=-1),
            -0.5 * jnp.sum(local_chord_z**2 * chord_measure, axis=-1),
            -0.5 * jnp.sum(local_chord_r * local_chord_z**2 * chord_measure, axis=-1),
        ),
        axis=-1,
    )
    chord_moments = jnp.sum(jnp.where(arc_mask[..., None], chord_moments, zero), axis=1)
    root_count = jnp.sum(crossing, axis=1)
    topology_valid = (
        (root_count <= 2 * _BICUBIC_ARC_CAPACITY)
        & ((root_count % 2) == 0)
        & (leaving_count == entering_count)
        & (leaving_count <= _BICUBIC_ARC_CAPACITY)
    )
    valid = topology_valid & jnp.all((~interval_mask) | interval_valid, axis=1)
    exact_correction = boundary_moments + arc_moments - base_moments
    chord_correction = arc_moments - chord_moments
    correction = jnp.where(
        (root_count == 2)[:, None], chord_correction, exact_correction
    )
    correction = jnp.where((root_count == 0)[:, None], zero, correction)
    correction = jnp.where(valid[:, None], correction, zero)
    return (
        correction,
        crossing_points,
        crossing,
        selected[0].reshape(selected[0].shape[0], -1),
        selected[1].reshape(selected[1].shape[0], -1),
        selected[2].reshape(selected[2].shape[0], -1),
        sample_mask.reshape(sample_mask.shape[0], -1),
        ordinate_gradient.reshape(ordinate_gradient.shape[0], -1),
        valid,
    )


def _single_arc_moment_correction(level, coefficient, corner_flux, dr, dz):
    """Return the chord correction for one endpoint-sign-changing cell arc."""
    all_points, all_crossings, inside = _bicubic_edge_crossings(
        level, coefficient, corner_flux, detect_even_roots=False
    )
    crossing_points = all_points.reshape(-1, 4, _BICUBIC_EDGE_ROOT_CAPACITY, 2)[:, :, 0]
    crossing = all_crossings.reshape(-1, 4, _BICUBIC_EDGE_ROOT_CAPACITY)[:, :, 0]
    following_inside = jnp.roll(inside, -1, axis=1)
    leaving = crossing & inside & ~following_inside
    entering = crossing & ~inside & following_inside
    leaving_index = jnp.argmax(leaving, axis=1)
    entering_index = jnp.argmax(entering, axis=1)
    start = jnp.take_along_axis(crossing_points, leaving_index[:, None, None], axis=1)[
        :, 0
    ]
    end = jnp.take_along_axis(crossing_points, entering_index[:, None, None], axis=1)[
        :, 0
    ]

    point = jnp.asarray(_BICUBIC_ARC_POINT, dtype=corner_flux.dtype)
    weight = jnp.asarray(_BICUBIC_ARC_WEIGHT, dtype=corner_flux.dtype)
    radial_span = end[:, 0] - start[:, 0]
    vertical_span = end[:, 1] - start[:, 1]
    radial_parameter = start[:, 0, None] + radial_span[:, None] * point
    vertical_seed = start[:, 1, None] + vertical_span[:, None] * point
    vertical_arc = _solve_bicubic_ordinate(
        coefficient[:, None],
        level,
        radial_parameter,
        vertical_seed,
        solve_vertical=True,
    )
    vertical_parameter = start[:, 1, None] + vertical_span[:, None] * point
    radial_seed = start[:, 0, None] + radial_span[:, None] * point
    radial_arc = _solve_bicubic_ordinate(
        coefficient[:, None],
        level,
        vertical_parameter,
        radial_seed,
        solve_vertical=False,
    )
    use_radial = jnp.abs(radial_span) >= jnp.abs(vertical_span)
    arc_radial = jnp.where(use_radial[:, None], radial_parameter, radial_arc)
    arc_vertical = jnp.where(use_radial[:, None], vertical_arc, vertical_parameter)

    local_radial = dr * (radial_parameter - 0.5)
    arc_height = dz * (vertical_arc - 0.5)
    line_height = dz * (vertical_seed - 0.5)
    radial_measure = dr * radial_span[:, None] * weight
    height_difference = arc_height - line_height
    height_squared_difference = arc_height**2 - line_height**2
    radial_correction = jnp.stack(
        (
            -jnp.sum(height_difference * radial_measure, axis=1),
            -jnp.sum(local_radial * height_difference * radial_measure, axis=1),
            -0.5 * jnp.sum(height_squared_difference * radial_measure, axis=1),
            -0.5
            * jnp.sum(
                local_radial * height_squared_difference * radial_measure, axis=1
            ),
        ),
        axis=1,
    )

    local_vertical = dz * (vertical_parameter - 0.5)
    arc_radius = dr * (radial_arc - 0.5)
    line_radius = dr * (radial_seed - 0.5)
    vertical_measure = dz * vertical_span[:, None] * weight
    radius_difference = arc_radius - line_radius
    radius_squared_difference = arc_radius**2 - line_radius**2
    vertical_correction = jnp.stack(
        (
            jnp.sum(radius_difference * vertical_measure, axis=1),
            0.5 * jnp.sum(radius_squared_difference * vertical_measure, axis=1),
            jnp.sum(local_vertical * radius_difference * vertical_measure, axis=1),
            0.5
            * jnp.sum(
                local_vertical * radius_squared_difference * vertical_measure,
                axis=1,
            ),
        ),
        axis=1,
    )
    correction = jnp.where(use_radial[:, None], radial_correction, vertical_correction)
    arc_value = _bicubic_derivatives(coefficient[:, None], arc_radial, arc_vertical)[0]
    two_crossings = jnp.sum(crossing, axis=1) == 2
    tolerance = jnp.asarray(2.0e-8, dtype=corner_flux.dtype) * jnp.maximum(
        jnp.ones((), dtype=corner_flux.dtype), jnp.abs(level)
    )
    valid = (
        two_crossings
        & jnp.all(jnp.isfinite(correction), axis=1)
        & (jnp.max(jnp.abs(arc_value - level), axis=1) < tolerance)
    )
    correction = jnp.where(valid[:, None], correction, 0.0)
    return correction, crossing_points, crossing, arc_radial, arc_vertical, valid


def _bicubic_stationary_point(coefficient, level, radial, vertical, *, radial_extremum):
    """Refine a coordinate extremum constrained to a bicubic level set."""
    dtype = radial.dtype
    level = jnp.asarray(level, dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    determinant_floor = jnp.asarray(1e-14, dtype=dtype)
    maximum_step = jnp.asarray(0.2, dtype=dtype)
    lower_bound = jnp.asarray(-0.1, dtype=dtype)
    upper_bound = jnp.asarray(1.1, dtype=dtype)

    def update(_, state):
        current_radial, current_vertical = state
        value, gradient_r, gradient_z, hessian_rr, hessian_rz, hessian_zz = (
            _bicubic_derivatives(coefficient, current_radial, current_vertical)
        )
        constrained_gradient = jnp.where(radial_extremum, gradient_z, gradient_r)
        second_r = jnp.where(radial_extremum, hessian_rz, hessian_rr)
        second_z = jnp.where(radial_extremum, hessian_zz, hessian_rz)
        determinant = gradient_r * second_z - gradient_z * second_r
        safe_determinant = jnp.where(
            jnp.abs(determinant) > determinant_floor,
            determinant,
            jnp.where(determinant < zero, -determinant_floor, determinant_floor),
        )
        residual = value - level
        radial_step = (second_z * residual - gradient_z * constrained_gradient) / (
            safe_determinant
        )
        vertical_step = (
            -second_r * residual + gradient_r * constrained_gradient
        ) / safe_determinant
        return (
            jnp.clip(
                current_radial - jnp.clip(radial_step, -maximum_step, maximum_step),
                lower_bound,
                upper_bound,
            ),
            jnp.clip(
                current_vertical - jnp.clip(vertical_step, -maximum_step, maximum_step),
                lower_bound,
                upper_bound,
            ),
        )

    radial, vertical = jax.lax.fori_loop(0, 10, update, (radial, vertical))
    value, gradient_r, gradient_z, *_ = _bicubic_derivatives(
        coefficient, radial, vertical
    )
    constrained_gradient = jnp.where(radial_extremum, gradient_z, gradient_r)
    level_tolerance = jnp.asarray(2.0e-8, dtype=dtype) * jnp.maximum(
        one, jnp.abs(level)
    )
    gradient_tolerance = jnp.asarray(2.0e-7, dtype=dtype)
    valid = (
        (radial >= zero)
        & (radial <= one)
        & (vertical >= zero)
        & (vertical <= one)
        & (jnp.abs(value - level) < level_tolerance)
        & (jnp.abs(constrained_gradient) < gradient_tolerance)
    )
    return radial, vertical, valid


def _integrate_bilinear(supports, corner_values, dr, dz, correction):
    """Integrate cellwise bilinear values over bicubic level-set clips."""
    lower_left, lower_right, upper_right, upper_left = jnp.moveaxis(
        corner_values, -1, 0
    )
    constant = 0.25 * (lower_left + lower_right + upper_right + upper_left)
    radial = 0.5 * (-lower_left + lower_right + upper_right - upper_left)
    vertical = 0.5 * (-lower_left - lower_right + upper_right + upper_left)
    cross = lower_left - lower_right + upper_right - upper_left
    area = supports.area + correction[:, 0]
    radial_moment = supports.first_area_moment[:, 0] + correction[:, 1]
    vertical_moment = supports.first_area_moment[:, 1] + correction[:, 2]
    mixed_moment = supports.second_area_moment[:, 0, 1] + correction[:, 3]
    return (
        constant * area
        + radial * radial_moment / dr
        + vertical * vertical_moment / dz
        + cross * mixed_moment / (dr * dz)
    )


def _clipped_surface_geometry(
    psi2d,
    psi_n_grid,
    core,
    radius,
    height,
    f_grid,
    psi_n_min,
    psi_n_max,
    n_surface_bins,
):
    """Return exact clipped-cell coarea bins and TORAX surface columns."""
    from nova.equilibrium.separatrix_clip import _traced_clip

    dtype = psi2d.dtype
    nz, nr = psi2d.shape
    dr = radius[1] - radius[0]
    dz = height[1] - height[0]
    node_index = jnp.arange(nz * nr).reshape(nz, nr)
    cell_nodes = jnp.stack(
        (
            node_index[:-1, :-1],
            node_index[:-1, 1:],
            node_index[1:, 1:],
            node_index[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1, 4)
    mesh_radius, mesh_height = jnp.meshgrid(radius, height)
    node_coordinates = jnp.stack((mesh_radius, mesh_height), axis=-1).reshape(-1, 2)
    centroids = jnp.mean(node_coordinates[cell_nodes], axis=1)
    cell_count = cell_nodes.shape[0]
    vertex_count = jnp.full(cell_count, 4, dtype=jnp.int32)

    psi_n_cells = psi_n_grid.reshape(-1)[cell_nodes]
    normalised_coefficient = _tensor_bicubic_coefficients(psi_n_grid)
    physical_coefficient = _tensor_bicubic_coefficients(psi2d)
    corner_radial = jnp.asarray((0.0, 1.0, 1.0, 0.0), dtype=dtype)
    corner_vertical = jnp.asarray((0.0, 0.0, 1.0, 1.0), dtype=dtype)
    _, gradient_r, gradient_z, *_ = _bicubic_derivatives(
        physical_coefficient[:, None], corner_radial, corner_vertical
    )
    gradient_flux = jnp.sqrt((gradient_r / dr) ** 2 + (gradient_z / dz) ** 2)
    radius_cells = node_coordinates[cell_nodes, 0]
    field_cells = f_grid.reshape(-1)[cell_nodes]
    gradient_psi = gradient_flux / _TWO_PI
    magnetic_field_squared = (gradient_psi**2 + field_cells**2) / radius_cells**2
    volume_weighted_values = jnp.stack(
        (
            _TWO_PI * radius_cells,
            _TWO_PI / radius_cells,
            jnp.full_like(radius_cells, _TWO_PI),
            _TWO_PI * gradient_flux**2 / radius_cells,
            _TWO_PI * radius_cells * gradient_psi,
            _TWO_PI * radius_cells * gradient_psi**2,
            _TWO_PI * gradient_psi**2 / radius_cells,
            _TWO_PI * radius_cells * magnetic_field_squared,
            _TWO_PI * radius_cells / jnp.maximum(magnetic_field_squared, 1e-30),
        ),
        axis=0,
    )
    eligible = (core > 0.0) | (psi_n_grid >= 1.0)
    flat_flux = psi_n_grid.reshape(-1)
    flat_eligible = eligible.reshape(-1)
    support_capacity = 8

    def signed_flux(level):
        return jnp.where(flat_eligible, level - flat_flux, -1.0)

    def required_vertex_count(level):
        inside = signed_flux(level)[cell_nodes] > 0.0
        crossing = inside != jnp.roll(inside, -1, axis=1)
        return jnp.max(jnp.sum(inside, axis=1) + jnp.sum(crossing, axis=1))

    def clip(level):
        return _traced_clip(
            node_coordinates,
            cell_nodes,
            vertex_count,
            centroids,
            support_capacity,
            signed_flux(level),
        )

    def cumulative(level):
        supports = clip(level)
        correction, *_ = _single_arc_moment_correction(
            level, normalised_coefficient, psi_n_cells, dr, dz
        )
        integrals = jax.vmap(
            lambda values: jnp.sum(
                _integrate_bilinear(supports, values, dr, dz, correction)
            )
        )(volume_weighted_values)
        return (
            integrals,
            jnp.max(supports.vertex_count),
            required_vertex_count(level),
        )

    levels = jnp.linspace(psi_n_min, psi_n_max, n_surface_bins + 1, dtype=dtype)
    psi_n_surface = 0.5 * (levels[:-1] + levels[1:])
    cumulative_values, cumulative_used, cumulative_required = jax.lax.map(
        cumulative, levels
    )
    shell_values = jnp.diff(cumulative_values, axis=0)
    shell_volume = jnp.maximum(shell_values[:, 0], 1e-30)
    surface_values = shell_values[:, 1:] / shell_volume[:, None]

    cell_origin = node_coordinates[cell_nodes[:, 0]]

    def extrema(level):
        supports = clip(level)
        _, crossing_points, crossing, arc_radial, arc_vertical, arc_valid = (
            _single_arc_moment_correction(
                level, normalised_coefficient, psi_n_cells, dr, dz
            )
        )
        samples = jnp.concatenate(
            (crossing_points, jnp.stack((arc_radial, arc_vertical), axis=-1)),
            axis=1,
        )
        sample_valid = jnp.concatenate(
            (crossing, jnp.broadcast_to(arc_valid[:, None], arc_radial.shape)), axis=1
        )

        def initial(coordinate, largest):
            fill = -jnp.inf if largest else jnp.inf
            candidate = jnp.where(sample_valid, samples[..., coordinate], fill)
            index = (
                jnp.argmax(candidate, axis=1)
                if largest
                else jnp.argmin(candidate, axis=1)
            )
            return jnp.take_along_axis(samples, index[:, None, None], axis=1)[:, 0]

        radial_low = initial(0, False)
        radial_high = initial(0, True)
        vertical_low = initial(1, False)
        vertical_high = initial(1, True)
        stationary = []
        stationary_valid = []
        for seed, radial_extremum in (
            (radial_low, True),
            (radial_high, True),
            (vertical_low, False),
            (vertical_high, False),
        ):
            point_r, point_z, point_valid = _bicubic_stationary_point(
                normalised_coefficient,
                level,
                seed[:, 0],
                seed[:, 1],
                radial_extremum=radial_extremum,
            )
            stationary.append(jnp.stack((point_r, point_z), axis=-1))
            stationary_valid.append(point_valid & arc_valid)
        samples = jnp.concatenate((samples, jnp.stack(stationary, axis=1)), axis=1)
        sample_valid = jnp.concatenate(
            (sample_valid, jnp.stack(stationary_valid, axis=1)), axis=1
        )
        radial = cell_origin[:, None, 0] + dr * samples[..., 0]
        vertical = cell_origin[:, None, 1] + dz * samples[..., 1]
        flat_valid = sample_valid.reshape(-1)
        flat_radial = radial.reshape(-1)
        flat_vertical = vertical.reshape(-1)
        r_in = jnp.min(jnp.where(flat_valid, flat_radial, jnp.inf))
        r_out = jnp.max(jnp.where(flat_valid, flat_radial, -jnp.inf))
        lower_slot = jnp.argmin(jnp.where(flat_valid, flat_vertical, jnp.inf))
        upper_slot = jnp.argmax(jnp.where(flat_valid, flat_vertical, -jnp.inf))
        return jnp.asarray(
            (
                r_in,
                r_out,
                flat_vertical[lower_slot],
                flat_vertical[upper_slot],
                flat_radial[lower_slot],
                flat_radial[upper_slot],
                jnp.max(supports.vertex_count),
                required_vertex_count(level),
            )
        )

    surface_extrema = jax.lax.map(extrema, psi_n_surface)
    (
        r_in,
        r_out,
        z_lower,
        z_upper,
        r_lower,
        r_upper,
        surface_used,
        surface_required,
    ) = surface_extrema.T
    edge_extrema = extrema(jnp.asarray(1.0, dtype=dtype))
    (
        edge_r_in,
        edge_r_out,
        edge_z_lower,
        edge_z_upper,
        edge_r_lower,
        edge_r_upper,
        edge_used,
        edge_required,
    ) = edge_extrema
    local_major = 0.5 * (r_in + r_out)
    local_minor = jnp.maximum(0.5 * (r_out - r_in), 1e-12)
    edge_major = 0.5 * (edge_r_in + edge_r_out)
    edge_minor = jnp.maximum(0.5 * (edge_r_out - edge_r_in), 1e-12)
    total_values, total_used, total_required = cumulative(jnp.asarray(1.0, dtype=dtype))
    total_volume = total_values[0]
    maximum_used = jnp.max(
        jnp.concatenate(
            (
                cumulative_used,
                surface_used,
                jnp.atleast_1d(edge_used),
                jnp.atleast_1d(total_used),
            )
        )
    )
    maximum_required = jnp.max(
        jnp.concatenate(
            (
                cumulative_required,
                surface_required,
                jnp.atleast_1d(edge_required),
                jnp.atleast_1d(total_required),
            )
        )
    )
    axis_position = jnp.argmin(jnp.where(core > 0, psi_n_grid, jnp.inf))
    axis_radius = mesh_radius.reshape(-1)[axis_position]
    dlevel = (psi_n_max - psi_n_min) / n_surface_bins
    return (
        {
            "pn_s": psi_n_surface,
            "dv_dpn": shell_values[:, 0] / dlevel,
            "inv_r2": surface_values[:, 0],
            "inv_r": surface_values[:, 1],
            "grad2_r2": surface_values[:, 2],
            "v_cum": 0.5 * (cumulative_values[:-1, 0] + cumulative_values[1:, 0]),
            "v_total": total_volume,
        },
        {
            "grad_psi_surface": surface_values[:, 3],
            "grad_psi2_surface": surface_values[:, 4],
            "grad_psi2_over_r2_surface": surface_values[:, 5],
            "b2_surface": surface_values[:, 6],
            "inv_b2_surface": surface_values[:, 7],
            "r_in_surface": r_in,
            "r_in_edge": edge_r_in,
            "r_out_surface": r_out,
            "r_out_edge": edge_r_out,
            "elongation_surface": (z_upper - z_lower) / (2.0 * local_minor),
            "elongation_edge": (edge_z_upper - edge_z_lower) / (2.0 * edge_minor),
            "delta_upper_surface": (local_major - r_upper) / local_minor,
            "delta_upper_edge": (edge_major - edge_r_upper) / edge_minor,
            "delta_lower_surface": (local_major - r_lower) / local_minor,
            "delta_lower_edge": (edge_major - edge_r_lower) / edge_minor,
            "axis_radius": jnp.asarray(axis_radius, dtype=dtype),
            "clipped_vertex_count_max": maximum_used,
            "clipped_vertex_count_required": maximum_required,
            "clipped_vertex_capacity": jnp.asarray(support_capacity),
        },
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_pressure",
        "n_diamagnetic",
        "n_radial_cells",
        "nonnegative",
    ),
)
def traced_assemble_flux_surface_geometry(
    surface_bins,
    psi2d,
    radius,
    height,
    inside_limiter,
    *,
    axis_psi,
    boundary_psi,
    profile_coefficients,
    coefficient_scale,
    ip_amperes,
    major_radius,
    boundary_toroidal_field,
    field_function_psi_n=None,
    field_function=None,
    n_pressure: int,
    n_diamagnetic: int,
    n_radial_cells: int = 24,
    nonnegative: bool = True,
    torax_columns=None,
):
    """Assemble fixed-shape transport metrics from connectivity surface bins.

    The input grid is ``psi2d[height, radius]`` in total poloidal flux [Wb].
    ``surface_bins`` is the output of
    :func:`nova.equilibrium.flux_surface_connectivity.traced_flux_surface_bins`:
    its
    ``n_surface_bins`` arrays live on increasing normalised-poloidal-flux
    mid-levels and may be nonuniform. ``profile_coefficients`` and
    ``coefficient_scale`` are the pressure-gradient columns followed by the
    diamagnetic columns; their product has toroidal-current-density units
    [A m-2]. A reader may instead supply ``field_function`` on
    ``field_function_psi_n``; that preserves the equilibrium's measured F and
    poloidal-flux profiles while the coarea moments supply its current profile.

    The returned dictionary is a JAX PyTree. Face arrays have
    ``n_radial_cells + 1`` entries, cell arrays have ``n_radial_cells`` entries,
    and all shapes are independent of the number of confined grid cells. The
    scalar ``valid`` is false for empty or ill-posed surfaces; callers can
    discard that slice without fabricating geometry.
    """
    dtype = jnp.asarray(psi2d).dtype
    psi2d = jnp.asarray(psi2d, dtype=dtype)
    radius = jnp.asarray(radius, dtype=dtype)
    height = jnp.asarray(height, dtype=dtype)
    inside_limiter = jnp.asarray(inside_limiter, dtype=bool)
    profile_coefficients = jnp.asarray(profile_coefficients, dtype=dtype)
    coefficient_scale = jnp.asarray(coefficient_scale, dtype=dtype)
    if (field_function is None) != (field_function_psi_n is None):
        raise ValueError(
            "field_function and field_function_psi_n must be supplied together"
        )

    poloidal_flux_span = boundary_psi - axis_psi
    safe_span = jnp.where(
        jnp.abs(poloidal_flux_span) > 1e-12, poloidal_flux_span, 1e-12
    )
    psi_n_grid = (psi2d - axis_psi) / safe_span
    core = _connectivity_core(psi_n_grid, inside_limiter)
    scaled_coefficients = profile_coefficients * coefficient_scale
    psi_n_profile = jnp.linspace(0.0, 1.0, 101, dtype=dtype)
    if field_function is None:
        diamagnetic_profile = (
            _traced_profile_shapes(
                psi_n_profile, n_diamagnetic, nonnegative=nonnegative
            )
            @ scaled_coefficients[n_pressure:]
        )
        f_profile, f_well_posed = _integrate_diamagnetic_drive(
            psi_n_profile,
            MU0 * major_radius * diamagnetic_profile,
            boundary_f=major_radius * boundary_toroidal_field,
            poloidal_flux_span=safe_span,
        )
    else:
        supplied_psi_n = jnp.asarray(field_function_psi_n, dtype=dtype)
        supplied_field_function = jnp.asarray(field_function, dtype=dtype)
        f_profile = jnp.interp(psi_n_profile, supplied_psi_n, supplied_field_function)
        f_well_posed = (
            (supplied_psi_n.ndim == 1)
            & (supplied_field_function.shape == supplied_psi_n.shape)
            & jnp.all(jnp.diff(supplied_psi_n) > 0.0)
            & jnp.all(jnp.isfinite(supplied_field_function))
            & jnp.all(jnp.abs(supplied_field_function) > 0.0)
        )

    psi_n_surface = jnp.asarray(surface_bins["pn_s"], dtype=dtype)
    direct_surface_integrals = "dv_dpn_edge" in surface_bins
    volume_derivative = jnp.asarray(surface_bins["dv_dpn"], dtype=dtype)
    volume_derivative_edge = jnp.asarray(
        surface_bins.get("dv_dpn_edge", volume_derivative[-1]), dtype=dtype
    )
    inverse_radius_squared = jnp.asarray(surface_bins["inv_r2"], dtype=dtype)
    inverse_radius_squared_edge = jnp.asarray(
        surface_bins.get("inv_r2_edge", inverse_radius_squared[-1]), dtype=dtype
    )
    inverse_radius = jnp.asarray(surface_bins["inv_r"], dtype=dtype)
    inverse_radius_edge = jnp.asarray(
        surface_bins.get("inv_r_edge", inverse_radius[-1]), dtype=dtype
    )
    gradient_squared_over_radius_squared = jnp.asarray(
        surface_bins["grad2_r2"], dtype=dtype
    )
    gradient_squared_over_radius_squared_edge = jnp.asarray(
        surface_bins.get("grad2_r2_edge", gradient_squared_over_radius_squared[-1]),
        dtype=dtype,
    )
    cumulative_volume = jnp.asarray(surface_bins["v_cum"], dtype=dtype)
    volume = jnp.asarray(surface_bins["v_total"], dtype=dtype)

    f_surface = jnp.interp(psi_n_surface, psi_n_profile, f_profile)
    f_grid = jnp.interp(psi_n_grid, psi_n_profile, f_profile)
    if torax_columns is None:
        _, torax_columns = _clipped_surface_geometry(
            psi2d,
            psi_n_grid,
            core,
            radius,
            height,
            f_grid,
            psi_n_surface[0] - 0.5 * (psi_n_surface[1] - psi_n_surface[0]),
            psi_n_surface[-1] + 0.5 * (psi_n_surface[-1] - psi_n_surface[-2]),
            psi_n_surface.size,
        )
    volume_derivative_per_flux = volume_derivative / jnp.abs(safe_span)
    safety_factor = (
        jnp.abs(f_surface)
        * inverse_radius_squared
        * volume_derivative_per_flux
        / _TWO_PI
    )
    safety_factor_edge = (
        jnp.abs(f_profile[-1])
        * inverse_radius_squared_edge
        * volume_derivative_edge
        / (jnp.abs(safe_span) * _TWO_PI)
    )
    toroidal_flux_surface = _cumulative_trapezoid_from_axis(
        psi_n_surface, safety_factor * jnp.abs(safe_span)
    )
    edge_safety_factor = (
        0.5 * (safety_factor[-1] + safety_factor_edge)
        if direct_surface_integrals
        else safety_factor[-1]
    )
    boundary_toroidal_flux = toroidal_flux_surface[-1] + (
        1.0 - psi_n_surface[-1]
    ) * edge_safety_factor * jnp.abs(safe_span)
    safe_boundary_toroidal_flux = jnp.maximum(boundary_toroidal_flux, 1e-30)
    rho_surface = jnp.sqrt(
        jnp.clip(toroidal_flux_surface / safe_boundary_toroidal_flux, 0.0, 1.0)
    )
    rho_surface = jax.lax.associative_scan(jnp.maximum, rho_surface)

    magnetic_field_squared = (
        gradient_squared_over_radius_squared / (4.0 * jnp.pi**2)
        + f_surface**2 * inverse_radius_squared
    )
    rho_face = jnp.linspace(0.0, 1.0, n_radial_cells + 1, dtype=dtype)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    psi_n_face = _surface_interpolation(rho_face, rho_surface, psi_n_surface, 0.0, 1.0)
    psi_n_cell = _surface_interpolation(rho_cell, rho_surface, psi_n_surface, 0.0, 1.0)
    f_face = _surface_interpolation(
        rho_face,
        rho_surface,
        f_surface,
        f_profile[0],
        major_radius * boundary_toroidal_field,
    )
    f_cell = 0.5 * (f_face[:-1] + f_face[1:])
    g3_face = _surface_interpolation(
        rho_face,
        rho_surface,
        inverse_radius_squared,
        inverse_radius_squared[0],
        inverse_radius_squared_edge,
    )
    g3_cell = 0.5 * (g3_face[:-1] + g3_face[1:])
    inverse_radius_cell = jnp.interp(psi_n_cell, psi_n_surface, inverse_radius)
    inverse_radius_face = _surface_interpolation(
        rho_face,
        rho_surface,
        inverse_radius,
        inverse_radius[0],
        inverse_radius_edge,
    )
    magnetic_field_squared_cell = jnp.interp(
        psi_n_cell, psi_n_surface, magnetic_field_squared
    )
    safety_factor_face = _surface_interpolation(
        rho_face,
        rho_surface,
        safety_factor,
        safety_factor[0],
        safety_factor_edge,
    )

    volume_face = _surface_interpolation(
        rho_face, rho_surface, cumulative_volume, 0.0, volume
    )
    volume_face = jax.lax.associative_scan(jnp.maximum, jnp.nan_to_num(volume_face))
    radial_spacing = 1.0 / n_radial_cells
    if direct_surface_integrals:
        volume_derivative_per_normalised_flux_face = _surface_interpolation(
            rho_face,
            rho_surface,
            volume_derivative,
            0.0,
            volume_derivative_edge,
        )
        normalised_flux_gradient = (
            2.0
            * rho_face
            * safe_boundary_toroidal_flux
            / jnp.maximum(
                jnp.abs(safety_factor_face * safe_span),
                jnp.asarray(1e-30, dtype=dtype),
            )
        )
        volume_derivative_face = (
            (volume_derivative_per_normalised_flux_face * normalised_flux_gradient)
            .at[0]
            .set(0.0)
        )
        volume_derivative_cell = 0.5 * (
            volume_derivative_face[:-1] + volume_derivative_face[1:]
        )
        volume_derivative_per_flux_face = _surface_interpolation(
            rho_face,
            rho_surface,
            volume_derivative_per_flux,
            volume_derivative_per_flux[0],
            volume_derivative_edge / jnp.abs(safe_span),
        )
        gradient_squared_face = _surface_interpolation(
            rho_face,
            rho_surface,
            gradient_squared_over_radius_squared,
            gradient_squared_over_radius_squared[0],
            gradient_squared_over_radius_squared_edge,
        )
    else:
        volume_derivative_face = jnp.gradient(volume_face, radial_spacing)
        volume_derivative_face = volume_derivative_face.at[0].set(0.0)
        volume_derivative_cell = jnp.diff(volume_face) / radial_spacing
        volume_derivative_per_flux_face = jnp.interp(
            psi_n_face, psi_n_surface, volume_derivative_per_flux
        )
        gradient_squared_face = jnp.interp(
            psi_n_face, psi_n_surface, gradient_squared_over_radius_squared
        )
    g2_face = (
        (volume_derivative_per_flux_face**2 * gradient_squared_face).at[0].set(0.0)
    )

    # Ampere's law supplies the enclosed current from the same coarea moments:
    # I = <Bp^2> int(dl/Bp) / mu0.  This keeps the geometry record tied to the
    # supplied equilibrium rather than to the profile ladder used by the solver.
    enclosed_current_surface = (
        gradient_squared_over_radius_squared
        / (4.0 * jnp.pi**2)
        * volume_derivative_per_flux
        / MU0
    )
    enclosed_current = _surface_interpolation(
        rho_face,
        rho_surface,
        enclosed_current_surface,
        0.0,
        enclosed_current_surface[-1],
    )
    current_edge = enclosed_current[-1]
    if field_function is None:
        enclosed_current = (
            enclosed_current
            * jnp.abs(ip_amperes)
            / jnp.maximum(jnp.abs(current_edge), 1e-30)
            * jnp.sign(current_edge)
        )
    enclosed_current = enclosed_current.at[0].set(0.0)

    diffusion_face = jnp.zeros_like(rho_face)
    diffusion_face = diffusion_face.at[1:].set(g2_face[1:] * g3_face[1:] / rho_face[1:])
    flux_sign = jnp.where(safe_span >= 0.0, 1.0, -1.0)
    safe_denominator = jnp.where(
        jnp.abs(diffusion_face[1:] * f_face[1:]) > 1e-30,
        diffusion_face[1:] * f_face[1:],
        1e-30,
    )
    poloidal_flux_gradient = (
        jnp.zeros_like(rho_face)
        .at[1:]
        .set(
            flux_sign
            * enclosed_current[1:]
            * (_16PI3 * MU0 * safe_boundary_toroidal_flux)
            / safe_denominator
        )
    )
    diffusion_mid = 0.5 * (diffusion_face[:-1] + diffusion_face[1:])
    f_mid = 0.5 * (f_face[:-1] + f_face[1:])
    current_mid = (
        flux_sign
        * diffusion_mid
        * 0.5
        * (poloidal_flux_gradient[:-1] + poloidal_flux_gradient[1:])
        * f_mid
        / (safe_boundary_toroidal_flux * _16PI3 * MU0)
    )
    readback_edge_current = 1.5 * current_mid[-1] - 0.5 * current_mid[-2]
    poloidal_flux_gradient = (
        poloidal_flux_gradient
        * jnp.abs(ip_amperes)
        / jnp.maximum(jnp.abs(readback_edge_current), 1e-30)
        * jnp.sign(readback_edge_current)
    )
    if field_function is None:
        psi_face = axis_psi + jnp.concatenate(
            [
                jnp.zeros(1, dtype=dtype),
                jnp.cumsum(
                    0.5
                    * (poloidal_flux_gradient[1:] + poloidal_flux_gradient[:-1])
                    * radial_spacing
                ),
            ]
        )
    else:
        psi_face = axis_psi + safe_span * psi_n_face

    axis_radius = torax_columns["axis_radius"]

    def torax_face(name, axis_value):
        values = torax_columns[name]
        edge_value = torax_columns.get(
            name.removesuffix("_surface") + "_edge", values[-1]
        )
        return _surface_interpolation(
            rho_face, rho_surface, values, axis_value, edge_value
        )

    grad_psi_face = torax_face("grad_psi_surface", 0.0)
    grad_psi2_face = torax_face("grad_psi2_surface", 0.0)
    grad_psi2_over_r2_face = torax_face("grad_psi2_over_r2_surface", 0.0)
    b2_face = torax_face("b2_surface", torax_columns["b2_surface"][0])
    inv_b2_face = torax_face("inv_b2_surface", torax_columns["inv_b2_surface"][0])
    r_in_face = torax_face("r_in_surface", axis_radius)
    r_out_face = torax_face("r_out_surface", axis_radius)
    elongation_face = torax_face(
        "elongation_surface", torax_columns["elongation_surface"][0]
    )
    delta_upper_face = torax_face(
        "delta_upper_surface", torax_columns["delta_upper_surface"][0]
    )
    delta_lower_face = torax_face(
        "delta_lower_surface", torax_columns["delta_lower_surface"][0]
    )
    int_dl_over_bp_face = volume_derivative_per_flux_face

    finite_arrays = (
        jnp.all(jnp.isfinite(psi_n_surface))
        & jnp.all(jnp.isfinite(volume_derivative))
        & jnp.all(jnp.isfinite(inverse_radius_squared))
        & jnp.all(jnp.isfinite(psi_face))
    )
    valid = (
        (jnp.abs(poloidal_flux_span) > 1e-12)
        & (jnp.sum(core) >= 200)
        & jnp.all(jnp.diff(psi_n_surface) > 0.0)
        & jnp.all(volume_derivative > 0.0)
        & (boundary_toroidal_flux > 0.0)
        & (jnp.abs(current_edge) > 1e-6 * jnp.maximum(jnp.abs(ip_amperes), 1.0))
        & jnp.all(diffusion_face[1:] > 0.0)
        & f_well_posed
        & finite_arrays
    )
    return {
        "rho_face": rho_face,
        "rho_cell": rho_cell,
        "psi_face": psi_face,
        "psi_n_face": psi_n_face,
        "psi_n_cell": psi_n_cell,
        "vpr_face": volume_derivative_face,
        "vpr_cell": volume_derivative_cell,
        "g2_face": g2_face,
        "g3_face": g3_face,
        "g3_cell": g3_cell,
        "f_face": f_face,
        "f_cell": f_cell,
        "b2_cell": magnetic_field_squared_cell,
        "inv_r_cell": inverse_radius_cell,
        "inv_r_face": inverse_radius_face,
        "phi_b": safe_boundary_toroidal_flux,
        "r0": major_radius,
        "ip_amperes": jnp.abs(ip_amperes),
        "axis_psi": axis_psi,
        "boundary_psi": boundary_psi,
        "volume": volume,
        "q_face": safety_factor_face,
        "volume_face": volume_face,
        "ip_profile_face": enclosed_current,
        "int_dl_over_bp_face": int_dl_over_bp_face,
        "grad_psi_face": grad_psi_face,
        "grad_psi2_face": grad_psi2_face,
        "grad_psi2_over_r2_face": grad_psi2_over_r2_face,
        "b2_face": b2_face,
        "inv_b2_face": inv_b2_face,
        "r_in_face": r_in_face,
        "r_out_face": r_out_face,
        "elongation_face": elongation_face,
        "delta_upper_face": delta_upper_face,
        "delta_lower_face": delta_lower_face,
        "clipped_vertex_count_max": torax_columns["clipped_vertex_count_max"],
        "clipped_vertex_count_required": torax_columns["clipped_vertex_count_required"],
        "clipped_vertex_capacity": torax_columns["clipped_vertex_capacity"],
        "gradient_moment_scale": jnp.asarray(
            _TWO_PI if field_function is not None else 1.0, dtype=dtype
        ),
        "flux_sign": flux_sign,
        "valid": valid,
    }


@partial(
    jax.jit,
    static_argnames=(
        "n_pressure",
        "n_diamagnetic",
        "n_radial_cells",
        "n_surface_bins",
        "nonnegative",
    ),
)
def traced_flux_surface_geometry(
    psi2d,
    radius,
    height,
    inside_limiter,
    *,
    axis_psi,
    boundary_psi,
    profile_coefficients,
    coefficient_scale,
    ip_amperes,
    major_radius,
    boundary_toroidal_field,
    field_function_psi_n=None,
    field_function=None,
    n_pressure: int,
    n_diamagnetic: int,
    n_radial_cells: int = 24,
    n_surface_bins: int = 28,
    psi_n_min=0.04,
    psi_n_max=0.985,
    nonnegative: bool = True,
):
    """Build fixed-shape flux-surface geometry directly from an equilibrium grid.

    This device entry point composes Nova's connectivity bins with
    :func:`traced_assemble_flux_surface_geometry`; it is safe under ``jit`` and
    ``vmap`` when slices share a machine grid and the static shape parameters.
    EQDSK-like callers may pass the paired ``field_function`` inputs to preserve
    the reader's measured F and poloidal-flux profiles.
    """
    poloidal_flux_span = boundary_psi - axis_psi
    safe_span = jnp.where(
        jnp.abs(poloidal_flux_span) > 1e-12, poloidal_flux_span, 1e-12
    )
    psi_n_grid = (psi2d - axis_psi) / safe_span
    core = _connectivity_core(psi_n_grid, inside_limiter)
    psi_n_profile = jnp.linspace(0.0, 1.0, 101, dtype=psi2d.dtype)
    if field_function is None:
        scaled_coefficients = profile_coefficients * coefficient_scale
        diamagnetic_profile = (
            _traced_profile_shapes(
                psi_n_profile, n_diamagnetic, nonnegative=nonnegative
            )
            @ scaled_coefficients[n_pressure:]
        )
        f_profile, _ = _integrate_diamagnetic_drive(
            psi_n_profile,
            MU0 * major_radius * diamagnetic_profile,
            boundary_f=major_radius * boundary_toroidal_field,
            poloidal_flux_span=safe_span,
        )
    else:
        f_profile = jnp.interp(
            psi_n_profile,
            jnp.asarray(field_function_psi_n, dtype=psi2d.dtype),
            jnp.asarray(field_function, dtype=psi2d.dtype),
        )
    f_grid = jnp.interp(psi_n_grid, psi_n_profile, f_profile)
    surface_bins, torax_columns = _clipped_surface_geometry(
        psi2d,
        psi_n_grid,
        core,
        radius,
        height,
        f_grid,
        psi_n_min,
        psi_n_max,
        n_surface_bins,
    )
    return traced_assemble_flux_surface_geometry(
        surface_bins,
        psi2d,
        radius,
        height,
        inside_limiter,
        axis_psi=axis_psi,
        boundary_psi=boundary_psi,
        profile_coefficients=profile_coefficients,
        coefficient_scale=coefficient_scale,
        ip_amperes=ip_amperes,
        major_radius=major_radius,
        boundary_toroidal_field=boundary_toroidal_field,
        field_function_psi_n=field_function_psi_n,
        field_function=field_function,
        n_pressure=n_pressure,
        n_diamagnetic=n_diamagnetic,
        n_radial_cells=n_radial_cells,
        nonnegative=nonnegative,
        torax_columns=torax_columns,
    )


def _axis_connected_core(psi_n, inside_limiter):
    """Return the axis-connected component inside the boundary surface."""
    confined = (psi_n < 1.0) & inside_limiter
    seed_position = jnp.argmin(jnp.where(confined, psi_n, jnp.inf).reshape(-1))
    seed = jnp.zeros(psi_n.size, dtype=bool).at[seed_position].set(True)
    return flood_fill_core(
        confined,
        seed.reshape(psi_n.shape),
        psi_n.shape[0] + psi_n.shape[1],
    )


def _regular_cell_mesh(radius, height):
    """Return fixed regular-grid cell connectivity and geometric payloads."""
    nz = height.size
    nr = radius.size
    node_index = jnp.arange(nz * nr).reshape(nz, nr)
    cell_nodes = jnp.stack(
        (
            node_index[:-1, :-1],
            node_index[:-1, 1:],
            node_index[1:, 1:],
            node_index[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1, 4)
    mesh_radius, mesh_height = jnp.meshgrid(radius, height)
    coordinates = jnp.stack((mesh_radius, mesh_height), axis=-1).reshape(-1, 2)
    centroids = jnp.mean(coordinates[cell_nodes], axis=1)
    vertex_count = jnp.full(cell_nodes.shape[0], 4, dtype=jnp.int32)
    return cell_nodes, coordinates, centroids, vertex_count, mesh_radius


def _masked_extremum(value, payload, valid, *, largest):
    """Reduce an extremum while carrying a same-slot payload."""
    fill = -jnp.inf if largest else jnp.inf
    candidate = jnp.where(valid, value, fill)
    slot = jnp.argmax(candidate) if largest else jnp.argmin(candidate)
    return candidate[slot], payload[slot]


def _surface_clips(
    psi2d,
    psi_n_grid,
    core,
    inside_limiter,
    radius,
    height,
    f_profile_psi_n,
    f_profile,
    psi_n_min,
    psi_n_max,
    n_surface_bins,
):
    """Integrate spline level sets and return FSA bins plus shape columns."""
    dtype = psi2d.dtype
    dr = radius[1] - radius[0]
    dz = height[1] - height[0]
    cell_nodes, coordinates, centroids, vertex_count, mesh_radius = _regular_cell_mesh(
        radius, height
    )
    normalised_spline = fit_tensor_spline(radius, height, psi_n_grid)
    physical_spline = fit_tensor_spline(radius, height, psi2d)
    normalised_coefficient = normalised_spline.cell_coefficients.reshape(
        -1, 4, 4
    ).astype(dtype)
    physical_coefficient = physical_spline.cell_coefficients.reshape(-1, 4, 4).astype(
        dtype
    )
    psi_n_cells = psi_n_grid.reshape(-1)[cell_nodes]

    radius_cells = coordinates[cell_nodes, 0]
    volume_weighted_values = jnp.stack(
        (
            _TWO_PI * radius_cells,
            _TWO_PI / radius_cells,
            jnp.full_like(radius_cells, _TWO_PI),
        ),
        axis=0,
    )

    flat_flux = psi_n_grid.reshape(-1)
    flat_eligible = ((core > 0.0) | (psi_n_grid >= 1.0)).reshape(-1)
    confined_seed = (core > 0.0) & inside_limiter
    seed_position = jnp.argmin(
        jnp.where(confined_seed, psi_n_grid, jnp.inf).reshape(-1)
    )
    seed = jnp.zeros(psi_n_grid.size, dtype=bool).at[seed_position].set(True)
    seed = seed.reshape(psi_n_grid.shape)

    def level_cell_participation(level):
        level_core = flood_fill_core(
            (psi_n_grid < level) & inside_limiter,
            seed,
            psi_n_grid.shape[0] + psi_n_grid.shape[1],
        )
        return jnp.any((level_core > 0.0).reshape(-1)[cell_nodes], axis=1)

    def signed_flux(level):
        return jnp.where(flat_eligible, level - flat_flux, -1.0)

    def required_vertex_count(level, participation):
        inside = signed_flux(level)[cell_nodes] > 0.0
        crossing = inside != jnp.roll(inside, -1, axis=1)
        count = jnp.sum(inside, axis=1) + jnp.sum(crossing, axis=1)
        return jnp.max(jnp.where(participation, count, 0))

    def clip(level, participation):
        supports = _traced_clip(
            coordinates,
            cell_nodes,
            vertex_count,
            centroids,
            _SUPPORT_CAPACITY,
            signed_flux(level),
        )
        return supports.qualify(participation)

    def corrected_clip(level, participation):
        supports = clip(level, participation)
        base_moments = jnp.stack(
            (
                supports.area,
                supports.first_area_moment[:, 0],
                supports.first_area_moment[:, 1],
                supports.second_area_moment[:, 0, 1],
            ),
            axis=1,
        )
        (
            correction,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            arc_weight,
            arc_sample_valid,
            arc_ordinate_derivative,
            arc_valid,
        ) = _bicubic_arc_moment_correction(
            level,
            normalised_coefficient,
            psi_n_cells,
            dr,
            dz,
            base_moments,
        )
        boundary_valid = jnp.all((~supports.boundary) | arc_valid)
        invalid_boundary = supports.boundary & ~arc_valid
        return (
            supports,
            correction,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            arc_weight,
            arc_sample_valid,
            arc_ordinate_derivative,
            arc_valid,
            boundary_valid,
            invalid_boundary,
        )

    def cumulative(level, participation):
        supports, correction, *_, boundary_valid, invalid_boundary = corrected_clip(
            level, participation
        )
        integrals = jax.vmap(
            lambda values: jnp.sum(
                _integrate_bilinear(supports, values, dr, dz, correction)
            )
        )(volume_weighted_values)
        return (
            integrals,
            jnp.max(supports.vertex_count),
            required_vertex_count(level, participation),
            boundary_valid,
            jnp.sum(invalid_boundary),
            jnp.argmax(invalid_boundary),
        )

    def arc_surface_average(level, participation):
        (
            supports,
            _,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            arc_weight,
            arc_sample_valid,
            arc_ordinate_derivative,
            arc_valid,
            boundary_valid,
            invalid_boundary,
        ) = corrected_clip(level, participation)
        _, physical_r, physical_z, *_ = _bicubic_derivatives(
            physical_coefficient[:, None], arc_r, arc_z
        )
        physical_gradient = jnp.hypot(physical_r / dr, physical_z / dz)
        radius_at_arc = cell_origin[:, None, 0] + dr * arc_r
        field_at_surface = jnp.interp(level, f_profile_psi_n, f_profile)
        gradient_psi = physical_gradient / _TWO_PI
        magnetic_field_squared = (
            gradient_psi**2 + field_at_surface**2
        ) / radius_at_arc**2

        sample_valid = supports.boundary[:, None] & arc_sample_valid
        weighted_volume = jnp.where(sample_valid, radius_at_arc * arc_weight, 0.0)
        denominator = jnp.sum(weighted_volume)
        safe_denominator = jnp.maximum(denominator, jnp.asarray(1e-30, dtype=dtype))
        integrands = jnp.stack(
            (
                1.0 / radius_at_arc**2,
                1.0 / radius_at_arc,
                physical_gradient**2 / radius_at_arc**2,
                gradient_psi,
                gradient_psi**2,
                gradient_psi**2 / radius_at_arc**2,
                magnetic_field_squared,
                1.0 / jnp.maximum(magnetic_field_squared, 1e-30),
            ),
            axis=0,
        )
        averages = jnp.sum(integrands * weighted_volume[None], axis=(1, 2))
        averages /= safe_denominator
        line_values = jnp.concatenate(
            (jnp.asarray((_TWO_PI * denominator,), dtype=dtype), averages)
        )
        minimum_ordinate_derivative = jnp.min(
            jnp.where(sample_valid, arc_ordinate_derivative, jnp.inf)
        )
        maximum_coarea_weight = jnp.max(jnp.where(sample_valid, arc_weight, 0.0))
        return (
            line_values,
            boundary_valid & (denominator > 0.0),
            jnp.sum(invalid_boundary),
            jnp.argmax(invalid_boundary),
            minimum_ordinate_derivative,
            maximum_coarea_weight,
        )

    levels = jnp.linspace(psi_n_min, psi_n_max, n_surface_bins + 1, dtype=dtype)
    surface_level = 0.5 * (levels[:-1] + levels[1:])
    cumulative_participation = jax.lax.map(level_cell_participation, levels)
    surface_participation = jax.lax.map(level_cell_participation, surface_level)
    interior_cumulative = jax.lax.map(
        lambda inputs: cumulative(inputs[0], inputs[1]),
        (levels[:-1], cumulative_participation[:-1]),
    )
    edge_cumulative = cumulative(levels[-1], cumulative_participation[-1])
    cumulative_values = jnp.concatenate(
        (interior_cumulative[0], edge_cumulative[0][None]), axis=0
    )
    cumulative_used = jnp.concatenate(
        (interior_cumulative[1], edge_cumulative[1][None])
    )
    cumulative_required = jnp.concatenate(
        (interior_cumulative[2], edge_cumulative[2][None])
    )
    cumulative_valid = jnp.concatenate(
        (interior_cumulative[3], edge_cumulative[3][None])
    )
    cumulative_invalid_count = jnp.concatenate(
        (interior_cumulative[4], edge_cumulative[4][None])
    )
    cumulative_invalid_cell = jnp.concatenate(
        (interior_cumulative[5], edge_cumulative[5][None])
    )
    cell_origin = coordinates[cell_nodes[:, 0]]
    interior_arc_average = jax.lax.map(
        lambda inputs: arc_surface_average(inputs[0], inputs[1]),
        (surface_level, surface_participation),
    )
    edge_arc_average = arc_surface_average(levels[-1], cumulative_participation[-1])

    def extrema(level, participation):
        (
            supports,
            _,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            _arc_weight,
            arc_sample_valid,
            _arc_ordinate_derivative,
            arc_valid,
            boundary_valid,
            invalid_boundary,
        ) = corrected_clip(level, participation)
        samples = jnp.concatenate(
            (crossing_points, jnp.stack((arc_r, arc_z), axis=-1)), axis=1
        )
        sample_valid = jnp.concatenate((crossing, arc_sample_valid), axis=1)
        sample_valid &= participation[:, None]

        def cell_seed(coordinate, *, largest):
            fill = -jnp.inf if largest else jnp.inf
            candidate = jnp.where(sample_valid, samples[..., coordinate], fill)
            index = (
                jnp.argmax(candidate, axis=1)
                if largest
                else jnp.argmin(candidate, axis=1)
            )
            return jnp.take_along_axis(samples, index[:, None, None], axis=1)[:, 0]

        seeds = (
            (cell_seed(0, largest=False), True),
            (cell_seed(0, largest=True), True),
            (cell_seed(1, largest=False), False),
            (cell_seed(1, largest=True), False),
        )
        stationary = []
        stationary_valid = []
        for seed, radial_extremum in seeds:
            point_r, point_z, point_valid = _bicubic_stationary_point(
                normalised_coefficient,
                level,
                seed[:, 0],
                seed[:, 1],
                radial_extremum=radial_extremum,
            )
            stationary.append(jnp.stack((point_r, point_z), axis=-1))
            stationary_valid.append(point_valid & arc_valid & participation)
        samples = jnp.concatenate((samples, jnp.stack(stationary, axis=1)), axis=1)
        sample_valid = jnp.concatenate(
            (sample_valid, jnp.stack(stationary_valid, axis=1)), axis=1
        )
        radial = cell_origin[:, None, 0] + dr * samples[..., 0]
        vertical = cell_origin[:, None, 1] + dz * samples[..., 1]
        flat_valid = sample_valid.reshape(-1)
        flat_radial = radial.reshape(-1)
        flat_vertical = vertical.reshape(-1)
        r_in, _ = _masked_extremum(
            flat_radial, flat_vertical, flat_valid, largest=False
        )
        r_out, _ = _masked_extremum(
            flat_radial, flat_vertical, flat_valid, largest=True
        )
        z_lower, r_lower = _masked_extremum(
            flat_vertical, flat_radial, flat_valid, largest=False
        )
        z_upper, r_upper = _masked_extremum(
            flat_vertical, flat_radial, flat_valid, largest=True
        )
        return jnp.asarray(
            (
                r_in,
                r_out,
                z_lower,
                z_upper,
                r_lower,
                r_upper,
                jnp.max(supports.vertex_count),
                required_vertex_count(level, participation),
                boundary_valid,
                jnp.sum(invalid_boundary),
                jnp.argmax(invalid_boundary),
            )
        )

    surface_extrema = jax.lax.map(
        lambda inputs: extrema(inputs[0], inputs[1]),
        (surface_level, surface_participation),
    )
    (
        r_in,
        r_out,
        z_lower,
        z_upper,
        r_lower,
        r_upper,
        surface_used,
        surface_required,
        surface_valid,
        surface_invalid_count,
        surface_invalid_cell,
    ) = surface_extrema.T
    edge_level = levels[-1]
    edge_extrema = extrema(edge_level, cumulative_participation[-1])
    (
        edge_r_in,
        edge_r_out,
        edge_z_lower,
        edge_z_upper,
        edge_r_lower,
        edge_r_upper,
        edge_used,
        edge_required,
        edge_valid,
        edge_invalid_count,
        edge_invalid_cell,
    ) = edge_extrema
    edge_supports, edge_correction, *_, edge_integral_valid, _ = corrected_clip(
        edge_level, cumulative_participation[-1]
    )
    edge_values = jax.vmap(
        lambda values: jnp.sum(
            _integrate_bilinear(edge_supports, values, dr, dz, edge_correction)
        )
    )(volume_weighted_values)
    edge_spacing = levels[-1] - levels[-2]
    boundary_fraction = (jnp.asarray(1.0, dtype=dtype) - edge_level) / edge_spacing
    total_values = edge_values + boundary_fraction * (
        cumulative_values[-1] - cumulative_values[-2]
    )
    local_major = 0.5 * (r_in + r_out)
    local_minor = jnp.maximum(0.5 * (r_out - r_in), 1e-12)
    edge_major = 0.5 * (edge_r_in + edge_r_out)
    edge_minor = jnp.maximum(0.5 * (edge_r_out - edge_r_in), 1e-12)
    maximum_used = jnp.max(
        jnp.concatenate((cumulative_used, surface_used, edge_used[None]))
    )
    maximum_required = jnp.max(
        jnp.concatenate((cumulative_required, surface_required, edge_required[None]))
    )
    axis_position = jnp.argmin(jnp.where(core > 0.0, psi_n_grid, jnp.inf))
    axis_radius = mesh_radius.reshape(-1)[axis_position]
    all_arcs_valid = (
        jnp.all(cumulative_valid)
        & jnp.all(interior_arc_average[1])
        & edge_arc_average[1]
        & jnp.all(surface_valid > 0.0)
        & (edge_valid > 0.0)
        & edge_integral_valid
    )
    diagnostic_count = jnp.concatenate(
        (
            cumulative_invalid_count,
            interior_arc_average[2],
            surface_invalid_count,
            edge_invalid_count[None],
        )
    )
    diagnostic_cell = jnp.concatenate(
        (
            cumulative_invalid_cell,
            interior_arc_average[3],
            surface_invalid_cell.astype(jnp.int32),
            edge_invalid_cell[None].astype(jnp.int32),
        )
    )
    diagnostic_level = jnp.concatenate(
        (levels, surface_level, surface_level, edge_level[None])
    )
    first_invalid_group = jnp.argmax(diagnostic_count > 0)
    arc_average = interior_arc_average[0]
    edge_arc_average_values = edge_arc_average[0]
    return (
        {
            "pn_s": surface_level,
            "dv_dpn": arc_average[:, 0],
            "dv_dpn_edge": edge_arc_average_values[0],
            "inv_r2": arc_average[:, 1],
            "inv_r2_edge": edge_arc_average_values[1],
            "inv_r": arc_average[:, 2],
            "inv_r_edge": edge_arc_average_values[2],
            "grad2_r2": arc_average[:, 3],
            "grad2_r2_edge": edge_arc_average_values[3],
            "v_cum": 0.5 * (cumulative_values[:-1, 0] + cumulative_values[1:, 0]),
            "v_total": total_values[0],
        },
        {
            "grad_psi_surface": arc_average[:, 4],
            "grad_psi_edge": edge_arc_average_values[4],
            "grad_psi2_surface": arc_average[:, 5],
            "grad_psi2_edge": edge_arc_average_values[5],
            "grad_psi2_over_r2_surface": arc_average[:, 6],
            "grad_psi2_over_r2_edge": edge_arc_average_values[6],
            "b2_surface": arc_average[:, 7],
            "b2_edge": edge_arc_average_values[7],
            "inv_b2_surface": arc_average[:, 8],
            "inv_b2_edge": edge_arc_average_values[8],
            "r_in_surface": r_in,
            "r_in_edge": edge_r_in,
            "r_out_surface": r_out,
            "r_out_edge": edge_r_out,
            "elongation_surface": (z_upper - z_lower) / (2.0 * local_minor),
            "elongation_edge": (edge_z_upper - edge_z_lower) / (2.0 * edge_minor),
            "delta_upper_surface": (local_major - r_upper) / local_minor,
            "delta_upper_edge": (edge_major - edge_r_upper) / edge_minor,
            "delta_lower_surface": (local_major - r_lower) / local_minor,
            "delta_lower_edge": (edge_major - edge_r_lower) / edge_minor,
            "axis_radius": jnp.asarray(axis_radius, dtype=dtype),
            "clipped_vertex_count_max": maximum_used,
            "clipped_vertex_count_required": maximum_required,
            "clipped_vertex_capacity": jnp.asarray(_SUPPORT_CAPACITY),
            "surface_arc_invalid_count": jnp.sum(diagnostic_count),
            "surface_arc_first_invalid_cell": diagnostic_cell[first_invalid_group],
            "surface_arc_first_invalid_level": diagnostic_level[first_invalid_group],
            "surface_arc_min_ordinate_derivative": jnp.min(
                jnp.concatenate((interior_arc_average[4], edge_arc_average[4][None]))
            ),
            "surface_arc_max_coarea_weight": jnp.max(
                jnp.concatenate((interior_arc_average[5], edge_arc_average[5][None]))
            ),
        },
        all_arcs_valid,
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_pressure",
        "n_diamagnetic",
        "n_radial_cells",
        "n_surface_bins",
        "nonnegative",
    ),
)
def extract_flux_surface_geometry(
    psi2d,
    radius,
    height,
    inside_limiter,
    *,
    axis_psi,
    boundary_psi,
    profile_coefficients,
    coefficient_scale,
    ip_amperes,
    major_radius,
    boundary_toroidal_field,
    field_function_psi_n=None,
    field_function=None,
    n_pressure: int,
    n_diamagnetic: int,
    n_radial_cells: int = 24,
    n_surface_bins: int = 28,
    psi_n_min=0.04,
    psi_n_max=0.985,
    nonnegative: bool = True,
):
    """Return the complete fixed-shape FSA record for one structured flux map."""
    psi2d = jnp.asarray(psi2d)
    radius = jnp.asarray(radius, dtype=psi2d.dtype)
    height = jnp.asarray(height, dtype=psi2d.dtype)
    inside_limiter = jnp.asarray(inside_limiter, dtype=bool)
    span = boundary_psi - axis_psi
    safe_span = jnp.where(jnp.abs(span) > 1e-12, span, 1e-12)
    psi_n_grid = (psi2d - axis_psi) / safe_span
    core = _axis_connected_core(psi_n_grid, inside_limiter)
    psi_n_profile = jnp.linspace(0.0, 1.0, 101, dtype=psi2d.dtype)
    if field_function is None:
        scaled = profile_coefficients * coefficient_scale
        diamagnetic_profile = (
            _traced_profile_shapes(
                psi_n_profile, n_diamagnetic, nonnegative=nonnegative
            )
            @ scaled[n_pressure:]
        )
        f_profile, _ = _integrate_diamagnetic_drive(
            psi_n_profile,
            MU0 * major_radius * diamagnetic_profile,
            boundary_f=major_radius * boundary_toroidal_field,
            poloidal_flux_span=safe_span,
        )
    else:
        f_profile = jnp.interp(
            psi_n_profile,
            jnp.asarray(field_function_psi_n, dtype=psi2d.dtype),
            jnp.asarray(field_function, dtype=psi2d.dtype),
        )
    surface_bins, torax_columns, arcs_valid = _surface_clips(
        psi2d,
        psi_n_grid,
        core,
        inside_limiter,
        radius,
        height,
        psi_n_profile,
        f_profile,
        psi_n_min,
        psi_n_max,
        n_surface_bins,
    )
    record = traced_assemble_flux_surface_geometry(
        surface_bins,
        psi2d,
        radius,
        height,
        inside_limiter,
        axis_psi=axis_psi,
        boundary_psi=boundary_psi,
        profile_coefficients=profile_coefficients,
        coefficient_scale=coefficient_scale,
        ip_amperes=ip_amperes,
        major_radius=major_radius,
        boundary_toroidal_field=boundary_toroidal_field,
        field_function_psi_n=field_function_psi_n,
        field_function=field_function,
        n_pressure=n_pressure,
        n_diamagnetic=n_diamagnetic,
        n_radial_cells=n_radial_cells,
        nonnegative=nonnegative,
        torax_columns=torax_columns,
    )
    return {
        **record,
        "valid": record["valid"] & arcs_valid,
        "surface_arc_valid": arcs_valid,
        "surface_arc_invalid_count": torax_columns["surface_arc_invalid_count"],
        "surface_arc_first_invalid_cell": torax_columns[
            "surface_arc_first_invalid_cell"
        ],
        "surface_arc_first_invalid_level": torax_columns[
            "surface_arc_first_invalid_level"
        ],
        "surface_arc_min_ordinate_derivative": torax_columns[
            "surface_arc_min_ordinate_derivative"
        ],
        "surface_arc_max_coarea_weight": torax_columns["surface_arc_max_coarea_weight"],
    }


__all__ = [
    "extract_flux_surface_geometry",
    "traced_assemble_flux_surface_geometry",
    "traced_flux_surface_geometry",
]
