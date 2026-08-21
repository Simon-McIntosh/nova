"""Traced structured-grid extraction of flux-surface-averaged geometry.

The composition and reuse boundaries follow
``scripts/geometry_service_reuse/report.md``: one axis-connected flood fill
establishes topology, a global tensor spline supplies every cell polynomial,
fixed-capacity clips carry Green-theorem moments, and masked reductions select
the constrained extrema.  The returned dictionary is the fixed-shape record
consumed by the transport geometry adapter.
"""

from functools import partial

import numpy as np

from nova.biot.greens import MU0
from nova.equilibrium.flux_surface_connectivity import flood_fill_core
from nova.equilibrium.separatrix_clip import _traced_clip
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.transport.current_diffusion import (
    _BICUBIC_ARC_WEIGHT,
    _bicubic_arc_moment_correction,
    _bicubic_derivatives,
    _bicubic_stationary_point,
    _integrate_bilinear,
    _integrate_diamagnetic_drive,
    _traced_profile_shapes,
    traced_assemble_flux_surface_geometry,
)
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp


_TWO_PI = 2.0 * np.pi
_SUPPORT_CAPACITY = 8


def _axis_connected_core(psi_n, inside_limiter):
    """Qualify every nested surface with one axis-connected flood fill."""
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
    cell_participation = jnp.any((core > 0.0).reshape(-1)[cell_nodes], axis=1)

    def signed_flux(level):
        return jnp.where(flat_eligible, level - flat_flux, -1.0)

    def required_vertex_count(level):
        inside = signed_flux(level)[cell_nodes] > 0.0
        crossing = inside != jnp.roll(inside, -1, axis=1)
        return jnp.max(jnp.sum(inside, axis=1) + jnp.sum(crossing, axis=1))

    def clip(level, *, separatrix):
        supports = _traced_clip(
            coordinates,
            cell_nodes,
            vertex_count,
            centroids,
            _SUPPORT_CAPACITY,
            signed_flux(level),
        )
        return supports.qualify(cell_participation) if separatrix else supports

    def corrected_clip(level, *, separatrix):
        supports = clip(level, separatrix=separatrix)
        correction, crossing_points, crossing, arc_r, arc_z, arc_valid = (
            _bicubic_arc_moment_correction(
                level, normalised_coefficient, psi_n_cells, dr, dz
            )
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
            arc_valid,
            boundary_valid,
            invalid_boundary,
        )

    def cumulative(level, *, separatrix):
        supports, correction, *_, boundary_valid, invalid_boundary = corrected_clip(
            level, separatrix=separatrix
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
            boundary_valid,
            jnp.sum(invalid_boundary),
            jnp.argmax(invalid_boundary),
        )

    def arc_surface_average(level, *, separatrix):
        (
            supports,
            _,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            arc_valid,
            boundary_valid,
            invalid_boundary,
        ) = corrected_clip(level, separatrix=separatrix)
        inside = psi_n_cells < level
        following_inside = jnp.roll(inside, -1, axis=1)
        leaving = crossing & inside & ~following_inside
        entering = crossing & ~inside & following_inside
        leaving_index = jnp.argmax(leaving, axis=1)
        entering_index = jnp.argmax(entering, axis=1)
        start = jnp.take_along_axis(
            crossing_points, leaving_index[:, None, None], axis=1
        )[:, 0]
        end = jnp.take_along_axis(
            crossing_points, entering_index[:, None, None], axis=1
        )[:, 0]
        radial_span = end[:, 0] - start[:, 0]
        vertical_span = end[:, 1] - start[:, 1]
        use_radial = jnp.abs(radial_span) >= jnp.abs(vertical_span)

        _, normalised_r, normalised_z, *_ = _bicubic_derivatives(
            normalised_coefficient[:, None], arc_r, arc_z
        )
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

        ordinate_derivative = jnp.where(use_radial[:, None], normalised_z, normalised_r)
        parameter_span = jnp.where(
            use_radial, jnp.abs(radial_span), jnp.abs(vertical_span)
        )
        derivative_floor = jnp.asarray(1e-14, dtype=dtype)
        coarea_weight = (
            dr
            * dz
            * parameter_span[:, None]
            * jnp.asarray(_BICUBIC_ARC_WEIGHT, dtype=dtype)[None]
            / jnp.maximum(jnp.abs(ordinate_derivative), derivative_floor)
        )
        sample_valid = supports.boundary & arc_valid
        weighted_volume = jnp.where(
            sample_valid[:, None], radius_at_arc * coarea_weight, 0.0
        )
        denominator = jnp.sum(weighted_volume)
        safe_denominator = jnp.maximum(denominator, jnp.asarray(1e-30, dtype=dtype))
        integrands = jnp.stack(
            (
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
        minimum_ordinate_derivative = jnp.min(
            jnp.where(sample_valid[:, None], jnp.abs(ordinate_derivative), jnp.inf)
        )
        maximum_coarea_weight = jnp.max(
            jnp.where(sample_valid[:, None], coarea_weight, 0.0)
        )
        return (
            averages,
            boundary_valid & (denominator > 0.0),
            jnp.sum(invalid_boundary),
            jnp.argmax(invalid_boundary),
            minimum_ordinate_derivative,
            maximum_coarea_weight,
        )

    levels = jnp.linspace(psi_n_min, psi_n_max, n_surface_bins + 1, dtype=dtype)
    surface_level = 0.5 * (levels[:-1] + levels[1:])
    interior_cumulative = jax.lax.map(
        lambda level: cumulative(level, separatrix=False), levels[:-1]
    )
    edge_cumulative = cumulative(levels[-1], separatrix=True)
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
    shell_values = jnp.diff(cumulative_values, axis=0)
    shell_volume = jnp.maximum(shell_values[:, 0], 1e-30)
    surface_values = shell_values[:, 1:] / shell_volume[:, None]
    cell_origin = coordinates[cell_nodes[:, 0]]
    interior_arc_average = jax.lax.map(
        lambda level: arc_surface_average(level, separatrix=False), surface_level
    )
    edge_arc_average = arc_surface_average(levels[-1], separatrix=True)

    def extrema(level, *, separatrix):
        (
            supports,
            _,
            crossing_points,
            crossing,
            arc_r,
            arc_z,
            arc_valid,
            boundary_valid,
            invalid_boundary,
        ) = corrected_clip(level, separatrix=separatrix)
        samples = jnp.concatenate(
            (crossing_points, jnp.stack((arc_r, arc_z), axis=-1)), axis=1
        )
        sample_valid = jnp.concatenate(
            (crossing, jnp.broadcast_to(arc_valid[:, None], arc_r.shape)), axis=1
        )
        sample_valid &= cell_participation[:, None]

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
            stationary_valid.append(point_valid & arc_valid & cell_participation)
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
                required_vertex_count(level),
                boundary_valid,
                jnp.sum(invalid_boundary),
                jnp.argmax(invalid_boundary),
            )
        )

    surface_extrema = jax.lax.map(
        lambda level: extrema(level, separatrix=False), surface_level
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
    edge_extrema = extrema(edge_level, separatrix=True)
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
        edge_level, separatrix=True
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
    dlevel = (psi_n_max - psi_n_min) / n_surface_bins
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
    return (
        {
            "pn_s": surface_level,
            "dv_dpn": shell_values[:, 0] / dlevel,
            "inv_r2": surface_values[:, 0],
            "inv_r": surface_values[:, 1],
            "grad2_r2": arc_average[:, 0],
            "v_cum": 0.5 * (cumulative_values[:-1, 0] + cumulative_values[1:, 0]),
            "v_total": total_values[0],
        },
        {
            "grad_psi_surface": arc_average[:, 1],
            "grad_psi2_surface": arc_average[:, 2],
            "grad_psi2_over_r2_surface": arc_average[:, 3],
            "b2_surface": arc_average[:, 4],
            "inv_b2_surface": arc_average[:, 5],
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


__all__ = ["extract_flux_surface_geometry"]
