"""Contract tests for traced structured-grid flux-surface extraction."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.flux_surface_extraction import (
    _bicubic_arc_moment_correction,
    _pack_boundary_band,
    _select_shape_representation,
    extract_flux_surface_geometry,
)
from nova.equilibrium.flux_surface_geometry import (
    BISECTION_STEPS,
    LADDER_PER_CELL,
    _ray_reach,
    _refine_axis,
)
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.transport import torax_geometry_from_fsa
from nova.transport.current_diffusion import (
    _bicubic_derivatives,
    _bicubic_edge_coordinates,
    _bicubic_edge_crossings,
    _bicubic_stationary_point,
    _solve_bicubic_ordinate,
)


def _analytic_input(
    points=33, *, elongation=1.55, triangularity=0.0, radial_quartic=0.0
):
    configure_dtypes()
    radius = jnp.linspace(2.3, 3.7, points)
    height = jnp.linspace(-1.05, 1.05, points)
    mesh_radius, mesh_height = jnp.meshgrid(radius, height)
    major_radius = 3.0
    minor_radius = 0.55
    normalized_height = mesh_height / (elongation * minor_radius)
    shifted_radius = (
        mesh_radius - major_radius - triangularity * minor_radius * normalized_height**2
    ) / minor_radius
    psi_n = (
        shifted_radius**2 + radial_quartic * shifted_radius**4 + normalized_height**2
    )
    return {
        "psi2d": -0.24 * psi_n,
        "radius": radius,
        "height": height,
        "inside_limiter": psi_n <= 1.08**2,
        "axis_psi": jnp.asarray(0.0),
        "boundary_psi": jnp.asarray(-0.24),
        "profile_coefficients": jnp.asarray([1.0, 0.0]),
        "coefficient_scale": jnp.asarray([8.0e5, 8.0e5]),
        "ip_amperes": jnp.asarray(5.0e5),
        "major_radius": jnp.asarray(major_radius),
        "boundary_toroidal_field": jnp.asarray(2.0),
        "n_pressure": 1,
        "n_diamagnetic": 1,
        "n_radial_cells": 12,
        "n_surface_bins": 14,
    }


def _extract(inputs):
    return extract_flux_surface_geometry(
        inputs["psi2d"],
        inputs["radius"],
        inputs["height"],
        inputs["inside_limiter"],
        axis_psi=inputs["axis_psi"],
        boundary_psi=inputs["boundary_psi"],
        profile_coefficients=inputs["profile_coefficients"],
        coefficient_scale=inputs["coefficient_scale"],
        ip_amperes=inputs["ip_amperes"],
        major_radius=inputs["major_radius"],
        boundary_toroidal_field=inputs["boundary_toroidal_field"],
        n_pressure=inputs["n_pressure"],
        n_diamagnetic=inputs["n_diamagnetic"],
        n_radial_cells=inputs["n_radial_cells"],
        n_surface_bins=inputs["n_surface_bins"],
    )


def _reference_module():
    path = Path(__file__).with_name("test_transport_geometry_reference.py")
    specification = spec_from_file_location("transport_geometry_reference", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _real_record(filename, cocos):
    reference = _reference_module()
    data = reference._nova_input(filename, cocos)
    major_radius = 0.5 * (float(np.max(data["xbdry"])) + float(np.min(data["xbdry"])))
    boundary_field = float(data["bcentr"]) * float(data["xcentr"]) / major_radius
    record = extract_flux_surface_geometry(
        jnp.asarray(np.asarray(data["psi"]).T),
        jnp.asarray(data["x"]),
        jnp.asarray(data["z"]),
        jnp.ones((int(data["nz"]), int(data["nx"])), dtype=bool),
        axis_psi=jnp.asarray(data["simagx"]),
        boundary_psi=jnp.asarray(data["sibdry"]),
        profile_coefficients=jnp.zeros(2),
        coefficient_scale=jnp.ones(2),
        ip_amperes=jnp.asarray(data["Ip"]),
        major_radius=jnp.asarray(major_radius),
        boundary_toroidal_field=jnp.asarray(boundary_field),
        field_function_psi_n=jnp.asarray(data["pnorm"]),
        field_function=jnp.asarray(data["fpol"]),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=24,
        n_surface_bins=96,
        psi_n_min=jnp.asarray(0.01),
        psi_n_max=jnp.asarray(0.99),
    )
    return data, record


def _host_contour_shape(data, levels):
    """Trace shape extrema with the host equilibrium spline and ray brackets."""
    radius = np.asarray(data["x"])
    height = np.asarray(data["z"])
    state = np.asarray(data["psi"]).ravel().reshape(radius.size, height.size)
    interpolant = RectBivariateSpline(radius, height, state, kx=3, ky=3, s=0)
    centre = _refine_axis(
        interpolant,
        (float(data["xmagx"]), float(data["zmagx"])),
        radius,
        height,
    )
    axis_flux = float(interpolant.ev(*centre))
    span = float(data["sibdry"]) - axis_flux
    levels = np.asarray(levels, dtype=np.float64)
    positive = levels > 0.0
    traced_levels = levels[positive]
    angle = 2.0 * np.pi * np.arange(512) / 512
    cosine = np.cos(angle)
    sine = np.sin(angle)
    reach = _ray_reach(centre, cosine, sine, radius, height)
    cosine = cosine[:, None]
    sine = sine[:, None]

    def normalized(distance):
        return (
            interpolant.ev(
                centre[0] + distance * cosine,
                centre[1] + distance * sine,
            )
            - axis_flux
        ) / span

    samples = LADDER_PER_CELL * max(radius.size, height.size)
    ladder = reach[:, None] * np.linspace(0.0, 1.0, samples + 1)[None]
    crossed = normalized(ladder)[:, None, :] >= traced_levels[None, :, None]
    index = np.argmax(crossed, axis=-1)
    lower = np.take_along_axis(ladder, np.maximum(index - 1, 0), axis=1)
    upper = np.take_along_axis(ladder, index, axis=1)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        below = normalized(middle) < traced_levels[None]
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    distance = 0.5 * (lower + upper)
    surface_radius = centre[0] + distance * cosine
    surface_height = centre[1] + distance * sine
    r_in = surface_radius.min(axis=0)
    r_out = surface_radius.max(axis=0)
    local_major = 0.5 * (r_in + r_out)
    local_minor = 0.5 * (r_out - r_in)
    columns = np.arange(traced_levels.size)
    upper_slot = np.argmax(surface_height, axis=0)
    lower_slot = np.argmin(surface_height, axis=0)
    values = {
        "R_in": r_in,
        "R_out": r_out,
        "elongation": (surface_height.max(axis=0) - surface_height.min(axis=0))
        / (2.0 * local_minor),
        "delta_upper": (local_major - surface_radius[upper_slot, columns])
        / local_minor,
        "delta_lower": (local_major - surface_radius[lower_slot, columns])
        / local_minor,
    }
    return {
        name: np.concatenate(([value[0]], value)) if not positive[0] else value
        for name, value in values.items()
    }


def test_bicubic_iterations_preserve_carry_dtype_and_float64_values():
    """Every fixed iteration remains generic without perturbing fp64 results."""
    configure_dtypes()
    results = {}
    for dtype in (jnp.float32, jnp.float64):
        crossing_coefficient = jnp.asarray(
            [[(row + column) / 3.0 for column in range(4)] for row in range(4)],
            dtype=dtype,
        )[None]
        corner_flux = jnp.asarray([[0.0, 1.0, 2.0, 1.0]], dtype=dtype)
        level = jnp.asarray(0.75, dtype=dtype)
        crossing, mask, _ = _bicubic_edge_crossings(
            level, crossing_coefficient, corner_flux
        )
        edge_r, edge_z = _bicubic_edge_coordinates(
            jnp.arange(4)[None], jnp.full((1, 4), 0.25, dtype=dtype)
        )
        ordinate = _solve_bicubic_ordinate(
            crossing_coefficient,
            level,
            jnp.asarray([[0.25]], dtype=dtype),
            jnp.asarray([[0.5]], dtype=dtype),
            solve_vertical=True,
        )
        squared_control = jnp.asarray(
            [0.25, -1.0 / 12.0, -1.0 / 12.0, 0.25], dtype=dtype
        )
        curved_coefficient = (squared_control[:, None] + squared_control[None, :])[None]
        curved_level = jnp.asarray(0.12, dtype=dtype)
        stationary = _bicubic_stationary_point(
            curved_coefficient,
            curved_level,
            jnp.asarray([0.8], dtype=dtype),
            jnp.asarray([0.5], dtype=dtype),
            radial_extremum=True,
        )
        assert bool(stationary[2][0])
        arrays = (crossing, edge_r, edge_z, ordinate, stationary[0], stationary[1])
        assert all(value.dtype == dtype for value in arrays)
        assert mask.dtype == jnp.bool_
        results[dtype] = tuple(np.asarray(value, dtype=np.float64) for value in arrays)

    for float32_value, float64_value in zip(
        results[jnp.float32], results[jnp.float64], strict=True
    ):
        np.testing.assert_allclose(float32_value, float64_value, rtol=2e-6, atol=2e-6)


def test_boundary_band_is_fixed_shape_and_overflow_fails_closed():
    """Boundary cells pack deterministically and report lost capacity."""
    corner_flux = jnp.asarray(
        (
            (0.0, 0.4, 0.6, 0.2),
            (0.6, 0.8, 0.9, 0.7),
            (0.1, 0.9, 0.8, 0.2),
            (0.3, 0.7, 0.6, 0.4),
        )
    )
    participation = jnp.asarray((True, True, True, True))
    indices, valid, count, overflow = _pack_boundary_band(
        jnp.asarray(0.5), corner_flux, participation, 2
    )
    np.testing.assert_array_equal(indices, (0, 2))
    np.testing.assert_array_equal(valid, (True, True))
    assert int(count) == 3
    assert bool(overflow)

    indices, valid, count, overflow = _pack_boundary_band(
        jnp.asarray(0.5), corner_flux, participation, 4
    )
    np.testing.assert_array_equal(indices[:3], (0, 2, 3))
    np.testing.assert_array_equal(valid, (True, True, True, False))
    assert int(count) == 3
    assert not bool(overflow)


def test_even_edge_root_pair_is_resolved_on_iter_inner_surface():
    """A cubic edge is bracketed on both sides of its interior minimum."""
    configure_dtypes()
    reference = _reference_module()
    data = reference._nova_input("iterhybrid_cocos17.eqdsk", 17)
    psi = jnp.asarray(np.asarray(data["psi"]).T)
    psi_n = (psi - data["simagx"]) / (data["sibdry"] - data["simagx"])
    coefficient = fit_tensor_spline(
        jnp.asarray(data["x"]), jnp.asarray(data["z"]), psi_n
    ).cell_coefficients.reshape(-1, 4, 4)
    cell = 9285
    row, column = divmod(cell, psi_n.shape[1] - 1)
    corners = jnp.asarray(
        (
            psi_n[row, column],
            psi_n[row, column + 1],
            psi_n[row + 1, column + 1],
            psi_n[row + 1, column],
        )
    )[None]
    level = jnp.asarray(0.040625, dtype=psi.dtype)
    points, crossing, _ = _bicubic_edge_crossings(
        level, coefficient[cell : cell + 1], corners
    )
    points = points.reshape(1, 4, 3, 2)
    crossing = crossing.reshape(1, 4, 3)
    top_radius = np.sort(np.asarray(points[0, 2, crossing[0, 2], 0]))
    np.testing.assert_allclose(
        top_radius, np.asarray((0.03308514, 0.42126222)), rtol=0.0, atol=2e-8
    )
    values = _bicubic_derivatives(
        coefficient[cell],
        jnp.asarray(top_radius),
        jnp.ones(2, dtype=psi.dtype),
    )[0]
    assert float(jnp.max(jnp.abs(values - level))) < 2e-8


def test_short_corner_arc_rejects_a_disconnected_stationary_split():
    """Reflected corner crossings stay separate from a remote spline branch."""
    configure_dtypes()
    coefficient = jnp.asarray(
        [
            [
                [
                    0.8925411017500524,
                    0.9161396415231808,
                    0.9404617198937791,
                    0.9657067643658207,
                ],
                [
                    0.8671276615558743,
                    0.8914403240931793,
                    0.9164302365820515,
                    0.9424437222102411,
                ],
                [
                    0.8413108568604310,
                    0.8664035350228159,
                    0.8919292635507745,
                    0.9187155804523564,
                ],
                [
                    0.8162481223885881,
                    0.8420013999453246,
                    0.8681904173648530,
                    0.8956971641040243,
                ],
            ],
            [
                [
                    0.8162481223885526,
                    0.8420013999452904,
                    0.8681904173648256,
                    0.8956971641040031,
                ],
                [
                    0.8413108568603900,
                    0.8664035350227755,
                    0.8919292635507445,
                    0.9187155804523355,
                ],
                [
                    0.8671276615558363,
                    0.8914403240931420,
                    0.9164302365820267,
                    0.9424437222102267,
                ],
                [
                    0.8925411017500188,
                    0.9161396415231480,
                    0.9404617198937589,
                    0.9657067643658089,
                ],
            ],
        ]
    )
    corners = jnp.asarray(
        [
            [
                0.8925411017500524,
                0.9657067643658207,
                0.8956971641040243,
                0.8162481223885881,
            ],
            [
                0.8162481223885526,
                0.8956971641040031,
                0.9657067643658089,
                0.8925411017500188,
            ],
        ]
    )
    level = jnp.asarray(0.81625)
    result = _bicubic_arc_moment_correction(
        level,
        coefficient,
        corners,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.zeros((2, 4)),
    )
    crossing_points = np.asarray(result[1]).reshape(2, 4, 3, 2)
    crossing = np.asarray(result[2]).reshape(2, 4, 3)
    assert np.array_equal(crossing.sum(axis=(1, 2)), np.asarray((2, 2)))
    radial_offset = 2.4301931262016296e-5
    vertical_offset = 2.497248351570392e-5
    expected = (
        (
            (2, np.asarray((radial_offset, 1.0))),
            (3, np.asarray((0.0, 1.0 - vertical_offset))),
        ),
        (
            (0, np.asarray((radial_offset, 0.0))),
            (3, np.asarray((0.0, vertical_offset))),
        ),
    )
    for cell, edge_roots in enumerate(expected):
        for edge, point in edge_roots:
            np.testing.assert_allclose(
                crossing_points[cell, edge, 0], point, rtol=0.0, atol=2e-8
            )
    sample_valid = np.asarray(result[6])
    assert np.array_equal(sample_valid.sum(axis=1), np.asarray((12, 12)))
    assert bool(jnp.all(result[-1]))
    for cell in range(2):
        values = _bicubic_derivatives(
            coefficient[cell],
            result[3][cell, sample_valid[cell]],
            result[4][cell, sample_valid[cell]],
        )[0]
        assert float(jnp.max(jnp.abs(values - level))) < 2e-8


def test_analytic_shape_gradient_jit_and_batch_contract():
    """Analytic shape, autodiff, compilation, and batching share one record."""
    inputs = _analytic_input(points=25, elongation=1.55, triangularity=0.12)
    record = _extract(inputs)
    assert bool(record["valid"])
    assert bool(record["surface_arc_valid"])
    assert record["rho_face"].shape == (13,)
    assert record["shape_axis_expansion_face"].shape == record["rho_face"].shape
    assert record["shape_boundary_cell_count_face"].shape == record["rho_face"].shape
    assert not bool(record["surface_cell_band_overflow"])
    assert int(record["surface_cell_band_max_count"]) <= int(
        record["surface_cell_band_capacity"]
    )
    full_cell_count = (inputs["psi2d"].shape[0] - 1) * (inputs["psi2d"].shape[1] - 1)
    assert int(record["surface_cell_band_capacity"]) < full_cell_count
    np.testing.assert_allclose(record["elongation_face"][-1], 1.55, rtol=4e-3)
    np.testing.assert_allclose(record["delta_upper_face"][-1], -0.12, atol=7e-3)
    np.testing.assert_allclose(record["delta_lower_face"][-1], -0.12, atol=7e-3)

    def shape_from_grid(grid):
        changed = dict(inputs)
        changed["psi2d"] = grid
        return _extract(changed)["elongation_face"][-1]

    shape_gradient = jax.grad(shape_from_grid)(inputs["psi2d"])
    slot = np.unravel_index(
        int(np.argmax(np.abs(np.asarray(shape_gradient)))), shape_gradient.shape
    )
    perturbation = jnp.zeros_like(inputs["psi2d"]).at[slot].set(1.0)
    epsilon = jnp.asarray(2.0e-6, dtype=inputs["psi2d"].dtype)
    central = (
        shape_from_grid(inputs["psi2d"] + epsilon * perturbation)
        - shape_from_grid(inputs["psi2d"] - epsilon * perturbation)
    ) / (2.0 * epsilon)
    relative_error = float(
        jnp.abs(central - shape_gradient[slot])
        / jnp.maximum(jnp.abs(central), jnp.asarray(1e-12))
    )
    print(
        f"shape_gradient slot={slot} autodiff={float(shape_gradient[slot]):.9e} "
        f"central={float(central):.9e} relative_error={relative_error:.9e}"
    )
    assert relative_error < 3e-3

    compiled = jax.jit(shape_from_grid)(inputs["psi2d"])
    np.testing.assert_allclose(compiled, record["elongation_face"][-1], rtol=1e-12)
    batched = jax.vmap(shape_from_grid)(
        jnp.stack((inputs["psi2d"], inputs["psi2d"] * 1.01))
    )
    assert batched.shape == (2,)
    np.testing.assert_allclose(batched, jnp.asarray([1.55, 1.55]), rtol=4e-3)


def test_shape_representation_gradient_routes_only_selected_faces():
    """The discrete shape choice sends gradients only to its selected values."""
    expansion = jnp.asarray((1.0, 2.0, 3.0))
    clipped = jnp.asarray((4.0, 5.0, 6.0))
    use_expansion = jnp.asarray((True, False, True))

    def selected_sum(expansion_values, clipped_values):
        return jnp.sum(
            _select_shape_representation(
                expansion_values,
                clipped_values,
                use_expansion,
            )
        )

    expansion_gradient, clipped_gradient = jax.grad(selected_sum, argnums=(0, 1))(
        expansion, clipped
    )
    np.testing.assert_array_equal(expansion_gradient, (1.0, 0.0, 1.0))
    np.testing.assert_array_equal(clipped_gradient, (0.0, 1.0, 0.0))


def test_analytic_surface_line_averages():
    """Arc coarea integrals recover ellipse volume and radius moments."""
    inputs = _analytic_input(points=33, elongation=1.55)
    record = _extract(inputs)
    psi_n_face = np.asarray(record["psi_n_face"])
    major_radius = float(inputs["major_radius"])
    minor_radius = 0.55
    expected_volume_derivative = (
        2.0
        * np.pi**2
        * major_radius
        * 1.55
        * minor_radius**2
        / abs(float(inputs["boundary_psi"] - inputs["axis_psi"]))
    )
    expected_inverse_radius_squared = 1.0 / (
        major_radius * np.sqrt(major_radius**2 - minor_radius**2 * psi_n_face)
    )
    np.testing.assert_allclose(
        record["int_dl_over_bp_face"][2:],
        expected_volume_derivative,
        rtol=1e-5,
    )
    np.testing.assert_allclose(record["inv_r_face"][2:], 1.0 / major_radius, rtol=1e-12)
    np.testing.assert_allclose(
        record["g3_face"][2:], expected_inverse_radius_squared[2:], rtol=1e-3
    )
    expected_volume_derivative_rho = (
        record["int_dl_over_bp_face"]
        * 2.0
        * record["rho_face"]
        * record["phi_b"]
        / record["q_face"]
    )
    # The innermost faces use the explicit linear axis limit of the surface
    # interpolator; outside that limit the direct coarea identity is exact.
    np.testing.assert_allclose(
        record["vpr_face"][4:], expected_volume_derivative_rho[4:], rtol=1e-12
    )


def test_extremum_radial_position_converges_faster_than_first_order():
    """Global spline extrema leave the first-order lattice regime."""
    resolutions = np.asarray([17, 25, 33])
    errors = []
    radial_quartic = 0.12
    edge_level = 0.985
    root_squared = (np.sqrt(1.0 + 4.0 * radial_quartic * edge_level) - 1.0) / (
        2.0 * radial_quartic
    )
    expected = 3.0 + 0.55 * np.sqrt(root_squared)
    for points in resolutions:
        record = _extract(
            _analytic_input(points=int(points), radial_quartic=radial_quartic)
        )
        errors.append(abs(float(record["r_out_face"][-1]) - expected))
    spacing = 1.0 / (resolutions - 1)
    order = float(np.polyfit(np.log(spacing), np.log(errors), 1)[0])
    print(f"extremum_errors={errors} convergence_order={order:.9f}")
    assert order > 1.0


def test_iter_shape_referees_localize_inner_surface_convention():
    """Two spline routes agree while TORAX's inner contour extension differs."""
    configure_dtypes()
    reference = _reference_module()
    data, record = _real_record("iterhybrid_cocos17.eqdsk", 17)
    service = torax_geometry_from_fsa(record)
    torax = reference._torax_geometry("iterhybrid_cocos17.eqdsk", 17)
    contour = reference._contour_geometry(
        "iterhybrid_cocos17.eqdsk", 17, record["rho_face"]
    )
    host = _host_contour_shape(data, contour.psi_norm)
    host_at_service_label = _host_contour_shape(data, record["psi_n_face"])

    def relative_error(actual, expected, mask=slice(2, None)):
        actual = np.asarray(actual)[mask]
        expected = np.asarray(expected)[mask]
        return float(
            np.max(np.abs(actual - expected))
            / max(float(np.max(np.abs(expected))), 1e-12)
        )

    service_upper = np.asarray(service.delta_upper_face)
    torax_upper = np.asarray(torax.delta_upper_face)
    service_host = relative_error(service_upper, host["delta_upper"])
    service_host_same_label = relative_error(
        service_upper, host_at_service_label["delta_upper"]
    )
    torax_host = relative_error(torax_upper, host["delta_upper"])
    outer = np.asarray(record["rho_face"]) >= 0.25
    outer_service_torax = relative_error(service_upper, torax_upper, outer)
    print(
        "ITER delta_upper service_host=%.9e service_host_same_psi=%.9e "
        "torax_host=%.9e outer_service_torax=%.9e"
        % (service_host, service_host_same_label, torax_host, outer_service_torax)
    )
    assert service_host < 0.05
    assert service_host_same_label < 0.03
    assert torax_host > 0.10
    assert outer_service_torax < 0.05

    for field in ("R_in", "R_out", "elongation"):
        service_values = getattr(service, f"{field}_face")
        torax_values = getattr(torax, f"{field}_face")
        assert relative_error(service_values, host[field]) < 0.01
        assert relative_error(torax_values, host[field]) < 0.01


@pytest.mark.parametrize(
    ("filename", "cocos"),
    [
        pytest.param(
            "iterhybrid_cocos17.eqdsk",
            17,
        ),
        pytest.param(
            "STEP_SPP_001_ECHD_ftop.eqdsk",
            1,
        ),
    ],
)
def test_real_equilibrium_reference_gates(filename, cocos):
    """The locked TORAX and contour thresholds remain executable evidence."""
    configure_dtypes()
    reference = _reference_module()
    _data, record = _real_record(filename, cocos)
    geometry = torax_geometry_from_fsa(record)
    torax_geometry = reference._torax_geometry(filename, cocos)
    contour = reference._contour_geometry(filename, cocos, record["rho_face"])
    contour_fields = {
        "Phi_face": np.abs(contour.toroidal_flux),
        "volume_face": contour.volume,
        "area_face": contour.area,
        "vpr_face": contour.volume_derivative * contour.boundary_rho_tor,
        "g0_face": contour.gradient_rho * contour.volume_derivative,
        "g1_face": contour.gradient_rho_squared * contour.volume_derivative**2,
        "g2_face": contour.gradient_rho_squared_over_radius_squared
        * contour.volume_derivative**2,
        "g3_face": contour.inverse_square_radius,
    }
    fields = (
        "Phi_face",
        "volume_face",
        "area_face",
        "vpr_face",
        "g0_face",
        "g1_face",
        "g2_face",
        "g3_face",
        "g2g3_over_rhon_face",
        "elongation_face",
        "delta_upper_face",
        "delta_lower_face",
    )
    torax_error = {}
    contour_error = {}
    for field in fields:
        torax_error[field] = reference._relative_error(
            getattr(geometry, field), getattr(torax_geometry, field)
        )
        if field in contour_fields:
            contour_error[field] = reference._relative_error(
                getattr(geometry, field), contour_fields[field]
            )
        print(
            f"{filename} {field} torax_relative_error={torax_error[field]:.9e} "
            f"contour_relative_error={contour_error.get(field, np.nan):.9e}"
        )

    failures = {}
    measured_baseline = {
        "iterhybrid_cocos17.eqdsk": {
            "Phi_face": 0.02208927376,
            "volume_face": 0.01752871900,
            "area_face": 0.01883412805,
            "vpr_face": 0.02305704449,
            "g0_face": 0.00891519730,
            "g1_face": 0.02024573833,
            "g2_face": 0.01886634587,
            "g2g3_over_rhon_face": 0.02625216689,
        },
        "STEP_SPP_001_ECHD_ftop.eqdsk": {
            "Phi_face": 0.02560021182,
            "volume_face": 0.01837201787,
            "area_face": 0.02128042988,
            "vpr_face": 0.01841215367,
            "g0_face": 0.03647981955,
            "g1_face": 0.08096168206,
            "g2_face": 0.03229869161,
            "g2g3_over_rhon_face": 0.05041074213,
        },
    }
    for field, expected in measured_baseline[filename].items():
        np.testing.assert_allclose(torax_error[field], expected, rtol=0.02, atol=2e-5)
    measured_contour_baseline = {
        "iterhybrid_cocos17.eqdsk": {
            "vpr_face": 0.02262632517,
            "g1_face": 0.02328956320,
        },
        "STEP_SPP_001_ECHD_ftop.eqdsk": {
            "vpr_face": 0.02349369153,
            "g1_face": 0.03884814627,
        },
    }
    for field, expected in measured_contour_baseline[filename].items():
        np.testing.assert_allclose(contour_error[field], expected, rtol=0.02, atol=2e-5)
    if filename.startswith("iterhybrid"):
        thresholds = {
            "g2g3_over_rhon_face": 0.0263,
            "g0_face": 0.0090,
            "g1_face": 0.0208,
            "g2_face": 0.0198,
            "elongation_face": 0.02,
        }
        failures.update(
            {
                field: error
                for field, error in torax_error.items()
                if field in thresholds and error > thresholds[field]
            }
        )
    else:
        # The cumulative clipped-cell fields retain their exact integrals but
        # are sampled on the rho labels induced by the direct-q averages.  The
        # two-sided baselines above pin that label effect independently.
        contour_thresholds = {"vpr_face": 0.0392, "g1_face": 0.0620}
        failures.update(
            {
                f"contour:{field}": error
                for field, error in contour_error.items()
                if field in contour_thresholds and error > contour_thresholds[field]
            }
        )
    diagnostic = {
        "invalid_count": int(record["surface_arc_invalid_count"]),
        "first_invalid_cell": int(record["surface_arc_first_invalid_cell"]),
        "first_invalid_level": float(record["surface_arc_first_invalid_level"]),
        "minimum_ordinate_derivative": float(
            record["surface_arc_min_ordinate_derivative"]
        ),
        "maximum_coarea_weight": float(record["surface_arc_max_coarea_weight"]),
    }
    print(f"{filename} arc_diagnostic={diagnostic}")
    if filename.startswith("iterhybrid"):
        assert bool(record["valid"]), diagnostic
        assert bool(record["surface_arc_valid"]), diagnostic
        assert diagnostic["invalid_count"] == 0
    else:
        assert bool(record["valid"]), diagnostic
    assert not failures, failures
