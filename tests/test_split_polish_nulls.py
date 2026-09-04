"""Checks for census-owned null selection with split-spline polishing."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.hex_cell_field_feasibility import (
    AXIS,
    LOBE_OFFSET,
    SADDLE,
    _base_flux,
    hex_lattice,
    solovev_flux,
)
from nova.equilibrium.flux_surface_connectivity import (
    polish_census_stationary_points,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from nova.linalg.split_spline import fit_split_spline
from tests.test_hex_flood_sn_secondary import (
    GEOMETRIES,
    GRID_SIZES,
    _topology_and_flux,
)


configure_dtypes()


def _carrier(shape: tuple[int, int]):
    centres, _, _ = hex_lattice(shape)
    radial = jnp.asarray(centres[..., 0])
    vertical = jnp.asarray(centres[..., 1])
    points = jnp.asarray(centres.reshape(-1, 2))
    values = solovev_flux(points).reshape(shape)
    level_set = (_base_flux(points) - LOBE_OFFSET**4).reshape(shape)
    return radial, vertical, values, level_set


def _selected_row(position: np.ndarray) -> jnp.ndarray:
    point = jnp.asarray(position)
    return jnp.r_[point, solovev_flux(point), 0.0]


def test_one_step_refines_only_the_two_census_selected_nulls():
    radial, vertical, values, _level_set = _carrier((41, 49))
    extremum_seed = _selected_row(AXIS + (0.005, 0.0))
    saddle_seed = _selected_row(SADDLE)
    boundary_value = solovev_flux(jnp.asarray(SADDLE))

    extremum, saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        boundary_value,
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
    )
    error = jnp.linalg.norm(
        jnp.stack((extremum[:2], saddle[:2])) - jnp.asarray((AXIS, SADDLE)),
        axis=-1,
    )
    active_basis_count = np.asarray(receipt["active_derivative_basis_count"]).tolist()

    print(
        f"solovev axis_error_m={float(error[0]):.12e} "
        f"saddle_error_m={float(error[1]):.12e} "
        f"iterations={np.asarray(receipt['iteration_count']).tolist()} "
        f"accepted={np.asarray(receipt['converged']).tolist()} "
        f"normalized_gradient={np.asarray(receipt['normalized_gradient']).tolist()} "
        f"roundoff_floor={np.asarray(receipt['roundoff_floor']).tolist()} "
        f"representation_floor={np.asarray(receipt['representation_floor']).tolist()} "
        f"active_basis_count={active_basis_count} "
        f"tolerance={np.asarray(receipt['stationarity_tolerance']).tolist()}"
    )
    # Half-offset support keeps both slots without claiming tensor authority.
    assert error[0] == pytest.approx(5.0e-3, abs=1.0e-12)
    assert error[1] == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_array_equal(np.asarray(receipt["complete_map"]), False)
    np.testing.assert_array_equal(np.asarray(receipt["spline_authored"]), False)
    assert receipt["fit_attempted"].tolist() == [True, True]
    assert receipt["fit_iterations"].tolist() == [1, 1]
    assert np.all(np.isfinite(np.asarray(receipt["fit_residual"])))
    assert not bool(receipt["converged"][0])
    assert receipt["iteration_count"][0] == 0
    assert not bool(receipt["seed_stationary"][0])
    assert bool(receipt["seed_stationary"][1]) or not bool(receipt["converged"][1])
    assert receipt["value"].shape == (2,)
    assert receipt["gradient"].shape == (2, 2)
    assert receipt["hessian"].shape == (2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(receipt["census_position_rz"]),
        np.asarray(jnp.stack((extremum_seed[:2], saddle_seed[:2]))),
    )


def _rescaled_solovev_polish(flux_scale: float, coordinate_scale: float):
    radial, vertical, values, _level_set = _carrier((41, 49))
    extremum_position = jnp.asarray(AXIS + (0.005, 0.0)) * coordinate_scale
    saddle_position = jnp.asarray(SADDLE) * coordinate_scale
    extremum_seed = jnp.r_[
        extremum_position,
        flux_scale * solovev_flux(jnp.asarray(AXIS + (0.005, 0.0))),
        0.0,
    ]
    saddle_seed = jnp.r_[
        saddle_position,
        flux_scale * solovev_flux(jnp.asarray(SADDLE)),
        0.0,
    ]
    extremum, saddle, receipt = polish_census_stationary_points(
        flux_scale * values,
        coordinate_scale * radial,
        coordinate_scale * vertical,
        flux_scale * solovev_flux(jnp.asarray(SADDLE)),
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
    )
    return extremum, saddle, receipt


def _rescaled_quadratic_polish(flux_scale: float, coordinate_scale: float):
    radial = coordinate_scale * jnp.linspace(-1.0, 1.0, 33)
    vertical = coordinate_scale * jnp.linspace(-1.0, 1.0, 33)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    values = flux_scale * (
        (radial_grid / coordinate_scale) ** 2 - (vertical_grid / coordinate_scale) ** 2
    )
    pitch = radial[1] - radial[0]
    sampled_saddle = jnp.asarray(
        (
            3.0 * pitch,
            0.0,
            flux_scale * 9.0 * (pitch / coordinate_scale) ** 2,
            0.0,
        )
    )
    absent_extremum = jnp.full_like(sampled_saddle, jnp.nan)
    return polish_census_stationary_points(
        values,
        radial,
        vertical,
        jnp.asarray(0.0),
        jnp.asarray(-1.0),
        absent_extremum,
        sampled_saddle,
    )


def _assert_dimensionless_receipts_invariant(unscaled, scaled):
    numerical_receipts = (
        "roundoff_floor",
        "representation_floor",
        "value_basis_norm",
        "fit_residual",
        "seed_normalized_gradient",
        "normalized_gradient",
        "normalized_value_change",
        "stationarity_tolerance",
        "value_consistency_tolerance",
    )
    exact_receipts = (
        "active_derivative_basis_count",
        "iteration_count",
        "fit_attempted",
        "fit_iterations",
        "hessian_type",
        "in_domain",
        "seed_stationary",
        "converged",  # The exposed convergence qualification is acceptance.
        "fit_converged",
        "representation_adequate",
        "value_replaced",
        "local_value_consistent",
        "spline_authored",
        "complete_map",
    )
    dimensional_exclusions = {
        "position_rz": "attempted positions carry coordinate units",
        "value": "the map-owned selected values carry flux units",
        "gradient_norm": "gradient norms carry flux per coordinate units",
        "gradient": "gradient components carry flux per coordinate units",
        "hessian": "Hessian entries carry flux per coordinate squared units",
        "interface_value": "the selected interface carries flux units",
        "derivative_basis_norm": (
            "the derivative operator carries inverse coordinate units"
        ),
        "sample_rms_residual": "the sample residual carries flux units",
        "census_position_rz": "census positions carry coordinate units",
        "selected_position_rz": "selected positions carry coordinate units",
        "selected_value": "selected values carry flux units",
        "fit_value": "the split fit's diagnostic value carries flux units",
        "spline_value": "the common-map value carries flux units",
        "local_value_evidence": "the seven-point evidence carries flux units",
    }
    compared_keys = set(numerical_receipts) | set(exact_receipts)
    assert compared_keys.isdisjoint(dimensional_exclusions)
    assert set(unscaled) == set(scaled)
    assert compared_keys | set(dimensional_exclusions) == set(unscaled)

    for name in numerical_receipts:
        unscaled_value = np.asarray(unscaled[name])
        scaled_value = np.asarray(scaled[name])
        np.testing.assert_array_equal(np.isnan(scaled_value), np.isnan(unscaled_value))
        finite = np.isfinite(unscaled_value) & np.isfinite(scaled_value)
        allowed = (
            1.0e-10 * np.maximum(np.abs(unscaled_value), np.abs(scaled_value)) + 1.0e-13
        )
        assert np.all(
            np.abs(unscaled_value[finite] - scaled_value[finite]) <= allowed[finite]
        ), name
    for name in exact_receipts:
        np.testing.assert_array_equal(
            np.asarray(scaled[name]), np.asarray(unscaled[name])
        )


def test_dimensionless_polish_receipts_are_rescaling_invariant():
    """Flux and coordinate units leave every resolved receipt unchanged.

    The mixed bound is relative at resolved scales.  Its 1e-13 absolute term is
    about 450 float64 machine epsilons, below which rescaling a physical-value
    solve may legitimately reorder cancellation in a roundoff-scale receipt.
    Acceptance decisions remain exact boolean invariants.
    """
    coordinate_scale = 1.0e-2
    unscaled_extremum, unscaled_saddle, unscaled = _rescaled_quadratic_polish(1.0, 1.0)
    scaled_extremum, scaled_saddle, scaled = _rescaled_quadratic_polish(
        1.0e3, coordinate_scale
    )
    # Authority flags and both evidence channels keep their roles under rescaling.
    _assert_dimensionless_receipts_invariant(unscaled, scaled)
    np.testing.assert_allclose(
        np.asarray(scaled_extremum[:2]) / coordinate_scale,
        np.asarray(unscaled_extremum[:2]),
        rtol=1.0e-10,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(scaled_saddle[:2]) / coordinate_scale,
        np.asarray(unscaled_saddle[:2]),
        rtol=1.0e-10,
        atol=1.0e-13,
    )
    _, _, solovev_unscaled = _rescaled_solovev_polish(1.0, 1.0)
    _, _, solovev_scaled = _rescaled_solovev_polish(1.0e3, coordinate_scale)
    _assert_dimensionless_receipts_invariant(solovev_unscaled, solovev_scaled)
    print(
        "solovev_rescaling_receipt "
        f"unscaled_roundoff={np.asarray(solovev_unscaled['roundoff_floor']).tolist()} "
        f"scaled_roundoff={np.asarray(solovev_scaled['roundoff_floor']).tolist()} "
        "unscaled_representation="
        f"{np.asarray(solovev_unscaled['representation_floor']).tolist()} "
        "scaled_representation="
        f"{np.asarray(solovev_scaled['representation_floor']).tolist()}"
    )


def test_sample_coincidence_does_not_imply_stationarity():
    radial = jnp.linspace(-1.0, 1.0, 33)
    vertical = jnp.linspace(-1.0, 1.0, 33)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    values = radial_grid**2 - vertical_grid**2
    pitch = radial[1] - radial[0]
    sampled_saddle = jnp.asarray((3.0 * pitch, 0.0, 9.0 * pitch**2, 0.0))
    absent_extremum = jnp.full_like(sampled_saddle, jnp.nan)

    _selected_extremum, selected_saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        jnp.asarray(0.0),
        jnp.asarray(-1.0),
        absent_extremum,
        sampled_saddle,
    )
    print(
        "sample_coincident_receipt "
        f"seed_gradient={float(receipt['seed_normalized_gradient'][1]):.12e} "
        f"roundoff_floor={float(receipt['roundoff_floor'][1]):.12e} "
        f"representation_floor={float(receipt['representation_floor'][1]):.12e}"
    )

    assert float(sampled_saddle[0]) in np.asarray(radial)
    assert float(sampled_saddle[1]) in np.asarray(vertical)
    assert not bool(receipt["seed_stationary"][1])
    assert int(receipt["iteration_count"][1]) == 1
    assert float(receipt["seed_normalized_gradient"][1]) > float(
        receipt["stationarity_tolerance"][1]
    )
    if bool(receipt["converged"][1]):
        moved = jnp.linalg.norm(selected_saddle[:2] - sampled_saddle[:2])
        coordinate_span = radial[-1] - radial[0]
        assert float(moved) > float(
            receipt["stationarity_tolerance"][1] * coordinate_span
        )
    else:
        np.testing.assert_array_equal(
            np.asarray(selected_saddle), np.asarray(sampled_saddle)
        )


def test_accepted_unmoved_polish_preserves_census_value_bits():
    radial = jnp.linspace(-1.0, 1.0, 33)
    vertical = jnp.linspace(-1.0, 1.0, 33)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    census_value = jnp.asarray(0.37)
    values = census_value + radial_grid**2 + vertical_grid**2
    selected_extremum = jnp.asarray((0.0, 0.0, census_value, -1.0))
    absent_saddle = jnp.full_like(selected_extremum, jnp.nan)

    extremum, _saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        jnp.asarray(1.0),
        jnp.asarray(-1.0),
        selected_extremum,
        absent_saddle,
    )

    assert bool(receipt["converged"][0])
    assert bool(receipt["seed_stationary"][0])
    assert int(receipt["iteration_count"][0]) == 0
    np.testing.assert_array_equal(
        np.asarray(receipt["selected_position_rz"][0]),
        np.asarray(selected_extremum[:2]),
    )
    # An accepted unmoved slot publishes the common-map value bit-for-bit.
    assert (
        np.asarray(extremum[2]).tobytes()
        == np.asarray(receipt["spline_value"][0]).tobytes()
    )
    assert bool(receipt["spline_authored"][0])
    assert bool(receipt["complete_map"][0])
    assert np.isfinite(float(receipt["local_value_evidence"][0]))
    assert np.isfinite(float(receipt["fit_value"][0]))
    assert np.asarray(extremum[2]).tobytes() != np.asarray(census_value).tobytes()
    assert bool(receipt["local_value_consistent"][0])
    assert not bool(receipt["value_replaced"][0])


def test_mast_unmoved_saddle_keeps_census_flux_and_reports_fit_misrepresentation():
    """An inadequate split fit remains evidence beside the published map flux.

    Production evidence comes from
    ``docs/figures/solver-convergence-regression/null-polish-attribution.json``;
    the selected saddle is corroborated by the 22086/43 pure row in
    ``docs/figures/topology-visual-corroboration/mast-topology-operands.npz``.
    """
    root = Path(__file__).parents[1]
    evidence_path = (
        root / "docs/figures/solver-convergence-regression/null-polish-attribution.json"
    )
    operand_path = (
        root / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
    )
    evidence = json.loads(evidence_path.read_text())
    slot = evidence["revisions"]["main_head"]["arms"]["22086/43 pure"]["solve"][
        "topology_qualification_polish_receipt"
    ]["slots"]["x"]
    with np.load(operand_path, allow_pickle=False) as operands:
        bank_saddle = operands["row_10_selected_x"][0]
        operand_coordinate = operands["row_10_cell_rz"]

    seed_position = jnp.asarray(slot["seed_position_rz_m"])
    census_value = jnp.asarray(slot["seed_value_wb"])
    published_fit_value = float(slot["polished_value_wb"])
    assert np.linalg.norm(np.asarray(seed_position) - bank_saddle) < 3.0e-3

    radial = jnp.asarray(np.unique(operand_coordinate[:, 0]))
    vertical = jnp.asarray(np.unique(operand_coordinate[:, 1]))
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    local_r = radial_grid - seed_position[0]
    local_z = vertical_grid - seed_position[1]
    checkerboard = (-1.0) ** jnp.indices(radial_grid.shape).sum(axis=0)
    values = published_fit_value + 0.08 * (local_r**2 - local_z**2)
    values = values + 0.06 * checkerboard
    selected_saddle = jnp.r_[seed_position, census_value, 0.0]
    absent_extremum = jnp.full_like(selected_saddle, jnp.nan)

    _extremum, saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        census_value,
        jnp.asarray(-1.0),
        absent_extremum,
        selected_saddle,
    )
    print(
        "mast_value_authority_receipt "
        f"census_value={float(saddle[2]):.17g} "
        f"fit_value={float(receipt['fit_value'][1]):.17g} "
        "sample_rms_residual="
        f"{float(receipt['sample_rms_residual'][1]):.17g} "
        f"representation_adequate={bool(receipt['representation_adequate'][1])}"
    )

    assert bool(receipt["converged"][1])
    # The common map publishes; split and local values remain evidence.
    assert not bool(receipt["seed_stationary"][1])
    np.testing.assert_array_equal(
        np.asarray(saddle[:2]), np.asarray(receipt["selected_position_rz"][1])
    )
    assert (
        np.asarray(saddle[2]).tobytes()
        == np.asarray(receipt["spline_value"][1]).tobytes()
    )
    assert bool(receipt["spline_authored"][1])
    assert bool(receipt["complete_map"][1])
    np.testing.assert_array_equal(
        np.asarray(receipt["value"]), np.asarray(receipt["selected_value"])
    )
    assert np.isfinite(float(receipt["fit_value"][1]))
    assert np.isfinite(float(receipt["local_value_evidence"][1]))
    assert float(receipt["value"][1]) != float(receipt["fit_value"][1])
    assert float(receipt["value"][1]) != float(receipt["local_value_evidence"][1])
    assert bool(receipt["local_value_consistent"][1])
    assert float(receipt["sample_rms_residual"][1]) > abs(
        float(receipt["fit_value"][1] - census_value)
    )
    assert not bool(receipt["representation_adequate"][1])
    assert not bool(receipt["value_replaced"][1])


def test_failed_fit_retains_the_census_rows_and_reports_failure():
    radial, vertical, values, _level_set = _carrier((13, 15))
    extremum_seed = _selected_row(AXIS + (0.005, 0.0))
    saddle_seed = _selected_row(SADDLE)

    extremum, saddle, receipt = polish_census_stationary_points(
        values.at[0, 0].set(jnp.nan),
        radial,
        vertical,
        solovev_flux(jnp.asarray(SADDLE)),
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
    )

    np.testing.assert_array_equal(np.asarray(extremum), np.asarray(extremum_seed))
    np.testing.assert_array_equal(np.asarray(saddle), np.asarray(saddle_seed))
    assert receipt["fit_converged"].tolist() == [False, False]
    assert receipt["converged"].tolist() == [False, False]


def test_unstructured_support_reports_polish_not_attempted():
    """Sparse carrier support retains the census with a truthful zero receipt."""
    radial, vertical, values, _level_set = _carrier((13, 15))
    sample_valid = jnp.zeros(values.shape, dtype=bool).at[::3, ::3].set(True)
    sparse_values = jnp.where(sample_valid, values, jnp.nan)
    extremum_seed = _selected_row(AXIS + (0.005, 0.0))
    saddle_seed = _selected_row(SADDLE)

    extremum, saddle, receipt = polish_census_stationary_points(
        sparse_values,
        radial,
        vertical,
        solovev_flux(jnp.asarray(SADDLE)),
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
        sample_valid,
    )

    np.testing.assert_array_equal(np.asarray(extremum), np.asarray(extremum_seed))
    np.testing.assert_array_equal(np.asarray(saddle), np.asarray(saddle_seed))
    assert receipt["fit_attempted"].tolist() == [False, False]
    assert receipt["fit_iterations"].tolist() == [0, 0]
    assert receipt["iteration_count"].tolist() == [0, 0]
    np.testing.assert_array_equal(np.asarray(receipt["fit_residual"]), 0.0)


def test_limited_read_fits_at_the_wall_contact_level():
    topology, psi, inside, _radius, _height = _topology_and_flux(GEOMETRIES[0], 33)
    grid_flux, _wall_flux = topology.split_flux_map(psi)
    extrema, saddles = topology.grid(grid_flux)
    provisional_extremum = topology.o_point_data(extrema, -1)
    census_saddle = topology.x_point_data(saddles, -1, provisional_extremum[2])
    result = topology.read_qualification(psi, -1, inside, int(TopologyClass.LIMITED))

    np.testing.assert_allclose(
        np.asarray(result.polish_receipt["interface_value"]),
        np.asarray((result.state.wall_point_flux, result.state.wall_point_flux)),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.isclose(
        float(result.state.wall_point_flux),
        float(census_saddle[2]),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("size", GRID_SIZES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda item: item.name)
def test_secondary_null_polish_receipt_is_explicit(geometry, size):
    topology, psi, inside, _radius, _height = _topology_and_flux(geometry, size)
    result = topology.read_qualification(psi, -1, inside, int(TopologyClass.DIVERTED))
    receipt = result.polish_receipt
    accepted = np.asarray(receipt["converged"]).tolist()
    normalized_gradient = np.asarray(receipt["normalized_gradient"]).tolist()
    normalized_value_change = np.asarray(receipt["normalized_value_change"]).tolist()
    value_tolerance = np.asarray(receipt["value_consistency_tolerance"]).tolist()
    tolerance = np.asarray(receipt["stationarity_tolerance"]).tolist()
    roundoff_floor = np.asarray(receipt["roundoff_floor"]).tolist()
    representation_floor = np.asarray(receipt["representation_floor"]).tolist()
    fit_converged = np.asarray(receipt["fit_converged"]).tolist()
    print(
        f"secondary_null_receipt case={geometry.name}-{size} "
        f"fit_converged={fit_converged} accepted={accepted} "
        f"normalized_gradient={normalized_gradient} "
        f"normalized_value_change={normalized_value_change} "
        f"value_tolerance={value_tolerance} tolerance={tolerance}"
        f" roundoff_floor={roundoff_floor} representation_floor={representation_floor}"
    )

    assert fit_converged == [True, True]
    assert np.all(np.isfinite(normalized_gradient))
    assert np.all(np.isfinite(normalized_value_change))
    assert np.all(np.asarray(tolerance) > 0.0)
    retained = ~np.asarray(receipt["converged"])
    np.testing.assert_array_equal(
        np.asarray(receipt["selected_position_rz"])[retained],
        np.asarray(receipt["census_position_rz"])[retained],
    )


def test_production_capacity_fit_receipts_report_solve_convergence():
    receipts = []
    for shape in ((33, 33), (65, 65)):
        radial, vertical, values, level_set = _carrier(shape)
        spline = fit_split_spline(radial, vertical, values, level_set)
        receipts.append(
            (
                shape,
                int(spline.solve_iterations),
                float(spline.solve_residual),
                bool(spline.solve_converged),
            )
        )

    print(f"production_fit_receipts={receipts}")
    assert [row[1] for row in receipts] == [1, 1]
    assert all(row[2] < 1.0e-10 for row in receipts)
    assert all(row[3] for row in receipts)
