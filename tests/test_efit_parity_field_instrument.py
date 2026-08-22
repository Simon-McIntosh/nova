from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_field_instrument import (
    BANKED_SOLVED_FIELD_ENERGY,
    BANKED_STORED_BOUNDARY_AREA,
    BANKED_STORED_BOUNDARY_VOLUME,
    EXPECTED_REBASELINED_SOLVED_FIELD_ENERGY,
    EXPECTED_REFERENCE_MAP_FIELD_ENERGY,
    PUBLISHED_REFERENCE_FIELD_ENERGY,
    _plot_fields,
    measure,
)


@pytest.fixture(scope="module")
def result():
    return measure()


def test_control_reuses_one_banked_observation_and_integrator(result):
    receipt, _fields = result
    execution = receipt["execution_contract"]
    assert execution["nonlinear_solve_calls"] == 0
    assert execution["equilibrium_observations_from_serialised_state"] == 1
    assert execution["reference_flux_source"] == "2*pi*efm/psirz"
    assert execution["reference_native_grid_shape"] == [65, 65]
    assert execution["nova_operator_grid_shape"] == [33, 33]
    assert execution["imported_integrator"].endswith("._boundary_moments")
    assert execution["support_policy"] == (
        "one 243-centroid stored-LCFS mask for both fields"
    )
    integrity = receipt["protected_banked_artifacts"]
    assert integrity["verified_digest_count"] == 23
    assert integrity["all_digests_match"] is True


def test_stored_boundary_support_and_accuracy_floor_are_explicit(result):
    receipt, _fields = result
    region = receipt["controlled_region"]
    assert region["contour"] == "stored_lcfs"
    assert region["contour_point_count"] == 151
    assert region["authoritative_cell_count"] == 243
    assert region["banked_solved_branch_cell_count"] == 234
    assert region["overlap_cell_count"] == 231
    assert region["symmetric_difference_cell_count"] == 15
    assert region["stored_only_cell_count"] == 12
    assert region["solved_only_cell_count"] == 3
    assert region["exact_poloidal_area_m2"] == BANKED_STORED_BOUNDARY_AREA
    assert region["exact_toroidal_volume_m3"] == BANKED_STORED_BOUNDARY_VOLUME
    assert region["cell_quadrature_poloidal_area_m2"] == pytest.approx(
        1.8414843762730015, rel=2.0e-15
    )
    assert region["cell_quadrature_toroidal_volume_m3"] == pytest.approx(
        8.831675636804732, rel=2.0e-15
    )
    assert region[
        "cell_quadrature_area_relative_error_from_exact_contour"
    ] == pytest.approx(0.0024246403684322626, rel=2.0e-15)
    assert region[
        "cell_quadrature_volume_relative_error_from_exact_contour"
    ] == pytest.approx(0.004475907677901958, rel=2.0e-15)


def test_three_field_energy_numbers_share_the_declared_support(result):
    receipt, _fields = result
    table = receipt["field_energy_comparison_table"]
    assert [row["field"] for row in table] == [
        "reference_own_map",
        "nova_solved_terminal_map",
        "reference_own_map",
    ]
    assert table[0]["support"] == "stored_lcfs_243_centroids"
    assert table[1]["support"] == "stored_lcfs_243_centroids"
    assert table[0]["operator"] == "nova"
    assert table[1]["operator"] == "nova"
    assert table[2]["operator"] == "reference_published"
    assert table[2]["field_energy_t2_m3"] == PUBLISHED_REFERENCE_FIELD_ENERGY
    assert table[0]["field_energy_t2_m3"] == pytest.approx(
        EXPECTED_REFERENCE_MAP_FIELD_ENERGY, rel=2.0e-15
    )
    assert table[1]["field_energy_t2_m3"] == pytest.approx(
        EXPECTED_REBASELINED_SOLVED_FIELD_ENERGY, rel=2.0e-15
    )
    assert np.all(np.isfinite([row["field_energy_t2_m3"] for row in table]))


def test_instrument_and_physics_ratios_close_multiplicatively(result):
    receipt, _fields = result
    table = receipt["field_energy_comparison_table"]
    reference_energy = table[0]["field_energy_t2_m3"]
    solved_energy = table[1]["field_energy_t2_m3"]
    control = receipt["instrument_control"]
    assert control[
        "nova_operator_on_reference_over_reference_published"
    ] == pytest.approx(reference_energy / PUBLISHED_REFERENCE_FIELD_ENERGY)
    assert control["nova_solved_over_nova_operator_on_reference"] == pytest.approx(
        solved_energy / reference_energy
    )
    assert control["multiplicative_closure_residual"] == pytest.approx(0.0, abs=2.0e-16)
    split = control["deficit_split_on_published_energy_scale"]
    assert split["instrument_fraction"] + split["physics_fraction"] == pytest.approx(
        1.0, abs=2.0e-15
    )
    assert split["additive_closure_residual_t2_m3"] == pytest.approx(0.0, abs=2.0e-16)


def test_support_rebaseline_is_visible_and_not_the_full_deficit(result):
    receipt, _fields = result
    region = receipt["controlled_region"]
    assert region["banked_solved_field_energy_t2_m3"] == BANKED_SOLVED_FIELD_ENERGY
    assert region["net_area_change_from_banked_support_m2"] == pytest.approx(
        0.0682031250471482, rel=2.0e-15
    )
    assert region["net_area_change_fraction_of_banked_support"] == pytest.approx(
        0.038461538461538464, rel=2.0e-15
    )
    assert region["support_shift_cannot_explain_prior_deficit"] is True
    assert (
        abs(region["signed_relative_field_energy_shift"])
        < region["prior_published_deficit_magnitude"]
    )


def test_all_outcome_branches_are_declared_before_the_measured_verdict(result):
    receipt, _fields = result
    declared = receipt["outcomes_declared_before_measurement"]
    assert declared["reference_map_near_reference_published"]["row_disposition"] == (
        "RETAIN_AS_PHYSICS_DIFFERENCE"
    )
    assert declared["reference_map_near_nova_solved"]["row_disposition"] == ("RETRACT")
    assert declared["reference_map_near_neither_endpoint"]["row_disposition"] == (
        "RETAIN_WITH_SPLIT_ATTRIBUTION"
    )
    control = receipt["instrument_control"]
    assert control["verdict"] == "INSTRUMENT_DIFFERENCE"
    assert control["row_disposition"] == "RETRACT"
    assert control[
        "nova_operator_on_reference_over_reference_published"
    ] == pytest.approx(0.6020235565543425, rel=2.0e-15)
    assert control["nova_solved_over_nova_operator_on_reference"] == pytest.approx(
        0.9994265612566154, rel=2.0e-15
    )
    assert control[
        "physics_signed_relative_deviation_after_instrument_division"
    ] == pytest.approx(-0.0005734387433845578, rel=2.0e-15)


def test_field_figure_uses_shared_support_and_colour_scale(result, tmp_path):
    receipt, fields = result
    assert np.count_nonzero(fields["inside"]) == 243
    assert fields["solved_field"].shape == (1089,)
    assert fields["reference_field"].shape == (1089,)
    figure = Path(tmp_path) / "field-control.png"
    _plot_fields(fields, receipt, figure)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
