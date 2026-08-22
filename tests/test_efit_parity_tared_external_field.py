from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_tared_external_field import (
    BANKED_STORED_FIELD_CLOSURE_WB,
    GAUGE_CONSTANT_WB,
    OUTPUT_FIGURE,
    REFERENCE_HALO_CURRENT_A,
    REPRESENTATIVE_SHOT,
    run_control,
)


@pytest.fixture(scope="module")
def result():
    return run_control()


def test_tare_closes_all_six_references_at_roundoff(result):
    receipt, runtime = result
    assert receipt["receipt"]["shot_count"] == 6
    assert len(runtime) == 6
    gate = receipt["closure_gate"]
    assert gate["passes"] is True
    assert gate["all_six_at_roundoff"] is True
    assert gate["maximum_sup_difference_wb"] <= BANKED_STORED_FIELD_CLOSURE_WB


def test_delta_star_current_is_partitioned_against_each_declared_support(result):
    receipt, _runtime = result
    rows = receipt["current_integral_table"]
    assert len(rows) == 6
    assert all(row["qualification_passes"] for row in rows)
    assert all(row["declared_support_valid_cell_count"] > 0 for row in rows)
    assert all(row["outside_declared_support_valid_cell_count"] > 0 for row in rows)
    for row in rows:
        assert row["total_valid_stencil_current_integral_a"] == pytest.approx(
            row["declared_support_current_integral_a"]
            + row["outside_declared_support_current_integral_a"],
            abs=2.0e-9,
        )
        assert np.isfinite(row["outside_over_banked_nova_halo"])
    assert (
        receipt["banked_comparison"]["nova_solved_outside_separatrix_current_a"]
        == REFERENCE_HALO_CURRENT_A
    )


def test_tare_uses_the_existing_operator_and_preserves_reference_gauge(result):
    receipt, runtime = result
    construction = receipt["tare_construction"]
    assert construction["delta_star_implementation"].endswith(
        ".conservation.delta_star"
    )
    assert construction["current_relation_implementation"].endswith(
        ".convention.delta_star_from_current_density"
    )
    assert receipt["gauge"]["additive_constant_wb"] == GAUGE_CONSTANT_WB == 0.0
    for item in runtime:
        profile = item["tare"]["profile"]
        assert profile.operator.prescribed_current_field.circuit_count == 1
        assert np.count_nonzero(np.asarray(profile.operator.external_current)) == 0
        assert item["tare"]["closure"]["sample_target_count"] == 0


def test_protected_banked_artifacts_and_control_caveat_are_explicit(result):
    receipt, _runtime = result
    integrity = receipt["protected_banked_artifacts"]
    assert integrity["verified_digest_count"] == 23
    assert integrity["all_digests_match"] is True
    caveat = receipt["control_caveat"].lower()
    assert "control and not a parity claim" in caveat
    assert "must never be cited as parity" in caveat


def test_external_field_figure_compares_tared_and_passive_inclusive_maps(result):
    receipt, _runtime = result
    comparison = receipt["passive_inclusive_comparison"]
    assert comparison["modeled_stored_circuit_count"] == 101
    assert comparison["ordinary_active_drive_zeroed"] is True
    figure = Path(comparison["figure"])
    assert figure == OUTPUT_FIGURE
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_all_six_current_constrained_solves_are_reported(result):
    receipt, _runtime = result
    assert len(receipt["per_shot"]) == 6
    assert len(receipt["six_reference_score_table"]) == 6
    aggregate = receipt["aggregate"]
    assert aggregate["banked_converged_plasma_roots"] == 1
    assert aggregate["registered_fixed_point_criterion"] == 1.0e-8
    assert aggregate["all_target_currents_exact"] is True
    assert aggregate["parity_claim"] is False


def test_raw_and_instrument_controlled_rows_are_banked_side_by_side(result):
    receipt, _runtime = result
    for row in receipt["per_shot"]:
        raw = row["raw_registered_rows"]["metrics"]
        controlled = row["instrument_controlled_rows"]
        assert np.isfinite(raw["lcfs"]["symmetric_mean_distance_m"])
        closed = controlled["lcfs_closed_branch"]
        assert closed["longest_polyline_fallback_used"] is False
        if closed["status"] == "scoreable":
            assert closed["closed_branch_point_count"] >= 4
            assert np.isfinite(closed["distance"]["symmetric_mean_distance_m"])
        else:
            assert closed["status"] == "unscoreable_no_closed_axis_branch"
            assert closed["closed_branch_point_count"] is None
            assert closed["distance"] is None
        matched = controlled["matched_stored_boundary_support"]
        assert matched["cell_count"] > 0
        assert np.isfinite(matched["poloidal_beta_signed_relative_deviation"])
        field = controlled["poloidal_field_energy_instrument_control"]
        assert field["multiplicative_closure_residual"] == pytest.approx(
            0.0, abs=2.0e-16
        )
        assert np.isfinite(field["instrument_controlled_signed_relative_deviation"])


def test_representative_field_instrument_ratio_reproduces_the_bank(result):
    receipt, _runtime = result
    row = next(
        item
        for item in receipt["per_shot"]
        if item["reference"]["shot"] == REPRESENTATIVE_SHOT
    )
    field = row["instrument_controlled_rows"][
        "poloidal_field_energy_instrument_control"
    ]
    assert field["nova_on_reference_over_reference_published"] == pytest.approx(
        field["banked_representative_instrument_ratio"], rel=2.0e-15
    )
    assert field["banked_representative_instrument_ratio"] == pytest.approx(
        0.6020235565543425, rel=2.0e-15
    )
