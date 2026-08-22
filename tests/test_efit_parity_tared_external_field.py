from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_tared_external_field import (
    BANKED_STORED_FIELD_CLOSURE_WB,
    GAUGE_CONSTANT_WB,
    OUTPUT_FIGURE,
    REFERENCE_HALO_CURRENT_A,
    measure_tares,
)


@pytest.fixture(scope="module")
def result():
    return measure_tares()


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
