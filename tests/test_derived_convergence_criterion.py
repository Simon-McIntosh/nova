from __future__ import annotations

import copy
import json
import math

import pytest

from benchmarks import derived_convergence_criterion as criterion


@pytest.fixture(scope="module")
def receipt() -> dict:
    return criterion.build_receipt()


@pytest.fixture(scope="module")
def refreshed_receipt() -> dict:
    return criterion.build_three_spacing_receipt()


def test_receipt_has_positive_values_for_the_frozen_six(receipt):
    rows = receipt["per_reference"]

    assert len(rows) == 6
    assert {row["reference"] for row in rows} == {
        "21978/35",
        "21983/35",
        "21985/51",
        "21986/46",
        "21989/55",
        "22086/43",
    }
    assert all(
        math.isfinite(row["derived_criterion"]) and row["derived_criterion"] > 0
        for row in rows
    )


def test_each_fit_holds_out_the_target_and_stays_within_its_stratum(receipt):
    strata = {row["reference"]: row["stratum"] for row in receipt["per_reference"]}

    for row in receipt["per_reference"]:
        fit = row["fit"]
        assert row["reference"] == fit["held_out_target"]
        assert row["reference"] not in fit["fit_references"]
        assert fit["target_residual_used_in_fit"] is False
        assert fit["target_mesh_pair_used_in_fit"] is False
        assert all(strata[peer] == row["stratum"] for peer in fit["fit_references"])


def test_mutating_a_gated_residual_does_not_change_any_criterion():
    mesh, topology, scorecard = criterion._load_sources()
    baseline = criterion.build_receipt_from_data(mesh, topology, scorecard)
    changed_scorecard = copy.deepcopy(scorecard)
    changed_scorecard["per_shot"][0]["closest_approach"]["residual"] = 0.987654321
    changed = criterion.build_receipt_from_data(mesh, topology, changed_scorecard)

    assert [row["derived_criterion"] for row in changed["per_reference"]] == [
        row["derived_criterion"] for row in baseline["per_reference"]
    ]
    assert changed["per_reference"][0]["gated_closest_residual_display_only"] == (
        0.987654321
    )


def test_coarse_mesh_criteria_equal_peer_geometric_means(receipt):
    rows = {row["reference"]: row for row in receipt["per_reference"]}
    coarse = {
        "21978/35": 0.010356795810607609,
        "21983/35": 0.010050357331518338,
        "21985/51": 0.01294253111102598,
        "21989/55": 0.017714765374879503,
        "22086/43": 0.011016939705021812,
    }

    assert rows["21983/35"]["derived_criterion"] == pytest.approx(coarse["21985/51"])
    assert rows["21985/51"]["derived_criterion"] == pytest.approx(coarse["21983/35"])
    assert rows["21978/35"]["derived_criterion"] == pytest.approx(
        math.sqrt(coarse["21989/55"] * coarse["22086/43"])
    )
    assert rows["21986/46"]["derived_criterion"] == pytest.approx(
        (coarse["21978/35"] * coarse["21989/55"] * coarse["22086/43"]) ** (1.0 / 3.0)
    )
    assert rows["21989/55"]["derived_criterion"] == pytest.approx(
        math.sqrt(coarse["21978/35"] * coarse["22086/43"])
    )
    assert rows["22086/43"]["derived_criterion"] == pytest.approx(
        math.sqrt(coarse["21978/35"] * coarse["21989/55"])
    )


def test_two_banked_strata_are_not_pooled(receipt):
    assert receipt["strata"]["closed-axis"]["frozen_references"] == [
        "21983/35",
        "21985/51",
    ]
    assert receipt["strata"]["confinement-construction"]["frozen_references"] == [
        "21978/35",
        "21986/46",
        "21989/55",
        "22086/43",
    ]
    assert receipt["criterion"]["stratification"].endswith("never pooled")


def test_circular_richardson_estimator_collapses_to_the_fine_residual(receipt):
    excluded = receipt["excluded_circular_estimator"]

    assert excluded["algebraic_collapse"][-1] == "E_f_i = R_fine_i"
    assert len(excluded["per_reference_numeric_check"]) == 5
    for row in excluded["per_reference_numeric_check"]:
        assert row["richardson_estimate"] == pytest.approx(
            row["fine_residual"], rel=2.0e-15
        )
        assert row["estimate_over_fine_residual"] == pytest.approx(1.0)


def test_domain_limits_and_low_order_qualifications_are_explicit(receipt):
    rows = {row["reference"]: row for row in receipt["per_reference"]}

    assert receipt["claim_bounds"]["third_mesh_available"] is False
    assert receipt["claim_bounds"]["independent_asymptotic_confirmation"] is False
    assert rows["21978/35"]["target_qualification"].startswith(
        "LEAST-TRUSTWORTHY TARGET"
    )
    assert "0.966596902394278" in rows["21978/35"]["target_qualification"]
    assert rows["22086/43"]["target_qualification"].startswith(
        "LEAST-TRUSTWORTHY TARGET"
    )
    assert "1.6371714162964488" in rows["22086/43"]["target_qualification"]


def test_receipt_is_banked_and_reproducible(tmp_path, receipt):
    checked = json.loads(criterion.OUTPUT_PATH.read_text())
    regenerated = criterion.write_receipt(tmp_path / "receipt.json")

    assert regenerated == receipt
    benchmark_source = str(criterion.BENCHMARK_SOURCE)
    checked["sources"].pop(benchmark_source)
    current_sources = copy.deepcopy(receipt["sources"])
    current_sources.pop(benchmark_source)
    assert checked == receipt | {"sources": current_sources}
    assert receipt["sources"][benchmark_source] == criterion._sha256(
        criterion.BENCHMARK_SOURCE
    )
    assert receipt["receipt"]["equilibrium_solves_run"] == 0
    assert receipt["criterion"]["registered_tolerance_changed"] is False
    assert set(receipt["sources"]) == {
        str(criterion.MESH_SOURCE),
        str(criterion.TOPOLOGY_SOURCE),
        str(criterion.GATED_RESIDUAL_SOURCE),
        str(criterion.BENCHMARK_SOURCE),
    }


def test_three_spacing_refresh_banks_both_criteria_for_the_frozen_six(
    refreshed_receipt,
):
    rows = refreshed_receipt["per_reference"]

    assert len(rows) == 6
    assert refreshed_receipt["receipt"]["refreshed_reference_count"] == 3
    assert refreshed_receipt["receipt"]["retained_reference_count"] == 3
    assert all(row["two_spacing_criterion"] > 0.0 for row in rows)
    assert all(row["refreshed_criterion"] > 0.0 for row in rows)


def test_third_rung_is_used_only_as_held_out_peer_evidence(refreshed_receipt):
    rows = {row["reference"]: row for row in refreshed_receipt["per_reference"]}

    for row in rows.values():
        fit = row["fit"]
        assert row["reference"] not in fit["fit_references"]
        assert fit["target_residual_used_in_fit"] is False
        assert fit["target_mesh_pair_used_in_fit"] is False
        assert all(
            level["reference"] != row["reference"] for level in fit["fit_levels"]
        )

    assert rows["21978/35"]["third_spacing_used"] is False
    assert all(
        row["third_spacing_used"]
        for reference, row in rows.items()
        if reference in {"21986/46", "21989/55", "22086/43"}
    )


def test_refresh_qualifications_are_stated_per_stratum(refreshed_receipt):
    closed = refreshed_receipt["strata"]["closed-axis"]
    confinement = refreshed_receipt["strata"]["confinement-construction"]

    assert closed["qualification_status"] == "RETAINED"
    assert closed["targets_with_three_spacing_fit"] == []
    assert confinement["qualification_status"] == "PARTIALLY_LIFTED"
    assert confinement["targets_with_three_spacing_fit"] == [
        "21986/46",
        "21989/55",
        "22086/43",
    ]
    assert confinement["targets_retaining_two_spacing_qualification"] == ["21978/35"]


def test_three_spacing_refresh_values(refreshed_receipt):
    rows = {row["reference"]: row for row in refreshed_receipt["per_reference"]}

    expected = {
        "21978/35": 0.013970057337880022,
        "21983/35": 0.012942531111025984,
        "21985/51": 0.010050357331518324,
        "21986/46": 0.01188975144741377,
        "21989/55": 0.010613918700193955,
        "22086/43": 0.012389584260990242,
    }
    assert {
        reference: row["refreshed_criterion"] for reference, row in rows.items()
    } == pytest.approx(expected)


def test_gated_residual_is_not_an_input_to_refreshed_criteria():
    mesh, topology, scorecard = criterion._load_sources()
    third_spacing_source = json.loads(criterion.THREE_SPACING_SOURCE.read_text())
    baseline = criterion.build_three_spacing_receipt_from_data(
        mesh, topology, scorecard, third_spacing_source
    )
    changed_scorecard = copy.deepcopy(scorecard)
    changed_scorecard["per_shot"][0]["closest_approach"]["residual"] = 0.987654321
    changed = criterion.build_three_spacing_receipt_from_data(
        mesh, topology, changed_scorecard, third_spacing_source
    )

    assert [row["refreshed_criterion"] for row in changed["per_reference"]] == [
        row["refreshed_criterion"] for row in baseline["per_reference"]
    ]


def test_three_spacing_receipt_is_banked_and_reproducible(tmp_path, refreshed_receipt):
    checked = json.loads(criterion.THREE_SPACING_OUTPUT_PATH.read_text())
    regenerated = criterion.write_three_spacing_receipt(tmp_path / "receipt.json")

    assert checked == regenerated == refreshed_receipt
    assert refreshed_receipt["receipt"]["equilibrium_solves_run"] == 0
    assert refreshed_receipt["criterion"]["registered_tolerance_changed"] is False
    assert str(criterion.THREE_SPACING_SOURCE) in refreshed_receipt["sources"]
