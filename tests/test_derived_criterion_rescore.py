from __future__ import annotations

import copy
import json

import pytest

from benchmarks import derived_criterion_rescore as rescore


@pytest.fixture(scope="module")
def receipt() -> dict:
    return rescore.build_receipt()


def test_comparison_table_covers_the_frozen_six(receipt):
    rows = receipt["comparison_table"]

    assert len(rows) == 6
    assert {row["reference"] for row in rows} == rescore.EXPECTED_REFERENCES
    for row in rows:
        assert row["closest_residual"] >= 0.0
        assert row["registered_criterion"] == 1.0e-8
        assert row["derived_criterion"] > 0.0
        assert row["registered_verdict"] in {
            "PASS",
            "FAIL",
            rescore.INVALID_ROOT_VERDICT,
        }
        assert row["derived_verdict"] in {
            "PASS",
            "FAIL",
            rescore.INVALID_ROOT_VERDICT,
        }


def test_registered_and_derived_counts_are_retained_side_by_side(receipt):
    counts = receipt["counts"]

    assert counts["registered_1e8"] == {
        "criterion": 1.0e-8,
        "numeric_threshold_met_count": 1,
        "invalid_physical_root_count": 1,
        "physical_convergence_count": 0,
        "display": "0 of 6",
    }
    assert counts["derived_held_out"] == {
        "per_reference_criterion": True,
        "numeric_threshold_met_count": 5,
        "invalid_physical_root_count": 1,
        "physical_convergence_count": 4,
        "display": "4 of 6",
    }
    assert counts["both_counts_retained"] is True


def test_per_reference_derived_verdicts_are_exact(receipt):
    rows = {row["reference"]: row for row in receipt["comparison_table"]}

    for reference in ("21978/35", "21983/35", "21985/51", "21986/46"):
        assert rows[reference]["derived_verdict"] == "PASS"
        assert rows[reference]["derived_margin"] > 0.0
    assert rows["22086/43"]["derived_verdict"] == "FAIL"
    assert rows["22086/43"]["derived_margin"] < 0.0
    assert rows["21989/55"]["derived_verdict"] == rescore.INVALID_ROOT_VERDICT


def test_vacuum_zero_is_visible_but_never_counted(receipt):
    rows = {row["reference"]: row for row in receipt["comparison_table"]}
    vacuum = rows["21989/55"]
    policy = receipt["vacuum_collapse_policy"]

    assert vacuum["closest_residual"] == 0.0
    assert vacuum["registered_numeric_threshold_met"] is True
    assert vacuum["derived_numeric_threshold_met"] is True
    assert vacuum["physical_root_valid"] is False
    assert vacuum["registered_verdict"] == rescore.INVALID_ROOT_VERDICT
    assert vacuum["derived_verdict"] == rescore.INVALID_ROOT_VERDICT
    assert policy["counted_as_converged"] is False
    assert "zero-current vacuum collapse" in policy["reason"]


def test_evidence_join_fails_closed_on_residual_disagreement():
    criterion = json.loads(rescore.CRITERION_SOURCE.read_text())
    scorecard = json.loads(rescore.SCORECARD_SOURCE.read_text())
    changed = copy.deepcopy(scorecard)
    changed["per_shot_table"][0]["closest_residual"] += 1.0e-6

    with pytest.raises(RuntimeError, match="evidence disagree"):
        rescore.build_receipt_from_data(criterion, changed)


def test_catalog_hold_implication_is_one_sentence_and_remains(receipt):
    verdict = receipt["verdict"]
    implication = verdict["catalog_hold_implication"]

    assert verdict["catalog_hold_remains"] is True
    assert implication.count(".") == 1
    assert "four of six" in implication
    assert "22086/43" in implication
    assert "21989/55" in implication


def test_claim_bounds_preserve_banked_only_scope(receipt):
    bounds = receipt["claim_bounds"]

    assert receipt["receipt"]["equilibrium_solves_run"] == 0
    assert bounds["new_equilibrium_solve"] is False
    assert bounds["banked_residuals_only"] is True
    assert bounds["registered_tolerance_changed"] is False
    assert bounds["derived_criterion_changed"] is False
    assert "two distinct mesh spacings" in bounds["criterion_limit_retained"]


def test_banked_receipt_matches_regeneration(tmp_path, receipt):
    checked = json.loads(rescore.OUTPUT_PATH.read_text())
    regenerated = rescore.write_receipt(tmp_path / "receipt.json")

    assert checked == regenerated == receipt
    assert set(receipt["sources"]) == {
        str(rescore.CRITERION_SOURCE),
        str(rescore.SCORECARD_SOURCE),
    }
