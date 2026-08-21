from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_recovered_current_sign_control as control


def _records() -> list[dict[str, object]]:
    records = []
    for index in range(8):
        baseline = float(index + 1)
        improves = index >= 4
        records.append(
            {
                "recovered_currents_improve_residual": improves,
                "predictors": {
                    "without_recovered_currents_fractional_rms": baseline,
                    "plasma_current_a": float((-1) ** index),
                    "recovery_relative_residual": float(index % 2),
                    "gauged_boundary_discrepancy_fractional_rms": float(index % 3),
                    "labelled_lcfs_elongation": 1.5,
                },
            }
        )
    return records


def test_preregistration_declares_cohort_predictors_and_ranking() -> None:
    declared = control.preregistration()
    assert declared["selection"]["frame_count"] >= 60
    assert declared["selection"]["shot_count"] >= 20
    assert declared["candidate_predictors_in_declared_order"] == list(
        control.PREDICTORS
    )
    assert declared["coefficients_fitted"] == 0
    assert declared["currents_adjusted"] == 0


def test_predictor_separation_finds_perfect_increasing_crossover() -> None:
    result = control.predictor_separation(
        np.arange(8.0), np.asarray([False] * 4 + [True] * 4)
    )
    assert result["orientation_invariant_auc"] == 1.0
    assert result["absolute_rank_biserial_separation"] == 1.0
    assert result["crossover_value"] == 3.5
    assert result["crossover_direction"] == "improving_above"
    assert result["crossover_balanced_accuracy"] == 1.0


def test_predictor_separation_is_orientation_invariant() -> None:
    outcome = np.asarray([False] * 4 + [True] * 4)
    ascending = control.predictor_separation(np.arange(8.0), outcome)
    descending = control.predictor_separation(-np.arange(8.0), outcome)
    assert (
        descending["orientation_invariant_auc"]
        == ascending["orientation_invariant_auc"]
    )
    assert descending["rank_direction"] == "smaller_values_improve"
    assert descending["crossover_direction"] == "improving_below"


def test_predictor_separation_refuses_one_class() -> None:
    try:
        control.predictor_separation(np.arange(4.0), np.ones(4, dtype=bool))
    except ValueError as error:
        assert "improving and worsening" in str(error)
    else:
        raise AssertionError("one-class ranking must fail closed")


def test_rank_predictors_uses_declared_order_after_metrics() -> None:
    ranked = control.rank_predictors(_records())
    assert ranked[0]["predictor"] == "without_recovered_currents_fractional_rms"
    assert ranked[0]["orientation_invariant_auc"] == 1.0
    assert {item["predictor"] for item in ranked} == set(control.PREDICTORS)


def test_baselines_are_compared_only_to_compatible_crossover() -> None:
    comparison = control.baseline_crossover_comparison(
        {
            "predictor": "without_recovered_currents_fractional_rms",
            "crossover_value": 1.0,
        }
    )
    assert comparison["applicable"]
    assert comparison["on_opposite_sides"]
    refused = control.baseline_crossover_comparison(
        {"predictor": "plasma_current_a", "crossover_value": 1.0}
    )
    assert not refused["applicable"]
    assert refused["on_opposite_sides"] is None


def test_lcfs_elongation_uses_physical_spans() -> None:
    row = {
        "efit_lcfs_n": np.asarray([8]),
        "efit_lcfs_r": [np.asarray([1.0, 1.5, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0])],
        "efit_lcfs_z": [np.asarray([-2.0, -2.0, -2.0, 0.0, 2.0, 2.0, 2.0, 0.0])],
    }
    assert control._lcfs_elongation(row, 0) == 4.0


def test_committed_receipt_carries_complete_sign_measurement() -> None:
    receipt = json.loads((control.DEFAULT_OUTPUT / control.RECEIPT_NAME).read_text())
    cohort = receipt["cohort"]
    assert cohort["frame_count"] >= 60
    assert cohort["shot_count"] >= 20
    assert cohort["all_shots_absent_from_affected_population"]
    assert cohort["improving_frame_count"] > 0
    assert cohort["worsening_frame_count"] > 0
    assert len(cohort["frames"]) == cohort["frame_count"]
    assert all(
        set(frame["predictors"]) == set(control.PREDICTORS)
        for frame in cohort["frames"]
    )
    assert all(
        "without_recovered_currents_fractional_rms" in frame
        and "with_recovered_currents_fractional_rms" in frame
        for frame in cohort["frames"]
    )
    assert [item["predictor"] for item in receipt["predictor_ranking"]] == [
        item["predictor"]
        for item in sorted(
            receipt["predictor_ranking"],
            key=lambda item: (
                -item["orientation_invariant_auc"],
                -item["crossover_balanced_accuracy"],
                item["declared_order"],
            ),
        )
    ]
    assert "crossover_value" in receipt["winning_predictor"]
    comparison = receipt["historical_and_replacement_baselines"]
    assert "on_opposite_sides" in comparison
