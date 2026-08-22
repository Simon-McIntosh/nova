"""Tests for fitted-current ensemble relation assessment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import diiid_circuit_ensemble_assessment as assessment


def test_banked_inputs_align_on_every_frame_identity() -> None:
    inputs = assessment.aligned_inputs(assessment.DEFAULT_OUTPUT)
    assert len(inputs["keys"]) == 60
    assert len({key[0] for key in inputs["keys"]}) == 20
    assert inputs["conductors"] == (
        "E567DN",
        "E567UP",
        "E89DN",
        "E89UP",
        "ECOILB",
    )


def test_through_origin_fit_and_intercept_test_distinguish_offset() -> None:
    predictor = np.linspace(-50_000.0, 40_000.0, 60)
    shots = np.repeat(np.arange(20), 3)
    noise = 150.0 * np.sin(np.arange(60, dtype=float))
    origin_response = 1.25 * predictor + noise
    offset_response = origin_response + 4_000.0
    np.testing.assert_allclose(
        assessment.through_origin_slope(predictor, origin_response),
        1.25,
        rtol=0.0,
        atol=1.0e-3,
    )
    origin = assessment.intercept_freedom_test(predictor, origin_response, shots)
    offset = assessment.intercept_freedom_test(predictor, offset_response, shots)
    assert origin["zero_intercept_not_rejected_at_0p05"]
    assert not offset["zero_intercept_not_rejected_at_0p05"]
    assert abs(offset["intercept_a"] - 4_000.0) < 100.0


def test_holm_adjustment_controls_the_complete_test_family() -> None:
    adjusted = assessment._holm_adjust([0.001, 0.01, 0.04, 0.20])
    np.testing.assert_allclose(adjusted, [0.004, 0.03, 0.08, 0.20])


def test_committed_receipt_reports_qualified_relations_and_failed_closure() -> None:
    path = assessment.DEFAULT_OUTPUT / assessment.ASSESSMENT_RECEIPT
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["policy"]["relation_fit_scope"] == "fit-once-static"
    assert receipt["policy"]["promotion_test"] == "both-required"
    assert receipt["cohort"] == {
        "frames": 60,
        "negative_ecoila_supply_frames": 35,
        "positive_ecoila_supply_frames": 25,
        "qualified_recorded_plasma_current_below_50ka_frames": 29,
        "recorded_plasma_current_at_or_above_50ka_frames": 31,
        "shots": 20,
    }
    expected = {
        "E567DN": (1.0016568244749882, 0.982223644561927),
        "E567UP": (1.0231285862990276, 0.9856421407471255),
        "E89DN": (1.0456240764323717, 0.9972019819366135),
        "E89UP": (1.0456947569496173, 0.9974088283729817),
        "ECOILB": (2.0009180770924457, 0.9969169329054671),
    }
    for name, (slope, correlation) in expected.items():
        conductor = receipt["conductors"][name]
        np.testing.assert_allclose(
            conductor["through_origin"]["slope"], slope, rtol=1.0e-12
        )
        np.testing.assert_allclose(
            conductor["through_origin"]["correlation"],
            correlation,
            rtol=1.0e-12,
        )
        assert conductor["leave_one_shot_out_prediction"]["r_squared"] > 0.94
        assert conductor["promotion_recommendation"]["fitted_current_relation"] == (
            "withhold"
        )
        assert (
            conductor["promotion_recommendation"]["joint_both_required_decision"]
            == "failed"
        )
        assert len(conductor["shot_stability"]["per_shot_through_origin_slopes"]) == 20
        assert len(conductor["frame_covariate_drift"]["tested"]) == 9
    assert receipt["conductors"]["ECOILB"]["intercept_freedom"][
        "zero_intercept_not_rejected_at_0p05"
    ]
    for name in ("E567DN", "E567UP", "E89DN", "E89UP"):
        assert not receipt["conductors"][name]["intercept_freedom"][
            "zero_intercept_not_rejected_at_0p05"
        ]
    reconciliation = receipt["residual_threshold_reconciliation"]
    assert reconciliation["closure_frames"] == 1
    assert reconciliation["required_closure_frames"] == 54
    assert reconciliation["total_frames"] == 60
    assert not reconciliation["passes"]
    np.testing.assert_allclose(
        reconciliation["threshold_ratio"]["median"],
        4.1240367695042295,
        rtol=1.0e-12,
    )
    assert "normalisation" in " ".join(
        receipt["conductors"]["ECOILB"]["promotion_recommendation"]["reasons"]
    )


def test_committed_figures_are_nonempty_png_artifacts() -> None:
    for name in (assessment.RELATION_FIGURE, assessment.RECONCILIATION_FIGURE):
        path = assessment.DEFAULT_OUTPUT / name
        assert path.is_file()
        assert path.stat().st_size > 50_000
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
