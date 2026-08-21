"""Tests for amplitude-sensitive DIII-D flux-space adjudication."""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import binary_dilation

from benchmarks import diiid_flux_space_adjudication as adjudication


def test_preregistration_fixes_arms_thresholds_and_decision_rule() -> None:
    path = adjudication.DEFAULT_OUTPUT / adjudication.PREREGISTRATION_NAME
    registration = json.loads(path.read_text(encoding="utf-8"))
    assert registration["scoring_started_before_commit"] is False
    assert registration["selection"]["frames"] == 60
    assert registration["selection"]["shots"] == 20
    assert set(registration["arms"]) == {"label_recovered", "circuit_derived"}
    assert registration["metric"]["cohort_median_used_for_threshold"] is False
    assert registration["amplitude_sensitivity"]["declared_perturbation_a"] == 1000.0
    assert registration["cohort_decision"]["required_frame_count"] == 54


def test_flux_metric_is_gauge_invariant_and_amplitude_sensitive() -> None:
    target = np.arange(49, dtype=float).reshape(7, 7) * 1.0e-5
    response = np.sin(np.arange(49, dtype=float)).reshape(7, 7) * 1.0e-8
    mask = np.ones((7, 7), dtype=bool)
    baseline = adjudication.gauge_free_rms(target, mask)
    np.testing.assert_allclose(
        adjudication.gauge_free_rms(target + 17.0, mask), baseline, rtol=1.0e-12
    )
    sensitivity = adjudication.perturbation_receipt(
        target,
        np.zeros_like(target),
        response,
        mask,
        baseline,
        1000.0,
    )
    assert sensitivity["field_response_rms_wb"] >= 1.0e-6
    assert sensitivity["passes_declared_sensitivity"]


def test_validation_mask_excludes_fit_boundary_and_dilated_core() -> None:
    core_rz = np.zeros((12, 10), dtype=bool)
    core_rz[5:7, 4:6] = True
    mask = adjudication.validation_mask(core_rz)
    assert mask.shape == (10, 12)
    assert not np.any(mask[:2, :])
    assert not np.any(mask[-2:, :])
    assert not np.any(mask[:, :2])
    assert not np.any(mask[:, -2:])
    assert not np.any(mask[binary_dilation(core_rz.T, iterations=2)])


def test_pattern_comparison_recovers_fixed_ohmic_shape() -> None:
    circuit = np.array([1.01, 0.99, 0.98, 0.97, 1.02])
    result = adjudication.pattern_metrics(20_000.0 * circuit, circuit)
    assert result["shape_relative_rms"] < 1.0e-14
    assert result["cosine"] > 1.0 - 1.0e-14
    assert result["temporal_eligible"]


def test_committed_receipt_scores_full_cohort_and_banks_figure() -> None:
    receipt_path = adjudication.DEFAULT_OUTPUT / adjudication.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["selection"]["frames"] == 60
    assert receipt["selection"]["shots"] == 20
    assert len(receipt["records"]) == 60
    assert receipt["metric"]["cohort_median_used_for_threshold"] is False
    assert receipt["amplitude_sensitivity"]["all_frames_all_coils_both_arms_pass"]
    assert all(
        len(item["label_recovered_currents_a"]) == 5 for item in receipt["records"]
    )
    assert all(
        len(item["circuit_derived_currents_a"]) == 5 for item in receipt["records"]
    )
    assert all(item["design_condition_number"] > 0.0 for item in receipt["records"])
    assert receipt["verdict"]["believed_current_authority"] in {
        "label_recovered",
        "circuit_derived",
        "undecidable",
    }
    assert receipt["verdict"]["statement"]
    assert (adjudication.DEFAULT_OUTPUT / adjudication.FIGURE_NAME).is_file()
