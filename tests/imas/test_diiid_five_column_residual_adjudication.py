"""Tests for the DIII-D five-column residual adjudication."""

from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_five_column_residual_adjudication as adjudication


def test_banked_cohort_is_exactly_the_committed_polarity_screened_set() -> None:
    cohort, source = adjudication.banked_cohort(
        adjudication.DEFAULT_DATA, adjudication.SOURCE_RECEIPT
    )
    assert len(cohort) == 60
    assert len({item.path.name for item in cohort}) == 20
    assert source["selection"]["all_selected_absent_from_polarity_population"]
    expected = [
        item["exact_clipped_moments"]["absolute_signed_fraction_of_extracted_current"]
        for item in source["records"]
    ]
    np.testing.assert_allclose(
        [item.expected_fraction for item in cohort], expected, rtol=0.0, atol=0.0
    )


def test_current_residual_separates_incomplete_edge_stencils() -> None:
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-0.6, 0.6, 7)
    metrics, density, valid = adjudication.current_residual_metrics(
        radius,
        height,
        np.zeros((7, 7)),
        np.ones((7, 7), dtype=bool),
        1.0e6,
    )
    assert metrics["absolute_signed_percent_of_extracted_current"] == 0.0
    assert metrics["interior_valid_nodes"] == 9
    assert metrics["edge_stencil"]["nodes"] == 40
    assert not metrics["edge_stencil"]["included_in_interior_current_comparison"]
    assert metrics["edge_stencil"]["delta_star_value"] is None
    assert np.all(np.isfinite(density[valid]))


def test_spatial_classifier_identifies_smooth_surviving_structure() -> None:
    shape = (21, 21)
    valid = np.ones(shape, dtype=bool)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    density = np.zeros(shape)
    density[valid] = 2.0
    radius_profile = np.linspace(-1.0, 1.0, shape[0])[:, None]
    vertical_profile = np.linspace(-1.0, 1.0, shape[1])[None, :]
    flux = radius_profile + 0.4 * vertical_profile
    omitted = np.stack(
        [
            np.sin((index + 1) * radius_profile)
            * np.cos((index + 1) * vertical_profile)
            for index in range(5)
        ]
    )
    result = adjudication.classify_spatial_residual(density, valid, flux.T, omitted)
    assert result["classification"] == "smooth"
    assert set(result["scores"]) == {
        "edge_concentrated",
        "smooth",
        "conductor_like",
    }
    assert len(result["projection_currents_a"]) == 5


def test_committed_receipt_contains_every_frame_and_explicit_verdict() -> None:
    receipt_path = adjudication.DEFAULT_OUTPUT / adjudication.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["selection"]["frames"] == 60
    assert receipt["selection"]["shots"] == 20
    assert receipt["selection"]["all_frames_absent_from_polarity_population"]
    assert len(receipt["records"]) == 60
    assert receipt["comparison"]["exact_tare_floor_percent"] == 0.4841505
    assert receipt["comparison"]["states"] == list(adjudication.STAGE_NAMES)
    assert all(
        set(record["residuals"]) == set(adjudication.STAGE_NAMES)
        for record in receipt["records"]
    )
    assert all(
        record["residuals"][name]["edge_stencil"][
            "included_in_interior_current_comparison"
        ]
        is False
        for record in receipt["records"]
        for name in adjudication.STAGE_NAMES
    )
    assert all(record["design_condition_number"] > 0.0 for record in receipt["records"])
    assert all(
        len(record["recovered_currents_a"]) == 5 for record in receipt["records"]
    )
    disagreement = receipt["current_disagreement"]
    assert disagreement["believed_authority"] == "ohmic_circuit_relation"
    assert disagreement["post_recovery_frames_above_exact_tare_floor"] == 30
    assert disagreement["post_recovery_frames_at_or_below_exact_tare_floor"] == 30
    assert "believed" in disagreement["verdict"]
    diagnostic = receipt["reuse_authority"]["diagnostic_manifest"]
    assert diagnostic["terminal"]
    assert diagnostic["status"] == "complete"
    assert sum(receipt["spatial_summary"]["classification_counts"].values()) == 60
    assert (adjudication.DEFAULT_OUTPUT / adjudication.COHORT_FIGURE_NAME).is_file()
    assert (adjudication.DEFAULT_OUTPUT / adjudication.SPATIAL_FIGURE_NAME).is_file()
