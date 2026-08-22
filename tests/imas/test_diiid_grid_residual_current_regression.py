"""Tests for full-grid omitted-conductor current regression."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import diiid_boundary_current_recovery as recovery
from benchmarks import diiid_grid_residual_current_regression as regression


def test_tikhonov_regression_recovers_coefficients_and_additive_gauge() -> None:
    generator = np.random.default_rng(119)
    design = generator.normal(size=(300, 5))
    expected = np.array([48_000.0, -6_000.0, 12_000.0, 3_500.0, -7_000.0])
    gauge = 0.037
    target = design @ expected + gauge
    solved = regression.tikhonov_regression(
        design, target, tuple(float(value) for value in np.logspace(-14, -6, 17))
    )
    np.testing.assert_allclose(solved["ampere_turns"], expected, rtol=2.0e-6)
    np.testing.assert_allclose(solved["gauge_wb"], gauge, rtol=0.0, atol=1.0e-9)
    assert regression._norm_metrics(solved["post_residual"])["rms_wb"] < 0.1


def test_regularization_diagnostics_bank_the_declared_sensitivity_sweep() -> None:
    radius = np.linspace(-1.0, 1.0, 121)
    design = np.column_stack([np.sin((index + 1) * radius) for index in range(5)])
    target = design @ np.arange(1.0, 6.0) + 2.0e-3 * np.cos(9.0 * radius)
    lambdas = tuple(float(value) for value in np.logspace(-10, 1, 12))
    diagnostics = regression.tikhonov_regression(design, target, lambdas)["diagnostics"]
    assert diagnostics["criterion"] == "minimum generalized cross-validation score"
    assert len(diagnostics["sweep"]) == len(lambdas)
    assert diagnostics["design_rank"] == 5
    assert diagnostics["scaled_gauge_free_design_condition_number"] >= 1.0
    assert diagnostics["one_decade_relative_current_change"] >= 0.0
    assert diagnostics["full_sweep_maximum_relative_current_change"] >= 0.0
    assert set(diagnostics["per_coil_sweep_ampere_turn_range"]) == set(
        recovery.OMITTED_COILS
    )


def test_turn_counts_are_read_in_exact_response_order() -> None:
    expected = np.array([48.0, 6.0, 6.0, 7.0, 7.0])
    receipt = {
        "coils": [
            {"coil": name, "signed_turn_sum": turns}
            for name, turns in zip(recovery.OMITTED_COILS, expected, strict=True)
        ]
    }
    np.testing.assert_array_equal(regression.turn_counts(receipt), expected)


def test_representative_selection_uses_six_distinct_shots() -> None:
    records = [
        {"shot": f"shot_{shot}.parquet", "frame": frame}
        for shot in range(8)
        for frame in range(3)
    ]
    selected = regression.representative_indices(records)
    assert len(selected) == 6
    assert len({records[index]["shot"] for index in selected}) == 6


def test_committed_receipt_banks_full_grid_fits_and_contours() -> None:
    path = regression.DEFAULT_OUTPUT / regression.RECEIPT_NAME
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["selection"]["frames"] == 60
    assert receipt["selection"]["shots"] == 20
    assert receipt["selection"]["all_frames_absent_from_polarity_population"]
    assert receipt["regression"]["grid_shape_zr"] == [65, 65]
    assert receipt["regression"]["grid_points_per_frame"] == 4225
    assert receipt["scope"]["calibration_uses_label_flux"]
    assert receipt["scope"]["circuit_promotion_claim"] is False
    assert len(receipt["ensemble_ready_fitted_current_table"]) == 60
    assert len(receipt["records"]) == 60
    assert all(record["grid_points"] == 4225 for record in receipt["records"])
    assert all(
        set(record["fitted_ampere_turns"]) == set(recovery.OMITTED_COILS)
        for record in receipt["records"]
    )
    assert all(
        set(record["equivalent_coil_currents_a"]) == set(recovery.OMITTED_COILS)
        for record in receipt["records"]
    )
    assert all(
        record["regularization"]["design_rank"] == 5
        and record["regularization"]["selected_lambda"] > 0.0
        and len(record["regularization"]["sweep"])
        == len(regression.REGULARISATION_GRID)
        for record in receipt["records"]
    )
    figures = receipt["artifacts"]["line_contour_figures"]
    assert len(figures) >= 6
    assert (
        len(
            {
                record["shot"]
                for record in receipt["records"]
                if record["contour_figure"] is not None
            }
        )
        >= 4
    )
    assert all(Path(figure).is_file() for figure in figures)
