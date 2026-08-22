"""Tests for full-grid omitted-conductor current regression."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import diiid_boundary_current_recovery as recovery
from benchmarks import diiid_exact_clipped_tare as tare
from benchmarks import diiid_five_column_residual_adjudication as prior
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


def test_read_path_oracle_recovers_declared_positive_currents(tmp_path: Path) -> None:
    generator = np.random.default_rng(817)
    shape = (17, 19)
    plasma = generator.normal(scale=0.02, size=shape)
    shipped_response = generator.normal(scale=1.0e-7, size=(3, *shape))
    shipped_currents = np.array([21_000.0, -13_000.0, 8_000.0])
    omitted_response = generator.normal(scale=1.0e-6, size=(5, *shape))
    counts = np.array([48.0, 6.0, 6.0, 7.0, 7.0])
    oracle = regression.production_read_path_oracle(
        tmp_path / "synthetic_corpus.parquet",
        plasma,
        650_000.0,
        shipped_response,
        shipped_currents,
        omitted_response,
        counts,
    )
    assert oracle["passes"]
    assert oracle["all_recovered_signs_positive"]
    assert oracle["maximum_relative_amplitude_error"] <= 1.0e-6
    assert oracle["reader"] == "benchmarks.diiid_exact_clipped_tare._read"
    assert oracle["convention_entry"] == "corpus_flux_to_nova_total"


def test_shipped_current_sampling_uses_direct_millisecond_coordinate() -> None:
    class Turns:
        applied_multiplier = 1.0

    class Conductor:
        name = "ECOILA"
        input_column = "magnetics_ECOILA"
        turns = Turns()

    class Description:
        conductors = (Conductor(),)

    row = {
        "magnetics_time": np.array([-100.0, 0.0, 100.0]),
        "magnetics_ECOILA": np.array([10.0, 20.0, 50.0]),
    }
    current = regression.shipped_currents_at_frame(
        row, Description(), ("ECOILA",), 50.0
    )
    np.testing.assert_allclose(current, [35_000.0], rtol=0.0, atol=1.0e-12)


def test_real_corpus_frame_times_are_inside_magnetics_trace() -> None:
    cohort, _source = prior.banked_cohort(prior.DEFAULT_DATA, prior.SOURCE_RECEIPT)
    for name in sorted({item.path.name for item in cohort}):
        row = tare._read(prior.DEFAULT_DATA / name, ("efit_times", "magnetics_time"))
        selected = [item for item in cohort if item.path.name == name]
        frame_times = np.asarray(
            [row["efit_times"][item.frame] for item in selected], dtype=float
        )
        np.testing.assert_allclose(
            frame_times,
            np.asarray([item.time_ms for item in selected]),
            rtol=0.0,
            atol=0.0,
        )
        invariant = regression.corpus_time_coordinate_invariant(row, frame_times, name)
        assert invariant["coordinate_unit"] == "ms"
        assert invariant["all_selected_frames_strictly_inside_trace"]
        assert invariant["minimum_strict_interior_margin_ms"] > 0.0


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
    time_invariant = receipt["selection"]["time_coordinate_invariant"]
    assert time_invariant["coordinate_unit"] == "ms"
    assert time_invariant["shots_checked"] == 20
    assert time_invariant["all_selected_frames_strictly_inside_trace"]
    assert "no unit conversion" in time_invariant["shipped_current_sampling"]
    assert receipt["regression"]["grid_shape_zr"] == [65, 65]
    assert receipt["regression"]["grid_points_per_frame"] == 4225
    assert receipt["scope"]["calibration_uses_label_flux"]
    assert receipt["scope"]["circuit_promotion_claim"] is False
    polarity = receipt["regression"]["polarity"]
    assert polarity["cohort_sign_is_measured_not_selected"]
    oracle = polarity["read_path_composition_oracle"]
    assert oracle["passes"]
    assert oracle["maximum_relative_amplitude_error"] <= 1.0e-6
    assert oracle["target_identity"] == (
        "label_total - exact_plasma - shipped_conductors"
    )
    assert polarity["response_orientation_invariant"]["passes"]
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
        record["same_frame_ecoila_current_a"] != 0.0
        and set(record["fitted_to_same_frame_ecoila_ratio"])
        == set(recovery.OMITTED_COILS)
        and set(record["absolute_amplitude_gap_a"]) == set(recovery.OMITTED_COILS)
        for record in receipt["records"]
    )
    expected_negative_frames = {
        "ECOILB": 36,
        "E567UP": 33,
        "E567DN": 33,
        "E89UP": 36,
        "E89DN": 36,
    }
    for name, count in expected_negative_frames.items():
        item = receipt["summary"]["per_coil"][name]
        assert item["equivalent_coil_current_a"]["median"] < 0.0
        assert item["negative_frames"] == count
        assert item["positive_frames"] == 60 - count
    np.testing.assert_allclose(
        receipt["summary"]["same_frame_ecoila_current_a"]["median"],
        -6924.264255400402,
        rtol=1.0e-12,
    )
    expected_ratio_and_gap = {
        "ECOILB": (2.020306759754309, 20631.091563518887),
        "E567UP": (1.0535218639420587, 2932.74096285845),
        "E567DN": (0.9688573779656977, 3560.9752007184993),
        "E89UP": (1.0706482258142718, 2362.025085170862),
        "E89DN": (1.0412952525775354, 2254.9428461228263),
    }
    for name, (ratio, gap) in expected_ratio_and_gap.items():
        item = receipt["summary"]["per_coil"][name]
        np.testing.assert_allclose(
            item["fitted_to_same_frame_ecoila_ratio"]["median"],
            ratio,
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            item["absolute_amplitude_gap_a"]["median"], gap, rtol=1.0e-12
        )
    np.testing.assert_allclose(
        receipt["summary"]["pre_fit_rms_wb"]["median"],
        0.04134593718835078,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        receipt["summary"]["post_fit_rms_wb"]["median"],
        0.006370599271128104,
        rtol=1.0e-12,
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
