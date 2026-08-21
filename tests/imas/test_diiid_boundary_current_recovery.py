from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_boundary_current_recovery as recovery


def test_preregistration_declares_required_scale_and_split() -> None:
    declared = recovery.preregistration()
    assert declared["selection"]["total_frames"] >= 200
    assert declared["selection"]["shots"] >= 20
    assert declared["prediction"]["split_unit"] == "shot"
    assert declared["prediction"]["features"] == list(recovery.SHIPPED_FEATURES)
    assert len(recovery.SHIPPED_FEATURES) == 20
    assert declared["root_existence_comparison"][
        "historical_reported_fractional_rms"
    ] == {
        "without_recovered_currents": 0.2156505171,
        "with_recovered_currents": 0.34392767843,
    }
    assert declared["root_existence_comparison"]["replacement_frames"] >= 5


def test_recover_currents_removes_only_additive_gauge() -> None:
    rng = np.random.default_rng(18)
    design = rng.normal(size=(80, 5))
    expected = np.asarray([3.0, -2.0, 0.5, 7.0, -4.0])
    result = recovery.recover_currents(design, design @ expected + 2.75)
    np.testing.assert_allclose(
        result["currents_a"], expected, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(result["gauge_wb_per_radian"], 2.75)
    assert result["relative_residual"] < 1.0e-12
    assert result["design_rank"] == 5


def test_recovery_condition_number_is_scale_free() -> None:
    rng = np.random.default_rng(22)
    design = rng.normal(size=(90, 5))
    current = rng.normal(size=5)
    first = recovery.recover_currents(design, design @ current)
    scaled = design * np.asarray([1.0e-4, 2.0, 1.0e3, 0.2, 50.0])
    second = recovery.recover_currents(scaled, design @ current)
    np.testing.assert_allclose(
        first["design_condition_number"],
        second["design_condition_number"],
        rtol=1.0e-12,
    )


def test_held_out_linear_split_does_not_mix_shots() -> None:
    rng = np.random.default_rng(9)
    shot_names = ["train-a"] * 8 + ["train-b"] * 8 + ["test"] * 8
    features = rng.normal(size=(24, 20))
    weights = rng.normal(size=(20, 5))
    targets = features @ weights + np.arange(5)
    result = recovery.fit_held_out_linear(
        features, targets, shot_names, {"train-a", "train-b"}
    )
    assert result["train_frames"] == 16
    assert result["held_out_frames"] == 8
    assert result["held_out_shots"] == ["test"]
    assert set(result["train_shots"]).isdisjoint(result["held_out_shots"])
    assert all(
        len(item["coefficients_a_per_released_channel_unit"]) == 20
        for item in result["targets"]
    )


def test_affected_shots_are_rejected_before_row_read(monkeypatch, tmp_path) -> None:
    affected = tmp_path / "affected.parquet"
    affected.touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("affected shot was read")

    monkeypatch.setattr(recovery, "_read", forbidden)
    try:
        recovery.select_cohort([affected], {affected.name})
    except RuntimeError as error:
        assert "insufficient unaffected" in str(error)
    else:
        raise AssertionError("an undersized cohort must fail closed")


def test_screened_root_selection_uses_distinct_unaffected_shots(tmp_path) -> None:
    selected = [
        recovery.SelectedFrame(tmp_path / f"shot-{index}.parquet", index, float(index))
        for index in range(7)
    ]
    frames = recovery.select_screened_root_frames(selected, {"unrelated.parquet"})
    assert len(frames) == 5
    assert len({item["shot"] for item in frames}) == 5
    assert all(item["shot"] != "unrelated.parquet" for item in frames)


def test_historical_screen_drops_only_affected_frame() -> None:
    records = [
        {
            "shot": f"shot-{index}",
            "frame": index,
            "original_fractional_rms": value,
            "recovered_current_fractional_rms": 2.0 * value,
        }
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 100.0))
    ]
    historical = {"frames": records}
    result = recovery.historical_screened_summary(historical, {"shot-4"})
    assert result["frame_count"] == 4
    assert result["dropped_frames"] == [{"shot": "shot-4", "frame": 4}]
    assert result["original_median_fractional_rms"] == 2.5
    assert result["recovered_current_median_fractional_rms"] == 5.0
    assert result["worsening_percent"] == 100.0
    assert result["recovered_currents_worsen_residual"]


def test_committed_receipt_carries_complete_measurement() -> None:
    path = recovery.DEFAULT_OUTPUT / recovery.RECEIPT_NAME
    receipt = json.loads(path.read_text())
    cohort = receipt["recovery_cohort"]
    assert cohort["frame_count"] >= 200
    assert cohort["shot_count"] >= 20
    assert cohort["all_frames_screened_free_of_affected_population"]
    assert len(cohort["frames"]) == cohort["frame_count"]
    assert all("design_condition_number" in frame for frame in cohort["frames"])
    assert all("relative_residual" in frame for frame in cohort["frames"])

    prediction = receipt["held_out_prediction"]
    assert len(prediction["targets"]) == 5
    assert all(
        len(target["coefficients_a_per_released_channel_unit"]) == 20
        for target in prediction["targets"]
    )
    root = receipt["root_existence"]
    assert root["replacement_polarity_screened"]["frame_count"] >= 5
    assert root["replacement_polarity_screened"][
        "all_shots_screened_free_of_affected_population"
    ]
    assert root["historical_polarity_screened_drop"]["frame_count"] == 4
    assert root["historical_polarity_screened_drop"]["affected_shot_count"] == 0
    assert root["historical_reported"] == {
        "without_recovered_currents_fractional_rms": 0.2156505171,
        "with_recovered_currents_fractional_rms": 0.34392767843,
        "worsening_percent": root["historical_reported"]["worsening_percent"],
    }
    assert "worsening_was_carried_by_affected_frame" in root["screening_verdict"]
