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
    assert (
        declared["root_existence_comparison"]["landed_free_boundary_fractional_rms"]
        == 0.2156505171
    )


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
    assert receipt["root_existence"]["frame_count"] == 5
    assert (
        receipt["root_existence"]["landed_baseline_fractional_rms"]
        == recovery.LANDED_FREE_BOUNDARY_FRACTIONAL_RMS
    )
