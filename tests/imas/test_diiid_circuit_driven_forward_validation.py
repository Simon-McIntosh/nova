from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "benchmarks" / "diiid_circuit_driven_forward_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_circuit_driven_forward_validation", MODULE_PATH
)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)

RECEIPT_PATH = (
    ROOT
    / "docs/figures/coil-circuit-discovery/circuit-driven-forward-validation"
    / study.RECEIPT_NAME
)


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text())


def test_selection_is_strictly_outside_calibration_and_polarity_banks() -> None:
    receipt = _receipt()
    selection = receipt["selection"]
    selected = {(item["shot"], item["frame"]) for item in selection["selected_frames"]}
    calibration_frames, calibration_shots, _bank = study.calibration_population()
    affected = study.polarity_population()

    assert len(selected) >= 5
    assert len({shot for shot, _frame in selected}) >= 3
    assert not selected & calibration_frames
    assert not {shot for shot, _frame in selected} & calibration_shots
    assert not {shot for shot, _frame in selected} & affected
    assert selection["calibration_bank"]["strictly_outside"] is True
    assert selection["calibration_bank"]["frame_count"] == 60
    assert selection["calibration_bank"]["shot_count"] == 20
    assert selection["all_finite_diverted"] is True
    assert selection["all_polarity_screened"] is True


def test_circuit_arm_has_complete_label_free_current_authority() -> None:
    receipt = _receipt()
    audit = receipt["current_path_audit"]

    assert audit["label_derived_current_reads"] == 0
    assert audit["per_frame_current_fits"] == 0
    assert audit["least_squares_updates"] == 0
    assert audit["unknown_current_parameters"] == 0
    assert len(audit["competition_current_channels"]) == 19
    for frame in receipt["frames"]:
        current = frame["circuit_driven"]["current_receipt"]
        assert current["complete_count"] == 24
        assert current["unknown_parameter_count"] == 0
        assert current["all_finite"] is True
        assert len(current["response_order"]) == 24
        assert len(current["conductors"]) == 24
        assert all(np.isfinite(item["value_a_turn"]) for item in current["conductors"])
        assert all(
            "label" not in item["authority"].lower() for item in current["conductors"]
        )
        assert frame["shipped_only"]["current_receipt"]["complete_count"] == 19


def test_receipt_carries_both_solve_arms_metrics_and_overlay() -> None:
    receipt = _receipt()

    assert receipt["aggregate"]["frame_count"] >= 5
    assert receipt["aggregate"]["shot_count"] >= 3
    assert (
        receipt["comparison"]["label_representability_ceiling_fractional_rms"] == 0.0429
    )
    assert (
        receipt["caveats"]["e89_systematic"][
            "E89UP_effective_gain_minus_integer_wiring"
        ]
        == 0.04569475694961733
    )
    for frame in receipt["frames"]:
        for arm in ("circuit_driven", "shipped_only"):
            solve = frame[arm]["solve"]
            assert isinstance(solve["converged"], bool)
            assert solve["fixed_point_relative_residual"] is not None
            assert solve["residual_trajectory"]
            assert solve["metrics"]["fractional_flux_rms"] is not None
            if solve["converged"]:
                assert solve["metrics"]["boundary_mean_separation_m"] is not None
                assert solve["metrics"]["x_point_separation_m"] is not None
    figure = ROOT / receipt["artifacts"]["overlay_figure"]
    assert figure.is_file()
    assert figure.stat().st_size > 10_000
