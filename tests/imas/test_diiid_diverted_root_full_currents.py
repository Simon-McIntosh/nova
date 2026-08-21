from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_diverted_root_full_currents as roots


def _frame(*, label_rms: float = 0.5, full_pass: bool = True) -> dict:
    diagnostic = {
        "fractional_rms": label_rms,
        "representability_ceiling": roots.LABEL_REPRESENTABILITY_CEILING,
        "used_as_pass_criterion": False,
    }
    full = roots.qualify_arm(
        residual=1.0e-8 if full_pass else 1.0e-4,
        finite=True,
        diverted=True,
        iterations=32,
        x_point=np.asarray([1.5, -1.0]),
        diagnostic=diagnostic,
        trace=np.asarray([1.0e-2, 1.0e-8]),
    )
    control = roots.qualify_arm(
        residual=1.0e-8,
        finite=True,
        diverted=False,
        iterations=32,
        x_point=np.asarray([np.nan, np.nan]),
        diagnostic=diagnostic,
        trace=np.asarray([1.0e-2, 1.0e-8]),
    )
    return {
        "shot": "shot.parquet",
        "frame": 2,
        "screened_out_of_affected_polarity_population": True,
        "arms": {
            roots.CURRENT_ARM_NAMES[0]: control,
            roots.CURRENT_ARM_NAMES[1]: full,
        },
    }


def test_declaration_separates_root_gate_from_label_diagnostic() -> None:
    declared = roots.preregistration()
    assert declared["solver"]["route"] == "newton_krylov"
    assert declared["solver"]["relative_residual_criterion"] == 1.0e-6
    assert declared["current_arms"]["poloidal_conductor_count"] == 24
    assert declared["current_arms"]["coefficients_fitted"] == 0
    assert declared["label_distance"]["representability_ceiling"] == 0.0429
    assert "never included" in declared["label_distance"]["role"]


def test_landed_inputs_are_distinct_screened_and_keep_exact_currents() -> None:
    frames = []
    for index in range(5):
        frames.append(
            {
                "shot": f"shot-{index}",
                "frame": index,
                "recovered_currents_a": {
                    name: 1000.0 * index + coil
                    for coil, name in enumerate(roots.OMITTED_COILS)
                },
            }
        )
    receipt = {
        "root_existence": {
            "replacement_polarity_screened": {
                "frame_count": 5,
                "all_shots_screened_free_of_affected_population": True,
                "frames": frames,
            }
        }
    }
    selected = roots.selected_inputs(receipt, set())
    assert len(selected) == 5
    assert selected[3].recovered_currents_a == tuple(3000.0 + i for i in range(5))


def test_control_can_converge_numerically_without_passing_topology() -> None:
    arm = _frame()["arms"][roots.CURRENT_ARM_NAMES[0]]
    assert arm["fixed_point"]["converged"]
    assert arm["topology"]["class"] == "limited"
    assert arm["simultaneously_converged_and_diverted"] is False


def test_label_distance_above_ceiling_does_not_fail_a_diverted_root() -> None:
    result = roots.summarize([_frame(label_rms=0.9) for _ in range(5)])
    assert min(result["full_current_label_fractional_rms"]) > 0.0429
    assert result["label_distance_is_diagnostic_only"]
    assert result["full_current_converged_diverted_frames"] == 5
    assert result["passed"]


def test_current_arms_append_recovered_values_without_adjustment() -> None:
    class Operator:
        external_current = np.arange(24.0)

    class Profile:
        operator = Operator()

    recovered = (41.0, 42.0, 43.0, 44.0, 45.0)
    arms = roots.current_arms(Profile(), recovered)
    np.testing.assert_array_equal(arms[0, -5:], 0.0)
    np.testing.assert_array_equal(arms[1, -5:], recovered)
    np.testing.assert_array_equal(arms[0, :-5], arms[1, :-5])


def test_committed_receipt_carries_five_paired_diverted_roots() -> None:
    receipt = json.loads((roots.DEFAULT_OUTPUT / roots.RECEIPT_NAME).read_text())
    result = receipt["result"]
    assert result["frame_count"] >= 5
    assert result["all_shots_screened_free_of_affected_population"]
    assert result["full_current_converged_diverted_frames"] == result["frame_count"]
    assert result["passed"]
    for frame in result["frames"]:
        assert frame["coefficients_fitted"] == 0
        assert frame["current_adjustments"] == 0
        assert len(frame["recovered_currents_a"]) == 5
        for name in roots.CURRENT_ARM_NAMES:
            arm = frame["arms"][name]
            assert "relative_residual" in arm["fixed_point"]
            assert "iterations" in arm["fixed_point"]
            assert "class" in arm["topology"]
            assert "x_point_rz_m" in arm["topology"]
            assert arm["label_map_diagnostic"]["used_as_pass_criterion"] is False
