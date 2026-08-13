"""Contracts for the real-slice convergence measurement harness."""

import numpy as np
import pytest

from benchmarks.real_slice_convergence import _norm_residual, _select, diagnose
from nova.imas.mast_solve_inputs import CorrectedSolveInputs


def _inputs(count: int = 6) -> CorrectedSolveInputs:
    return CorrectedSolveInputs(
        shot=21978,
        time_s=np.arange(count, dtype=float),
        coil_channels=("coil",),
        coil_currents_a=np.arange(count, dtype=float)[:, None],
        sensor_channels=("probe",),
        sensor_signals=(10.0 + np.arange(count, dtype=float))[:, None],
        sensor_units=("T",),
        plasma_current_a=100.0 + np.arange(count, dtype=float),
        corrections=(),
    )


def test_slice_selection_requires_five_real_rows_and_preserves_alignment():
    selected = _select(_inputs(), (0, 1, 2, 4, 5))

    assert selected.slice_count == 5
    np.testing.assert_array_equal(selected.time_s, [0.0, 1.0, 2.0, 4.0, 5.0])
    np.testing.assert_array_equal(selected.sensor_signals[:, 0], [10, 11, 12, 14, 15])
    with pytest.raises(ValueError, match="at least five"):
        _select(_inputs(), (0, 1, 2, 3))


def test_norm_conversion_keeps_the_fixed_point_denominator():
    measured = _norm_residual(np.array([2.0, -1.0]), np.array([1.0, -0.5]))

    assert measured.relative_sup == pytest.approx(0.5)
    assert measured.relative_rms == pytest.approx(np.sqrt(0.15625))
    assert measured.rms_per_sup == pytest.approx(np.sqrt(0.625))


def test_diagnosis_does_not_compare_the_fixed_point_scalar_to_profile_bound():
    slice_rows = [
        {
            "fixed_point_norm": {"relative_sup": 1.0},
            "whitened_magnetics_rms": 0.9,
        }
        for _ in range(5)
    ]

    def scheme(value, converted, magnetics):
        return [
            {
                "reported_relative_sup": value,
                "post_budget_norm_diagnostic": {"relative_rms": converted},
                "whitened_magnetics_rms": magnetics,
            }
            for _ in range(5)
        ]

    payload = {
        "registered_profile_bound": 0.075868,
        "accelerator": {"evaluation_count": 20},
        "slices": slice_rows,
        "schemes": {
            "picard": scheme(0.8, 0.2, 0.85),
            "anderson": scheme(0.7, 0.15, 0.83),
            "newton_krylov": scheme(0.6, 0.1, 0.8),
        },
        "sensor_scale_counterfactuals": {
            "full_scale_full_seed": scheme(0.6, 0.1, 0.8),
            "full_scale_local_seed": scheme(0.61, 0.1, 0.8),
            "local_scale_full_seed": scheme(1.0, 0.1, 0.8),
            "local_scale_local_seed": scheme(1.01, 0.1, 0.8),
        },
    }

    result = diagnose(payload)

    assert result["dominant_candidate"] == "whitening_or_sensor_scale_mismatch"
    assert result["median_newton_converted_rms"] == pytest.approx(0.1)
    assert result["converted_rms_over_registered_bound"] == pytest.approx(
        0.1 / 0.075868
    )
    assert result["sensor_scale_counterfactual_median_relative_sup"] == {
        "full_scale_full_seed": {
            "median_finite_relative_sup": pytest.approx(0.6),
            "finite_slice_count": 5,
            "nonfinite_slice_count": 0,
        },
        "full_scale_local_seed": {
            "median_finite_relative_sup": pytest.approx(0.61),
            "finite_slice_count": 5,
            "nonfinite_slice_count": 0,
        },
        "local_scale_full_seed": {
            "median_finite_relative_sup": pytest.approx(1.0),
            "finite_slice_count": 5,
            "nonfinite_slice_count": 0,
        },
        "local_scale_local_seed": {
            "median_finite_relative_sup": pytest.approx(1.01),
            "finite_slice_count": 5,
            "nonfinite_slice_count": 0,
        },
    }
    assert (
        result["exclusions"]["insufficient_accelerator_budget"][
            "equal_evaluation_budget"
        ]
        == 20
    )
