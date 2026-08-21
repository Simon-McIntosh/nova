"""Contracts for the plasma-current constrained forward-map measurement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import diiid_current_pinned_forward as pinned
from nova.equilibrium.stencil_mesh import CellCurrentMoments


def test_declaration_keeps_the_constraint_explicit_and_the_system_square() -> None:
    declaration = pinned.preregistration()

    assert declaration["target_current"]["status"].startswith("declared constraint")
    assert "not available" in declaration["target_current"]["inference_availability"]
    assert declaration["shared_inputs"]["coefficients_fitted"] == 0
    assert declaration["shared_inputs"]["currents_adjusted"] == 0
    assert declaration["arms"]["pinned_eliminated"]["unknowns_and_rows"].startswith(
        "N flux unknowns and N flux"
    )
    assert declaration["arms"]["pinned_augmented"]["alphas"] == [0.1, 1.0, 10.0]


def test_lambda_elimination_enforces_current_exactly() -> None:
    target = 720_000.0
    unscaled = 600_000.0
    amplitude = pinned._lambda_value(target, unscaled)

    assert amplitude == pytest.approx(1.2)
    assert amplitude * unscaled == pytest.approx(target)


def test_lambda_guard_fails_loudly_without_clipping() -> None:
    with pytest.raises(pinned.LambdaOutOfBand, match="outside"):
        pinned._lambda_value(1.0, 1.0e-9)


def test_common_amplitude_scales_every_current_moment() -> None:
    moments = CellCurrentMoments(
        np.asarray([1.0, 2.0]),
        np.asarray([3.0, 4.0]),
        np.asarray([5.0, 6.0]),
    )
    scaled = pinned._scaled_moments(moments, 2.5)

    np.testing.assert_allclose(scaled.cell_current, [2.5, 5.0])
    np.testing.assert_allclose(scaled.radial_moment, [7.5, 10.0])
    np.testing.assert_allclose(scaled.vertical_moment, [12.5, 15.0])


def test_summary_requires_control_reproduction_before_positive_verdict() -> None:
    eigenvalue = {
        "absolute_dominant_eigenvalue_estimate": 1.1,
        "finite": True,
    }

    def arm(residual: float, amplitude: float, passed: bool) -> dict:
        return {
            "relative_residual": residual,
            "amplitude": amplitude,
            "dominant_map_eigenvalue": eigenvalue,
            "simultaneously_meets_1e-6_and_diverted": passed,
            "lambda_guard_triggered": False,
        }

    records = []
    for index in range(5):
        records.append(
            {
                "shot": f"shot-{index}",
                "screened_out_of_affected_polarity_population": True,
                "augmented_verdict_stable_across_alpha": True,
                "arms": {
                    "unpinned": arm(pinned.UNPINNED_PLATEAU_CONTROL, 1.0, False),
                    "pinned_eliminated": arm(1.0e-8, 1.2, True),
                    "pinned_augmented": arm(1.0e-8, 1.2, True),
                },
            }
        )
    summary = pinned.summarize(records)

    assert summary["unpinned_control_reproduced"]
    assert summary["current_pinning_removes_vacuum_root_and_orbiting_plateau"]


def test_generated_receipt_carries_five_screened_three_arm_frames() -> None:
    path = pinned.DEFAULT_OUTPUT / pinned.RECEIPT_NAME
    receipt = json.loads(path.read_text())
    result = receipt["result"]

    assert result["frame_count"] >= 5
    assert result["distinct_shots"] >= 5
    assert result["all_shots_screened_free_of_affected_population"]
    assert result["unpinned_control_reproduced"]
    assert len(result["frames"]) >= 5
    for frame in result["frames"]:
        assert frame["same_label_branch_seed_all_arms"]
        assert frame["poloidal_conductor_count"] == 24
        assert set(frame["arms"]) == set(pinned.ARM_NAMES)
        assert set(frame["augmented_alpha_trials"]) == {"0.1", "1.0", "10.0"}
        for arm in pinned.ARM_NAMES:
            measured = frame["arms"][arm]
            assert "relative_residual" in measured
            assert "iterations" in measured
            assert "topology" in measured
            assert "amplitude" in measured
            assert "dominant_map_eigenvalue" in measured
    assert (pinned.DEFAULT_OUTPUT / pinned.FIGURE_NAME).is_file()


def test_source_does_not_modify_or_import_an_equilibrium_implementation() -> None:
    source = Path(pinned.__file__).read_text()

    assert "nova/equilibrium" not in source
    assert "prescribed benchmark constraint" in source
