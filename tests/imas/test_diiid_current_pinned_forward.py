"""Contracts for the plasma-current constrained forward-map measurement."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import diiid_current_pinned_forward as pinned
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS


def test_declaration_keeps_the_constraint_explicit_and_the_system_square() -> None:
    declaration = pinned.preregistration()

    assert declaration["target_current"]["status"].startswith("declared constraint")
    assert (
        "admissible competition input"
        in declaration["target_current"]["inference_availability"]
    )
    assert declaration["selection"]["absolute_recorded_ip_floor_a"] == 200_000.0
    assert (
        declaration["selection"]["preflight_reproducibility_relative_tolerance"]
        == 2.0e-6
    )
    assert len(declaration["selection"]["cohort_declared_before_solver_scoring"]) == 5
    assert all(
        item["seed_lambda"] > 0.0
        for item in declaration["selection"]["cohort_declared_before_solver_scoring"]
    )
    assert "excluded" in declaration["selection"]["low_current_control_fixture"]["role"]
    assert declaration["shared_inputs"]["coefficients_fitted"] == 0
    assert declaration["shared_inputs"]["currents_adjusted"] == 0
    assert declaration["arms"]["pinned_eliminated"]["unknowns_and_rows"].startswith(
        "N flux unknowns and N flux"
    )
    assert set(declaration["arms"]) == set(pinned.ARM_NAMES)
    assert declaration["dropped_comparison_arm"]["alpha_stability"] == (
        "not applicable"
    )
    assert "does not expose" in declaration["dropped_comparison_arm"]["reason"]


def test_description_driven_arm_uses_the_fixed_circuit_without_free_currents(
    monkeypatch,
) -> None:
    names = (*POLOIDAL_CONDUCTORS, *pinned.OMITTED_COILS)
    values = np.arange(1.0, 25.0)
    captured = {}

    class Resolution:
        def __init__(self):
            self.unknown_indices = np.asarray([], dtype=int)
            self.prescribed_standard_deviation_a = np.zeros(24)
            self.names = names

        @staticmethod
        def current(unknown):
            assert unknown == ()
            return values

    def complete(profile, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            profile=("completed", profile),
            resolution=Resolution(),
            response_receipt={"current_authority": "pf_active circuit"},
        )

    monkeypatch.setattr(
        pinned,
        "dataset_machine_description",
        lambda row, source_row: SimpleNamespace(physical=(row, source_row)),
    )
    monkeypatch.setattr(
        pinned,
        "shipped_current_at",
        lambda row, description, shipped_names, time_ms: {
            name: float(index + 1) for index, name in enumerate(shipped_names)
        },
    )
    monkeypatch.setattr(pinned, "complete_profile_current_adapter", complete)

    profile, shipped, current, receipt = pinned.description_driven_currents(
        {"_source_path": "shot.parquet"}, object(), 2200.0
    )

    assert profile[0] == "completed"
    assert captured["use_circuit"] is True
    assert captured["shipped_names"] == POLOIDAL_CONDUCTORS
    np.testing.assert_array_equal(current, values)
    np.testing.assert_array_equal(shipped[:19], values[:19])
    np.testing.assert_array_equal(shipped[19:], np.zeros(5))
    assert receipt["unknown_parameter_count"] == 0
    assert receipt["response_order"] == list(names)
    assert receipt["response"]["current_authority"] == "pf_active circuit"


def test_lambda_elimination_enforces_current_exactly() -> None:
    target = 720_000.0
    unscaled = 600_000.0
    amplitude = pinned._lambda_value(target, unscaled)

    assert amplitude == pytest.approx(1.2)
    assert amplitude * unscaled == pytest.approx(target)


def test_lambda_guard_fails_loudly_without_clipping() -> None:
    with pytest.raises(pinned.LambdaOutOfBand, match="outside"):
        pinned._lambda_value(1.0, 1.0e-9)


@pytest.mark.parametrize("unscaled", [0.0, -1.0, np.nan])
def test_lambda_guard_checks_admissibility_before_division(unscaled: float) -> None:
    with pytest.raises(pinned.LambdaOutOfBand, match="outside"):
        pinned._lambda_value(1.0, unscaled)


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


def test_power_iteration_receipt_is_strict_json_serializable() -> None:
    result = pinned.power_iteration(lambda state: 0.5 * state, np.ones(8))

    assert result["finite"] is True
    assert 0.4 < result["absolute_dominant_eigenvalue_estimate"] < 0.6
    json.dumps(result, allow_nan=False)


def test_unconverged_nonfinite_values_are_explicitly_not_computed() -> None:
    serial = pinned._serialise_arm(
        {
            "state": np.ones(2),
            "mapped": lambda state: state,
            "relative_residual": float("inf"),
            "current_relative_error": float("inf"),
            "current_constraint_required": True,
            "amplitude": float("inf"),
            "topology": "limited",
            "x_point_rz_m": np.asarray([np.nan, np.nan]),
            "lambda_guard_triggered": True,
        }
    )

    assert serial["relative_residual"] is None
    assert serial["current_relative_error"] is None
    assert serial["amplitude"] is None
    assert serial["qualified_equilibrium_metrics"]["status"] == "not-computed"
    assert set(serial["not_computed_fields"]) == {
        "amplitude",
        "current_relative_error",
        "relative_residual",
        "x_point_rz_m",
    }
    assert "zero is never used" in serial["sentinel_policy"]
    json.dumps(serial, allow_nan=False)


def test_summary_requires_representative_current_and_diverted_convergence() -> None:
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
                "absolute_target_plasma_current_a": 500_000.0,
                "target_and_unscaled_source_same_sign": True,
                "unconstrained_current_controls": {
                    "shipped_20": {"relative_residual": 0.12},
                    "full_24": {"relative_residual": 0.03},
                    "shipped_to_full_residual_ratio": 4.0,
                },
                "arms": {
                    "unpinned": arm(pinned.UNPINNED_PLATEAU_CONTROL, 1.0, False),
                    "pinned_eliminated": arm(1.0e-8, 1.2, True),
                },
            }
        )
    low_control = {"historical_full_24_plateau_reproduced": True}
    summary = pinned.summarize(records, low_control)

    assert summary["all_frames_absolute_recorded_ip_at_least_200ka"]
    assert summary["all_frames_target_and_source_same_sign"]
    assert summary["representative_current_unpinned_comparison"][
        "historical_3_53x_holds_at_representative_current"
    ]
    assert summary["current_pinning_removes_vacuum_root_and_orbiting_plateau"]


def test_label_recovered_frame_102_remains_an_explicit_diagnostic_control() -> None:
    path = pinned.DEFAULT_OUTPUT / pinned.CHECKPOINT_NAME
    frame = json.loads(path.read_text().splitlines()[0])
    diagnostic = {
        "shot": frame["shot"],
        "frame": frame["frame"],
        "arms": frame["diagnostic_label_recovered"]["arms"],
    }
    eliminated = diagnostic["arms"]["pinned_eliminated"]

    assert frame["diagnostic_label_recovered"]["uses_reconstruction_label"]
    assert frame["diagnostic_label_recovered"]["frame_102_landed_control_reproduced"]
    assert pinned.diagnostic_frame_102_reproduced(diagnostic)
    assert eliminated["relative_residual"] == pytest.approx(
        1.5899788903681545e-9,
        rel=0.0,
        abs=pinned.DIAGNOSTIC_RESIDUAL_ABSOLUTE_TOLERANCE,
    )
    assert eliminated["relative_residual"] <= pinned.RELATIVE_RESIDUAL_CRITERION
    assert eliminated["iterations"] == 4
    assert eliminated["topology"] == "diverted"


def test_source_does_not_modify_or_import_an_equilibrium_implementation() -> None:
    source = Path(pinned.__file__).read_text()

    assert "nova/equilibrium" not in source
    assert "prescribed input" in source
    assert "same class as the coil currents" in source
