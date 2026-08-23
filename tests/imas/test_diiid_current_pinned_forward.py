"""Contracts for the plasma-current constrained forward-map measurement."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import diiid_current_pinned_forward as pinned
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_description import CIRCUIT_DRIVEN_CONDUCTORS, POLOIDAL_CONDUCTORS


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
    assert declaration["shared_inputs"]["current_authority"] == (
        "nova.imas.diiid_current fixed wiring"
    )
    assert not declaration["shared_inputs"][
        "label_recovered_current_prescriptions_used"
    ]
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
    names = (*POLOIDAL_CONDUCTORS, *CIRCUIT_DRIVEN_CONDUCTORS)
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


def test_constrained_arm_drives_the_public_target_current_seam() -> None:
    captured = {}
    target = 720_000.0

    class Operator:
        @staticmethod
        def normalised_current_moments(state, target_current, requested_class=None):
            captured["normalised_target"] = target_current
            captured["normalised_class"] = requested_class
            return SimpleNamespace(cell_current=np.asarray([target_current])), 1.2

        @staticmethod
        def read(state):
            return None, SimpleNamespace(
                diverted=True,
                x_point=np.asarray([1.4, -1.1]),
            )

    class Profile:
        operator = Operator()

        @staticmethod
        def flux_map(current, requested_class=None, target_current=None):
            captured["current"] = np.asarray(current)
            captured["target"] = target_current
            captured["requested_class"] = requested_class
            return lambda state: state

    result = pinned.solve_constrained(
        Profile(),
        np.ones(2),
        np.asarray([3.0]),
        target,
    )

    assert captured["target"] == target
    assert captured["normalised_target"] == target
    assert captured["requested_class"] == TopologyClass.DIVERTED
    assert captured["normalised_class"] == TopologyClass.DIVERTED
    assert result["relative_residual"] == 0.0
    assert result["current_relative_error"] == 0.0
    assert result["amplitude"] == pytest.approx(1.2)
    assert result["topology"] == "diverted"


def test_benchmark_has_no_local_amplitude_elimination_copy() -> None:
    source = Path(pinned.__file__).read_text()

    assert "def eliminated_map" not in source
    assert "def _lambda_value" not in source
    assert "def _scaled_moments" not in source
    assert "safe_unscaled" not in source
    assert "target_current=target_current_a" in source
    assert "attempted_amplitude = target / unscaled" not in source
    assert "current_normalisation_amplitude(target, unscaled)" in source


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


def test_committed_receipt_uses_only_circuit_driven_current_prescriptions() -> None:
    path = pinned.DEFAULT_OUTPUT / pinned.CHECKPOINT_NAME
    frame = json.loads(path.read_text().splitlines()[0])
    assert "diagnostic_label_recovered" not in frame
    authority = frame["inference_current_authority"]
    assert authority["uses_circuit"] is True
    assert authority["unknown_parameter_count"] == 0
    assert len(authority["currents_a"]) == 24

    receipt = json.loads((pinned.DEFAULT_OUTPUT / pinned.RECEIPT_NAME).read_text())
    choice = receipt["current_arm_choice"]
    assert choice["uses_pf_active_circuit"] is True
    assert choice["label_recovered_current_prescriptions_used"] is False
    assert receipt["authorities"]["recovery_bank_role"] == (
        "shot and frame selection only"
    )
    low = receipt["result"]["low_current_control_fixture"]
    assert low["current_normalisation_guard_triggered"] is True
    assert "outside" in low["current_normalisation_guard_termination"]


def test_cross_codegen_policy_qualifies_the_excluded_low_current_control() -> None:
    control = {
        "full_24_unpinned": {"relative_residual": 0.034912343846758294},
        "historical_full_24_plateau": 0.03491124178554655,
    }

    qualified = pinned._qualify_cross_codegen_control(
        control,
        banked_absolute_difference=pinned.LOW_CURRENT_BANKED_ABSOLUTE_DIFFERENCE,
    )

    assert qualified["historical_full_24_plateau_absolute_difference"] == pytest.approx(
        1.1020612117447208e-6,
        rel=0.0,
        abs=1.0e-18,
    )
    assert qualified["cross_codegen_absolute_tolerance"] == 2.0e-6
    assert (
        qualified["historical_full_24_plateau_absolute_difference_decimal"]
        == "1.1020612117447208e-6"
    )
    assert qualified["historical_full_24_plateau_reproduced"]
    assert qualified["control_excluded_from_cohort_conclusions"]


def test_post_check_updates_only_the_banked_receipt(tmp_path: Path) -> None:
    control = {
        "full_24_unpinned": {"relative_residual": 0.034912343846758294},
        "historical_full_24_plateau": 0.03491124178554655,
        "historical_full_24_plateau_reproduced": False,
    }
    receipt = {"result": {"low_current_control_fixture": control}}
    path = tmp_path / pinned.RECEIPT_NAME
    path.write_text(json.dumps(receipt))

    updated = pinned.recheck_banked_receipt(tmp_path)

    updated_control = updated["result"]["low_current_control_fixture"]
    assert updated_control["historical_full_24_plateau_reproduced"]
    assert updated_control["control_excluded_from_cohort_conclusions"]
    assert json.loads(path.read_text()) == updated


def test_source_does_not_modify_or_import_an_equilibrium_implementation() -> None:
    source = Path(pinned.__file__).read_text()

    assert "nova/equilibrium" not in source
    assert "prescribed input" in source
    assert "same class as the coil currents" in source
