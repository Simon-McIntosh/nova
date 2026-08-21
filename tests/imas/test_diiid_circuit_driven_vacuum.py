"""Contracts for the fixed ohmic-circuit vacuum-field comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import diiid_circuit_driven_vacuum as circuit
from benchmarks.diiid_boundary_current_recovery import OMITTED_COILS


def test_circuit_model_uses_fixed_scales_from_shipped_ecoila() -> None:
    currents = circuit.circuit_currents(10_000.0)

    assert currents["ECOILB"] == pytest.approx(10_172.0)
    assert currents["E567UP"] == pytest.approx(9_929.0)
    assert currents["E567DN"] == pytest.approx(9_823.0)
    assert currents["E89UP"] == pytest.approx(9_806.0)
    assert currents["E89DN"] == pytest.approx(10_165.0)
    assert set(currents) == set(OMITTED_COILS)


def test_current_models_have_zero_label_and_unity_controls() -> None:
    recovered = (66_000.0, 40_000.0, 36_000.0, 30_000.0, 33_000.0)
    models = circuit.current_models(50_000.0, recovered)

    assert set(models) == {*circuit.MODEL_NAMES, circuit.SENSITIVITY_NAME}
    assert not any(models["shipped_20_only"].values())
    assert tuple(models["label_recovered"][name] for name in OMITTED_COILS) == recovered
    assert set(models[circuit.SENSITIVITY_NAME].values()) == {50_000.0}


def test_spread_reproduces_the_landed_label_exemplar() -> None:
    spread = circuit.current_spread(circuit.LANDED_LABEL_CURRENT_EXEMPLAR_A)

    assert spread["maximum_to_minimum_absolute_ratio"] == pytest.approx(
        2.196, rel=5.0e-4
    )


def test_summary_counts_diverted_topology_and_x_point_motion() -> None:
    def measured(name: str, moved: bool) -> dict:
        solve = {
            "relative_residual": 1.0e-8,
            "terminal_topology": "diverted",
            "x_point_separation_from_label_m": 0.2 if moved else 0.6,
        }
        if name == "circuit_derived":
            solve["moves_x_point_toward_divertor_leg"] = moved
        return {
            "vacuum_field_first": {
                "gauge_free_fractional_rms": 0.1,
                "additive_gauge_wb_per_radian": 0.01,
            },
            "current_pinned_solve_second": solve,
        }

    records = []
    for index in range(5):
        models = {
            name: measured(name, name == "circuit_derived")
            for name in circuit.MODEL_NAMES
        }
        records.append(
            {
                "shot": circuit.NAMED_SHOT if index == 0 else f"shot-{index}",
                "frame": circuit.NAMED_FRAME if index == 0 else index,
                "labelled_diverted": True,
                "absent_from_603_shot_polarity_population": True,
                "models": models,
                "sensitivity": {
                    circuit.SENSITIVITY_NAME: measured(circuit.SENSITIVITY_NAME, False)
                },
            }
        )

    result = circuit.summarize(records)

    assert result["frame_count"] == 5
    assert result["distinct_shot_count"] == 5
    assert result["named_frame_present"]
    assert result["circuit_derived_moves_x_point_toward_divertor_leg_count"] == 5
    assert result["models"]["circuit_derived"]["diverted_terminal_count"] == 5


def test_generated_receipt_keeps_vacuum_first_and_zero_fits() -> None:
    receipt = json.loads((circuit.DEFAULT_OUTPUT / circuit.RECEIPT_NAME).read_text())

    assert receipt["measurement_order"][0].startswith("vacuum field")
    assert receipt["current_models"]["coefficients_fitted"] == 0
    assert receipt["current_models"]["circuit_derived"]["inference_admissible"]
    assert not receipt["current_models"]["label_recovered"]["inference_admissible"]
    assert receipt["selection"]["frame_count"] >= 5
    assert receipt["selection"]["distinct_shot_count"] >= 5
    assert receipt["selection"]["named_frame_present"]
    assert receipt["selection"]["all_labelled_diverted"]
    assert receipt["selection"]["all_absent_from_polarity_population"]
    assert len(receipt["frames"]) >= 5
    for frame in receipt["frames"]:
        assert frame["coefficients_fitted"] == 0
        assert set(frame["models"]) == set(circuit.MODEL_NAMES)
        assert set(frame["sensitivity"]) == {circuit.SENSITIVITY_NAME}
        for model in (*circuit.MODEL_NAMES, circuit.SENSITIVITY_NAME):
            arm = (
                frame["models"][model]
                if model in circuit.MODEL_NAMES
                else frame["sensitivity"][model]
            )
            assert "gauge_free_fractional_rms" in arm["vacuum_field_first"]
            assert "additive_gauge_wb_per_radian" in arm["vacuum_field_first"]
            solve = arm["current_pinned_solve_second"]
            assert "relative_residual" in solve
            assert "iterations" in solve
            assert "terminal_topology" in solve
            assert "x_point_rz_m" in solve
            assert "x_point_separation_from_label_m" in solve
    assert (circuit.DEFAULT_OUTPUT / circuit.FIGURE_NAME).is_file()


def test_source_treats_equilibrium_as_read_only() -> None:
    source = Path(circuit.__file__).read_text()

    assert "nova/equilibrium" not in source
    assert "coefficients_fitted" in source
    assert "SHIPPED" in source
