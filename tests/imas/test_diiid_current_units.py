from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_current_units.py"
SPEC = importlib.util.spec_from_file_location("diiid_current_units", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_comparison_declaration_precedes_results_and_is_quantitative():
    declaration = MODULE.comparison_declaration()
    assert declaration["declared_before_comparison"] is True
    assert declaration["cohort_size"] == 20
    assert declaration["sample_count_difference_required"] == 0
    assert declaration["time_max_abs_difference_tolerance_ms"] == 0.01
    assert declaration["trace_absolute_tolerance_a_turn"] == 1.0e-3


def test_shot_match_uses_ampere_turns_and_every_common_channel():
    result = MODULE.compare_shot_arrays(
        netcdf_time_ms=np.array([0.0, 1.0]),
        netcdf_currents_a={
            "ECOILA": np.array([10.0, 20.0]),
            "F1A": np.array([1.0, 2.0]),
        },
        netcdf_turns={"ECOILA": 10.0, "F1A": 50.0},
        competition_time_ms=np.array([0.0, 1.005]),
        competition_currents_ka_turn={
            "ECOILA": np.array([0.1, 0.2]),
            "F1A": np.array([0.05, 0.1]),
        },
    )
    assert result["matched"] is True
    assert np.isclose(result["max_abs_time_difference_ms"], 0.005)
    assert result["common_channel_count"] == 2
    assert all(
        channel["netcdf_to_competition_rms_ratio"] == 1.0
        for channel in result["channels"]
    )

    changed = MODULE.compare_shot_arrays(
        netcdf_time_ms=np.array([0.0, 1.0]),
        netcdf_currents_a={"F1A": np.array([1.0, 2.0])},
        netcdf_turns={"F1A": 50.0},
        competition_time_ms=np.array([0.0]),
        competition_currents_ka_turn={"F1A": np.array([0.05])},
    )
    assert changed["sample_count_difference"] == -1
    assert changed["matched"] is False


def test_family_interpretation_retains_physical_qualification():
    result = MODULE.interpret_family_contrast(
        ohmic_max_current_a=55_500.0,
        f_max_current_a=93.0,
        ohmic_max_ampere_turn=2_600_000.0,
        f_max_ampere_turn=5_100.0,
        equilibrium_slice_count=340,
        x_point_counts=[2] * 340,
    )
    assert result["supported_reading"] == "unit_inconsistency_between_families"
    assert result["confidence"] == "medium"
    assert result["numeric_evidence"]["x_point_count_histogram"] == {"2": 340}
    assert "not an independent current calibration" in result["reason"]


def test_wiring_dependency_consumes_summary_without_pairwise_matrix(tmp_path):
    path = tmp_path / "wiring.json"
    path.write_text(
        json.dumps(
            {
                "pair_count": 276,
                "classification_counts": {"independent": 276},
                "competition_19_conductor_table": {
                    "shipped": ["F1A", "ECOILA"],
                    "netcdf_coils_absent_from_competition_table": ["ECOILB"],
                },
                "pairwise_current_relationships": [
                    {"coil_a": "ECOILA", "coil_b": "ECOILB"}
                ],
            }
        )
    )
    result = MODULE.consume_wiring_result(path)
    assert result["status"] == "consumed"
    assert result["pair_count"] == 276
    assert "pairwise_current_relationships" not in result


def test_amplitude_figure_is_written_on_log_axes(tmp_path):
    path = tmp_path / "amplitudes.png"
    MODULE.plot_amplitudes(
        {
            "per_coil": [
                {
                    "name": "ECOILA",
                    "family": "ohmic",
                    "max_abs_current_a": 55_000.0,
                    "max_abs_ampere_turn": 2_640_000.0,
                },
                {
                    "name": "F1A",
                    "family": "F-coil",
                    "max_abs_current_a": 50.0,
                    "max_abs_ampere_turn": 2_900.0,
                },
            ]
        },
        path,
    )
    assert path.is_file()
    assert path.stat().st_size > 1_000
