from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_coil_wiring.py"
SPEC = importlib.util.spec_from_file_location("diiid_coil_wiring", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOLERANCES = {
    "current_absolute_tolerance_A": 1e-6,
    "current_relative_tolerance": 1e-12,
    "proportional_residual_relative_tolerance": 1e-10,
    "proportional_correlation_floor": 0.9999999999,
}


def test_pairwise_classifier_distinguishes_wiring_relationships():
    current = np.array([-3.0, -1.0, 2.0, 5.0])
    same = MODULE.compare_pair(current, current, TOLERANCES)
    opposite = MODULE.compare_pair(current, -current, TOLERANCES)
    assert same["classification"] == "identical"
    assert opposite["classification"] == "exactly_negated"

    scaled = MODULE.compare_pair(current, 2.5 * current, TOLERANCES)
    assert scaled["classification"] == "proportional"
    assert scaled["ratio_second_to_first"] == 2.5

    unrelated = MODULE.compare_pair(
        current, np.array([1.0, -4.0, 3.0, -2.0]), TOLERANCES
    )
    assert unrelated["classification"] == "independent"


def test_correlation_is_scale_free_and_handles_constant_traces():
    current = np.array([-2.0, 0.0, 1.0, 7.0])
    assert np.isclose(MODULE.scale_free_correlation(current, 11.0 * current), 1.0)
    assert np.isclose(MODULE.scale_free_correlation(current, -3.0 * current), -1.0)
    assert MODULE.scale_free_correlation(np.ones(4), current) is None


def test_classification_codes_pin_receipt_and_figure_meanings():
    assert MODULE.CLASS_CODES == {
        "identical": 0,
        "exactly_negated": 1,
        "proportional": 2,
        "independent": 3,
    }


def test_competition_table_has_nineteen_poloidal_conductors():
    assert len(MODULE.POLOIDAL_CONDUCTORS) == 19
    assert "ECOILA" in MODULE.POLOIDAL_CONDUCTORS
    assert "ECOILB" not in MODULE.POLOIDAL_CONDUCTORS
