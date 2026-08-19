import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_plasma_subtraction_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_plasma_subtraction_gate", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_preregistration_is_written_before_scoring(tmp_path):
    path = gate.write_preregistration(tmp_path)
    receipt = json.loads(path.read_text())
    assert receipt["coefficients_fitted"] == 0
    assert receipt["score_region"] == "all finite nodes on the shipped 65 by 65 grid"
    assert receipt["registered_pooled_quiescent_r2_bar"] == (
        gate.RECORDED_COIL_ONLY_EXTERIOR_R2
    )


def test_gauge_score_removes_only_additive_constant():
    actual = np.array([[2.0, 3.0], [5.0, 8.0]])
    score, squared_error, total, gauge, residual = gate.gauge_r_squared(
        actual, actual - 7.0
    )
    assert score == 1.0
    assert squared_error == 0.0
    assert total > 0.0
    assert gauge == 7.0
    np.testing.assert_array_equal(residual, 0.0)


def test_population_reports_pooled_and_shot_quartiles():
    records = []
    for shot in ("a", "b"):
        for value in (0.8, 0.9):
            records.append(
                {
                    "shot": shot,
                    "population": "quiescent",
                    "r2": value,
                    "squared_error": 1.0 - value,
                    "total_sum_squares": 1.0,
                    "current_projection_fractional_rms": 0.1,
                    "reliable_surfaces": 10,
                }
            )
    result = gate._population(records, "quiescent")
    assert result["frames"] == 4
    assert result["shots"] == 2
    np.testing.assert_allclose(result["pooled_r2"], 0.85)
    np.testing.assert_allclose(result["shot_r2"]["median"], 0.85)


def test_plasma_mask_follows_supplied_lcfs_polygon():
    axis = np.linspace(-1.0, 1.0, 5)
    row = {
        "efit_lcfs_n": [4],
        "efit_lcfs_r": [[-0.6, 0.6, 0.6, -0.6]],
        "efit_lcfs_z": [[-0.6, -0.6, 0.6, 0.6]],
    }
    mask = gate._plasma_mask(row, 0, axis, axis)
    assert mask[2, 2]
    assert not mask[0, 0]
