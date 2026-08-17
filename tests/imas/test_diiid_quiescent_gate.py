import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_vacuum_quiescent_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_vacuum_quiescent_gate", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_smoothed_native_derivative_uses_declared_window():
    time = np.arange(0.0, 101.0)
    currents = np.column_stack([2.0 * time, -0.5 * time])
    derivative = gate.smoothed_native_derivative(time, currents, np.array([50.0]))
    np.testing.assert_allclose(derivative, [[2000.0, -500.0]])


def test_population_selection_requires_all_sensitivity_bands(tmp_path):
    metric = np.array(
        [10, 20, 30, 60, 70, 80, 120, 130, 140, 300, 310, 320, 330, 340, 350]
    )
    shot = gate.ShotCensus(
        tmp_path / "shot.parquet", np.arange(metric.size), metric, np.ones(metric.size)
    )
    selected, frames = gate.select_populations([shot], shot_count=1)
    assert selected == [shot]
    assert frames[shot.path]["quiescent"].tolist() == [0, 1, 2, 3, 4, 5]
    assert len(frames[shot.path]["transient"]) == 6


def test_normalised_flux_uses_axis_and_lcfs_values():
    radius = np.linspace(1.0, 2.0, 5)
    height = np.linspace(-0.5, 0.5, 5)
    rr, zz = np.meshgrid(radius, height)
    flux = (rr - 1.5) ** 2 + zz**2
    row = {
        "efit_grid_R": radius,
        "efit_grid_Z": height,
        "efit_psirz": [flux],
        "efit_r_axis": [1.5],
        "efit_z_axis": [0.0],
        "efit_lcfs_n": [4],
        "efit_lcfs_r": [[1.25, 1.5, 1.75, 1.5]],
        "efit_lcfs_z": [[0.0, 0.25, 0.0, -0.25]],
    }
    normalised = gate.normalised_flux(row, 0)
    np.testing.assert_allclose(normalised[2, 2], 0.0)
    np.testing.assert_allclose(normalised[2, 1], 1.0)


def test_r2_removes_only_additive_gauge():
    actual = np.array([2.0, 3.0, 5.0, 8.0])
    score, squared_error, total = gate._r2(actual, actual - 12.0)
    assert score == 1.0
    assert squared_error == 0.0
    assert total > 0.0


def test_low_plasma_current_uses_kiloampere_channel_units():
    current_ka = np.array([-50.0, -49.9, 0.0, 49.9, 50.0])
    assert gate.low_plasma_current_mask(current_ka).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_label_map_current_converts_per_radian_flux(monkeypatch):
    captured = {}

    def apply(radius, height, total_flux):
        captured["radius"] = radius
        captured["height"] = height
        captured["total_flux"] = total_flux
        return "receipt"

    monkeypatch.setattr(gate, "apply_delta_star", apply)
    radius = np.array([1.0, 1.5, 2.0])
    height = np.array([-1.0, 0.0, 1.0])
    flux_per_radian = np.arange(9.0).reshape(3, 3)
    assert gate.label_map_current(radius, height, flux_per_radian) == "receipt"
    np.testing.assert_allclose(captured["total_flux"], 2.0 * np.pi * flux_per_radian.T)
