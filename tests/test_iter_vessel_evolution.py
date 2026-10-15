"""Controls for the ITER vessel-current evolution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "iter_vessel_evolution"))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "iter_vessel_evolution" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_positive_control_recovers_analytic_decay():
    """A single ring under a coil-current ramp matches the analytic L/R response."""
    measure = _module("measure")
    result = measure.positive_control(inductance=1.0, resistance=1.0)
    assert result["resolved"]
    assert result["relative_error"] < 1e-6


def test_resistance_sets_decay_time():
    """The analytic peak scales inversely with the ring resistance."""
    measure = _module("measure")
    slow = measure.positive_control(inductance=2.0, resistance=1.0)
    fast = measure.positive_control(inductance=2.0, resistance=4.0)
    assert abs(fast["tau_s"] - 0.25 * slow["tau_s"]) < 1e-12
    assert abs(fast["peak_current_a"]) < abs(slow["peak_current_a"])


def test_assignment_uses_nearest_loop():
    """A circuit is assigned to the nearest loop centroid."""
    measure = _module("measure")
    circuits = np.array([[1.0, 0.0], [9.0, 0.0], [5.1, 0.0]])
    loops = np.array([[1.0, 0.0], [9.0, 0.0], [5.0, 0.0]])
    owner = measure.assign_to_loops(circuits, loops)
    assert list(owner) == [0, 1, 2]


def test_group_comparison_flags_an_out_of_tolerance_loop():
    """A loop whose residual exceeds the stated tolerance is not counted within it."""
    measure = _module("measure")
    times = np.linspace(0.0, 1.0, 11)
    i_passive = np.zeros((11, 2))
    i_passive[:, 0] = 100.0
    i_passive[:, 1] = 100.0
    circuit_centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    loop_centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    dina_times = times
    dina_currents = np.zeros((11, 2))
    dina_currents[:, 0] = 100.0
    dina_currents[:, 1] = 1000.0
    result = measure.group_comparison(
        times,
        i_passive,
        circuit_centroids,
        loop_centroids,
        ["good", "bad"],
        dina_times,
        dina_currents,
        absolute_a=20.0,
        relative=0.0,
    )
    by_name = {row["name"]: row for row in result["rows"]}
    assert by_name["good"]["within_tolerance"]
    assert not by_name["bad"]["within_tolerance"]
    assert result["loops_within_tolerance"] == 1
