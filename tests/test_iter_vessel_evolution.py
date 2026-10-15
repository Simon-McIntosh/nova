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


def test_integration_matches_analytic_lr_decay():
    """The integrated ring current follows the analytic L/R decay.

    Asserts the *propagated* state against the closed-form L/R response, so a
    change to the recurrence -- a scaled interval, a dropped decay factor --
    reddens here.
    """
    from nova.circuit.propagate import integrate_eddy_ode

    tau = 1.0
    linkage, rate, ramp, duration = 0.5, 1.0, 0.1, 1.0
    times = np.linspace(0.0, duration, 4001)
    psi = linkage * np.clip(rate * times, 0.0, rate * ramp)
    numeric, _ = integrate_eddy_ode(np.array([tau]), times, psi[:, None])

    peak = -linkage * rate * tau * (1.0 - np.exp(-ramp / tau))
    analytic = np.where(
        times <= ramp,
        -linkage * rate * tau * (1.0 - np.exp(-times / tau)),
        peak * np.exp(-(times - ramp) / tau),
    )
    error = np.max(np.abs(numeric[:, 0] - analytic)) / abs(peak)
    assert error < 1e-6

    fast, _ = integrate_eddy_ode(np.array([0.25 * tau]), times, psi[:, None])
    assert np.abs(fast[:, 0]).max() < np.abs(numeric[:, 0]).max()


def test_assignment_uses_containment():
    """A circuit is assigned to the loop whose cross-section contains it."""
    from shapely.geometry import box

    measure = _module("measure")
    circuits = np.array([[1.0, 0.0], [9.0, 0.0], [1.0, 40.0]])
    loops = [
        box(-1.0, -1.0, 3.0, 1.0),
        box(7.0, -1.0, 11.0, 1.0),
    ]
    owner, unassigned, ambiguous = measure.assign_by_containment(circuits, loops)
    assert list(owner) == [0, 1, -1]
    assert unassigned == 1
    assert ambiguous == 0


def test_assignment_flags_a_circuit_inside_two_loops():
    """A circuit inside more than one DINA loop is ambiguous, not forced onto one."""
    from shapely.geometry import box

    measure = _module("measure")
    circuits = np.array([[1.0, 0.0]])
    loops = [box(-1.0, -1.0, 3.0, 1.0), box(0.0, -1.0, 5.0, 1.0)]
    owner, _unassigned, ambiguous = measure.assign_by_containment(circuits, loops)
    assert list(owner) == [-2]
    assert ambiguous == 1


def test_group_comparison_flags_an_out_of_tolerance_loop():
    """A loop whose residual exceeds the stated tolerance is not counted within it."""
    from shapely.geometry import box

    measure = _module("measure")
    times = np.linspace(0.0, 1.0, 11)
    i_passive = np.zeros((11, 2))
    i_passive[:, 0] = 100.0
    i_passive[:, 1] = 100.0
    circuit_centroids = np.array([[1.0, 0.0], [9.0, 0.0]])
    loop_polygons = [box(0.0, -1.0, 2.0, 1.0), box(8.0, -1.0, 10.0, 1.0)]
    dina_times = times
    dina_currents = np.zeros((11, 2))
    dina_currents[:, 0] = 100.0
    dina_currents[:, 1] = 1000.0
    result = measure.group_comparison(
        times,
        i_passive,
        circuit_centroids,
        loop_polygons,
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
