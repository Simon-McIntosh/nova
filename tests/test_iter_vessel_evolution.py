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


def test_passive_flux_uses_the_sensor_grid_owner():
    """The flux at points is the owner's grid coupling, not a second point kernel."""
    from nova.circuit.conductor import ConductorSet, SensorSet
    from nova.circuit.linkage import sensor_grid_couplings

    measure = _module("measure")
    conductors = ConductorSet(
        r=np.array([3.0, 6.0, 8.0, 8.5, 9.0]),
        z=np.array([4.0, 4.0, 3.0, 1.0, -1.0]),
        dr=np.array([0.4, 0.4, 0.2, 0.2, 0.2]),
        dz=np.array([0.6, 0.6, 0.2, 0.2, 0.2]),
        current_share=np.ones(5),
        circuit=np.arange(5),
    )
    circuits = np.array([2, 3, 4])
    points = np.array([[9.5, 0.0], [9.0, 3.0]])
    times = np.linspace(0.0, 1.0, 5)
    i_passive = np.outer(np.sin(times), np.array([1e3, 2e3, 3e3]))

    flux = measure.passive_flux_at_points(
        conductors, circuits, i_passive, times, points
    )
    sensors = SensorSet(points[:, 0], points[:, 1], np.zeros(2), np.ones(2, dtype=bool))
    _, coupling = sensor_grid_couplings(
        conductors, circuits, sensors, points[:, 0], points[:, 1]
    )
    assert np.allclose(flux, i_passive @ coupling.T)
    assert flux.shape == (times.size, points.shape[0])


def test_driver_assembly_couples_sections_passive_and_plasma():
    """The driver assembly runs on a reduced case: coupled flux and coil voltage."""
    from nova.circuit.conductor import ConductorSet
    from nova.circuit.linkage import circuit_linkage_matrix

    evolve = _module("evolve")
    n_section, n_passive = 3, 4
    n_total = n_section + n_passive
    conductors = ConductorSet(
        r=np.linspace(3.0, 9.0, n_total),
        z=np.linspace(4.0, -4.0, n_total),
        dr=np.full(n_total, 0.3),
        dz=np.full(n_total, 0.3),
        current_share=np.ones(n_total),
        circuit=np.arange(n_total),
    )
    matrix = circuit_linkage_matrix(conductors, range(n_total), section_n_max=3)
    series = evolve.series_matrix(np.array([0, 0, 1]))
    times = np.linspace(0.0, 2.0, 9)
    i_channel = np.zeros((times.size, series.shape[1]))
    i_channel[:, 0] = 1e4 * times
    i_channel[:, 1] = -1e4 * times
    i_passive = 1e3 * np.exp(-times)[:, None] * np.ones((1, n_passive))
    psi_plasma_section = 0.5 * np.sin(times)[:, None] * np.ones((1, n_section))

    psi_coil, voltage_coil = evolve.psis_and_voltages(
        times, series, matrix, i_channel, i_passive, psi_plasma_section
    )
    assert psi_coil.shape == (times.size, series.shape[1])
    assert voltage_coil.shape == (times.size, series.shape[1])
    assert np.max(np.abs(voltage_coil[:, 0])) > 0

    tau = np.array([0.2, 0.5, 1.0])
    drive = np.gradient(psi_plasma_section, times, axis=0) * tau[None, :]
    state = evolve.integrate(tau, np.eye(n_section), times, drive)
    assert state.shape == (times.size, n_section)
    assert np.abs(state).max() > 0
