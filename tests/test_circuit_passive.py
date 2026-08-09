"""Passive L/R circuit system, its eigen-reduction and its propagation.

The load-bearing contracts:

* the mode history is the EXACT zero-order-hold solution of
  ``da/dt + a/tau = -dPsi/dt`` (pinned against dense sub-step integration), and
  the three evaluations of that recurrence -- arbitrary cadence, uniform-cadence
  filter, JAX scan -- agree to round-off;
* a constant drive produces no eddy state, and a state left alone decays purely
  exponentially at its own time constant;
* a uniform resistance scale maps every ``tau -> tau/scale`` with the
  eigenvectors invariant;
* raw-cadence integration reproduces the slice-cadence result exactly when the
  raw drive IS piecewise-linear between slices, and inherits the pre-slice
  history when it is not;
* a synthetic shell vessel yields physical decay times, L-orthonormal
  eigenvectors, and shape-consistent drive couplings;
* held-back measured channels never become drive columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.circuit.conductor import ConductorSet, SensorSet
from nova.circuit.passive import (
    PassiveCircuit,
    PassiveEigenbasis,
    build_passive_circuit_system,
    eddy_history,
    load_circuit_system,
    load_eigenbasis,
    mode_relevance,
    predict_circuit_currents,
    raw_eddy_trajectory,
    reduce_passive_system,
    save_circuit_system,
    save_eigenbasis,
)
from nova.circuit.propagate import (
    integrate_eddy_ode,
    zoh_mode_response,
)
from nova.jax.config import Precision, configure_dtypes
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.circuit.propagate import scan_eddy_modes

RNG = np.random.default_rng(11)


def _toy_basis(n_modes=3, n_channels=4, n_cells=20) -> PassiveEigenbasis:
    """A synthetic eigenbasis: only the mode maps matter to the propagation."""
    return PassiveEigenbasis(
        tau=np.array([0.030, 0.012, 0.004])[:n_modes],
        v=RNG.normal(size=(7, n_modes)),
        a_sensor=RNG.normal(size=(9, n_modes)),
        g_grid=RNG.normal(size=(48, n_modes)),
        m_channel=RNG.normal(size=(n_modes, n_channels)),
        m_cell=RNG.normal(size=(n_modes, n_cells)) * 1e-3,
        resistivity=7.2e-7,
    )


def _sequences(basis, n_t=16, seed=0):
    """Random drive and plasma-cell current histories on an uneven cadence."""
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.008, 0.03, size=n_t))
    i_channel = np.cumsum(
        rng.normal(0, 50.0, size=(n_t, basis.m_channel.shape[1])), axis=0
    )
    i_cell = np.abs(rng.normal(0, 100.0, size=(n_t, basis.m_cell.shape[1])))
    return times, i_channel, i_cell


def _shell_vessel() -> tuple[ConductorSet, SensorSet, np.ndarray, np.ndarray]:
    """A synthetic axisymmetric vessel: shell segments plus two driven coils.

    Circuits 0-11 subdivide a closed rectangular shell (centre column, outer
    cylinder, two end plates) into 3 mm-walled segments; circuits 100 and 101 are
    the driven up/down coil pair.  Enough structure for physical L/R times without
    any machine-description dependency.
    """
    rows = []
    circuit = 0
    for height in np.linspace(-1.0, 1.0, 3):  # centre column
        rows.append((circuit, 0.30, height, 0.003, 0.667, 1.0))
        circuit += 1
    for height in np.linspace(-1.0, 1.0, 3):  # outer cylinder
        rows.append((circuit, 1.70, height, 0.003, 0.667, 1.0))
        circuit += 1
    for radius in np.linspace(0.45, 1.55, 3):  # end plates
        rows.append((circuit, radius, 1.05, 0.367, 0.003, 1.0))
        circuit += 1
        rows.append((circuit, radius, -1.05, 0.367, 0.003, 1.0))
        circuit += 1
    rows.append((100, 1.20, 0.70, 0.10, 0.10, 100.0))
    rows.append((101, 1.20, -0.70, 0.10, 0.10, 100.0))
    columns = np.asarray(rows, dtype=np.float64)
    conductors = ConductorSet(
        circuit=columns[:, 0].astype(np.int64),
        r=columns[:, 1],
        z=columns[:, 2],
        dr=columns[:, 3],
        dz=columns[:, 4],
        current_share=columns[:, 5],
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    sensors = SensorSet(
        r=1.0 + 0.75 * np.cos(angle),
        z=1.15 * np.sin(angle),
        angle=angle,
        is_flux=np.arange(angle.size) % 3 == 0,
    )
    grid_r, grid_z = np.meshgrid(np.linspace(0.4, 1.6, 9), np.linspace(-0.9, 0.9, 11))
    return conductors, sensors, grid_r.ravel(), grid_z.ravel()


def _shell_system(**kwargs):
    conductors, sensors, grid_r, grid_z = _shell_vessel()
    system = build_passive_circuit_system(
        conductors,
        sensors,
        grid_r,
        grid_z,
        passive_circuits=range(12),
        channel_circuits={"upper_current": [100], "lower_current": [101]},
        **kwargs,
    )
    return conductors, sensors, system


# --- propagation ------------------------------------------------------------
def test_constant_drive_leaves_no_eddy_state():
    basis = _toy_basis()
    times = np.linspace(0.0, 0.3, 12)
    i_channel = np.full((12, basis.m_channel.shape[1]), 3.0e3)
    i_cell = np.full((12, basis.m_cell.shape[1]), 40.0)
    state, swing = eddy_history(basis, times, i_channel, i_cell)
    assert np.abs(state).max() == 0.0
    assert np.abs(swing).max() == 0.0


def test_eddy_history_matches_dense_sub_step_integration():
    """The ZOH update equals the dense sub-stepped ODE solution for the
    piecewise-linear linked flux it assumes."""
    basis = _toy_basis()
    times, i_channel, i_cell = _sequences(basis, n_t=10)
    state, _swing = eddy_history(basis, times, i_channel, i_cell)

    psi = i_channel @ basis.m_channel.T + i_cell @ basis.m_cell.T
    reference = np.zeros(basis.n_modes)
    for step in range(1, times.size):
        interval = times[step] - times[step - 1]
        n_sub = 4000
        sub = interval / n_sub
        rate = (psi[step] - psi[step - 1]) / interval
        for _ in range(n_sub):  # exponential-Euler sub-steps, constant drive
            decay = np.exp(-sub / basis.tau)
            reference = decay * reference + (1.0 - decay) * (-basis.tau * rate)
        assert np.allclose(state[step], reference, rtol=1e-4, atol=1e-12)


def test_state_decays_purely_once_the_drive_stops():
    basis = _toy_basis()
    basis = PassiveEigenbasis(
        tau=basis.tau,
        v=basis.v,
        a_sensor=basis.a_sensor,
        g_grid=basis.g_grid,
        m_channel=RNG.normal(size=(3, 1)),
        m_cell=np.zeros((3, 2)),
        resistivity=basis.resistivity,
    )
    times = np.array([0.0, 0.01, 0.05, 0.15])
    i_channel = np.array([[0.0], [1000.0], [1000.0], [1000.0]])
    state, _ = eddy_history(basis, times, i_channel, np.zeros((4, 2)))
    # steps two and three carry no drive: pure exponential decay of step one
    np.testing.assert_allclose(
        state[2], state[1] * np.exp(-(0.05 - 0.01) / basis.tau), rtol=1e-12
    )
    np.testing.assert_allclose(
        state[3], state[2] * np.exp(-(0.15 - 0.05) / basis.tau), rtol=1e-12
    )


def test_the_three_integrators_agree():
    """Arbitrary-cadence reference, uniform-cadence filter and JAX scan are the
    same recurrence, so they agree to round-off on the same mode flux."""
    basis = _toy_basis()
    times, i_channel, i_cell = _sequences(basis, n_t=12)
    reference, swing = eddy_history(basis, times, i_channel, i_cell)
    psi_mode = i_channel @ basis.m_channel.T + i_cell @ basis.m_cell.T
    state, mode_swing = integrate_eddy_ode(basis.tau, times, psi_mode)
    np.testing.assert_array_equal(state, reference)
    np.testing.assert_array_equal(mode_swing, swing)

    uniform_times = np.arange(400) * 1e-3
    uniform_psi = np.cumsum(RNG.normal(size=(400, basis.n_modes)), axis=0) * 1e-4
    exact, _ = integrate_eddy_ode(basis.tau, uniform_times, uniform_psi)
    filtered = zoh_mode_response(basis.tau, 1e-3, uniform_psi)
    np.testing.assert_allclose(filtered, exact, rtol=1e-10, atol=1e-18)


def test_voltage_drive_matches_the_reference_and_its_steady_state():
    tau = np.array([0.030, 0.008])
    times = np.arange(600) * 1e-3
    psi = np.zeros((600, 2))
    voltage = np.zeros((600, 2))
    voltage[100:] = np.array([2.0, -1.0])  # step voltage
    exact, _ = integrate_eddy_ode(tau, times, psi, voltage_mode=voltage)
    filtered = zoh_mode_response(tau, 1e-3, psi, voltage_mode=voltage)
    np.testing.assert_allclose(filtered, exact, rtol=1e-10, atol=1e-16)
    # the constant-voltage steady state of da/dt + a/tau = v is a = tau v
    np.testing.assert_allclose(exact[-1], tau * voltage[-1], rtol=1e-3)


@pytest.mark.slow
def test_jax_scan_matches_the_host_integrator_and_batches():
    configure_dtypes()
    basis = _toy_basis()
    times, i_channel, i_cell = _sequences(basis, n_t=20)
    psi_mode = i_channel @ basis.m_channel.T + i_cell @ basis.m_cell.T
    reference, swing = integrate_eddy_ode(basis.tau, times, psi_mode)

    state, mode_swing = scan_eddy_modes(basis.tau, times, psi_mode)
    np.testing.assert_allclose(np.asarray(state), reference, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(mode_swing), swing, rtol=1e-12, atol=0.0)

    # a batch of drive histories propagates in one call, shapes fixed
    batch = jnp.stack([psi_mode, 2.0 * psi_mode, -psi_mode])
    batched = scan_eddy_modes(basis.tau, times, batch)[0]
    assert batched.shape == (3, times.size, basis.n_modes)
    np.testing.assert_allclose(
        np.asarray(batched[1]), 2.0 * reference, rtol=1e-12, atol=1e-18
    )


def test_jax_scan_precision_is_selected_per_call():
    """General automatic propagation is fp64 and explicit fp32 remains available."""
    configure_dtypes()
    tau = np.array([0.03, 0.008])
    times = np.linspace(0.0, 0.1, 8)
    psi = np.column_stack([times, -2.0 * times])

    automatic, _ = scan_eddy_modes(tau, times, psi)
    single, _ = scan_eddy_modes(tau, times, psi, precision=Precision.SINGLE)

    assert automatic.dtype == jnp.float64
    assert single.dtype == jnp.float32


def test_raw_cadence_equals_slice_cadence_for_a_piecewise_linear_drive():
    """Densifying a piecewise-linear flux changes nothing under exact ZOH, so
    with no pre-slice history the two cadences agree exactly."""
    basis = _toy_basis()
    times, i_channel, i_cell = _sequences(basis, n_t=8)
    slice_state, _ = eddy_history(basis, times, i_channel, i_cell)

    raw_times = np.unique(
        np.concatenate(
            [
                np.linspace(times[step - 1], times[step], 11)
                for step in range(1, times.size)
            ]
        )
    )
    raw_channel = np.column_stack(
        [
            np.interp(raw_times, times, i_channel[:, column])
            for column in range(i_channel.shape[1])
        ]
    )
    at_slices, raw_state = raw_eddy_trajectory(
        basis, raw_times, raw_channel, times, i_cell
    )
    assert raw_state.shape == (raw_times.size, basis.n_modes)
    np.testing.assert_allclose(at_slices, slice_state, rtol=1e-10, atol=1e-14)


def test_pre_slice_plasma_flux_follows_the_measured_current():
    """Before the first slice the plasma mode flux follows the measured current
    with the first slice's flux pattern -- shape frozen, amplitude following."""
    basis = _toy_basis()
    raw_times = np.linspace(0.0, 0.1, 101)
    raw_channel = np.zeros((raw_times.size, basis.m_channel.shape[1]))
    slice_times = np.array([0.05, 0.08])
    i_cell = np.abs(RNG.normal(0, 50.0, size=(2, basis.m_cell.shape[1])))
    ip_raw = np.clip(np.interp(raw_times, [0.02, 0.05], [0.0, 2.0e5]), 0, None)

    _, with_plasma = raw_eddy_trajectory(
        basis, raw_times, raw_channel, slice_times, i_cell, ip_raw=ip_raw
    )
    _, without = raw_eddy_trajectory(basis, raw_times, raw_channel, slice_times, i_cell)
    before = raw_times < 0.05
    assert np.abs(with_plasma[before]).max() > 0.0
    assert np.abs(without[before][:-1]).max() == 0.0


def test_uniform_tau_scale_scales_the_decay_exactly():
    basis = _toy_basis()
    raw_times = np.linspace(0.0, 0.2, 201)
    raw_channel = np.zeros((raw_times.size, basis.m_channel.shape[1]))
    raw_channel[raw_times >= 0.01] = 1.0e3  # one step, then flat
    slice_times = np.array([0.15, 0.19])
    i_cell = np.zeros((2, basis.m_cell.shape[1]))
    scale = 2.0
    _, nominal = raw_eddy_trajectory(basis, raw_times, raw_channel, slice_times, i_cell)
    _, scaled = raw_eddy_trajectory(
        basis, raw_times, raw_channel, slice_times, i_cell, tau_scale=scale
    )
    early, late = 100, 180
    interval = raw_times[late] - raw_times[early]
    np.testing.assert_allclose(
        nominal[late] / nominal[early], np.exp(-interval / basis.tau), rtol=1e-10
    )
    np.testing.assert_allclose(
        scaled[late] / scaled[early],
        np.exp(-scale * interval / basis.tau),
        rtol=1e-10,
    )


# --- the circuit system ----------------------------------------------------
def test_shell_vessel_decay_times_are_physical():
    """A steel shell vessel's L/R times land in the tens-of-milliseconds band,
    with L-orthonormal eigenvectors and shape-consistent couplings."""
    conductors, sensors, system = _shell_system()
    assert system.n_circuits == 12
    assert system.channels == ["lower_current", "upper_current"]
    tau, vectors = system.mode_system()
    assert tau.max() > 1e-3
    assert tau.max() < 1.0
    assert tau.min() > 1e-6
    # v is L-orthonormal: v' L v = I
    np.testing.assert_allclose(
        vectors.T @ system.lmat @ vectors, np.eye(12), rtol=1e-8, atol=1e-10
    )
    assert system.a_circuit.shape == (sensors.n_sensors, 12)
    assert system.m_channel.shape == (12, 2)
    assert np.all(system.r_diag > 0)
    # the centre-column segments sit at the smallest radius of the set
    assert system.centroid_r.min() == pytest.approx(0.30)


def test_reduction_keeps_the_most_relevant_modes_slowest_first():
    _conductors, sensors, system = _shell_system()
    cell_index = np.arange(0, system.g_grid.shape[0], 7)
    basis = reduce_passive_system(
        system,
        sensor_scale=np.ones(sensors.n_sensors),
        n_modes=5,
        cell_index=cell_index,
    )
    assert basis.n_modes == 5
    assert np.all(np.diff(basis.tau) <= 1e-12)  # slowest first
    assert basis.a_sensor.shape == (sensors.n_sensors, 5)
    assert basis.g_grid.shape == (system.g_grid.shape[0], 5)
    assert basis.m_cell.shape == (5, cell_index.size)
    # reciprocity: the plasma-cell coupling IS the grid column at those cells
    np.testing.assert_allclose(basis.m_cell, basis.g_grid[cell_index].T)
    # the kept set is the top of the relevance ranking
    tau, vectors = system.mode_system()
    relevance = mode_relevance(
        tau, system.a_circuit @ vectors, np.ones(sensors.n_sensors)
    )
    assert set(np.argsort(relevance)[::-1][:5]) == {
        int(np.argmin(np.abs(tau - kept))) for kept in basis.tau
    }


def test_uniform_resistance_multiplier_scales_every_decay_time():
    _conductors, _sensors, system = _shell_system()
    tau, vectors = system.mode_system()
    scaled_tau, scaled_vectors = system.mode_system(np.full(12, 4.0))
    np.testing.assert_allclose(scaled_tau, tau / 4.0, rtol=1e-9)
    np.testing.assert_allclose(
        np.abs(scaled_vectors), np.abs(vectors), rtol=1e-6, atol=1e-9
    )


def test_bad_resistance_multipliers_fail_loud():
    _conductors, _sensors, system = _shell_system()
    with pytest.raises(ValueError, match="shape"):
        system.mode_system(np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        system.mode_system(np.full(12, -1.0))


def test_a_held_back_channel_never_becomes_a_drive_column():
    """Moving an instrumented circuit into the passive set drops its channel from
    the drive columns and records which row it measures."""
    conductors, sensors, grid_r, grid_z = _shell_vessel()
    system = build_passive_circuit_system(
        conductors,
        sensors,
        grid_r,
        grid_z,
        passive_circuits=[*range(12), 101],
        channel_circuits={"upper_current": [100], "lower_current": [101]},
        measured_circuits={"lower_current": 101},
    )
    assert system.channels == ["upper_current"]
    assert system.measured_channel_row == {"lower_current": 12}
    assert system.n_circuits == 13
    assert system.m_channel.shape == (13, 1)


def test_plasma_drive_adds_a_transient_the_coils_alone_do_not():
    """A plasma current ramp induces circuit currents on top of the drive-only
    prediction; a zero / non-finite current sample contributes nothing."""
    conductors, _sensors, system = _shell_system()
    times = np.linspace(0.0, 0.05, 51)
    i_channel = np.zeros((times.size, 2))  # coil-quiet: plasma drive only
    ip = np.clip(np.interp(times, [0.01, 0.03], [0.0, 4.0e5]), 0.0, None)
    axis = np.column_stack([np.full(times.size, 1.0), np.zeros(times.size)])
    drive_only, full = predict_circuit_currents(
        system,
        conductors,
        times,
        i_channel,
        ["upper_current", "lower_current"],
        ip_amperes=ip,
        axis_rz=axis,
    )
    assert np.abs(drive_only).max() == 0.0
    assert np.abs(full).max() > 0.0
    # the induced currents oppose the rising plasma current (Lenz)
    assert np.sum(full[np.argmax(ip > 0) + 1]) < 0.0
    # without the plasma arguments the two returns are the same array
    quiet, quiet_full = predict_circuit_currents(
        system, conductors, times, i_channel, ["upper_current", "lower_current"]
    )
    np.testing.assert_array_equal(quiet, quiet_full)


def test_circuit_system_and_eigenbasis_round_trip(tmp_path):
    _conductors, sensors, system = _shell_system()
    save_circuit_system(tmp_path / "system.npz", system)
    back = load_circuit_system(tmp_path / "system.npz")
    np.testing.assert_array_equal(back.lmat, system.lmat)
    np.testing.assert_array_equal(back.r_diag, system.r_diag)
    assert back.channels == system.channels
    assert back.measured_channel_row == system.measured_channel_row
    assert back.resistivity == system.resistivity

    solver = PassiveCircuit(
        system=system,
        cell_index=np.arange(0, system.g_grid.shape[0], 7),
        sensor_scale=np.ones(sensors.n_sensors),
        n_modes=4,
    )
    basis = solver.reduce()
    save_eigenbasis(tmp_path / "basis.npz", basis)
    restored = load_eigenbasis(tmp_path / "basis.npz")
    np.testing.assert_array_equal(restored.tau, basis.tau)
    np.testing.assert_array_equal(restored.m_cell, basis.m_cell)
    assert restored.volt_channel is None
