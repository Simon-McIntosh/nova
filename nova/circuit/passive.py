"""Passive structure carried as an L/R eigenmode circuit system.

The vessel, coil cases and in-vessel structures form closed toroidal current
paths nobody drives and nobody measures (with the exception of a few instrumented
coil cases).  They are the machine's magnetic memory: a coil or plasma current
swing induces currents in them that decay on their own L/R times, adding field at
the plasma that no per-slice fit of the measured drives can account for.

This module reduces that structure to a linear circuit system and propagates it:

* **inductance** is EXACT from geometry -- the two-section flux linkage of
  :mod:`nova.circuit.linkage` on the finite-area axisymmetric kernels.  It is a
  prior no learner should re-fit.
* **resistance** is not.  The nominal toroidal-ring resistance
  ``2 pi r rho / (dr dz)`` at the true cross-section is a starting point; the
  real conducting paths of a welded 3-D shell are what
  :mod:`nova.circuit.resistance` calibrates against coil-only intervals.  The
  resistance degrees of freedom live in CIRCUIT space, diagonal (toroidal rings
  share no conductor path), so a candidate resistance model costs one cheap
  generalised eigensolve while the geometry-exact linkage never changes.
* **reduction** keeps the modes whose history the measurements can actually see:
  the relevance ``tau * ||a_sensor / scale||`` ranks a mode by how long it lives
  times how strongly it shows up in whitened sensor units.  A slow mode the
  sensors can see is exactly a mode whose history a per-slice fit cannot absorb.
* **propagation** is the exact zero-order-hold integration of
  :mod:`nova.circuit.propagate`.

Held-back measured circuits: when a machine instruments some of its passive
circuits (a coil case with a current transducer), those measurements are far more
valuable as HELD-BACK targets than as inputs.  Moving them into the passive set
makes their currents predicted from the remaining drives through the mutual
couplings, and the measurement becomes the strongest per-circuit test of both L
and R the corpus offers.  Their channels then never appear as drive columns.

Conventions: total poloidal flux ``Phi = 2 pi R A_phi`` [Wb], explicit ``mu0``
inside the kernels, raw SI throughout.  Mode amplitudes are in the
L-orthonormal eigencoordinates everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nova.biot.greens import greens_psi
from nova.circuit.conductor import ConductorSet, SensorSet
from nova.circuit.linkage import (
    channel_flux_linkage,
    circuit_linkage_matrix,
    guard_positive_definite,
    ring_resistance,
    sensor_grid_couplings,
)
from nova.circuit.propagate import integrate_eddy_ode

#: nominal stainless-steel resistivity [Ohm m] -- the SCALE at which the ring
#: resistances start; a bounded cross-shot multiplier is the calibrated unknown
NOMINAL_STEEL_RESISTIVITY = 7.2e-7


@dataclass
class PassiveCircuitSystem:
    """Circuit-space L/R system of a passive set, before eigen-reduction.

    ``lmat`` ``(n_circuits, n_circuits)`` is the positive-definite two-section
    flux linkage [Wb/A] and ``r_diag`` the diagonal nominal ring resistances
    [Ohm].  ``a_circuit`` / ``g_grid`` are the per-ampere sensor signatures and
    grid flux columns; ``m_channel`` ``(n_circuits, n_channels)`` the flux each
    circuit links per ampere of each measured drive channel [Wb/A].

    ``measured_channel_row`` is non-empty only when instrumented circuits were
    moved INTO the passive set: it names which circuit row each held-back
    channel measures, and those channels are then absent from ``channels``.
    """

    circuits: np.ndarray
    centroid_r: np.ndarray
    centroid_z: np.ndarray
    lmat: np.ndarray
    r_diag: np.ndarray
    a_circuit: np.ndarray
    g_grid: np.ndarray
    m_channel: np.ndarray
    channels: list[str]
    measured_channel_row: dict[str, int]
    resistivity: float
    section_scale: np.ndarray

    @property
    def n_circuits(self) -> int:
        """Return the number of passive circuits carried."""
        return int(self.circuits.size)

    def mode_system(self, r_multipliers: np.ndarray | None = None):
        """Solve ``R v = (1/tau) L v`` for one candidate resistance model.

        Returns ``(tau, v)`` with ``tau`` the decay times [s] and ``v`` the
        L-orthonormal eigenvectors mapping mode amplitudes to circuit currents.
        ``r_multipliers`` scales the diagonal ring resistances per circuit -- the
        calibration hook; the geometry-exact linkage and every coupling stay
        fixed while the data-led resistance model reshapes the modes.
        """
        from scipy.linalg import eigh

        r_diag = self.r_diag
        if r_multipliers is not None:
            multiplier = np.asarray(r_multipliers, dtype=np.float64)
            if multiplier.shape != r_diag.shape:
                raise ValueError(
                    f"r_multipliers shape {multiplier.shape} != circuits {r_diag.shape}"
                )
            if np.any(~np.isfinite(multiplier)) or np.any(multiplier <= 0):
                raise ValueError("r_multipliers must be finite and positive")
            r_diag = r_diag * multiplier
        rate, vectors = eigh(np.diag(r_diag), self.lmat)
        return 1.0 / np.clip(rate, 1e-12, None), vectors


@dataclass
class PassiveEigenbasis:
    """L/R eigenmodes of a passive set, reduced to the modes kept.

    ``tau`` ``(n_modes,)`` are the physical decay times [s], slowest first;
    ``v`` ``(n_circuits, n_modes)`` the L-orthonormal eigenvector block mapping a
    mode amplitude to circuit currents.  ``a_sensor`` ``(n_sensors, n_modes)``
    and ``g_grid`` ``(n_points, n_modes)`` map a mode amplitude to sensor
    readings / grid flux; ``m_channel`` ``(n_modes, n_channels)`` and
    ``m_cell`` ``(n_modes, n_cells)`` are the flux linkages each mode picks up
    per ampere of drive channel / plasma cell current -- the DRIVE couplings.

    ``volt_channel`` ``(n_modes, n_channels)`` [Ohm] carries voltage-type drive
    couplings when the circuit topology includes a galvanic term (a case wired
    across its winding sees the winding's terminal voltage); ``None`` otherwise.
    """

    tau: np.ndarray
    v: np.ndarray
    a_sensor: np.ndarray
    g_grid: np.ndarray
    m_channel: np.ndarray
    m_cell: np.ndarray
    resistivity: float
    volt_channel: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        """Return the number of modes retained."""
        return int(self.tau.size)


def build_passive_circuit_system(
    conductors: ConductorSet,
    sensors: SensorSet,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    *,
    passive_circuits,
    channel_circuits: dict[str, list[int]],
    measured_circuits: dict[str, int] | None = None,
    channel_gain: dict[str, float] | None = None,
    resistivity: float = NOMINAL_STEEL_RESISTIVITY,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
) -> PassiveCircuitSystem:
    """Circuit-space L, R and couplings of a passive set -- pure geometry.

    ``passive_circuits`` are the circuits carried as state; ``channel_circuits``
    maps each measured drive channel to the circuits it energises.
    ``measured_circuits`` maps an instrumented channel to the passive circuit it
    measures: those circuits are held back (state predicted from the remaining
    drives, the measurement kept as a target) and their channels are dropped from
    the drive columns.  Pass them in ``passive_circuits`` as well.

    Inductance is the two-section linkage, positive-definite-guarded; resistance
    the true-cross-section ring resistance at ``resistivity``.  Build once per
    machine description and reuse across slices.
    """
    circuits = sorted(int(c) for c in passive_circuits)
    if not circuits:
        raise ValueError("the passive set is empty")
    measured_circuits = dict(measured_circuits or {})
    held_back = set(measured_circuits)
    driven = {
        channel: sorted(int(c) for c in energised)
        for channel, energised in channel_circuits.items()
        if channel not in held_back
    }

    lmat = guard_positive_definite(
        circuit_linkage_matrix(
            conductors,
            circuits,
            section_scale_frac=section_scale_frac,
            section_n_max=section_n_max,
        )
    )
    centroid_r, centroid_z = conductors.centroids(circuits)
    a_circuit, g_grid = sensor_grid_couplings(
        conductors, circuits, sensors, grid_r, grid_z
    )
    channels, m_channel = channel_flux_linkage(
        conductors,
        circuits,
        driven,
        channel_gain=channel_gain,
        section_scale_frac=section_scale_frac,
        section_n_max=section_n_max,
    )
    row_of = {circuit: index for index, circuit in enumerate(circuits)}
    return PassiveCircuitSystem(
        circuits=np.asarray(circuits, dtype=np.int64),
        centroid_r=centroid_r,
        centroid_z=centroid_z,
        lmat=lmat,
        r_diag=ring_resistance(conductors, circuits, resistivity),
        a_circuit=a_circuit,
        g_grid=g_grid,
        m_channel=m_channel,
        channels=channels,
        measured_channel_row={
            channel: row_of[int(circuit)]
            for channel, circuit in measured_circuits.items()
        },
        resistivity=float(resistivity),
        section_scale=conductors.section_scale(circuits),
    )


def mode_relevance(
    tau: np.ndarray, a_modes: np.ndarray, sensor_scale: np.ndarray
) -> np.ndarray:
    """History relevance ``tau * ||a_sensor / scale||`` of each mode.

    How long a mode lives times how strongly it shows up in whitened sensor
    units.  A slow mode the sensors can see is exactly a mode whose history a
    per-slice fit cannot absorb, which is what makes this the reduction ranking.
    """
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    return tau * np.linalg.norm(a_modes / scale[:, np.newaxis], axis=0)


def select_modes(
    tau: np.ndarray, a_modes: np.ndarray, sensor_scale: np.ndarray, n_modes: int
) -> np.ndarray:
    """Indices of the ``n_modes`` most history-relevant modes, slowest first."""
    keep = np.argsort(mode_relevance(tau, a_modes, sensor_scale))[::-1][: int(n_modes)]
    return keep[np.argsort(tau[keep])[::-1]]


def reduce_passive_system(
    system: PassiveCircuitSystem,
    *,
    sensor_scale: np.ndarray,
    n_modes: int = 12,
    cell_index: np.ndarray,
    r_multipliers: np.ndarray | None = None,
) -> PassiveEigenbasis:
    """Eigen-reduce a circuit system to its most history-relevant modes.

    ``cell_index`` indexes the plasma current cells into the grid rows of
    ``system.g_grid``; the plasma drive coupling follows by reciprocity --
    ``m_cell = g_grid[cells].T``, since the flux a mode links per ampere of cell
    current equals the flux the cell sees per ampere of mode current.
    """
    tau, vectors = system.mode_system(r_multipliers)
    a_modes = system.a_circuit @ vectors
    keep = select_modes(tau, a_modes, sensor_scale, n_modes)

    v_keep = vectors[:, keep]
    g_modes = system.g_grid @ v_keep
    return PassiveEigenbasis(
        tau=tau[keep],
        v=v_keep,
        a_sensor=a_modes[:, keep],
        g_grid=g_modes,
        m_channel=v_keep.T @ system.m_channel,
        m_cell=g_modes[np.asarray(cell_index), :].T,
        resistivity=float(system.resistivity),
    )


def build_passive_eigenbasis(
    conductors: ConductorSet,
    sensors: SensorSet,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    *,
    passive_circuits,
    channel_circuits: dict[str, list[int]],
    cell_index: np.ndarray,
    sensor_scale: np.ndarray,
    n_modes: int = 12,
    measured_circuits: dict[str, int] | None = None,
    channel_gain: dict[str, float] | None = None,
    resistivity: float = NOMINAL_STEEL_RESISTIVITY,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
    r_multipliers: np.ndarray | None = None,
) -> PassiveEigenbasis:
    """Build and eigen-reduce a passive set in one call.

    Convenience over :func:`build_passive_circuit_system` followed by
    :func:`reduce_passive_system`; keep the circuit system when several
    resistance models are to be tried, since only the eigensolve depends on them.
    """
    system = build_passive_circuit_system(
        conductors,
        sensors,
        grid_r,
        grid_z,
        passive_circuits=passive_circuits,
        channel_circuits=channel_circuits,
        measured_circuits=measured_circuits,
        channel_gain=channel_gain,
        resistivity=resistivity,
        section_scale_frac=section_scale_frac,
        section_n_max=section_n_max,
    )
    return reduce_passive_system(
        system,
        sensor_scale=sensor_scale,
        n_modes=n_modes,
        cell_index=cell_index,
        r_multipliers=r_multipliers,
    )


def eddy_history(
    basis: PassiveEigenbasis,
    times: np.ndarray,
    i_channel: np.ndarray,
    i_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode state along a slice sequence from its own drives.

    The mode flux is ``Psi_m(t) = m_channel_m . i_channel(t) +
    m_cell_m . i_cell(t)``.  Returns ``(a, u)`` -- the mode state and the
    per-step flux swing.  ``a[0] = 0`` takes the first slice as the eddy
    reference; :func:`raw_eddy_trajectory` removes that approximation by
    integrating the raw-cadence drives from the stream start instead.
    """
    psi_mode = (
        np.asarray(i_channel, dtype=np.float64) @ basis.m_channel.T
        + np.asarray(i_cell, dtype=np.float64) @ basis.m_cell.T
    )
    return integrate_eddy_ode(basis.tau, times, psi_mode)


def raw_eddy_trajectory(
    basis: PassiveEigenbasis,
    raw_times: np.ndarray,
    i_channel_raw: np.ndarray,
    slice_times: np.ndarray,
    i_cell_slices: np.ndarray,
    *,
    ip_raw: np.ndarray | None = None,
    tau_scale: float | np.ndarray = 1.0,
    voltage_mode_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode state at the reconstructed slices from RAW-cadence integration.

    The mode flux is assembled at the raw measurement cadence and integrated from
    the raw stream start with ``a = 0`` (the pre-drive machine is quiescent), so
    the slice sequence inherits the full drive history -- solenoid precharge,
    breakdown flux swing -- instead of the ``a[0] = 0`` slice-cadence
    approximation.

    Drive term: ``m_channel . i_channel_raw(t)`` per raw sample.  Plasma term:
    the exact per-slice mode flux ``m_cell . i_cell(t_slice)`` -- the FULL
    time-varying current distribution, so a changing plasma shape, position and
    internal inductance enter the flux swing as ``d(M(t) Ip(t))/dt`` rather than
    the fixed-mutual approximation ``M dIp/dt`` -- linearly interpolated to the
    raw cadence (interpolation commutes with the fixed linear map ``m_cell``).
    Before the first slice the first slice's flux PATTERN is amplitude-followed
    with the measured plasma current ``ip_raw``; without it the plasma term is
    zero-ramped from the raw start.

    ``tau_scale`` is the bounded resistance-scale degree of freedom: a UNIFORM
    scalar leaves the eigenvectors invariant and maps every ``tau -> tau/scale``
    exactly, while a per-mode array is the diagonal-in-eigenbasis approximation
    to a structured resistivity change -- bounded, calibrated across shots, never
    per-slice.

    Returns ``(a_slices, a_raw)`` in the L-orthonormal mode coordinates.
    """
    raw_times = np.asarray(raw_times, dtype=np.float64)
    slice_times = np.asarray(slice_times, dtype=np.float64)
    psi_channel = np.asarray(i_channel_raw, dtype=np.float64) @ basis.m_channel.T

    psi_cell_slices = np.asarray(i_cell_slices, dtype=np.float64) @ basis.m_cell.T
    psi_cell_raw = np.empty_like(psi_channel)
    for mode in range(basis.n_modes):
        # constant-extrapolates outside the slice span
        psi_cell_raw[:, mode] = np.interp(
            raw_times, slice_times, psi_cell_slices[:, mode]
        )
    before = raw_times < slice_times[0]
    if np.any(before):
        if ip_raw is not None:
            ip_raw = np.asarray(ip_raw, dtype=np.float64)
            ip_first = float(np.interp(slice_times[0], raw_times, ip_raw))
            fraction = np.zeros(int(before.sum()))
            if abs(ip_first) > 1e-12:
                fraction = np.clip(ip_raw[before] / ip_first, 0.0, 1.0)
            psi_cell_raw[before] = fraction[:, np.newaxis] * psi_cell_slices[0]
        else:
            psi_cell_raw[before] = 0.0

    tau = basis.tau / np.asarray(tau_scale, dtype=np.float64)
    a_raw, _swing = integrate_eddy_ode(
        tau,
        raw_times,
        psi_channel + psi_cell_raw,
        voltage_mode=voltage_mode_raw,
    )
    a_slices = np.column_stack(
        [
            np.interp(slice_times, raw_times, a_raw[:, mode])
            for mode in range(basis.n_modes)
        ]
    )
    return a_slices, a_raw


def predict_circuit_currents(
    system: PassiveCircuitSystem,
    conductors: ConductorSet,
    times: np.ndarray,
    i_channel: np.ndarray,
    channels: list[str],
    *,
    ip_amperes: np.ndarray | None = None,
    axis_rz: np.ndarray | None = None,
    r_multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Passive circuit currents from the measured drives, quiescent start.

    Exact-ZOH eigenmode integration of the full circuit system (no mode
    reduction) driven by the measured drive history ``i_channel`` ``(n_t,
    n_channels)`` in ``channels`` order; channels the system does not carry
    contribute zero.

    When ``ip_amperes`` ``(n_t,)`` and ``axis_rz`` ``(n_t, 2)`` are given, the
    PLASMA current's own flux swing drives the structure too, as a toroidal
    filament at the given axis trace.  The Lenz-antiparallel image currents a
    fast current ramp induces add vertical field at the plasma in the CONFINING
    direction while they last, and decay on the structure's L/R times once the
    ramp holds -- a drive term absent from every drives-only prediction.
    Non-finite or zero-current samples contribute no plasma linkage, which is the
    pre-plasma convention: before breakdown the structure is drive-driven only.

    Returns ``(i_drive_only, i_full)`` in ``system.circuits`` order (identical
    when the plasma drive is omitted).  A consumer injects
    ``system.g_grid @ i_full[t]`` on the grid or ``system.a_circuit @ i_full[t]``
    at the sensors.
    """
    times = np.asarray(times, dtype=np.float64)
    i_channel = np.asarray(i_channel, dtype=np.float64)
    column_of = {channel: index for index, channel in enumerate(system.channels)}
    m_drive = np.zeros((system.n_circuits, len(channels)))
    for index, channel in enumerate(channels):
        if channel in column_of:
            m_drive[:, index] = system.m_channel[:, column_of[channel]]
    tau, vectors = system.mode_system(r_multipliers)
    psi_drive = i_channel @ m_drive.T
    a_drive, _swing = integrate_eddy_ode(tau, times, psi_drive @ vectors)
    i_drive_only = a_drive @ vectors.T
    if ip_amperes is None or axis_rz is None:
        return i_drive_only, i_drive_only

    # plasma -> circuit linkage per slice: the share-weighted flux per ampere of
    # a loop at the (time-varying) axis, vectorised over all passive filaments
    ip_amperes = np.asarray(ip_amperes, dtype=np.float64)
    axis_rz = np.asarray(axis_rz, dtype=np.float64)
    rows = np.concatenate(conductors.rows(system.circuits))
    owner = np.concatenate(
        [
            np.full(group.size, index, dtype=np.int64)
            for index, group in enumerate(conductors.rows(system.circuits))
        ]
    )
    psi_plasma = np.zeros((times.size, system.n_circuits))
    for step in range(times.size):
        if (
            not np.isfinite(ip_amperes[step])
            or ip_amperes[step] == 0.0
            or not np.all(np.isfinite(axis_rz[step]))
        ):
            continue
        psi = np.atleast_1d(
            greens_psi(
                conductors.r[rows],
                conductors.z[rows],
                float(axis_rz[step, 0]),
                float(axis_rz[step, 1]),
            )
        )
        psi_plasma[step] = np.bincount(
            owner,
            weights=conductors.current_share[rows] * psi,
            minlength=system.n_circuits,
        ) * float(ip_amperes[step])
    a_full, _swing = integrate_eddy_ode(tau, times, (psi_drive + psi_plasma) @ vectors)
    return i_drive_only, a_full @ vectors.T


@dataclass
class PassiveCircuit:
    """Passive structure solver: build the L/R system, reduce it, propagate it.

    The stateful face over this module.  Construction is pure geometry, so one
    instance serves a whole campaign; ``reduce`` is where a calibrated resistance
    model enters, and ``eddy_history`` / ``raw_trajectory`` propagate a drive
    history through the retained modes.

    ``sensor_scale`` is the per-channel measurement scale the mode-relevance
    ranking whitens by (a robust noise estimate, not a fit residual).
    """

    system: PassiveCircuitSystem
    cell_index: np.ndarray
    sensor_scale: np.ndarray
    n_modes: int = 12

    @classmethod
    def build(
        cls,
        conductors: ConductorSet,
        sensors: SensorSet,
        grid_r: np.ndarray,
        grid_z: np.ndarray,
        *,
        passive_circuits,
        channel_circuits: dict[str, list[int]],
        cell_index: np.ndarray,
        sensor_scale: np.ndarray,
        n_modes: int = 12,
        **kwargs,
    ) -> PassiveCircuit:
        """Build the circuit system from a machine description."""
        return cls(
            system=build_passive_circuit_system(
                conductors,
                sensors,
                grid_r,
                grid_z,
                passive_circuits=passive_circuits,
                channel_circuits=channel_circuits,
                **kwargs,
            ),
            cell_index=np.asarray(cell_index),
            sensor_scale=np.asarray(sensor_scale, dtype=np.float64),
            n_modes=int(n_modes),
        )

    def reduce(self, r_multipliers: np.ndarray | None = None) -> PassiveEigenbasis:
        """Eigen-reduce under a candidate resistance model."""
        return reduce_passive_system(
            self.system,
            sensor_scale=self.sensor_scale,
            n_modes=self.n_modes,
            cell_index=self.cell_index,
            r_multipliers=r_multipliers,
        )


def save_circuit_system(path: Path | str, system: PassiveCircuitSystem) -> None:
    """Persist a circuit system -- the linkage build is minutes of kernels."""
    np.savez_compressed(
        path,
        circuits=system.circuits,
        centroid_r=system.centroid_r,
        centroid_z=system.centroid_z,
        lmat=system.lmat,
        r_diag=system.r_diag,
        a_circuit=system.a_circuit,
        g_grid=system.g_grid,
        m_channel=system.m_channel,
        channels=np.array(system.channels),
        measured_channel_row=np.frombuffer(
            json.dumps(system.measured_channel_row).encode(), dtype=np.uint8
        ),
        resistivity=np.float64(system.resistivity),
        section_scale=system.section_scale,
    )


def load_circuit_system(path: Path | str) -> PassiveCircuitSystem:
    """Read back a persisted circuit system."""
    with np.load(path) as stored:
        return PassiveCircuitSystem(
            circuits=stored["circuits"],
            centroid_r=stored["centroid_r"],
            centroid_z=stored["centroid_z"],
            lmat=stored["lmat"],
            r_diag=stored["r_diag"],
            a_circuit=stored["a_circuit"],
            g_grid=stored["g_grid"],
            m_channel=stored["m_channel"],
            channels=[str(channel) for channel in stored["channels"]],
            measured_channel_row={
                channel: int(row)
                for channel, row in json.loads(
                    stored["measured_channel_row"].tobytes()
                ).items()
            },
            resistivity=float(stored["resistivity"]),
            section_scale=stored["section_scale"],
        )


def save_eigenbasis(path: Path | str, basis: PassiveEigenbasis) -> None:
    """Persist an eigenbasis -- the build is minutes of kernel sums."""
    extra = (
        {"volt_channel": basis.volt_channel} if basis.volt_channel is not None else {}
    )
    np.savez_compressed(
        path,
        tau=basis.tau,
        v=basis.v,
        a_sensor=basis.a_sensor,
        g_grid=basis.g_grid,
        m_channel=basis.m_channel,
        m_cell=basis.m_cell,
        resistivity=np.float64(basis.resistivity),
        **extra,
    )


def load_eigenbasis(path: Path | str) -> PassiveEigenbasis:
    """Read back a persisted eigenbasis."""
    with np.load(path) as stored:
        return PassiveEigenbasis(
            tau=stored["tau"],
            v=stored["v"],
            a_sensor=stored["a_sensor"],
            g_grid=stored["g_grid"],
            m_channel=stored["m_channel"],
            m_cell=stored["m_cell"],
            resistivity=float(stored["resistivity"]),
            volt_channel=(
                stored["volt_channel"] if "volt_channel" in stored.files else None
            ),
        )


__all__ = [
    "NOMINAL_STEEL_RESISTIVITY",
    "PassiveCircuit",
    "PassiveCircuitSystem",
    "PassiveEigenbasis",
    "build_passive_circuit_system",
    "build_passive_eigenbasis",
    "eddy_history",
    "load_circuit_system",
    "load_eigenbasis",
    "mode_relevance",
    "predict_circuit_currents",
    "raw_eddy_trajectory",
    "reduce_passive_system",
    "save_circuit_system",
    "save_eigenbasis",
    "select_modes",
]
