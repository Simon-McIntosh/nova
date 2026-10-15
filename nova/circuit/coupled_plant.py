"""The coupled ITER plant, recovered from a machine description through one call.

This is the single imported surface the switched-converter model needs.  Given a
machine description (its sectioned conductors, coil wiring and reduced passive
circuit system) and a plasma source (an IMAS equilibrium with the coil-current
series that goes with it), :func:`coupled_plant` returns one :class:`CoupledPlant`
-- the coil inductance and resistance, the reduced passive eigenmodes with their
decay times and their coupling to every coil section, the plasma flux drive at
every coil section and passive mode against time, the reduction error against
frequency, and provenance.

The plasma source is duck-typed and minimal, so a DINA case and a reconstructed
measured pulse reach the call the same way: it exposes ``coil_current_times``,
``coil_currents`` and ``prescribed_currents()`` (the flux-function current
distributions at ``plasma_times``).  The plasma is never solved here; it is
prescribed from the source's flux functions against the source's own current.

The build is minutes of kernel sums, so the result is cached through nova's
serialisers.  The cache location resolves through
:class:`nova.database.filepath.FilePath`; the call returns the same plant cold
and from its cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nova.circuit.conductor import ConductorSet
from nova.circuit.passive import PassiveCircuitSystem
from nova.imas.prescribed_plasma import plasma_linked_flux_history

__all__ = ["CoupledPlant", "coupled_plant"]


@dataclass
class CoupledPlant:
    """The coupled plant a switched-converter model takes as a state-space multiport.

    ``coil_names`` names the ``n_coil`` terminals.  ``coil_inductance``
    ``(n_coil, n_coil)`` and ``coil_resistance`` ``(n_coil,)`` are the terminal
    L and R.  ``mode_tau`` ``(n_modes,)`` are the reduced passive decay times [s]
    and ``mode_coil_coupling`` ``(n_modes, n_coil)`` the flux each mode links per
    ampere of each terminal [Wb/A] -- the drive from coil to mode.  The state is
    the mode amplitude ``a_m`` with ``da_m/dt = (Psi_m - a_m) / tau_m`` and
    ``Psi_m = mode_coil_coupling[m] . i_coil + mode_plasma_flux[m](t)``.

    ``section_plasma_flux`` ``(n_times, n_section)`` is the plasma flux linked at
    every coil section against ``plasma_times``, and ``mode_plasma_flux``
    ``(n_times, n_modes)`` its projection onto the reduced modes -- the plasma's
    drive on the plant.  ``reduction_error`` ``(n_frequencies,)`` is the reduced
    plant's terminal-impedance error against the full model over
    ``frequencies``.
    """

    coil_names: list[str]
    coil_inductance: np.ndarray
    coil_resistance: np.ndarray
    mode_tau: np.ndarray
    mode_coil_coupling: np.ndarray
    section_plasma_flux: np.ndarray
    mode_plasma_flux: np.ndarray
    plasma_times: np.ndarray
    frequencies: np.ndarray
    reduction_error: np.ndarray
    provenance: dict

    @property
    def n_coil(self) -> int:
        """Return the number of coil terminals."""
        return len(self.coil_names)

    @property
    def n_modes(self) -> int:
        """Return the number of passive modes retained."""
        return int(self.mode_tau.size)

    def terminal_impedance(self, frequency_hz: np.ndarray) -> np.ndarray:
        """Reduced plant terminal impedance ``(n_freq, n_coil, n_coil)`` [Ohm]."""
        return _terminal_impedance(
            self.coil_inductance,
            self.coil_resistance,
            self.mode_coil_coupling.T,
            self.mode_tau,
            np.atleast_1d(np.asarray(frequency_hz, dtype=np.float64)),
        )


def coupled_plant(
    machine,
    source=None,
    *,
    n_modes: int | None = None,
    frequencies: np.ndarray | None = None,
    error_tolerance: float = 1e-2,
    cache_dir: Path | str | None = None,
    provenance: dict | None = None,
    refresh: bool = False,
) -> CoupledPlant:
    """Return the coupled plant for ``machine`` driven by ``source``.

    ``machine`` carries the sectioned conductor set and its wiring:
    ``conductors`` (:class:`~nova.circuit.conductor.ConductorSet`),
    ``series`` ``(n_section, n_coil)``, ``coil_names``, ``passive_circuits``,
    ``coil_inductance`` ``(n_coil, n_coil)``, ``coil_resistance`` ``(n_coil,)``
    and ``passive_system`` (:class:`~nova.circuit.passive.PassiveCircuitSystem`).

    ``source`` is the plasma source: ``coil_current_times``, ``coil_currents``
    ``(n_t, n_coil)`` and ``prescribed_currents()`` (one flux-function current
    distribution per ``plasma_times``).  Pass ``source=None`` for the plant
    without a plasma drive; the plasma flux fields are then empty and the
    terminal impedance is unaffected, so a machine's reduction can be measured
    without a plasma source in hand.

    The reduced passive modes are the most coil-relevant, grown until the
    terminal impedance matches the full model to ``error_tolerance`` over
    ``frequencies`` (0.1 Hz to 5 kHz by default) or capped at ``n_modes``.
    """
    conductors: ConductorSet = machine.conductors
    series = np.asarray(machine.series, dtype=np.float64)
    coil_names = list(machine.coil_names)
    passive_circuits = np.asarray(machine.passive_circuits, dtype=np.int64)
    coil_inductance = np.asarray(machine.coil_inductance, dtype=np.float64)
    coil_resistance = np.asarray(machine.coil_resistance, dtype=np.float64)
    system: PassiveCircuitSystem = machine.passive_system

    if frequencies is None:
        frequencies = np.geomspace(0.1, 5000.0, 21)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))

    provenance = dict(provenance or {})
    provenance.setdefault("nova_version", _nova_revision())
    provenance.setdefault("machine", getattr(machine, "label", "unspecified"))
    provenance.setdefault(
        "plasma_source",
        "none" if source is None else getattr(source, "label", "unspecified"),
    )

    directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    digest = _cache_digest(
        machine, source, n_modes, frequencies, error_tolerance, provenance
    )
    directory = directory / digest
    plant_path = directory / "plant.npz"

    if plant_path.exists() and not refresh:
        return load_coupled_plant(plant_path)

    tau, vectors = system.mode_system()
    coupling = system.m_channel.T @ vectors  # (n_coil, n_modes_all)
    keep, error = select_terminal_modes(
        coil_inductance,
        coil_resistance,
        coupling,
        tau,
        frequencies,
        n_modes=n_modes,
        tolerance=error_tolerance,
    )
    mode_tau = tau[keep]
    mode_coil_coupling = coupling[:, keep].T  # (n_modes, n_coil)

    if source is None:
        plasma_times = np.empty(0, dtype=np.float64)
        section_plasma_flux = np.empty((0, series.shape[0]))
        mode_plasma_flux = np.empty((0, mode_tau.size))
    else:
        plasma_times = np.asarray(source.plasma_times, dtype=np.float64)
        currents = source.prescribed_currents()
        circuits, plasma_flux = plasma_linked_flux_history(currents, conductors)
        row_of = {int(circuit): index for index, circuit in enumerate(circuits)}
        section_plasma_flux = plasma_flux[
            :, _section_columns(circuits, conductors, series)
        ]
        passive_columns = [row_of[int(c)] for c in passive_circuits]
        mode_plasma_flux = plasma_flux[:, passive_columns] @ vectors[:, keep]

    plant = CoupledPlant(
        coil_names=coil_names,
        coil_inductance=coil_inductance,
        coil_resistance=coil_resistance,
        mode_tau=mode_tau,
        mode_coil_coupling=mode_coil_coupling,
        section_plasma_flux=section_plasma_flux,
        mode_plasma_flux=mode_plasma_flux,
        plasma_times=plasma_times,
        frequencies=frequencies,
        reduction_error=error,
        provenance=provenance,
    )
    directory.mkdir(parents=True, exist_ok=True)
    save_coupled_plant(plant_path, plant)
    return plant


def _section_columns(circuits, conductors, series) -> list[int]:
    """Column of each winding section in the plasma-flux circuit ordering."""
    row_of = {int(circuit): index for index, circuit in enumerate(circuits)}
    sections = sorted(
        {
            int(circuit)
            for circuit in conductors.circuits
            if int(circuit) < series.shape[0]
        }
    )
    return [row_of[circuit] for circuit in sections]


def _terminal_impedance(
    coil_inductance, coil_resistance, mode_coupling, mode_tau, frequency_hz
) -> np.ndarray:
    """Terminal impedance with the retained passive modes shorted.

    ``mode_coupling`` is ``(n_coil, n_modes)``.  The bare winding plus the
    frequency-dependent screening the retained modes add.  With no modes the
    result is exactly the bare winding.
    """
    omega = 2 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    coil = np.asarray(coil_inductance, dtype=np.float64)
    result = (
        1j * omega[:, None, None] * coil
        + np.diag(np.asarray(coil_resistance, dtype=np.float64))[None, :, :]
    )
    coupling = np.asarray(mode_coupling, dtype=np.float64)
    if coupling.shape[0]:
        rate = 1.0 / np.asarray(mode_tau, dtype=np.float64)
        screened = np.einsum(
            "im,jm,fm->fij",
            coupling,
            coupling,
            1.0 / (rate[None, :] + 1j * omega[:, None]),
            optimize=True,
        )
        result = result + omega[:, None, None] ** 2 * screened
    return result


def _relative_error(reduced, full) -> float:
    scale = np.abs(full)
    scale = np.where(scale > 0, scale, 1.0)
    return float(np.max(np.abs(reduced - full) / scale))


def _reduction_error(
    coil_inductance, coil_resistance, coupling, tau, frequencies, keep=None
) -> np.ndarray:
    """Reduced minus full terminal impedance, relative, per frequency."""
    full = _terminal_impedance(
        coil_inductance, coil_resistance, coupling, tau, frequencies
    )
    if keep is None:
        return np.zeros(frequencies.size)
    reduced = _terminal_impedance(
        coil_inductance, coil_resistance, coupling[:, keep], tau[keep], frequencies
    )
    scale = np.abs(full)
    scale = np.where(scale > 0, scale, 1.0)
    return np.max(np.abs(reduced - full) / scale, axis=(1, 2))


def select_terminal_modes(
    coil_inductance,
    coil_resistance,
    coupling,
    tau,
    frequencies,
    *,
    n_modes: int | None = None,
    tolerance: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Modes whose retention holds the terminal impedance to ``tolerance``.

    Modes are ranked by the strength of their coupling to the coils, so the
    ranking ignores passive detail the terminals cannot see.  Returns
    ``(keep, error)`` -- the retained indices and the per-frequency relative
    error they achieve.  ``keep`` is grown from the strongest coupling until the
    error is within ``tolerance`` over every frequency, or capped at ``n_modes``
    or the full set.  The full set reproduces the model exactly.
    """
    n_all = tau.size
    if n_all == 0:
        return np.argsort(tau)[::-1], np.zeros(frequencies.size)
    if n_modes is None:
        n_modes = n_all
    n_modes = int(np.clip(n_modes, 1, n_all))
    strength = np.linalg.norm(np.asarray(coupling, dtype=np.float64), axis=0)
    order = np.argsort(strength)[::-1]
    best_keep = order[:n_modes]
    best_error = _reduction_error(
        coil_inductance, coil_resistance, coupling, tau, frequencies, best_keep
    )
    # Doubling the retained count on each trial keeps the search logarithmic in
    # the mode count, so a full set of wall circuits is still tractable.
    count = 1
    while True:
        keep = order[:count]
        error = _reduction_error(
            coil_inductance, coil_resistance, coupling, tau, frequencies, keep
        )
        if float(np.max(error)) <= tolerance:
            return keep, error
        if count == n_modes:
            break
        count = min(count * 2, n_modes)
    return best_keep, best_error


def _nova_revision() -> str:
    """The nova revision the plant was built at."""
    try:
        import nova

        return str(getattr(nova, "__version__", "unknown"))
    except Exception:  # pragma: no cover - version lookup is best-effort provenance
        return "unknown"


def _default_cache_dir() -> Path:
    """Cache root resolved through nova's FilePath."""
    from nova.database.filepath import FilePath

    return Path(
        FilePath("user_cache.nova.iter-passive-plasma-circuit.coupled-plant").path
    )


def _cache_digest(
    machine, source, n_modes, frequencies, error_tolerance, provenance
) -> str:
    """A stable digest of everything the cached plant depends on."""
    digest = hashlib.sha256()
    for name in ConductorSet._columns:
        digest.update(np.asarray(getattr(machine.conductors, name)).tobytes())
    digest.update(np.asarray(machine.series).tobytes())
    digest.update(np.asarray(machine.passive_circuits).tobytes())
    digest.update(np.asarray(machine.coil_inductance).tobytes())
    digest.update(np.asarray(machine.coil_resistance).tobytes())
    digest.update(np.asarray(machine.passive_system.lmat).tobytes())
    digest.update(np.asarray(machine.passive_system.r_diag).tobytes())
    digest.update(np.asarray(frequencies).tobytes())
    digest.update(
        np.asarray([n_modes if n_modes is not None else -1, error_tolerance]).tobytes()
    )
    digest.update(json.dumps(provenance, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:20]


def save_coupled_plant(path: Path | str, plant: CoupledPlant) -> None:
    """Persist a coupled plant -- the build is minutes of kernel sums."""
    np.savez_compressed(
        path,
        coil_names=np.array(plant.coil_names),
        coil_inductance=plant.coil_inductance,
        coil_resistance=plant.coil_resistance,
        mode_tau=plant.mode_tau,
        mode_coil_coupling=plant.mode_coil_coupling,
        section_plasma_flux=plant.section_plasma_flux,
        mode_plasma_flux=plant.mode_plasma_flux,
        plasma_times=plant.plasma_times,
        frequencies=plant.frequencies,
        reduction_error=plant.reduction_error,
        provenance=np.frombuffer(
            json.dumps(plant.provenance, default=str).encode(), dtype=np.uint8
        ),
    )


def load_coupled_plant(path: Path | str) -> CoupledPlant:
    """Read back a persisted coupled plant."""
    with np.load(path) as stored:
        return CoupledPlant(
            coil_names=[str(name) for name in stored["coil_names"]],
            coil_inductance=stored["coil_inductance"],
            coil_resistance=stored["coil_resistance"],
            mode_tau=stored["mode_tau"],
            mode_coil_coupling=stored["mode_coil_coupling"],
            section_plasma_flux=stored["section_plasma_flux"],
            mode_plasma_flux=stored["mode_plasma_flux"],
            plasma_times=stored["plasma_times"],
            frequencies=stored["frequencies"],
            reduction_error=stored["reduction_error"],
            provenance=json.loads(stored["provenance"].tobytes()),
        )
