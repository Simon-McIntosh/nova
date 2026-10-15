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

The build is minutes of kernel sums, so the result is cached through the circuit
tier's serialisers, :func:`nova.circuit.passive.save_coupled_plant` and its
loader, re-exported here.  The cache location resolves through
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
from nova.circuit.passive import (
    PassiveCircuitSystem,
    load_coupled_plant,
    mode_relevance,
    save_coupled_plant,
    select_modes_to_tolerance,
    terminal_impedance_spectrum,
)
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

    ``section_names`` names the ``n_section`` winding sections, and
    ``section_coil`` ``(n_section,)`` is the coil each section belongs to in its
    series order.  ``section_inductance`` ``(n_section, n_section)``,
    ``section_resistance`` ``(n_section,)`` and ``mode_section_coupling``
    ``(n_modes, n_section)`` are the section-level L, R and mode couplings; the
    terminal fields are their series reduction over ``section_coil``
    (``coil_inductance == series.T . section_inductance . series`` and so on).
    The section level is what a switched-converter model reads to predict the
    internal section voltages of a sectioned winding.

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
    section_names: list[str]
    section_coil: np.ndarray
    section_inductance: np.ndarray
    section_resistance: np.ndarray
    mode_tau: np.ndarray
    mode_coil_coupling: np.ndarray
    mode_section_coupling: np.ndarray
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
    def n_section(self) -> int:
        """Return the number of coil winding sections."""
        return int(self.section_inductance.shape[0])

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

    The machine also carries the sectioned winding's own linkage --
    ``section_names``, ``section_coil`` ``(n_section,)``, ``section_inductance``
    ``(n_section, n_section)``, ``section_resistance`` ``(n_section,)`` and the
    passive-section mutual ``section_passive_mutual`` ``(n_passive, n_section)``
    -- whose series reduction is the terminal fields above.  Omit them and the
    section fields come back empty.

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
    section_names = [str(name) for name in getattr(machine, "section_names", [])]
    section_coil = np.asarray(
        getattr(machine, "section_coil", np.empty(0, dtype=np.int64)), dtype=np.int64
    )
    section_inductance = np.asarray(
        getattr(machine, "section_inductance", np.empty((0, 0))), dtype=np.float64
    )
    section_resistance = np.asarray(
        getattr(machine, "section_resistance", np.empty(0)), dtype=np.float64
    )
    section_passive_mutual = np.asarray(
        getattr(machine, "section_passive_mutual", np.empty((0, 0))), dtype=np.float64
    )

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
    if (
        section_passive_mutual.shape[0] == vectors.shape[0]
        and section_passive_mutual.size
    ):
        mode_section_coupling = (vectors.T @ section_passive_mutual)[keep]
    else:
        mode_section_coupling = np.empty((mode_tau.size, section_inductance.shape[0]))

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
        section_names=section_names,
        section_coil=section_coil,
        section_inductance=section_inductance,
        section_resistance=section_resistance,
        mode_tau=mode_tau,
        mode_coil_coupling=mode_coil_coupling,
        mode_section_coupling=mode_section_coupling,
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
    frequency-dependent screening the retained modes add, screened through the
    owner :func:`~nova.circuit.passive.terminal_impedance_spectrum` in its
    already-reduced-mode form.  With no modes the result is exactly the bare
    winding.
    """
    return terminal_impedance_spectrum(
        coil_inductance,
        frequency_hz=frequency_hz,
        terminal_resistance=coil_resistance,
        mode_coupling=mode_coupling,
        mode_rate=1.0 / np.asarray(mode_tau, dtype=np.float64),
    )


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

    Modes are ranked by the strength of their coupling to the coils through the
    passive tier's relevance owner, so the ranking ignores passive detail the
    terminals cannot see, and the retained set is grown by
    :func:`~nova.circuit.passive.select_modes_to_tolerance` until the terminal
    impedance it leaves is within ``tolerance`` over every frequency, or capped
    at ``n_modes`` or the full set.  Returns ``(keep, error)`` -- the retained
    indices and the per-frequency relative error they achieve.  The full set
    reproduces the model exactly.
    """
    coupling = np.asarray(coupling, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)

    def error_of_keep(keep):
        return _reduction_error(
            coil_inductance, coil_resistance, coupling, tau, frequencies, keep
        )

    return select_modes_to_tolerance(
        mode_relevance(np.ones(tau.size), coupling),
        error_of_keep,
        n_modes=n_modes,
        tolerance=tolerance,
    )


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
    for name in ("section_coil", "section_inductance", "section_resistance"):
        section = getattr(machine, name, None)
        if section is not None:
            digest.update(np.asarray(section).tobytes())
    digest.update(np.asarray(machine.passive_system.lmat).tobytes())
    digest.update(np.asarray(machine.passive_system.r_diag).tobytes())
    digest.update(np.asarray(frequencies).tobytes())
    digest.update(
        np.asarray([n_modes if n_modes is not None else -1, error_tolerance]).tobytes()
    )
    digest.update(json.dumps(provenance, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:20]
