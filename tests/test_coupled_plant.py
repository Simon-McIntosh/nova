"""The coupled plant through one call: cold, cached, reduced and perturbed."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nova.circuit.conductor import ConductorSet
from nova.circuit.coupled_plant import (
    _terminal_impedance,
    coupled_plant,
    load_coupled_plant,
)
from nova.circuit.linkage import circuit_linkage_matrix, ring_resistance
from nova.circuit.passive import NOMINAL_STEEL_RESISTIVITY, PassiveCircuitSystem
from nova.imas.profiles import PrescribedCurrent


@dataclass
class _Machine:
    """A reduced sectioned machine: two coils over four sections, 6 rings."""

    conductors: ConductorSet
    series: np.ndarray = None
    coil_names: list = None
    passive_circuits: np.ndarray = None
    coil_inductance: np.ndarray = None
    coil_resistance: np.ndarray = None
    passive_system: PassiveCircuitSystem = None
    label: str = "reduced-two-coil"


@dataclass
class _Source:
    """A minimal plasma source: an equilibrium's prescribed currents and coils."""

    plasma_times: np.ndarray
    coil_current_times: np.ndarray
    coil_currents: np.ndarray
    _currents: list
    label: str = "reduced-dina"

    def prescribed_currents(self):
        return list(self._currents)


def _reduced_machine() -> _Machine:
    n_section, n_coil, n_passive = 4, 2, 6
    r = np.array([3.0, 3.6, 6.0, 6.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0])
    z = np.array([4.0, 3.2, 4.0, 3.2, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    dr = np.full(10, 0.4)
    dz = np.array([0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    share = np.ones(10)
    circuit = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    conductors = ConductorSet(r, z, dr, dz, share, circuit)

    series = np.zeros((n_section, n_coil))
    series[np.arange(n_section), np.array([0, 0, 1, 1])] = 1.0
    coil_names = ["PFa", "PFb"]

    matrix = circuit_linkage_matrix(conductors, range(10), section_n_max=3)
    passive = np.arange(n_passive) + n_section
    system = PassiveCircuitSystem(
        circuits=passive,
        centroid_r=r[passive],
        centroid_z=z[passive],
        lmat=matrix[n_section:, n_section:],
        r_diag=ring_resistance(conductors, passive, NOMINAL_STEEL_RESISTIVITY),
        a_circuit=np.empty((0, n_passive)),
        g_grid=np.empty((0, n_passive)),
        m_channel=matrix[n_section:, :n_section] @ series,
        channels=coil_names,
        measured_channel_row={},
        resistivity=NOMINAL_STEEL_RESISTIVITY,
        section_scale=conductors.section_scale(passive),
    )
    return _Machine(
        conductors=conductors,
        series=series,
        coil_names=coil_names,
        passive_circuits=passive,
        coil_inductance=series.T @ matrix[:n_section, :n_section] @ series,
        coil_resistance=np.array([1e-4, 1e-4]),
        passive_system=system,
    )


def _source(n_slices: int = 4) -> _Source:
    radius = np.linspace(5.0, 7.0, 9)
    height = np.linspace(-2.0, 2.0, 7)
    grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
    shape = np.exp(-(((grid_r - 6.0) / 1.2) ** 2 + (grid_z / 1.5) ** 2))
    cell_current = 3e5 * shape / shape.sum()
    times = np.linspace(1.0, 4.0, n_slices)
    currents = [
        PrescribedCurrent(
            radius=radius,
            height=height,
            current_density=cell_current,
            cell_current=cell_current,
            boundary=np.array([[5.2, 0.0], [6.8, 0.0]]),
            boundary_source="reduced",
            boundary_fraction=1.0,
            endpoint_gap=0.0,
            scale_factor=1.0,
            centroid=(6.0, 0.0),
            plasma_current=float(cell_current.sum()),
        )
        for _ in range(n_slices)
    ]
    coil_times = np.linspace(0.0, 5.0, 11)
    coil_currents = np.column_stack(
        [1e5 * np.sin(coil_times), 1e5 * np.cos(coil_times)]
    )
    return _Source(
        plasma_times=times,
        coil_current_times=coil_times,
        coil_currents=coil_currents,
        _currents=currents,
    )


def test_cold_and_cached_plants_are_identical(tmp_path):
    machine, source = _reduced_machine(), _source()
    cold = coupled_plant(machine, source, cache_dir=tmp_path)
    cached = coupled_plant(machine, source, cache_dir=tmp_path)
    assert any(tmp_path.glob("*/plant.npz"))
    for field in (
        "coil_inductance",
        "coil_resistance",
        "mode_tau",
        "mode_coil_coupling",
        "section_plasma_flux",
        "mode_plasma_flux",
        "plasma_times",
        "reduction_error",
    ):
        assert np.array_equal(getattr(cold, field), getattr(cached, field)), field
    assert cold.provenance == cached.provenance


def test_reduced_terminal_impedance_matches_full(tmp_path):
    machine, source = _reduced_machine(), _source()
    plant = coupled_plant(machine, source, cache_dir=tmp_path)
    frequencies = np.geomspace(0.1, 5000.0, 21)

    tau, vectors = machine.passive_system.mode_system()
    full_coupling = machine.passive_system.m_channel.T @ vectors
    full = _terminal_impedance(
        plant.coil_inductance, plant.coil_resistance, full_coupling, tau, frequencies
    )
    reduced = plant.terminal_impedance(frequencies)

    scale = np.abs(full)
    scale = np.where(scale > 0, scale, 1.0)
    relative = np.max(np.abs(reduced - full) / scale, axis=(1, 2))
    assert relative.max() < 0.01, relative
    assert np.max(plant.reduction_error) < 0.01


def test_the_cache_is_written_by_the_circuit_tier_serialisers(tmp_path):
    """The call caches through the serialisers nova.circuit.passive ships."""
    from nova.circuit import coupled_plant as plant_module
    from nova.circuit import passive

    assert plant_module.save_coupled_plant is passive.save_coupled_plant
    assert plant_module.load_coupled_plant is passive.load_coupled_plant

    machine, source = _reduced_machine(), _source()
    cold = coupled_plant(machine, source, cache_dir=tmp_path)
    path = next(tmp_path.glob("*/plant.npz"))
    restored = passive.load_coupled_plant(path)
    for field in ("mode_tau", "mode_coil_coupling", "section_plasma_flux"):
        np.testing.assert_array_equal(
            getattr(restored, field), getattr(cold, field)
        )
    assert restored.provenance == cold.provenance


def test_negative_control_perturbed_cache_fails(tmp_path):
    machine, source = _reduced_machine(), _source()
    coupled_plant(machine, source, cache_dir=tmp_path)
    plant_path = next(tmp_path.glob("*/plant.npz"))

    stored = dict(np.load(plant_path))
    stored["mode_tau"] = stored["mode_tau"] * 1.5  # one array's decay times moved
    np.savez_compressed(plant_path, **stored)
    perturbed = load_coupled_plant(plant_path)

    frequencies = np.geomspace(0.1, 5000.0, 21)
    reference = coupled_plant(machine, source, cache_dir=tmp_path / "fresh")
    assert not np.array_equal(perturbed.mode_tau, reference.mode_tau)
    assert not np.allclose(
        perturbed.terminal_impedance(frequencies),
        reference.terminal_impedance(frequencies),
    )


def test_real_sectioned_set_reduction_is_within_one_percent():
    """The real ITER sectioned plant holds 1% per coil from 0.1 Hz to 5 kHz.

    Loads the cached sectioned system and the machine-description wiring rather
    than rebuilding the passive mesh, so the check is seconds rather than the
    minutes the refinement takes.
    """
    import pytest

    root = Path(__file__).resolve().parents[1]
    cache = Path(
        "/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit"
        "/sectioned-conductor-set/68df15665ab2ea45f427"
    )
    if not (cache / "linkage.npy").exists():
        pytest.skip("the sectioned conductor cache is not present on this host")

    spec = importlib.util.spec_from_file_location(
        "iter_sectioned_build", root / "scripts" / "iter_sectioned_conductor_set/build.py"
    )
    build = importlib.util.module_from_spec(spec)
    sys.modules["iter_sectioned_build"] = build
    try:
        spec.loader.exec_module(build)
    except Exception:  # pragma: no cover - the machine description is not local: skip
        pytest.skip("the ITER machine description is not reachable from this host")

    from nova.circuit.coupled_plant import _terminal_impedance, coupled_plant
    from nova.circuit.passive import load_circuit_system

    matrix = np.load(cache / "linkage.npy")
    system = load_circuit_system(cache / "system.npz")
    conductors, coil_names, owner, *_unused = build.source_conductors()
    owner = np.asarray(owner, dtype=np.int64)
    series = np.zeros((owner.size, len(coil_names)))
    series[np.arange(owner.size), owner] = 1.0

    n_section = owner.size
    machine = _Machine(
        conductors=conductors,
        series=series,
        coil_names=list(coil_names),
        passive_circuits=np.asarray(system.circuits, dtype=np.int64),
        coil_inductance=series.T @ matrix[:n_section, :n_section] @ series,
        coil_resistance=np.zeros(series.shape[1]),
        passive_system=system,
        label="iter-sectioned-12mm",
    )

    frequencies = np.geomspace(0.1, 5000.0, 101)
    plant = coupled_plant(machine, None, frequencies=frequencies)

    tau, vectors = system.mode_system()
    coupling = system.m_channel.T @ vectors
    full = _terminal_impedance(
        plant.coil_inductance,
        plant.coil_resistance,
        coupling,
        tau,
        frequencies,
    )
    diag = np.arange(machine.coil_inductance.shape[0])
    error = np.abs(
        plant.terminal_impedance(frequencies)[:, diag, diag] - full[:, diag, diag]
    ) / np.abs(full[:, diag, diag])

    assert error.max() < 1e-2, dict(
        zip(machine.coil_names, np.round(error.max(axis=0), 12))
    )
    assert plant.n_coil == len(machine.coil_names)
