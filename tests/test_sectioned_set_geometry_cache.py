"""The sectioned-set cache carries the refined geometry and one loader drives it.

A sectioned-set entry keeps its linkage and passive system; this module checks
that it also keeps the refined conductor geometry a plasma source's flux is read
at, that one loader assembles the machine from it, and that an entry without the
geometry is refused by name rather than silently served.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nova.circuit.conductor import (
    ConductorSet,
    PolygonSection,
    load_conductor_geometry,
    subdivide_passive_thickness,
)
from nova.circuit.coupled_plant import coupled_plant
from nova.circuit.linkage import circuit_linkage_matrix, ring_resistance
from nova.circuit.passive import (
    SECTIONED_GEOMETRY_FILE,
    NOMINAL_STEEL_RESISTIVITY,
    PassiveCircuitSystem,
    SectionedGeometryMissing,
    load_coupled_plant,
    load_sectioned_machine,
    save_circuit_system,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_ENTRY = Path(
    "/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit"
    "/sectioned-conductor-set/68df15665ab2ea45f427"
)
PINNED_PLANT = Path(
    "/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit"
    "/coupled-plant/28148563521d92a2198b/plant.npz"
)
REFINEMENT_MAX_THICKNESS_M = 0.04


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "iter_sectioned_build_geometry",
        ROOT / "scripts" / "iter_sectioned_conductor_set/build.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["iter_sectioned_build_geometry"] = module
    spec.loader.exec_module(module)
    return module


def _small_source():
    """Two coil sections over a two-element passive wall with a polygon arm."""
    wall = np.array([[7.8, -0.05], [8.6, -0.05], [8.6, 0.05], [7.8, 0.05]])
    conductors = ConductorSet(
        r=[3.0, 3.6, 8.2, 9.0],
        z=[2.0, 1.0, 0.05, -1.0],
        dr=[0.4, 0.4, 0.8, 0.3],
        dz=[0.6, 0.6, 0.1, 0.2],
        current_share=[1.0, 1.0, 1.0, 0.5],
        circuit=[0, 1, 2, 3],
        polygon_sections=(PolygonSection(2, wall),),
    )
    thickness = np.array([0.10, 0.08])
    passive = np.array([2, 3])
    return conductors, passive, thickness


def _refine_small():
    conductors, passive, thickness = _small_source()
    return subdivide_passive_thickness(
        conductors,
        passive,
        thickness,
        REFINEMENT_MAX_THICKNESS_M,
        surface_layer_m=0.02,
    )


def _write_entry(
    entry: Path, refined, coil_names, section_coil, *, with_geometry: bool
):
    """Write a sectioned-set entry the way the build does."""
    entry.mkdir(parents=True, exist_ok=True)
    n_section = len(section_coil)
    circuits = np.array([*range(n_section), *refined.circuits])
    matrix = circuit_linkage_matrix(refined.conductors, circuits, section_n_max=3)
    np.save(entry / "linkage.npy", matrix)
    series = np.zeros((n_section, len(coil_names)))
    series[np.arange(n_section), section_coil] = 1.0
    save_circuit_system(
        entry / "system.npz",
        PassiveCircuitSystem(
            circuits=refined.circuits,
            centroid_r=refined.conductors.centroids(refined.circuits)[0],
            centroid_z=refined.conductors.centroids(refined.circuits)[1],
            lmat=matrix[n_section:, n_section:],
            r_diag=ring_resistance(
                refined.conductors, refined.circuits, NOMINAL_STEEL_RESISTIVITY
            ),
            a_circuit=np.empty((0, len(refined.circuits))),
            g_grid=np.empty((0, len(refined.circuits))),
            m_channel=matrix[n_section:, :n_section] @ series,
            channels=list(coil_names),
            measured_channel_row={},
            resistivity=NOMINAL_STEEL_RESISTIVITY,
            section_scale=refined.conductors.section_scale(refined.circuits),
        ),
    )
    if with_geometry:
        build = _build_module()
        build.write_geometry(entry, refined, coil_names, section_coil)
    return circuits


def _small_fixture(tmp_path, *, with_geometry: bool = True):
    entry = tmp_path / "entry"
    refined = _refine_small()
    circuits = _write_entry(
        entry, refined, ["PFa", "PFb"], np.array([0, 1]), with_geometry=with_geometry
    )
    return entry, refined, circuits


def _assert_geometry_equal(loaded, rebuild):
    """Every geometry column and the circuit/parent map are bit-identical."""
    for name in ConductorSet._columns:
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded.conductors, name)),
            np.asarray(getattr(rebuild.conductors, name)),
        )
    np.testing.assert_array_equal(loaded.circuits, rebuild.circuits)
    np.testing.assert_array_equal(loaded.parent, rebuild.parent)
    np.testing.assert_array_equal(loaded.resistance_factor, rebuild.resistance_factor)
    left = loaded.conductors.polygon_sections
    right = rebuild.conductors.polygon_sections
    assert [int(p.circuit) for p in left] == [int(p.circuit) for p in right]
    for one, other in zip(left, right, strict=True):
        np.testing.assert_array_equal(
            np.asarray(one.vertices), np.asarray(other.vertices)
        )
        assert one.current_share == other.current_share


def test_geometry_round_trips_bit_identical(tmp_path):
    """The stored refined geometry reloads exactly as a fresh rebuild computes it."""
    entry = tmp_path / "entry"
    refined = _refine_small()
    build = _build_module()
    path = build.write_geometry(entry, refined, ["PFa", "PFb"], np.array([0, 1]))
    assert path is not None, "the build wrote no geometry beside the linkage"
    assert Path(path).exists(), "the geometry file is missing from the entry"

    rebuild = _refine_small()
    loaded = load_conductor_geometry(entry / SECTIONED_GEOMETRY_FILE)
    _assert_geometry_equal(loaded, rebuild)


def test_loader_returns_the_machine_coupled_plant_takes(tmp_path):
    """The loader's machine drives `coupled_plant` and rebuilds the geometry."""
    entry, refined, _circuits = _small_fixture(tmp_path)
    machine = load_sectioned_machine(entry)
    _assert_geometry_equal(machine.subdivision, refined)
    np.testing.assert_array_equal(
        machine.passive_circuits, np.asarray(machine.passive_system.circuits)
    )
    assert machine.section_names == ["PFa section 1", "PFb section 1"]
    np.testing.assert_array_equal(machine.series, np.array([[1.0, 0.0], [0.0, 1.0]]))
    plant = coupled_plant(machine, None, cache_dir=tmp_path / "plant")
    assert plant.n_coil == len(machine.coil_names)
    assert plant.n_section == machine.section_coil.size


def test_entry_without_geometry_is_refused_by_name(tmp_path):
    """An entry holding only linkage.npy and system.npz cannot yield a machine."""
    entry, _refined, _circuits = _small_fixture(tmp_path, with_geometry=False)
    assert not (entry / SECTIONED_GEOMETRY_FILE).exists()
    assert (entry / "linkage.npy").exists() and (entry / "system.npz").exists()
    with pytest.raises(SectionedGeometryMissing, match="geometry.npz"):
        load_sectioned_machine(entry)


def test_pinned_plant_entry_loads_unchanged():
    """The pinned plant power_convertor reads is still loadable as written."""
    if not PINNED_PLANT.exists():
        pytest.skip("the pinned coupled-plant entry is not present on this host")
    plant = load_coupled_plant(PINNED_PLANT)
    # The pinned entry predates the section block, so it carries terminals only.
    assert plant.n_coil == 14
    assert plant.n_section == 0
    assert plant.n_modes > 0


def test_conductor_geometry_reads_from_the_cache(tmp_path, monkeypatch):
    """`evolve.conductor_geometry` loads the entry and never rebuilds it."""
    import scripts.iter_vessel_evolution.evolve as evolve

    entry, refined, _circuits = _small_fixture(tmp_path)
    surface_layer = 0.012
    monkeypatch.setattr(evolve, "CONDUCTOR_CACHE", tmp_path)
    monkeypatch.setattr(evolve, "MESH_KEY", {surface_layer: "entry"})
    calls = []
    from nova.circuit import conductor as conductor_module

    def counted(*args, **kwargs):
        calls.append(args)
        raise AssertionError("subdivide_passive_thickness was called on the load path")

    monkeypatch.setattr(conductor_module, "subdivide_passive_thickness", counted)
    conductors, coil_names, owner, subdivision = evolve.conductor_geometry(
        surface_layer
    )

    assert calls == [], "the geometry was rebuilt instead of loaded"
    assert coil_names == ["PFa", "PFb"]
    np.testing.assert_array_equal(np.asarray(owner), np.array([0, 1]))
    _assert_geometry_equal(subdivision, refined)
    np.testing.assert_array_equal(conductors.r, np.asarray(refined.conductors.r))


def test_real_entry_geometry_round_trips(tmp_path):
    """The real 12 mm entry carries geometry bit-identical to a rebuild."""
    if not (REAL_ENTRY / "linkage.npy").exists():
        pytest.skip("the sectioned conductor cache is not present on this host")
    build = _build_module()
    try:
        conductors, coil_names, owner, thickness, *_unused = build.source_conductors()
    except Exception:  # pragma: no cover - the description is not reachable: skip
        pytest.skip("the ITER machine description is not reachable from this host")

    n_section = len(owner)
    source_passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors,
        source_passive,
        thickness,
        REFINEMENT_MAX_THICKNESS_M,
        surface_layer_m=0.012,
    )

    entry = tmp_path / REAL_ENTRY.name
    entry.mkdir(parents=True)
    for name in ("linkage.npy", "system.npz"):
        (entry / name).write_bytes((REAL_ENTRY / name).read_bytes())
    path = build.write_geometry(entry, refined, coil_names, owner)
    assert path is not None and Path(path).exists(), "no geometry was written"

    machine = load_sectioned_machine(entry)
    _assert_geometry_equal(machine.subdivision, refined)
    np.testing.assert_array_equal(machine.passive_circuits, refined.circuits)
    assert list(machine.coil_names) == list(coil_names)
    assert machine.section_coil.size == n_section
    # The pinned entry is a pre-change one: it holds no geometry file.
    assert not (REAL_ENTRY / SECTIONED_GEOMETRY_FILE).exists()
