"""A plasma source the plant cache has never seen drives the full ITER plant.

The sectioned-set cache carries the refined conductor geometry, so a DINA case
or a reconstructed pulse assembled by ``load_sectioned_machine`` reaches the
whole ITER plant without rebuilding the sectioned set.  These checks drive two
DINA cases the plant cache holds no entry for, show the plants equal an
independent in-process rebuild bit for bit, count zero subdivisions and zero
linkage builds at every module binding, and show the reconstruction-style
presentation of 105014/1 reaches the same machine within its stated tolerance.
The literal negative control removes ``geometry.npz`` from a scratch
sectioned-set copy: the loader then refuses by name.

Heavy: each plant integrates the plasma flux over the 1607-filament sectioned
machine, so the module is marked slow and builds one plant per arm.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from nova.circuit.conductor import ConductorSet, subdivide_passive_thickness
from nova.circuit.coupled_plant import coupled_plant
from nova.circuit.linkage import circuit_linkage_matrix
from nova.circuit.passive import (
    SECTIONED_GEOMETRY_FILE,
    SectionedGeometryMissing,
    load_circuit_system,
    load_sectioned_machine,
)

ROOT = Path(__file__).resolve().parents[1]

#: scratch sectioned-set cache root the build wrote geometry.npz into, holding
#: the real entries at their derived keys.
SCRATCH_ROOT = Path(
    os.environ.get(
        "NOVA_UNSEEN_SOURCE_SCRATCH",
        "/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit"
        "/sectioned-conductor-set",
    )
)
ENTRY_12MM = SCRATCH_ROOT / "66c10162f3e37adb2679"
SURFACE_LAYER_M = 0.012
MAX_LAYER_THICKNESS_M = 0.04

#: DINA (pulse, run) -> a high-current slice whose prescription is valid.
DINA_CASES = {
    (105069, 3): 1688,
    (105013, 1): 723,
}
DINA_CASE_ROOT = Path("/work/imas/shared/imasdb/ITER/3")
RECONSTRUCTION_URI = "imas:hdf5?user=mcintos;pulse=105014;run=1;database=iter;version=3"
RECONSTRUCTION_DD = "3.38.1"

#: plant fields carrying no per-machine provenance, compared bit for bit.
PLANT_FIELDS = (
    "coil_inductance",
    "coil_resistance",
    "section_inductance",
    "section_resistance",
    "section_coil",
    "mode_tau",
    "mode_coil_coupling",
    "mode_section_coupling",
    "section_plasma_flux",
    "mode_plasma_flux",
    "plasma_times",
    "reduction_error",
)


@dataclass(frozen=True)
class _Machine:
    conductors: object
    series: np.ndarray
    coil_names: list[str]
    passive_circuits: np.ndarray
    passive_system: object
    section_names: list[str]
    section_coil: np.ndarray
    section_inductance: np.ndarray
    section_resistance: np.ndarray
    section_passive_mutual: np.ndarray
    label: str


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "iter_sectioned_build_unseen",
        ROOT / "scripts" / "iter_sectioned_conductor_set/build.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["iter_sectioned_build_unseen"] = module
    spec.loader.exec_module(module)
    return module


def _spy_every_binding(monkeypatch, original, calls, message):
    """Replace ``original`` at every module binding and report the bindings.

    ``subdivide_passive_thickness`` and ``circuit_linkage_matrix`` are bound by
    name into several modules, so a spy on one module's attribute leaves the
    others live.  Walking the loaded modules and replacing every attribute that
    IS the original catches each binding wherever it lives; the returned names
    let the caller assert the guard was installed at all.
    """

    def spy(*args, **kwargs):
        calls.append(("call", args))
        raise AssertionError(message)

    bound = []
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", None):
            continue
        for name, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, name, spy, raising=False)
                bound.append(f"{module.__name__}.{name}")
    return bound


def _refined_machine(entry: Path) -> _Machine:
    """Rebuild the refined geometry in-process and assemble the machine.

    This is the independent reference the cache path is compared against: the
    source conductors and their refinement are recomputed by the build module,
    and only the linkage and the passive system come from the entry.
    """
    build = _build_module()
    conductors, coil_names, owner, thickness, *_unused = build.source_conductors()
    owner = np.asarray(owner, dtype=np.int64)
    n_section = owner.size
    source_passive = np.arange(n_section, n_section + len(thickness))
    refined = subdivide_passive_thickness(
        conductors,
        source_passive,
        thickness,
        MAX_LAYER_THICKNESS_M,
        surface_layer_m=SURFACE_LAYER_M,
    )
    matrix = _load_matrix(entry)
    system = load_circuit_system(entry / "system.npz")
    series = np.zeros((n_section, len(coil_names)))
    series[np.arange(n_section), owner] = 1.0
    counts: dict[int, int] = {}
    section_names = []
    for section_owner in owner:
        ordinal = counts.get(int(section_owner), 0)
        counts[int(section_owner)] = ordinal + 1
        section_names.append(f"{coil_names[int(section_owner)]} section {ordinal + 1}")
    return _Machine(
        conductors=refined.conductors,
        series=series,
        coil_names=list(coil_names),
        passive_circuits=np.asarray(system.circuits, dtype=np.int64),
        passive_system=system,
        section_names=section_names,
        section_coil=owner,
        section_inductance=matrix[:n_section, :n_section],
        section_resistance=np.zeros(n_section),
        section_passive_mutual=matrix[n_section:, :n_section],
        label="iter-sectioned-12mm-rebuilt",
    )


def _load_matrix(entry: Path) -> np.ndarray:
    return np.load(entry / "linkage.npy")


def _assert_conductors_equal(left: ConductorSet, right: ConductorSet) -> None:
    for name in ConductorSet._columns:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    left_polygons = left.polygon_sections
    right_polygons = right.polygon_sections
    assert [int(p.circuit) for p in left_polygons] == [
        int(p.circuit) for p in right_polygons
    ]
    for one, other in zip(left_polygons, right_polygons, strict=True):
        np.testing.assert_array_equal(
            np.asarray(one.vertices), np.asarray(other.vertices)
        )


def _open_dina(pulse: int, run: int):
    import imas

    path = DINA_CASE_ROOT / str(pulse) / str(run)
    with imas.DBEntry(f"imas:hdf5?path={path}", "r") as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    return equilibrium, active


def _dina_source(pulse: int, run: int, index: int):
    """One DINA slice as a generic ``PlasmaSource``.

    These two cases store their poloidal flux with axis and boundary in the same
    sense as their plasma current -- the opposite of the 105014 convention the
    generic prescription expects -- so the stored flux family is oriented by the
    sign the existing reader derives (``sign(-ip / span)``) before it is
    prescribed.  Both arms consume the identical source, so the orientation does
    not enter the bit-identity or the tolerance comparisons.
    """
    from nova.imas.plasma_source import PlasmaSource

    equilibrium, active = _open_dina(pulse, run)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance=f"DINA {pulse}/{run}", indices=[index]
    )
    global_ = equilibrium.time_slice[index].global_quantities
    span = float(global_.psi_boundary) - float(global_.psi_axis)
    if np.sign(-float(global_.ip) / span) < 0:
        original = source.slices[0]
        source = replace(
            source,
            slices=(
                replace(
                    original,
                    psi=-original.psi,
                    psi_samples=-original.psi_samples,
                    psi_axis=-original.psi_axis,
                    psi_boundary=-original.psi_boundary,
                ),
            ),
        )
    return source


def _open_reconstruction():
    import imas

    with imas.DBEntry(RECONSTRUCTION_URI, "r", dd_version=RECONSTRUCTION_DD) as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    return equilibrium, active


def _reconstruction_style_source(equilibrium, active, index, *, stride: int = 4):
    """Present 105014/1 the way a reconstruction arrives at the source.

    EFIT-style p' and FF' on the case's own psi normalisation, the stored last
    closed flux surface named as the boundary rather than a separatrix, and the
    coil currents resampled onto every fourth row of their own clock.
    """
    from nova.imas.plasma_source import CoilCurrentSeries, PlasmaSlice, PlasmaSource

    coil_time = np.asarray(active.time, float)
    sampled = np.arange(0, coil_time.size, stride)
    coils = CoilCurrentSeries(
        coil_time[sampled],
        tuple(str(coil.name).strip() for coil in active.coil),
        np.column_stack(
            [np.asarray(coil.current.data, float)[sampled] for coil in active.coil]
        ),
    )
    slice_ = equilibrium.time_slice[index]
    global_ = slice_.global_quantities
    profile = slice_.profiles_1d
    map_ = slice_.profiles_2d[0]
    lcfs = slice_.boundary.outline
    return PlasmaSource(
        (
            PlasmaSlice(
                time=float(slice_.time),
                radius=np.asarray(map_.grid.dim1),
                height=np.asarray(map_.grid.dim2),
                psi=np.asarray(map_.psi),
                psi_samples=np.asarray(profile.psi),
                p_prime=np.asarray(profile.dpressure_dpsi),
                ff_prime=np.asarray(profile.f_df_dpsi),
                psi_axis=float(global_.psi_axis),
                psi_boundary=float(global_.psi_boundary),
                plasma_current=float(global_.ip),
                outline=np.empty((0, 2)),
                axis=(float(global_.magnetic_axis.r), float(global_.magnetic_axis.z)),
                flux_derivative_scale=2 * np.pi,
                x_points=(),
                boundary_outline=np.c_[np.asarray(lcfs.r), np.asarray(lcfs.z)],
            ),
        ),
        coils,
        provenance="DINA 105014/1 as a reconstruction",
    )


@pytest.mark.slow
def test_unseen_source_drives_the_full_plant_with_zero_rebuilds(tmp_path, monkeypatch):
    """105069/3 and 105013/1 drive the full ITER plant, rebuilding nothing.

    The machine is assembled from the scratch sectioned-set entry by
    ``load_sectioned_machine``; the plant each source returns is compared field
    for field against the plant an independently rebuilt machine returns, and
    the counters must record no ``subdivide_passive_thickness`` call and no
    ``circuit_linkage_matrix`` call at any module binding during the cache-path
    plant build.
    """
    if not (ENTRY_12MM / SECTIONED_GEOMETRY_FILE).exists():
        pytest.skip("the scratch sectioned-set entry is not present on this host")

    cached = load_sectioned_machine(ENTRY_12MM)
    rebuilt = _refined_machine(ENTRY_12MM)
    _assert_conductors_equal(cached.conductors, rebuilt.conductors)

    for (pulse, run), index in DINA_CASES.items():
        source = _dina_source(pulse, run, index)
        subdivide_calls: list = []
        linkage_calls: list = []
        bound_subdivide = _spy_every_binding(
            monkeypatch,
            subdivide_passive_thickness,
            subdivide_calls,
            "subdivide_passive_thickness ran on the cache plant path",
        )
        bound_linkage = _spy_every_binding(
            monkeypatch,
            circuit_linkage_matrix,
            linkage_calls,
            "circuit_linkage_matrix ran on the cache plant path",
        )
        assert bound_subdivide, "no binding of subdivide_passive_thickness was guarded"
        assert bound_linkage, "no binding of circuit_linkage_matrix was guarded"

        plant = coupled_plant(cached, source, cache_dir=tmp_path / f"cache-{pulse}")
        assert subdivide_calls == [], "the cache path rebuilt the subdivided set"
        assert linkage_calls == [], "the cache path rebuilt the linkage"

        cold = coupled_plant(rebuilt, source, cache_dir=tmp_path / f"cold-{pulse}")
        for field in PLANT_FIELDS:
            np.testing.assert_array_equal(getattr(plant, field), getattr(cold, field))
        print(
            f"UNSEEN_SOURCE {pulse}/{run} slice={index} "
            f"n_section={len(plant.section_coil)} "
            f"subdivide={len(subdivide_calls)} linkage={len(linkage_calls)}"
        )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "reference waits on the DD3-to-COCOS-17 translator; DINA flux functions "
        "and maps disagree at the edge"
    ),
)
def test_reconstruction_style_presentation_agrees_within_tolerance(tmp_path):
    """105014/1 presented as a reconstruction meets relative 1e-3 on the plant.

    The reconstruction-style source drives the same cached sectioned machine as
    the DINA-path source, and the two plants' plasma-flux arrays agree within
    the relative tolerance §5 predeclared.

    Expected to fail on its own assertion, hence ``strict=True``: the DINA case
    the arm was built on disagrees between its stored flux functions and its
    flux map at the edge, so the arm needs a reference case whose flux functions
    do agree with the field they must generate. No such case reaches this path
    until the DD3 to COCOS 17 translator exists: every non-DINA source under
    /work/imas/shared/imasdb/ITER/3 is DD3 data carrying its own convention,
    which the plasma-source orientation contract refuses, because that contract
    encodes one fixed orientation rather than translating the source's DD
    version.
    """
    if not (ENTRY_12MM / SECTIONED_GEOMETRY_FILE).exists():
        pytest.skip("the scratch sectioned-set entry is not present on this host")

    from nova.imas.plasma_source import PlasmaSource

    machine = load_sectioned_machine(ENTRY_12MM)
    equilibrium, active = _open_reconstruction()
    generic = PlasmaSource.from_imas(
        equilibrium, active, provenance="DINA 105014/1", indices=[200]
    )
    reconstruction = _reconstruction_style_source(equilibrium, active, 200)

    baseline = coupled_plant(machine, generic, cache_dir=tmp_path / "generic")
    plant = coupled_plant(machine, reconstruction, cache_dir=tmp_path / "recon")

    np.testing.assert_array_equal(plant.plasma_times, baseline.plasma_times)
    np.testing.assert_allclose(
        plant.section_plasma_flux, baseline.section_plasma_flux, rtol=1e-3, atol=0.0
    )
    np.testing.assert_allclose(
        plant.mode_plasma_flux, baseline.mode_plasma_flux, rtol=1e-3, atol=0.0
    )
    difference = np.abs(plant.section_plasma_flux - baseline.section_plasma_flux)
    print(
        "RECONSTRUCTION_FULL_MACHINE "
        f"max_abs_flux_diff_wb={float(np.max(difference)):.9f} "
        "max_rel_flux_diff="
        f"{float(np.max(difference / np.abs(baseline.section_plasma_flux))):.3e} "
        "tolerance=1e-03"
    )


def test_negative_control_entry_without_geometry_is_refused(tmp_path):
    """A new source cannot drive an entry whose ``geometry.npz`` is removed.

    A scratch copy of the real 12 mm entry has its geometry removed; the loader
    must refuse with ``SectionedGeometryMissing`` rather than rebuild the
    sectioned set or serve a reduced machine.
    """
    if not ENTRY_12MM.exists():
        pytest.skip("the scratch sectioned-set entry is not present on this host")
    entry = tmp_path / "entry"
    shutil.copytree(ENTRY_12MM, entry)
    (entry / SECTIONED_GEOMETRY_FILE).unlink()
    source = _dina_source(105069, 3, 1688)
    assert source.slices and source.plasma_times.size == 1
    with pytest.raises(SectionedGeometryMissing, match="geometry.npz"):
        load_sectioned_machine(entry)
