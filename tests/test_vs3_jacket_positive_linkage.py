"""The ITER coil and passive linkage must store positive energy.

The cached plant's coil Schur complement ``L - M^T M`` is indefinite: the eight
VS3 steel jackets screen more flux than the vertical-coil inductance holds.  Two
defects are separated here.  The first is geometry: ``build.py`` collapses each
annular conductor cross-section to an area-equivalent square at the coil centre,
so the jacket square (22.97 mm) lies inside the conductor square (27.56 mm) it
should surround -- the machine description puts the jacket annulus at
26.5-29.5 mm around a 16.95-23.0 mm conductor.  The second is the raw passive
block's own negatives, which lie in the vessel rows and are independent of the
jackets.

The reduced geometry fixture below reads only the two vertical-coil sections and
the eight jacket rows, so the attribution runs in seconds rather than rebuilding
the whole sectioned set.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nova.circuit.conductor import (
    ConductorSet,
    annulus_sector_columns,
    annulus_sector_vertices,
)
from nova.circuit.screening import plasma_self_inductance
from scripts.iter_sectioned_conductor_set.build import (
    ANNULUS_SECTORS,
    source_conductors,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
#: the sectioned-set root the entry arms read.  A fenced worker cannot write the
#: shared cache, so a build under test names its own root here and the entry arms
#: measure the entries that build produced rather than skipping on an uninstalled
#: one.
ENTRY_ROOT = Path(
    os.environ.get("NOVA_SECTIONED_SET_ROOT", CACHE / "sectioned-conductor-set")
)
PINNED_PLANT = CACHE / "coupled-plant/28148563521d92a2198b/plant.npz"
ENTRIES = {
    "12mm": ENTRY_ROOT / "66c10162f3e37adb2679",
    "6mm": ENTRY_ROOT / "f3a24fdd12eae6348f3a",
}
#: the eight jacket source centroids, upper then lower (r, z) [m]
JACKET_CENTROID = np.array(
    [
        [5.806659, 4.939808],
        [5.863305, 4.911286],
        [5.780324, 4.885989],
        [5.836434, 4.856539],
        [7.550558, -2.500018],
        [7.496333, -2.541098],
        [7.511397, -2.449317],
        [7.456658, -2.490061],
    ]
)


def _vertical_arms():
    """The VSU/VSL conductor circuits and jacket circuits, as build.py reads them."""
    conductors, coil_names, owner, _thickness, _res, passive_name = source_conductors()
    n_section = len(owner)
    jackets = [i for i, n in enumerate(passive_name) if "VS3" in n and "Jacket" in n]
    assert len(jackets) == 8 * ANNULUS_SECTORS, "the eight VS3 jackets must be present"
    vsu, vsl = coil_names.index("VSU"), coil_names.index("VSL")
    sections = [i for i, o in enumerate(owner) if o in (vsu, vsl)]
    series = np.zeros((len(sections), 2))
    for k, i in enumerate(sections):
        series[k, 0 if owner[i] == vsu else 1] = 1.0
    circuits = [int(c) for c in sections] + [n_section + i for i in jackets]
    return conductors, series, circuits


def _schur(conductors, circuits, series, finite_source):
    from nova.circuit.linkage import circuit_linkage_matrix

    full = circuit_linkage_matrix(
        conductors, circuits, section_n_max=6, workers=1, finite_source=finite_source
    )
    n_sec = full.shape[0] - 8 * ANNULUS_SECTORS
    coil = series.T @ full[:n_sec, :n_sec] @ series
    mutual = series.T @ full[:n_sec, n_sec:]
    passive = full[n_sec:, n_sec:]
    schur = coil - mutual @ np.linalg.solve(passive, mutual.T)
    return coil, schur, np.linalg.eigvalsh(schur)


def test_cached_plant_coil_schur_is_indefinite_at_the_vertical_coils():
    """A unit vertical-coil direction stores negative energy in the cached plant."""
    with np.load(PINNED_PLANT) as stored:
        coil = np.asarray(stored["coil_inductance"], float)
        coupling = np.asarray(stored["mode_coil_coupling"], float)
    schur = coil - coupling.T @ coupling
    values = np.linalg.eigvalsh(schur)
    assert int((values < 0).sum()) == 2
    assert values.min() == pytest.approx(-3.8377767548e-08, rel=1e-6)
    _w, vectors = np.linalg.eigh(schur)
    names = [str(n) for n in np.load(PINNED_PLANT)["coil_names"]]
    leading = {names[i] for i in np.argsort(-np.abs(vectors[:, 0]))[:1]}
    assert leading <= {"VSU", "VSL"}


@pytest.mark.parametrize("label", ["12mm", "6mm"])
def test_repaired_passive_block_is_positive_definite(label):
    """The repaired entry's passive block stores positive energy, unfloored.

    Recorded pre-repair evidence, kept as the number the repair had to move:
    the raw passive block of the entry at the retired key held 37 (12 mm) and
    175 (6 mm) negative eigenvalues, with minima -2.527958e-7 H and
    -9.450930e-8 H, none supported on the eight jacket rows.  Those numbers are
    a statement about the retired entry, not about this one; the entry this
    test reads is the repaired build and carries no negative eigenvalue.
    """
    entry = ENTRIES[label]
    matrix = np.load(entry / "linkage.npy")
    system = np.load(entry / "system.npz")
    n_passive = system["lmat"].shape[0]
    n_section = matrix.shape[0] - n_passive
    assert n_section > 0, "the stored linkage lost the finite-source coil block"
    block = matrix[n_section:, n_section:]
    values = np.linalg.eigvalsh(block)
    assert int((values < 0).sum()) == 0, values.min()
    assert values.min() > 0


def test_repaired_reduced_vertical_block_is_positive_definite():
    """The reduced vertical-coil Schur complement stores positive energy.

    Recorded pre-repair evidence, kept as the number the repair had to move:
    with the area-equivalent square representation this reduced block held two
    negative eigenvalues with a smallest value of -1.5106e-8 H.  The annulus
    sectors the build now lays down give a positive-definite block, and the
    separate finite-source rule for the coil self-block is not what makes it so.
    """
    conductors, series, circuits = _vertical_arms()
    _coil, _schur_matrix, values = _schur(conductors, circuits, series, False)
    assert int((values < 0).sum()) == 0, values
    assert values.min() > 0
    _coil, _schur_matrix, hybrid = _schur(conductors, circuits, series, True)
    assert int((hybrid < 0).sum()) == 0, hybrid


#: machine-description cross-sections: VS3 conductor and its steel jacket [m]
CONDUCTOR_ANNULUS = (0.01695, 0.02300)
JACKET_ANNULUS = (0.0265, 0.0295)


def _annulus_objects():
    """The eight VS3 conductor and eight jacket annuli as (key, r, z) records.

    Each sector carries the centroid of its own quadrangle, so an annulus centre
    is the mean of the centres of its sectors, not any one sector row.
    """
    conductors, coil_names, owner, _thickness, _res, passive_name = source_conductors()
    n_section = len(owner)
    vsu, vsl = coil_names.index("VSU"), coil_names.index("VSL")
    coils = []
    for coil_index, label in ((vsu, "VSU"), (vsl, "VSL")):
        rows = np.flatnonzero(owner == coil_index).reshape(-1, ANNULUS_SECTORS)
        for run in rows:
            coils.append(
                (
                    label,
                    float(np.mean(conductors.r[run])),
                    float(np.mean(conductors.z[run])),
                )
            )
    element_rows: dict[str, list[int]] = {}
    for row, name in enumerate(passive_name):
        if "VS3" not in name or "Jacket" not in name:
            continue
        element_rows.setdefault(str(name).split(",")[-1].strip(), []).append(
            n_section + row
        )
    jackets = []
    for label, rows in element_rows.items():
        index = np.asarray(rows)
        jackets.append(
            (
                label,
                float(np.mean(conductors.r[index])),
                float(np.mean(conductors.z[index])),
            )
        )
    assert len(coils) == 8 and len(jackets) == 8
    return coils, jackets


def _sector_vertices(r0, z0, ri, ro, n_sector, k):
    """One annular-sector quadrangle, from the subdivision's own owner."""
    return annulus_sector_vertices(r0, z0, ri, ro, n_sector, k)


def _annulus_ports(records, annulus, n_sector, first_circuit):
    """One annulus as ``n_sector`` equal-share sector circuits, with shapes.

    The sector rows and their shaped sections come from
    :func:`annulus_sector_columns`, the owner of the annulus subdivision, so
    this arm and the sectioned-set build cut an annulus the same way.
    """
    polygons = []
    columns: dict[str, list[float]] = {name: [] for name in ConductorSet._columns}
    circuit = first_circuit
    labels = []
    ri, ro = annulus
    for key, r0, z0 in records:
        rows, sections = annulus_sector_columns(
            r0, z0, ri, ro, 1.0, circuit, n_sector, circuit_stride=1
        )
        polygons.extend(sections)
        for name in ConductorSet._columns:
            columns[name].extend(float(value) for value in rows[name])
        labels.extend((key, circuit + index) for index in range(n_sector))
        circuit += n_sector
    return ConductorSet(**columns, polygon_sections=tuple(polygons)), labels


def _annulus_arm(n_sector=8, workers=4):
    """Coil and jacket port matrices of the IDS-faithful annulus arm.

    Each conductor and jacket is its machine-description annulus, cut into
    ``n_sector`` equal-share sector polygons in separate circuits.  The returned
    linkage entry folds the SOURCE share once, and each sector row carries the
    same share, so the share is already in the matrix twice over -- source and
    observer -- and the port reduction is the plain sum over the annulus's
    sectors.  Weighting again would count the share a third time.
    """
    from nova.circuit.linkage import circuit_linkage_matrix

    coils, jackets = _annulus_objects()
    conductors, labels = _annulus_ports(coils, CONDUCTOR_ANNULUS, n_sector, 0)
    jacket_set, jacket_labels = _annulus_ports(
        jackets, JACKET_ANNULUS, n_sector, len(coils) * n_sector
    )
    combined = _concat(conductors, jacket_set)
    n_circuit = len(labels) + len(jacket_labels)
    matrix = circuit_linkage_matrix(
        combined, range(n_circuit), section_n_max=6, workers=workers
    )
    weight = 1.0
    coil_keys = sorted({key for key, _ in labels})
    jacket_keys = [key for key, _ in jacket_labels[::n_sector]]
    wc = np.zeros((n_circuit, len(coil_keys)))
    wj = np.zeros((n_circuit, len(jacket_keys)))
    for index, (key, _circuit) in enumerate(labels):
        wc[index, coil_keys.index(key)] = weight
    for index, (key, _circuit) in enumerate(jacket_labels):
        wj[len(labels) + index, jacket_keys.index(key)] = weight
    return wc.T @ matrix @ wc, wc.T @ matrix @ wj, wj.T @ matrix @ wj


def _concat(first, second):
    """One ConductorSet holding both, keeping shaped sections and circuit ids."""

    columns = {
        name: np.concatenate([getattr(first, name), getattr(second, name)])
        for name in ConductorSet._columns
    }
    return ConductorSet(
        **columns,
        polygon_sections=(*first.polygon_sections, *second.polygon_sections),
    )


def test_annulus_sector_arm_is_positive_definite_and_does_not_over_screen():
    """The IDS-faithful annulus sectors store positive energy and screen below one."""
    L, M, P = _annulus_arm(n_sector=ANNULUS_SECTORS)
    schur = L - M @ np.linalg.solve(P, M.T)
    values = np.linalg.eigvalsh(schur)
    assert int((values < 0).sum()) == 0, values
    assert values.min() > 0
    lam = np.linalg.eigvalsh(np.linalg.solve(L, M @ np.linalg.solve(P, M.T)))
    assert lam.max() < 1.0
    coupling = np.abs(M / np.sqrt(np.diag(L)[:, None] * np.diag(P)[None, :]))
    assert coupling.max() < 1.0


def test_annulus_sector_arm_reproduces_the_ring_self_inductance():
    """The arm's absolute inductance matches the closed-form ring it discretises."""
    from nova.circuit.linkage import circuit_linkage_matrix

    n = ANNULUS_SECTORS
    conductors, jackets = _annulus_objects()
    single, _labels = _annulus_ports(conductors[:1], CONDUCTOR_ANNULUS, n, 0)
    jacket_only, _jacket_labels = _annulus_ports(jackets[:1], JACKET_ANNULUS, n, n)
    matrix = circuit_linkage_matrix(
        _concat(single, jacket_only), range(2 * n), section_n_max=6, workers=1
    )
    # each sector row and its shaped section already carry the sector share
    # 1/n, so the matrix entries hold the share twice over -- source and
    # observer -- and the port reduction is the plain sum: weighting again
    # would count the share a third time and shrink the arm by 1/n
    weight = np.zeros(2 * n)
    weight[:n] = 1.0
    arm = float(weight @ matrix @ weight)
    radius = conductors[0][1]
    reference = float(
        plasma_self_inductance(radius, np.sqrt(np.prod(CONDUCTOR_ANNULUS)))
    )
    assert arm == pytest.approx(reference, rel=0.03), (arm, reference)


def test_jacket_annulus_lies_outside_the_conductor_it_surrounds():
    """The build places each VS3 jacket annulus outside the conductor it screens.

    Recorded pre-repair evidence, kept as the geometry the repair had to change:
    ``source_conductors`` collapsed each annulus to a square of side sqrt(area)
    which put the jacket's 22.974 mm square inside the conductor's 27.556 mm
    square at the same centre, inverting the machine description's radial
    ordering.
    """
    conductors, _series, circuits = _vertical_arms()
    coils, jackets = _annulus_objects()
    polygons = conductors.polygon_by_circuit()
    conductor_circuit = int(circuits[0])
    jacket_circuit = int(circuits[-8 * ANNULUS_SECTORS])
    _key, conductor_r, conductor_z = coils[0]
    _jkey, jacket_r, jacket_z = jackets[0]
    conductor_vertices = np.linalg.norm(
        np.asarray(polygons[conductor_circuit].vertices)
        - np.array([conductor_r, conductor_z]),
        axis=1,
    )
    jacket_vertices = np.linalg.norm(
        np.asarray(polygons[jacket_circuit].vertices) - np.array([jacket_r, jacket_z]),
        axis=1,
    )
    assert float(conductor_vertices.min()) == pytest.approx(
        CONDUCTOR_ANNULUS[0], abs=1e-4
    )
    assert float(conductor_vertices.max()) == pytest.approx(
        CONDUCTOR_ANNULUS[1], abs=1e-4
    )
    assert float(jacket_vertices.min()) == pytest.approx(JACKET_ANNULUS[0], abs=1e-4)
    assert float(jacket_vertices.max()) == pytest.approx(JACKET_ANNULUS[1], abs=1e-4)
    assert jacket_vertices.min() > conductor_vertices.max()
