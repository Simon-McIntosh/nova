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

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
PINNED_PLANT = CACHE / "coupled-plant/28148563521d92a2198b/plant.npz"
ENTRIES = {
    "12mm": (CACHE / "sectioned-conductor-set/68df15665ab2ea45f427", 37, -2.527958e-07),
    "6mm": (CACHE / "sectioned-conductor-set/06ff06d3eb7a4448f0fe", 175, -9.450930e-08),
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


def _load_build():
    spec = importlib.util.spec_from_file_location(
        "iter_sectioned_build_jacket",
        ROOT / "scripts" / "iter_sectioned_conductor_set/build.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["iter_sectioned_build_jacket"] = module
    spec.loader.exec_module(module)
    return module


def _vertical_arms():
    """The VSU/VSL conductor sections and jacket rows, as build.py reads them."""
    import numpy as np

    builder = _load_build()
    conductors, coil_names, owner, _thickness, _res, passive_name = (
        builder.source_conductors()
    )
    n_section = len(owner)
    jackets = [i for i, n in enumerate(passive_name) if "VS3" in n and "Jacket" in n]
    assert len(jackets) == 8, "the eight VS3 jacket loops must be present"
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
    n_sec = full.shape[0] - 8
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
def test_passive_block_negatives_lie_outside_the_jacket_rows(label):
    """The raw passive block is indefinite independently of the eight jackets."""
    entry, count, floor = ENTRIES[label]
    matrix = np.load(entry / "linkage.npy")
    system = np.load(entry / "system.npz")
    n_passive = system["lmat"].shape[0]
    n_section = matrix.shape[0] - n_passive
    block = matrix[n_section:, n_section:]
    values, vectors = np.linalg.eigh(block)
    negatives = np.flatnonzero(values < 0)
    assert negatives.size == count
    assert values.min() == pytest.approx(floor, rel=1e-5)
    radius = np.asarray(system["centroid_r"], float)
    height = np.asarray(system["centroid_z"], float)
    jacket_rows = {
        int(np.argmin(np.abs(radius - r) + np.abs(height - z)))
        for r, z in JACKET_CENTROID
    }
    peaks = {int(np.argmax(np.abs(vectors[:, k]))) for k in negatives}
    assert peaks.isdisjoint(jacket_rows)


def test_reduced_vertical_block_reproduces_the_two_negatives():
    """The jacket square collapse, as build.py lays it down, is indefinite."""
    conductors, series, circuits = _vertical_arms()
    _coil, _schur_matrix, values = _schur(conductors, circuits, series, False)
    assert int((values < 0).sum()) == 2
    assert values.min() == pytest.approx(-1.51e-08, rel=0.1)
    # the quadrature mismatch (coil block finite-source vs hybrid mutual) is not
    # the cause: unifying the rule moves the smallest eigenvalue by ~2e-9.
    _coil, _schur_matrix, hybrid = _schur(conductors, circuits, series, True)
    assert hybrid.min() == pytest.approx(values.min(), rel=0.01)


#: machine-description cross-sections: VS3 conductor and its steel jacket [m]
CONDUCTOR_ANNULUS = (0.01695, 0.02300)
JACKET_ANNULUS = (0.0265, 0.0295)


def _annulus_objects():
    """The eight VS3 conductor and eight jacket annuli as (key, r, z) records."""
    builder = _load_build()
    conductors, coil_names, owner, _thickness, _res, passive_name = (
        builder.source_conductors()
    )
    n_section = len(owner)
    vsu, vsl = coil_names.index("VSU"), coil_names.index("VSL")
    coils = []
    for i, o in enumerate(owner):
        if o in (vsu, vsl):
            coils.append(
                (
                    "VSU" if o == vsu else "VSL",
                    float(conductors.r[i]),
                    float(conductors.z[i]),
                )
            )
    jackets = []
    for j, name in enumerate(passive_name):
        if "VS3" in name and "Jacket" in name:
            row = int(np.flatnonzero(conductors.circuit == n_section + j)[0])
            jackets.append(
                (
                    name.split(",")[-1].strip(),
                    float(conductors.r[row]),
                    float(conductors.z[row]),
                )
            )
    assert len(coils) == 8 and len(jackets) == 8
    return coils, jackets


def _sector_vertices(r0, z0, ri, ro, n_sector, k):
    """One annular-sector quadrangle: outer arc then inner arc, k = 0..n-1."""
    t0 = 2 * np.pi * k / n_sector
    t1 = 2 * np.pi * (k + 1) / n_sector
    return np.asarray(
        [
            (r0 + ro * np.cos(t0), z0 + ro * np.sin(t0)),
            (r0 + ro * np.cos(t1), z0 + ro * np.sin(t1)),
            (r0 + ri * np.cos(t1), z0 + ri * np.sin(t1)),
            (r0 + ri * np.cos(t0), z0 + ri * np.sin(t0)),
        ],
        float,
    )


def _annulus_ports(records, annulus, n_sector, first_circuit):
    """One annulus as ``n_sector`` equal-share sector circuits, with shapes."""
    from nova.circuit.conductor import ConductorSet, PolygonSection

    polygons = []
    columns = {name: [] for name in ConductorSet._columns}
    circuit = first_circuit
    labels = []
    ri, ro = annulus
    for key, r0, z0 in records:
        for k in range(n_sector):
            vertices = _sector_vertices(r0, z0, ri, ro, n_sector, k)
            polygons.append(PolygonSection(circuit, vertices, 1.0 / n_sector))
            centre = vertices.mean(axis=0)
            span = np.ptp(vertices, axis=0)
            for name, value in zip(
                ConductorSet._columns,
                (
                    centre[0],
                    centre[1],
                    max(span[0], 1e-6),
                    max(span[1], 1e-6),
                    1.0,
                    circuit,
                ),
                strict=True,
            ):
                columns[name].append(value)
            labels.append((key, circuit))
            circuit += 1
    return ConductorSet(**columns, polygon_sections=tuple(polygons)), labels


def _annulus_arm(n_sector=8, workers=4):
    """Coil and jacket port matrices of the IDS-faithful annulus arm.

    Each conductor and jacket is its machine-description annulus, cut into
    ``n_sector`` equal-share sector polygons in separate circuits.  The returned
    linkage entry folds the SOURCE share once, so the physical port reduction --
    the share applied to source and observer alike -- weights each sector by the
    square root of its share.  That weight is what makes the summed arm
    reproduce the closed-form ring self-inductance of one conductor.
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
    weight = 1.0 / np.sqrt(n_sector)
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
    from nova.circuit.conductor import ConductorSet

    columns = {
        name: np.concatenate([getattr(first, name), getattr(second, name)])
        for name in ConductorSet._columns
    }
    return ConductorSet(
        **columns,
        polygon_sections=(*first.polygon_sections, *second.polygon_sections),
    )


def _ring_self(major, ri, ro):
    """Closed-form self-inductance of a thin ring of mean radius ``major`` [H]."""
    rho = np.sqrt(ri * ro)
    return 4e-7 * np.pi * major * (np.log(8 * major / rho) - 2)


def test_annulus_sector_arm_repairs_the_negativity_its_square_leaves():
    """Red: the square collapse.  Green: the IDS-faithful annulus sectors."""
    conductors, series, circuits = _vertical_arms()
    _coil, _schur_matrix, square_values = _schur(conductors, circuits, series, False)
    assert int((square_values < 0).sum()) == 2
    assert square_values.min() < 0

    L, M, P = _annulus_arm(n_sector=8)
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

    conductors, jackets = _annulus_objects()
    single, _labels = _annulus_ports(conductors[:1], CONDUCTOR_ANNULUS, 8, 0)
    jacket_only, _jacket_labels = _annulus_ports(jackets[:1], JACKET_ANNULUS, 8, 8)
    matrix = circuit_linkage_matrix(
        _concat(single, jacket_only), range(16), section_n_max=6, workers=1
    )
    wc = np.zeros((16, 1))
    wc[:8, 0] = 1.0 / np.sqrt(8)
    arm = float((wc.T @ matrix @ wc)[0, 0])
    reference = _ring_self(conductors[0][1], *CONDUCTOR_ANNULUS)
    assert arm == pytest.approx(reference, rel=0.03), (arm, reference)


def test_jacket_square_lies_inside_the_conductor_it_should_surround():
    """The builder's area-equivalent squares invert the IDS radial ordering."""
    conductors, _series, circuits = _vertical_arms()
    section_row = int(np.flatnonzero(conductors.circuit == circuits[0])[0])
    jacket_row = int(np.flatnonzero(conductors.circuit == circuits[-8])[0])
    conductor_side = float(conductors.dr[section_row]) * 1000
    jacket_side = float(conductors.dr[jacket_row]) * 1000
    assert conductor_side == pytest.approx(27.556, abs=0.01)
    assert jacket_side == pytest.approx(22.974, abs=0.01)
    # same centre, and the jacket square is the smaller of the two
    assert float(conductors.r[section_row]) == pytest.approx(
        float(conductors.r[jacket_row])
    )
    assert jacket_side < conductor_side
