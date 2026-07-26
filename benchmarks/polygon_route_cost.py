"""What the exact polygon-section route costs, closed form against quadrature.

The exact treatment of a polygonal conductor section comes two ways --
:func:`nova.biot.polygon.polygon_greens`, which reduces the section integral to a
contour sum and does the remaining angular integral by boundary quadrature, and
:func:`nova.biot.polygonanalytic.polygon_analytic_greens`, which does that
integral in closed form too. Choosing between them is orthogonal to how pairs are
binned, so there are four arrangements to cost, not two:

===================  ==========================================================
arrangement          treatment
===================  ==========================================================
exact-quadrature     the converged boundary rule on every pair (shipped default)
exact-closed         the closed form on every pair
banded-quadrature    converged rule near, reduced rule mid, moment filament far
banded-closed        closed form near, reduced rule mid, moment filament far
===================  ==========================================================

with the bare point filament as the floor everything is measured against.

Each measurement is a subcommand, so each runs in its own interpreter:

``column``
    per-pair cost of one source section against a plasma-grid-like target cloud.
    A whole column, not a spiral sample, because a banded arrangement's cost is a
    MIXTURE -- its per-pair rate is meaningless without the band populations that
    weight it, and those come from the target cloud's shape.
``population`` / ``build-population``
    the band populations themselves, for the idealised sections and for a real
    plasma grid's own cells, so the mixture above can be read rather than inferred.
``band-cost``
    what each band's treatment costs on exactly the pairs the scheme routes to it,
    so a column rate is the population-weighted sum of these and any surprise in a
    column figure has to be explainable here.
``distance`` / ``batch``
    the two exact kernels against target distance at a fixed batch width, and
    against batch width inside the band. Between them these say WHY a column rate
    and a near-band rate disagree: one of the two variables moves the answer and
    the other does not.
``build``
    a real plasma-grid solve through each arrangement, which adds the per-call
    dispatch, the operator composition and the tessellation that a kernel-only
    figure leaves out.
``sweep`` / ``collect``
    ``sweep`` runs every one of the above in a fresh subprocess and appends the
    records to a file; ``collect`` turns that file into the tables and the figure,
    and extends the measured per-pair rates to a projected 2000-cell first build on
    one core, on a host pool and on one device.

One variant per process, first run, no in-process repeat: repeating a biot solve
inside one interpreter warms the allocator and the allocator's free lists and
reports roughly 40% fast, which would invent a speedup the first build never sees.
Module import is resolved BEFORE each timer starts -- it costs of order a second,
three orders above what the point kernel then spends on a whole column, and it is
paid once per process rather than once per pair, so timing it alongside the kernel
would report the same fixed number for every arrangement.

Run it on a compute node. On a shared login node the same call has been observed
to vary five-fold, which is larger than every difference these tables resolve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06
"""Hexagon circumradius of a ~560-cell ITER plasma mesh, at its major radius.

The same section the three-band scheme's acceptance sweeps use
(:mod:`tests.test_biotbandedcoupling`), so the per-pair figures here sit beside
the recorded ones without a shape correction.
"""

LATTICE_CELLS = 2000
"""Target cloud size, as a cell count of the hexagonal tiling it lays out.

Reproduces the 2,339-target column the recorded banded per-column figures used.
"""

BUILD_CELLS = 500
"""``dplasma`` for the plasma-grid build, matching the tracked point baseline.

The tracked figure is a ~560-cell solve of the same first wall; asking for 500
cells of a hexagonal tiling inside it delivers that mesh, so the build times here
compare directly against it.
"""

BUILD_WALL = {"ellip": [4.2, -0.4, 1.25, 4.2]}
"""First wall of the tracked plasma baseline."""

PROJECTED_CELLS = 2000
"""Build size the first-build budget is stated at: its all-to-all pair count."""

HOST_POOL_SCALING = 8.78
"""Measured 16-core scaling of the tiled assembly, for the pool projection.

A measured SCALING applied to a measured single-core rate is a projection, not a
measurement, and is labelled as one wherever it is reported.
"""

DEVICE_RATE = 5.5
"""Steady-state cost per pair of the traced closed form on one H200 [us]."""

DEVICE_QUADRATURE_RATE = 5.3
"""Steady-state cost per pair of the traced boundary quadrature on one H200 [us]."""

DEVICE_COMPILE = {"cold": 101.9, "warm cache": 8.45, "warm evaluator": 0.0}
"""Compile the traced closed form pays, by what is already warm [s].

Cold is a fresh process with no on-disk cache; warm cache is a second process
hitting ``NOVA_COMPILATION_CACHE``; warm evaluator is a later build in the same
process, which retraces nothing.
"""

DEVICE_QUADRATURE_COMPILE = {"cold": 1.36, "warm cache": 0.23, "warm evaluator": 0.0}
"""The same, for the traced boundary quadrature -- two orders below the above."""

FIRST_BUILD_BUDGET = 20.0 * 60.0
"""First-build budget the operator-assembly route is chosen against [s]."""

TRACKED_POINT_BUILD = 0.963
"""Tracked plasma-grid solve through the point kernel [s].

The asv figure for the same first wall and mesh, which repeats the solve inside
one process -- so it is a WARM number, and a fresh-process first build of the same
work is expected to sit above it rather than on it.
"""

RECORDED = {
    "point": 0.42,
    "exact-quadrature": 869.0,
    "exact-closed": 171.4,
    "banded-quadrature": 13.1,
}
"""Per-pair figures already on record [us], for the hexagon, to compare against.

``exact-quadrature`` is the recorded whole-column figure (869 hexagon / 1008
wall-clipped); the 857.8 also on record is the same rule over a spiral sample of
512 targets, which is a different cloud. ``banded-closed`` has no recorded figure
-- it is what this driver adds.
"""


# --- the sections and the target cloud ---------------------------------------


def hexagon(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return the plasma cell section, a regular hexagon of circumradius ``radius``."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def clipped_cell(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return a hexagon with one corner cut by a straight wall.

    Its third moments survive where a regular hexagon's vanish by symmetry, which
    is what pushes its far seam out and so changes the band populations -- and the
    closed form's cost is corner-organised, so the extra corner changes that too.
    """
    corner = list(hexagon(r0, z0, radius))
    return np.array(
        corner[:2]
        + [[r0 - 0.35 * radius, z0 + 0.75 * radius]]
        + [[r0 - 0.95 * radius, z0 + 0.30 * radius]]
        + corner[3:]
    )


SECTIONS = {"hexagon": hexagon, "wall-clipped": clipped_cell}


def hex_lattice(cells=LATTICE_CELLS, radius=CELL_RADIUS):
    """Return the centres of a hexagonal tiling, as a plasma grid lays them out."""
    pitch = np.sqrt(3.0) * radius
    reach = int(np.ceil(np.sqrt(cells / np.pi)))
    span = np.arange(-reach, reach + 1)
    row, column = np.meshgrid(span, span)
    centre_r = R0 + pitch * (column + 0.5 * (row % 2))
    centre_z = Z0 + pitch * np.sqrt(3.0) / 2.0 * row
    keep = np.hypot(centre_r - R0, centre_z - Z0) <= reach * pitch
    return centre_r[keep], centre_z[keep]


# --- the arrangements --------------------------------------------------------

ARRANGEMENTS = {
    "point": {},
    "exact-quadrature": {"banded": False, "closed_form": False},
    "exact-closed": {"banded": False, "closed_form": True},
    "banded-quadrature": {"banded": True, "closed_form": False},
    "banded-closed": {"banded": True, "closed_form": True},
}
"""Keyword arguments each arrangement passes to ``PolySection.configured``.

``point`` is not a ``PolySection`` arrangement at all -- it is the bare
centroid-filament ring, the floor -- and is handled separately.
"""


def column_kernel(arrangement, target_r, target_z, vertices):
    """Return a no-argument callable for one source column, imports already done.

    The imports are resolved HERE and not inside the timed call. Importing
    ``nova.biot.greens`` costs of order a second on a quiet node -- three orders
    above what the point kernel then spends on the whole column -- so timing the
    import alongside the kernel would report the same fixed number for every
    arrangement and hide the very difference being measured. It is a real cost of a
    fresh process, reported separately as the setup time; it is paid once per
    process and not once per pair, so it does not belong in a per-pair rate.

    Everything but ``point`` goes through ``PolySection.section_greens``, the one
    dispatch the shipped element uses, so the figure is the route's cost and not a
    kernel called around it.
    """
    if arrangement == "point":
        from nova.biot.greens import greens_bz_br, greens_psi, section_centroid

        centre = section_centroid(vertices)

        def call():
            """Return the bare centroid-filament column, the cost floor."""
            psi = greens_psi(target_r, target_z, centre[0], centre[1])
            bz, br = greens_bz_br(target_r, target_z, centre[0], centre[1])
            return psi, br, bz

        return call

    from nova.biot.polysection import PolySection

    def call():
        """Return the column through the configured polygon-section arrangement."""
        with PolySection.configured(**ARRANGEMENTS[arrangement]):
            return PolySection.section_greens(target_r, target_z, vertices)

    return call


# --- per-pair cost of one source column --------------------------------------


def _checksum(*parts):
    """Return a summed marker over the finite entries, and how many are not.

    The target cloud puts one target on the section's own centroid, where a bare
    point filament is singular and the polygon kernels are not -- which is the
    reason the polygon treatment exists. Summing only the finite entries keeps the
    marker readable and reports the singular count instead of hiding it.
    """
    values = np.concatenate([np.asarray(part).ravel() for part in parts])
    finite = np.isfinite(values)
    return float(np.sum(values[finite])), int(np.count_nonzero(~finite))


def measure_column(arrangement, section):
    """Return one record: per-pair cost of one section against the target cloud."""
    vertices = SECTIONS[section]()
    target_r, target_z = hex_lattice()
    setup = time.perf_counter()
    call = column_kernel(arrangement, target_r, target_z, vertices)
    setup = time.perf_counter() - setup
    start = time.perf_counter()
    psi, br, bz = call()
    seconds = time.perf_counter() - start
    psi_checksum, singular = _checksum(psi)
    field_checksum, _ = _checksum(br, bz)
    return {
        "measurement": "column",
        "arrangement": arrangement,
        "section": section,
        "pairs": int(target_r.size),
        "seconds": seconds,
        "setup_seconds": setup,
        "us_per_pair": 1e6 * seconds / target_r.size,
        "psi_checksum": psi_checksum,
        "field_checksum": field_checksum,
        "singular": singular,
    }


BANDS = ("near-quadrature", "near-closed", "mid", "far")
"""The treatments a banded arrangement spends its pairs on, costed separately.

Both near kernels appear because which one serves the near band is the choice
being made, and a column rate is the population-weighted sum of these -- so a
column figure that surprises has to be explainable from this table or it is wrong.
"""


def band_kernel(treatment, target_r, target_z, vertices):
    """Return a callable for one band's treatment on the pairs given, imports done.

    The pairs handed in are the ones the scheme would actually route to this band,
    so the figure includes the per-call cost of a batch that small. That is not an
    artefact to be factored out: it is what the arrangement pays. A near band
    holding a few tens of pairs out of thousands amortises a kernel's fixed setup
    over very little, and a kernel with more setup can lose there even while it wins
    per pair on a wide batch.
    """
    from nova.biot.bandedcoupling import MID_RULE, near_greens
    from nova.biot.greens import moment_filament
    from nova.biot.polygon import polygon_greens

    match treatment:
        case "near-quadrature" | "near-closed":
            closed = treatment == "near-closed"
            return lambda: near_greens(target_r, target_z, vertices, closed_form=closed)
        case "mid":
            return lambda: polygon_greens(
                target_r,
                target_z,
                vertices,
                n_panels=MID_RULE[0],
                n_nodes=MID_RULE[1],
            )
        case "far":
            return lambda: moment_filament(target_r, target_z, vertices)
    raise KeyError(treatment)


def measure_band_cost(section, treatment):
    """Return one record: what one band's treatment costs on its own pairs.

    The band assignment comes from the same target cloud the column figures use, so
    the per-band rates here weighted by the populations reconstruct those figures.
    """
    from nova.biot.bandedcoupling import band

    vertices = SECTIONS[section]()
    target_r, target_z = hex_lattice()
    assignment = band(target_r, target_z, vertices)
    index = 1 if treatment == "mid" else 2 if treatment == "far" else 0
    chosen = assignment == index
    setup = time.perf_counter()
    call = band_kernel(treatment, target_r[chosen], target_z[chosen], vertices)
    setup = time.perf_counter() - setup
    start = time.perf_counter()
    psi, br, bz = call()
    seconds = time.perf_counter() - start
    pairs = int(np.count_nonzero(chosen))
    psi_checksum, _ = _checksum(psi)
    return {
        "measurement": "band_cost",
        "section": section,
        "treatment": treatment,
        "pairs": pairs,
        "fraction": pairs / assignment.size,
        "seconds": seconds,
        "setup_seconds": setup,
        "us_per_pair": 1e6 * seconds / pairs,
        "psi_checksum": psi_checksum,
    }


BATCH_WIDTHS = (8, 16, 64, 256, 1024, 4096)
"""Near-band batch widths the two exact kernels are compared over.

The near band of a plasma-grid column holds a few tens of pairs, and a per-pair
rate measured on thousands does not describe that. Both kernels have a fixed cost
per CALL -- rule construction for the quadrature, corner temporaries for the closed
form -- and the two amortise differently, so which one is cheaper is a function of
the batch width and not a constant. Where the two cross is what decides whether the
closed form can serve a near band at all.
"""


def measure_batch(section, treatment, pairs):
    """Return one record: a near kernel's per-pair cost at one batch width.

    The targets sit on a ring inside the near band, at 1.5 section radii of contour
    distance, so every pair is one the near band would own -- widening the batch and
    not the distance.
    """
    vertices = SECTIONS[section]()
    from nova.biot.bandedcoupling import section_radius

    angle = np.linspace(0.0, 2.0 * np.pi, pairs, endpoint=False)
    offset = 1.5 * section_radius(vertices)
    centre = np.asarray(vertices).mean(axis=0)
    target_r = centre[0] + offset * np.cos(angle)
    target_z = centre[1] + offset * np.sin(angle)
    setup = time.perf_counter()
    call = band_kernel(treatment, target_r, target_z, vertices)
    setup = time.perf_counter() - setup
    start = time.perf_counter()
    psi, _, _ = call()
    seconds = time.perf_counter() - start
    psi_checksum, _ = _checksum(psi)
    return {
        "measurement": "batch",
        "section": section,
        "treatment": treatment,
        "pairs": int(pairs),
        "seconds": seconds,
        "setup_seconds": setup,
        "us_per_pair": 1e6 * seconds / pairs,
        "psi_checksum": psi_checksum,
    }


DISTANCE_OFFSETS = (0.2, 0.7, 1.2, 2.2, 4.0, 6.8, 12.0, 30.0)
"""Centroid offsets, in section radii, the exact kernels are costed at.

Includes both of the scheme's seams so the bands can be read off the sweep.
"""

DISTANCE_PAIRS = 256
"""Pairs per ring in the distance sweep -- wide enough that a per-call fixed cost
is not what is being compared, so the sweep isolates DISTANCE."""


def measure_distance(section, treatment, offset, pairs=DISTANCE_PAIRS):
    """Return one record: a kernel's per-pair cost on a ring at one distance.

    The boundary quadrature spends the same fixed node count on every pair whatever
    the distance; the closed form does not -- its two per-corner residuals are graded
    integrals whose difficulty is set by how close the target comes to the corner's
    own edges. So the two kernels' relative cost is a FUNCTION of distance, and a
    single ratio quoted from a target cloud is really a ratio at that cloud's mixture
    of distances. Every ring here carries the same number of pairs, so what varies
    between records is distance alone.
    """
    from nova.biot.bandedcoupling import contour_distance, section_radius

    vertices = SECTIONS[section]()
    radius = section_radius(vertices)
    angle = np.linspace(0.0, 2.0 * np.pi, pairs, endpoint=False)
    centre = np.asarray(vertices).mean(axis=0)
    target_r = centre[0] + offset * radius * np.cos(angle)
    target_z = centre[1] + offset * radius * np.sin(angle)
    contour = contour_distance(target_r, target_z, vertices) / radius
    setup = time.perf_counter()
    call = band_kernel(treatment, target_r, target_z, vertices)
    setup = time.perf_counter() - setup
    start = time.perf_counter()
    psi, _, _ = call()
    seconds = time.perf_counter() - start
    psi_checksum, _ = _checksum(psi)
    return {
        "measurement": "distance",
        "section": section,
        "treatment": treatment,
        "offset": float(offset),
        "contour": float(np.mean(contour)),
        "pairs": int(pairs),
        "seconds": seconds,
        "setup_seconds": setup,
        "us_per_pair": 1e6 * seconds / pairs,
        "psi_checksum": psi_checksum,
    }


def measure_population(section, target_r=None, target_z=None, label=None):
    """Return one record: what fraction of the pairs each band holds.

    This is what turns a per-band cost into a column rate, so a banded
    arrangement's per-pair figure cannot be read without it.
    """
    from nova.biot.bandedcoupling import band, mid_limit, section_skew

    vertices = SECTIONS[section]() if isinstance(section, str) else section
    if target_r is None:
        target_r, target_z = hex_lattice()
    assignment = band(target_r, target_z, vertices)
    fraction = [
        float(np.count_nonzero(assignment == index) / assignment.size)
        for index in (0, 1, 2)
    ]
    return {
        "measurement": "population",
        "section": label or section,
        "pairs": int(assignment.size),
        "cells": 1,
        "corners": int(len(vertices)),
        "skew": float(section_skew(vertices)),
        "far_seam": float(mid_limit(vertices)),
        "near": fraction[0],
        "mid": fraction[1],
        "far": fraction[2],
    }


# --- a real plasma-grid build ------------------------------------------------


def _plasma_coilset(cells=BUILD_CELLS, polysection=False):
    """Return an unsolved coilset of the tracked plasma baseline's first wall.

    Plasma cells ship with ``segment="circle"`` -- the point-filament ring -- so
    routing the build through the polygon-section element means relabelling the
    subframe's own ``segment`` column, which is what ``Solve.generator`` maps to
    an element class. Nothing else about the frame changes: the cells already
    carry their own section polygons.
    """
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dplasma=-cells, tplasma="hex")
    coilset.firstwall.insert(BUILD_WALL, turn="hex", Ic=1e6)
    if polysection:
        subframe = coilset.subframe
        subframe.loc[np.asarray(subframe.plasma), "segment"] = "polysection"
    return coilset


def measure_build(arrangement, cells=BUILD_CELLS):
    """Return one record: wall time of a plasma-grid solve through one arrangement.

    The mesh, the element imports and the grid instance are all resolved before the
    timer starts, so the figure is the operator build -- kernel, per-column
    dispatch, composition and tessellation -- and nothing that precedes it. That is
    also what the tracked point figure measures, which repeats the solve inside one
    warm process; a fresh process still pays each kernel's own first call here,
    which the tracked one does not.
    """
    from nova.biot.polysection import PolySection

    coilset = _plasma_coilset(cells, polysection=arrangement != "point")
    count = int(np.asarray(coilset.subframe.plasma).sum())
    grid = coilset.plasmagrid
    with PolySection.configured(**ARRANGEMENTS.get(arrangement, {})):
        start = time.perf_counter()
        grid.solve()
        seconds = time.perf_counter() - start
    psi = np.asarray(grid.data.Psi)
    return {
        "measurement": "build",
        "arrangement": arrangement,
        "cells": count,
        "pairs": count * count,
        "seconds": seconds,
        "us_per_pair": 1e6 * seconds / (count * count),
        "psi_checksum": float(np.sum(psi)),
    }


def grid_sections(coilset):
    """Return each plasma cell's section vertices, closing vertex dropped.

    The same reduction ``PolySection._section_vertices`` performs, so what is
    measured here is the geometry the element itself sees.
    """
    plasma = np.asarray(coilset.subframe.plasma)
    sections = []
    for poly in np.asarray(coilset.subframe["poly"])[plasma]:
        vertices = np.asarray(poly.points, dtype=np.float64)[:, [0, 2]]
        if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        sections.append(vertices)
    return sections


def measure_build_population(cells=BUILD_CELLS):
    """Return the real grid's band populations, aggregated over every source column.

    Not a sampled cell: the build's cost is the sum over all of them, and a real
    plasma grid is not one repeated section. Its interior is a regular hexagonal
    tiling, but the wall cuts the boundary ring into polygons of three to twelve
    corners including slivers two orders below a cell in radius -- and both of the
    things that set cost depend on that. The closed form is organised BY CORNER, so
    a twelve-corner cell costs about twice a hexagon; and the far seam is set by the
    section's own skew, so a cut cell keeps far more pairs on the mid rule than a
    symmetric one does. Reported alongside is the idealised-section population, so
    the difference between the two is visible rather than assumed.
    """
    from nova.biot.bandedcoupling import (
        MID_LIMIT,
        band,
        mid_limit,
        section_radius,
        section_skew,
    )

    coilset = _plasma_coilset(cells, polysection=True)
    plasma = np.asarray(coilset.subframe.plasma)
    target_r = np.asarray(coilset.subframe.x)[plasma]
    target_z = np.asarray(coilset.subframe.z)[plasma]
    sections = grid_sections(coilset)
    counts = np.zeros(3, dtype=np.int64)
    corners, skew, seam, radius = [], [], [], []
    for vertices in sections:
        assignment = band(target_r, target_z, vertices)
        for index in range(3):
            counts[index] += int(np.count_nonzero(assignment == index))
        corners.append(len(vertices))
        skew.append(section_skew(vertices))
        seam.append(mid_limit(vertices))
        radius.append(section_radius(vertices))
    fraction = counts / counts.sum()
    corners = np.asarray(corners)
    skew = np.asarray(skew)
    return [
        {
            "measurement": "build_population",
            "section": f"{len(sections)}-cell plasma grid",
            "pairs": int(counts.sum()),
            "cells": len(sections),
            "corners": int(np.median(corners)),
            "corner_range": [int(corners.min()), int(corners.max())],
            "six_corner_cells": int(np.count_nonzero(corners == 6)),
            "radius_range": [float(min(radius)), float(max(radius))],
            "skew": float(np.median(skew)),
            "wide_seam_cells": int(np.count_nonzero(np.asarray(seam) > MID_LIMIT)),
            "far_seam": float(np.median(seam)),
            "near": float(fraction[0]),
            "mid": float(fraction[1]),
            "far": float(fraction[2]),
        }
    ]


# --- collection, tables, projection -----------------------------------------


def _median(records, key="us_per_pair"):
    """Return the median and the observed range over repeated processes."""
    values = sorted(record[key] for record in records)
    return float(np.median(values)), (values[0], values[-1])


def _group(records, measurement, *keys):
    """Return records of one measurement grouped by the named fields."""
    grouped: dict[tuple, list] = {}
    for record in records:
        if record.get("measurement") != measurement:
            continue
        grouped.setdefault(tuple(record[key] for key in keys), []).append(record)
    return grouped


def read_records(path):
    """Return the JSON records in a file, one per line, blank lines ignored."""
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def column_table(records):
    """Return the per-pair cost table as markdown rows."""
    grouped = _group(records, "column", "arrangement", "section")
    lines = [
        "| arrangement | hexagon us/pair | wall-clipped us/pair | recorded hexagon |",
        "|---|---|---|---|",
    ]
    for arrangement in ARRANGEMENTS:
        cells = []
        for section in SECTIONS:
            found = grouped.get((arrangement, section))
            if not found:
                cells.append("--")
                continue
            median, span = _median(found)
            cells.append(f"{median:.1f} ({span[0]:.1f}-{span[1]:.1f}, n={len(found)})")
        recorded = RECORDED.get(arrangement)
        lines.append(
            f"| {arrangement} | {cells[0]} | {cells[1]} | "
            f"{'--' if recorded is None else f'{recorded:.1f}'} |"
        )
    return "\n".join(lines)


def population_rows(records):
    """Return one population record per section, idealised first then the real grid."""
    rows: list[dict] = []
    for measurement in ("population", "build_population"):
        for record in records:
            if record.get("measurement") != measurement:
                continue
            if record["section"] not in [row["section"] for row in rows]:
                rows.append(record)  # pure geometry: a repeated process repeats it
    return rows


def population_table(records):
    """Return the band-population table as markdown rows."""
    lines = [
        "| section | corners | skew | far seam | near | mid | far |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in population_rows(records):
        corners = str(row["corners"])
        if row.get("corner_range") and row["corner_range"][0] != row["corner_range"][1]:
            corners = (
                f"{row['corner_range'][0]}-{row['corner_range'][1]} "
                f"(median {row['corners']})"
            )
        lines.append(
            f"| {row['section']} | {corners} | {row['skew']:.1e} | "
            f"{row['far_seam']:.1f} | {row['near']:.2%} | {row['mid']:.2%} | "
            f"{row['far']:.2%} |"
        )
    return "\n".join(lines)


def band_cost_table(records):
    """Return the per-band cost table, and each band's share of the column."""
    grouped = _group(records, "band_cost", "treatment", "section")
    lines = [
        "| treatment | pairs | hexagon us/pair | wall-clipped us/pair | "
        "hexagon share of column | wall-clipped share |",
        "|---|---|---|---|---|---|",
    ]
    for treatment in BANDS:
        cells, shares, pairs = [], [], []
        for section in SECTIONS:
            found = grouped.get((treatment, section))
            if not found:
                cells.append("--")
                shares.append("--")
                continue
            median, span = _median(found)
            cells.append(f"{median:.1f} ({span[0]:.1f}-{span[1]:.1f})")
            shares.append(f"{1e-6 * median * found[0]['pairs']:.4f} s")
            pairs.append(str(found[0]["pairs"]))
        lines.append(
            f"| {treatment} | {'/'.join(dict.fromkeys(pairs))} | {cells[0]} | "
            f"{cells[1]} | {shares[0]} | {shares[1]} |"
        )
    return "\n".join(lines)


def _batch_crossover(grouped):
    """Return the narrowest measured batch at which the closed form is not dearer.

    The width the two kernels swap places at, to within the sweep's own spacing --
    the number that decides whether a band can be served by the closed form.
    """
    for width in sorted({key[1] for key in grouped}):
        rates = [
            _median(grouped[(treatment, width)])[0]
            for treatment in ("near-quadrature", "near-closed")
            if (treatment, width) in grouped
        ]
        if len(rates) == 2 and rates[1] <= rates[0]:
            return width
    return None


def batch_table(records):
    """Return the near-kernel batch-width table, and where the two cross."""
    grouped = _group(records, "batch", "treatment", "pairs")
    widths = sorted({key[1] for key in grouped})
    lines = [
        "| pairs in the call | quadrature us/pair | closed form us/pair | ratio |",
        "|---|---|---|---|",
    ]
    for width in widths:
        rates = []
        for treatment in ("near-quadrature", "near-closed"):
            found = grouped.get((treatment, width))
            rates.append(_median(found)[0] if found else None)
        if None in rates:
            continue
        lines.append(
            f"| {width} | {rates[0]:.1f} | {rates[1]:.1f} | "
            f"{rates[0] / rates[1]:.2f}x |"
        )
    return "\n".join(lines)


def distance_table(records):
    """Return the cost-against-distance table for the two exact kernels."""
    grouped = _group(records, "distance", "treatment", "offset")
    offsets = sorted({key[1] for key in grouped})
    lines = [
        "| centroid offset / a | contour distance / a | quadrature us/pair | "
        "closed form us/pair | closed / quadrature |",
        "|---|---|---|---|---|",
    ]
    for offset in offsets:
        rates, contour = [], None
        for treatment in ("near-quadrature", "near-closed"):
            found = grouped.get((treatment, offset))
            rates.append(_median(found)[0] if found else None)
            if found:
                contour = found[0]["contour"]
        if None in rates:
            continue
        lines.append(
            f"| {offset:.1f} | {contour:.2f} | {rates[0]:.1f} | {rates[1]:.1f} | "
            f"{rates[1] / rates[0]:.2f} |"
        )
    return "\n".join(lines)


def build_table(records):
    """Return the plasma-grid build table as markdown rows."""
    grouped = _group(records, "build", "arrangement")
    point = grouped.get(("point",))
    baseline = _median(point, "seconds")[0] if point else None
    lines = [
        "| arrangement | build s | us/pair | x point build |",
        "|---|---|---|---|",
    ]
    for arrangement in ARRANGEMENTS:
        found = grouped.get((arrangement,))
        if not found:
            continue
        median, span = _median(found, "seconds")
        rate = _median(found)[0]
        ratio = "1" if baseline is None else f"{median / baseline:.0f}"
        lines.append(
            f"| {arrangement} | {median:.2f} ({span[0]:.2f}-{span[1]:.2f}, "
            f"n={len(found)}) | {rate:.1f} | {ratio} |"
        )
    return "\n".join(lines)


def projection(records, cells=PROJECTED_CELLS):
    """Return the projected first-build table for a square all-to-all operator.

    One core is the measured column rate. The host pool divides it by the measured
    tiled scaling, which makes it a PROJECTION and not a measurement. The device
    column exists only for the two arrangements that have a traced tile kernel:
    the banded arrangements bin pairs into three shapes per section and no traced
    kernel does that, so a device figure for them would be invented.
    """
    pairs = cells * cells
    grouped = _group(records, "column", "arrangement", "section")
    device = {
        "exact-closed": (DEVICE_RATE, DEVICE_COMPILE),
        "exact-quadrature": (DEVICE_QUADRATURE_RATE, DEVICE_QUADRATURE_COMPILE),
    }
    rows = []
    for arrangement in ARRANGEMENTS:
        found = grouped.get((arrangement, "hexagon"))
        if not found:
            continue
        rate = _median(found)[0]
        one_core = 1e-6 * rate * pairs
        row = {
            "arrangement": arrangement,
            "us_per_pair": rate,
            "one core": one_core,
            "host pool": one_core / HOST_POOL_SCALING,
        }
        if arrangement in device:
            rate, compile_cost = device[arrangement]
            for state, seconds in compile_cost.items():
                row[f"device {state}"] = 1e-6 * rate * pairs + seconds
        rows.append(row)
    return {"measurement": "projection", "cells": cells, "pairs": pairs, "rows": rows}


def projection_table(projected):
    """Return the projection as markdown rows, with the budget verdict per route."""
    columns = ["one core", "host pool", "device cold", "device warm cache"]
    lines = [
        "| arrangement | us/pair | " + " | ".join(columns) + " | inside budget |",
        "|---|---|" + "---|" * (len(columns) + 1),
    ]
    for row in projected["rows"]:
        cells = [
            "--" if row.get(name) is None else f"{row[name] / 60.0:.2f} min"
            for name in columns
        ]
        inside = [
            name
            for name in columns
            if row.get(name) is not None and row[name] <= FIRST_BUILD_BUDGET
        ]
        lines.append(
            f"| {row['arrangement']} | {row['us_per_pair']:.1f} | "
            + " | ".join(cells)
            + f" | {', '.join(inside) if inside else 'none'} |"
        )
    return "\n".join(lines)


# --- the figure --------------------------------------------------------------


def figure(records, path):
    """Write the four-panel cost figure: per pair, by band, by build, projected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplot_mosaic(
        [
            ["per_pair", "population", "distance", "batch"],
            ["build", "projection", "summary", "summary"],
        ],
        figsize=(23.0, 10.5),
    )
    _panel_per_pair(axes["per_pair"], records)
    _panel_population(axes["population"], records)
    _panel_distance(axes["distance"], records)
    _panel_batch(axes["batch"], records)
    _panel_build(axes["build"], records)
    _panel_projection(axes["projection"], records)
    _panel_summary(axes["summary"], records)
    fig.suptitle(
        "Cost of the exact polygon-section route: closed form against boundary "
        "quadrature\ncompute node, one core, fresh process per variant, "
        "median of 3, module import outside the timer",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    fig.savefig(path, dpi=130)
    return path


ARRANGEMENT_COLOUR = {
    "point": "0.55",
    "exact-quadrature": "C3",
    "exact-closed": "C0",
    "banded-quadrature": "C1",
    "banded-closed": "C2",
}


def _panel_per_pair(axis, records):
    """Per-pair cost by arrangement and section, against the recorded figures."""
    grouped = _group(records, "column", "arrangement", "section")
    names = [
        name
        for name in ARRANGEMENTS
        if any((name, section) in grouped for section in SECTIONS)
    ]
    width = 0.38
    for offset, (section, hatch) in enumerate(zip(SECTIONS, ("", "///"))):
        heights, places, colours = [], [], []
        for index, name in enumerate(names):
            found = grouped.get((name, section))
            if not found:
                continue
            heights.append(_median(found)[0])
            places.append(index + (offset - 0.5) * width)
            colours.append(ARRANGEMENT_COLOUR[name])
        axis.bar(
            places,
            heights,
            width,
            hatch=hatch,
            label=section,
            color=colours,
            edgecolor="k",
            linewidth=0.6,
        )
        for place, height in zip(places, heights):
            axis.annotate(
                f"{height:.1f}",
                (place, height),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
                rotation=45,
            )
    for name in ("exact-quadrature", "banded-quadrature"):
        value = RECORDED[name]
        axis.axhline(value, color="k", ls="--", lw=1.0)
        axis.annotate(
            f"recorded {name} {value:.1f}",
            (-0.45, value),
            textcoords="offset points",
            xytext=(0, -3),
            fontsize=7.5,
            va="top",
            ha="left",
            color="k",
        )
    axis.set_yscale("log")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    axis.set_ylabel("cost per target-source pair  [us]")
    axis.set_title(
        f"(a) one source column against a {LATTICE_CELLS}-cell target cloud",
        fontsize=10,
    )
    axis.legend(fontsize=8, title="section", title_fontsize=8)
    axis.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    axis.set_ylim(top=axis.get_ylim()[1] * 3.0)


def _panel_population(axis, records):
    """Band populations, and what each band contributes to the column rate."""
    rows = population_rows(records)
    places = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for band_name, colour in (("near", "C3"), ("mid", "C1"), ("far", "C2")):
        heights = np.array([row[band_name] for row in rows])
        axis.barh(
            places,
            heights,
            left=bottom,
            color=colour,
            label=band_name,
            edgecolor="k",
            linewidth=0.5,
        )
        for place, height, base in zip(places, heights, bottom):
            if height > 0.06:
                axis.annotate(
                    f"{height:.1%}",
                    (base + 0.5 * height, place),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottom = bottom + heights
    axis.set_yticks(places)
    axis.set_yticklabels(
        [
            f"{row['section']}\n{row['corners']} corners, seam "
            f"{row['far_seam']:.1f} a\nnear {row['near']:.2%}"
            for row in rows
        ],
        fontsize=8,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("fraction of target-source pairs")
    axis.set_title(
        "(b) band populations -- the mixture a banded per-pair rate is", fontsize=10
    )
    axis.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))


BAND_COLOUR = {
    "near-quadrature": "C3",
    "near-closed": "C0",
    "mid": "C1",
    "far": "C2",
}


def _panel_distance(axis, records):
    """The two exact kernels against distance, with the scheme's seams marked.

    Neither kernel's cost depends on where the target sits: the quadrature spends a
    fixed node count at any distance, and the closed form's per-corner residuals take
    a fixed graded rule. So a per-pair rate can be quoted for a whole column without
    a distance weighting -- and the near band's cost, which does differ, has to be
    explained by something other than distance. That is what the next panel measures.
    """
    from nova.biot.bandedcoupling import MID_LIMIT, NEAR_LIMIT

    grouped = _group(records, "distance", "treatment", "offset")
    for treatment, style in (("near-quadrature", "o-"), ("near-closed", "s-")):
        offsets = sorted(key[1] for key in grouped if key[0] == treatment)
        if not offsets:
            continue
        axis.plot(
            [grouped[(treatment, offset)][0]["contour"] for offset in offsets],
            [_median(grouped[(treatment, offset)])[0] for offset in offsets],
            style,
            color=BAND_COLOUR[treatment],
            label=treatment.replace("near-", ""),
            lw=1.8,
            ms=5,
        )
    for value, name in ((NEAR_LIMIT, "near/mid seam"), (MID_LIMIT, "mid/far seam")):
        axis.axvline(value, color="k", ls=":", lw=1.1)
        axis.annotate(
            name,
            (value, axis.get_ylim()[1]),
            rotation=90,
            textcoords="offset points",
            xytext=(3, -4),
            fontsize=7.5,
            va="top",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("contour distance / section radius")
    axis.set_ylabel(f"cost per pair, {DISTANCE_PAIRS} pairs a ring  [us]")
    axis.set_title("(c) both exact kernels against distance", fontsize=10)
    axis.legend(fontsize=8, title="exact kernel", title_fontsize=8, loc="lower left")
    axis.grid(ls=":", lw=0.5, alpha=0.6)


def _panel_batch(axis, records):
    """The two exact kernels against batch width -- what a thin near band pays.

    Both kernels carry a cost per CALL that no pair count divides away, and the two
    amortise it differently: the quadrature builds one angular rule and reuses it
    across the batch, while the closed form holds several corner parts live at once.
    The width the three-band scheme actually hands the near band is marked, and it is
    the width at which the choice is made -- not the width a whole column gives.
    """
    grouped = _group(records, "batch", "treatment", "pairs")
    for treatment, style in (("near-quadrature", "o-"), ("near-closed", "s-")):
        widths = sorted(key[1] for key in grouped if key[0] == treatment)
        if not widths:
            continue
        axis.plot(
            widths,
            [_median(grouped[(treatment, width)])[0] for width in widths],
            style,
            color=BAND_COLOUR[treatment],
            label=treatment.replace("near-", ""),
            lw=1.8,
            ms=5,
        )
    crossover = _batch_crossover(grouped)
    if crossover:
        axis.axvline(crossover, color="0.4", lw=1.0)
        axis.annotate(
            f"the two cross at\n{crossover} pairs a call",
            (crossover, 0.06),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=8,
            va="bottom",
            color="0.25",
        )
    measured = _group(records, "band_cost", "treatment", "section")
    near = measured.get(("near-quadrature", "hexagon"))
    if near:
        pairs, total = near[0]["pairs"], round(near[0]["pairs"] / near[0]["fraction"])
        axis.axvline(pairs, color="k", ls=":", lw=1.2)
        axis.annotate(
            f"near band of one column:\n{pairs} pairs of {total}",
            (pairs, axis.get_ylim()[1]),
            textcoords="offset points",
            xytext=(5, -6),
            fontsize=8,
            va="top",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("pairs in one kernel call")
    axis.set_ylabel("cost per pair  [us]")
    axis.set_title("(d) both exact kernels against batch width", fontsize=10)
    axis.legend(fontsize=8, title="exact kernel", title_fontsize=8, loc="upper right")
    axis.grid(ls=":", lw=0.5, alpha=0.6)


def _panel_summary(axis, records):
    """The headline numbers as text, so the figure can be read without the tables."""
    axis.axis("off")
    column = _group(records, "column", "arrangement", "section")
    build = _group(records, "build", "arrangement")

    def rate(arrangement, section="hexagon"):
        found = column.get((arrangement, section))
        return _median(found)[0] if found else float("nan")

    def solve(arrangement):
        found = build.get((arrangement,))
        return _median(found, "seconds")[0] if found else float("nan")

    exact_ratio = rate("exact-quadrature") / rate("exact-closed")
    band_ratio = rate("banded-closed") / rate("banded-quadrature")
    lines = [
        ("(g) what the measurement decides", 11.5, "bold"),
        ("", 9, "normal"),
        ("EXACT EVERYWHERE, per pair (hexagon column):", 9.5, "bold"),
        (f"   boundary quadrature   {rate('exact-quadrature'):8.1f} us", 9.5, "normal"),
        (f"   closed form           {rate('exact-closed'):8.1f} us", 9.5, "normal"),
        (f"   closed form is {exact_ratio:.1f}x CHEAPER -- adopt it here", 9.5, "bold"),
        ("", 8, "normal"),
        ("THREE BANDS, whole column (hexagon):", 9.5, "bold"),
        (
            f"   quadrature near band  {rate('banded-quadrature'):8.1f} us",
            9.5,
            "normal",
        ),
        (f"   closed-form near band {rate('banded-closed'):8.1f} us", 9.5, "normal"),
        (
            f"   closed form is {band_ratio:.1f}x DEARER -- the near band is",
            9.5,
            "bold",
        ),
        ("   too thin to amortise it; see panel (d)", 9.5, "bold"),
        ("", 8, "normal"),
        ("PLASMA-GRID BUILD (fresh process):", 9.5, "bold"),
        (
            f"   point filament        {solve('point'):8.2f} s"
            f"   (tracked warm {TRACKED_POINT_BUILD:.3f} s)",
            9.5,
            "normal",
        ),
        (
            f"   exact quadrature      {solve('exact-quadrature'):8.2f} s"
            f"   {solve('exact-quadrature') / solve('point'):6.0f}x point",
            9.5,
            "normal",
        ),
        (
            f"   exact closed form     {solve('exact-closed'):8.2f} s"
            f"   {solve('exact-closed') / solve('point'):6.0f}x point",
            9.5,
            "normal",
        ),
        (
            f"   banded quadrature     {solve('banded-quadrature'):8.2f} s"
            f"   {solve('banded-quadrature') / solve('point'):6.0f}x point",
            9.5,
            "normal",
        ),
        (
            f"   banded closed form    {solve('banded-closed'):8.2f} s"
            f"   {solve('banded-closed') / solve('point'):6.0f}x point",
            9.5,
            "normal",
        ),
    ]
    height = 0.97
    for text, size, weight in lines:
        axis.text(
            0.0,
            height,
            text,
            transform=axis.transAxes,
            fontsize=size,
            fontweight=weight,
            family="monospace" if text.startswith("   ") else None,
            va="top",
        )
        height -= 0.042 if text else 0.022


def _panel_build(axis, records):
    """Measured plasma-grid build time by arrangement, against the point build."""
    grouped = _group(records, "build", "arrangement")
    names = [name for name in ARRANGEMENTS if (name,) in grouped]
    heights = [_median(grouped[(name,)], "seconds")[0] for name in names]
    cells = grouped[(names[0],)][0]["cells"] if names else 0
    axis.bar(
        range(len(names)),
        heights,
        color=[ARRANGEMENT_COLOUR[name] for name in names],
        edgecolor="k",
        linewidth=0.6,
    )
    point = heights[names.index("point")] if "point" in names else None
    for index, height in enumerate(heights):
        text = f"{height:.2f} s"
        if point:
            text += f"\n{height / point:.0f}x"
        axis.annotate(
            text,
            (index, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
        )
    if point:
        axis.axhline(point, color="k", ls=":", lw=1.0)
    axis.axhline(TRACKED_POINT_BUILD, color="k", ls="--", lw=1.0)
    axis.annotate(
        f"tracked point build {TRACKED_POINT_BUILD:.3f} s\n(warm: repeated in process)",
        (len(names) - 0.4, TRACKED_POINT_BUILD),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=8,
        va="bottom",
        ha="right",
    )
    axis.set_yscale("log")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    axis.set_ylabel("plasma-grid solve  [s]")
    axis.set_title(
        f"(e) a real {cells}-cell plasma-grid build, fresh process", fontsize=10
    )
    axis.grid(axis="y", ls=":", lw=0.5, alpha=0.6)


def _panel_projection(axis, records):
    """Projected first build against core count and device, with the budget line."""
    projected = projection(records)
    columns = ["one core", "host pool", "device cold", "device warm cache"]
    places = np.arange(len(columns))
    for row in projected["rows"]:
        values = [row.get(name) for name in columns]
        keep = [index for index, value in enumerate(values) if value is not None]
        axis.plot(
            places[keep],
            [values[index] / 60.0 for index in keep],
            "o-",
            color=ARRANGEMENT_COLOUR[row["arrangement"]],
            label=row["arrangement"],
            lw=1.6,
            ms=5,
        )
    axis.axhline(FIRST_BUILD_BUDGET / 60.0, color="k", ls="--", lw=1.2)
    axis.annotate(
        f"{FIRST_BUILD_BUDGET / 60.0:.0f} min first-build budget",
        (len(columns) - 1, FIRST_BUILD_BUDGET / 60.0),
        textcoords="offset points",
        xytext=(0, 4),
        ha="right",
        fontsize=8,
    )
    axis.set_yscale("log")
    axis.set_xticks(places)
    axis.set_xticklabels(
        [
            "1 core",
            f"16 cores\n(x{HOST_POOL_SCALING} projected)",
            "H200\ncold compile",
            "H200\nwarm cache",
        ],
        fontsize=8,
    )
    axis.set_ylabel(f"projected {PROJECTED_CELLS}-cell first build  [min]")
    axis.set_title(
        f"(f) projected {PROJECTED_CELLS}-cell build "
        f"({PROJECTED_CELLS**2 / 1e6:.0f}M pairs), hexagon rates",
        fontsize=10,
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", ls=":", lw=0.5, alpha=0.6)


# --- command line ------------------------------------------------------------


def sweep(path, repeats=3, cells=BUILD_CELLS, skip=()):
    """Run every measurement in its own interpreter, appending records to a file.

    The protocol lives here rather than in a shell loop so that reproducing the
    tables cannot accidentally reuse a warm process: each measurement is a fresh
    ``python -m`` of this module, and the repeats are separate processes rather
    than a loop inside one. ``skip`` drops named subcommand arguments -- a build
    variant too slow for the queue's wall, say -- and the tables report the
    process count behind each entry, so a dropped one is visible.
    """
    import subprocess

    repeated = ("column", "build", "band-cost", "batch", "distance")
    calls = [["population"], ["build-population"]]
    for arrangement in ARRANGEMENTS:
        for section in SECTIONS:
            calls.append(["column", arrangement, section])
    for section in SECTIONS:
        for treatment in BANDS:
            calls.append(["band-cost", section, treatment])
    for treatment in ("near-quadrature", "near-closed"):
        for width in BATCH_WIDTHS:
            calls.append(["batch", "hexagon", treatment, str(width)])
        for offset in DISTANCE_OFFSETS:
            calls.append(["distance", "hexagon", treatment, str(offset)])
    for arrangement in ARRANGEMENTS:
        calls.append(["build", arrangement, "--cells", str(cells)])
    with open(path, "a") as stream:
        for call in calls:
            if any(token in skip for token in call):
                print(f"skipped {' '.join(call)}", flush=True)
                continue
            for repeat in range(repeats if call[0] in repeated else 1):
                start = time.perf_counter()
                done = subprocess.run(
                    [sys.executable, "-m", "benchmarks.polygon_route_cost", *call],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if done.returncode != 0:
                    print(f"FAILED {' '.join(call)}\n{done.stderr[-2000:]}", flush=True)
                    continue
                for line in done.stdout.splitlines():
                    if line.startswith("{"):
                        stream.write(line + "\n")
                stream.flush()
                print(
                    f"{' '.join(call)} [{repeat + 1}/{repeats}] "
                    f"{time.perf_counter() - start:.1f} s wall",
                    flush=True,
                )
    return path


def main(argv=None):
    """Dispatch one measurement, or collect a file of them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    column = sub.add_parser("column", help="per-pair cost of one source column")
    column.add_argument("arrangement", choices=list(ARRANGEMENTS))
    column.add_argument("section", choices=list(SECTIONS))
    sub.add_parser("population", help="band populations of the target cloud")
    cost = sub.add_parser("band-cost", help="what one band's treatment costs")
    cost.add_argument("section", choices=list(SECTIONS))
    cost.add_argument("treatment", choices=list(BANDS))
    batch = sub.add_parser("batch", help="a near kernel at one batch width")
    batch.add_argument("section", choices=list(SECTIONS))
    batch.add_argument("treatment", choices=["near-quadrature", "near-closed"])
    batch.add_argument("pairs", type=int)
    reach = sub.add_parser("distance", help="a kernel on a ring at one distance")
    reach.add_argument("section", choices=list(SECTIONS))
    reach.add_argument("treatment", choices=list(BANDS))
    reach.add_argument("offset", type=float)
    build = sub.add_parser("build", help="a real plasma-grid solve")
    build.add_argument("arrangement", choices=list(ARRANGEMENTS))
    build.add_argument("--cells", type=int, default=BUILD_CELLS)
    sub.add_parser("build-population", help="band populations of the real grid")
    run = sub.add_parser("sweep", help="every measurement, one fresh process each")
    run.add_argument("records")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--cells", type=int, default=BUILD_CELLS)
    run.add_argument("--skip", nargs="*", default=[])
    collect = sub.add_parser("collect", help="tables and figure from a record file")
    collect.add_argument("records")
    collect.add_argument("--figure", default=None)

    args = parser.parse_args(argv)
    match args.command:
        case "sweep":
            sweep(args.records, args.repeats, args.cells, skip=tuple(args.skip))
        case "column":
            print(json.dumps(measure_column(args.arrangement, args.section)))
        case "band-cost":
            print(json.dumps(measure_band_cost(args.section, args.treatment)))
        case "batch":
            print(json.dumps(measure_batch(args.section, args.treatment, args.pairs)))
        case "distance":
            print(
                json.dumps(measure_distance(args.section, args.treatment, args.offset))
            )
        case "population":
            for section in SECTIONS:
                print(json.dumps(measure_population(section)))
        case "build":
            print(json.dumps(measure_build(args.arrangement, args.cells)))
        case "build-population":
            for record in measure_build_population():
                print(json.dumps(record))
        case "collect":
            records = read_records(args.records)
            print("## coupling kernel cost per pair\n")
            print(column_table(records))
            print("\n## band populations\n")
            print(population_table(records))
            print("\n## cost of each band's treatment, on its own pairs\n")
            print(band_cost_table(records))
            print("\n## the two exact kernels against DISTANCE, 256 pairs a ring\n")
            print(distance_table(records))
            print("\n## the two exact kernels against BATCH WIDTH, inside the band\n")
            print(batch_table(records))
            print("\n## plasma-grid build\n")
            print(build_table(records))
            print(f"\n## projected {PROJECTED_CELLS}-cell first build\n")
            print(projection_table(projection(records)))
            if args.figure:
                print(f"\nfigure: {figure(records, args.figure)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
