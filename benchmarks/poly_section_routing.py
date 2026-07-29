"""Evidence for the cross-section a swept winding presents, and what routing costs.

Four panels, one per claim this rung rests on.

**Top left -- what section a swept winding actually presents.**  Three answers to
the same question, drawn on top of each other: the section's exact corners, the
frame's ``poly`` column, and the box a width-and-height descriptor bounds.  The
column is built from the sweep's own float64 corner loops rather than from the vtk
mesh those loops build, because VTK's points default to single precision and a mesh
round trip lands a corner authored at 2.97 on 2.96999979 -- 8e-09 relative, four
orders above the closed-form reduction's own round-off.  The panel annotates both
deviations, and the corner count the projection hands over before its collinear
runs are collapsed.

**Top right -- the ratio the routing flip corrects, per section shape.**  ``Bow``
integrates the RECTANGLE a section's width and height bound while normalising by
the section's own area, so every section that does not fill its bounding box comes
back too large by the reciprocal of the fraction it fills: one for a rectangle,
4/3 for a hexagon, 4/pi for a disc and for an ellipse.  Measured through the frame
on all five rows, because the error is a normalisation and is therefore the same on
each.

**Bottom left -- a hollow section as a coupled pair.**  An annulus is the outer
boundary at current density ``+j`` and the interior boundary as a core at ``-j``,
both of them solid sections the reduction already evaluates.  The panel checks the
pair against the annulus integrated DIRECTLY -- partitioned into solid
quadrilateral cells between corresponding corners of the two boundaries, summed
with no cancellation between members -- so the agreement is a statement about the
superposition rather than about the kernel.

**Bottom right -- what a round section costs, and what a coarser one saves.**  A
disc or an ellipse is correct only at the corner count its generator resolves, and
the cost of a closed-form section reduction tracks the corner count.  The panel puts
the measured cost per pair against the measured error of the same section at coarser
resolutions, both against the 512-corner reference, so the decision on whether
sixty-four corners is the default is read off a curve rather than argued.

Timings come from a JSON file of this module's own ``--measure`` records -- one
variant per process on a compute node -- rather than being measured here, because a
figure script is the last place a benchmark should run::

    for v in $(python benchmarks/poly_section_routing.py --measure list); do
        python benchmarks/poly_section_routing.py --measure "$v" >> cost.jsonl
    done
    python benchmarks/poly_section_routing.py routing.png cost.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np
from scipy.constants import mu_0

RADIUS, ELEVATION = 3.0, 0.2
WIDTH, THICKNESS = 0.06, 0.04
HOLLOW = 0.2  # hollowness factor, 1 - r/R
SWEEP = (0.3, 1.9)
FIELD_ATTRS = ["Ax", "Ay", "Bx", "By", "Bz"]
ROWS = (r"$A_r$", r"$A_\phi$", r"$B_r$", r"$B_\phi$", r"$B_z$")
REPEATS = 3
N_TARGET = 512

TARGET_R = np.array([3.5, 2.5, 3.1, 3.0, 3.3])
TARGET_PHI = np.array([0.2, 1.0, 2.4, -0.4, 4.5])
TARGET_Z = np.array([0.5, -0.2, 0.25, 0.35, -0.5])

# Corner counts a disc generator can be asked for, as quadrant segments.  The
# reference is four times the shipped resolution, which is where the row error of
# the shipped one stops moving; the cost sweep stops short of it because a section
# reduction costs one evaluation per corner and 512 of them over 512 pairs is two
# minutes of wall for a point already established by the curve below it.
QUADRANT_SEGMENTS = (2, 4, 8, 16, 32, 128)
COST_SEGMENTS = (2, 4, 8, 16, 32)
REFERENCE_SEGMENTS = 128

SHAPES = {
    "rectangle": {"rect": (0, 0, WIDTH, THICKNESS)},
    "hexagon": {"hex": (0, 0, WIDTH, THICKNESS)},
    "disc": {"disc": (0, 0, WIDTH, WIDTH)},
    "ellipse": {"ellipse": (0, 0, WIDTH, THICKNESS)},
}
FILL = {
    "rectangle": 1.0,
    "hexagon": 4.0 / 3.0,
    "disc": 4.0 / np.pi,
    "ellipse": 4.0 / np.pi,
}

# Measured 2026-07-27, sun_debug node, one core, fresh process per variant, median
# of 3, 512 pairs through the frame.  Overridden by a JSON file of --measure records.
#
# ``Bow`` is flat in the section -- 180.5 to 182.7 across all four shapes -- because
# it integrates four corners whatever the section is; the polygon element is linear
# in the corner count at about 161 microseconds per corner per pair, so a disc's
# sixty-four corners cost 11.9 times a hexagon's six and 57 times ``Bow``.  A hollow
# section pays for both of its members, which is why ``polybow-skin`` is twice
# ``polybow-disc`` and ``bow-skin`` is not quite twice ``bow-disc``.
COST = {
    "bow-rectangle": 181.1,
    "bow-hexagon": 182.7,
    "bow-disc": 180.5,
    "bow-ellipse": 181.7,
    "bow-skin": 301.8,
    "polybow-rectangle": 570.4,
    "polybow-hexagon": 867.7,
    "polybow-disc": 10128.6,
    "polybow-ellipse": 10075.0,
    "polybow-skin": 20370.1,
    "polybow-disc-8": 1332.9,
    "polybow-disc-16": 2642.0,
    "polybow-disc-32": 5375.6,
    "polybow-disc-64": 10327.3,
    "polybow-disc-128": 20106.3,
}


# ---------------------------------------------------------------------------
# The frame, and the rows read back out of it.


def solved(cross_section, segment=None, sweep=SWEEP, targets=None):
    """Return a solved coilset carrying one thickened winding of that section."""
    from nova.frame.coilset import CoilSet

    start, end = sweep
    angle = np.array([start, 0.5 * (start + end), end])
    coilset = CoilSet(field_attrs=FIELD_ATTRS)
    coilset.winding.insert(
        np.stack(
            [
                RADIUS * np.cos(angle),
                RADIUS * np.sin(angle),
                ELEVATION * np.ones_like(angle),
            ],
            axis=-1,
        ),
        cross_section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    if targets is None:
        targets = np.stack(
            [TARGET_R * np.cos(TARGET_PHI), TARGET_R * np.sin(TARGET_PHI), TARGET_Z],
            axis=-1,
        )
    coilset.point.solve(targets)
    return coilset


def cylindrical_rows(coilset):
    """Return the five rows per ampere in the target's own cylindrical basis."""
    matrix = np.stack(
        [np.asarray(coilset.point.data[attr]).sum(axis=1) for attr in FIELD_ATTRS]
    )
    potential_x, potential_y, field_x, field_y, field_z = matrix
    cosine, sine = np.cos(TARGET_PHI), np.sin(TARGET_PHI)
    return np.stack(
        [
            mu_0 * (potential_x * cosine + potential_y * sine),
            mu_0 * (-potential_x * sine + potential_y * cosine),
            field_x * cosine + field_y * sine,
            -field_x * sine + field_y * cosine,
            field_z,
        ]
    )


def descriptor_box(coilset):
    """Return the rectangle the frame's width and height bound, at the arc."""
    width = float(np.asarray(coilset.subframe["width"])[0])
    height = float(np.asarray(coilset.subframe["height"])[0])
    return np.array(
        [
            [RADIUS - width / 2, ELEVATION - height / 2],
            [RADIUS + width / 2, ELEVATION - height / 2],
            [RADIUS + width / 2, ELEVATION + height / 2],
            [RADIUS - width / 2, ELEVATION + height / 2],
        ]
    )


def exact_corners(name):
    """Return the section's own corners at the arc, from its generator."""
    from nova.geometry.polygen import PolyGen
    from nova.geometry.section import collapse_collinear

    args = SHAPES[name][next(iter(SHAPES[name]))]
    shape = PolyGen(name)(RADIUS, ELEVATION, *args[2:])
    return collapse_collinear(np.asarray(shape.exterior.coords, dtype=np.float64))


def mesh_corners(coilset):
    """Return the section a vtk mesh round trip reports, for the precision it costs."""
    from nova.geometry.volume import TriShell

    vtk = np.asarray(coilset.subframe["vtk"])[0]
    return np.asarray(TriShell(vtk, ahull=True, alpha=None).poly.points)[:, [0, 2]]


# ---------------------------------------------------------------------------
# Disc resolution: the generator's own corner count as a knob.


def disc(segments):
    """Return a disc descriptor resolved to ``4 * segments`` corners."""
    return {"disc": (0, 0, WIDTH, WIDTH, segments)}


def disc_rows(segments):
    """Return the five rows of a disc winding at that resolution."""
    return cylindrical_rows(solved(disc(segments)))


def disc_error():
    """Return the corner count, area deficit and row error against the reference."""
    from nova.geometry.polygen import PolyGen

    exact = np.pi / 4 * WIDTH**2
    reference = disc_rows(REFERENCE_SEGMENTS)
    record = []
    for segments in QUADRANT_SEGMENTS:
        polygon = PolyGen("disc")(0.0, 0.0, WIDTH, WIDTH, segments)
        rows = disc_rows(segments)
        scale = np.max(np.abs(reference), axis=1)[:, None]
        record.append(
            dict(
                corners=4 * segments,
                area_deficit=1.0 - polygon.area / exact,
                row_error=float(np.max(np.abs(rows - reference) / scale)),
            )
        )
    return record


# ---------------------------------------------------------------------------
# The hollow pair, against the annulus integrated directly.


def annulus_partition(coilset):
    """Return a partition of the annulus into solid quadrilateral cells."""
    from nova.biot.polybow import section_corners

    outer, core = (
        section_corners(poly) for poly in np.asarray(coilset.subframe["poly"])[:2]
    )
    return [
        np.array(
            [outer[i], outer[(i + 1) % len(outer)], core[(i + 1) % len(core)], core[i]]
        )
        for i in range(len(outer))
    ]


def cell_integral(cells, sweep=SWEEP):
    """Return the five rows of a partition, at the partition's own density."""
    from nova.biot.polygonarc import polygon_arc_greens

    start, end = sweep
    total = np.zeros((5, len(TARGET_R)))
    area = 0.0
    for cell in cells:
        rolled = np.roll(cell, -1, axis=0)
        cell_area = 0.5 * abs(
            float(np.sum(cell[:, 0] * rolled[:, 1] - rolled[:, 0] * cell[:, 1]))
        )
        total += cell_area * np.stack(
            polygon_arc_greens(TARGET_R, TARGET_Z, TARGET_PHI, cell, start, end)
        )
        area += cell_area
    return total / area, area


def hollow_record():
    """Return the pair against the direct integral, per hollow section name."""
    record = {}
    for label, key in (("box", "box"), ("skin", "sk")):
        coilset = solved({key: (0, 0, WIDTH, HOLLOW)})
        want, area = cell_integral(annulus_partition(coilset))
        got = cylindrical_rows(coilset)
        scale = np.max(np.abs(want), axis=1)[:, None]
        outer, core = (
            poly.poly.area for poly in np.asarray(coilset.subframe["poly"])[:2]
        )
        column = float(np.asarray(coilset.subframe["area"])[0])
        density = 1.0 / column
        record[label] = dict(
            by_row=np.max(np.abs(got - want) / scale, axis=1),
            cells=len(annulus_partition(coilset)),
            partition_area=area,
            column_area=column,
            current_sum=density * outer - density * core,
        )
    return record


# ---------------------------------------------------------------------------
# Cost, one variant per process.


def _timed(call, pairs):
    """Return the median of ``REPEATS`` timings, in microseconds per pair."""
    elapsed = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return 1e6 * statistics.median(elapsed) / pairs


def cloud(count=N_TARGET, reach=8.0 * WIDTH):
    """Return a spiral target cloud ringing the section, azimuths across the sweep."""
    turn = np.linspace(0.0, 6.0 * np.pi, count)
    radius = RADIUS + reach * (0.3 + 0.7 * turn / turn[-1]) * np.cos(turn)
    height = ELEVATION + reach * (0.3 + 0.7 * turn / turn[-1]) * np.sin(turn)
    azimuth = np.linspace(SWEEP[0] - 1.0, SWEEP[1] + 1.0, count)
    return np.stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), height], axis=-1
    )


def _cost(cross_section, segment):
    """Return an element class's per-pair cost through the frame, same cloud."""
    from nova.frame.coilset import CoilSet

    targets = cloud()
    start, end = SWEEP
    angle = np.array([start, 0.5 * (start + end), end])
    coilset = CoilSet(field_attrs=FIELD_ATTRS)
    coilset.winding.insert(
        np.stack(
            [
                RADIUS * np.cos(angle),
                RADIUS * np.sin(angle),
                ELEVATION * np.ones_like(angle),
            ],
            axis=-1,
        ),
        cross_section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment

    def once():
        coilset.point.solve(targets)

    return _timed(once, len(targets)), len(targets)


VARIANTS = {
    **{
        f"{element}-{name}": (
            lambda name=name, segment=segment: _cost(SHAPES[name], segment)
        )
        # ``Bow`` has to be NAMED now: a thickened arc routes to the polygon-section
        # element on its own, which is the flip this rung landed.
        for element, segment in (("bow", "bow"), ("polybow", None))
        for name in SHAPES
    },
    **{
        f"polybow-disc-{4 * segments}": (
            lambda segments=segments: _cost(disc(segments), None)
        )
        for segments in COST_SEGMENTS
    },
    "polybow-skin": lambda: _cost({"sk": (0, 0, WIDTH, HOLLOW)}, None),
    "bow-skin": lambda: _cost({"sk": (0, 0, WIDTH, HOLLOW)}, "bow"),
}


def measure(name):
    """Run one cost variant and print its record."""
    if name == "list":
        print("\n".join(sorted(VARIANTS)))
        return
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; try 'list'")
    cost, pairs = VARIANTS[name]()
    print(
        json.dumps(
            {
                "variant": name,
                "microseconds_per_pair": round(cost, 4),
                "pairs": pairs,
                "repeats": REPEATS,
            }
        )
    )


def load_cost(path):
    """Return the cost table, overridden by a JSON or JSON-lines record file."""
    cost = dict(COST)
    if path is None:
        return cost
    with open(path) as handle:
        for line in handle:
            if not (line := line.strip()):
                continue
            record = json.loads(line)
            for entry in record if isinstance(record, list) else [record]:
                cost[entry["variant"]] = entry["microseconds_per_pair"]
    return cost


# ---------------------------------------------------------------------------
# The figure.


def section_panel(axes):
    """Draw the three answers to what section a swept winding presents."""
    from nova.biot.polybow import section_corners

    coilset = solved(SHAPES["hexagon"])
    stored = np.asarray(coilset.subframe["poly"])[0]
    column = section_corners(stored)
    exact = exact_corners("hexagon")
    box = descriptor_box(coilset)
    mesh = mesh_corners(coilset)

    def ring(points):
        return np.r_[points, points[:1]]

    axes.plot(
        *ring(box).T, color="C3", lw=1.2, ls="--", label="descriptor box, what Bow adds"
    )
    axes.fill(*ring(exact).T, color="0.85", lw=0)
    axes.plot(*ring(exact).T, color="0.25", lw=2.0, label="section, exact corners")
    axes.plot(
        *column.T, "o", color="C0", ms=8, mfc="none", mew=1.6, label="poly column"
    )
    axes.plot(*mesh.T, "x", color="C1", ms=6, mew=1.4, label="via the vtk mesh")

    order = [
        int(np.argmin(np.linalg.norm(exact - corner, axis=1))) for corner in column
    ]
    column_error = float(np.max(np.abs(column - exact[order])))
    mesh_error = float(
        np.max(np.min(np.linalg.norm(mesh[:, None] - exact[None], axis=-1), axis=1))
    )
    axes.set_title("the section a swept winding presents", fontsize=9)
    axes.set_xlabel("$r$ [m]")
    axes.set_ylabel("$z$ [m]")
    axes.set_aspect("equal")
    axes.margins(0.22)
    axes.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
    axes.text(
        0.02,
        0.14,
        f"worst corner deviation\n"
        f"  poly column  {column_error:.1e} m\n"
        f"  via the mesh {mesh_error:.1e} m\n"
        f"{len(np.asarray(stored.points))} stored corners"
        f" -> {len(column)} after collapse",
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.95),
    )


def ratio_panel(axes):
    """Draw the ratio the routing flip corrects, per section shape."""
    names = list(SHAPES)
    offset = np.arange(len(names))
    for row in range(5):
        measured = []
        for name in names:
            bow = cylindrical_rows(solved(SHAPES[name], segment="bow"))
            poly = cylindrical_rows(solved(SHAPES[name]))
            measured.append(float(np.median(bow[row] / poly[row])))
        axes.plot(
            offset + 0.06 * (row - 2),
            measured,
            "o",
            ms=5,
            color=f"C{row}",
            label=ROWS[row],
        )
    for position, name in enumerate(names):
        axes.hlines(FILL[name], position - 0.35, position + 0.35, color="0.4", lw=1.2)
        axes.annotate(
            {1.0: "1", 4 / 3: "4/3", 4 / np.pi: r"4/$\pi$"}[FILL[name]],
            (position + 0.38, FILL[name]),
            fontsize=7,
            va="center",
        )
    axes.set_xticks(offset)
    axes.set_xticklabels(names, fontsize=7)
    axes.set_ylabel("Bow / polygon section")
    axes.set_ylim(0.95, 1.42)
    axes.set_title("what routing to Bow overstates, by shape", fontsize=9)
    axes.legend(
        fontsize=6.5, ncol=5, loc="upper left", columnspacing=0.8, handletextpad=0.2
    )


def hollow_panel(axes):
    """Draw the coupled pair against the annulus integrated directly."""
    record = hollow_record()
    offset = np.arange(5)
    for position, (label, entry) in enumerate(record.items()):
        axes.semilogy(
            offset + 0.16 * (position - 0.5),
            np.maximum(entry["by_row"], 1e-16),
            "o",
            ms=6,
            color=f"C{position}",
            label=f"{label}: {entry['cells']} cells, "
            f"$\\Sigma I$ = {entry['current_sum']:.12f}",
        )
    axes.set_xticks(offset)
    axes.set_xticklabels(ROWS)
    axes.set_ylabel("worst deviation / row scale")
    axes.set_ylim(1e-14, 1e-6)
    axes.set_title("hollow $+j$ / $-j$ pair vs the direct annulus integral", fontsize=9)
    axes.legend(fontsize=6.5, loc="upper left")


def resolution_panel(axes, cost):
    """Draw what a round section costs against what a coarser one gives up."""
    # The reference itself is dropped: its row error against itself is zero, which
    # says nothing and drags a log axis five decades to say it.
    record = [
        entry for entry in disc_error() if entry["corners"] < 4 * REFERENCE_SEGMENTS
    ]
    corners = np.array([entry["corners"] for entry in record])
    error = np.array([entry["row_error"] for entry in record])
    deficit = np.array([abs(entry["area_deficit"]) for entry in record])
    axes.loglog(
        corners,
        error,
        "o-",
        color="C0",
        ms=5,
        label=f"row error vs {4 * REFERENCE_SEGMENTS} corners",
    )
    axes.loglog(
        corners, deficit, "s--", color="C2", ms=5, label="area deficit vs the circle"
    )
    axes.axvline(64, color="0.6", lw=1.0, ls=":")
    axes.annotate(
        "shipped\nresolution",
        (64, error[0]),
        fontsize=6.5,
        color="0.35",
        ha="right",
        va="top",
    )
    axes.set_xlabel("disc corners")
    axes.set_ylabel("relative error")
    axes.set_title("a round section: cost against error", fontsize=9)

    twin = axes.twinx()
    measured = [
        (4 * segments, cost.get(f"polybow-disc-{4 * segments}", 0.0))
        for segments in COST_SEGMENTS
    ]
    measured = [(count, value) for count, value in measured if value > 0]
    if measured:
        twin.plot(
            *np.array(measured).T, "^-", color="C3", ms=5, label=r"cost [$\mu$s/pair]"
        )
        # As TEXT rather than as lines on the twin axis: a horizontal line drawn
        # against the right-hand scale reads as a level on the left-hand one, and
        # the two carry different quantities.
        axes.text(
            0.97,
            0.03,
            f"for comparison, same cloud:\n"
            f"  Bow          {cost.get('bow-disc', 0.0):8.1f} $\\mu$s/pair\n"
            f"  hexagon, 6   {cost.get('polybow-hexagon', 0.0):8.1f} $\\mu$s/pair",
            transform=axes.transAxes,
            va="bottom",
            ha="right",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.95),
        )
    twin.set_yscale("log")
    twin.set_ylabel(r"cost [$\mu$s/pair]", color="C3")
    twin.tick_params(axis="y", colors="C3")
    handles, labels = axes.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    axes.legend(handles + extra[0], labels + extra[1], fontsize=6.5, loc="center left")


def figure(path="poly_section_routing.png", cost_path=None):
    """Render the four panels and write the PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost = load_cost(cost_path)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))
    section_panel(axes[0, 0])
    ratio_panel(axes[0, 1])
    hollow_panel(axes[1, 0])
    resolution_panel(axes[1, 1], cost)
    fig.suptitle(
        "the poly cross-section, the routing flip, and hollow coupled pairs",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170)
    print(path)


def main(argv):
    """Measure one variant, or render the figure."""
    if argv and argv[0] == "--measure":
        measure(argv[1] if len(argv) > 1 else "list")
        return
    figure(*argv[:2]) if argv else figure()


if __name__ == "__main__":
    main(sys.argv[1:])
