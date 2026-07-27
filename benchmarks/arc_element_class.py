"""Evidence for the polygon-section arc's element class: the section, the gap, the cost.

Three panels, one per claim the class rests on.

**Left -- what the frame's section descriptor means, and what ``Bow`` integrates
instead.**  A swept winding of hexagonal section is routed to ``Bow`` today, which
thickens its arc into the RECTANGLE its width and height bound while normalising
by the frame's area, which is the hexagon's.  The panel draws both and the ratio
the mismatch produces, which is the reciprocal of the area fraction a regular
hexagon fills -- exactly four thirds, on every row, because the error is a
normalisation rather than a shape.

**Middle -- the element class is the reduction and a transform, and the transform
is measured.**  Each row's worst deviation between the class and the bare
reduction, over four sweeps: the class's rows are formed in the source's own local
frame, rotated to global and read back in the target's cylindrical basis, so the
panel measures a round trip through two rotation matrices and nothing else.
Beside it, the same rows against ``Bow`` at zero edge slope, which is four orders
coarser because it is a fixed-node zeta quadrature's accuracy rather than a
transform's round-off.

**Right -- the cost, measured through the frame so the two elements are compared
on the same terms.**  ``Bow`` and this class over the same target cloud, the same
sweep and the same coilset, on the rectangle they share and on the hexagon only
one of them can represent.  The bare reduction is marked on the rectangle bar,
where the frame's section is the same one the bare variant uses, so the gap
between them is what the frame itself costs.

Timings are read from a JSON file of ``benchmarks/arc_section_cost.py`` records --
one variant per process on a compute node -- rather than measured here, because a
figure script is the last place a benchmark should run.  Run as
``python benchmarks/arc_element_class.py [output.png] [cost.json]``.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from nova.biot.polygonarc import polygon_arc_greens
from nova.frame.coilset import CoilSet
from nova.geometry.polygen import PolyGen

RADIUS, ELEVATION = 3.0, 0.2
WIDTH, THICKNESS = 0.06, 0.04
FIELD_ATTRS = ["Ax", "Ay", "Bx", "By", "Bz"]
ROWS = (r"$A_r$", r"$A_\phi$", r"$B_r$", r"$B_\phi$", r"$B_z$")

SWEEPS = {
    "0.3 → 1.9": (0.3, 1.9),
    "−1.0 → 2.5": (-1.0, 2.5),
    "5.6 → 7.4": (5.6, 7.4),
    "0 → π": (0.0, np.pi),
}

TARGET_R = np.array([3.5, 2.5, 3.1, 3.0, 3.3])
TARGET_PHI = np.array([0.2, 1.0, 2.4, -0.4, 4.5])
TARGET_Z = np.array([0.5, -0.2, 0.25, 0.35, -0.5])

# Measured 2026-07-26, sun_debug node, one core, fresh process per variant, median
# of 3, 512 pairs.  Overridden by a JSON file of the benchmark's own records.
COST = {
    "bow-rectangle": 182.1,
    "polybow-rectangle": 550.1,
    "bow-hexagon": 186.5,
    "polybow-hexagon": 926.6,
    "arc-rectangle": 472.2,
}


def solved(section, sweep, segment=None):
    """Return a solved coilset carrying one swept winding of that section."""
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
        section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    coilset.point.solve(
        np.stack(
            [TARGET_R * np.cos(TARGET_PHI), TARGET_R * np.sin(TARGET_PHI), TARGET_Z],
            axis=-1,
        )
    )
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


def corners(coilset):
    """Return the section corners the frame's descriptor implies, at the arc."""
    name = PolyGen(str(np.asarray(coilset.subframe["section"])[0])).shape
    shape = PolyGen(name)(
        RADIUS,
        ELEVATION,
        float(np.asarray(coilset.subframe["width"])[0]),
        float(np.asarray(coilset.subframe["height"])[0]),
    )
    points = np.asarray(shape.exterior.coords, dtype=np.float64)
    gap = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return points[gap > 1e-9 * np.max(np.ptp(points, axis=0))]


def section(axes):
    """Draw the hexagon against the box Bow integrates, and the ratio it costs."""
    hexagonal = solved({"hex": (0, 0, WIDTH, THICKNESS)}, SWEEPS["0.3 → 1.9"], "bow")
    cell = corners(hexagonal)
    width = float(np.asarray(hexagonal.subframe["width"])[0])
    height = float(np.asarray(hexagonal.subframe["height"])[0])
    box = np.array(
        [
            [RADIUS - width / 2, ELEVATION - height / 2],
            [RADIUS + width / 2, ELEVATION - height / 2],
            [RADIUS + width / 2, ELEVATION + height / 2],
            [RADIUS - width / 2, ELEVATION + height / 2],
        ]
    )
    for shape, colour, label in ((box, "C3", "Bow's box"), (cell, "C0", "the section")):
        loop = np.vstack([shape, shape[:1]])
        axes.fill(loop[:, 0], loop[:, 1], color=colour, alpha=0.16)
        axes.plot(loop[:, 0], loop[:, 1], color=colour, lw=1.8, label=label)
    axes.set_aspect("equal")
    axes.set_xlabel("r [m]")
    axes.set_ylabel("z [m]")
    axes.legend(loc="lower left", fontsize=8, framealpha=0.9)

    ratio = cylindrical_rows(hexagonal) / cylindrical_rows(
        solved({"hex": (0, 0, WIDTH, THICKNESS)}, SWEEPS["0.3 → 1.9"], "polybow")
    )
    axes.text(
        RADIUS,
        ELEVATION,
        "every row, every target\n"
        f"bow / polybow = {np.min(ratio):.4f} … {np.max(ratio):.4f}\n"
        r"$4/3$ = 1.3333, the reciprocal of the area"
        "\nfraction a regular hexagon fills",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92),
    )
    axes.set_title(
        "A hexagonal winding routes to Bow, which\n"
        "integrates the box its width and height bound",
        fontsize=9,
    )


def transform(axes):
    """Plot the class against the bare reduction, and against Bow, per row."""
    against_bare = np.zeros(5)
    against_bow = np.zeros(5)
    for sweep in SWEEPS.values():
        rectangle = {"rect": (0, 0, WIDTH, THICKNESS)}
        through = solved(rectangle, sweep, "polybow")
        got = cylindrical_rows(through)
        bare = np.stack(
            polygon_arc_greens(TARGET_R, TARGET_Z, TARGET_PHI, corners(through), *sweep)
        )
        bow = cylindrical_rows(solved(rectangle, sweep))
        scale = np.max(np.abs(bare), axis=1)
        against_bare = np.maximum(
            against_bare, np.max(np.abs(got - bare), axis=1) / scale
        )
        against_bow = np.maximum(
            against_bow, np.max(np.abs(bow - bare), axis=1) / scale
        )
    position = np.arange(5)
    axes.bar(
        position - 0.19, against_bare, 0.38, color="C0", label="class vs reduction"
    )
    axes.bar(position + 0.19, against_bow, 0.38, color="C1", label="Bow vs reduction")
    axes.set_yscale("log")
    axes.set_xticks(position)
    axes.set_xticklabels(ROWS)
    axes.set_ylabel("worst deviation, each row against its own scale")
    axes.set_ylim(top=10.0 ** np.ceil(np.log10(np.max(against_bow)) + 1.4))
    axes.legend(loc="upper left", fontsize=8, ncols=2)
    axes.grid(axis="y", alpha=0.3)
    axes.set_title(
        "The class adds a frame transform and nothing else;\n"
        "Bow's own quadrature is four orders coarser",
        fontsize=9,
    )


def cost(axes, table):
    """Draw the per-pair cost of the two elements, measured through the frame."""
    position = np.arange(2)
    bow = [table["bow-rectangle"], table["bow-hexagon"]]
    poly = [table["polybow-rectangle"], table["polybow-hexagon"]]
    axes.bar(position - 0.2, bow, 0.4, color="C1", label="Bow")
    axes.bar(position + 0.2, poly, 0.4, color="C0", label="PolyBow")
    axes.plot(
        [0.2],
        [table["arc-rectangle"]],
        "_",
        ms=26,
        mew=2.0,
        color="0.2",
        label="the reduction alone",
    )
    for index, (one, other) in enumerate(zip(bow, poly)):
        axes.text(index, other + 45, f"{other / one:.2f}×", ha="center", fontsize=9)
    axes.annotate(
        "and 4/3 out",
        xy=(0.8, bow[1]),
        xytext=(0.62, 430),
        ha="center",
        fontsize=8.5,
        color="C3",
        arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
    )
    axes.set_xticks(position)
    axes.set_xticklabels(["rectangle", "hexagon"])
    axes.set_ylabel("µs / pair, through the frame")
    axes.set_ylim(0, 1180)
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(axis="y", alpha=0.3)
    axes.set_title(
        "Cost on the same coilset and the same cloud:\n"
        "the frame is 16 % of the rectangle's reduction",
        fontsize=9,
    )


def main(path: str, records: str | None = None) -> None:
    """Build the three panels and write the figure."""
    table = dict(COST)
    if records is not None:
        with open(records) as source:
            for line in source:
                if line.startswith("{"):
                    entry = json.loads(line)
                    table[entry["variant"]] = entry["microseconds_per_pair"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    section(axes[0])
    transform(axes[1])
    cost(axes[2], table)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    print(path)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "arc_element_class.png",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
