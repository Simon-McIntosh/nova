"""Plot the measured operator bound and the refusal's witness.

The left panel reads every receipt's largest metric-weighted quotient against
the bound, on an axis scaled to the readings so a difference of parts per
million is visible.  The right panel maps the share of the graded operator's
witness over the iron cross-section with the wall drawn, so the region carrying
the direction is visible rather than only tabulated.

    python scripts/iron_operator_bound/figure.py --output <run-dir> \
        --figure <docs/figures/.../operator-bound.png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from nova.jax.config import configure_dtypes  # noqa: E402

configure_dtypes()

from nova.biot import magnetisation as iron  # noqa: E402

BOUND = 1.0
SUBJECT = "#3b6ea5"
LIMIT = "#c62828"
POPULATION = "#929292"
SHARE = 0.01
LABELS = (
    ("uniform-0.08", "uniform 80"),
    ("graded-16-80", "graded 16/80"),
    ("uniform-0.04", "uniform 40"),
)


def receipt(output, name):
    """Return one receipt as a dictionary, or None when it is absent."""
    path = output / name
    return json.loads(path.read_text()) if path.exists() else None


def style():
    """Apply the shared inline style sheet, or its rules by hand."""
    try:
        plt.style.use("data-ink")
    except OSError:
        matplotlib.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
                "legend.frameon": False,
            }
        )


def bound_panel(axis, output):
    """Draw each operator's largest quotient against the bound.

    The quotients sit within a few parts per million of the bound while the
    planted control sits five percent above it, so a bar from zero draws every
    operator as the same shape and a linear zoom on the operators pushes the
    control off the panel.  The axis therefore carries the excess over the bound
    on a symmetric-log scale: zero is the bound and five decades of reading are
    legible at once.  Every measured reading falls past the bound, so the
    non-positive half of the axis is empty and is not drawn.
    Each mark is labelled with its excess over the bound, which is the quantity
    the axis states, and the series are told apart by colour and marker: the
    graded operator under study takes the limit colour, the uniform pair is the
    neutral population it is read against, and the planted control carries its
    own open marker because it tests the instrument rather than the operator.
    """
    rows = []
    for label, title in LABELS:
        row = receipt(output, f"bound-{label}.json")
        if row is not None:
            rows.append((title, float(row["maximum"]), label))
    control = receipt(output, "control-uniform-0.08.json")
    if control is not None:
        rows.append(("planted control", float(control["maximum"]), "control"))
    names = [name for name, _, _ in rows]
    values = np.asarray([value for _, value, _ in rows])
    excess = values - BOUND
    position = np.arange(len(rows))
    scale = max(float(np.max(np.abs(excess))), 1e-12)
    axis.set_xscale("symlog", linthresh=min(1e-7, scale), linscale=0.6)
    axis.set_xlim(0.0, 1.6 * scale)
    decade = int(np.floor(np.log10(scale)))
    steps = [10.0**power for power in (decade, decade - 2, decade - 4) if power >= -7]
    ticks = sorted({0.0} | set(steps))
    axis.set_xticks(ticks)
    axis.xaxis.set_minor_locator(NullLocator())

    def tick_label(tick):
        if tick == 0.0:
            return "0"
        exponent = int(round(np.log10(abs(tick))))
        return "$10^{%d}$" % exponent

    axis.set_xticklabels([tick_label(tick) for tick in ticks])
    for index, offset, kind in zip(position, excess, (row[2] for row in rows)):
        if kind == "graded-16-80":
            colour, face, marker = LIMIT, LIMIT, "o"
        elif kind == "control":
            colour, face, marker = "#1d1d1d", "none", "s"
        else:
            colour, face, marker = POPULATION, POPULATION, "o"
        axis.plot(
            [offset],
            [index],
            marker=marker,
            markersize=9,
            markerfacecolor=face,
            markeredgecolor=colour,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=3,
        )
        axis.text(
            offset,
            index - 0.24,
            f"{offset:+.2e}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=colour,
        )
    axis.set_yticks(position)
    axis.set_yticklabels(names)
    axis.set_xlabel("largest $x^{T}QGx\\,/\\,(\\mu_0\\,x^{T}Qx)$ minus the bound")
    axis.set_ylim(len(rows) - 0.5, -0.9)


def witness_panel(axis, output, inputs):
    """Draw the cells that carry the graded witness over the cross-section.

    The share is per cell, and the direction's energy sits in a handful of
    them, so every cell is drawn as the population in one neutral and the cells
    at or above ``SHARE`` are drawn over them in the subject colour.  A filled
    colour map would spend ink on cells the witness does not touch and would
    put a scale beside a quantity the page states in numbers.
    """
    shares = np.load(output / "shares-graded-16-80.npy")
    from scripts.iron_operator_bound.persisted import cell_size

    field = cell_size("graded-16-80", inputs)
    mesh = iron._mesh(inputs["region"], field)
    polygons = [np.asarray(polygon) for polygon in mesh.polygons]
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=POPULATION,
            edgecolors="none",
            linewidths=0.0,
            zorder=1,
            label="iron",
        )
    )
    carrying = np.flatnonzero(np.abs(shares) >= SHARE)
    centroids = np.asarray(mesh.centroids)
    share = np.abs(shares[carrying])
    axis.scatter(
        centroids[carrying, 0],
        centroids[carrying, 1],
        s=60.0 + 900.0 * share / max(float(share.max()), 1e-12),
        color=SUBJECT,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="carries the witness",
    )
    region = inputs["region"]
    parts = [region] if region.geom_type == "Polygon" else list(region.geoms)
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            coordinates = np.asarray(ring.coords)
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#1d1d1d",
                linewidth=1.2,
                zorder=3,
            )
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    return len(carrying)


def check_labels(figure):
    """Refuse to write a figure whose text leaves the canvas or overlaps.

    A figure is an artifact: a clipped axis unit or two labels on top of each
    other is a defect in the deliverable, not a cosmetic preference.  The check
    names the offending text so the placement can be fixed rather than the
    check suppressed.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    boxes = []
    for text in figure.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        extent = text.get_window_extent(renderer)
        if extent.width == 0.0 or extent.height == 0.0:
            continue
        if (
            extent.x0 < canvas.x0
            or extent.x1 > canvas.x1
            or extent.y0 < canvas.y0
            or extent.y1 > canvas.y1
        ):
            raise SystemExit(f"text runs off the canvas: {text.get_text()!r}")
        boxes.append((text.get_text(), extent))
    for position, (first, one) in enumerate(boxes):
        for second, other in boxes[position + 1 :]:
            if one.overlaps(other):
                raise SystemExit(f"text overlaps: {first!r} and {second!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    style()
    from scripts.iron_graded_refinement.measure import west_inputs

    inputs = west_inputs()
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 7.2))
    bound_panel(left, args.output)
    carrying = witness_panel(right, args.output, inputs)
    print(f"{carrying} cells carry at least {SHARE:.0%} of the witness")
    figure.tight_layout()
    check_labels(figure)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=100)
    figure.savefig(args.figure.with_suffix(".svg"))
    print(args.figure)


if __name__ == "__main__":
    main()
