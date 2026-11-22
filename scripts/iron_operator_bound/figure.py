"""Plot the measured operator bound and the refusal's witness.

The left panel reads every receipt's largest metric-weighted quotient against
the bound and against the seed-state limit the origin slope implies.  The right
panel maps the share of the graded operator's witness over the iron
cross-section with the wall drawn, so the region carrying the direction is
visible rather than only tabulated.

    python scripts/iron_operator_bound/figure.py --output <run-dir> \
        --figure <docs/figures/.../operator-bound.png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from nova.jax.config import configure_dtypes  # noqa: E402

configure_dtypes()

from nova.biot import magnetisation as iron  # noqa: E402

BOUND = 1.0
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
    """Draw each operator's largest quotient against the bound and the limit."""
    rows = []
    for label, title in LABELS:
        row = receipt(output, f"bound-{label}.json")
        if row is not None:
            rows.append((title, float(row["maximum"])))
    control = receipt(output, "control-uniform-0.08.json")
    if control is not None:
        rows.append(("planted control", float(control["maximum"])))
    names = [name for name, _ in rows]
    values = np.asarray([value for _, value in rows])
    position = np.arange(len(rows))
    axis.barh(position, values, height=0.52, color="#3b6ea5", zorder=2)
    axis.axvline(BOUND, color="#4d4d4d", linewidth=1.2, linestyle=":", zorder=3)
    witness = receipt(output, "witness-graded-16-80.json")
    head = -0.75
    if witness is not None:
        limit = float(witness["seed_definiteness_limit"])
        axis.axvline(limit, color="#8a5a00", linewidth=1.2, linestyle="--", zorder=3)
        axis.text(
            limit,
            head,
            "seed limit",
            color="#8a5a00",
            fontsize=14,
            ha="right",
            va="center",
        )
    axis.text(
        BOUND, head, "bound", color="#4d4d4d", fontsize=14, ha="left", va="center"
    )
    for index in position:
        axis.text(
            values[index] * 1.01,
            index,
            f"{values[index]:.3f}",
            va="center",
            ha="left",
            fontsize=14,
            color="#222222",
        )
    axis.set_yticks(position)
    axis.set_yticklabels(names)
    axis.set_xlabel("largest $x^{T}QGx\\,/\\,(\\mu_0\\,x^{T}Qx)$")
    axis.set_ylim(len(rows) - 0.45, -1.1)
    axis.invert_yaxis()


def witness_panel(axis, output, inputs):
    """Draw the graded witness's share over the iron cross-section.

    The share is signed, so the map is diverging about zero: a cell either adds
    to the numerator the bound constrains or subtracts from it.
    """
    shares = np.load(output / "shares-graded-16-80.npy")
    from scripts.iron_operator_bound.persisted import cell_size

    field = cell_size("graded-16-80", inputs)
    mesh = iron._mesh(inputs["region"], field)
    polygons = [np.asarray(polygon) for polygon in mesh.polygons]
    magnitude = float(np.max(np.abs(shares)))
    collection = PolyCollection(
        polygons,
        array=shares,
        cmap="RdBu_r",
        clim=(-magnitude, magnitude),
        edgecolors="none",
        linewidths=0.0,
    )
    axis.add_collection(collection)
    region = inputs["region"]
    parts = [region] if region.geom_type == "Polygon" else list(region.geoms)
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            coordinates = np.asarray(ring.coords)
            axis.plot(
                coordinates[:, 0], coordinates[:, 1], color="#1d1d1d", linewidth=1.2
            )
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.set_axis_off()
    return collection


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
    collection = witness_panel(right, args.output, inputs)
    bar = figure.colorbar(
        collection, ax=right, orientation="horizontal", shrink=0.55, pad=0.02
    )
    bar.set_label("cell share of $w^{T}QGw$")
    bar.outline.set_visible(False)
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=100)
    figure.savefig(args.figure.with_suffix(".svg"))
    print(args.figure)


if __name__ == "__main__":
    main()
