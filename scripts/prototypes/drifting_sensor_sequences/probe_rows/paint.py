"""Draw the two-code contour panel from ``panel.py``'s payload.

Line contours only, on one shared level array, every null set visible.  The
generating state's own field is drawn beneath both reconstructions, grey and
dashed, with its axis and its saddle marked, so a reader sees what each code
was fitted to and how far each code's nulls stand from the ones that generated
the data.  The subject, nova, carries the heavier contour and the filled
markers; imas-efit is a clean green drawn lighter and with open markers, since
the caption calls it the comparison and two filled sets of nearly coincident
markers cannot be told apart.  The wall is the machine's limiter ring.  A
committed figure is an SVG written through the repository's checked save
function, which refuses a figure whose labels clip or overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.path
import numpy as np

# The subject is blue; imas-efit is a clean green.  The burnt orange used before
# reads as dirt beside a clean hue and is reserved for agentic marks.
NOVA = "#1f5fa8"
EFIT = "#2e8b57"
REFERENCE = "#4d4d4d"
WIDTH = 7.2
HEIGHT = 7.2


def draw(axes, data, codes=("nova", "efit")):
    import matplotlib.lines as lines

    from nova.media.poloidal import draw_flux_contours, draw_wall

    radius = np.asarray(data["radius"], float)
    height = np.asarray(data["height"], float)
    levels = np.asarray(data["levels"], float)
    wall_r = np.asarray(data["wall_r"], float)
    wall_z = np.asarray(data["wall_z"], float)
    grid_r, grid_z = np.meshgrid(radius, height)
    ring = matplotlib.path.Path(np.column_stack([wall_r, wall_z]))
    inside = ring.contains_points(
        np.column_stack([grid_r.ravel(), grid_z.ravel()])
    ).reshape(grid_r.shape)
    # A raster carries field across its whole rectangle; only the vessel
    # interior is a field the machine can hold, so points outside the limiter
    # ring carry nothing to contour.
    blank = np.where(inside, 1.0, np.nan)
    if "psi_ref" in data:
        draw_flux_contours(
            axes,
            radius,
            height,
            np.asarray(data["psi_ref"], float).T * blank,
            levels,
            color=REFERENCE,
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
    # The subject is the figure's, so it is the heavier line and it is drawn
    # last: a comparison contour a third thicker than the subject inverts what
    # the figure is about.
    ink = {"nova": (NOVA, 1.6), "efit": (EFIT, 0.9)}
    for code in sorted(codes, key=lambda name: name == "nova"):
        colour, width = ink[code]
        draw_flux_contours(
            axes,
            radius,
            height,
            (np.asarray(data["psi_%s" % code], float).T * blank),
            levels,
            color=colour,
            linewidth=width,
            zorder=4 if code == "nova" else 3,
        )
    draw_wall(axes, radius=wall_r, height=wall_z)
    # The comparison code's markers are open, which is what the caption says
    # and what lets both null sets stay legible where they nearly coincide.
    efit_axis = np.asarray(data["axis_efit"], float).ravel()
    axes.plot(
        [float(efit_axis[0])],
        [float(efit_axis[1])],
        marker="s",
        markersize=10.0,
        markerfacecolor="white",
        markeredgecolor=EFIT,
        markeredgewidth=2.0,
        linestyle="none",
        zorder=5,
    )
    efit_points = np.asarray(data["x_points_efit"], float).reshape(-1, 2)
    efit_points = (
        efit_points[np.all(np.isfinite(efit_points), axis=1)]
        if efit_points.size
        else efit_points
    )
    if efit_points.size:
        axes.plot(
            efit_points[:, 0],
            efit_points[:, 1],
            marker="D",
            markersize=11.0,
            markerfacecolor="white",
            markeredgecolor=EFIT,
            markeredgewidth=2.0,
            linestyle="none",
            zorder=5,
        )
    if "psi_nova" in data:
        nova_axis = np.asarray(data["axis_nova"], float).ravel()
        axes.plot(
            [float(nova_axis[0])],
            [float(nova_axis[1])],
            marker="^",
            markersize=11.0,
            markerfacecolor=NOVA,
            markeredgecolor=NOVA,
            linestyle="none",
            zorder=6,
        )
        nova_points = np.asarray(data["x_points_nova"], float).reshape(-1, 2)
        if nova_points.size:
            axes.plot(
                nova_points[:, 0],
                nova_points[:, 1],
                marker="P",
                markersize=11.0,
                markerfacecolor=NOVA,
                markeredgecolor=NOVA,
                linestyle="none",
                zorder=6,
            )
    if "psi_ref" in data:
        ref_axis = np.asarray(data["axis_ref"], float).ravel()
        axes.plot(
            [float(ref_axis[0])],
            [float(ref_axis[1])],
            marker="o",
            markersize=10.0,
            markerfacecolor="none",
            markeredgecolor=REFERENCE,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=2,
        )
        ref_points = np.asarray(data["x_points_ref"], float).reshape(-1, 2)
        ref_points = ref_points[np.all(np.isfinite(ref_points), axis=1)]
        if ref_points.size:
            axes.plot(
                ref_points[:, 0],
                ref_points[:, 1],
                marker="x",
                markersize=9.0,
                markeredgecolor=REFERENCE,
                markeredgewidth=1.6,
                linestyle="none",
                zorder=2,
            )
    axes.set_xlim(wall_r.min() - 0.35, wall_r.max() + 0.35)
    axes.set_ylim(wall_z.min() - 0.35, wall_z.max() + 0.35)
    axes.set_aspect("equal")
    axes.set_axis_off()
    # An axis turned off still carries its tick label artists, and the checked
    # save reads every text artist whether it is drawn or not.
    axes.set_xticks([])
    axes.set_yticks([])
    labels = {"nova": "nova", "efit": "imas-efit"}
    handles = [
        lines.Line2D(
            [],
            [],
            color=ink[code][0],
            lw=2.4 if code == "nova" else 1.6,
            label=labels[code],
        )
        for code in sorted(codes, key=lambda name: name == "nova")
    ]
    if "psi_ref" in data:
        handles.append(
            lines.Line2D(
                [], [], color=REFERENCE, lw=1.2, ls="--", label="generating state"
            )
        )
    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codes", default="nova,efit")
    parser.add_argument("--png", default="")
    args = parser.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.edge_sheet.figure import save_checked

    data = np.load(args.input)
    figure, axes = plt.subplots(figsize=(WIDTH, HEIGHT))
    codes = tuple(code for code in args.codes.split(",") if code)
    handles = draw(axes, data, codes)
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = save_checked(figure, str(output), png=args.png or None)
    print("%s labels=%d" % (output, len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
