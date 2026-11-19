"""Draw the two-code contour panel from ``panel.py``'s payload.

Line contours only, on one shared level array, both null sets in their own
styles: solid triangle for the axis, filled cross for the admitted saddle,
hollow markers for other nulls.  The wall is drawn from the machine's limiter
ring.  ``--codes`` selects the codes a panel draws, so one code can be shown
alone; ``--raster`` writes a 200 dpi PNG where a reader wants a raster, and no
raster is written beside a committed SVG.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib
import matplotlib.path
import numpy as np

BLUE = "#1f5fa8"
ORANGE = "#c06a1b"


def draw(axes, data, codes=("nova", "efit")):
    import matplotlib.lines as lines

    from nova.media.ink import DEFAULT_INK
    from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

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
    # interior is a field the machine can hold, so every point outside the
    # limiter ring carries nothing to contour.
    blank = np.where(inside, 1.0, np.nan)
    ink = {"nova": (BLUE, 0.6), "efit": (ORANGE, 1.4)}
    for code in codes:
        colour, width = ink[code]
        draw_flux_contours(
            axes,
            radius,
            height,
            (np.asarray(data["psi_%s" % code], float).T * blank),
            levels,
            color=colour,
            linewidth=width,
        )
    draw_wall(axes, radius=wall_r, height=wall_z)
    for code in codes:
        colour = ink[code][0]
        style = dataclasses.replace(DEFAULT_INK, axis_color=colour, xpoint_color=colour)
        points = np.asarray(data["x_points_%s" % code], float).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)]
        draw_nulls(
            axes,
            magnetic_axis=np.asarray(data["axis_%s" % code], float),
            x_points=points if points.size else None,
            style=style,
        )
    axes.set_xlim(wall_r.min() - 0.35, wall_r.max() + 0.35)
    axes.set_ylim(wall_z.min() - 0.35, wall_z.max() + 0.35)
    axes.set_aspect("equal")
    axes.set_axis_off()
    labels = {"nova": "nova, reconstructed slice", "efit": "imas-efit, same slice"}
    inks = {"nova": BLUE, "efit": ORANGE}
    return [
        lines.Line2D(
            [],
            [],
            color=inks[code],
            lw=0.8 if code == "nova" else 1.4,
            label=labels[code],
        )
        for code in codes
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--codes", default="nova,efit")
    parser.add_argument("--raster", default="")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.input)
    figure, axes = plt.subplots(figsize=(5.6, 7.8))
    codes = tuple(code for code in args.codes.split(",") if code)
    handles = draw(axes, data, codes)
    axes.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    if args.title:
        axes.set_title(args.title, fontsize=9, color="#333333")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    # A raster is for inspection only; the committed figure is the SVG.
    if args.raster:
        figure.savefig(Path(args.raster), dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
