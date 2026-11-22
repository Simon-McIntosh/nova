"""Draw the two-code contour panel from ``panel.py``'s payload.

Line contours only, on one shared level array, both null sets visible.  Each
code carries its own colour and its own marker shape, so the two null sets stay
legible where they nearly coincide: nova, the subject, takes the filled
triangle for its axis and the filled cross for its admitted saddle on top of
the map; imas-efit takes a larger open square and a rotated cross in green,
drawn beneath.  The wall is the machine's limiter ring.  A committed figure is
an SVG written through the repository's checked save function, which refuses a
figure whose labels clip or overlap.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib
import matplotlib.path
import numpy as np

# The subject is blue; imas-efit is a clean green.  The burnt orange used before
# reads as dirt beside a clean hue and is reserved for agentic marks.
NOVA = "#1f5fa8"
EFIT = "#2e8b57"
WIDTH = 14.0
HEIGHT = 7.2


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
    # interior is a field the machine can hold, so points outside the limiter
    # ring carry nothing to contour.
    blank = np.where(inside, 1.0, np.nan)
    ink = {"nova": (NOVA, 0.9), "efit": (EFIT, 1.6)}
    # The comparison code goes down first, the subject last, so the map the
    # figure is about is never buried under the one it is set against.
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
        )
    draw_wall(axes, radius=wall_r, height=wall_z)
    styles = {
        "nova": DEFAULT_INK,
        "efit": dataclasses.replace(
            DEFAULT_INK,
            axis_marker="s",
            axis_markersize=9.5,
            xpoint_marker="+",
            xpoint_markersize=13.0,
            xpoint_markeredgewidth=2.0,
        ),
    }
    for code in sorted(codes, key=lambda name: name == "nova"):
        colour = ink[code][0]
        style = dataclasses.replace(
            styles[code],
            axis_color=colour,
            xpoint_color=colour,
            strike_color=colour,
            zorder_markers=6 if code == "nova" else 5,
        )
        points = np.asarray(data["x_points_%s" % code], float).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)] if points.size else points
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
    labels = {"nova": "nova", "efit": "imas-efit"}
    return [
        lines.Line2D(
            [],
            [],
            color=ink[code][0],
            lw=2.4 if code == "nova" else 1.4,
            label=labels[code],
        )
        for code in sorted(codes, key=lambda name: name == "nova")
    ]


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
    figure, axes = plt.subplots(figsize=(7.2, 7.2))
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