"""Draw the two-code contour panel from ``panel.py``'s payload.

Line contours only, on one shared level array, both null sets in their own
styles: solid triangle for the axis, filled cross for the admitted saddle,
hollow markers for other nulls.  The wall is drawn from the machine's limiter
ring.  The raster is written at 200 dpi beside the SVG so a reader can inspect
it without an SVG viewer.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

BLUE = "#1f5fa8"
ORANGE = "#c06a1b"


def draw(axes, data):
    import matplotlib.lines as lines

    from nova.media.ink import DEFAULT_INK
    from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

    radius = np.asarray(data["radius"], float)
    height = np.asarray(data["height"], float)
    levels = np.asarray(data["levels"], float)
    draw_flux_contours(
        axes,
        radius,
        height,
        np.asarray(data["psi_nova"], float),
        levels,
        color=BLUE,
        linewidth=0.7,
    )
    draw_flux_contours(
        axes,
        radius,
        height,
        np.asarray(data["psi_efit"], float),
        levels,
        color=ORANGE,
        linewidth=0.6,
        linestyles="dashed",
    )
    wall_r = np.asarray(data["wall_r"], float)
    wall_z = np.asarray(data["wall_z"], float)
    draw_wall(axes, radius=wall_r, height=wall_z)
    for code, colour in (("nova", BLUE), ("efit", ORANGE)):
        style = dataclasses.replace(DEFAULT_INK, axis_color=colour, xpoint_color=colour)
        points = np.asarray(data["x_points_%s" % code], float).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)]
        draw_nulls(
            axes,
            magnetic_axis=np.asarray(data["axis_%s" % code], float),
            x_points=points if points.size else None,
            style=style,
        )
    axes.set_aspect("equal")
    axes.set_axis_off()
    return [
        lines.Line2D([], [], color=BLUE, lw=0.9, label="nova"),
        lines.Line2D([], [], color=ORANGE, lw=0.9, ls="--", label="imas-efit"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.input)
    figure, axes = plt.subplots(figsize=(5.6, 7.8))
    handles = draw(axes, data)
    axes.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    if args.title:
        axes.set_title(args.title, fontsize=9, color="#333333")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".png"), dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
