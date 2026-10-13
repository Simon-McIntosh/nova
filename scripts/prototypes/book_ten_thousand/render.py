"""Render the cell-current discrepancy on the diverted poloidal carrier."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import draw_nulls, draw_wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    direct = np.load(args.data / "diverted-10000-cells.npz")
    measured = np.load(args.data / "diverted-10000-combined.npz")
    centre = direct["centre"]
    delta = measured["booked"][0] - measured["reference"][0]
    scale = np.max(np.abs(measured["reference"]))
    peak = float(np.max(np.abs(delta)) / scale)
    assert np.count_nonzero(delta > 0.1 * peak * scale) > 0
    figure, axes = plt.subplots(figsize=(14, 8), dpi=100)
    mesh = tri.Triangulation(centre[:, 0], centre[:, 1])
    positive = np.maximum(delta / scale, 0.0)
    levels = peak * np.asarray((0.1, 0.3, 0.5, 0.7, 0.9))
    contours = axes.tricontour(
        mesh,
        positive,
        levels=levels,
        colors="#a33636",
        linewidths=(1.2, 1.5, 1.8, 2.1, 2.4),
    )
    axes.clabel(contours, inline=True, fontsize=12, fmt=lambda x: f"{x:.0%}")
    wall = direct["wall"]
    draw_wall(axes, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
    analytic_style = DEFAULT_INK.variant(
        axis_color="#555555",
        xpoint_color="#555555",
        axis_markersize=12,
        xpoint_markersize=13,
    )
    read_style = DEFAULT_INK.variant(
        axis_color="#a33636",
        xpoint_color="#a33636",
        axis_markersize=6,
        xpoint_markersize=7,
    )
    draw_nulls(
        axes,
        magnetic_axis=direct["axis"],
        x_points=direct["saddle"][None],
        style=analytic_style,
    )
    read_saddles = measured["read_x_points"][measured["read_x_point_valid"]]
    draw_nulls(
        axes,
        magnetic_axis=measured["read_axis"],
        x_points=read_saddles,
        style=read_style,
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()
    axes.margins(0.03)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="svg", facecolor="white")
    plt.close(figure)
    print(f"FIGURE={args.output} PEAK_RELATIVE_CELL_CURRENT_DIFFERENCE={peak:.12g}")


if __name__ == "__main__":
    main()
