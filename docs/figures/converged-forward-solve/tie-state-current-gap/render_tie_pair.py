"""Render the tie field's independent contour topology beside the read's choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes

HERE = Path(__file__).resolve().parent
CELL_COLOUR = "#1f6f8b"
READ_COLOUR = "#b46b2d"
FIELD_COLOUR = "#4a4a4a"


def draw_panel(axes, row, data, wall, physical_levels, zoom):
    """Draw one offset from an independently sampled physical field."""
    key = str(row["offset"]).replace(".", "_")
    r, z = data["r"], data["z"]
    flux = np.ma.masked_where(~data[key + "_inside"], data[key + "_flux"])
    axis = next(point for point in row["independent"] if point["kind"] == "maximum")
    saddles = sorted(
        (point for point in row["independent"] if point["kind"] == "saddle"),
        key=lambda point: point["position"][1],
    )
    lower, upper = saddles
    units = (WallUnit(wall[:, 0], wall[:, 1], kind="vessel", closed=True),)
    poloidal_axes(axes)
    if zoom:
        persistence = axis["level"] - upper["level"]
        display = (flux - upper["level"]) / persistence
        levels = [0, 0.1, 0.25, 0.5, 0.75]
        axes.contour(
            r,
            z,
            display,
            levels=levels[1:],
            colors=FIELD_COLOUR,
            linewidths=1.1,
            zorder=1,
        )
        axes.contour(
            r, z, display, levels=[0], colors=CELL_COLOUR, linewidths=2.0, zorder=7
        )
        axes.set_xlim(1.37, 1.67)
        axes.set_ylim(0.77, 1.13)
    else:
        axes.contour(
            r,
            z,
            flux,
            levels=physical_levels,
            colors=FIELD_COLOUR,
            linewidths=0.9,
            linestyles="solid",
            zorder=1,
        )
        axes.contour(
            r,
            z,
            flux,
            levels=[upper["level"]],
            colors=CELL_COLOUR,
            linewidths=1.8,
            zorder=7,
        )
        axes.set_xlim(float(wall[:, 0].min()) - 0.04, float(wall[:, 0].max()) + 0.04)
        axes.set_ylim(float(wall[:, 1].min()) - 0.04, float(wall[:, 1].max()) + 0.04)
    poloidal.draw_wall(axes, units=units, linestyle="solid")
    for cell in row["read"]["support"]:
        polygon = np.asarray(cell["vertices"])
        axes.plot(
            np.r_[polygon[:, 0], polygon[0, 0]],
            np.r_[polygon[:, 1], polygon[0, 1]],
            color=CELL_COLOUR,
            lw=2.5,
            zorder=5,
        )
    poloidal.draw_nulls(
        axes,
        magnetic_axis=np.asarray(axis["position"]),
        x_points=np.asarray([upper["position"]]),
        other_x_points=np.asarray([lower["position"]]),
        contain=units,
    )
    read = row["read"]
    for point, size in ((read["axis"], 130), (read["saddle"], 220)):
        axes.scatter(
            point[0],
            point[1],
            marker="o",
            s=size,
            facecolors="none",
            edgecolors=READ_COLOUR,
            linewidths=1.8,
            zorder=9,
        )
    axes.set_title(
        f"{row['offset']:.5f}" + (" · saddle detail" if zoom else " · vessel"),
        loc="left",
        pad=12,
        fontsize=14,
    )


def render(receipt, field_data, figure_path, preview):
    rows = json.loads(Path(receipt).read_text())
    with np.load(field_data) as data:
        wall = data["wall"]
        physical_levels = [-0.01, -0.007, -0.004, -0.001, 0.002, 0.005, 0.007, 0.0085]
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 11.5),
            facecolor=DEFAULT_INK.figure_facecolor,
            gridspec_kw={"height_ratios": [1.3, 1]},
        )
        for column, row in enumerate(rows):
            draw_panel(axes[0, column], row, data, wall, physical_levels, False)
            draw_panel(axes[1, column], row, data, wall, physical_levels, True)
        legend = [
            Line2D([], [], color=FIELD_COLOUR, lw=1.2, label="field contours"),
            Line2D(
                [],
                [],
                color=CELL_COLOUR,
                lw=2,
                label="top saddle level and booked cells",
            ),
            Line2D(
                [],
                [],
                marker=DEFAULT_INK.axis_marker,
                color=DEFAULT_INK.axis_color,
                linestyle="none",
                markersize=10,
                label="field maximum",
            ),
            Line2D(
                [],
                [],
                marker=DEFAULT_INK.xpoint_marker,
                color=DEFAULT_INK.xpoint_color,
                linestyle="none",
                markersize=10,
                label="field top saddle",
            ),
            Line2D(
                [],
                [],
                marker=DEFAULT_INK.xpoint_marker,
                markerfacecolor="none",
                markeredgecolor=DEFAULT_INK.xpoint_color,
                linestyle="none",
                markersize=10,
                label="field bottom saddle",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=READ_COLOUR,
                linestyle="none",
                markersize=12,
                label="read axis and saddle candidate",
            ),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            fontsize=13,
        )
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.04, hspace=0.12
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, format="svg")
        if preview:
            preview.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview, format="png", dpi=100)
        plt.close(fig)
    print(f"FIGURE={figure_path}")
    if preview:
        print(f"PREVIEW={preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument(
        "--figure", type=Path, default=HERE / "tie-pair-current-gap.svg"
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    render(args.receipt, args.field, args.figure, args.preview)


if __name__ == "__main__":
    main()
