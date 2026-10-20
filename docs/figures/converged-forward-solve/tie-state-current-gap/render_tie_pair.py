"""Render the tie field's independent contour topology beside the read's choices."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
        axes.set_ylim(0.70, 1.13)
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
        for ring in cell.get("rings", [cell["vertices"]]):
            polygon = np.asarray(ring)
            axes.plot(
                np.r_[polygon[:, 0], polygon[0, 0]],
                np.r_[polygon[:, 1], polygon[0, 1]],
                color="#333333",
                linestyle="--",
                lw=2.0,
                zorder=5,
            )
    poloidal.draw_nulls(
        axes,
        magnetic_axis=np.asarray(axis["position"]),
        x_points=np.asarray([upper["position"]]),
        other_x_points=np.asarray([lower["position"]]),
        style=replace(DEFAULT_INK, axis_color=FIELD_COLOUR, xpoint_color=FIELD_COLOUR),
        contain=units,
    )
    read = row["read"]
    read_style = replace(
        DEFAULT_INK,
        axis_color=READ_COLOUR,
        xpoint_color=READ_COLOUR,
        axis_markersize=6,
        xpoint_markersize=6,
    )
    roots = np.asarray(read["root_positions"])
    other = roots[
        (np.linalg.norm(roots - np.asarray(read["axis"]), axis=1) > 1e-8)
        & (np.linalg.norm(roots - np.asarray(read["saddle"]), axis=1) > 1e-8)
    ]
    poloidal.draw_nulls(
        axes,
        magnetic_axis=read["axis"],
        x_points=np.asarray([read["saddle"]]) if read["boundary_class"] else None,
        other_x_points=other,
        style=read_style,
        contain=units,
    )


def null_handles():
    """Capture legend handles from the same null artists used in the panels."""
    handles = []
    for label, keyword in (
        ("reference maximum", {"magnetic_axis": np.array([0.0, 0.0])}),
        ("reference separator", {"x_points": np.array([[0.0, 0.0]])}),
        ("other saddle", {"other_x_points": np.array([[0.0, 0.0]])}),
        (
            "read maximum",
            {
                "magnetic_axis": np.array([0.0, 0.0]),
                "style": replace(
                    DEFAULT_INK, axis_color=READ_COLOUR, axis_markersize=6
                ),
            },
        ),
        (
            "read separator",
            {
                "x_points": np.array([[0.0, 0.0]]),
                "style": replace(
                    DEFAULT_INK, xpoint_color=READ_COLOUR, xpoint_markersize=6
                ),
            },
        ),
    ):
        scratch, axis = plt.subplots()
        keyword.setdefault(
            "style",
            replace(DEFAULT_INK, axis_color=FIELD_COLOUR, xpoint_color=FIELD_COLOUR),
        )
        poloidal.draw_nulls(axis, **keyword)
        artist = axis.get_lines()[0]
        artist.set_label(label)
        handles.append(artist)
        plt.close(scratch)
    return handles


def check_text(figure):
    """Refuse clipped or overlapping visible labels before saving the artifact."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = [
        text
        for text in figure.findobj(matplotlib.text.Text)
        if text.get_visible() and text.get_text()
    ]
    boxes = [(text, text.get_window_extent(renderer)) for text in labels]
    for text, box in boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"clipped label: {text.get_text()}")
    for index, (text, box) in enumerate(boxes):
        for peer, other in boxes[index + 1 :]:
            if box.overlaps(other):
                raise ValueError(
                    f"overlapping labels: {text.get_text()} / {peer.get_text()}"
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
        (contour_handle,) = axes[0, 0].plot(
            [], [], color=FIELD_COLOUR, lw=1.2, label="field contours"
        )
        (support_handle,) = axes[0, 0].plot(
            [], [], color=CELL_COLOUR, lw=2, label="physical separator"
        )
        (rings_handle,) = axes[0, 0].plot(
            [], [], color="#333333", linestyle="--", lw=2, label="booked rings"
        )
        legend = [contour_handle, support_handle, rings_handle, *null_handles()]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.015),
            frameon=False,
            fontsize=13,
        )
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.04, hspace=0.12
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        check_text(fig)
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
