"""Draw the promoted nodes on the converged coil edit's poloidal field.

The panel carries the converged state's flux as line contours, the separatrix
the state assembles and the separatrix the field of the currents assembles,
both null sets, the first wall, and every promoted node.  Everything is read
from the measurement's own field archive, so the panel draws the numbers the
receipt reports rather than a second evaluation of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "promoted-nodes-near-xpoint"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
    sample_cubic_controls,
)

SUBJECT = "#d97706"
IMAGED = "#2563eb"
STATE = "#1f2937"
POPULATION = "#929292"


def _assert_text_inside(figure):
    """Refuse to write a figure whose text runs off the canvas or onto itself."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    boxes = []
    for text in figure.findobj(matplotlib.text.Text):
        label = text.get_text().strip()
        if not text.get_visible() or not label:
            continue
        box = text.get_window_extent(renderer)
        if box.width <= 0.0 or box.height <= 0.0:
            continue
        if (
            box.x0 < canvas.x0 - 1.0
            or box.y0 < canvas.y0 - 1.0
            or box.x1 > canvas.x1 + 1.0
            or box.y1 > canvas.y1 + 1.0
        ):
            raise ValueError(f"text {label!r} extends past the canvas")
        boxes.append((label, box))
    for index, (label, box) in enumerate(boxes):
        for other_label, other in boxes[index + 1 :]:
            overlap = box.intersection(box, other)
            if overlap is not None and overlap.width > 1.0 and overlap.height > 1.0:
                raise ValueError(f"text {label!r} overlaps {other_label!r}")


def render(
    fields_path: Path, receipt_path: Path, svg_path: Path, png_path: Path
) -> dict:
    fields = np.load(fields_path, allow_pickle=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    radius = np.asarray(fields["radius"], dtype=np.float64)
    height = np.asarray(fields["height"], dtype=np.float64)
    state_field = np.asarray(fields["state_field"], dtype=np.float64)
    promoted = np.asarray(fields["promoted_coordinate"], dtype=np.float64)
    wall_units = receipt["wall_units"]

    radius_mesh, height_mesh = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[radius_mesh.ravel(), height_mesh.ravel()]
    wall_polygon = np.concatenate(
        [np.c_[unit["radius"], unit["height"]] for unit in wall_units], axis=0
    )

    finite = np.isfinite(state_field)
    interior = state_field[finite]
    levels = np.linspace(
        float(np.quantile(interior, 0.02)), float(np.quantile(interior, 0.98)), 13
    )[1:-1]

    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    grid = figure.add_gridspec(1, 2, width_ratios=(2.1, 1.0), wspace=0.04)
    axis = figure.add_subplot(grid[0, 0])
    zoom = figure.add_subplot(grid[0, 1])
    poloidal_axes(axis)
    zoom.set_axis_off()
    zoom.grid(False)
    axis.margins(0.01)

    contours = draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        state_field.T.ravel(),
        levels,
        wall_polygon,
        color=POPULATION,
        linewidth=1.2,
    )
    for collection in contours if isinstance(contours, list) else (contours,):
        collection.set_zorder(1)

    state_polyline = np.asarray(fields["state_closed_polyline"], dtype=np.float64)
    image_polyline = np.asarray(fields["image_closed_polyline"], dtype=np.float64)
    axis.plot(
        state_polyline[:, 0],
        state_polyline[:, 1],
        color=STATE,
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label="separatrix assembled on the converged state",
    )
    axis.plot(
        image_polyline[:, 0],
        image_polyline[:, 1],
        color=IMAGED,
        linestyle="-",
        linewidth=2.0,
        zorder=4,
        label="separatrix assembled on the field of the currents",
    )
    for controls_key, valid_key, colour, style, width in (
        ("state_open_controls", "state_open_valid", STATE, "--", 1.6),
        ("image_open_controls", "image_open_valid", IMAGED, "-", 2.0),
    ):
        controls = np.asarray(fields[controls_key], dtype=np.float64)
        valid = np.asarray(fields[valid_key], dtype=bool)
        for slot in range(valid.shape[0]):
            leg = sample_cubic_controls(controls[slot], valid[slot])
            if leg.shape[0] > 1:
                axis.plot(
                    leg[:, 0],
                    leg[:, 1],
                    color=colour,
                    linestyle=style,
                    linewidth=width,
                    zorder=3 if style == "--" else 4,
                )

    for unit in wall_units:
        draw_wall(
            axis,
            np.asarray(unit["radius"], dtype=np.float64),
            np.asarray(unit["height"], dtype=np.float64),
            close=bool(unit["closed"]),
            linestyle=None if unit["closed"] else "--",
        )

    state_saddle = np.asarray(fields["state_saddle"], dtype=np.float64).reshape(2)
    image_saddle = np.asarray(fields["image_saddle"], dtype=np.float64).reshape(2)
    magnetic_axis = np.asarray(fields["axis"], dtype=np.float64).reshape(2)

    # Every mark takes its slot from the null painter rather than from a second
    # vocabulary: the axis goes in the axis slot and both saddles in the
    # X-point slot, each in its own colour, so a reader who knows the machine's
    # null marks reads the same shapes here.
    axis_style = DEFAULT_INK.variant(axis_color=STATE)
    state_style = DEFAULT_INK.variant(xpoint_color=STATE)
    image_style = DEFAULT_INK.variant(
        xpoint_color=IMAGED,
        xpoint_markersize=DEFAULT_INK.xpoint_markersize + 1.5,
    )
    draw_nulls(axis, magnetic_axis, style=axis_style, contain=wall_polygon)
    draw_nulls(axis, x_points=[state_saddle], style=state_style, contain=wall_polygon)
    draw_nulls(axis, x_points=[image_saddle], style=image_style, contain=wall_polygon)
    axis.scatter(
        promoted[:, 0],
        promoted[:, 1],
        s=26,
        marker="D",
        facecolor="none",
        edgecolor=SUBJECT,
        linewidth=0.9,
        zorder=6,
        label=f"promoted node ({promoted.shape[0]})",
    )

    # The two saddles sit a few millimetres apart, which is under one panel
    # pixel on the whole-machine view, so the shift this node reports is only
    # readable on a magnified window.  The window is square in the poloidal
    # plane and its scale bar gives the reader the length without an axis.
    span = float(np.linalg.norm(image_saddle - state_saddle))
    centre = 0.5 * (state_saddle + image_saddle)
    half = max(1.5e-3, 3.0 * span)
    zoom.plot(state_polyline[:, 0], state_polyline[:, 1], color=STATE, ls="--", lw=1.8)
    zoom.plot(image_polyline[:, 0], image_polyline[:, 1], color=IMAGED, lw=2.6)
    zoom_axis_style = axis_style.variant(axis_markersize=16.0)
    zoom_state_style = state_style.variant(
        xpoint_markersize=15.0, xpoint_markeredgewidth=2.6
    )
    zoom_image_style = image_style.variant(
        xpoint_markersize=15.0, xpoint_markeredgewidth=2.6
    )
    draw_nulls(zoom, magnetic_axis, style=zoom_axis_style)
    draw_nulls(zoom, x_points=[state_saddle], style=zoom_state_style)
    draw_nulls(zoom, x_points=[image_saddle], style=zoom_image_style)
    zoom.plot(
        [state_saddle[0], image_saddle[0]],
        [state_saddle[1], image_saddle[1]],
        color=POPULATION,
        lw=1.2,
        zorder=4,
    )
    zoom.annotate(
        f"{1.0e3 * span:.1f} mm",
        xy=centre,
        xytext=(0.06, 0.16),
        textcoords="axes fraction",
        fontsize=14,
        color="#374151",
        arrowprops={"arrowstyle": "-", "color": POPULATION, "lw": 1.0},
        zorder=7,
    )
    bar = 0.5e-3
    zoom.plot(
        [centre[0] - 0.40 * half, centre[0] - 0.40 * half + bar],
        [centre[1] - 0.92 * half, centre[1] - 0.92 * half],
        color="#374151",
        lw=2.0,
    )
    zoom.text(
        centre[0] - 0.40 * half + 0.5 * bar,
        centre[1] - 0.86 * half,
        "0.5 mm",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#374151",
    )
    zoom.set_xlim(centre[0] - half, centre[0] + half)
    zoom.set_ylim(centre[1] - half, centre[1] + half)
    zoom.set_aspect("equal")
    zoom.set_anchor("W")

    figure.subplots_adjust(
        left=0.012, right=0.995, bottom=0.012, top=0.905, wspace=0.02
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=SUBJECT,
            marker="D",
            markersize=7,
            markerfacecolor="none",
            lw=0,
            label=f"promoted node ({promoted.shape[0]})",
        ),
        plt.Line2D([], [], color=STATE, lw=1.6, ls="--", label="converged state"),
        plt.Line2D([], [], color=POPULATION, lw=1.2, label="converged state flux"),
    ]
    if image_polyline.shape[0]:
        handles.insert(
            1,
            plt.Line2D([], [], color=IMAGED, lw=2.4, label="field of the currents"),
        )
    handles.extend(
        [
            plt.Line2D(
                [],
                [],
                color=STATE,
                marker=DEFAULT_INK.xpoint_marker,
                markersize=8,
                lw=0,
                label="saddle, converged state",
            ),
            plt.Line2D(
                [],
                [],
                color=IMAGED,
                marker=DEFAULT_INK.xpoint_marker,
                markersize=8,
                lw=0,
                label="saddle, field of the currents",
            ),
        ]
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            color=STATE,
            marker=DEFAULT_INK.axis_marker,
            markersize=8,
            lw=0,
            label="magnetic axis",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.995),
    )
    _assert_text_inside(figure)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg_path, format="svg", metadata={"Date": None}, bbox_inches=None)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, format="png", dpi=100, bbox_inches=None)
    plt.close(figure)
    return {
        "svg": str(svg_path),
        "png": str(png_path),
        "levels": levels.tolist(),
        "promoted_marked": int(promoted.shape[0]),
        "state_polyline_vertices": int(state_polyline.shape[0]),
        "image_polyline_vertices": int(image_polyline.shape[0]),
        "saddle_separation_mm": float(1.0e3 * span),
        "zoom_window_mm": float(2.0e3 * half),
        "magnetic_axis_rz": magnetic_axis.tolist(),
        "state_saddle_rz": state_saddle.tolist(),
        "image_saddle_rz": image_saddle.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render(arguments.fields, arguments.receipt, arguments.svg, arguments.png),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
