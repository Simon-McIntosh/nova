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
from nova.media.poloidal import draw_nulls, draw_scattered_contours, draw_wall  # noqa: E402

SUBJECT = "#d97706"
IMAGED = "#2563eb"
STATE = "#1f2937"
POPULATION = "#929292"


def _cubic_points(controls, valid, samples=8):
    """Sample one branch's valid cubic segments into an ordered polyline."""
    points = []
    index_array = np.nonzero(np.asarray(valid, dtype=bool).reshape(-1))[0]
    for index in index_array:
        control = np.asarray(controls[index], dtype=np.float64).reshape(4, 2)
        parameter = np.linspace(0.0, 1.0, samples, endpoint=index == index_array[0])
        one_minus = 1.0 - parameter
        weight = np.stack(
            (
                one_minus**3,
                3 * one_minus**2 * parameter,
                3 * one_minus * parameter**2,
                parameter**3,
            ),
            axis=1,
        )
        points.append(weight @ control)
    return np.concatenate(points) if points else np.empty((0, 2))


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

    figure, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    poloidal_axes(axis)
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

    image_polyline = np.asarray(fields["image_closed_polyline"], dtype=np.float64)
    state_polyline = np.asarray(fields["state_closed_polyline"], dtype=np.float64)
    axis.plot(
        state_polyline[:, 0],
        state_polyline[:, 1],
        color=STATE,
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label="separatrix assembled on the converged state",
    )
    for controls_key, valid_key in (
        ("state_open_controls", "state_open_valid"),

    ):
        controls = np.asarray(fields[controls_key], dtype=np.float64)
        valid = np.asarray(fields[valid_key], dtype=bool)
        colour = STATE
        style = "-"
        width = 2.4
        for slot in range(valid.shape[0]):
            leg = _cubic_points(controls[slot], valid[slot])
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

    state_style = DEFAULT_INK.variant(axis_color=STATE, xpoint_color=STATE)
    image_style = DEFAULT_INK.variant(
        axis_color=IMAGED, xpoint_color=IMAGED, axis_markersize=9, xpoint_markersize=10
    )
    draw_nulls(
        axis,
        np.asarray(fields["state_saddle"], dtype=np.float64),
        None,
        style=state_style,
        contain=wall_polygon,
    )
    draw_nulls(
        axis,
        np.asarray(fields["image_saddle"], dtype=np.float64),
        None,
        style=image_style,
        contain=wall_polygon,
    )
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
    axis.set_aspect("equal")
    axis.autoscale_view()
    figure.legend(
        handles=[
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
            plt.Line2D(
                [],
                [],
                color=STATE,
                lw=1.6,
                ls="--",
                label="separatrix, converged state",
            ),
            plt.Line2D([], [], color=POPULATION, lw=1.2, label="converged state flux"),
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.02),
    )
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
