"""Draw the promoted nodes and the near-X-point directions off one receipt.

The left panel is the whole machine: unfilled line contours of the converged
state and of the field of the currents on one shared level array, the wall, the
promoted nodes, both null sets, and a locator for the magnification.  The right
panel is that magnification, carrying all four branches leaving the null in
each field beside the separatrix asymptotes of the state's polished saddle,
the directions d with d'Hd = 0.
Every number the figure writes into a figure is read from the receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from nova.media.poloidal import (  # noqa: E402
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    sample_cubic_controls,
)

WIDTH_IN, HEIGHT_IN = 14.0, 7.2
STATE_COLOR = "#222222"
IMAGE_COLOR = "#1f5fa8"
POPULATION = "#929292"
HIGHLIGHT = "#b3261e"
LEVELS = 21
HOT_WB = 1.0e-3
ASYMPTOTE = "#6b6b6b"  # the four X-arm asymptotes at the null
ZOOM_M = 6.0e-3
DEG = "\N{DEGREE SIGN}"


def levels_for(state_values, image_values):
    pooled = np.concatenate(
        [
            np.asarray(state_values, float).ravel(),
            np.asarray(image_values, float).ravel(),
        ]
    )
    pooled = pooled[np.isfinite(pooled)]
    return np.linspace(float(pooled.min()), float(pooled.max()), LEVELS)


def assert_inside(figure, artists, name):
    """Refuse a figure whose labels leave the canvas or overlap each other."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    boxes = [
        (artist.get_text(), artist.get_window_extent(renderer)) for artist in artists
    ]
    problems = []
    for text, box in boxes:
        if (
            box.x0 < canvas.x0 - 1.0
            or box.x1 > canvas.x1 + 1.0
            or box.y0 < canvas.y0 - 1.0
            or box.y1 > canvas.y1 + 1.0
        ):
            problems.append("label %r leaves the canvas" % text)
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if boxes[first][1].overlaps(boxes[second][1]):
                problems.append(
                    "labels %r and %r overlap" % (boxes[first][0], boxes[second][0])
                )
    if problems:
        raise RuntimeError("%s: %s" % (name, "; ".join(sorted(set(problems)))))
    return len(boxes)


def _ray(axes, origin, bearing, length, colour, text, dy, width=1.4, head=True):
    """Draw one direction ray and its direct label; return both artists."""
    direction = np.array(
        [math.cos(math.radians(bearing)), math.sin(math.radians(bearing))]
    )
    end = np.asarray(origin, float) + length * direction
    axes.annotate(
        "",
        xy=end,
        xytext=np.asarray(origin, float),
        arrowprops={"arrowstyle": "-|>" if head else "-", "color": colour, "lw": width},
        zorder=12,
    )
    return axes.text(
        end[0],
        end[1] + dy,
        text,
        color=colour,
        fontsize=11,
        ha="center",
        va="bottom" if dy > 0 else "top",
        zorder=14,
    )


def open_branches(fields, name):
    """Return every open branch of one field, from either archive layout.

    A shared display-grid assembly keeps the resolved points directly, while
    the solve's own archive keeps spline controls beside a validity mask.
    """
    prefix = "%s_open_" % name
    direct = sorted(key for key in fields if key[len(prefix) :].isdigit())
    if direct:
        return [np.asarray(fields[key], float) for key in direct]
    controls = np.asarray(fields["%s_open_controls" % name], float)
    valid = np.asarray(fields["%s_open_valid" % name], bool)
    return [
        np.asarray(sample_cubic_controls(controls[slot], valid[slot]), float)
        for slot in range(controls.shape[0])
    ]


def render(receipt, fields, destination, angles=None):
    """Draw the two-panel figure and return the written path."""
    radius = np.asarray(fields["display_radius"], float)
    height = np.asarray(fields["display_height"], float)
    state_flux = np.asarray(fields["display_state_flux"], float).T
    image_flux = np.asarray(fields["display_flux"], float).T
    levels = levels_for(state_flux, image_flux)
    promoted = np.asarray(fields["promoted_coordinate"], float)
    difference = np.asarray(
        [row["image_minus_state_wb"] for row in receipt["promoted"]], float
    )
    hot = np.abs(difference) > HOT_WB
    saddle = np.asarray(receipt["x_point"]["state_rz"], float)
    image_saddle = np.asarray(receipt["x_point"]["image_rz"], float)
    axis_rz = np.asarray(receipt["nulls"]["axis_rz"], float)

    figure = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.62, 1.0),
        wspace=0.03,
        left=0.015,
        right=0.99,
        bottom=0.09,
        top=0.85,
    )
    main = figure.add_subplot(grid[0, 0])
    zoom = figure.add_subplot(grid[0, 1])
    labelled = []

    wall_units = receipt["wall_units"]
    for axes in (main, zoom):
        for unit in wall_units:
            draw_wall(
                axes,
                np.asarray(unit["radius"], dtype=float),
                np.asarray(unit["height"], dtype=float),
                close=bool(unit["closed"]),
                linestyle=None if unit["closed"] else "--",
            )
        draw_flux_contours(
            axes, radius, height, image_flux, levels, linewidth=0.5, color="#cccccc"
        )
        draw_flux_contours(
            axes, radius, height, state_flux, levels, linewidth=0.6, color="#4a4a4a"
        )
        drawn = [
            (fields["state_closed_polyline"], STATE_COLOR, (0, (5, 2))),
            (fields["image_closed_polyline"], IMAGE_COLOR, (0, (2, 2))),
        ]
        for name, colour, dash in (
            ("state", STATE_COLOR, (0, (5, 2))),
            ("image", IMAGE_COLOR, (0, (2, 2))),
        ):
            for points in open_branches(fields, name):
                if points.shape[0] > 2:
                    drawn.append((points, colour, dash))
        for polyline, colour, dash in drawn:
            polyline = np.asarray(polyline, float)
            if polyline.ndim == 2 and polyline.shape[0] > 2:
                axes.plot(
                    polyline[:, 0],
                    polyline[:, 1],
                    color=colour,
                    linewidth=1.6,
                    linestyle=dash,
                    zorder=8,
                )
        axes.scatter(
            promoted[~hot, 0],
            promoted[~hot, 1],
            s=16,
            color=POPULATION,
            zorder=9,
            linewidths=0,
        )
        axes.scatter(
            promoted[hot, 0],
            promoted[hot, 1],
            s=26,
            color=HIGHLIGHT,
            marker="D",
            zorder=10,
            linewidths=0,
        )
        draw_nulls(
            axes,
            magnetic_axis=axis_rz,
            x_points=saddle[None, :],
            other_x_points=image_saddle[None, :],
        )
        axes.set_aspect("equal")
        axes.set_axis_off()

    main.add_patch(
        Rectangle(
            saddle - 0.5 * ZOOM_M,
            ZOOM_M,
            ZOOM_M,
            fill=False,
            edgecolor=IMAGE_COLOR,
            linewidth=1.0,
            zorder=11,
        )
    )
    half = 0.5 * ZOOM_M
    zoom.set_xlim(saddle[0] - half, saddle[0] + half)
    zoom.set_ylim(saddle[1] - half, saddle[1] + half)

    if angles is not None:
        for entry in angles["state"]["asymptotic"]["directions"]:
            direction = np.asarray(entry["unit"], float)
            both = np.concatenate(
                [saddle[None, :] - 1.6 * half * direction[None, :],
                 saddle[None, :] + 1.6 * half * direction[None, :]]
            )
            zoom.plot(
                both[:, 0],
                both[:, 1],
                color=ASYMPTOTE,
                linewidth=0.9,
                linestyle=(0, (1, 2)),
                zorder=13,
            )

    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=STATE_COLOR,
                linestyle=(0, (5, 2)),
                linewidth=1.6,
                label="separatrix, converged state",
            ),
            Line2D(
                [],
                [],
                color=IMAGE_COLOR,
                linestyle=(0, (2, 2)),
                linewidth=1.6,
                label="separatrix, field of the currents",
            ),
            Line2D(
                [],
                [],
                color=POPULATION,
                marker="o",
                linestyle="none",
                markersize=5,
                label="promoted node",
            ),
            Line2D(
                [],
                [],
                color=HIGHLIGHT,
                marker="D",
                linestyle="none",
                markersize=5,
                label="promoted, |image - state| above 1 mWb",
            ),
            Line2D(
                [],
                [],
                color="#4a4a4a",
                linewidth=1.4,
                label="poloidal flux, converged state",
            ),
            Line2D(
                [],
                [],
                color="#cccccc",
                linewidth=1.4,
                label="poloidal flux, field of the currents",
            ),
            Line2D(
                [],
                [],
                color=ASYMPTOTE,
                linewidth=1.2,
                linestyle=(0, (1, 2)),
                label="X-arm asymptotes at the null",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.995),
    )

    assert_inside(figure, labelled, "promoted-nodes-panel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return {
        "figure": str(destination),
        "labels": len(labelled),
        "hot_nodes": int(hot.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=Path, default=None)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    fields = dict(np.load(arguments.fields))
    angles = (
        json.loads(arguments.angles.read_text(encoding="utf-8"))
        if arguments.angles is not None
        else None
    )
    print(json.dumps(render(receipt, fields, arguments.output, angles)))


if __name__ == "__main__":
    main()
