"""Draw one fast-route certificate row against its analytic oracle."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
)

NEUTRAL = "#676767"
CANVAS_INCHES = (14.0, 7.2)
DPI = 200
CM_PER_INCH = 2.54


def check_text(figure):
    """Refuse a figure whose text leaves the canvas or overlaps."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    artists = [
        artist
        for artist in figure.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for artist, box in zip(artists, boxes, strict=True):
        if (
            box.x0 < 0
            or box.y0 < 0
            or box.x1 > figure.bbox.width
            or box.y1 > figure.bbox.height
        ):
            raise ValueError(f"text leaves canvas: {artist.get_text()}")
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if boxes[first].overlaps(boxes[second]):
                raise ValueError(
                    f"text overlaps: {artists[first].get_text()}, "
                    f"{artists[second].get_text()}"
                )


def outline(axes, point, marker, size, color, zorder):
    axes.plot(
        point[0],
        point[1],
        marker=marker,
        markersize=size,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=2.2,
        linestyle="none",
        zorder=zorder,
    )


def render(state_path, row_path, output_path, preview_path, title, cells=None):
    import jax
    from tests.equilibrium import test_topology_read as fixture

    with np.load(state_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    row = json.loads(Path(row_path).read_text())
    if "rows" in row:
        matches = [
            item for item in row["rows"] if cells is None or item["cells"] == cells
        ]
        if len(matches) != 1:
            raise ValueError(f"no single row for cells={cells}")
        row = matches[0]
    kind = row["kind"]
    _, analytic, wall, _, _ = fixture._analytic_inputs(kind)
    coordinates = data["nodes"]
    analytic_flux = np.asarray(jax.vmap(analytic.value)(coordinates))
    solved = data["solved_flux"]
    solved_roots = data["solved_roots"]
    analytic_roots = data["analytic_roots"]
    span = float(np.ptp(analytic_flux))
    levels = np.linspace(
        float(np.asarray(analytic_flux).min()) + 0.1 * span,
        float(np.asarray(analytic_flux).max()) - 0.1 * span,
        9,
    )
    feature = DEFAULT_INK.flux_color
    figure, axes = plt.subplots(figsize=CANVAS_INCHES, dpi=100)
    poloidal_axes(axes)
    draw_scattered_contours(
        axes,
        coordinates[:, 0],
        coordinates[:, 1],
        solved,
        levels,
        wall,
        color=feature,
        linewidth=3.0,
        zorder=4,
    )
    # The two fields agree to about 1e-5 of the span, so the oracle is drawn
    # above the route on a thinner dashed line rather than beneath it.
    reference = draw_scattered_contours(
        axes,
        coordinates[:, 0],
        coordinates[:, 1],
        analytic_flux,
        levels,
        wall,
        color=NEUTRAL,
        linewidth=1.6,
        zorder=6,
    )
    reference.set_linestyle((0, (4, 2.6)))
    # A white casing keeps the dashes legible where they run along the route.
    reference.set_path_effects(
        [patheffects.withStroke(linewidth=3.4, foreground="white")]
    )
    draw_wall(axes, wall[:, 0], wall[:, 1])
    oracle = row["oracle"]
    saddle = oracle.get("saddle_ratio")
    # A limited row has no X-point; its second stationary point is the limiter
    # contact, which must not be drawn with the saddle marker.
    boundary_marker = "X" if saddle is not None else "s"
    boundary_label = "X-point" if saddle is not None else "boundary"
    oracle_style = DEFAULT_INK.variant(
        axis_color=NEUTRAL,
        xpoint_color=NEUTRAL,
        axis_marker="^",
        axis_markersize=22,
        xpoint_marker=boundary_marker,
        xpoint_markersize=22,
    )
    draw_nulls(
        axes, analytic_roots[0], analytic_roots[1:], style=oracle_style, contain=wall
    )
    outline(axes, analytic_roots[0], "^", 16, NEUTRAL, 8)
    if len(analytic_roots) > 1:
        outline(axes, analytic_roots[1], boundary_marker, 16, NEUTRAL, 8)
    solved_style = DEFAULT_INK.variant(
        axis_color=feature,
        xpoint_color=feature,
        axis_markersize=12,
        xpoint_marker=boundary_marker,
        xpoint_markersize=12,
        zorder_markers=9,
    )
    draw_nulls(
        axes, solved_roots[0], solved_roots[1:], style=solved_style, contain=wall
    )
    axes.set_aspect("equal")
    lower, upper = np.min(wall, axis=0), np.max(wall, axis=0)
    margin = 0.04 * (upper - lower)
    axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    xpoint = "n/a" if saddle is None else f"{saddle:.2e}"
    caption = (
        f"{title}: residual {row['fast']['fresh_residual']:.2e}, converged "
        f"{row['fast']['converged']}; errors axis {oracle['axis_ratio']:.2e}, "
        f"X-point {xpoint}, boundary {oracle['boundary_ratio']:.2e}\n"
        f"budgets 5e-4 for axis and X-point, 1e-4 for boundary"
    )
    handles = [
        Line2D([], [], color=feature, lw=3, label="fast-route flux"),
        Line2D(
            [],
            [],
            color=feature,
            marker="^",
            markersize=10,
            linestyle="none",
            label="fast axis",
        ),
        Line2D(
            [],
            [],
            color=feature,
            marker=boundary_marker,
            markersize=10,
            linestyle="none",
            label=f"fast {boundary_label}",
        ),
        Line2D([], [], color=NEUTRAL, lw=1.6, ls="--", label="oracle flux"),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="^",
            markersize=14,
            markerfacecolor="white",
            linestyle="none",
            label="oracle axis",
        ),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker=boundary_marker,
            markersize=14,
            markerfacecolor="white",
            linestyle="none",
            label=f"oracle {boundary_label}",
        ),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.wall_color,
            lw=DEFAULT_INK.wall_linewidth,
            label="wall",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    figure.text(
        0.5, 0.055, caption, ha="center", va="center", fontsize=14, linespacing=1.35
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.15)
    check_text(figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview_path, format="png", dpi=DPI)
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    print(
        "PANEL "
        + json.dumps(
            {
                "output": str(output_path),
                "preview": str(preview_path),
                "caption": caption,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--cells", type=int)
    args = parser.parse_args()
    render(args.state, args.row, args.output, args.preview, args.title, args.cells)
