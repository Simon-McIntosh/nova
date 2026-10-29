"""Draw a converged bank flux field against its analytic oracle."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
)


def draw_outline(axes, point, marker, size, color, zorder):
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


def check_text(figure):
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


def render(state_path, output_path, preview_path):
    with np.load(state_path, allow_pickle=False) as state:
        data = {name: np.asarray(state[name]) for name in state.files}
    if not bool(data["converged"]):
        raise ValueError("only a converged state can be published as the oracle panel")
    coordinates = data["coordinates"]
    wall = data["wall"]
    analytic = data["analytic_flux"]
    solved = data["solved_flux"]
    oracle_roots = data["analytic_roots"]
    solved_roots = data["solved_roots"]
    span = float(np.ptp(analytic))
    levels = np.linspace(
        float(np.min(analytic)) + 0.1 * span, float(np.max(analytic)) - 0.1 * span, 7
    )
    feature = DEFAULT_INK.flux_color
    neutral = "#676767"
    figure, axes = plt.subplots(figsize=(14, 7.2), dpi=100)
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
        zorder=3,
    )
    reference = draw_scattered_contours(
        axes,
        coordinates[:, 0],
        coordinates[:, 1],
        analytic,
        levels,
        wall,
        color=neutral,
        linewidth=2.6,
        zorder=2,
    )
    reference.set_linestyle("dashed")
    draw_wall(axes, wall[:, 0], wall[:, 1])
    oracle_style = DEFAULT_INK.variant(
        axis_color=neutral,
        xpoint_color=neutral,
        axis_marker="^",
        axis_markersize=14,
        xpoint_marker="D",
        xpoint_markersize=14,
    )
    draw_nulls(
        axes,
        oracle_roots[0],
        oracle_roots[1:],
        style=oracle_style,
        contain=wall,
    )
    draw_outline(axes, oracle_roots[0], "^", 14, neutral, 8)
    draw_outline(axes, oracle_roots[1], "D", 14, neutral, 8)
    solved_style = DEFAULT_INK.variant(
        axis_color=feature,
        xpoint_color=feature,
        axis_markersize=7,
        xpoint_markersize=8,
        zorder_markers=9,
    )
    draw_nulls(
        axes,
        solved_roots[0],
        solved_roots[1:],
        style=solved_style,
        contain=wall,
    )
    axes.set_aspect("equal")
    handles = [
        Line2D([], [], color=feature, lw=3, label="bank flux"),
        Line2D([], [], color=neutral, lw=2.6, ls="--", label="oracle flux"),
        Line2D([], [], color=neutral, lw=2.4, label="wall"),
        Line2D(
            [],
            [],
            color=feature,
            marker="^",
            markersize=7,
            linestyle="none",
            label="bank axis",
        ),
        Line2D(
            [],
            [],
            color=neutral,
            marker="^",
            markersize=12,
            markerfacecolor="white",
            linestyle="none",
            label="oracle axis",
        ),
        Line2D(
            [],
            [],
            color=feature,
            marker="X",
            markersize=8,
            linestyle="none",
            label="bank saddle",
        ),
        Line2D(
            [],
            [],
            color=neutral,
            marker="D",
            markersize=12,
            markerfacecolor="white",
            linestyle="none",
            label="oracle saddle",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.03)
    check_text(figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview_path, format="png", dpi=200)
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    image = output_path.read_text()
    image, count = re.subn(
        r'width="[^"]+" height="[^"]+" viewBox=',
        'width="31.1cm" height="16cm" viewBox=',
        image,
        count=1,
    )
    assert count == 1
    output_path.write_text(
        "\n".join(line.rstrip() for line in image.splitlines()) + "\n"
    )
    print(
        f"PANEL={output_path} PREVIEW={preview_path} "
        f"LEVELS={levels.tolist()} RESIDUAL={float(data['residual_sup'])} "
        f"AXIS_RATIO={float(data['axis_ratio'])} "
        f"BOUNDARY_RATIO={float(data['boundary_ratio'])}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()
    render(args.state, args.output, args.preview)
