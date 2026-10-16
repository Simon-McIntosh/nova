"""Render measured Taylor slopes, solve histories, and terminal flux contours."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_scattered_contours, draw_wall

assert jax.config.jax_enable_x64 is True


BLUE = "#326c88"
RUST = "#9a5f47"
GREY = "#666666"
BLOCKS = ("current", "radial", "vertical")


def _style():
    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "fine-grid-convergence"
    plt.rcParams["axes.grid"] = False


def _save(figure, axis, output):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.canvas.draw()
    canvas = figure.bbox
    artists = [
        text
        for text in figure.findobj(plt.Text)
        if text.get_visible() and text.get_text()
    ]
    boxes = [(text.get_text(), text.get_window_extent()) for text in artists]
    for label, box in boxes:
        if not canvas.contains(box.x0, box.y0) or not canvas.contains(box.x1, box.y1):
            raise ValueError(f"text outside canvas: {label}")
    for first, (left_name, left) in enumerate(boxes):
        for right_name, right in boxes[first + 1 :]:
            overlap = left.intersection(left, right)
            if overlap is not None and overlap.width > 1 and overlap.height > 1:
                raise ValueError(f"text overlap: {left_name} / {right_name}")
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def taylor(receipts, output):
    _style()
    figure, axis = plt.subplots(figsize=(14, 8), dpi=100)
    for receipt, color in zip(receipts, (BLUE, RUST), strict=True):
        rows = receipt["directions"]["newton"]
        steps = np.asarray([row["step"] for row in rows])
        remainder = np.asarray([row["blocks"] for row in rows])
        slopes = np.log(remainder[:-1] / remainder[1:]) / np.log(
            steps[:-1, None] / steps[1:, None]
        )
        locations = np.sqrt(steps[:-1] * steps[1:])
        for column, (name, dash) in enumerate(
            zip(BLOCKS, ("-", "--", ":"), strict=True)
        ):
            axis.semilogx(
                locations,
                slopes[:, column],
                color=color,
                linestyle=dash,
                linewidth=2.6,
                label=f"{receipt['cells']:,} {name}",
            )
    axis.axhline(2, color=GREY, linestyle=":", linewidth=1.2)
    axis.set_xlim(2e-8, 5e-3)
    axis.set_xticks(np.logspace(-7, -3, 5))
    axis.set_xlabel("moment step")
    axis.set_ylabel("Taylor remainder slope")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    _save(figure, axis, output)


def residual_histories(receipts, output):
    _style()
    figure, axis = plt.subplots(figsize=(14, 8), dpi=100)
    all_values = []
    for receipt, color in zip(receipts, (GREY, BLUE, RUST), strict=True):
        rows = receipt["history"]
        values = [receipt["centre_residual"]] + [
            row["residual"] for row in rows if row["accepted"]
        ]
        all_values.extend(values)
        axis.semilogy(
            np.arange(len(values)),
            values,
            color=color,
            linewidth=3,
            marker="o",
            markersize=5,
            label=f"{receipt['cells']:,} cells",
        )
    largest = max(
        sum(row["accepted"] for row in receipt["history"]) for receipt in receipts
    )
    axis.set_xlim(-0.1, largest + 0.1)
    axis.set_xticks(np.arange(largest + 1))
    axis.axhline(1e-12, color=GREY, linestyle=":", linewidth=1.2)
    lower = min(min(all_values), 1e-12) / 2
    upper = max(all_values) * 2
    axis.set_ylim(lower, upper)
    powers = np.arange(
        int(np.ceil(np.log10(lower))), int(np.floor(np.log10(upper))) + 1, 2
    )
    axis.set_yticks(10.0**powers)
    axis.set_xlabel("accepted correction")
    axis.set_ylabel("scaled residual")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    _save(figure, axis, output)


def terminal_panel(data, output):
    """Replace fast_read.render_contours.panel without figure-local prose."""
    points = np.asarray(data["coordinates"])
    wall = np.asarray(data["wall"])
    solved = np.asarray(data["solved_flux"])
    analytic = np.asarray(data["analytic_flux"])
    lower = max(np.min(solved), np.min(analytic))
    upper = min(np.max(solved), np.max(analytic))
    levels = np.linspace(lower, upper, 9)[1:-1]
    figure, axis = plt.subplots(figsize=(14, 8), dpi=100)
    poloidal_axes(axis)
    draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        solved,
        levels,
        wall,
        color=DEFAULT_INK.flux_color,
        linewidth=3.0,
    )
    reference_lines = draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        analytic,
        levels,
        wall,
        color="#444444",
        linewidth=2.6,
    )
    reference_lines.set_linestyles("dashed")
    draw_wall(axis, wall[:, 0], wall[:, 1])
    solved_roots = np.asarray(data["solved_roots"])
    analytic_roots = np.asarray(data["analytic_roots"])
    reference_style = DEFAULT_INK.variant(
        axis_color="#444444",
        xpoint_color="#444444",
        axis_markersize=10,
        xpoint_markersize=11,
    )
    solved_style = DEFAULT_INK.variant(
        axis_color=DEFAULT_INK.flux_color,
        xpoint_color=DEFAULT_INK.flux_color,
    )
    draw_nulls(
        axis,
        analytic_roots[0],
        analytic_roots[1:],
        style=reference_style,
        contain=wall,
    )
    draw_nulls(
        axis,
        solved_roots[0],
        solved_roots[1:],
        style=solved_style,
        contain=wall,
    )
    axis.autoscale_view()
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, action="append", default=[])
    parser.add_argument("--solve", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        if len(args.probe) != 2:
            parser.error("two fine-mesh derivative receipts are required")
        taylor(
            [json.loads(path.read_text()) for path in args.probe],
            args.output / "taylor-slopes.svg",
        )
    if args.solve:
        if len(args.solve) != 3:
            parser.error("three solve receipts are required")
        receipts = [json.loads(path.read_text()) for path in args.solve]
        residual_histories(receipts, args.output / "residual-histories.svg")
        for receipt in receipts:
            terminal_panel(
                receipt["panel"],
                args.output / f"terminal-{receipt['cells']}.svg",
            )


if __name__ == "__main__":
    main()
