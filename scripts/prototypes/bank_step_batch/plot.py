"""Draw per-slice wall against slices per batch, one series per step stage."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import numpy as np

from scripts.prototypes.bank_step_booking.plot import check_text

STAGES = (
    ("residual_and_booking", "residual and booking", "#4c78a8", "-"),
    ("local_booking_tangent", "booking tangent", "#e45756", "-"),
    ("geometry_tangent", "geometry tangent", "#54a24b", "-"),
    ("assembly", "assembly", "#b279a2", "-"),
    ("lu_factor", "LU factor", "#f58518", "-"),
    ("whole_step", "whole step", "#1b1b1b", "-"),
)


def stage_walls(path):
    walls = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item["kind"] == "WALL":
            walls[(item["name"], item["slices"])] = item["per_slice_ms"]
        elif item["kind"] == "STAGE":
            walls[(item["name"], item["slices"])] = item["per_slice_ms"]
    return walls


def main(stage_log, svg, png):
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    walls = stage_walls(Path(stage_log))
    batches = sorted({slices for _name, slices in walls})
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for name, label, colour, style in STAGES:
        values = [walls.get((name, count), np.nan) for count in batches]
        ax.plot(
            batches, values, color=colour, linewidth=3.0, linestyle=style,
            marker="o", markersize=7, label=label, zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(count) for count in batches])
    ax.set_xlabel("slices per batch")
    ax.set_ylabel("per-slice wall [ms]")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=14, loc="upper right", ncols=3)
    ax.set_xlim(batches[0] * 0.85, batches[-1] * 3.2)
    finite = [value for value in walls.values() if value]
    ax.set_ylim(min(finite) * 0.6, max(finite) * 1.8)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        (item, item.get_window_extent(renderer))
        for item in fig.findobj(match=plt.Text)
        if item.get_visible() and item.get_text()
    ]
    for index, (first, a) in enumerate(boxes):
        for second, b in boxes[index + 1 :]:
            if a.overlaps(b):
                raise ValueError(
                    f"overlapping figure text: {first.get_text()} / {second.get_text()}"
                )
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.stages, arguments.svg, arguments.png)