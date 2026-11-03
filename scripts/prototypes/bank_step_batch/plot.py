"""Draw per-slice wall against slices per batch, one series per step stage."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import numpy as np


# The booking tangent and its assembly land within about a pixel of each other,
# so neither may be drawn opaquely over the other. The assembly is drawn as a
# wide hollow ring and the tangent as a small filled square on top of it, kept
# apart also by dash: the ring shows around every mark the tangent places.
STAGES = (
    ("read", "read", "#6f4fa8", "-", "D", 5, 7, False),
    ("residual_and_booking", "residual and booking", "#4c78a8", "-", "o", 3, 7, False),
    ("local_booking_tangent", "booking tangent", "#2f6b5e", "--", "s", 6, 6, False),
    ("geometry_tangent", "geometry tangent", "#54a24b", "-", "o", 3, 7, False),
    ("assembly", "assembly", "#b279a2", "-", "o", 4, 11, True),
    ("lu_factor", "LU factor", "#b8560f", "-", "o", 3, 7, False),
    ("whole_step", "whole step", "#1b1b1b", "-", "o", 7, 7, False),
)


def stage_walls(path):
    walls = {}
    for line in path.read_text().splitlines():
        kind, _, payload = line.partition(" ")
        if kind not in {"WALL", "STAGE"}:
            continue
        item = json.loads(payload)
        walls[(item["name"], item["slices"])] = item["per_slice_ms"]
    return walls


def main(stage_log, svg, png):
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    walls = stage_walls(Path(stage_log))
    batches = sorted({slices for _name, slices in walls})
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for name, label, colour, style, marker, order, size, hollow in STAGES:
        values = [walls.get((name, count), np.nan) for count in batches]
        ax.plot(
            batches,
            values,
            color=colour,
            linewidth=3.0,
            linestyle=style,
            marker=marker,
            markersize=size,
            markerfacecolor="none" if hollow else colour,
            markeredgewidth=2.0 if hollow else 1.0,
            label=label,
            zorder=order,
        )
        # A series that stops early is a truncated measurement, not a value
        # that happened to be missing: mark where it ends.
        if len(values) > 1 and np.isnan(values[-1]) and np.isfinite(values[-2]):
            last = next(
                count
                for count, value in zip(reversed(batches), reversed(values))
                if np.isfinite(value)
            )
            height = walls[(name, last)]
            ax.plot(
                [last],
                [height],
                marker="x",
                markersize=11,
                markeredgewidth=2.6,
                color=colour,
                linestyle="none",
                zorder=order + 1,
            )
            ax.annotate(
                f"{label} truncated at {last}",
                xy=(last, height),
                xytext=(4, -16),
                textcoords="offset points",
                fontsize=14,
                color=colour,
                ha="left",
                va="top",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(count) for count in batches])
    ax.set_xlabel("slices per batch")
    ax.set_ylabel("per-slice wall [ms]")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=14, loc="upper right", ncols=4)
    ax.set_xlim(batches[0] * 0.85, batches[-1] * 1.15)
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
    fig.savefig(svg)
    fig.savefig(png, dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.stages, arguments.svg, arguments.png)