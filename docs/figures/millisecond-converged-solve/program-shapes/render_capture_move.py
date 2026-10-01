"""Compare the bucket figure's drawn ratios between two renderings of it.

Each input is a bucket figure SVG written by render_buckets.py, so the ratios
compared here are the ones the figure actually draws rather than ones read back
from a receipt. Usage:

    render_capture_move.py --retired <old.svg> --current <new.svg> --output <path>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = re.compile(r'<path d="([^"]*)"[^>]*style="([^"]*)"')
RECT_COLOUR = {
    "#1f77b4": "support_capacity",
    "#ff7f0e": "mesh_cell_capacity",
    "#2ca02c": "mesh_axis_nodes",
    "#d62728": "sample_node_capacity",
    "#9467bd": "state_size",
    "#8c564b": "wall_node_capacity",
    "#e377c2": "arc_capacity",
}


def _numbers(path_data: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+\.?\d*", path_data)]


def drawn_ratios(svg_text: str) -> dict[str, list[float]]:
    """The ratio each drawn bar encodes, grouped by the axis it belongs to."""
    rectangles = []
    verticals = []
    dashed = []
    for data, style in PATH.findall(svg_text):
        values = _numbers(data)
        fill = re.search(r"fill:\s*(#[0-9a-fA-F]{6})", style)
        if fill is not None and fill.group(1) in RECT_COLOUR and len(values) == 8:
            xs, ys = values[0::2], values[1::2]
            if len(set(xs)) == 2 and len(set(ys)) == 2:
                rectangles.append(
                    (min(xs), max(xs), max(ys), RECT_COLOUR[fill.group(1)])
                )
        elif len(values) == 4 and values[0] == values[2]:
            (dashed if "stroke-dasharray" in style else verticals).append(values[0])

    spine_xs = sorted(set(verticals))
    origin = min(
        x for x in spine_xs if any(abs(box[0] - x) < 1e-6 for box in rectangles)
    )
    edge = min(x for x in spine_xs if x > origin)
    unit = [x for x in dashed if origin < x < edge]
    if len(unit) != 1:
        raise ValueError(f"expected one ratio-one line, saw {unit}")
    scale = unit[0] - origin

    ratios: dict[str, list[float]] = {}
    for left, right, _, axis in rectangles:
        if abs(left - origin) < 1e-6:
            ratios.setdefault(axis, []).append((right - origin) / scale)
    for values in ratios.values():
        values.sort()
    return ratios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retired", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    retired = drawn_ratios(args.retired.read_text(encoding="utf-8"))
    current = drawn_ratios(args.current.read_text(encoding="utf-8"))
    # Any change to how many capacities an axis observes moves this figure's
    # grouping, so a mismatch is refused rather than drawn mislabelled.
    counts = sorted(len(values) for values in current.values())
    if counts != [2, 2, 2, 3, 4, 4, 5]:
        raise ValueError(f"unexpected bars per capacity axis: {counts}")
    axes = sorted(current, key=lambda name: -max(current[name]))
    moved_axes = {
        name: max(abs(a - b) for a, b in zip(retired[name], current[name])) > 1e-3
        for name in axes
    }

    figure, panel = plt.subplots(figsize=(7.0, 4.2))
    for index, name in enumerate(axes):
        before, after = retired[name], current[name]
        moved = moved_axes[name]
        panel.plot([min(before), max(after)], [index, index], color="#cccccc", zorder=1)
        panel.plot(
            before,
            [index] * len(before),
            "o",
            markerfacecolor="none",
            markeredgecolor="#7f7f7f",
            markersize=6,
            label="retired capture" if index == 0 else None,
            zorder=2,
        )
        panel.plot(
            after,
            [index] * len(after),
            "o",
            color="#3b6ea5" if moved > 1e-3 else "#7f7f7f",
            markersize=6,
            label="regenerated capture" if index == 0 else None,
            zorder=3,
        )
    panel.axvline(1.0, color="black", linewidth=0.8, linestyle="dashed")
    panel.set_yticks(range(len(axes)))
    panel.set_yticklabels(
        [f"{name} (moved)" if moved_axes[name] else name for name in axes], fontsize=8
    )
    panel.set_xlabel(
        "padding ratio drawn: bucketed floor over realised capacity", fontsize=8
    )
    panel.set_title(
        "bucket figure bars, retired against regenerated capture",
        fontsize=9,
        loc="left",
    )
    panel.tick_params(labelsize=8)
    panel.legend(fontsize=7, loc="lower right")
    panel.grid(axis="x", color="#eeeeee", linewidth=0.5)
    panel.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    figure.savefig(args.output.with_suffix(".svg"))
    for name in axes:
        print(
            f"{name}: retired {sorted(retired[name])} current {sorted(current[name])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
