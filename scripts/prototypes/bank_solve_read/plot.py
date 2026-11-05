"""Draw the per-slice converged wall of the two arms from their receipt."""

import argparse
import json
from math import ceil
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[3]
FIGURE = (
    ROOT
    / "docs/figures/sub-millisecond-forward-path/bank-solve-read/per-slice-wall.svg"
)


def records(log):
    """Every receipt line, tagged with the arm its header announced."""
    arm = None
    found = {}
    for line in Path(log).read_text().splitlines():
        if line.startswith("HEADER "):
            arm = json.loads(line.partition(" ")[2])["arm"]
        for label in ("SINGLE", "BATCH"):
            if line.startswith(label + " "):
                found[(arm, label)] = json.loads(line.partition(" ")[2])
    return found


def save_checked(fig, output):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [
        item for item in fig.findobj(Text) if item.get_visible() and item.get_text()
    ]
    boxes = [(item, item.get_window_extent(renderer)) for item in texts]
    for item, box in boxes:
        if not all(canvas.contains(*corner) for corner in box.get_points()):
            raise ValueError(f"text outside canvas: {item.get_text()}")
    for index, (left, left_box) in enumerate(boxes):
        for right, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                raise ValueError(f"text overlap: {left.get_text()} / {right.get_text()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    root = ET.parse(output)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(output, encoding="unicode", xml_declaration=True)
    print(f"FIGURE={output} TEXT_CHECK=passed CANVAS=1400x720", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args()
    found = records(args.log)
    rows = [
        ("base · 1 slice", found[("base", "SINGLE")]["wall_ms"], "#929292"),
        ("head · 1 slice", found[("head", "SINGLE")]["wall_ms"], "#286eae"),
        (
            "base · 16 slices, amortised per slice",
            found[("base", "BATCH")]["per_slice_ms"],
            "#929292",
        ),
        (
            "head · 16 slices, amortised per slice",
            found[("head", "BATCH")]["per_slice_ms"],
            "#286eae",
        ),
    ]
    walls = [row[1] for row in rows]
    scale, unit = (1000, "s") if max(walls) >= 1000 else (1, "ms")
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    positions = [3, 2, 1, 0]
    for y, (name, value, colour) in zip(positions, rows, strict=True):
        scaled = value / scale
        ax.hlines(y, 0, scaled, color=colour, linewidth=3.0, zorder=2)
        ax.plot(scaled, y, "o", color=colour, markersize=10, zorder=3)
        ax.annotate(
            f"{scaled:.2f} {unit}",
            (scaled, y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=14,
            color=colour,
            zorder=4,
        )
    ax.set_yticks(positions, [row[0] for row in rows])
    values = [value / scale for value in walls]
    tick = 0.5 if max(values) <= 2 else float(ceil(max(values) / 4))
    upper = tick * ceil(max(values) / tick) * 1.35
    ax.set_xlim(0, upper)
    ax.set_xticks([tick * index for index in range(ceil(upper / tick) + 1)])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel(f"Converged solve wall per slice [{unit}]")
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.16, top=0.96)
    save_checked(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()