"""Draw a linear per-slice wall comparison from the converged device receipt."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[3]
FIGURE = (
    ROOT
    / "docs/figures/sub-millisecond-forward-path/bank-solve-driver/per-slice-wall.svg"
)


def record(log, label):
    for line in Path(log).read_text().splitlines():
        if line.startswith(label + " "):
            return json.loads(line.partition(" ")[2])
    raise ValueError(f"{label} is missing from {log}")


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
                raise ValueError(
                    f"text overlap: {left.get_text()} / {right.get_text()}"
                )
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
    single = record(args.log, "SINGLE")
    prototype = record(args.log, "PROTOTYPE")
    batch = record(args.log, "BATCH")
    assert single["converged"] and prototype["converged"]
    assert all(batch["converged"])
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    names = ["prototype · 1 slice", "staged · 1 slice", "staged · 16 slices"]
    walls = [prototype["wall_ms"], single["wall_ms"], batch["per_slice_ms"]]
    colours = ["#929292", "#286eae", "#286eae"]
    for y, value, colour in zip((2, 1, 0), walls, colours, strict=True):
        ax.hlines(y, 0, value, color=colour, linewidth=3.0, zorder=2)
        ax.plot(value, y, "o", color=colour, markersize=10, zorder=3)
    ax.set_yticks((2, 1, 0), names)
    ax.set_xlim(0, max(walls) * 1.12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Converged solve wall per slice [ms]")
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.18, top=0.95)
    save_checked(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
