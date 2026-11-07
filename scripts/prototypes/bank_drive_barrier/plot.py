"""Draw the fixed-state fresh residual against drive perturbation."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def validate_text(fig, ax):
    """Refuse clipped or colliding text before writing the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.texts,
    ]
    boxes = [
        (artist.get_text(), artist.get_window_extent(renderer))
        for artist in artists
        if artist.get_visible() and artist.get_text()
    ]
    for name, box in boxes:
        if not fig.bbox.contains(*box.get_points()[0]) or not fig.bbox.contains(
            *box.get_points()[1]
        ):
            raise ValueError(f"clipped figure text: {name}")
    for index, (name, box) in enumerate(boxes):
        for other_name, other_box in boxes[index + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"overlapping figure text: {name} / {other_name}")
    for label in ax.texts:
        box = label.get_window_extent(renderer)
        for line in ax.lines:
            path = line.get_path().transformed(line.get_transform())
            if path.intersects_bbox(box, filled=False):
                raise ValueError(f"figure label covers data: {label.get_text()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for line in args.log.read_text().splitlines():
        if line.startswith("PROBE "):
            rows.append(json.loads(line[6:]))
    if len(rows) != 6:
        raise ValueError("expected six fresh residual probes")
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2))
    x = np.array([row["epsilon"] or 1e-13 for row in rows])
    y = np.array([row["residual_max"] for row in rows])
    ax.plot(x, y, color="#2e6fa8", linewidth=3, marker="o", markersize=8, zorder=3)
    for row, position, residual in zip(rows, x, y, strict=True):
        flipped = [
            name
            for name, key in (
                ("membership", "membership_flip"),
                ("cut set", "cut_flip"),
                ("normal form", "normal_flip"),
                ("fragments", "fragment_flip"),
                ("topology class", "class_flip"),
                ("qualification", "qualified_flip"),
                ("X-point", "xvalid_flip"),
            )
            if row[key]
        ]
        if flipped:
            ax.annotate(
                ", ".join(flipped),
                (position, residual),
                xytext=(0, 24),
                textcoords="offset points",
                ha="left",
                va="bottom",
                color="#2e6fa8",
                fontsize=14,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-14, 3e-4)
    ax.set_ylim(min(y) / 3, max(y) * 30)
    ax.set_xticks(x, ["0", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴"])
    ax.set_xlabel("drive perturbation")
    ax.set_ylabel("fresh residual norm")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.17, top=0.92)
    validate_text(fig, ax)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg")
    image = args.output.read_text()
    image = re.sub(
        r'width="[^"]+" height="[^"]+"',
        'width="31.1cm" height="16cm"',
        image,
        count=1,
    )
    args.output.write_text(image)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
