"""Render the measured cell-integral and solve comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


GREY = "#929292"
BLUE = "#326c88"
REFERENCE = "#666666"


def _save(figure, output):
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.canvas.draw()
    canvas = figure.bbox
    artists = [
        artist
        for artist in figure.findobj(plt.Text)
        if artist.get_visible() and artist.get_text()
    ]
    bounds = [(artist.get_text(), artist.get_window_extent()) for artist in artists]
    for label, box in bounds:
        if not canvas.contains(box.x0, box.y0) or not canvas.contains(box.x1, box.y1):
            raise ValueError(f"text outside canvas: {label}")
    for position, (left_name, left) in enumerate(bounds):
        for right_name, right in bounds[position + 1 :]:
            overlap = left.intersection(left, right)
            if overlap is not None and overlap.width > 1 and overlap.height > 1:
                raise ValueError(f"text overlap: {left_name} / {right_name}")
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def agreement(receipts, output):
    plt.style.use("data-ink")
    plt.rcParams["axes.grid"] = False
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=100, sharey=True)
    for axis, receipt in zip(axes, receipts, strict=True):
        rows = receipt["rows"]
        ids = np.asarray([row["cell"] for row in rows])
        positions = np.arange(len(ids))
        errors = np.asarray([max(row["relative_error"]) for row in rows])
        errors = np.maximum(errors, 1e-18)
        largest = int(np.argmax(errors))
        axis.scatter(positions, errors, color=GREY, s=45, zorder=2)
        axis.scatter(positions[largest], errors[largest], color=BLUE, s=70, zorder=3)
        axis.axhline(1e-12, color=REFERENCE, linestyle=":", linewidth=1.2, zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(3e-19, 3e-11)
        axis.set_xticks(positions, labels=ids)
        axis.set_xlabel(f"cell ({receipt['cells']:,} total)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("scaled relative integral difference")
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=GREY, label="cell"),
            Line2D(
                [], [], marker="o", linestyle="", color=BLUE, label="largest difference"
            ),
            Line2D([], [], linestyle=":", color=REFERENCE, label="10⁻¹² limit"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    _save(figure, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.agreement:
        if len(args.agreement) != 3:
            parser.error("three carrier receipts are required")
        agreement(
            [json.loads(path.read_text()) for path in args.agreement],
            args.output / "per-cell-agreement.svg",
        )


if __name__ == "__main__":
    main()
