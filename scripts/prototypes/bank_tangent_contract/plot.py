"""Compare warm booking wall and device storage on one H200."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox

COLORS = {"base": "#929292", "head": "#2e8b57"}


def record(path, label):
    for line in path.read_text().splitlines():
        if line.startswith(label + " "):
            return json.loads(line[len(label) + 1 :])
    raise ValueError(f"missing {label} receipt in {path}")


def main(base, head, out):
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    categories = (("booking tangent", "whole step"), ("one slice", "16 lanes"))
    series = {
        "base": (
            (record(base, "TANGENT")["wall_ms"], record(base, "STEP")["wall_ms"]),
            (
                record(base, "TANGENT")["memory"]["temp_size_in_bytes"] / 1e9,
                record(base, "BATCH_COMPILED")["memory"]["temp_size_in_bytes"] / 1e9,
            ),
        ),
        "head": (
            (record(head, "TANGENT")["wall_ms"], record(head, "STEP")["wall_ms"]),
            (
                record(head, "TANGENT")["memory"]["temp_size_in_bytes"] / 1e9,
                record(head, "BATCH_COMPILED")["memory"]["temp_size_in_bytes"] / 1e9,
            ),
        ),
    }
    for panel, axis in enumerate(axes):
        for arm, offset in (("base", -0.13), ("head", 0.13)):
            values = series[arm][panel]
            x = np.arange(2) + offset
            axis.scatter(x, values, s=140, color=COLORS[arm], label=arm, zorder=3)
            for position, value in zip(x, values, strict=True):
                axis.text(
                    position,
                    value + (1.1 if panel == 0 else 1.6),
                    f"{value:.1f}",
                    color=COLORS[arm],
                    ha="center",
                    va="bottom",
                )
        axis.set_xticks((0, 1), categories[panel])
        axis.set_xlim(-0.38, 1.38)
        axis.set_ylim(0, 70 if panel else 54)
        axis.set_ylabel("wall [ms]" if panel == 0 else "temporary device memory [GB]")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.82, wspace=0.28)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text()
    ]
    bounds = [(artist, artist.get_window_extent(renderer)) for artist in labels]
    for artist, box in bounds:
        if not fig.bbox.contains(*box.get_points()[0]) or not fig.bbox.contains(
            *box.get_points()[1]
        ):
            raise ValueError(f"text outside canvas: {artist.get_text()}")
    for index, (first, first_box) in enumerate(bounds):
        for second, second_box in bounds[index + 1 :]:
            if first_box.overlaps(second_box):
                raise ValueError(
                    f"text collision: {first.get_text()} / {second.get_text()}"
                )
    for panel, axis in enumerate(axes):
        for arm, offset in (("base", -0.13), ("head", 0.13)):
            for position, value in zip(
                np.arange(2) + offset, series[arm][panel], strict=True
            ):
                x, y = axis.transData.transform((position, value))
                mark = Bbox.from_bounds(x - 8, y - 8, 16, 16)
                for artist, box in bounds:
                    if re.fullmatch(r"\d+\.\d", artist.get_text()) and box.overlaps(
                        mark
                    ):
                        raise ValueError(f"text over data: {artist.get_text()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    svg = out.read_text()
    svg = re.sub(
        r'width="[^"]+" height="[^"]+"', 'width="31.1cm" height="16cm"', svg, count=1
    )
    out.write_text(svg)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.base, args.head, args.out)
