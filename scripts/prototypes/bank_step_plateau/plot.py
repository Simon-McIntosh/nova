"""Plot same-state directional errors for each measured derivative component."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


BLOCKS = ("full", "booking", "geometry", "flux")


def check_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    boxes = [
        (item, item.get_window_extent(renderer))
        for item in fig.findobj(match=plt.Text)
        if item.get_visible() and item.get_text()
    ]
    for item, box in boxes:
        if not canvas.contains(box.x0, box.y0) or not canvas.contains(box.x1, box.y1):
            raise ValueError(f"clipped figure text: {item.get_text()}")
    for index, (first, a) in enumerate(boxes):
        for second, b in boxes[index + 1 :]:
            if a.overlaps(b):
                raise ValueError(
                    f"overlapping figure text: {first.get_text()} / {second.get_text()}"
                )


def main(head, base, svg, png):
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    rows = {
        "head": json.loads(head.read_text()),
        "base": json.loads(base.read_text()),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex=True)
    for axis, block in zip(axes.flat, BLOCKS, strict=True):
        for name, color, linestyle in (
            ("base", "#929292", "--"),
            ("head", "#2865b2", "-"),
        ):
            selected = [
                row
                for row in rows[name]
                if row["h_over_x"] == 1e-5 and row["plus_valid"] and row["minus_valid"]
            ]
            x = np.asarray([row["iterate"] for row in selected])
            y = np.asarray([max(row[block], 1e-16) for row in selected])
            axis.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=3 if name == "head" else 2.6,
                marker="o",
                markersize=5,
                zorder=3 if name == "head" else 2,
            )
        axis.set_yscale("log")
        axis.set_ylabel(f"{block} relative error")
        axis.set_xticks(np.arange(len({row["iterate"] for row in rows["head"]})))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    for axis in axes[-1]:
        axis.set_xlabel("accepted iterate")
    fig.legend(
        [axes.flat[0].lines[1], axes.flat[0].lines[0]],
        ["head P100", "base P100"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.89, bottom=0.13, hspace=0.33, wspace=0.27
    )
    check_text(fig)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    tree = ET.parse(svg)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(svg, encoding="unicode", xml_declaration=True)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(f"FIGURE svg={svg} png={png} h_over_x=1e-5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("head", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    main(args.head, args.base, args.svg, args.png)
