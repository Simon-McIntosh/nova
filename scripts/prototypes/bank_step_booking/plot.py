"""Draw measured warm bank stages before and after a tangent change."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    ("read", "read (outside step)"),
    ("residual_and_booking", "residual and booking"),
    ("local_booking_tangent", "booking tangent"),
    ("jacobian_with_geometry", "full Jacobian"),
    ("lu_factor", "LU factor"),
    ("whole_step", "fresh whole step"),
)


def records(path, prefix):
    return [
        json.loads(line[len(prefix) :])
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]


def check_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    artists = [
        item
        for item in fig.findobj(match=plt.Text)
        if item.get_visible() and item.get_text()
    ]
    boxes = [(item, item.get_window_extent(renderer)) for item in artists]
    for item, box in boxes:
        if not canvas.contains(box.x0, box.y0) or not canvas.contains(box.x1, box.y1):
            raise ValueError(f"clipped figure text: {item.get_text()}")
    for index, (first, a) in enumerate(boxes):
        for second, b in boxes[index + 1 :]:
            if a.overlaps(b):
                raise ValueError(
                    f"overlapping figure text: {first.get_text()} / {second.get_text()}"
                )


def main(baseline, after, svg, png):
    plt.style.use("data-ink")
    before = {item["name"]: item["median_ms"] for item in records(baseline, "STAGE ")}
    current = {item["name"]: item["median_ms"] for item in records(after, "STAGE ")}
    names = [label for _, label in STAGES]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for offset, values, color, label in (
        (-0.15, before, "#929292", "batched baseline"),
        (0.15, current, "#2e8b57", "shared boundary"),
    ):
        for index, (name, _) in enumerate(STAGES):
            value = values[name]
            ax.plot([0, value], [index + offset] * 2, color=color, linewidth=2.6)
            ax.scatter([value], [index + offset], s=65, color=color, zorder=3)
        ax.plot([], [], color=color, linewidth=3, label=label)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(before.values()), max(current.values())) * 1.08)
    ax.set_xlabel("titan P100 warm wall [ms]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.14)
    check_text(fig)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    root = ET.parse(svg)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(svg, encoding="unicode", xml_declaration=True)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(
        f"FIGURE svg={svg} png={png} "
        f"booking_before_ms={before['local_booking_tangent']:.3f} "
        f"booking_after_ms={current['local_booking_tangent']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.after, args.svg, args.png)
