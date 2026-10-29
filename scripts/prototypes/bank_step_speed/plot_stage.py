"""Draw isolated warm-stage walls beside the whole bank step."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "read": "read (outside step)",
    "residual_and_booking": "flux and booking",
    "local_booking_tangent": "booking tangent",
    "geometry_tangent_one_direction": "geometry tangent, one direction",
    "jacobian_without_geometry": "Jacobian, frozen geometry",
    "jacobian_with_geometry": "Jacobian, full geometry",
    "equilibration": "equilibration",
    "lu_factor": "LU factor",
    "lu_solve": "LU solve",
    "whole_step": "whole step",
}


def stages(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("STAGE "):
            record = json.loads(line[6:])
            values[record["name"]] = record["median_ms"]
    assert set(LABELS) <= values.keys()
    return values


def readable(value):
    return f"{value:.2f}" if value < 10 else f"{value:.0f}"


def main(baseline, after, svg, png):
    plt.style.use("data-ink")
    before = stages(baseline)
    current = stages(after) if after else None
    names = list(LABELS)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 7.2))
    if current:
        ax.barh(
            y - 0.19,
            [before[name] for name in names],
            0.34,
            color="#929292",
            label="baseline",
        )
        ax.barh(
            y + 0.19,
            [current[name] for name in names],
            0.34,
            color="#2e8b57",
            label="current",
        )
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
        for index, name in enumerate(names):
            for offset, value, color in (
                (-0.19, before[name], "#666666"),
                (0.19, current[name], "#2e8b57"),
            ):
                ax.text(
                    value * 1.08,
                    index + offset,
                    readable(value),
                    va="center",
                    color=color,
                )
    else:
        colors = ["#2e8b57" if name == "whole_step" else "#929292" for name in names]
        ax.barh(y, [before[name] for name in names], 0.54, color=colors)
        for index, name in enumerate(names):
            value = before[name]
            ax.text(
                value * 1.08,
                index,
                readable(value),
                va="center",
                color="#2e8b57" if name == "whole_step" else "#666666",
            )
    ax.axvline(0.5, color="#777777", linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 1000)
    ax.set_yticks(y, [LABELS[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("warm wall [ms]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.87, bottom=0.13)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    root = ET.parse(svg)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(svg, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.after, args.svg, args.png)
