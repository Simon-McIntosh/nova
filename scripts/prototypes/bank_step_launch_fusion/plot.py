"""Draw the P100 warm bank step stages on a linear wall-time axis."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_step_booking.plot import check_text, records


STAGES = (
    ("residual_and_booking", "residual and booking"),
    ("local_booking_tangent", "booking tangent"),
    ("jacobian_with_geometry", "full Jacobian"),
    ("lu_factor", "LU factor"),
    ("whole_step", "fresh whole step"),
)


def main(baseline: Path, checkpoint: Path, after: Path, svg: Path):
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    before = {item["name"]: item["median_ms"] for item in records(baseline, "STAGE ")}
    intermediate = {
        item["name"]: item["median_ms"] for item in records(checkpoint, "STAGE ")
    }
    current = {item["name"]: item["median_ms"] for item in records(after, "STAGE ")}
    y = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for offset, values, color, label, marker, style in (
        (-0.22, before, "#929292", "base", "o", "-"),
        (0.0, intermediate, "#3478b5", "geometry batch", "s", "--"),
        (0.22, current, "#2e8b57", "booking batch", "D", "-"),
    ):
        for index, (name, _) in enumerate(STAGES):
            value = values[name]
            ax.plot(
                [0, value],
                [index + offset] * 2,
                color=color,
                linewidth=2.6,
                linestyle=style,
            )
            ax.scatter(
                [value], [index + offset], s=65, color=color, marker=marker, zorder=3
            )
        ax.plot([], [], color=color, linewidth=3, linestyle=style, label=label)
    ax.set_yticks(y, [label for _, label in STAGES])
    ax.invert_yaxis()
    maximum = max(
        before["whole_step"], intermediate["whole_step"], current["whole_step"]
    )
    ax.set_xlim(0, maximum * 1.08)
    ax.set_xticks(np.arange(0, np.ceil(maximum / 200) * 200 + 1, 200))
    ax.set_xlabel("titan P100 warm wall [ms]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.14)
    check_text(fig)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    plt.close(fig)
    root = ET.parse(svg)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(svg, encoding="unicode", xml_declaration=True)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(f"FIGURE {svg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.checkpoint, args.after, args.svg)
