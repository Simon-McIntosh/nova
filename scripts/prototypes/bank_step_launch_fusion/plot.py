"""Draw warm bank step stages on separate linear device axes."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.prototypes.bank_step_booking.plot import check_text, records


STAGES = (
    ("residual_and_booking", "residual and booking"),
    ("local_booking_tangent", "booking tangent"),
    ("jacobian_with_geometry", "full Jacobian"),
    ("lu_factor", "LU factor"),
    ("whole_step", "fresh whole step"),
)


def stage_walls(path: Path):
    return {item["name"]: item["median_ms"] for item in records(path, "STAGE ")}


def draw_device(ax, series, device, *, show_labels):
    y = np.arange(len(STAGES))
    maximum = 0.0
    for offset, values, color, _label, marker, style in series:
        for index, (name, _) in enumerate(STAGES):
            value = values[name]
            maximum = max(maximum, value)
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
    if show_labels:
        ax.set_yticks(y, [label for _, label in STAGES])
    else:
        ax.tick_params(axis="y", labelleft=False)
    scale = 10 ** np.floor(np.log10(maximum / 5))
    step = next(unit * scale for unit in (1, 2, 5, 10) if unit * scale >= maximum / 5)
    upper = np.ceil(maximum / step) * step
    ax.set_xlim(0, upper * 1.08)
    ax.set_xticks(np.arange(0, upper + step / 2, step))
    ax.set_xlabel(f"{device} warm wall [ms]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def main(
    baseline: Path,
    checkpoint: Path,
    after: Path,
    h200_base: Path,
    h200_after: Path,
    svg: Path,
):
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    before = stage_walls(baseline)
    intermediate = stage_walls(checkpoint)
    current = stage_walls(after)
    h_before = stage_walls(h200_base)
    h_current = stage_walls(h200_after)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
    draw_device(
        axes[0],
        (
            (-0.22, before, "#929292", "base", "o", "-"),
            (0.0, intermediate, "#3478b5", "geometry batch", "s", "--"),
            (0.22, current, "#2e8b57", "booking batch", "D", "-"),
        ),
        "titan P100",
        show_labels=True,
    )
    draw_device(
        axes[1],
        (
            (-0.15, h_before, "#929292", "base", "o", "-"),
            (0.15, h_current, "#2e8b57", "booking batch", "D", "-"),
        ),
        "H200",
        show_labels=False,
    )
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linewidth=3,
                linestyle=style,
                label=label,
            )
            for color, marker, style, label in (
                ("#929292", "o", "-", "base"),
                ("#3478b5", "s", "--", "geometry batch"),
                ("#2e8b57", "D", "-", "booking batch"),
            )
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.97),
        ncol=3,
    )
    fig.subplots_adjust(left=0.28, right=0.97, wspace=0.20, top=0.84, bottom=0.14)
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
    parser.add_argument("h200_base", type=Path)
    parser.add_argument("h200_after", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    main(
        args.baseline,
        args.checkpoint,
        args.after,
        args.h200_base,
        args.h200_after,
        args.svg,
    )
