"""Draw the production row program's graph share and cold compile per source block."""

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import pwd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "point_program_attribution"

from nova.media.ink import DEFAULT_INK, trace_axes
from scripts.edge_sheet.figure import save_checked

_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
_STYLE = _HOME / ".config/matplotlib/stylelib/data-ink.mplstyle"
plt.style.use(str(_STYLE) if _STYLE.exists() else "data-ink")

STYLE = DEFAULT_INK.variant(label_fontsize=14.0)


def draw_share(panel, blocks):
    blocks = [block for block in blocks if block["block"] != "identity"]
    blocks = sorted(blocks, key=lambda block: -block["graph_share"])
    positions = list(range(len(blocks)))
    panel.barh(
        positions,
        [100.0 * block["graph_share"] for block in blocks],
        color=[DEFAULT_INK.flux_color if block["erased"] else DEFAULT_INK.contour_color
               for block in blocks],
        height=0.68,
        zorder=4,
    )
    panel.set_yticks(positions)
    panel.set_yticklabels([block["label"] for block in blocks])
    panel.invert_yaxis()
    panel.set_xlabel("lowered graph removed [% of 6,088,067 bytes]")
    widest = max(100.0 * block["graph_share"] for block in blocks)
    panel.set_xlim(0.0, widest * 1.22)
    for position, block in zip(positions, blocks):
        share = 100.0 * block["graph_share"]
        text = "erased" if block["erased"] else "%.1f" % share
        panel.text(
            max(share, 0.0) + widest * 0.02,
            position,
            text,
            va="center",
            fontsize=14,
        )
    return widest


def draw_compile(panel, arms):
    positions = list(range(len(arms)))
    seconds = [arm["compile_seconds"] for arm in arms]
    panel.bar(
        positions,
        seconds,
        color=[DEFAULT_INK.contour_color, DEFAULT_INK.flux_color, DEFAULT_INK.flux_color],
        width=0.56,
        zorder=4,
    )
    panel.set_xticks(positions)
    panel.set_xticklabels([arm["label"].replace(" ", "\n") for arm in arms])
    panel.set_ylabel("cold H200 compile [s]")
    tallest = max(seconds)
    step = 50.0 if tallest > 150.0 else 10.0
    ticks = [step * index for index in range(1 + int(tallest * 1.25 / step))]
    panel.set_yticks(ticks)
    panel.set_ylim(0.0, ticks[-1] + step * 1.6)
    for position, arm in zip(positions, arms):
        panel.text(
            position,
            arm["compile_seconds"] + tallest * 0.02,
            "%.1f" % arm["compile_seconds"],
            ha="center",
            va="bottom",
            fontsize=14,
        )
    held = arms[0]["compile_seconds"] - arms[1]["compile_seconds"]
    if held <= 0.005:
        return tallest
    panel.annotate(
        "",
        xy=(1, arms[1]["compile_seconds"]),
        xytext=(1, arms[0]["compile_seconds"]),
        arrowprops={"arrowstyle": "<->", "color": DEFAULT_INK.axis_color, "lw": 1.2},
    )
    panel.text(
        1.42,
        arms[0]["compile_seconds"] - held / 2,
        "%.1f s held" % held,
        fontsize=13,
        va="center",
    )
    return tallest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    receipt = json.loads(args.attribution.read_text())
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(15.4, 7.4), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    left = trace_axes(left, STYLE)
    right = trace_axes(right, STYLE)
    draw_share(left, receipt["blocks"])
    draw_compile(right, receipt["arms"])
    figure.legend(
        handles=[
            Patch(color=DEFAULT_INK.flux_color, label="stub removed nothing (erased)"),
            Patch(color=DEFAULT_INK.contour_color, label="share of graph or compile"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    texts = save_checked(figure, str(args.output), str(args.png) if args.png else None)
    print("wrote %s with %d text artists" % (args.output, len(texts)))
    print("labels: " + " | ".join(texts))


if __name__ == "__main__":
    main()