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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "point_program_attribution"

from nova.media.ink import DEFAULT_INK, trace_axes
from scripts.edge_sheet.figure import save_checked

_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
_STYLE = _HOME / ".config/matplotlib/stylelib/data-ink.mplstyle"
plt.style.use(str(_STYLE) if _STYLE.exists() else "data-ink")

STYLE = DEFAULT_INK.variant(label_fontsize=14.0)

# One colour per state, read off the block itself rather than off its position:
# the control is what the others are measured against, a stubbed block is one the
# compiler sees, and an erased block removed nothing.
STATE_COLOUR = {
    "control": DEFAULT_INK.contour_color,
    "stubbed": DEFAULT_INK.flux_color,
    "erased": "none",
}
ERASED_EDGE = "0.35"
LABEL_COLOUR = DEFAULT_INK.wall_color
SPAN_COLOUR = "0.35"

SHORT_LABEL = {
    "value path": "value path (primal moments)",
    "gradient": "derivative tower",
    "regular Hessian at corner heights": "corner-height Hessian",
    "near-corner weighted residual tangents": "residual tangents",
    "elliptic terms": "elliptic terms",
}
ARM_SHORT = {
    "unchanged program": "control",
    "corner-height Hessian stubbed": "corner-height",
    "gradient and Hessian stubbed": "derivative tower",
}


def block_state(block):
    return "erased" if block["erased"] else "stubbed"


def draw_share(panel, blocks, control_bytes):
    blocks = [block for block in blocks if block["block"] != "identity"]
    blocks = sorted(blocks, key=lambda block: -block["graph_share"])
    positions = list(range(len(blocks)))
    panel.barh(
        positions,
        [100.0 * block["graph_share"] for block in blocks],
        color=[STATE_COLOUR[block_state(block)] for block in blocks],
        edgecolor=[ERASED_EDGE if block["erased"] else "none" for block in blocks],
        linewidth=1.0,
        height=0.68,
        zorder=4,
    )
    panel.set_yticks(positions)
    panel.set_yticklabels(
        [SHORT_LABEL.get(block["label"], block["label"]) for block in blocks]
    )
    panel.invert_yaxis()
    panel.set_xlabel(f"lowered graph removed [% of {control_bytes:,} bytes]")
    widest = max(100.0 * block["graph_share"] for block in blocks)
    panel.set_xlim(0.0, widest * 1.22)
    for position, block in zip(positions, blocks):
        share = 100.0 * block["graph_share"]
        if block["erased"]:
            text = "erased — stub removes nothing"
            offset = widest * 0.02
        else:
            text = "%.1f" % share
            offset = widest * 0.02
        panel.text(
            max(share, 0.0) + offset,
            position,
            text,
            va="center",
            fontsize=14,
            color=LABEL_COLOUR,
        )
    return widest


def draw_compile(panel, arms):
    positions = list(range(len(arms)))
    seconds = [arm["compile_seconds"] for arm in arms]
    states = ["control"] + ["stubbed"] * (len(arms) - 1)
    panel.bar(
        positions,
        seconds,
        color=[STATE_COLOUR[state] for state in states],
        width=0.5,
        zorder=1,
    )
    panel.set_xticks(positions)
    panel.set_xticklabels([ARM_SHORT.get(arm["label"], arm["label"]) for arm in arms])
    panel.set_ylabel("cold H200 compile [s]")
    tallest = max(seconds)
    step = 50.0 if tallest > 150.0 else 10.0
    panel.set_yticks([step * index for index in range(1 + int(tallest * 1.25 / step))])
    panel.set_ylim(0.0, tallest * 1.20)
    for position, arm in zip(positions, arms):
        panel.text(
            position,
            arm["compile_seconds"] + tallest * 0.02,
            "%.1f" % arm["compile_seconds"],
            ha="center",
            va="bottom",
            fontsize=14,
        )
    # Each span sits in the clear gap beside its bar rather than across a bar, and
    # is drawn in neutral grey because nothing here is a broken limit.
    control_seconds = arms[0]["compile_seconds"]
    spans = (
        (
            0.5,
            arms[1]["compile_seconds"],
            control_seconds,
            "%.1f s held by the corner-height Hessian"
            % (control_seconds - arms[1]["compile_seconds"]),
        ),
        (
            1.5,
            arms[2]["compile_seconds"],
            control_seconds,
            "%.1f s held by all derivative instantiations"
            % (control_seconds - arms[2]["compile_seconds"]),
        ),
    )
    for xpos, low, high, label in spans:
        panel.annotate(
            "",
            xy=(xpos, low),
            xytext=(xpos, high),
            arrowprops={
                "arrowstyle": "<->",
                "color": SPAN_COLOUR,
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=2,
        )
        panel.plot(
            [xpos - 0.14, xpos + 0.02],
            [high, high],
            color=SPAN_COLOUR,
            lw=1.0,
            zorder=2,
        )
        panel.plot(
            [xpos - 0.14, xpos + 0.02],
            [low, low],
            color=SPAN_COLOUR,
            lw=1.0,
            zorder=2,
        )
        panel.text(
            xpos + 0.10,
            (low + high) / 2.0,
            label,
            fontsize=11.5,
            rotation=90,
            va="center",
            ha="left",
            color=SPAN_COLOUR,
            zorder=2,
        )
    panel.set_xlim(-0.75, 2.95)
    return tallest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    receipt = json.loads(args.attribution.read_text())
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(17.6, 7.4), gridspec_kw={"width_ratios": [1.45, 1.25]}
    )
    left = trace_axes(left, STYLE)
    right = trace_axes(right, STYLE)
    draw_share(left, receipt["blocks"], receipt["control_bytes"])
    draw_compile(right, receipt["arms"])
    figure.legend(
        handles=[
            Patch(color=STATE_COLOUR["control"], label="control (unstubbed)"),
            Patch(
                color=STATE_COLOUR["stubbed"],
                label="block stubbed, visible to the compiler",
            ),
            Patch(
                facecolor="none",
                edgecolor=ERASED_EDGE,
                linestyle=":",
                label="erased: stub removes nothing — zero-length bar, "
                "marked in text, no share",
            ),
            Line2D(
                [0],
                [0],
                color=SPAN_COLOUR,
                lw=1.2,
                label="seconds the block holds",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    texts = save_checked(figure, str(args.output), str(args.png) if args.png else None)
    print("wrote %s with %d text artists" % (args.output, len(texts)))
    print("labels: " + " | ".join(texts))


if __name__ == "__main__":
    main()
