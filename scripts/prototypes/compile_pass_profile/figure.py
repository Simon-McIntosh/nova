"""Draw time per HLO pass for the point jet and the split of its compile wall."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "compile-pass-profile"

from nova.media.ink import DEFAULT_INK, trace_axes

plt.style.use("data-ink")

HLO_COLOUR = DEFAULT_INK.flux_color
STAGE_COLOUR = DEFAULT_INK.contour_color
STYLE = DEFAULT_INK.variant(label_fontsize=14.0)

CODEGEN = ("ir-no-opt", "ir-with-opt", "priority_fusion_dump", "llvm")


def codegen_stage(label):
    """Name the codegen stage a dump belongs to, or None for an HLO pass."""
    return next((token for token in CODEGEN if token in label.lower()), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", default="point_jet-4")
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())[args.program]

    rows = []
    for row in profile["table"][: args.depth]:
        stage = codegen_stage(row["pass"])
        rows.append(
            {
                "label": stage or row["pass"].split("/")[-1],
                "seconds": row["seconds"],
                "colour": STAGE_COLOUR if stage else HLO_COLOUR,
            }
        )

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    left = trace_axes(left, STYLE)
    right = trace_axes(right, STYLE)

    positions = list(range(len(rows)))
    left.barh(
        positions,
        [row["seconds"] for row in rows],
        color=[row["colour"] for row in rows],
        height=0.68,
        zorder=3,
    )
    left.set_yticks(positions)
    left.set_yticklabels([row["label"] for row in rows])
    left.invert_yaxis()
    left.set_xlabel("time per pass [s]")
    widest = max(row["seconds"] for row in rows)
    for position, row in zip(positions, rows):
        left.text(
            row["seconds"] + widest * 0.012,
            position,
            "%.1f" % row["seconds"],
            va="center",
            fontsize=14,
        )
    left.set_xlim(0, widest * 1.16)

    before = profile["seconds_before_first_dump"]
    passes = profile["attributed_seconds"]
    after = profile["seconds_after_last_dump"]
    wall = profile["compile_seconds"]
    split = (
        ("before first\ndump", before, STAGE_COLOUR),
        ("passes", passes, HLO_COLOUR),
        ("after last\ndump", after, STAGE_COLOUR),
    )
    slots = list(range(len(split)))
    right.bar(
        slots,
        [entry[1] for entry in split],
        color=[entry[2] for entry in split],
        width=0.56,
        zorder=3,
    )
    right.set_xticks(slots)
    right.set_xticklabels([entry[0] for entry in split])
    right.set_ylabel("compile wall [s]")
    right.set_ylim(0, wall * 1.15)
    for slot, entry in zip(slots, split):
        right.text(
            slot,
            entry[1] + wall * 0.012,
            "%.2f" % entry[1],
            ha="center",
            va="bottom",
            fontsize=14,
        )

    figure.legend(
        handles=[
            Patch(color=HLO_COLOUR, label="HLO pass"),
            Patch(color=STAGE_COLOUR, label="codegen stage"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(args.output, format="svg", metadata={"Date": None})
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
