"""Draw time per HLO pass for the point jet beside the positive control."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: styling below sets sizes explicitly

HLO = "#3b6ea5"
STAGE = "#929292"
LIMIT = "#b3261e"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.figsize": (14, 7.2),
    }
)


def stage_of(label):
    lowered = label.lower()
    if any(token in lowered for token in ("llvm", "ir", "ptx", "codegen", "gpu")):
        return STAGE
    return HLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jet", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
    jet = json.loads(args.jet.read_text())
    control = json.loads(args.control.read_text())

    rows = sorted(jet["table"], key=lambda row: -row["seconds"])[: args.depth]
    labels = [row["pass"].split("/")[-1] for row in rows]
    values = [row["seconds"] for row in rows]
    colours = [stage_of(row["pass"]) for row in rows]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )

    positions = range(len(rows))
    left.barh(list(positions), values, color=colours, height=0.68, zorder=3)
    left.set_yticks(list(positions))
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlabel("time per pass [s]")
    top = max(values)
    for position, row, value in zip(positions, rows, values):
        left.text(
            value + top * 0.015,
            position,
            "%.2f s, %.0f%%" % (value, 100 * row["share_of_attributed"]),
            va="center",
            fontsize=14,
            color="#333333",
        )
    left.set_xlim(0, top * 1.28)
    left.set_title(
        "point jet: %d passes attributed, %.1f s of %.1f s compile"
        % (
            jet["distinct_passes"],
            jet["attributed_seconds"],
            jet["compile_seconds"],
        ),
        fontsize=14,
    )

    bars = [
        (
            "HLO passes",
            jet["attributed_seconds"],
            HLO,
        ),
        (
            "codegen tail",
            jet["seconds_after_last_dump"] or 0.0,
            STAGE,
        ),
    ]
    positions = range(len(bars))
    right.bar(
        list(positions),
        [entry[1] for entry in bars],
        color=[entry[2] for entry in bars],
        width=0.6,
        zorder=3,
    )
    right.set_xticks(list(positions))
    right.set_xticklabels([entry[0] for entry in bars])
    right.set_ylabel("seconds of the compile wall")
    right.set_title("point jet compile wall split", fontsize=14)
    for position, entry in zip(positions, bars):
        right.text(
            position,
            entry[1],
            "%.1f s" % entry[1],
            ha="center",
            va="bottom",
            fontsize=14,
            color="#333333",
        )
    right.set_ylim(0, max(entry[1] + entry[2] for entry in bars) * 1.15 * 0 + jet["compile_seconds"] * 1.12)

    control_cover = 100 * (control["attributed_seconds"] / control["compile_seconds"])
    if control_cover < 90:
        right.text(
            0.02,
            0.02,
            "control 64-op chain: %.0f%% of %.2f s attributed"
            % (control_cover, control["compile_seconds"]),
            transform=right.transAxes,
            fontsize=14,
            color=LIMIT,
        )

    figure.tight_layout()
    figure.savefig(args.output, dpi=100)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()