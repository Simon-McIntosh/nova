"""Draw time per HLO pass for the point jet beside the positive control."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HLO = "#3b6ea5"
STAGE = "#929292"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
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
    if any(token in lowered for token in ("llvm", "ir-", ".ll", "autotuner")):
        return STAGE
    return HLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
    profiles = json.loads(args.profile.read_text())
    jet = profiles["point_jet-4"]
    control = profiles["control-1024"]
    small = profiles["control-64"]

    rows = jet["table"][: args.depth]
    labels = [row["pass"].split("/")[-1] for row in rows]
    values = [row["seconds"] for row in rows]
    colours = [stage_of(row["pass"]) for row in rows]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    positions = list(range(len(rows)))
    left.barh(positions, values, color=colours, height=0.68, zorder=3)
    left.set_yticks(positions)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlabel("time per pass [s]")
    top = max(values)
    for position, row, value in zip(positions, rows, values):
        left.text(
            value + top * 0.012,
            position,
            "%.1f s" % value,
            va="center",
            fontsize=14,
            color="#333333",
        )
    left.set_xlim(0, top * 1.22)

    split = [
        ("passes", jet["attributed_seconds"], HLO),
        ("before the\nfirst dump", jet["seconds_before_first_dump"], STAGE),
        ("after the\nlast dump", jet["seconds_after_last_dump"], STAGE),
    ]
    slots = list(range(len(split)))
    right.bar(
        slots,
        [entry[1] for entry in split],
        color=[entry[2] for entry in split],
        width=0.58,
        zorder=3,
    )
    right.set_xticks(slots)
    right.set_xticklabels([entry[0] for entry in split])
    right.set_ylabel("seconds of the compile wall")
    right.set_ylim(0, jet["compile_seconds"] * 1.5)
    for slot, entry in zip(slots, split):
        right.text(
            slot,
            entry[1],
            "%.1f" % entry[1],
            ha="center",
            va="bottom",
            fontsize=14,
            color="#333333",
        )
    right.text(
        0.03,
        0.99,
        "control %d-op chain:\n%.2f s, %.0f%% attributed\ncontrol %d-op chain:\n%.2f s, %.0f%% attributed"
        % (
            control["size"],
            control["compile_seconds"],
            100 * control["coverage_of_compile"],
            small["size"],
            small["compile_seconds"],
            100 * small["coverage_of_compile"],
        ),
        transform=right.transAxes,
        fontsize=14,
        color="#333333",
        va="top",
    )

    figure.tight_layout()
    figure.savefig(args.output, dpi=100)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()