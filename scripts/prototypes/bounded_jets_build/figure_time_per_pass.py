# ruff: noqa: E402
"""Draw time per HLO pass for both cold-compile arms of one program.

The two arms are the revision-pinned base receipt and this tree's head receipt.
Passes are ordered by the base arm's own cost and drawn on one shared time axis,
so a pass that got cheaper cannot hide behind an independent scale; the right
panel states each arm's attributed and total compile wall.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "bounded-jets-build"

from nova.media.ink import DEFAULT_INK, trace_axes

plt.style.use("data-ink")

BASE_COLOUR = DEFAULT_INK.contour_color
HEAD_COLOUR = DEFAULT_INK.flux_color
STYLE = DEFAULT_INK.variant(label_fontsize=14.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()

    arms = {}
    for arm, path in (("base", args.base), ("head", args.head)):
        receipt = json.loads(path.read_text())
        arms[arm] = {
            row["pass"].split("/")[-1]: row["seconds"] for row in receipt["table"]
        }
        arms[arm + "_total"] = receipt["compile_seconds"]
        arms[arm + "_attributed"] = receipt["attributed_seconds"]

    order = sorted(
        (label for label in arms["base"] if label in arms["head"]),
        key=lambda label: arms["base"][label],
        reverse=True,
    )[: args.depth]
    widest = max(max(arms["base"][label], arms["head"][label]) for label in order)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    left = trace_axes(left, STYLE)
    right = trace_axes(right, STYLE)

    positions = list(range(len(order)))
    offset = 0.19
    left.barh(
        [position - offset for position in positions],
        [arms["base"][label] for label in order],
        color=BASE_COLOUR,
        height=0.34,
        zorder=3,
    )
    left.set_yticks(positions)
    left.set_yticklabels(order)
    left.invert_yaxis()
    left.set_xlabel("time per pass [s]")
    left.set_xlim(0, widest * 1.18)
    for position, label in zip(positions, order):
        left.text(
            widest * 0.012,
            position - offset,
            "%.1f" % arms["base"][label],
            va="center",
            ha="left",
            fontsize=12,
            color="white",
            zorder=4,
        )

    left.barh(
        [position + offset for position in positions],
        [arms["head"][label] for label in order],
        color=HEAD_COLOUR,
        height=0.34,
        zorder=3,
    )
    for position, label in zip(positions, order):
        left.text(
            arms["head"][label] + widest * 0.012,
            position + offset,
            "%.1f" % arms["head"][label],
            va="center",
            fontsize=12,
        )

    totals = [
        ("attributed", arms["base_attributed"], arms["head_attributed"]),
        ("cold compile", arms["base_total"], arms["head_total"]),
    ]
    slots = list(range(len(totals)))
    for slot, (label, base, head) in enumerate(totals):
        right.bar(
            [slot - offset],
            [base],
            color=BASE_COLOUR,
            width=0.34,
            zorder=3,
        )
        right.bar(
            [slot + offset],
            [head],
            color=HEAD_COLOUR,
            width=0.34,
            zorder=3,
        )
        right.text(
            slot - offset,
            base,
            "%.1f" % base,
            ha="center",
            va="bottom",
            fontsize=13,
        )
        right.text(
            slot + offset,
            head,
            "%.1f" % head,
            ha="center",
            va="bottom",
            fontsize=13,
        )
    ceiling = max(arms["base_total"], arms["head_total"])
    right.set_xticks(slots)
    right.set_xticklabels([label for label, _, _ in totals])
    right.set_ylabel("seconds")
    right.set_ylim(0, ceiling * 1.16)

    figure.legend(
        handles=[
            Patch(color=BASE_COLOUR, label="base receipt"),
            Patch(color=HEAD_COLOUR, label="head receipt"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="svg", metadata={"Date": None})
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
