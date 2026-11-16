# ruff: noqa: E402
"""Draw time per HLO pass for both cold-compile arms of one program.

The two arms are the revision-pinned base receipt and this tree's head receipt.
Passes are ordered by the base arm's own cost and drawn on one shared time axis,
so a pass that got cheaper cannot hide behind an independent scale; labels sit
outside their bars in dark text so both arms read the same way, and the right
panel states each arm's attributed and whole cold-compile wall.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "bounded-jets-build"

from matplotlib import rcParams

from nova.media.ink import DEFAULT_INK, trace_axes
from scripts.prototypes.bank_solve_profile.plot import save_checked

# Mid grey that holds about 3:1 against white for the base bars.
BASE_COLOUR = "#929292"
HEAD_COLOUR = DEFAULT_INK.flux_color
LABEL_COLOUR = "#1a1a1a"
STYLE = DEFAULT_INK.variant(label_fontsize=14.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()

    plt.style.use("data-ink")

    arms = {}
    for arm, path in (("base", args.base), ("head", args.head)):
        receipt = json.loads(path.read_text())
        table = receipt["table"]
        arms[arm] = {row["pass"].split("/")[-1]: row["seconds"] for row in table}
        arms[arm + "_passes"] = len(table)
        arms[arm + "_total"] = receipt["compile_seconds"]
        arms[arm + "_attributed"] = receipt["attributed_seconds"]
        arms[arm + "_label"] = "%s %s \u2014 whole wall %.1f s" % (
            arm,
            receipt["revision"][:9],
            receipt["compile_seconds"],
        )

    order = sorted(
        (label for label in arms["base"] if label in arms["head"]),
        key=lambda label: arms["base"][label],
        reverse=True,
    )[: args.depth]
    widest = max(max(arms["base"][label], arms["head"][label]) for label in order)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.8), gridspec_kw={"width_ratios": [2.1, 1.0]}
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
    left.barh(
        [position + offset for position in positions],
        [arms["head"][label] for label in order],
        color=HEAD_COLOUR,
        height=0.34,
        zorder=3,
    )
    left.set_yticks(positions)
    left.set_yticklabels(order)
    left.invert_yaxis()
    left.set_xlabel(
        "time per pass [s] \u2014 the %d costliest of the base arm's %d passes"
        % (len(order), arms["base_passes"])
    )
    left.set_xlim(0, widest * 1.26)
    left.set_xticks(np.arange(0, widest * 1.26, 25))
    for position, label in zip(positions, order):
        for arm, slot in (("base", -offset), ("head", offset)):
            left.text(
                arms[arm][label] + widest * 0.012,
                position + slot,
                "%.1f" % arms[arm][label],
                va="center",
                ha="left",
                fontsize=11,
                color=LABEL_COLOUR,
                zorder=4,
            )

    totals = [
        ("attributed", arms["base_attributed"], arms["head_attributed"]),
        ("cold compile", arms["base_total"], arms["head_total"]),
    ]
    slots = list(range(len(totals)))
    for slot, (label, base, head) in enumerate(totals):
        right.bar([slot - offset], [base], color=BASE_COLOUR, width=0.34, zorder=3)
        right.bar([slot + offset], [head], color=HEAD_COLOUR, width=0.34, zorder=3)
        right.text(
            slot - offset,
            base,
            "%.1f" % base,
            ha="center",
            va="bottom",
            fontsize=13,
            color=LABEL_COLOUR,
        )
        right.text(
            slot + offset,
            head,
            "%.1f" % head,
            ha="center",
            va="bottom",
            fontsize=13,
            color=LABEL_COLOUR,
        )
    ceiling = max(arms["base_total"], arms["head_total"])
    right.set_xticks(slots)
    right.set_xticklabels([label for label, _, _ in totals])
    right.set_ylabel("whole cold-compile wall [s]")
    right.set_ylim(0, ceiling * 1.16)
    # Name the ticks explicitly: autoscaling before the limit was set left a
    # tick above the top of the axes, which the clip check refuses.
    right.set_yticks(np.arange(0, ceiling * 1.16, 50))

    figure.legend(
        handles=[
            Patch(color=BASE_COLOUR, label=arms["base_label"]),
            Patch(color=HEAD_COLOUR, label=arms["head_label"]),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save_checked(figure, args.output)
    if args.png is not None:
        figure.savefig(args.png, dpi=150)
        print("PREVIEW=%s" % args.png, flush=True)
    print("wrote %s" % args.output)
    print("rc text color %s" % rcParams["text.color"], flush=True)


if __name__ == "__main__":
    main()
