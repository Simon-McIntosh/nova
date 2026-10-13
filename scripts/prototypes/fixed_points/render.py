"""Plot synchronized fixed-point latency from persisted measurement rows."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    receipts = [
        json.loads(path.read_text()) for path in sorted(args.rows.glob("*.json"))
    ]
    plt.style.use("data-ink")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for column, kind in enumerate(("limited", "diverted")):
        for row_index, state in enumerate(("analytic", "picard")):
            axis = axes[row_index, column]
            axis.axhline(1, color="0.5", linewidth=1.2, linestyle=":")
            axis.text(
                0.03,
                0.96,
                kind + ", " + state,
                transform=axis.transAxes,
                va="top",
                fontsize=20,
            )
            for arm, colour, style in [
                ("cached-jet", "0.15", "-"),
                ("kernel", "0.45", "--"),
            ]:
                points = sorted(
                    (
                        receipt["cells"],
                        item["median_seconds"] * 1000,
                        item["p99_seconds"] * 1000,
                    )
                    for receipt in receipts
                    if receipt["kind"] == kind
                    for item in receipt["rows"]
                    if item["arm"] == arm and item["state_kind"] == state
                )
                if not points:
                    continue
                cells, median, high = zip(*points)
                axis.plot(
                    cells,
                    median,
                    color=colour,
                    linestyle=style,
                    linewidth=3,
                    marker="o",
                    markersize=6,
                )
                axis.vlines(cells, median, high, colors=colour, linewidth=1.2)
                axis.annotate(
                    arm,
                    (cells[-1], median[-1]),
                    xytext=(-4, 8),
                    textcoords="offset points",
                    ha="right",
                    color=colour,
                    fontsize=20,
                )
            axis.set_yscale("log")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(False)
            if row_index == 1:
                axis.set_xlabel("Cells")
            if column == 0:
                axis.set_ylabel("Fixed-point call [ms]")
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
