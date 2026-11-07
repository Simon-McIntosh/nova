"""Mismatch between the assembled Jacobian and the fresh-read central difference.

One grouped bar per direction class, one colour per state, logarithmic
ordinate. The subject is the analytic Jacobian the step assembles; the
booking-to-central-difference series is drawn beside it because that is the
term the table names.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ("cut", "normal", "whole", "axis", "saddle", "random")
LABELS = {
    "cut": "cut",
    "normal": "normal form",
    "whole": "whole",
    "axis": "axis",
    "saddle": "saddle",
    "random": "random",
}
COLOURS = {"stalled-1e-2": "#c0392b", "certificate": "#2e8b57"}
NAMES = {"stalled-1e-2": "stalled 1e-2", "certificate": "certificate"}
STEP = "1e-06"
WIDTH = 0.36


def median_of(state, klass, key):
    values = [row[key] for row in state["directions"] if row["class"] == klass]
    values = [value for value in values if np.isfinite(value)]
    return float(np.median(values)) if values else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    states = {state["label"]: state for state in receipt["states"]}
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    positions = np.arange(len(CLASSES))
    series = (
        ("analytic-full", "Jacobian as assembled"),
        ("booking-to-cd", "booking term to central difference"),
    )
    for axis, (key, title) in zip(axes, series, strict=True):
        for offset, (label, state) in zip((-WIDTH / 2, WIDTH / 2), states.items(), strict=True):
            heights = [median_of(state, klass, f"{key}:h{STEP}:rel") for klass in CLASSES]
            axis.bar(
                positions + offset,
                heights,
                WIDTH,
                color=COLOURS[label],
                label=NAMES[label],
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels([LABELS[klass] for klass in CLASSES], rotation=0)
        axis.set_title(title, fontsize=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.axhline(1.0, color="#929292", linewidth=1.2, linestyle=":", zorder=2)
    axes[0].set_ylabel("relative mismatch at step 1e-6")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(args.output)
    figure.savefig(output, format="svg", facecolor="white")
    figure.savefig(output.with_suffix(".png"), format="png", dpi=140, facecolor="white")
    print("wrote", output)


if __name__ == "__main__":
    main()