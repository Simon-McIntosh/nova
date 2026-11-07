"""Plot the relative Jacobian mismatch per direction class at both states."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ("cut", "normal", "whole", "axis", "saddle", "random")
COLOURS = {
    "stalled-1e-2": "#c0392b",
    "certificate": "#2e8b57",
}
GLYPH = {"stalled-1e-2": "o", "certificate": "s"}
STEP = "1e-06"


def series(state, name):
    values = [
        row[f"analytic-full:h{STEP}:rel"]
        for row in state["directions"]
        if row["class"] == name
    ]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    states = {state["label"]: state for state in receipt["states"]}
    figure, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    width = 0.34
    positions = np.arange(len(CLASSES))
    for offset, (label, state) in zip((-width / 2, width / 2), states.items(), strict=True):
        heights = []
        for name in CLASSES:
            values = series(state, name)
            heights.append(float(np.median(values)) if values else np.nan)
        axis.bar(
            positions + offset,
            heights,
            width,
            color=COLOURS[label]
             if label in COLOURS
             else "#929292",
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASSES)
    axis.set_ylabel("relative mismatch at step 1e-6")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output, facecolor="white")
    print("wrote", args.output)


if __name__ == "__main__":
    main()