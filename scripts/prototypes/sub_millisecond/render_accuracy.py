"""Plot the cut-rule error response to finer fixed subcell sampling."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import trace_axes


def render(limited_path: Path, diverted_path: Path, output: Path):
    plt.style.use(
        "/home/ITER/mcintos/Code/dotfiles/matplotlib/stylelib/data-ink.mplstyle"
    )
    rows = (
        ("limited", json.loads(limited_path.read_text()), "#3366cc", "solid"),
        ("diverted", json.loads(diverted_path.read_text()), "#a46322", "dashed"),
    )
    figure, axes = plt.subplots(figsize=(14, 5), dpi=100)
    trace_axes(axes)
    for label, receipt, color, linestyle in rows:
        counts = np.asarray((16, 64))
        error = np.asarray(
            (receipt["rows"]["4"]["map_error"], receipt["rows"]["8"]["map_error"])
        )
        axes.plot(counts, error, color=color, linewidth=3, linestyle=linestyle)
        axes.scatter(counts, error, color=color, s=55, zorder=4)
        label_y = 1.8e-2 if label == "diverted" else error[-1]
        axes.text(73, label_y, label, color=color, fontsize=20, va="center")
    axes.axhline(1e-2, color="#666666", linewidth=1.2, linestyle="dotted")
    axes.text(73, 8e-3, "acceptance ceiling", color="#555555", fontsize=20, va="center")
    axes.set_yscale("log")
    axes.set_xlim(10, 125)
    axes.set_ylim(1e-4, 3e-2)
    axes.set_xticks((16, 64))
    axes.set_xticklabels(("4×4", "8×8"))
    axes.set_xlabel("fixed subcells per fan triangle", fontsize=22)
    axes.set_ylabel("analytic flux-map error", fontsize=22)
    axes.tick_params(labelsize=20, width=1.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("limited", type=Path)
    parser.add_argument("diverted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.limited, args.diverted, args.output)


if __name__ == "__main__":
    main()
