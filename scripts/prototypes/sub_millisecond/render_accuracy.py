"""Plot subcell and quadratic-arc cut error on the analytic certificates."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import trace_axes


def render(
    limited_path: Path,
    diverted_path: Path,
    bell_limited_path: Path,
    bell_diverted_path: Path,
    output: Path,
):
    plt.style.use(
        "/home/ITER/mcintos/Code/dotfiles/matplotlib/stylelib/data-ink.mplstyle"
    )
    rows = (
        (
            "limited",
            json.loads(limited_path.read_text()),
            json.loads(bell_limited_path.read_text()),
            "#3366cc",
            "solid",
        ),
        (
            "diverted",
            json.loads(diverted_path.read_text()),
            json.loads(bell_diverted_path.read_text()),
            "#a46322",
            "dashed",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100, sharey=True)
    for axis in axes:
        trace_axes(axis)
        axis.axhline(1e-2, color="#666666", linewidth=1.2, linestyle="dotted")
        axis.set_yscale("log")
        axis.set_ylim(8e-5, 3e-2)
        axis.tick_params(labelsize=20, width=1.2)
    for label, subcell, bell, color, linestyle in rows:
        subcell_error = np.asarray(
            (subcell["rows"]["4"]["map_error"], subcell["rows"]["8"]["map_error"])
        )
        arc_error = np.asarray(
            [bell["rows"][str(count)]["map_error"] for count in (0, 4, 8)]
        )
        axes[0].plot(
            (0, 1), subcell_error, color=color, linewidth=3, linestyle=linestyle
        )
        axes[0].scatter((0, 1), subcell_error, color=color, s=55, zorder=4)
        axes[1].plot(
            (0, 1, 2), arc_error, color=color, linewidth=3, linestyle=linestyle
        )
        axes[1].scatter((0, 1, 2), arc_error, color=color, s=55, zorder=4)
        axes[1].text(2.15, arc_error[-1], label, color=color, fontsize=20, va="center")
    axes[0].set_xlim(-0.2, 1.5)
    axes[1].set_xlim(-0.2, 2.8)
    axes[0].set_xticks((0, 1), ("4×4", "8×8"))
    axes[1].set_xticks((0, 1, 2), ("chord", "4 arcs", "8 arcs"))
    axes[0].set_xlabel("fixed subcell rule", fontsize=22)
    axes[1].set_xlabel("local Bell cut", fontsize=22)
    axes[0].set_ylabel("analytic flux-map error", fontsize=22)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("limited", type=Path)
    parser.add_argument("diverted", type=Path)
    parser.add_argument("bell_limited", type=Path)
    parser.add_argument("bell_diverted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(
        args.limited,
        args.diverted,
        args.bell_limited,
        args.bell_diverted,
        args.output,
    )


if __name__ == "__main__":
    main()
