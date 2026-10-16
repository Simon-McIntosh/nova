"""Render the measured cost of the saddle-anchored map against its baseline."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTHS = (1, 16, 128)
ARMS = (
    ("axis-facing", "axis", "0.15", "--"),
    ("saddle-anchored", "saddle", "#7a4f9e", "-"),
)
SIZE_LIMIT_MB = 50.0


def render(rows, output, preview):
    """Draw per-state wall and generated size against batch width."""
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)
    wall_axis, size_axis = axes

    for label, key, colour, style in ARMS:
        wall = np.array(
            [rows["arms"][key][str(width)]["per_state_wall_ms"] for width in WIDTHS]
        )
        size = np.array(
            [
                rows["arms"][key][str(width)]["executable_bytes"] / 1e6
                for width in WIDTHS
            ]
        )
        wall_axis.plot(
            WIDTHS, wall, color=colour, linestyle=style, linewidth=2.8, label=label
        )
        size_axis.plot(
            WIDTHS, size, color=colour, linestyle=style, linewidth=2.8, label=label
        )

    size_axis.axhline(SIZE_LIMIT_MB, color="0.45", linewidth=1.2, linestyle=":")
    size_axis.text(
        WIDTHS[0],
        SIZE_LIMIT_MB,
        f"bound {SIZE_LIMIT_MB:g} MB",
        color="0.35",
        va="bottom",
    )

    for axis, ylabel in (
        (wall_axis, "per-state wall [ms]"),
        (size_axis, "executable [MB]"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks(WIDTHS, labels=[str(value) for value in WIDTHS])
        axis.set_xlim(0.9, 260)
        axis.set_xlabel("batch width")
        axis.set_ylabel(ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)

    handles, labels = wall_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "speed-gate-282.svg", metadata={"Date": None})
    preview.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview / "speed-gate-282-preview.png", dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.gate.read_text())
    render(rows, args.output, args.preview)
    print(
        f"RENDERED ratio={rows['saddle_over_axis_width128']:.4f} "
        f"pass={rows['gate_pass']} cells={rows['cells']}"
    )


if __name__ == "__main__":
    main()
