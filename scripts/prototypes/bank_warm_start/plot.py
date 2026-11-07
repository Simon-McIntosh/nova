"""Batch reads per slice for the three drive steps, in the data-ink style."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.media.ink import DEFAULT_INK, trace_axes

CANVAS = (14.0, 7.2)
COLOURS = {1e-5: "#2f6fd0", 1e-3: "#d97b16", 1e-2: "#7a4fbf"}
LABELS = {1e-5: "step 1e-5", 1e-3: "step 1e-3", 1e-2: "step 1e-2"}


def style():
    """The house sheet plus the ink module's trace furniture, at page sizes."""
    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    plt.rcParams.update({"axes.labelsize": 16, "legend.fontsize": 14})
    return DEFAULT_INK.variant(label_fontsize=14.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.report.read_text())
    ink = style()
    figure, axes = plt.subplots(figsize=CANVAS, dpi=100)
    trace_axes(axes, ink)
    arm = summary["arms"]["runner1"]["table"]
    steps = sorted({row["step"] for row in arm if row["slice"] > 0})
    offsets = {step: (index - (len(steps) - 1) / 2) * 0.16 for index, step in enumerate(steps)}
    for step in steps:
        rows = [row for row in arm if row["step"] == step and row["slice"] > 0]
        xs = [row["slice"] + offsets[step] for row in rows]
        ys = [row["reads_max"] for row in rows]
        axes.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=COLOURS[step],
            label=LABELS[step],
            zorder=3 if step == steps[-1] else 2,
        )
    axes.set_xlabel("slice index")
    axes.set_ylabel("batch reads per slice [count]")
    axes.set_xticks(sorted({row["slice"] for row in arm if row["slice"] > 0}))
    axes.set_yscale("log")
    axes.set_yticks([3, 10, 30, 100, 500])
    axes.set_yticklabels(["3", "10", "30", "100", "500"])
    axes.set_ylim(2.4, 900)
    axes.legend(
        frameon=False,
        ncol=len(steps),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        fontsize=ink.label_fontsize,
    )
    figure.tight_layout()
    figure.savefig(args.out, format="svg")
    figure.savefig(args.out.with_suffix(".png"), format="png", dpi=140)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()