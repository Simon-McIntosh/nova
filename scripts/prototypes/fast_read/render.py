"""Render measured moving-axis validity and read cost."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "fixed-shape-read"
import matplotlib.pyplot as plt
import numpy as np


COLOURS = {"limited": "#466b76", "diverted": "#987251"}
NEUTRAL = "#333333"
FAILURE = "#a33132"


def axes_style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.tick_params(labelsize=20, width=1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.spines["left"].set_linewidth(1.2)


def validity(rows, output):
    figure, axes = plt.subplots(2, 1, figsize=(14, 11), dpi=100, sharex=True)
    for axis, kind in zip(axes, ("limited", "diverted"), strict=True):
        data = rows[kind]["rows"]
        x = np.asarray([row["offset"] for row in data])
        actual = np.asarray([row["axis"][1] for row in data])
        expected = np.asarray([row["oracle_axis"][1] for row in data])
        valid = np.asarray([row["valid"] for row in data])
        axis.plot(x, expected, color=NEUTRAL, linestyle="--", linewidth=2.6)
        axis.plot(x, actual, color=COLOURS[kind], linewidth=3.0)
        axis.scatter(x[~valid], actual[~valid], color=FAILURE, s=45, zorder=3)
        axis.text(
            0.02,
            0.95,
            f"{kind}: {int(valid.sum())}/41 valid",
            transform=axis.transAxes,
            color=COLOURS[kind],
            fontsize=20,
            va="top",
        )
        axis.set_ylabel("axis Z [m]", fontsize=22)
        axes_style(axis)
    axes[-1].set_xlabel("vertical-field offset [span/pitch]", fontsize=22)
    axes[-1].annotate(
        "analytic axis",
        (x[-1], expected[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=NEUTRAL,
        fontsize=20,
    )
    figure.subplots_adjust(hspace=0.15, right=0.82)
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def cost(reads, output):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    labels = ("limited", "diverted")
    x = np.arange(2)
    baseline = [reads[kind]["baseline_wall_ms"] for kind in labels]
    candidate = [reads[kind]["wall_ms"] for kind in labels]
    kernels = [reads[kind]["trace"]["gpu_kernel_events"] for kind in labels]
    axes[0].plot(x, baseline, color=NEUTRAL, linewidth=2.6, marker="o")
    axes[0].plot(x, candidate, color=COLOURS["limited"], linewidth=3, marker="o")
    axes[0].axhline(0.12, color=FAILURE, linestyle=":", linewidth=1.2)
    axes[0].text(1.06, baseline[-1], "existing read", color=NEUTRAL, fontsize=20)
    axes[0].text(
        1.06, candidate[-1], "fixed-shape read", color=COLOURS["limited"], fontsize=20
    )
    axes[0].text(1.06, 0.12, "0.12 ms target", color=FAILURE, fontsize=20)
    axes[0].set_ylabel("wall per read [ms]", fontsize=22)
    axes[1].plot(x, kernels, color=COLOURS["diverted"], linewidth=3, marker="o")
    axes[1].text(
        1.06, kernels[-1], "fixed-shape read", color=COLOURS["diverted"], fontsize=20
    )
    axes[1].set_ylabel("GPU kernels per read", fontsize=22)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=20)
        axis.set_xlim(-0.15, 1.8)
        axes_style(axis)
    figure.subplots_adjust(wspace=0.45, right=0.82)
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def stages(reads, fused, output):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    labels = ("read", "map", "correction")
    x = np.arange(3)
    for kind in ("limited", "diverted"):
        data = fused[kind]
        walls = (reads[kind]["wall_ms"], data["map_wall_ms"], data["slice_wall_ms"])
        kernels = (
            reads[kind]["trace"]["gpu_kernel_events"],
            data["map_trace"]["gpu_kernel_events"],
            data["slice_trace"]["gpu_kernel_events"],
        )
        for axis, values in zip(axes, (walls, kernels), strict=True):
            axis.plot(x, values, color=COLOURS[kind], linewidth=3, marker="o")
            axis.text(2.08, values[-1], kind, color=COLOURS[kind], fontsize=20)
    axes[0].set_ylabel("wall per call [ms]", fontsize=22)
    axes[1].set_ylabel("GPU kernels per call", fontsize=22)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=20)
        axis.set_xlim(-0.2, 2.7)
        axes_style(axis)
    figure.subplots_adjust(wspace=0.45, right=0.82)
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sweeps = {
        kind: json.loads((args.run / f"{kind}-sweep.json").read_text())
        for kind in ("limited", "diverted")
    }
    reads = {
        kind: json.loads((args.run / f"{kind}-edge.json").read_text())
        for kind in ("limited", "diverted")
    }
    validity(sweeps, args.output / "axis-sweep.svg")
    cost(reads, args.output / "read-cost.svg")
    fused_paths = {
        kind: args.run / f"{kind}-fused-all.json" for kind in ("limited", "diverted")
    }
    if all(path.is_file() for path in fused_paths.values()):
        fused = {
            kind: json.loads(path.read_text()) for kind, path in fused_paths.items()
        }
        if all("slice_trace" in row for row in fused.values()):
            stages(reads, fused, args.output / "stage-cost.svg")


if __name__ == "__main__":
    main()
