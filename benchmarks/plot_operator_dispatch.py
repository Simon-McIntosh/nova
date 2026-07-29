"""Plot the operator-overhead, banding, and GPU-dispatch evidence."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "docs"
    / "figures"
    / "polybow-arc-section"
    / "operator_dispatch_evidence.json"
)
DEFAULT_FIGURE = DEFAULT_DATA.with_name("operator_dispatch_and_banding.png")


def plot(data: dict, output: pathlib.Path) -> None:
    """Write a three-panel decision figure."""
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

    overhead = data["frame_overhead"]
    for name, color in (("straight", "#c14953"), ("finite_arc", "#2d6a9f")):
        rows = overhead[name]
        axes[0].plot(
            [row["pairs"] for row in rows],
            [row["ratio"] for row in rows],
            marker="o",
            linewidth=2,
            label=name.replace("_", " "),
            color=color,
        )
    axes[0].axhline(1.0, color="0.45", linewidth=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="pairs in one solve",
        ylabel="frame / direct wall",
    )
    axes[0].set_title("Frame overhead is geometry-dependent")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, which="both")

    banding = data["arc_banding"]
    placement = banding["placement_worst_beyond_seam"]
    labels = ["centroid\n+ moments", "RMS\n+ moments", "bare RMS"]
    values = [
        placement["centroid_moment"],
        placement["rms_moment"],
        placement["rms_bare"],
    ]
    axes[1].bar(labels, values, color=["#2d6a9f", "#6b8eaa", "#c14953"])
    axes[1].axhline(
        1e-6, color="0.2", linestyle="--", linewidth=1, label="1e-6 envelope"
    )
    axes[1].set(yscale="log", ylabel="worst relative error beyond seam")
    axes[1].set_title("RMS placement does not buy accuracy")
    wall = banding["wall_4096_targets"]
    axes[1].text(
        0.03,
        0.04,
        f"4096 targets: {wall['speedup']:.2f}× faster\n"
        f"{100 * wall['exact_fraction']:.1f}% exact; "
        f"error {wall['worst_relative']:.1e}",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=0.25, axis="y", which="both")

    gpu = data["gpu_dispatch"]
    ring = gpu["ring_quadrature_320_by_320"]
    arc = gpu["finite_arc_one_pair"]
    names = ["ring\n1 GPU", "ring\n4 GPU", "arc host\n4096-wide", "arc GPU\n1-wide"]
    rates = [
        ring["one_h200_tile_80_previous_baseline_us_per_pair"],
        ring["four_h200_tile_80"]["warm_us_per_pair"],
        overhead["finite_arc"][-1]["direct_us_per_pair"],
        arc["warm_us_per_pair"],
    ]
    bars = axes[2].bar(names, rates, color=["#2d6a9f", "#3d9970", "#8c6d31", "#c14953"])
    axes[2].set(yscale="log", ylabel="warm microseconds per pair")
    axes[2].set_title("Shard rings; band arcs")
    axes[2].bar_label(
        bars, labels=[f"{value:.2g}" for value in rates], padding=3, fontsize=8
    )
    axes[2].text(
        0.03,
        0.96,
        f"arc cold compile: {arc['cold_seconds'] / 60:.1f} min\n"
        f"4-GPU ring cold: {ring['four_h200_tile_80']['cold_seconds']:.2f} s",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
    )
    axes[2].grid(alpha=0.25, axis="y", which="both")

    figure.suptitle(
        "Biot operator assembly: choose the route by geometry and batch shape",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output)


if __name__ == "__main__":
    main()
