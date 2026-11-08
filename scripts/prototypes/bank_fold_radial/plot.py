"""Render sampled branch conditioning with separate matrix scales."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.prototypes.bank_fold_vertical.plot import checked_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharex=True)
    for ax, arm, color, label in zip(
        axes,
        ("unpinned", "pinned"),
        ("#929292", "#2e6fa8"),
        (
            "bank matrix: smallest singular value",
            "radius-bordered matrix: smallest singular value",
        ),
        strict=True,
    ):
        rows = receipt["condition"][arm]
        ax.plot(
            [r["drive"] for r in rows],
            [r["smallest"] for r in rows],
            linestyle="none",
            marker="o",
            markersize=8,
            color=color,
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_xlim(0.9995, 1.0105)
        ax.set_xticks([1.0, 1.005, 1.01], labels=["1.000", "1.005", "1.010"])
        ax.set_xlabel("drive")
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(7e-7, 1e-3)
    axes[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
    axes[1].set_ylim(6e-7, 3e-5)
    axes[1].set_yticks([1e-6, 1e-5])
    axes[0].text(
        0.04,
        1.05,
        "unpinned · balanced moment blocks",
        transform=axes[0].transAxes,
        fontsize=14,
    )
    axes[1].text(
        0.04,
        1.05,
        "radius pin · row 10⁻³ · source 1 mT",
        transform=axes[1].transAxes,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.84, wspace=0.45)
    checked_save(fig, args.output)
    fig.savefig(args.png, dpi=100)
    print("RENDER canvas=1400x720 text_bounds=pass text_collisions=pass markers_only=1")
    plt.close(fig)


if __name__ == "__main__":
    main()
