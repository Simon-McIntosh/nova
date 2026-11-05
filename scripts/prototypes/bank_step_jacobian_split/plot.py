"""Draw paired whole-batch Jacobian walls from the measured receipt."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.prototypes.bank_solve_read.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    rows = {
        row["lanes"]: row["timings_ms"]
        for row in json.loads(args.receipt.read_text())["rows"]
        if row["step"] == 0
    }
    parts = [
        ("local", "Local booking sensitivities"),
        ("bank", "Bank channels + first contraction"),
        ("geometry", "Fresh geometry, whole"),
        ("stencil", "  Stencil construction"),
        ("channels", "  Geometry channel calls"),
        ("closing", "  Closing contraction"),
        ("matrix", "Matrix formation"),
        ("whole", "Whole Jacobian"),
    ]
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for lanes, offset, colour in ((1, -0.18, "#929292"), (16, 0.18, "#286eae")):
        for index, (key, _) in enumerate(parts):
            value = rows[lanes][key]
            y = index + offset
            ax.barh(
                y,
                value,
                height=0.27,
                color=colour,
                label=f"{lanes} slice" + ("s" if lanes > 1 else "")
                if index == 0
                else None,
                zorder=2,
            )
            ax.annotate(
                f"{value:.3f}" if value < 1 else f"{value:.2f}",
                (value, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=14,
                color="#333333",
            )
    ax.set_yticks(range(len(parts)), [name for _, name in parts])
    ax.set_ylim(len(parts) - 0.4, -0.6)
    upper = max(row["whole"] for row in rows.values()) * 1.2
    ax.set_xlim(0, upper)
    ax.set_xticks(range(0, int(upper) + 1, 50))
    ax.set_xlabel("Wall per batch [ms]")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.68, 0.99), ncols=2, frameon=False)
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.12, top=0.91)
    save_checked(fig, args.output)
    fig.savefig(args.preview, dpi=100)
    print(f"RENDER={args.preview}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
