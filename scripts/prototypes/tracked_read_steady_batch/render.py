"""Draw the batched steady read's per-slice wall against the batch size."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, trace_axes

BOUND_MS = 0.2
ORDER = ("single", "four", "sixteen", "sixtyfour")
LABEL = {"single": "1", "four": "4", "sixteen": "16", "sixtyfour": "64"}
# The bound and its label carry a scale, so the label is a neutral that clears
# AA at 14 pt on white rather than the faint grey a family of marks uses.
NEUTRAL = "#6b6b6b"
ARMS = (
    ("batch", "one branch for the batch", None, 3.0, "o"),
    ("vmap_conditional", "vmap of the read's conditional", "grey", 2.6, "s"),
    ("vmap_carried", "vmap carried, no branch", "black", 2.6, "^"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text())
    sizes = np.asarray([rows[f"{key}/batch"]["batch"] for key in ORDER], dtype=float)
    style = DEFAULT_INK
    fig, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    trace_axes(axis, style)

    for arm, label, tone, width, marker in ARMS:
        values = np.asarray([rows[f"{key}/{arm}"]["per_slice_ms"] for key in ORDER])
        color = (
            style.flux_color
            if tone is None
            else (NEUTRAL if tone == "grey" else "#1a1a1a")
        )
        axis.plot(
            sizes,
            values,
            color=color,
            linewidth=width,
            linestyle="-" if tone is None else "--",
            marker=marker,
            markersize=6,
            label=label,
            zorder=4 if tone is None else 3,
        )
        index = ORDER.index("sixteen")
        axis.annotate(
            label,
            (sizes[index], values[index]),
            textcoords="offset points",
            xytext=(0, -16 if tone != "grey" else 10),
            ha="center",
            color=color,
            fontsize=14,
        )

    axis.axhline(
        BOUND_MS, color=style.flux_color, linewidth=1.2, linestyle=":", zorder=2
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([LABEL[key] for key in ORDER])
    axis.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.6)
    axis.text(
        sizes[-1] * 1.05,
        BOUND_MS,
        f"{BOUND_MS} ms bound",
        color=NEUTRAL,
        fontsize=14,
        va="center",
        ha="left",
        zorder=5,
    )
    axis.set_xlabel("slices per batch", fontsize=16)
    axis.set_ylabel("wall per slice [ms]", fontsize=16)
    axis.legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
