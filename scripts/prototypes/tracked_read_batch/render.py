"""Draw the read's per-slice wall against the batch size on one H200."""

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
GREY = "#929292"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text())
    sizes = np.asarray([rows[key]["batch"] for key in ORDER], dtype=float)
    per_slice = np.asarray([rows[key]["per_slice_ms"] for key in ORDER])

    style = DEFAULT_INK
    fig, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    trace_axes(axis, style)
    axis.plot(
        sizes,
        per_slice,
        color=style.sol_color,
        linewidth=3.0,
        marker="o",
        markersize=7,
        zorder=3,
    )
    axis.axhline(BOUND_MS, color=GREY, linewidth=1.2, linestyle=":", zorder=2)
    axis.text(
        sizes[0] / 1.3,
        BOUND_MS * 1.08,
        f"{BOUND_MS} ms per slice",
        color=GREY,
        fontsize=14,
        va="bottom",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes)
    axis.set_xticklabels([LABEL[key] for key in ORDER])
    axis.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
    axis.set_ylim(0.0, max(per_slice) * 1.2)
    axis.set_xlabel("slices per batch", fontsize=16)
    axis.set_ylabel("wall per slice [ms]", fontsize=16)
    index = ORDER.index("sixteen")
    axis.annotate(
        f"{per_slice[index]:.3f} ms at 16",
        (sizes[index], per_slice[index]),
        textcoords="offset points",
        xytext=(12, 10),
        color=style.sol_color,
        fontsize=14,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg")
    fig.savefig(args.output.with_suffix(".png"), dpi=100)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()