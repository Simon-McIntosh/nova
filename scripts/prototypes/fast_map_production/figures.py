"""Render the fast-map figures from the H200 receipts the measurement rows wrote.

The input is the directory of JSON receipts the measure rows produce, so a
figure is rebuilt from the same numbers the research record cites rather than
from a second copy of the arithmetic. Each figure is written as one SVG under
the served figures directory; a PNG twin is rendered beside the run receipts
only for the producer's own inspection.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RIGHT = "#1f6f8b"
LEFT = "#9b5a2a"
ORIGINAL = "#5c5c66"


def _style():
    import matplotlib.style

    matplotlib.style.use("data-ink")


def _load(directory, name):
    return json.loads((directory / name).read_text())


def convergence(directory, out):
    corrected = _load(directory, "diverted-250-solve.json")
    control = _load(directory, "control-original-250-solve.json")
    figure, axis = plt.subplots(figsize=(14, 5))
    for receipt, colour, label in (
        (corrected, RIGHT, "corrected read and booking"),
        (control, ORIGINAL, "original read and booking (control)"),
    ):
        rows = receipt["history"]
        residual = [row["residual"] for row in rows]
        steps = np.arange(1, len(rows) + 1)
        axis.plot(steps, residual, color=colour, linewidth=3.0, zorder=3)
        axis.annotate(
            f"{label}\n{residual[-1]:.3e}",
            (steps[-1], residual[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=colour,
            fontsize=14,
            va="center",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.5, 15)
    axis.set_xlabel("correction")
    axis.set_ylabel("scaled sup residual")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(out / "diverted-convergence-traces.svg")
    plt.close(figure)


def scans(directory, out):
    diverted = _load(directory, "diverted-250-probe.json")
    limited = _load(directory, "limited-250-probe.json")
    figure, panels = plt.subplots(1, 2, figsize=(14, 5))
    for axis, receipt, colour, title, label in (
        (
            panels[0],
            diverted,
            RIGHT,
            "diverted stalled iterate",
            "Newton direction",
        ),
        (
            panels[1],
            limited,
            LEFT,
            "limited converged state",
            "Newton direction",
        ),
    ):
        rows = receipt["scan"]
        index = np.arange(1, len(rows) + 1)
        residual = [row["residual"] for row in rows]
        axis.plot(index, residual, color=colour, linewidth=3.0, zorder=3)
        peak = max(rows, key=lambda row: row["residual"])
        axis.text(
            0.02,
            0.96,
            f"max {peak['residual']:.3e}\nat step {peak['step']:.3g}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=colour,
            fontsize=14,
        )
        axis.margins(y=0.18)
        axis.set_yscale("log")
        axis.set_xlabel("scan step (increasing step scale)")
        axis.set_ylabel("scaled sup residual")
        axis.set_title(title, fontsize=16)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(out / "stalled-scan-residuals.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    arguments = parser.parse_args()
    _style()
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.inspection.mkdir(parents=True, exist_ok=True)
    convergence(arguments.receipts, arguments.output)
    scans(arguments.receipts, arguments.output)
    for name in ("diverted-convergence-traces", "stalled-scan-residuals"):
        source = arguments.output / f"{name}.svg"
        print(f"wrote {source}", flush=True)


if __name__ == "__main__":
    main()