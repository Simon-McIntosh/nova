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
    gated = _load(directory, "diverted-250-wallgate.json")
    figure, axis = plt.subplots(figsize=(14, 5))
    for receipt, colour, label in (
        (control, ORIGINAL, "original read and booking (control)"),
        (gated, LEFT, "wall eligibility restored"),
        (corrected, RIGHT, "branchless wall candidate"),
    ):
        rows = receipt["history"]
        residual = [row["residual"] for row in rows]
        steps = np.arange(1, len(rows) + 1)
        axis.plot(
            steps,
            residual,
            color=colour,
            linewidth=3.0,
            zorder=3,
            label=f"{label}: {residual[-1]:.3e}")
    axis.set_yscale("log")
    axis.set_xlim(0.5, len(corrected["history"]) + 0.5)
    axis.set_xlabel("correction")
    axis.set_ylabel("scaled sup residual")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=14, loc="center right")
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


def taylor(directory, out):
    """Taylor remainders at the converged limited state and the stalled diverted one.

    A smooth map's remainder falls as the square of the step, so a log-log slope
    of two is the differentiability a Taylor remainder tests for. Both states
    reach that slope over the decade above each series' roundoff floor. They
    differ in extent rather than in slope: the converged limited state's two
    directions are defined through the largest step, while three of the
    diverted iterate's eleven rows are undefined at the largest steps, which is
    the interval the descent cannot cross.
    """
    limited = _load(directory, "limited-250-probe.json")
    diverted = _load(directory, "diverted-250-probe.json")
    figure, axis = plt.subplots(figsize=(14, 5.5))
    series = (
        (limited["stalled_taylor"], LEFT, "limited converged, Newton direction"),
        (
            limited["stalled_random_taylor"],
            "#7a5c1e",
            "limited converged, random direction",
        ),
        (diverted["stalled_taylor"], RIGHT, "diverted stalled, Newton direction"),
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("two-norm Taylor remainder")
    largest = 0.0
    for rows, colour, label in series:
        step = [row["step"] for row in rows]
        remainder = [row["remainder_two"] for row in rows]
        runs, current = [], []
        for point, value in zip(step, remainder):
            if value == value:
                current.append((point, value))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        run = max(runs, key=len)
        axis.plot(
            [point for point, _ in run],
            [value for _, value in run],
            color=colour,
            linewidth=2.6,
            zorder=3,
        )
        end = max(run, key=lambda item: item[0])
        largest = max(largest, end[0])
        axis.annotate(
            label,
            end,
            textcoords="offset points",
            xytext=(8, 0),
            color=colour,
            fontsize=13,
            va="center",
        )
    axis.set_xlim(left=1e-8, right=largest * 9)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("two-norm Taylor remainder")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(out / "taylor-remainder-slopes.svg")
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
    taylor(arguments.receipts, arguments.output)
    for name in (
        "diverted-convergence-traces",
        "stalled-scan-residuals",
        "taylor-remainder-slopes",
    ):
        source = arguments.output / f"{name}.svg"
        print(f"wrote {source}", flush=True)


if __name__ == "__main__":
    main()