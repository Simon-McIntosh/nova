"""Draw the diverted correction scan and resolution response."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.fast_read.render_contours import panel


BLUE = "#326c88"
RUST = "#9a5f47"
GREY = "#666666"


def _style():
    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "diverted-correction"
    plt.rcParams["axes.grid"] = False


def _finish(figure, axes, path):
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)


def _label(axis, x, y, name, color):
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite):
        index = finite[-1]
        axis.text(x[index], y[index], name, color=color, ha="right")


def step_scan(data, output):
    _style()
    rows = data["scan"][1:]
    step = np.asarray([row["step"] for row in rows])
    residual = np.asarray([row["residual"] for row in rows])
    cell = np.asarray([row["cell_current_residual"] for row in rows])
    fraction = np.asarray([row["membership_71"] for row in rows])
    saddle = np.asarray([row["saddle_moments_71"][0] for row in rows])
    ordinary = np.asarray([row["ordinary_moments_71"][0] for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100, sharex=True)
    axes[0].semilogx(step, residual, color=BLUE, linewidth=3)
    axes[0].semilogx(step, np.abs(cell), color=RUST, linewidth=2.6)
    axes[0].set_ylabel("scaled residual")
    _label(axes[0], step, residual, "global", BLUE)
    _label(axes[0], step, np.abs(cell), "cell 71", RUST)
    axes[1].semilogx(step, fraction, color=BLUE, linewidth=3)
    axes[1].set_ylabel("cell 71 booked fraction")
    axes[2].semilogx(step, saddle, color=RUST, linewidth=3)
    axes[2].semilogx(step, ordinary, color=GREY, linestyle="--", linewidth=2.6)
    axes[2].set_ylabel("cell 71 current moment [A]")
    axes[2].set_xlabel("Newton step scale")
    _label(axes[2], step, saddle, "saddle", RUST)
    present = np.flatnonzero(np.isfinite(ordinary) & (np.abs(ordinary) > 1e-12))
    if len(present):
        index = present[-1]
        axes[2].text(
            step[index], ordinary[index], "ordinary candidate", color=GREY, ha="right"
        )
    initial_class = data["scan"][0]["boundary_class"]
    switched = [row["step"] for row in rows if row["boundary_class"] != initial_class]
    if switched:
        for axis in axes:
            axis.axvline(switched[0], color=GREY, linestyle=":", linewidth=1.2)
        axes[0].text(
            switched[0] * 1.1,
            axes[0].get_ylim()[1] * 0.8,
            "wall path opens",
            color=GREY,
        )
    for axis in axes:
        axis.set_xlim(step[0], step[-1])
    _finish(figure, axes, output)


def resolution(rows, output):
    _style()
    cells = np.asarray([row["cells"] for row in rows])
    residual = np.asarray([row["residual"] for row in rows])
    map_error = np.asarray([row["map_error"] for row in rows])
    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    axis.loglog(cells, residual, "o-", color=RUST, linewidth=3, markersize=9)
    axis.loglog(cells, map_error, "o--", color=BLUE, linewidth=2.6, markersize=9)
    axis.set_xlabel("carrier cells")
    axis.set_ylabel("relative error")
    axis.text(cells[-1], residual[-1], "stall residual", color=RUST, ha="right")
    axis.text(cells[-1], map_error[-1], "analytic map error", color=BLUE, ha="right")
    axis.set_xticks(cells, labels=[str(cell) for cell in cells])
    _finish(figure, (axis,), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--terminal", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        step_scan(
            json.loads(args.probe.read_text()), args.output / "step-scale-scan.svg"
        )
    if args.resolution:
        resolution(
            json.loads(args.resolution.read_text()), args.output / "resolution.svg"
        )
    for receipt in args.terminal:
        data = json.loads(receipt.read_text())
        panel(data["panel"], args.output / f"{receipt.stem}-terminal.svg")


if __name__ == "__main__":
    main()
