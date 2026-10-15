"""Render measured continuity, fidelity, and terminal flux contours."""

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
    plt.rcParams["svg.hashsalt"] = "fast-map-continuity"
    plt.rcParams["axes.grid"] = False


def _save(figure, axes, output):
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def step_scan(baseline, candidate, output):
    _style()
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100, sharex=True)
    for data, name, color, dash in (
        (baseline, "unchanged", GREY, "--"),
        (candidate, "coherent booking", BLUE, "-"),
    ):
        rows = data["scan"][1:]
        step = np.asarray([row["step"] for row in rows])
        residual = np.asarray([row["residual"] for row in rows])
        wall = np.asarray([row["wall_candidate"] for row in rows])
        support = np.asarray([len(row["support_ids"]) for row in rows])
        axes[0].semilogx(step, residual, color=color, linestyle=dash, linewidth=3)
        axes[1].semilogx(step, wall, color=color, linestyle=dash, linewidth=3)
        axes[2].semilogx(step, support, color=color, linestyle=dash, linewidth=3)
        axes[0].text(step[-1], residual[-1], name, color=color, ha="right")
    axes[0].set_ylabel("scaled residual")
    axes[1].set_ylabel("wall candidate [Wb]")
    axes[2].set_ylabel("support cells")
    axes[2].set_xlabel("Newton step scale")
    _save(figure, axes, output)


def resolution(receipts, output):
    _style()
    cells = np.asarray([row["cells"] for row in receipts])
    error = np.asarray([row["centre_map_flux_error"] for row in receipts])
    residual = np.asarray([row["centre_residual"] for row in receipts])
    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    axis.loglog(cells, error, "o-", color=BLUE, linewidth=3, markersize=9)
    axis.loglog(cells, residual, "o--", color=RUST, linewidth=2.6, markersize=9)
    axis.set_xlabel("carrier cells")
    axis.set_ylabel("relative error")
    axis.set_xticks(cells, labels=[str(cell) for cell in cells])
    axis.text(cells[-1], error[-1], "map image", color=BLUE, ha="right")
    axis.text(cells[-1], residual[-1], "seed residual", color=RUST, ha="right")
    _save(figure, (axis,), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-probe", type=Path)
    parser.add_argument("--candidate-probe", type=Path)
    parser.add_argument("--map", type=Path, action="append", default=[])
    parser.add_argument("--terminal", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.baseline_probe and args.candidate_probe:
        step_scan(
            json.loads(args.baseline_probe.read_text()),
            json.loads(args.candidate_probe.read_text()),
            args.output / "step-scale-scan.svg",
        )
    if args.map:
        resolution(
            [json.loads(path.read_text()) for path in args.map],
            args.output / "resolution.svg",
        )
    for receipt in args.terminal:
        data = json.loads(receipt.read_text())
        panel(data["panel"], args.output / f"{receipt.stem}-terminal.svg")


if __name__ == "__main__":
    main()
