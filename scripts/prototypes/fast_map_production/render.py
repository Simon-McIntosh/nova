"""Render measured continuity, fidelity, and terminal flux contours."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
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
    figure, axes = plt.subplots(5, 1, figsize=(14, 18), dpi=100, sharex=True)
    for data, name, color, dash in (
        (baseline, "unchanged", GREY, "--"),
        (candidate, "continuous wall", RUST, "-"),
    ):
        rows = data["scan"][1:]
        step = np.asarray([row["step"] for row in rows])
        residual = np.asarray([row["residual"] for row in rows])
        wall = np.asarray([row["wall_candidate"] for row in rows])
        support = np.asarray([len(row["support_ids"]) for row in rows])
        membership = np.asarray([row["membership_71"] for row in rows])
        current = np.asarray([row["cell_current_residual"] for row in rows])
        axes[0].semilogx(step, residual, color=color, linestyle=dash, linewidth=3)
        axes[1].semilogx(step, wall, color=color, linestyle=dash, linewidth=3)
        axes[2].semilogx(step, support, color=color, linestyle=dash, linewidth=3)
        axes[3].semilogx(step, membership, color=color, linestyle=dash, linewidth=3)
        axes[4].semilogx(step, current, color=color, linestyle=dash, linewidth=3)
        axes[0].text(step[-1], residual[-1], name, color=color, ha="right")
    axes[0].set_ylabel("scaled residual")
    axes[1].set_ylabel("wall candidate [Wb]")
    axes[2].set_ylabel("support cells")
    axes[3].set_ylabel("cell 71 membership")
    axes[4].set_ylabel("cell 71 scaled current residual")
    axes[4].set_xlabel("Newton step scale")
    _save(figure, axes, output)


def step_window(candidate, output):
    """Show the local support exchange without compressing its step axis."""
    _style()
    rows = candidate["scan"][1:]
    step = np.asarray([row["step"] for row in rows]) * 1e5
    quantities = (
        ("residual", "scaled residual", 1),
        ("wall_candidate", "wall candidate [µWb]", 1e6),
        ("membership_71", "cell 71 membership", 1),
        ("cell_current_residual", "cell 71 scaled current residual", 1),
    )
    figure, axes = plt.subplots(4, 1, figsize=(14, 14), dpi=100, sharex=True)
    for axis, (key, label, scale) in zip(axes, quantities, strict=True):
        value = np.asarray([row[key] for row in rows]) * scale
        axis.plot(step, value, color=RUST, linewidth=3)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Newton step scale [10⁻⁵]")
    _save(figure, axes, output)


def direction_scans(receipts, output):
    """Compare three fine scans against a common Newton-step scale."""
    _style()
    figure, axes = plt.subplots(5, 1, figsize=(14, 18), dpi=100, sharex=True)
    quantities = (
        ("residual", "scaled residual", 1),
        ("wall_candidate", "wall candidate [µWb]", 1e6),
        ("support_ids", "support cells", 1),
        ("membership_71", "cell 71 membership", 1),
        ("cell_current_residual", "cell 71 scaled current residual", 1),
    )
    for data, name, linestyle in zip(
        receipts, ("Newton", "seed 17", "seed 29"), ("-", "--", ":"), strict=True
    ):
        rows = data["scan"][1:]
        step = np.asarray([row["step"] for row in rows]) * 1e5
        for axis, (key, label, scale) in zip(axes, quantities, strict=True):
            value = (
                np.asarray(
                    [
                        len(row[key]) if key == "support_ids" else row[key]
                        for row in rows
                    ]
                )
                * scale
            )
            axis.plot(
                step, value, color=RUST, linestyle=linestyle, linewidth=3, label=name
            )
            axis.set_ylabel(label)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Newton step scale [10⁻⁵]")
    _save(figure, axes, output)


def stalled_correction(receipt, output):
    """Show the smooth minimum below the line search's trial-scale floor."""
    _style()
    rows = receipt["scan"]
    step = np.asarray([row["step"] for row in rows])
    residual = np.asarray([row["residual"] for row in rows])
    near = step <= 0.002
    step, residual = step[near], residual[near]
    minimum = np.argmin(residual)
    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    axis.plot(step, residual, color=RUST, linewidth=3)
    axis.plot(step[minimum], residual[minimum], "o", color=RUST, markersize=10)
    axis.axvline(1 / 2048, color=GREY, linestyle=":", linewidth=1.2)
    label_height = residual.max() - 0.1 * (residual.max() - residual.min())
    axis.text(1 / 2048, label_height, "smallest trial", color=GREY)
    axis.set_xlabel("Newton step scale")
    axis.set_ylabel("scaled residual")
    _save(figure, (axis,), output)


def resolution(receipts, output, unchanged=None, lean=None):
    _style()
    cells = np.asarray([row["cells"] for row in receipts])
    error = np.asarray([row["centre_map_flux_error"] for row in receipts])
    residual = np.asarray([row["centre_residual"] for row in receipts])
    figure, axis = plt.subplots(figsize=(14, 7), dpi=100)
    axis.loglog(cells, error, "o-", color=BLUE, linewidth=3, markersize=9)
    axis.loglog(cells, residual, "o--", color=RUST, linewidth=2.6, markersize=9)
    if unchanged:
        axis.loglog(cells, unchanged, "o:", color=GREY, linewidth=2.6, markersize=9)
        axis.text(
            cells[-1] * 1.04, unchanged[-1], "unchanged map", color=GREY, va="center"
        )
    if lean:
        axis.loglog(cells, lean, "o-.", color="#355f4d", linewidth=2.6, markersize=9)
        axis.text(cells[-1] * 1.04, lean[-1], "lean map", color="#355f4d", va="center")
    axis.set_xlabel("carrier cells")
    axis.set_ylabel("relative error")
    axis.xaxis.set_major_locator(FixedLocator(cells))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([str(cell) for cell in cells])
    axis.set_xlim(cells[0] * 0.8, cells[-1] * 1.7)
    axis.text(cells[-1] * 1.04, error[-1], "profile map", color=BLUE, va="center")
    axis.text(cells[-1] * 1.04, residual[-1], "seed residual", color=RUST, va="center")
    _save(figure, (axis,), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-probe", type=Path)
    parser.add_argument("--candidate-probe", type=Path)
    parser.add_argument("--window-probe", type=Path)
    parser.add_argument("--direction-probe", type=Path, action="append", default=[])
    parser.add_argument("--stall-probe", type=Path)
    parser.add_argument("--map", type=Path, action="append", default=[])
    parser.add_argument("--unchanged-error", type=float, action="append", default=[])
    parser.add_argument("--lean-error", type=float, action="append", default=[])
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
    if args.window_probe:
        step_window(
            json.loads(args.window_probe.read_text()), args.output / "step-window.svg"
        )
    if args.direction_probe:
        if len(args.direction_probe) != 3:
            parser.error("--direction-probe requires exactly three receipts")
        direction_scans(
            [json.loads(path.read_text()) for path in args.direction_probe],
            args.output / "direction-scans.svg",
        )
    if args.stall_probe:
        stalled_correction(
            json.loads(args.stall_probe.read_text()), args.output / "stall-scan.svg"
        )
    if args.map:
        resolution(
            [json.loads(path.read_text()) for path in args.map],
            args.output / "resolution.svg",
            args.unchanged_error,
            args.lean_error,
        )
    for receipt in args.terminal:
        data = json.loads(receipt.read_text())
        panel(data["panel"], args.output / f"{receipt.stem}-terminal.svg")


if __name__ == "__main__":
    main()
