"""Render convergence traces and terminal poloidal panels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.fast_read.render_contours import panel


COLORS = {"krylov": "#68717a", "gmres": "#326c88", "dense": "#9a5f47"}


def _style():
    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "fused-convergence"
    plt.rcParams["axes.grid"] = False


def residual_history(results, destination):
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100, sharey=True)
    for axis, carrier in zip(axes, ("limited", "diverted"), strict=True):
        for method in ("krylov", "gmres", "dense"):
            data = results.get((carrier, method))
            if not data or not data.get("history"):
                continue
            residual = [entry["residual"] for entry in data["history"]]
            x = np.arange(1, len(residual) + 1)
            axis.semilogy(x, residual, color=COLORS[method], linewidth=3)
            axis.text(x[-1] + 0.2, residual[-1], method, color=COLORS[method])
        axis.axhline(1e-12, color="#666666", linestyle=":", linewidth=1.2)
        axis.text(0.03, 0.97, carrier, transform=axis.transAxes, va="top")
        axis.set_xlabel("accepted correction")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("scaled moment residual")
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)


def taylor_remainder(data, destination):
    _style()
    figure, axis = plt.subplots(figsize=(14, 6), dpi=100)
    rows = data["taylor"]
    h = np.asarray([row["step"] for row in rows])
    colors = ("#326c88", "#9a5f47", "#686c58")
    names = ("current", "radial", "vertical")
    for block, (color, name) in enumerate(zip(colors, names, strict=True)):
        remainder = np.asarray([row["remainder"][block] for row in rows])
        axis.loglog(h, remainder, color=color, linewidth=3)
        axis.text(h[4] * 0.8, remainder[4], name, color=color, ha="right")
    guide = 3 * rows[2]["remainder"][0] * (h / h[2]) ** 2
    axis.loglog(h[1:4], guide[1:4], color="#666666", linestyle=":", linewidth=1.2)
    axis.text(h[2], guide[2], "slope 2", color="#666666", va="bottom")
    axis.set_xlabel("relative perturbation")
    axis.set_ylabel("Taylor remainder")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for carrier in ("limited", "diverted"):
        for method in ("krylov", "gmres", "dense"):
            path = args.receipts / f"{carrier}-{method}-accepted.json"
            if not path.exists():
                path = args.receipts / f"{carrier}-{method}.json"
            if path.exists():
                results[(carrier, method)] = json.loads(path.read_text())
    residual_history(results, args.output / "residual-history.svg")
    taylor = args.receipts / "diverted-taylor.json"
    if taylor.exists() and "taylor" in json.loads(taylor.read_text()):
        taylor_remainder(
            json.loads(taylor.read_text()), args.output / "taylor-remainder.svg"
        )
    for carrier in ("limited", "diverted"):
        for method in ("dense", "gmres", "krylov"):
            result = results.get((carrier, method), {})
            if "panel" not in result:
                panel_receipt = args.receipts / f"{carrier}-{method}-panel.json"
                if panel_receipt.exists():
                    result = json.loads(panel_receipt.read_text())
            if "panel" in result:
                panel(
                    result["panel"],
                    args.output / f"{carrier}-{method}-terminal.svg",
                )


if __name__ == "__main__":
    main()
