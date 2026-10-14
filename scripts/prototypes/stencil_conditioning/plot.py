"""Draw the spatial and numerical consequences of ring conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("data-ink")

KINDS = ("limited", "diverted")
CELLS = (550, 2000, 5000, 10000)
COLOR = {"limited": "#346080", "diverted": "#337c72"}


def _read(root: Path) -> dict:
    return {
        (kind, cells): (
            json.loads((root / f"{kind}-{cells}.json").read_text()),
            np.load(root / f"{kind}-{cells}.npz"),
        )
        for kind in KINDS
        for cells in CELLS
    }


def _spatial(rows: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, kind in zip(axes, KINDS):
        wall = np.asarray(rows[(kind, 10000)][0]["wall"])
        axis.plot(*np.vstack((wall, wall[0])).T, color="0.35", lw=1.8)
        for cells in CELLS:
            _record, array = rows[(kind, cells)]
            worst = int(np.argmax(array["condition"]))
            centre = array["centre"][array["stencil"][worst, 0]]
            color = "#af3834" if kind == "diverted" and cells == 10000 else COLOR[kind]
            axis.plot(*centre, "o", color=color, markersize=8)
            axis.annotate(
                f"{cells:,}",
                centre,
                xytext=(7, 7),
                textcoords="offset points",
                color=color,
                fontsize=16,
            )
        axis.text(0.04, 0.95, kind, transform=axis.transAxes, va="top", fontsize=20)
        axis.set_aspect("equal")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _condition(rows: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 6))
    for kind in KINDS:
        maximum = [rows[(kind, cells)][0]["condition"]["max"] for cells in CELLS]
        axis.plot(CELLS, maximum, marker="o", lw=3, color=COLOR[kind])
        axis.annotate(
            kind,
            (CELLS[-1], maximum[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=COLOR[kind],
        )
    axis.axhline(1000, color="0.55", linestyle=":", lw=1.2)
    axis.text(550, 1100, "admission limit", color="0.45", fontsize=20)
    axis.set(yscale="log", xlabel="requested cells", ylabel="maximum ring condition")
    axis.set_xlim(350, 11500)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _accuracy(rows: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, (field, label) in zip(
        axes,
        (
            ("gradient_error", "relative gradient error"),
            ("operator_error", "relative elliptic error"),
        ),
    ):
        for kind in KINDS:
            for cells in CELLS:
                record, array = rows[(kind, cells)]
                condition = array["condition"]
                scale = record["accuracy"][field.replace("error", "scale")]
                error = array[field] / scale
                sample = np.linspace(
                    0, len(condition) - 1, min(80, len(condition)), dtype=int
                )
                sample = np.unique(np.r_[sample, np.argsort(condition)[-10:]])
                axis.scatter(
                    condition[sample],
                    error[sample],
                    s=12,
                    color=COLOR[kind],
                    label=kind if cells == 550 else None,
                )
        axis.axvline(1000, color="0.55", linestyle=":", lw=1.2)
        axis.set(
            xscale="log", yscale="log", xlabel="normalised ring condition", ylabel=label
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _read(args.root)
    _spatial(rows, args.output / "ring-locations.svg")
    _condition(rows, args.output / "condition-by-cells.svg")
    _accuracy(rows, args.output / "analytic-error-by-condition.svg")


if __name__ == "__main__":
    main()
