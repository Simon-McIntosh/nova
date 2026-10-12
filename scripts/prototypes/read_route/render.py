"""Plot read-position and membership errors from completed row receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _completed_rows(directory: Path, kind: str, cells: int) -> list[dict]:
    receipt = directory / f"{kind}-{cells}.json"
    log = directory / f"{kind}-{cells}.log"
    if not log.exists() or not log.read_text().rstrip().endswith("EXIT=0"):
        return []
    if receipt.exists():
        return json.loads(receipt.read_text())["rows"]
    return [
        json.loads(line.removeprefix("READ_ROW "))
        for line in log.read_text().splitlines()
        if line.startswith("READ_ROW ")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    plt.style.use("data-ink")
    figure, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    colours = {"limited": "#427e7b", "diverted": "#856895"}
    for kind, colour in colours.items():
        for representation, linestyle in (("sampled", "--"), ("kernel", "-")):
            rows = []
            for cells in (2000, 5000, 10000):
                rows += [
                    row
                    for row in _completed_rows(args.rows, kind, cells)
                    if row["representation"] == representation
                ]
            if not rows:
                continue
            x = np.asarray([row["cells"] for row in rows])
            position = np.asarray(
                [
                    max(row["axis_error_pitches"], row["x_error_pitches"] or 0)
                    for row in rows
                ]
            )
            membership = np.asarray(
                [
                    max(
                        row["smooth_membership_error"],
                        row["saddle_membership_error"] or 0,
                    )
                    for row in rows
                ]
            )
            for axis, y in zip(axes, (position, membership), strict=True):
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=3,
                    linestyle=linestyle,
                    color=colour,
                    label=f"{kind} {representation}",
                )
    axes[0].axhline(1, color="0.5", linewidth=1.2, linestyle=":")
    axes[0].set_ylabel("Null position error [pitches]")
    axes[1].set_ylabel("Maximum membership error [median cell area]")
    axes[1].set_xlabel("Realised cells")
    axes[1].set_yscale("log")
    axes[1].set_xticks((2000, 5000, 10000))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    axes[0].legend(frameon=False, ncol=2)
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
