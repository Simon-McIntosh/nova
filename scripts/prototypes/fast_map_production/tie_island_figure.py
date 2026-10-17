"""Draw the two tie islands from the probe scans.

Two panels, one per resolution, each with the competing levels and the boundary
flux above and the unnormalised booked current below, on the offset scale of
the state that produces the island. The bisected switch is drawn as a dotted
line in each panel.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("data-ink")
except OSError:
    pass

BOUNDARY = "#3b6ea5"
CURRENT = "#b06f1f"
WALL = "#929292"
SADDLE = "#222222"
SWITCH = "#929292"

PANELS = [
    "282 cells",
    "550 cells",
]


def series(path):
    """Read a probe scan into the arrays the panels draw."""
    scan = json.loads(Path(path).read_text())["scan"]
    return {
        "offset": np.array([row["offset"] for row in scan]),
        "wall": np.array([row["wall_candidate"] for row in scan]),
        "saddle": np.array([row["saddle_candidate"] for row in scan]),
        "flux": np.array([row["boundary_flux"] for row in scan]),
        "current": np.array([row["raw_current"] for row in scan]),
        "class": np.array([row["boundary_class"] for row in scan]),
    }


def draw(axes, data, title):
    """Plot one resolution: levels and flux above, booked current below."""
    top, bottom = axes
    top.plot(data["offset"], data["wall"], color=WALL, lw=2.4, zorder=2)
    top.plot(
        data["offset"], data["saddle"], color=SADDLE, lw=2.6, ls=(0, (6, 3)), zorder=4
    )
    top.plot(data["offset"], data["flux"], color=BOUNDARY, lw=3.0, zorder=3)
    bottom.plot(data["offset"], data["current"], color=CURRENT, lw=3.0, zorder=3)
    switched = np.flatnonzero(np.diff(data["class"]) != 0)
    for index in switched:
        left = data["offset"][index]
        right = data["offset"][index + 1]
        for axis in (top, bottom):
            axis.axvline((left + right) / 2, color=SWITCH, lw=1.2, ls=":", zorder=1)
    for axis in (top, bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    top.set_title(title, fontsize=14, loc="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    figure, grid = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
    left = series(args.left)
    right = series(args.right)
    draw(grid[:, 0], left, PANELS[0])
    draw(grid[:, 1], right, PANELS[1])
    grid[0, 0].set_ylabel("level [Wb]", fontsize=16)
    grid[1, 0].set_ylabel("booked current [A]", fontsize=16)
    grid[1, 0].set_xlabel("exterior jet offset", fontsize=16)
    grid[1, 1].set_xlabel("exterior jet offset", fontsize=16)
    for axis in grid.ravel():
        axis.tick_params(labelsize=14)
    handles = [
        plt.Line2D([], [], color=WALL, lw=2.4, label="wall contact level"),
        plt.Line2D([], [], color=SADDLE, lw=2.6, ls=(0, (6, 3)), label="saddle level"),
        plt.Line2D([], [], color=BOUNDARY, lw=3.0, label="boundary flux"),
        plt.Line2D([], [], color=CURRENT, lw=3.0, label="booked current"),
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.output, dpi=100)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
