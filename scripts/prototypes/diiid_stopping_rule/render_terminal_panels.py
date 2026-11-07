"""Draw a measured terminal state as a line-contour poloidal panel.

The panel carries the governed wall, the EFIT label boundary, the terminal flux
contours on stated levels and the nulls the arm recorded (axis and admitted
saddle), so a reader can see where the state actually sat when the solve ended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--arm", default="default")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    data = np.load(arguments.artifact)
    flux = data[
        f"frames.{arguments.frame}.branches.{arguments.branch}.arms.{arguments.arm}._flux_grid"
    ]
    radius, height = data["radius"], data["height"]
    wall = data["wall"]
    label_boundary = data["label_boundary"]
    record = json.loads(arguments.record.read_text())
    arm = record["frames"][str(arguments.frame)]["branches"][arguments.branch]["arms"][
        arguments.arm
    ]
    axis = np.asarray(arm["presence"]["axis_rz_m"], dtype=float)
    saddle = arm.get("x_point_rz_m")
    saddles = (
        np.asarray(saddle, dtype=float).reshape(1, 2) if saddle else np.zeros((0, 2))
    )

    span = float(np.ptp(flux[np.isfinite(flux)]))
    finite = flux[np.isfinite(flux)]
    levels = np.linspace(finite.min() + 0.1 * span, finite.max() - 0.1 * span, 7)

    figure, axes = plt.subplots(figsize=(14, 7.2), dpi=100)
    poloidal_axes(axes)
    axes.contour(
        radius,
        height,
        flux,
        levels=levels,
        colors=DEFAULT_INK.flux_color,
        linewidths=2.6,
        zorder=4,
    )
    axes.plot(
        label_boundary[:, 0],
        label_boundary[:, 1],
        color="#929292",
        linestyle="--",
        linewidth=2.4,
        label="EFIT boundary",
    )
    draw_wall(axes, wall[:, 0], wall[:, 1])
    draw_nulls(axes, axis, saddles, style=DEFAULT_INK, contain=wall)
    axes.set_aspect("equal")
    lower, upper = wall.min(axis=0), wall.max(axis=0)
    margin = 0.06 * (upper - lower)
    axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    handles = [
        Line2D([], [], color=DEFAULT_INK.flux_color, lw=2.6, label="solved flux"),
        Line2D([], [], color="#929292", ls="--", lw=2.4, label="EFIT boundary"),
    ]
    axes.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=14,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    print(f"WROTE {arguments.output} residual={arm['terminal_residual']:.6g}")


if __name__ == "__main__":
    main()
