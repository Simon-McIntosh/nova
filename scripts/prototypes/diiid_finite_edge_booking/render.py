"""Draw one flux-contour panel per finite-edge row.

Each panel is the row's own terminal state: the solved flux as line contours at
the normalised levels the current is clipped on, the first wall, the magnetic
axis, the achieved X-point and EFIT's X-point. The normalised levels come from
the levels the solve itself used, recorded by the row's own design read, so a
panel cannot show a separatrix the source was not clipped at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from scripts.prototypes.diiid_xpoint_attribution.attribute import field
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)

ROWS = (
    ("frame-137-diverted", "137 diverted"),
    ("frame-146-diverted", "146 diverted"),
    ("frame-146-limited", "146 limited"),
    ("frame-179-limited", "179 limited"),
)
PSI_LEVELS = np.array([0.02, 0.2, 0.4, 0.6, 0.8, 0.95, 1.02])


def read(folder: Path) -> dict:
    """Return one row's terminal state, its receipts and its discriminators."""

    data = dict(np.load(folder / "terminal.npz"))
    return dict(
        data=data,
        receipt=json.loads((folder / "receipt.json").read_text()),
        shadow=json.loads((folder / "shadow-image.json").read_text()),
        source=json.loads((folder / "core-analysis.json").read_text()),
        edge=json.loads((folder / "finite-edge.json").read_text()),
    )


def panel(ax, row, label, limits):
    """Draw one row's terminal state and return its caption lines."""

    data, shadow, edge = row["data"], row["shadow"], row["edge"]
    radius, height = data["radius"], data["height"]
    topology = row["receipt"]["terminal_topology"]
    axis = np.asarray(topology["axis_rz_m"], dtype=float)
    x_point = np.asarray(shadow["lower_x_rz_m"], dtype=float)
    efit_x = np.asarray(shadow["efit_x_rz_m"], dtype=float)
    axis_flux = float(edge["axis_flux_wb"])
    boundary_flux = float(edge["boundary_flux_wb"])
    interpolant = field(radius, height, data["flux"])
    grid_r = np.linspace(limits["radius"][0], limits["radius"][1], 161)
    grid_z = np.linspace(limits["height"][0], limits["height"][1], 201)
    normalised = (interpolant(grid_r, grid_z) - axis_flux) / (boundary_flux - axis_flux)
    poloidal_axes(ax)
    draw_flux_contours(
        ax,
        grid_r,
        grid_z,
        normalised.T,
        PSI_LEVELS,
        color=SOLVED,
        linewidth=1.2,
        wall=data["limiter"],
        zorder=3,
    )
    draw_flux_contours(
        ax,
        grid_r,
        grid_z,
        normalised.T,
        [1.0],
        color=SOLVED,
        linewidth=3.0,
        wall=data["limiter"],
        zorder=4,
    )
    draw_nulls(
        ax,
        magnetic_axis=axis,
        x_points=x_point[None, :],
        style=replace(DEFAULT_INK, axis_color=SOLVED, xpoint_color=SOLVED),
        contain=data["limiter"],
    )
    ax.plot(
        *efit_x,
        ls="none",
        marker="X",
        ms=9,
        mfc="none",
        mec=REFERENCE,
        mew=1.6,
        zorder=15,
    )
    draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=5)
    ax.set_xlim(limits["radius"])
    ax.set_ylim(limits["height"])
    return (
        label,
        f"residual {shadow['residual']:.2g}; converged {shadow['converged']}",
        f"lower X {100 * shadow['lower_x_distance_m']:.2f} cm; "
        f"band {edge['max_cell_difference_a']:.3g} A",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    rows = [read(args.input / "edge" / name) for name, _ in ROWS]
    points = np.array(
        [
            point
            for row in rows
            for point in (row["shadow"]["lower_x_rz_m"], row["shadow"]["efit_x_rz_m"])
        ]
    )
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    limits = dict(
        radius=(centre[0] - 0.42, centre[0] + 0.42),
        height=(centre[1] - 0.42, centre[1] + 0.42),
    )
    figure = plt.figure(figsize=CANVAS, dpi=100)
    axes = [
        figure.add_axes(box)
        for box in (
            (0.02, 0.20, 0.23, 0.60),
            (0.26, 0.20, 0.23, 0.60),
            (0.51, 0.20, 0.23, 0.60),
            (0.75, 0.20, 0.23, 0.60),
        )
    ]
    notes = []
    for ax, row, (_, label) in zip(axes, rows, ROWS, strict=True):
        lines = panel(ax, row, label, limits)
        for offset, text in zip((0.155, 0.105, 0.055), lines, strict=True):
            notes.append(
                figure.text(
                    ax.get_position().x0 + 0.115, offset, text, ha="center", fontsize=14
                )
            )
    handles = [
        Line2D([], [], color=SOLVED, lw=1.6, label="solved flux"),
        Line2D([], [], color=SOLVED, lw=3.0, label="clipped level"),
        Line2D([], [], color=WALL, lw=1.5, label="first wall"),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="magnetic axis"),
        Line2D([], [], color=SOLVED, marker="x", ls="none", label="solved X-point"),
        Line2D([], [], color=REFERENCE, marker="X", ls="none", label="EFIT X-point"),
    ]
    figure.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    name = "terminal-states"
    save(
        figure,
        axes,
        notes,
        args.output / (name + ".svg"),
        args.previews / (name + ".png"),
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
