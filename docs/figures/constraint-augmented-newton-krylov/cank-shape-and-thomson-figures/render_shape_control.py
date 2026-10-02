"""Regenerate the shape-steering and steering-authority poloidal panels.

Both panels compare a commanded shape move against the boundary the solve reached.
Each receipt carries its arm's convergence state, so the panels are drawn with the
axes off and the first wall present, and every arm that did not converge is drawn
dashed and named in the panel legend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from nova.equilibrium.wall_mask import vessel_unit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "docs/figures/constraint-augmented-newton-krylov/shape-control"
GEOMETRY = ROOT / "nova/catalog/mast_geometry.json"

PREVIOUS_COLOR = "#9e9e9e"
ELONGATION_COLOR = "#238b45"
SCAN_COLORS = {"5 mm": "#c6dbef", "10 mm": "#6baed6", "20 mm": "#2171b5"}
TURNING_POINT_KEYS = ("outer_m", "upper_m", "inner_m", "lower_m")


def _load(name):
    return json.loads((FIGURE_DIR / name).read_text())


def _mast_wall():
    catalogue = json.loads(GEOMETRY.read_text())["configurations"]
    geometry = next(iter(catalogue.values()))["geometry"]
    limiter = np.asarray(geometry["limiter"], float)
    return (vessel_unit(limiter[:, 0], limiter[:, 1], name="MAST limiter"),)


def _turning_points(container):
    return np.asarray([container[key] for key in TURNING_POINT_KEYS], float)


def _arm_state(arm):
    """Name the arm's recorded state and whether the panel must mark it refused."""
    if arm["achieved_boundary_rz_m"] is None:
        return "no qualified boundary", True
    if not bool(arm["qualified"]):
        return "NOT CONVERGED " + str(arm["termination"]), True
    return "converged", False


def _panel(axes, wall, previous, specs, title):
    poloidal_axes(axes)
    poloidal.draw_wall(axes, units=wall)
    boundary = np.asarray(previous[0], float)
    points = np.asarray(previous[1], float)
    axes.plot(
        boundary[:, 0],
        boundary[:, 1],
        color=PREVIOUS_COLOR,
        linewidth=2.4,
        zorder=DEFAULT_INK.zorder_separatrix,
    )
    axes.plot(
        points[:, 0],
        points[:, 1],
        linestyle="none",
        marker="x",
        color=PREVIOUS_COLOR,
        markersize=7.0,
        markeredgewidth=1.0,
        zorder=DEFAULT_INK.zorder_markers,
    )
    handles = [
        Line2D([], [], color=PREVIOUS_COLOR, linewidth=2.4, label="previous boundary"),
        Line2D(
            [],
            [],
            color=PREVIOUS_COLOR,
            linestyle="none",
            marker="x",
            markersize=7.0,
            label="previous turning points",
        ),
    ]
    for arm, color, label in specs:
        state, refused = _arm_state(arm)
        linestyle = "dashed" if refused else "solid"
        arm_boundary = arm["achieved_boundary_rz_m"]
        if arm_boundary is not None:
            achieved = np.asarray(arm_boundary, float)
            axes.plot(
                achieved[:, 0],
                achieved[:, 1],
                color=color,
                linewidth=3.0,
                linestyle=linestyle,
                zorder=DEFAULT_INK.zorder_flux,
            )
        commanded = _turning_points(arm["commanded_turning_points"])
        axes.plot(
            commanded[:, 0],
            commanded[:, 1],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=8.0,
            markeredgewidth=1.4,
            zorder=DEFAULT_INK.zorder_markers,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=3.0,
                linestyle=linestyle,
                label=label + ": " + state,
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markersize=8.0,
                label=label + " commanded",
            )
        )
    axes.relim()
    axes.autoscale_view()
    axes.legend(
        handles=handles,
        fontsize=7.0,
        frameon=False,
        loc="upper left",
        borderaxespad=0.2,
        handlelength=1.6,
        labelspacing=0.3,
    )
    axes.set_title(title, fontsize=11)


def render_shape_steering():
    data = _load("two-arms.json")
    previous = (
        data["converged"]["boundary_rz_m"],
        data["converged"]["target"]["flux_points_rz_m"],
    )
    wall = _mast_wall()
    palette = ["#2171b5", ELONGATION_COLOR]
    figure, axes_row = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
        dpi=DEFAULT_INK.figure_dpi,
        constrained_layout=True,
    )
    labels = ["upper point +2 cm", "elongation +5%"]
    for axes, arm, color, label in zip(axes_row, data["arms"], palette, labels):
        _panel(axes, wall, previous, [(arm, color, label)], label)
    figure.suptitle(
        "Bounding-box shape steering on 22086/43: commanded turning points against the achieved boundary",  # noqa: E501
        fontsize=13,
    )
    figure.savefig(FIGURE_DIR / "shape-steering.png")
    plt.close(figure)


def render_steering_authority():
    data = _load("steering-authority.json")
    previous = (
        data["converged"]["boundary_rz_m"],
        data["previous_turning_points_rz_m"],
    )
    wall = _mast_wall()
    panels = [
        ("newton_krylov_lifted", "elongation"),
        ("newton_krylov_lifted", "upper-scan"),
        ("reduced", "elongation"),
        ("reduced", "upper-scan"),
    ]
    figure, axes_grid = plt.subplots(
        2,
        2,
        figsize=(14, 14),
        dpi=DEFAULT_INK.figure_dpi,
        constrained_layout=True,
    )
    for axes, (route, family) in zip(axes_grid.ravel(), panels):
        entries = [
            entry
            for entry in data["arms"]
            if entry["route"] == route and entry["arm_family"] == family
        ]
        specs = []
        for entry in entries:
            command = entry["command_label"]
            color = (
                ELONGATION_COLOR
                if family == "elongation"
                else SCAN_COLORS.get(command, "#2171b5")
            )
            specs.append((entry["arm"], color, command))
        _panel(axes, wall, previous, specs, route + ": " + family)
    figure.suptitle(
        "Steering authority on 22086/43: commanded shape against the achieved boundary, per route",  # noqa: E501
        fontsize=13,
    )
    figure.savefig(FIGURE_DIR / "steering-authority.png")
    plt.close(figure)


def main():
    render_shape_steering()
    render_steering_authority()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
