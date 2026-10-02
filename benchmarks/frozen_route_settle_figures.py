"""Draw the frozen-route settle evidence figures from the persisted solve.

Job 1280457 (H200, betelgeuse) solved the bootstrapped accelerator machine at
the after revision and persisted the arrays this module redraws.  See the
evidence fragment for the measured receipts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes
from nova.media.sources.frame import coerce_wall_units


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs/figures/forward-solver-route-integrity/frozen-route-settle-evidence"
NPZ = FIGDIR / "route-settle-data.npz"
SUMMARY = FIGDIR / "route-settle-summary.json"
BEFORE = (
    ROOT
    / "docs/figures/millisecond-converged-solve/newton-krylov-route/fixed-point-rca/receipt.json"
)
AFTER = (
    ROOT
    / "docs/figures/forward-solver-route-integrity/frozen-route-settle-gate/rca-after/receipt.json"
)
TOL = 1.0e-6
LINE = 2.6
LEVEL_COUNT = 22


def trips(path):
    route = json.loads(path.read_text())["route"]
    values = np.asarray(route["active_set_residuals"], dtype=float)
    reason = route["termination_reason_name"]
    return values[np.isfinite(values) & (values > 0.0)], reason


def residual_figure():
    data = np.load(NPZ)
    summary = json.loads(SUMMARY.read_text())
    style = DEFAULT_INK
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.6), constrained_layout=True, facecolor=style.figure_facecolor
    )
    trace_axes(axes[0], style)
    trace = np.asarray(data["picard_trace"], dtype=float)
    keep = np.isfinite(trace) & (trace > 0.0)
    axes[0].plot(np.arange(trace.size)[keep], trace[keep], color=style.flux_color, linewidth=LINE)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("map evaluation", fontsize=style.label_fontsize)
    axes[0].set_ylabel("fixed-point residual", fontsize=style.label_fontsize)
    axes[0].set_title("Picard reference route", fontsize=style.label_fontsize + 1)
    trace_axes(axes[1], style)
    before, before_reason = trips(BEFORE)
    after, after_reason = trips(AFTER)
    after = np.r_[after, summary["newton_krylov"]["residual"]]
    axes[1].plot(
        np.arange(before.size), before, color=style.trace_cursor_color,
        linewidth=LINE, marker="o", markersize=style.trace_markersize,
    )
    axes[1].plot(
        np.arange(after.size), after, color=style.trace_cursor_color,
        linewidth=LINE, marker="o", markersize=style.trace_markersize, linestyle="--",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("active-set step", fontsize=style.label_fontsize)
    axes[1].set_title("Newton-Krylov before---after", fontsize=style.label_fontsize + 1)
    axes[1].set_xlim(-0.4, 5.0)
    for item in axes:
        item.axhline(
            TOL, color=style.contour_color,
            linewidth=style.contour_linewidth, linestyle=":",
        )
        item.text(
            0.02, TOL, "registered tolerance", transform=item.get_xaxis_transform(),
            color=style.contour_color, fontsize=style.label_fontsize, va="bottom",
        )
    axes[1].text(
        after.size - 1, after[-1], "  %s" % after_reason,
        color=style.trace_cursor_color, fontsize=style.label_fontsize, va="center",
    )
    figure.savefig(FIGDIR / "route-residual-before-after.png", dpi=style.figure_dpi)
    figure.savefig(FIGDIR / "route-residual-before-after.svg")
    plt.close(figure)


def panel_figure():
    data = np.load(NPZ)
    summary = json.loads(SUMMARY.read_text())
    style = DEFAULT_INK
    wall = coerce_wall_units(data["wall"])
    levels = poloidal.contour_levels(
        data["picard_grid"], LEVEL_COUNT,
        boundary=summary["picard"]["boundary_flux"], axis=summary["picard"]["axis_flux"],
    )
    figure, axes = plt.subplots(
        figsize=(7.4, 7.4), constrained_layout=True, facecolor=style.figure_facecolor
    )
    poloidal_axes(axes, style)
    poloidal.draw_flux_contours(
        axes, data["radius"], data["height"], data["picard_grid"], levels,
        color=style.contour_color, linewidth=1.6, wall=wall,
        )
    poloidal.draw_flux_contours(
        axes, data["radius"], data["height"], data["newton_grid"], levels,
        color=style.flux_color, linewidth=style.flux_linewidth, wall=wall,
    )
    poloidal.draw_wall(axes, units=wall, style=style)
    points = np.asarray(data["newton_xpoint"], dtype=float).reshape(-1, 2)
    axis_point = data["newton_axis"]
    if np.all(np.isfinite(points)):
        poloidal.draw_nulls(axes, magnetic_axis=axis_point, x_points=points, style=style)
    else:
        poloidal.draw_nulls(axes, magnetic_axis=axis_point, style=style)
    figure.savefig(
        FIGDIR / "terminal-flux-panel.png", dpi=style.figure_dpi
    )
    figure.savefig(FIGDIR / "terminal-flux-panel.svg")
    plt.close(figure)


if __name__ == "__main__":
    residual_figure()
    panel_figure()