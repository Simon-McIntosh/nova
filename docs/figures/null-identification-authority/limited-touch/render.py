"""Render the converged limited-boundary wall-contact comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parent


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / name) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _hollow_lines(axes, first: int) -> None:
    for line in axes.lines[first:]:
        line.set_markerfacecolor("none")
        line.set_markeredgewidth(1.0)


def main() -> int:
    receipt = json.loads((ROOT / "receipt.json").read_text())
    before = _load("row-16-before.npz")
    after = _load("row-16-terminal-state.npz")
    rows = (receipt["before"]["rows"][0], receipt["after"]["rows"][0])
    states = (before, after)
    radius = before["radius_axis"]
    height = before["height_axis"]
    wall = before["wall"]
    shared_levels = np.unique(
        np.r_[
            np.linspace(
                min(float(before["raster"].min()), float(after["raster"].min())),
                max(float(before["raster"].max()), float(after["raster"].max())),
                18,
            ),
            [row["boundary_flux_wb"] for row in rows],
        ]
    )
    own_style = DEFAULT_INK.variant(axis_marker="^", axis_markersize=7.0)
    other_style = DEFAULT_INK.variant(
        axis_marker="^",
        axis_markersize=9.0,
        axis_color="#3366cc",
    )
    figure, axes_row = plt.subplots(
        1,
        2,
        figsize=(8.4, 4.8),
        dpi=DEFAULT_INK.figure_dpi,
        constrained_layout=True,
    )
    titles = ("Before · detached-lobe contact", "After · axis-connected contact")
    for index, (axes, state, row, title) in enumerate(
        zip(axes_row, states, rows, titles, strict=True)
    ):
        poloidal_axes(axes)
        poloidal.draw_flux_contours(
            axes,
            radius,
            height,
            state["raster"],
            shared_levels,
        )
        poloidal.draw_flux_contours(
            axes,
            radius,
            height,
            state["raster"],
            [row["boundary_flux_wb"]],
            color=DEFAULT_INK.separatrix_color,
            linewidth=DEFAULT_INK.separatrix_linewidth,
        )
        poloidal.draw_wall(axes, wall[:, 0], wall[:, 1])
        counterpart = rows[1 - index]
        first = len(axes.lines)
        poloidal.draw_nulls(
            axes,
            magnetic_axis=counterpart["axis_position_m"],
            style=other_style,
            contain=wall,
        )
        _hollow_lines(axes, first)
        poloidal.draw_nulls(
            axes,
            magnetic_axis=row["axis_position_m"],
            style=own_style,
            contain=wall,
        )
        contact = row["contact_position_m"]
        axes.plot(
            contact[0],
            contact[1],
            marker="o",
            markerfacecolor="#ffffff",
            markeredgecolor="#cc0000",
            markeredgewidth=1.2,
            markersize=5.5,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers,
        )
        axes.set_title(title, fontsize=9.0)
        axes.text(
            0.02,
            0.02,
            (
                f"boundary {row['boundary_flux_wb']:.6f} Wb\n"
                f"wall gap {row['contour_min_wall_distance_m']:.6f} m"
            ),
            transform=axes.transAxes,
            fontsize=7.5,
            verticalalignment="bottom",
            bbox=DEFAULT_INK.label_bbox,
            zorder=DEFAULT_INK.zorder_label,
        )
    radial_span = float(wall[:, 0].max() - wall[:, 0].min())
    vertical_span = float(wall[:, 1].max() - wall[:, 1].min())
    pad = 0.04 * max(radial_span, vertical_span)
    for axes in axes_row:
        axes.set_xlim(float(wall[:, 0].min()) - pad, float(wall[:, 0].max()) + pad)
        axes.set_ylim(float(wall[:, 1].min()) - pad, float(wall[:, 1].max()) + pad)
    figure.suptitle(
        "MAST 27079 · 35 ms · converged limited read\n"
        "shared contour levels; solid triangle: this read's axis; "
        "hollow triangle: counterpart axis; circle: published wall contact",
        fontsize=9.0,
    )
    figure.savefig(ROOT / "row-16-before-after.png", dpi=180)
    figure.savefig(ROOT / "row-16-before-after.svg")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
