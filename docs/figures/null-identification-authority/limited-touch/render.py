"""Render the converged limited-boundary wall-contact comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.connectivity_boundary import _points_inside_polygon
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parent
DETACHED_COLOR = "#c4c4c4"
DETACHED_LINEWIDTH = 0.5


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / name) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _split_axis_enclosing(segments, axis):
    """Partition a level's segments into the axis-enclosing one and the rest."""
    enclosing, detached = [], []
    for segment in segments:
        points = np.asarray(segment, dtype=float)
        if points.shape[0] < 3:
            detached.append(points)
            continue
        inside = _points_inside_polygon(
            np.asarray([axis[0]]),
            np.asarray([axis[1]]),
            points[:, 0],
            points[:, 1],
        )
        (enclosing if bool(np.asarray(inside)[0]) else detached).append(points)
    return enclosing, detached


def _draw_boundary_components(axes, radius, height, flux, level, axis):
    """Draw one level: the axis-enclosing component in boundary style.

    A level of a diverted map also carries components that do not enclose the
    axis. Drawing the whole level set in boundary style paints those as the
    boundary, which is how a detached divertor lobe came to read as the
    published contact. They are faint here instead.
    """
    drawn = poloidal.draw_flux_contours(axes, radius, height, flux, [level])
    segments = [np.asarray(segment) for segment in drawn.allsegs[0]]
    drawn.remove()
    enclosing, detached = _split_axis_enclosing(segments, axis)
    for segment in detached:
        axes.plot(
            segment[:, 0],
            segment[:, 1],
            color=DETACHED_COLOR,
            linewidth=DETACHED_LINEWIDTH,
            solid_capstyle="round",
            zorder=DEFAULT_INK.zorder_flux,
        )
    for segment in enclosing:
        axes.plot(
            segment[:, 0],
            segment[:, 1],
            color=DEFAULT_INK.separatrix_color,
            linewidth=DEFAULT_INK.separatrix_linewidth,
            solid_capstyle="round",
            zorder=DEFAULT_INK.zorder_separatrix,
        )
    return enclosing, detached


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
        _enclosing, detached = _draw_boundary_components(
            axes,
            radius,
            height,
            state["raster"],
            row["boundary_flux_wb"],
            np.asarray(row["axis_position_m"], dtype=float),
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
        if detached:
            largest = max(detached, key=lambda segment: len(segment))
            axes.text(
                float(np.mean(largest[:, 0])),
                float(np.mean(largest[:, 1])),
                "detached",
                fontsize=7.0,
                fontstyle="italic",
                color=DETACHED_COLOR,
                horizontalalignment="center",
                verticalalignment="center",
                zorder=DEFAULT_INK.zorder_label,
            )
        axes.set_title(title, fontsize=9.0)
        axes.text(
            0.02,
            0.02,
            (
                f"boundary {row['boundary_flux_wb']:.6f} Wb\n"
                f"contact gap {row['contour_min_wall_distance_m']:.6f} m"
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
        "shared contour levels; boundary style marks only the axis-enclosing "
        "component, faint marks detached lobes; solid triangle: this read's axis; "
        "hollow triangle: counterpart axis; circle: published wall contact",
        fontsize=8.0,
    )
    figure.savefig(ROOT / "row-16-before-after.png", dpi=180)
    figure.savefig(ROOT / "row-16-before-after.svg")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
