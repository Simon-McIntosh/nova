"""Render the per-frame poloidal flux-contour panels for the convergence record.

Each panel draws the bank's terminal diverted field as line contours and, on
the same axes and at the same normalised levels, the EFIT label field.  Both
null sets are drawn in the ``nova.media`` vocabulary, the first wall is drawn
from the governed limiter ring, and the caption carries the terminal residual
and the converged flag.

Frame 144's limited seed has no solved field: it refuses at axis identification,
before any topology read, so that seed has no panel.  Frame 144's diverted seed
solves and is drawn like every other frame.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.text import Text

from benchmarks import diiid_forward_gs_match as bench
from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    inside_wall_units,
)

OPERAND_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/evidence/operands/diiid-forward-gs"
)
LEVEL_COUNT = 9


def _display_operand(frame: int) -> Path | None:
    matches = sorted(OPERAND_ROOT.glob(f"frame-display-*-{frame}.npz"))
    return matches[0] if matches else None


def _oriented(flux: np.ndarray, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return the flux map in the (height, radius) order the painter requires."""

    array = np.asarray(flux, dtype=float)
    if array.shape == (len(height), len(radius)):
        return array
    if array.shape == (len(radius), len(height)):
        return array.T
    raise ValueError(
        f"flux shape {array.shape} does not match radius {len(radius)} "
        f"and height {len(height)}"
    )


def _bilinear(radius, height, flux, point) -> float:
    r = float(
        np.interp(point[0], radius, np.arange(len(radius)), left=np.nan, right=np.nan)
    )
    z = float(
        np.interp(point[1], height, np.arange(len(height)), left=np.nan, right=np.nan)
    )
    if not np.isfinite(r) or not np.isfinite(z):
        return float("nan")
    ri = int(round(r))
    zi = int(round(z))
    return float(flux[zi, ri])


def _edge_flux(radius, height, flux, boundary) -> float:
    """Return the median flux the map carries on the label boundary."""

    values = np.asarray(
        [_bilinear(radius, height, flux, point) for point in np.asarray(boundary)],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _label_frame(row: dict, frame: int):
    radius, height = bench.canonical_axes(row)
    flux = corpus_flux_to_nova_total(
        np.asarray(row["efit_psirz"][frame], dtype=float)
    ).T
    count = int(row["efit_lcfs_n"][frame])
    boundary = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    axis = np.asarray(
        [row["efit_r_axis"][frame], row["efit_z_axis"][frame]], dtype=float
    )
    return radius, height, flux, boundary, axis


def _normalised_levels(axis_value: float, edge_value: float) -> np.ndarray:
    if not np.isfinite(axis_value) or not np.isfinite(edge_value):
        return np.empty(0)
    return np.linspace(axis_value, edge_value, LEVEL_COUNT + 2)[1:-1]


def _shared_levels(pairs: list[tuple[float, float]]) -> np.ndarray:
    """Return one level array spanning every supplied axis-to-boundary range.

    Both fields are drawn at these same physical levels, which is what makes a
    mismatch between them visible.  Taking the union rather than the solved
    field's own range is what keeps each field's closed surfaces on the page:
    spaced over one field's range alone, the other's contours collapse onto a
    few levels or leave the array entirely.
    """

    finite = [
        (float(axis_value), float(edge_value))
        for axis_value, edge_value in pairs
        if np.isfinite(axis_value) and np.isfinite(edge_value)
    ]
    if not finite:
        return np.empty(0)
    low = min(min(pair) for pair in finite)
    high = max(max(pair) for pair in finite)
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        return np.empty(0)
    return np.linspace(low, high, LEVEL_COUNT + 2)[1:-1]


def _null_handle(marker: str, filled: bool, color: str, label: str):
    return plt.Line2D(
        [],
        [],
        marker=marker,
        linestyle="none",
        color=color,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markersize=6.5,
        markeredgewidth=DEFAULT_INK.xpoint_markeredgewidth,
        label=label,
    )


def render_frame(
    frame: int,
    record: dict,
    row: dict,
    wall: np.ndarray,
    output: Path,
) -> dict:
    figure, axes = plt.subplots(figsize=(14.0, 7.2))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.13)
    poloidal_axes(axes)
    branch = record["banked_branches"]["diverted"]
    banked = record.get("banked_read") or {}
    operand = _display_operand(frame)
    drawn: dict[str, object] = {"frame": frame, "solved_field": False}
    label_radius, label_height, label_flux, label_boundary, label_axis = _label_frame(
        row, frame
    )
    label_flux = _oriented(label_flux, label_radius, label_height)
    gauge = float(record.get("additive_gauge_wb") or 0.0)
    label_x = np.asarray(banked.get("reference_x_points_rz_m") or [], dtype=float)
    label_axis_value = _bilinear(label_radius, label_height, label_flux, label_axis)
    label_edge_value = _edge_flux(
        label_radius, label_height, label_flux, label_boundary
    )
    ranges = [(label_axis_value - gauge, label_edge_value - gauge)]
    handles = []
    solved_axis = solved_x = None
    saddle_flux = None
    if operand is not None:
        with np.load(operand) as data:
            radius = np.asarray(data["display_radius"], dtype=float)
            height = np.asarray(data["display_height"], dtype=float)
            flux = _oriented(np.asarray(data["flux"], dtype=float), radius, height)
            saddle_flux = float(data["saddle_flux"])
        solved_axis = np.asarray(banked["nova_axis_rz_m"], dtype=float).reshape(-1)[:2]
        solved_x = np.asarray(banked["nova_x_point_rz_m"], dtype=float).reshape(-1, 2)
        axis_value = _bilinear(radius, height, flux, solved_axis)
        edge_value = _edge_flux(radius, height, flux, label_boundary)
        ranges.append((axis_value, edge_value))
        levels = _shared_levels(ranges)
        draw_flux_contours(
            axes,
            radius,
            height,
            flux,
            levels,
            color=DEFAULT_INK.flux_color,
            wall=wall,
        )
        draw_flux_contours(
            axes,
            radius,
            height,
            flux,
            [saddle_flux],
            color=DEFAULT_INK.separatrix_color,
            linewidth=DEFAULT_INK.separatrix_linewidth,
            wall=wall,
        )
        inside = np.asarray(
            inside_wall_units(np.asarray(solved_x, dtype=float), wall), dtype=bool
        )
        solved_x_inside = bool(np.all(inside))
        others = [solved_x[~inside]] if np.any(~inside) else []
        if label_x.size:
            others.append(label_x)
        draw_nulls(
            axes,
            magnetic_axis=solved_axis,
            x_points=np.asarray(solved_x, dtype=float)[inside],
            contain=wall,
            other_x_points=(np.concatenate(others) if others else None),
        )
        handles.append(
            plt.Line2D(
                [], [], color=DEFAULT_INK.flux_color, lw=1.6, label="Nova solved flux"
            )
        )
        handles.append(
            plt.Line2D(
                [],
                [],
                color=DEFAULT_INK.separatrix_color,
                lw=1.6,
                label="solved separatrix",
            )
        )
        handles.append(
            _null_handle(
                DEFAULT_INK.axis_marker, True, DEFAULT_INK.axis_color, "solved axis"
            )
        )
        handles.append(
            _null_handle(
                DEFAULT_INK.xpoint_marker,
                True,
                DEFAULT_INK.xpoint_color,
                "solved saddle",
            )
        )
        axis_inside = bool(
            inside_wall_units(np.asarray(solved_axis, dtype=float)[None, :], wall)[0]
        )
        if not axis_inside:
            axes.plot(
                solved_axis[0],
                solved_axis[1],
                marker=DEFAULT_INK.axis_marker,
                markerfacecolor="none",
                markeredgecolor=DEFAULT_INK.axis_color,
                markeredgewidth=DEFAULT_INK.xpoint_markeredgewidth,
                markersize=DEFAULT_INK.axis_markersize,
                linestyle="none",
                zorder=7,
            )
            handles.append(
                _null_handle(
                    DEFAULT_INK.axis_marker,
                    False,
                    DEFAULT_INK.axis_color,
                    "solved axis (outside wall)",
                )
            )
        drawn.update(
            {
                "solved_field": True,
                "levels": [float(value) for value in levels],
                "separatrix_flux_wb": saddle_flux,
                "solved_axis_rz_m": solved_axis.tolist(),
                "solved_x_point_rz_m": solved_x.tolist(),
                "solved_x_point_inside_wall": solved_x_inside,
                "solved_axis_inside_wall": axis_inside,
            }
        )
    else:
        levels = _shared_levels(ranges)
    draw_flux_contours(
        axes,
        label_radius,
        label_height,
        label_flux,
        np.asarray(levels, dtype=float) + gauge,
        color=DEFAULT_INK.contour_color,
        linewidth=1.0,
        wall=wall,
    )
    handles.insert(
        0 if not drawn["solved_field"] else 1,
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.contour_color,
            lw=1.2,
            label="EFIT label flux (gauge-aligned)",
        ),
    )
    axes.plot(
        label_axis[0],
        label_axis[1],
        marker=DEFAULT_INK.axis_marker,
        markerfacecolor="none",
        markeredgecolor=DEFAULT_INK.contour_color,
        markeredgewidth=DEFAULT_INK.xpoint_markeredgewidth,
        markersize=DEFAULT_INK.axis_markersize,
        linestyle="none",
        zorder=6,
    )
    handles.append(
        _null_handle(
            DEFAULT_INK.axis_marker, False, DEFAULT_INK.contour_color, "label axis"
        )
    )
    if label_x.size:
        handles.append(
            _null_handle(
                DEFAULT_INK.xpoint_marker,
                False,
                DEFAULT_INK.contour_color,
                "label saddle",
            )
        )
    draw_wall(axes, wall[:, 0], wall[:, 1], close=True)
    handles.append(
        plt.Line2D([], [], color=DEFAULT_INK.wall_color, lw=1.4, label="first wall")
    )
    axes.set_xlim(float(wall[:, 0].min()) - 0.15, float(wall[:, 0].max()) + 0.15)
    axes.set_ylim(float(wall[:, 1].min()) - 0.15, float(wall[:, 1].max()) + 0.15)

    terminal_residual = branch.get("residual")
    converged = record.get("converged")
    parts = [f"frame {frame}", str(record["shot"])]
    if terminal_residual is not None:
        parts.append(f"diverted terminal residual {terminal_residual:.3e}")
        parts.append(f"converged {str(converged).lower()}")
    else:
        parts.append("refused: no qualified magnetic axis, no solved field")
    achieved = branch.get("achieved_class")
    consistent = branch.get("topology_consistent")
    if achieved:
        parts.append(
            f"kept terminal topology {achieved} "
            f"(consistent with its diverted seed: {str(consistent).lower()})"
        )
    if drawn["solved_field"]:
        parts.append(
            "solved saddle "
            + (
                "inside the wall"
                if drawn["solved_x_point_inside_wall"]
                else "OUTSIDE the wall (a real solved null, drawn hollow)"
            )
        )
        if not drawn["solved_axis_inside_wall"]:
            parts.append("solved axis OUTSIDE the wall")
    parts.append("tolerance 1e-5")
    caption = textwrap.fill("  |  ".join(parts), width=150)
    figure.text(
        0.5, 0.02, caption, ha="center", va="bottom", fontsize=11, color="#333333"
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    figure.savefig(output.with_suffix(".png"), format="png", dpi=100)
    plt.close(figure)
    drawn["svg"] = str(output)
    drawn["terminal_residual"] = terminal_residual
    drawn["converged"] = converged
    drawn["legend_entries"] = [handle.get_label() for handle in handles]
    return drawn


def _artifact_grid(archive: Path) -> dict[str, np.ndarray]:
    """Return the measured terminal field and its lattice from one saved arm."""

    with np.load(archive, allow_pickle=True) as data:
        keys = [key for key in data.files if key.endswith("_flux_grid")]
        if len(keys) != 1:
            raise ValueError(f"{archive} carries {len(keys)} flux grids, expected one")
        grid = np.asarray(data[keys[0]], dtype=float)
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        return {
            "flux": _oriented(grid, radius, height),
            "radius": radius,
            "height": height,
            "label": _oriented(np.asarray(data["label"], dtype=float), radius, height),
            "wall": np.asarray(data["wall"], dtype=float),
            "limiter": np.asarray(data["limiter"], dtype=float),
            "label_boundary": np.asarray(data["label_boundary"], dtype=float),
            "key": np.asarray(keys[0]),
        }


def _lattice_extremum(
    radius: np.ndarray, height: np.ndarray, flux: np.ndarray, wall: np.ndarray
) -> np.ndarray:
    """Return the lattice point carrying the field's own extremum inside the wall."""

    inside = np.asarray(
        inside_wall_units(
            np.c_[
                np.meshgrid(radius, height)[0].ravel(),
                np.meshgrid(radius, height)[1].ravel(),
            ],
            wall,
        ),
        dtype=bool,
    ).reshape(flux.shape)
    masked = np.where(inside, flux, np.nan)
    peak = (
        np.nanargmax(masked)
        if abs(np.nanmax(masked)) >= abs(np.nanmin(masked))
        else np.nanargmax(-masked)
    )
    row, column = np.unravel_index(peak, flux.shape)
    return np.asarray([radius[column], height[row]], dtype=float)


def _check_text_artists(figure, output: Path) -> list[str]:
    """Return the text artists that clip the canvas or overlap another.

    A figure whose labels collide or run off the canvas is not delivered, so the
    renderer refuses to write one and names the offending artists.
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    boxes = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        boxes.append((text.get_text()[:40], box))
    problems = []
    for label, box in boxes:
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            problems.append(
                f"clipped: {label!r} "
                f"x=[{box.x0:.0f},{box.x1:.0f}] y=[{box.y0:.0f},{box.y1:.0f}]"
            )
    for index, (first_label, first) in enumerate(boxes):
        for second_label, second in boxes[index + 1 :]:
            if first.overlaps(second):
                problems.append(f"overlap: {first_label!r} with {second_label!r}")
    if problems:
        print(f"REFUSED {output.name}: " + "; ".join(problems), flush=True)
    return problems


def _refined_field(radius, height, flux, padding=0.02):
    """Interpolate the solved map onto a fine grid, once per panel."""

    spline = RegularGridInterpolator(
        (radius, height), flux, method="cubic", bounds_error=False, fill_value=np.nan
    )
    dense_r = np.linspace(radius[0] - padding, radius[-1] + padding, 601)
    dense_z = np.linspace(height[0] - padding, height[-1] + padding, 601)
    grid_r, grid_z = np.meshgrid(dense_r, dense_z, indexing="ij")
    dense = spline((grid_r, grid_z))
    return dense_r, dense_z, dense


def _boundary_curve(dense_r, dense_z, dense, level, saddle, window=0.22):
    """Return the level set, locally refined around the saddle where admitted.

    A 33 x 33 lattice carries a level set through a saddle as one smooth curve:
    at that resolution the four legs merge.  Contouring the same interpolant on a
    dense grid restores the crossing, and the window keeps the cost local to the
    region a reader looks at when judging an X-point.
    """

    fine_r = np.linspace(
        max(saddle[0] - window, dense_r[0]), min(saddle[0] + window, dense_r[-1]), 241
    )
    fine_z = np.linspace(
        max(saddle[1] - window, dense_z[0]), min(saddle[1] + window, dense_z[-1]), 241
    )
    grid_r, grid_z = np.meshgrid(fine_r, fine_z, indexing="ij")
    spline = RegularGridInterpolator(
        (dense_r, dense_z), dense, method="linear", bounds_error=False, fill_value=np.nan
    )
    local = spline((grid_r, grid_z))
    return fine_r, fine_z, local


def _draw_boundary(axes, dense_r, dense_z, dense, level, saddle, colour, width, refine):
    """Draw the bounding surface, refined at the saddle when it passes through it.

    Only a separatrix-bounded terminal has its bounding level at the saddle, so
    only that case is contoured on the local window.  A limiter-bounded level
    lies outside that window and would come back empty.
    """

    handles = []
    if refine and saddle.size == 2 and np.isfinite(level):
        fine_r, fine_z, local = _boundary_curve(dense_r, dense_z, dense, level, saddle)
        used = axes.contour(
            fine_r,
            fine_z,
            local,
            levels=[level],
            colors=[colour],
            linewidths=[width],
            linestyles="solid",
        )
    else:
        used = axes.contour(
            dense_r,
            dense_z,
            dense,
            levels=[level],
            colors=[colour],
            linewidths=[width],
            linestyles="solid",
        )
    handles.append(used)
    return handles


def _caption_lines(parts, width=110):
    """Wrap the caption on its field separators, never inside one."""

    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}  |  {part}"
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def render_measured_frame(
    frame: int,
    receipt_frame: dict,
    artifact: Path,
    output: Path,
    boundary: dict | None = None,
    branch: str = "diverted",
) -> dict:
    """Draw one frame's MEASURED terminal state as line contours.

    The solved field, the label field, the lattice, the limiter and the bounding
    flux all come from the arm's own archive and its measured analysis, so the
    panel is the state the solve reached rather than a re-derivation of it.
    """

    boundary = boundary or {}
    grid = _artifact_grid(artifact)
    radius, height = grid["radius"], grid["height"]
    wall = grid["wall"]
    limiter = grid["limiter"]
    flux, label = grid["flux"], grid["label"]
    measured = receipt_frame["measured"]["branches"][branch]["arms"]["default"]
    terminal_residual = measured.get("terminal_residual")
    converged = bool(receipt_frame.get("converged"))
    presence = measured.get("presence") or {}
    solved_axis = np.asarray(presence.get("axis_rz_m") or [], dtype=float)
    solved_x = np.asarray(measured.get("x_point_rz_m") or [], dtype=float).reshape(-1, 2)
    labels = receipt_frame.get("measured") or {}
    label_axis = np.asarray(labels.get("label_axis_rz_m") or [], dtype=float)
    label_x = np.asarray(labels.get("label_saddle_rz_m") or [], dtype=float).reshape(-1)
    if label_axis.size != 2:
        label_axis = _lattice_extremum(radius, height, label, wall)

    bounding_flux = boundary.get("bounding_flux")
    bounds = boundary.get("bounds", "unknown")
    contact = np.asarray(boundary.get("contact_rz_m") or [], dtype=float)

    figure, (axes, trace) = plt.subplots(
        1, 2, figsize=(14.0, 7.2), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    figure.subplots_adjust(left=0.04, right=0.97, top=0.80, bottom=0.24, wspace=0.14)
    poloidal_axes(axes)

    solved_axis_value = _bilinear(radius, height, flux, solved_axis)
    solved_edge_value = _edge_flux(radius, height, flux, grid["label_boundary"])
    label_axis_value = _bilinear(radius, height, label, label_axis)
    label_edge_value = _edge_flux(radius, height, label, grid["label_boundary"])
    levels = _shared_levels(
        [(solved_axis_value, solved_edge_value), (label_axis_value, label_edge_value)]
    )

    # The reference is drawn first and thin; the subject, the solved field, is
    # drawn over it and heavier, because the solved state is the evidence.
    draw_flux_contours(
        axes,
        radius,
        height,
        label,
        levels,
        color=DEFAULT_INK.contour_color,
        linewidth=0.7,
        wall=wall,
    )
    draw_flux_contours(
        axes,
        radius,
        height,
        flux,
        levels,
        color=DEFAULT_INK.flux_color,
        linewidth=1.0,
        wall=wall,
    )

    dense_r, dense_z, dense = _refined_field(radius, height, flux)
    saddle_flux = (
        _bilinear(radius, height, flux, solved_x[0]) if solved_x.size else float("nan")
    )
    if np.isfinite(bounding_flux):
        _draw_boundary(
            axes,
            dense_r,
            dense_z,
            dense,
            float(bounding_flux),
            solved_x[0] if solved_x.size else np.zeros(0),
            DEFAULT_INK.separatrix_color,
            2.0,
            bounds == "separatrix",
        )
    if bounds == "limiter" and np.isfinite(saddle_flux):
        # The saddle level set is not the boundary here; it is drawn thin and
        # dashed so a reader can see the surface that would have bounded the
        # plasma had the limiter not reached it first.
        axes.contour(
            dense_r,
            dense_z,
            dense,
            levels=[saddle_flux],
            colors=[DEFAULT_INK.contour_color],
            linewidths=[0.9],
            linestyles=["dashed"],
        )

    if np.asarray(grid["label_boundary"]).size:
        boundary_points = np.asarray(grid["label_boundary"], dtype=float)
        axes.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            color=DEFAULT_INK.contour_color,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=3,
        )

    axis_inside = (
        bool(inside_wall_units(solved_axis[None, :], limiter)[0])
        if solved_axis.size
        else False
    )
    inside = (
        np.asarray(inside_wall_units(solved_x, limiter), dtype=bool)
        if solved_x.size
        else np.zeros(0, dtype=bool)
    )
    others = [solved_x[~inside]] if solved_x.size and np.any(~inside) else []
    draw_nulls(
        axes,
        magnetic_axis=solved_axis if solved_axis.size else None,
        x_points=solved_x[inside] if solved_x.size else None,
        contain=limiter,
        other_x_points=np.concatenate(others) if others else None,
    )
    if solved_x.size:
        axes.plot(
            solved_x[inside, 0] if np.any(inside) else [],
            solved_x[inside, 1] if np.any(inside) else [],
            marker=DEFAULT_INK.xpoint_marker,
            markerfacecolor=DEFAULT_INK.xpoint_color,
            markeredgecolor="white",
            markeredgewidth=1.4,
            markersize=DEFAULT_INK.xpoint_markersize,
            linestyle="none",
            zorder=8,
        )
    if label_axis.size == 2:
        axes.plot(
            label_axis[0],
            label_axis[1],
            marker=DEFAULT_INK.axis_marker,
            markerfacecolor="none",
            markeredgecolor=DEFAULT_INK.contour_color,
            markeredgewidth=1.6,
            markersize=DEFAULT_INK.axis_markersize + 3.0,
            linestyle="none",
            zorder=7,
        )
    if label_x.size == 2 and np.all(np.isfinite(label_x)):
        axes.plot(
            label_x[0],
            label_x[1],
            marker=DEFAULT_INK.xpoint_marker,
            markerfacecolor="none",
            markeredgecolor=DEFAULT_INK.contour_color,
            markeredgewidth=1.4,
            markersize=DEFAULT_INK.xpoint_markersize + 3.0,
            linestyle="none",
            zorder=7,
        )

    draw_wall(axes, limiter[:, 0], limiter[:, 1], close=True)
    if contact.size == 2 and np.all(np.isfinite(contact)):
        axes.plot(
            contact[0],
            contact[1],
            marker=DEFAULT_INK.strike_marker,
            markerfacecolor=DEFAULT_INK.strike_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=DEFAULT_INK.strike_markersize + 2.0,
            linestyle="none",
            zorder=9,
        )
    axes.plot(
        np.append(wall[:, 0], wall[0, 0]),
        np.append(wall[:, 1], wall[0, 1]),
        color="#929292",
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        zorder=2,
    )
    extent = np.vstack([limiter, wall])
    axes.set_xlim(float(extent[:, 0].min()) - 0.05, float(extent[:, 0].max()) + 0.05)
    axes.set_ylim(float(extent[:, 1].min()) - 0.05, float(extent[:, 1].max()) + 0.05)

    # The band the map leaves empty carries the trajectory the brief asks for.
    residual_history = np.asarray(measured.get("residual_history") or [], dtype=float)
    trips = np.arange(residual_history.size)
    trace.semilogy(
        trips,
        residual_history,
        color=DEFAULT_INK.flux_color,
        linewidth=1.6,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    trace.axhline(1e-5, color="#929292", linewidth=1.2, linestyle=(0, (1, 2)))
    trace.annotate(
        "tolerance 1e-5",
        xy=(0.02, 1e-5),
        xytext=(0.02, 1e-4),
        fontsize=12,
        color="#555555",
        va="bottom",
    )
    mask_history = np.asarray(measured.get("mask_history") or [], dtype=float)
    for trip, difference in enumerate(mask_history):
        if trip < residual_history.size and np.isfinite(residual_history[trip]):
            trace.annotate(
                f"{int(difference)}",
                (trip, residual_history[trip]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=11,
                color="#555555",
            )
    trace.set_xlabel("active-set trip", fontsize=16)
    trace.set_ylabel("relative residual", fontsize=16)
    trace.set_xticks(trips)
    span = max(int(trips[-1]) if trips.size else 0, 1)
    trace.set_xlim(-0.5, span + 0.5)
    trace.tick_params(labelsize=14)
    for side in ("top", "right"):
        trace.spines[side].set_visible(False)
    trace.text(
        0.98,
        0.95,
        "mask difference above each point",
        transform=trace.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="#555555",
    )

    cells = int(presence.get("closed_core_cells") or 0)
    ratio = presence.get("confined_to_target_ratio")
    parts = [
        f"frame {frame}",
        f"{branch} terminal residual {terminal_residual:.3e}",
        f"converged {str(converged).lower()}",
        f"axis R={solved_axis[0]:.3f} Z={solved_axis[1]:+.3f} m",
        f"axis {'inside' if axis_inside else 'OUTSIDE'} the limiter",
        f"bounded by the {bounds}",
        f"{cells} closed core cells",
        f"confined current {ratio:.3f}x target" if ratio is not None else "no current read",
    ]
    figure.text(
        0.5,
        0.035,
        _caption_lines(parts),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )

    handles = [
        plt.Line2D(
            [], [], color=DEFAULT_INK.flux_color, lw=1.0, label="Nova solved flux"
        ),
        plt.Line2D(
            [], [], color=DEFAULT_INK.contour_color, lw=0.7, label="EFIT label flux"
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.separatrix_color,
            lw=2.0,
            label=f"bounding {'limiter surface' if bounds == 'limiter' else 'separatrix'}",
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.contour_color,
            lw=0.8,
            ls=(0, (4, 3)),
            label="EFIT label boundary",
        ),
        _null_handle(
            DEFAULT_INK.axis_marker, True, DEFAULT_INK.axis_color, "solved axis"
        ),
        _null_handle(
            DEFAULT_INK.xpoint_marker,
            True,
            DEFAULT_INK.xpoint_color,
            "solved X-point",
        ),
        _null_handle(
            DEFAULT_INK.axis_marker, False, DEFAULT_INK.contour_color, "label axis"
        ),
        _null_handle(
            DEFAULT_INK.xpoint_marker, False, DEFAULT_INK.contour_color, "label X-point"
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.strike_color,
            marker=DEFAULT_INK.strike_marker,
            markersize=DEFAULT_INK.strike_markersize + 2.0,
            lw=0,
            label="limiter contact",
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.wall_color,
            lw=1.0,
            label="DIII-D limiter (wall)",
        ),
        plt.Line2D(
            [], [], color="#929292", lw=1.0, ls=(0, (2, 3)), label="grid domain bound"
        ),
    ]
    if bounds == "limiter":
        handles.append(
            plt.Line2D(
                [],
                [],
                color=DEFAULT_INK.contour_color,
                lw=0.9,
                ls="dashed",
                label="saddle level (not bounding)",
            )
        )
    if solved_x.size and not np.all(inside):
        handles.append(
            _null_handle(
                DEFAULT_INK.axis_marker,
                False,
                DEFAULT_INK.axis_color,
                "solved axis (outside wall)",
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
        frameon=False,
        fontsize=10.5,
    )
    problems = _check_text_artists(figure, output)
    if problems:
        plt.close(figure)
        return {"frame": frame, "refused": problems}
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    figure.savefig(output.with_suffix(".png"), format="png", dpi=100)
    plt.close(figure)
    return {
        "frame": frame,
        "svg": str(output),
        "png": str(output.with_suffix(".png")),
        "artifact": str(artifact),
        "terminal_residual": terminal_residual,
        "converged": converged,
        "solved_axis_rz_m": solved_axis.tolist(),
        "solved_axis_inside_limiter": axis_inside,
        "closed_core_cells": cells,
        "confined_to_current_ratio": ratio,
        "bounds": bounds,
        "bounding_flux": bounding_flux,
        "psi_saddle": boundary.get("psi_saddle"),
        "psi_limiter": boundary.get("psi_limiter"),
        "achieved_class": boundary.get("achieved_class"),
        "contact_rz_m": boundary.get("contact_rz_m"),
        "flux_grid_key": str(grid["key"]),
        "levels": [float(value) for value in levels],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[179, 44, 144, 146, 137]
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("docs/figures/diiid-forward-convergence/convergence-receipt.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        type=Path,
        default=None,
        help="measured boundary competition per frame and branch",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "read each frame's measured terminal field from "
            "frame-<n>-diverted-default.npz in this directory instead of "
            "re-deriving it"
        ),
    )
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text())
    if arguments.artifact_dir is not None:
        rendered = []
        analysis = {}
        if arguments.boundary is not None and arguments.boundary.exists():
            analysis = json.loads(arguments.boundary.read_text())["frames"]
        for frame in arguments.frames:
            archive = arguments.artifact_dir / f"frame-{frame}-diverted-default.npz"
            output = arguments.output_dir / f"frame-{frame}-flux-contours.svg"
            measured = (analysis.get(str(frame)) or {}).get("diverted") or {}
            rendered.append(
                render_measured_frame(
                    frame,
                    receipt["frames"][str(frame)],
                    archive,
                    output,
                    measured.get("default"),
                )
            )
            print(json.dumps(rendered[-1], sort_keys=True, default=float), flush=True)
            # A terminal the solver records as diverted can be bounded by its
            # limiter instead.  Where the bounding surface is the separatrix, the
            # X-point legs are the state a reader has to judge, so that arm is
            # drawn beside the frame's diverted terminal.
            for branch, arms in (analysis.get(str(frame)) or {}).items():
                entry = (arms or {}).get("default") or {}
                if branch == "diverted" or entry.get("bounds") != "separatrix":
                    continue
                extra = arguments.artifact_dir / f"frame-{frame}-{branch}-default.npz"
                if not extra.exists():
                    continue
                extra_output = (
                    arguments.output_dir / f"frame-0-flux-contours.svg"
                ).with_name(f"frame-{frame}-{branch}-flux-contours.svg")
                rendered.append(
                    render_measured_frame(
                        frame,
                        receipt["frames"][str(frame)],
                        extra,
                        extra_output,
                        entry,
                        branch,
                    )
                )
                print(
                    json.dumps(rendered[-1], sort_keys=True, default=float), flush=True
                )
        refused = [item["frame"] for item in rendered if "refused" in item]
        arguments.record.write_text(
            json.dumps({"panels": rendered}, indent=2, default=float) + "\n"
        )
        if refused:
            raise SystemExit(f"panels refused by the text check: {refused}")
        return
    partial = json.loads(
        Path(
            "docs/figures/diiid-forward-onboarding/forward-gs-2026-11/"
            "forward_gs_partial_receipt.json"
        ).read_text()
    )
    partial_by_frame = {
        int(record["frame"]): record for record in partial["frame_records"]
    }
    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    wall, _statement = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE, bench.DEFAULT_MACHINE_ARTIFACT_DIGEST
    )
    rendered = []
    for frame in arguments.frames:
        item = [entry for entry in selected if entry.frame == frame][0]
        row = bench._read(
            item.path,
            bench._LABEL_COLUMNS
            + bench._GEOMETRY_COLUMNS
            + bench._CURRENT_COLUMNS
            + bench._PLASMA_CURRENT_COLUMNS,
        )
        record = dict(receipt["frames"][str(frame)])
        source = partial_by_frame[frame]
        record.setdefault("banked_read", source.get("banked_read"))
        record["additive_gauge_wb"] = (source.get("metrics") or {}).get(
            "additive_gauge_wb"
        )
        output = arguments.output_dir / f"frame-{frame}-flux-contours.svg"
        rendered.append(render_frame(frame, record, row, wall, output))
        print(json.dumps(rendered[-1], sort_keys=True), flush=True)
    arguments.record.write_text(
        json.dumps({"panels": rendered}, indent=2, default=float) + "\n"
    )


if __name__ == "__main__":
    main()
