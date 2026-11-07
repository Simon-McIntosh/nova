"""Render the per-frame poloidal flux-contour panels for the convergence record.

Each panel draws the bank's terminal diverted field as line contours and, on
the same axes and at the same normalised levels, the EFIT label field.  Both
null sets are drawn in the ``nova.media`` vocabulary, the first wall is drawn
from the governed limiter ring, and the caption carries the terminal residual
and the converged flag.

Frame 144 has no solved field: the solve refuses before a topology read.  Its
panel shows the EFIT label contours and nulls with the wall, which is what that
frame's evidence is.
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
            problems.append(f"clipped: {label!r}")
    for index, (first_label, first) in enumerate(boxes):
        for second_label, second in boxes[index + 1 :]:
            if first.overlaps(second):
                problems.append(f"overlap: {first_label!r} with {second_label!r}")
    if problems:
        print(f"REFUSED {output.name}: " + "; ".join(problems), flush=True)
    return problems


def render_measured_frame(
    frame: int,
    receipt_frame: dict,
    artifact: Path,
    output: Path,
) -> dict:
    """Draw one frame's MEASURED terminal state as line contours.

    The solved field, the label field, the lattice and the wall all come from
    the arm's own archive, so the panel is the state the solve reached rather
    than a re-derivation of it.  Nulls are the measured ones from the receipt.
    """

    grid = _artifact_grid(artifact)
    radius, height = grid["radius"], grid["height"]
    wall = grid["wall"]
    flux, label = grid["flux"], grid["label"]
    measured = receipt_frame["measured"]["branches"]["diverted"]["arms"]["default"]
    terminal_residual = measured.get("terminal_residual")
    converged = bool(receipt_frame.get("converged"))
    presence = measured.get("presence") or {}
    solved_axis = np.asarray(presence.get("axis_rz_m") or [], dtype=float)
    solved_x = np.asarray(measured.get("x_point_rz_m") or [], dtype=float).reshape(
        -1, 2
    )
    label_axis = _lattice_extremum(radius, height, label, wall)

    figure, axes = plt.subplots(figsize=(14.0, 7.2))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.13)
    poloidal_axes(axes)

    solved_axis_value = _bilinear(radius, height, flux, solved_axis)
    solved_edge_value = _edge_flux(radius, height, flux, grid["label_boundary"])
    label_axis_value = _bilinear(radius, height, label, label_axis)
    label_edge_value = _edge_flux(radius, height, label, grid["label_boundary"])
    levels = _shared_levels(
        [(solved_axis_value, solved_edge_value), (label_axis_value, label_edge_value)]
    )
    draw_flux_contours(
        axes, radius, height, flux, levels, color=DEFAULT_INK.flux_color, wall=wall
    )
    saddle_flux = (
        _bilinear(radius, height, flux, np.asarray(solved_x, dtype=float)[0])
        if solved_x.size
        else float("nan")
    )
    if np.isfinite(saddle_flux):
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
    draw_flux_contours(
        axes,
        radius,
        height,
        label,
        levels,
        color=DEFAULT_INK.contour_color,
        linewidth=1.0,
        wall=wall,
    )
    axis_inside = (
        bool(inside_wall_units(solved_axis[None, :], wall)[0])
        if solved_axis.size
        else False
    )
    inside = (
        np.asarray(inside_wall_units(solved_x, wall), dtype=bool)
        if solved_x.size
        else np.zeros(0, dtype=bool)
    )
    others = [solved_x[~inside]] if solved_x.size and np.any(~inside) else []
    draw_nulls(
        axes,
        magnetic_axis=solved_axis if solved_axis.size else None,
        x_points=solved_x[inside] if solved_x.size else None,
        contain=wall,
        other_x_points=np.concatenate(others) if others else None,
    )
    draw_wall(axes, wall[:, 0], wall[:, 1], close=True)
    axes.set_xlim(float(wall[:, 0].min()) - 0.15, float(wall[:, 0].max()) + 0.15)
    axes.set_ylim(float(wall[:, 1].min()) - 0.15, float(wall[:, 1].max()) + 0.15)

    handles = [
        plt.Line2D(
            [], [], color=DEFAULT_INK.flux_color, lw=1.6, label="Nova solved flux"
        ),
        plt.Line2D(
            [], [], color=DEFAULT_INK.contour_color, lw=1.2, label="EFIT label flux"
        ),
        _null_handle(
            DEFAULT_INK.axis_marker, True, DEFAULT_INK.axis_color, "solved axis"
        ),
        _null_handle(
            DEFAULT_INK.xpoint_marker, True, DEFAULT_INK.xpoint_color, "solved saddle"
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.separatrix_color,
            lw=1.2,
            label="solved separatrix",
        ),
        _null_handle(
            DEFAULT_INK.axis_marker, False, DEFAULT_INK.contour_color, "label axis"
        ),
        plt.Line2D([], [], color=DEFAULT_INK.wall_color, lw=1.4, label="first wall"),
    ]
    if not axis_inside and solved_axis.size:
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

    cells = int(presence.get("closed_core_cells") or 0)
    ratio = presence.get("confined_to_target_ratio")
    parts = [
        f"frame {frame}",
        f"diverted terminal residual {terminal_residual:.3e}",
        f"converged {str(converged).lower()}",
        f"axis R={solved_axis[0]:.3f} Z={solved_axis[1]:+.3f} m",
        f"{cells} closed core cells",
        f"confined current {ratio:.3f}x target"
        if ratio is not None
        else "no current read",
        "tolerance 1e-5",
    ]
    figure.text(
        0.5,
        0.02,
        textwrap.fill("  |  ".join(parts), width=150),
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
        frameon=False,
        fontsize=11,
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
        "solved_axis_inside_wall": axis_inside,
        "closed_core_cells": cells,
        "confined_to_current_ratio": ratio,
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
        for frame in arguments.frames:
            archive = arguments.artifact_dir / f"frame-{frame}-diverted-default.npz"
            output = arguments.output_dir / f"frame-{frame}-flux-contours.svg"
            rendered.append(
                render_measured_frame(
                    frame, receipt["frames"][str(frame)], archive, output
                )
            )
            print(json.dumps(rendered[-1], sort_keys=True, default=float), flush=True)
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
