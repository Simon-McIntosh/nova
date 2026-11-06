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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_forward_gs_match as bench
from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

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


def render_frame(
    frame: int,
    record: dict,
    row: dict,
    wall: np.ndarray,
    output: Path,
) -> dict:
    figure, axes = plt.subplots(figsize=(14.0, 7.2))
    poloidal_axes(axes)
    branch = record["branches"]["diverted"]
    banked = record.get("banked_read") or {}
    operand = _display_operand(frame)
    drawn: dict[str, object] = {"frame": frame, "solved_field": False}
    label_radius, label_height, label_flux, label_boundary, label_axis = _label_frame(
        row, frame
    )
    label_flux = _oriented(label_flux, label_radius, label_height)
    label_x = np.asarray(banked.get("reference_x_points_rz_m") or [], dtype=float)
    label_settled = label_x.size > 0

    if operand is not None:
        with np.load(operand) as data:
            radius = np.asarray(data["display_radius"], dtype=float)
            height = np.asarray(data["display_height"], dtype=float)
            flux = _oriented(np.asarray(data["flux"], dtype=float), radius, height)
            saddle_rz = np.asarray(data["saddle_rz"], dtype=float)
            saddle_flux = float(data["saddle_flux"])
        solved_axis = np.asarray(banked["nova_axis_rz_m"], dtype=float)
        solved_x = np.asarray(banked["nova_x_point_rz_m"], dtype=float)
        axis_value = _bilinear(radius, height, flux, solved_axis)
        levels = _normalised_levels(axis_value, saddle_flux)
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
        draw_nulls(
            axes,
            magnetic_axis=solved_axis,
            x_points=solved_x[None, :],
            other_x_points=label_x if label_settled else None,
        )
        drawn.update(
            {
                "solved_field": True,
                "levels": [float(value) for value in levels],
                "separatrix_flux_wb": saddle_flux,
                "saddle_rz_m": saddle_rz.tolist(),
            }
        )
    label_axis_value = _bilinear(label_radius, label_height, label_flux, label_axis)
    label_edge_value = _edge_flux(
        label_radius, label_height, label_flux, label_boundary
    )
    gauge = float(record.get("additive_gauge_wb") or 0.0)
    if drawn["solved_field"]:
        label_levels = np.asarray(drawn["levels"], dtype=float) + gauge
    else:
        label_levels = _normalised_levels(label_axis_value, label_edge_value)
    draw_flux_contours(
        axes,
        label_radius,
        label_height,
        label_flux,
        label_levels,
        color=DEFAULT_INK.contour_color,
        linewidth=0.8,
        alpha=0.95,
        wall=wall,
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
    draw_wall(axes, wall[:, 0], wall[:, 1], close=True)
    axes.set_xlim(float(wall[:, 0].min()) - 0.15, float(wall[:, 0].max()) + 0.15)
    axes.set_ylim(float(wall[:, 1].min()) - 0.15, float(wall[:, 1].max()) + 0.15)

    terminal_residual = branch.get("residual")
    converged = record.get("converged")
    if terminal_residual is not None:
        caption = (
            f"frame {frame}  {record['shot']}  |  diverted terminal residual "
            f"{terminal_residual:.3e}  |  converged {str(converged).lower()}  |  "
            f"tolerance 1e-5"
        )
    else:
        caption = (
            f"frame {frame}  {record['shot']}  |  refused: "
            "no qualified magnetic axis, no solved field drawn"
        )
    figure.text(
        0.5, 0.015, caption, ha="center", va="bottom", fontsize=12, color="#333333"
    )
    handles = [
        plt.Line2D(
            [], [], color=DEFAULT_INK.flux_color, lw=1.6, label="Nova solved flux"
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.contour_color,
            lw=1.2,
            label="EFIT label flux (gauge-aligned)",
        ),
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.separatrix_color,
            lw=1.6,
            label="solved separatrix",
        ),
        plt.Line2D([], [], color=DEFAULT_INK.wall_color, lw=1.4, label="first wall"),
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=12
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    figure.savefig(
        output.with_suffix(".png"), format="png", dpi=100, bbox_inches="tight"
    )
    plt.close(figure)
    drawn["svg"] = str(output)
    drawn["terminal_residual"] = terminal_residual
    drawn["converged"] = converged
    return drawn


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
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text())
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
