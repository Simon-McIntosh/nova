"""Compare saved DIII-D geometry before and after the diagnostic translation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.prototypes.diiid_xpoint_offset.decompose import read_case, spline
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall


def draw_critical_points(axes, census, shift, style, minimum_marker, size):
    """Mark all enumerated extrema and saddles, including unconfined roots."""
    for point in census["points"]:
        coordinate = np.asarray(point["rz_m"]) + shift
        before = len(axes.lines)
        if point["kind"] == "maximum":
            draw_nulls(
                axes,
                magnetic_axis=coordinate,
                style=replace(style, axis_markersize=size),
            )
        elif point["kind"] == "saddle":
            draw_nulls(
                axes,
                x_points=coordinate[None],
                style=replace(style, xpoint_markersize=size),
            )
        else:
            axes.plot(
                *coordinate,
                marker=minimum_marker,
                markersize=size,
                linestyle="none",
                color=style.axis_color,
                zorder=style.zorder_markers,
            )
        if not point["in_plasma"]:
            for artist in axes.lines[before:]:
                artist.set_markerfacecolor("none")


def render(entry, row, output, preview):
    """Show the original geometry and the actual residual after alignment."""
    data, _ = read_case(entry)
    radius, height = data["radius"], data["height"]
    field, ref = [spline(radius, height, data[key]) for key in ("flux", "label")]
    translation = np.asarray(row["fit"]["translation_m"])
    ref_axis = np.asarray(row["reference_axis_rz_m"])
    ref_saddle = np.asarray(row["reference_saddle_rz_m"])
    saddle = np.asarray(row["saddle_rz_m"])
    axis_flux, boundary_flux = (
        row["reference_core"]["max_flux_wb"],
        float(ref.ev(*ref_saddle)),
    )
    levels = np.sort(
        boundary_flux
        + (axis_flux - boundary_flux) * np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    )
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    figure, axes = plt.subplots(1, 4, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.2, top=0.74, wspace=0.12)
    notes = []
    solved_style = replace(
        DEFAULT_INK,
        axis_color=SOLVED,
        xpoint_color=SOLVED,
        axis_markersize=9,
        xpoint_markersize=10,
        zorder_markers=10,
    )
    ref_style = replace(
        DEFAULT_INK,
        axis_color=REFERENCE,
        xpoint_color=REFERENCE,
        axis_marker="v",
        xpoint_marker="P",
        axis_markersize=9,
        xpoint_markersize=10,
        zorder_markers=20,
    )
    for index, ax in enumerate(axes):
        poloidal_axes(ax)
        shift = translation if index in (1, 3) else np.zeros(2)
        gauge = row["fit"]["gauge_wb" if index in (1, 3) else "gauge_before_wb"]
        if index == 3:
            both = np.stack((ref_saddle, saddle + shift))
            lo, hi = both.min(axis=0) - 0.16, both.max(axis=0) + 0.16
            # A square viewport preserves the local geometry and its comparison.
            centre = (lo + hi) / 2
            half = max(hi - lo) / 2
            limits = (centre - half, centre + half)
        elif index == 2:
            peak = row["solved_core"]["max_flux_wb"]
            core = [
                p["rz_m"]
                for p in row["solved_core"]["points"]
                if p["in_plasma"]
                and p["flux_wb"] > peak - 0.2 * (peak - data["boundary"][2])
            ]
            bounds = np.vstack((core, [ref_axis]))
            limits = (bounds.min(axis=0) - 0.06, bounds.max(axis=0) + 0.06)
        else:
            limits = (
                np.array([radius[0], height[0]]),
                np.array([radius[-1], height[-1]]),
            )
        lo, hi = limits
        dense_r = np.linspace(lo[0], hi[0], 181)
        dense_z = np.linspace(lo[1], hi[1], 281)
        rr, zz = np.meshgrid(dense_r, dense_z, indexing="ij")
        reference = ref(dense_r, dense_z)
        translated = (
            field.ev((rr - shift[0]).ravel(), (zz - shift[1]).ravel()).reshape(rr.shape)
            - gauge
        )
        valid = (
            (rr - shift[0] >= radius[0])
            & (rr - shift[0] <= radius[-1])
            & (zz - shift[1] >= height[0])
            & (zz - shift[1] <= height[-1])
        )
        translated = np.where(valid, translated, np.nan)
        panel_levels = levels
        if index == 2:
            peak_level = row["solved_core"]["max_flux_wb"] - gauge
            panel_levels = np.unique(
                np.r_[
                    levels,
                    peak_level - np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]),
                ]
            )
        draw_flux_contours(
            ax,
            dense_r,
            dense_z,
            reference.T,
            panel_levels,
            color=REFERENCE,
            linewidth=1.5,
            wall=data["limiter"],
        )
        # Dashed reference lines remain visible under the solid solved field.
        for collection in ax.collections:
            collection.set_linestyle("dashed")
        draw_flux_contours(
            ax,
            dense_r,
            dense_z,
            translated.T,
            panel_levels,
            color=SOLVED,
            linewidth=1.8,
            wall=data["limiter"],
            zorder=3,
        )
        reference_boundary = draw_flux_contours(
            ax,
            dense_r,
            dense_z,
            reference.T,
            [boundary_flux],
            color=REFERENCE,
            linewidth=2.2,
            wall=data["limiter"],
            zorder=4,
        )
        reference_boundary.set_linestyle("dashed")
        solved_boundary = float(data["boundary"][2]) - gauge
        draw_flux_contours(
            ax,
            dense_r,
            dense_z,
            translated.T,
            [solved_boundary],
            color=SOLVED,
            linewidth=2.4,
            wall=data["limiter"],
            zorder=5,
        )
        draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=6)
        size = 7 if index == 2 else 5
        draw_critical_points(ax, row["solved_core"], shift, solved_style, "o", size)
        draw_critical_points(
            ax, row["reference_core"], np.zeros(2), ref_style, "s", size
        )
        draw_nulls(
            ax,
            x_points=(saddle + shift)[None],
            style=replace(solved_style, zorder_markers=30),
        )
        draw_nulls(
            ax, x_points=ref_saddle[None], style=replace(ref_style, zorder_markers=40)
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        box = ax.get_position()
        name = (
            "original",
            "after translation",
            "core critical points",
            "lower X after translation",
        )[index]
        notes.append(
            figure.text((box.x0 + box.x1) / 2, 0.79, name, ha="center", fontsize=16)
        )
    handles = [
        Line2D([], [], color=SOLVED, lw=2.4, label="solved"),
        Line2D([], [], color=REFERENCE, lw=2.2, ls="--", label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="solved maximum"),
        Line2D([], [], color=REFERENCE, marker="v", ls="none", label="EFIT maximum"),
        Line2D([], [], color=SOLVED, marker="X", ls="none", label="solved saddle"),
        Line2D([], [], color=REFERENCE, marker="P", ls="none", label="EFIT saddle"),
        Line2D([], [], color=SOLVED, marker="o", ls="none", label="solved minimum"),
        Line2D([], [], color=REFERENCE, marker="s", ls="none", label="EFIT minimum"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    before = np.linalg.norm(row["xpoint_offset_cm"])
    after = np.linalg.norm(row["xpoint_after_translation_cm"])
    notes.append(
        figure.text(
            0.5,
            0.12,
            f"applied (ΔR, ΔZ) = ({100 * translation[0]:+.2f}, "
            f"{100 * translation[1]:+.2f}) cm; "
            f"lower-X error {before:.2f} → {after:.2f} cm",
            ha="center",
            fontsize=14,
        )
    )
    notes.append(
        figure.text(
            0.5,
            0.065,
            f"gauge-free flux RMS {1000 * row['fit']['rms_before_wb']:.1f} → "
            f"{1000 * row['fit']['rms_after_wb']:.1f} mWb; "
            "shared levels per panel; core adds near-peak levels",
            ha="center",
            fontsize=14,
        )
    )
    notes.append(
        figure.text(
            0.5,
            0.025,
            f"main-plasma maxima: solved {row['solved_core']['maxima_count']}, "
            f"EFIT {row['reference_core']['maxima_count']}; "
            "hollow markers: outside the main plasma",
            ha="center",
            fontsize=14,
        )
    )
    save(figure, axes, notes, output, preview)
    plt.close(figure)
    return dict(
        file=str(output),
        preview=str(preview),
        shared_levels_wb=levels.tolist(),
        reference_xpoint_zorder=40,
        solved_xpoint_zorder=30,
        solved_critical_points=len(row["solved_core"]["points"]),
        reference_critical_points=len(row["reference_core"]["points"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.inputs.read_text())
    rows = json.loads(args.receipt.read_text())["rows"]
    figures = []
    for case in ("frame-137-diverted", "frame-146-diverted"):
        entry = next(e for e in entries if e["arm"] == "head" and e["case"] == case)
        row = next(e for e in rows if e["arm"] == "head" and e["case"] == case)
        figures.append(
            render(
                entry, row, args.output / f"{case}.svg", args.previews / f"{case}.png"
            )
        )
    (args.previews / "figure-checks.json").write_text(
        json.dumps(figures, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
