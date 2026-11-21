"""Render the measured coupling differences and paired saved flux fields."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import numpy as np

from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    inside_wall_units,
)
from scripts.prototypes.diiid_single_kernel_coupling.batch import ROWS
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)

HIGHEST_DIFFERENCE_COLOR = "#b8327e"


def load(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def panel(figure, left, width, wall):
    axes = figure.add_axes((left, 0.14, width, 0.71))
    draw_wall(axes, radius=wall[:, 0], height=wall[:, 1], color=WALL, linewidth=1.5)
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_xlim(wall[:, 0].min() - 0.06, wall[:, 0].max() + 0.06)
    axes.set_ylim(wall[:, 1].min() - 0.06, wall[:, 1].max() + 0.06)
    return axes


def difference_panels(args):
    from benchmarks import diiid_forward_gs_match as bench

    folder = args.input / "base/frame-137-diverted"
    data = load(folder / "coupling-differences.npz")
    records = json.loads((folder / "coupling-differences.json").read_text())
    wall, _ = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE, bench.DEFAULT_MACHINE_ARTIFACT_DIGEST
    )
    assert wall.ndim == 2 and wall.shape[1] == 2
    figure = plt.figure(figsize=CANVAS, dpi=100)
    levels = (0.04, 0.05, 0.06, 0.08, 0.12)
    styles = ("solid", (0, (6, 2)), (0, (2, 2)), "dotted", "solid")
    widths = (2.6, 2.6, 2.6, 2.6, 3.0)
    colors = (SOLVED,) * (len(levels) - 1) + (HIGHEST_DIFFERENCE_COLOR,)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                linewidth=width,
                linestyle=style,
                label=f"{level:.2f}%",
            )
            for level, style, width, color in zip(
                levels, styles, widths, colors, strict=True
            )
        ],
        loc="upper center",
        ncol=5,
        frameon=False,
        handlelength=4.4,
    )
    axes, notes, levels_record = [], [], {}
    for index, name in enumerate(("grid", "wall", "sample")):
        left = 0.04 + index * 0.32
        ax = panel(figure, left, 0.28, wall)
        ax.set_xlim(data["radius"][0] - 0.03, data["radius"][-1] + 0.03)
        ax.set_ylim(data["height"][0] - 0.03, data["height"][-1] + 0.03)
        axes.append(ax)
        values = 100.0 * data[name]
        contours = ax.contour(
            data["radius"],
            data["height"],
            values.T,
            levels=levels,
            colors=colors,
            linewidths=widths,
            linestyles=styles,
        )
        figure.canvas.draw()
        for segment in contours.allsegs[-1]:
            if not len(segment):
                continue
            point = segment[np.argmin(segment[:, 0])]
            label_x = ax.get_position().x0 - 0.014
            label_y = figure.transFigure.inverted().transform(
                ax.transData.transform(point)
            )[1]
            notes.append(
                figure.text(
                    label_x,
                    label_y,
                    f"{levels[-1]:.2f}%",
                    color=HIGHEST_DIFFERENCE_COLOR,
                    ha="right",
                    va="center",
                )
            )
            figure.add_artist(
                ConnectionPatch(
                    xyA=(label_x + 0.006, label_y),
                    coordsA=figure.transFigure,
                    xyB=point,
                    coordsB=ax.transData,
                    color=HIGHEST_DIFFERENCE_COLOR,
                    linewidth=1.2,
                )
            )
        levels_record[name] = list(levels)
        notes.append(figure.text(left + 0.14, 0.89, f"{name} targets", ha="center"))
        metric = records["targets"][name]
        notes.append(
            figure.text(
                left + 0.14,
                0.07,
                f"median {100 * metric['relative_median']:.3f}% · "
                f"maximum {100 * metric['relative_max']:.3f}%",
                ha="center",
                fontsize=12,
            )
        )
    save(
        figure,
        axes,
        notes,
        args.output / "uniform-response-difference.svg",
        args.previews / "uniform-response-difference.png",
    )
    plt.close(figure)
    return levels_record


def flux_panels(args, frame, branch):
    case = f"frame-{frame}-{branch}"
    folders = {arm: args.input / arm / case for arm in args.arms}
    data = {arm: load(path / "terminal.npz") for arm, path in folders.items()}
    receipts = {
        arm: json.loads((path / "receipt.json").read_text())
        for arm, path in folders.items()
    }
    solves = {
        arm: json.loads((path / "solve-receipt.json").read_text())
        for arm, path in folders.items()
    }
    reference = data[args.arms[0]]
    radius, height, wall = (reference[key] for key in ("radius", "height", "limiter"))
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    inside = inside_wall_units(np.c_[rr.ravel(), zz.ravel()], wall).reshape(rr.shape)
    label = reference["label"]
    levels = np.linspace(float(label[inside].min()), float(label[inside].max()), 16)[
        1:-1
    ]
    figure = plt.figure(figsize=CANVAS, dpi=100)
    figure.legend(
        handles=[
            Line2D([], [], color=SOLVED, linewidth=3, label="solved flux and nulls"),
            Line2D(
                [],
                [],
                color=REFERENCE,
                linestyle="--",
                linewidth=2.6,
                label="EFIT flux and nulls",
            ),
            Line2D([], [], color=WALL, linewidth=1.5, label="physical wall"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    axes, notes, gauges = [], [], {}
    for index, arm in enumerate(args.arms):
        left = 0.35 if len(args.arms) == 1 else 0.12 + index * 0.46
        current = data[arm]
        np.testing.assert_array_equal(current["label"], label)
        ax = panel(figure, left, 0.30, wall)
        axes.append(ax)
        gauge = float(np.median(label[inside] - current["flux"][inside]))
        gauges[arm] = gauge
        contours = draw_flux_contours(
            ax,
            radius,
            height,
            label.T,
            levels,
            color=REFERENCE,
            linewidth=2.6,
            wall=wall,
            zorder=2,
        )
        contours.set_linestyle("--")
        draw_flux_contours(
            ax,
            radius,
            height,
            (current["flux"] + gauge).T,
            levels,
            color=SOLVED,
            linewidth=3.0,
            wall=wall,
            zorder=3,
        )
        topology = receipts[arm]["terminal_topology"]
        lower = topology["lower_saddle_rz_m"]
        admitted = topology["boundary_class"] == "diverted"
        solved_style = replace(DEFAULT_INK, axis_color=SOLVED, xpoint_color=SOLVED)
        reference_style = replace(
            DEFAULT_INK,
            axis_color=REFERENCE,
            xpoint_color=REFERENCE,
            axis_markersize=9,
            xpoint_markersize=10,
        )
        draw_nulls(
            ax,
            magnetic_axis=current["reference_axis"],
            x_points=np.asarray(receipts[arm]["efit_x_point_rz_m"])[None],
            style=reference_style,
            contain=wall,
        )
        draw_nulls(
            ax,
            magnetic_axis=topology["axis_rz_m"],
            x_points=np.asarray(topology["selected_saddle_rz_m"])[None]
            if admitted
            else None,
            other_x_points=np.asarray(lower)[None]
            if lower is not None and not admitted
            else None,
            style=solved_style,
            contain=wall,
        )
        notes.append(
            figure.text(
                left + 0.15,
                0.87,
                "hybrid uniform" if arm == "base" else "polygon uniform",
                ha="center",
            )
        )
        solve = solves[arm]
        notes.append(
            figure.text(
                left + 0.15,
                0.075,
                f"residual {solve['residual']:.3e} · converged {solve['converged']}",
                ha="center",
                fontsize=12,
            )
        )
        distance = topology["lower_saddle_distance_m"]
        offset = (
            "unqualified lower saddle"
            if distance is None
            else f"lower X-point offset {100 * distance:.3f} cm"
        )
        notes.append(figure.text(left + 0.15, 0.035, offset, ha="center", fontsize=12))
    save(
        figure, axes, notes, args.output / f"{case}.svg", args.previews / f"{case}.png"
    )
    plt.close(figure)
    return dict(levels_wb=levels.tolist(), additive_gauge_wb=gauges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--differences-only", action="store_true")
    mode.add_argument("--flux-only", action="store_true")
    parser.add_argument(
        "--arms", nargs="+", choices=("base", "head"), default=("base", "head")
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        choices=[f"{frame}-{branch}" for frame, branch in ROWS],
    )
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    metadata_path = args.previews / "render-metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else dict(flux={})
    )
    if not args.flux_only:
        metadata["difference_levels_percent"] = difference_panels(args)
    if not args.differences_only:
        for frame, branch in ROWS:
            name = f"{frame}-{branch}"
            if args.rows is None or name in args.rows:
                metadata["flux"][name] = flux_panels(args, frame, branch)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
