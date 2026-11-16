"""Draw the four terminal flux states and the observed limited-row trajectory."""

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

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
LEVELS = np.asarray(
    [-2.0, -1.6, -1.0, -0.6, -0.2, -0.02, 0.0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0]
)


def panel(folder, output, previews):
    data = dict(np.load(folder / "terminal.npz"))
    receipt = json.loads((folder / "receipt.json").read_text())
    census = json.loads((folder / "core-analysis.json").read_text())
    trips = json.loads((folder / "trips.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.15, top=0.79, wspace=0.25)
    solved_null = np.asarray(receipt["terminal_topology"]["lower_saddle_rz_m"])
    reference_null = np.asarray(receipt["efit_x_point_rz_m"])
    solved_field = field(data["radius"], data["height"], data["flux"])
    saddle_level = float(solved_field.ev(*solved_null)) - float(data["boundary"][2])
    shared_levels = (
        np.unique(np.r_[LEVELS, saddle_level]) if abs(saddle_level) > 0.05 else LEVELS
    )
    centre = (solved_null + reference_null) / 2
    half = max(0.15, np.max(np.abs(solved_null - reference_null)) / 2 + 0.08)
    limits = (
        ((0.8, 2.6), (-1.7, 1.5)),
        ((centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)),
    )
    for ax, (r_limits, z_limits) in zip(axes, limits, strict=True):
        poloidal_axes(ax)
        radius = np.linspace(*r_limits, 181)
        height = np.linspace(*z_limits, 241)
        for key, axis, null, color, dash, owner in (
            (
                "label",
                data["reference_axis"],
                reference_null,
                REFERENCE,
                "--",
                "reference_census",
            ),
            ("flux", data["axis"][:2], solved_null, SOLVED, "-", "census"),
        ):
            interpolant = field(data["radius"], data["height"], data[key])
            boundary = (
                float(np.mean(interpolant.ev(*data["label_boundary"].T)))
                if key == "label"
                else float(data["boundary"][2])
            )
            values = interpolant(radius, height) - boundary
            for levels, width in (
                (shared_levels[shared_levels != 0], 1.2),
                ([0.0], 3.0),
            ):
                lines = draw_flux_contours(
                    ax,
                    radius,
                    height,
                    values.T,
                    levels,
                    color=color,
                    linewidth=width,
                    wall=data["limiter"],
                    zorder=2 if key == "label" else 3,
                )
                lines.set_linestyle(dash)
            reference = key == "label"
            admitted = reference or receipt["branch"] == "diverted"
            other = [
                point["rz_m"]
                for point in census[owner]["points"]
                if point["kind"] == "saddle"
                and (
                    not admitted
                    or np.linalg.norm(np.asarray(point["rz_m"]) - null) > 1e-5
                )
            ]
            style = replace(
                DEFAULT_INK,
                axis_color=color,
                xpoint_color=color,
                axis_marker="v" if reference else "^",
                axis_markersize=10 if reference else 7,
                xpoint_marker="s" if reference else "X",
                xpoint_markersize=12 if reference else 8,
                zorder_markers=13 if reference else 14,
            )
            start = len(ax.lines)
            draw_nulls(
                ax,
                magnetic_axis=axis,
                x_points=np.asarray([null]) if admitted else None,
                other_x_points=np.asarray(other),
                style=style,
                contain=data["limiter"],
            )
            if reference:
                for line in ax.lines[start:]:
                    line.set_markerfacecolor("none")
        draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=5)
        ax.set_xlim(r_limits)
        ax.set_ylim(z_limits)
    notes = [
        figure.text(0.27, 0.08, "full vessel", ha="center"),
        figure.text(0.73, 0.08, "lower saddle region", ha="center"),
    ]
    handles = [
        Line2D([], [], color=SOLVED, lw=2.6, label="solved flux"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT flux"),
        Line2D([], [], color=WALL, lw=1.5, label="first wall"),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="solved axis"),
        Line2D([], [], color=SOLVED, marker="X", ls="none", label="solved saddle"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="s",
            mfc="none",
            ls="none",
            label="EFIT saddle",
        ),
    ]
    figure.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    save(
        figure,
        axes,
        notes,
        output / (folder.name + ".svg"),
        previews / (folder.name + ".png"),
    )
    print("SHARED_LEVELS", folder.name, shared_levels.tolist(), flush=True)
    print(
        "CAPTION",
        folder.name,
        json.dumps(
            dict(
                residual=receipt["residual"],
                converged=receipt["converged"],
                **trips["terminal"],
            )
        ),
        flush=True,
    )
    plt.close(figure)


def trajectory(root, diagnosis, output, previews):
    current = json.loads((root / "edge/frame-146-limited/trips.json").read_text())
    previous = json.loads(diagnosis.read_text())
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.1, right=0.96, bottom=0.17, top=0.83, wspace=0.3)
    current_cells = [row["returned"]["connected_cells"] for row in current["trips"]]
    current_heights = [row["returned"]["axis_rz_m"][1] for row in current["trips"]]
    for cells, heights, color, dash, label in (
        (
            previous["connected_cells"],
            previous["axis_height_m"],
            WALL,
            "--",
            "diagnosis",
        ),
        (current_cells, current_heights, SOLVED, "-", "inactive image masked"),
    ):
        height = [100 * value for value in heights]
        for ax, values in zip(axes, (cells, height), strict=True):
            ax.plot(
                np.arange(1, len(cells) + 1),
                values,
                color=color,
                ls=dash,
                lw=3.0 if label == "inactive image masked" else 2.6,
                zorder=3 if label == "inactive image masked" else 2,
                marker="o",
                label=label,
            )
            ax.set_xlabel("outer trip")
            ax.set_xticks(
                np.arange(
                    1, max(len(current_cells), len(previous["connected_cells"])) + 1
                )
            )
    axes[0].set_ylabel("connected cells")
    axes[1].set_ylabel("axis height [cm]")
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    save(
        figure,
        [],
        [],
        output / "limited-trajectory.svg",
        previews / "limited-trajectory.png",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none"})
    for frame, branch in ROWS:
        panel(
            args.input / "edge" / f"frame-{frame}-{branch}", args.output, args.previews
        )
    trajectory(args.input, args.diagnosis, args.output, args.previews)


if __name__ == "__main__":
    main()
