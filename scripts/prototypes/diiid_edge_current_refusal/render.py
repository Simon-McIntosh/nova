"""Draw the recorded support transitions and residual histories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
LEVELS = np.array([-0.02, 0.0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0])
REFUSED = "#bf3446"


def load(root, frame, branch):
    folder = root / "edge" / f"frame-{frame}-{branch}"
    return (
        dict(np.load(folder / "terminal.npz")),
        json.loads((folder / "receipt.json").read_text()),
        json.loads((folder / "spatial.json").read_text()),
        json.loads((folder / "core-analysis.json").read_text()),
    )


def points(entries):
    nonempty = [np.asarray(entry) for entry in entries if len(entry)]
    return np.unique(np.vstack(nonempty), axis=0) if nonempty else np.empty((0, 2))


def panel(root, frame, branch, output, previews):
    data, receipt, spatial, census = load(root, frame, branch)
    changed = points(
        [
            trip[key]
            for trip in spatial["trips"]
            for key in ("entering_rz_m", "leaving_rz_m")
        ]
    )
    last_inner = len(spatial["trips"]) - 1
    refused = points(
        [
            proposal["comparisons"][0]["changed_rz_m"]
            for proposal in spatial["proposals"]
            if proposal["inner"] == last_inner and not proposal["accepted"]
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.81, bottom=0.14, wspace=0.2)
    for ax, marks, color, marker in zip(
        axes, (changed, refused), (SOLVED, REFUSED), ("s", "o"), strict=True
    ):
        poloidal_axes(ax)
        radius, height = np.linspace(0.9, 2.4, 111), np.linspace(-1.45, 1.35, 151)
        for key, axis, null, shade, dash, owner in (
            (
                "label",
                data["reference_axis"],
                receipt["efit_x_point_rz_m"],
                REFERENCE,
                "--",
                "reference_census",
            ),
            (
                "flux",
                data["axis"][:2],
                receipt["terminal_topology"]["lower_saddle_rz_m"],
                SOLVED,
                "-",
                "census",
            ),
        ):
            interpolant = field(data["radius"], data["height"], data[key])
            boundary = (
                float(np.mean(interpolant.ev(*data["label_boundary"].T)))
                if key == "label"
                else float(data["boundary"][2])
            )
            values = interpolant(radius, height) - boundary
            for levels, width in ((LEVELS[LEVELS != 0], 1.0), ([0], 2.6)):
                lines = draw_flux_contours(
                    ax,
                    radius,
                    height,
                    values.T,
                    levels,
                    color=shade,
                    linewidth=width,
                    wall=data["limiter"],
                    zorder=2 if key == "label" else 3,
                )
                lines.set_linestyle(dash)
            reference = key == "label"
            style = replace(
                DEFAULT_INK,
                axis_color=shade,
                xpoint_color=shade,
                axis_marker="v" if reference else "^",
                axis_markersize=10 if reference else 7,
                xpoint_marker="s" if reference else "X",
                xpoint_markersize=12 if reference else 8,
                zorder_markers=13 if reference else 12,
            )
            other = [
                point["rz_m"]
                for point in census[owner]["points"]
                if point["kind"] == "saddle"
                and np.linalg.norm(np.asarray(point["rz_m"]) - null) > 1e-5
            ]
            start = len(ax.lines)
            draw_nulls(
                ax,
                magnetic_axis=axis,
                x_points=np.asarray([null]),
                other_x_points=np.asarray(other),
                style=style,
                contain=data["limiter"],
            )
            if reference:
                for line in ax.lines[start:]:
                    line.set_markerfacecolor("none")
        draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=5)
        if len(marks):
            ax.plot(
                *marks.T,
                ls="none",
                marker=marker,
                ms=6,
                mfc="none",
                mec=color,
                mew=1.3,
                zorder=14,
            )
        ax.set_xlim(0.9, 2.4)
        ax.set_ylim(-1.45, 1.35)
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
        Line2D(
            [],
            [],
            color=SOLVED,
            marker="s",
            mfc="none",
            ls="none",
            label="support changed",
        ),
        Line2D(
            [],
            [],
            color=REFUSED,
            marker="o",
            mfc="none",
            ls="none",
            label="refused proposal",
        ),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="solved axis"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="v",
            mfc="none",
            ls="none",
            label="EFIT axis",
        ),
        Line2D([], [], color=SOLVED, marker="X", ls="none", label="solved null"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="s",
            mfc="none",
            ls="none",
            label="EFIT null",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=5,
        fontsize=12,
    )
    notes = [
        figure.text(
            0.26, 0.08, f"{len(changed)} cells change across inner solves", ha="center"
        ),
        figure.text(
            0.74,
            0.08,
            f"{len(refused)} cells change in terminal refused proposals",
            ha="center",
        ),
    ]
    name = f"frame-{frame}-{branch}"
    save(figure, axes, notes, output / f"{name}.svg", previews / f"{name}.png")
    plt.close(figure)
    return dict(
        case=name,
        changed_cells=len(changed),
        terminal_refused_cells=len(refused),
        residual=receipt["residual"],
        converged=receipt["converged"],
        levels_wb=LEVELS.tolist(),
    )


def traces(root, output, previews):
    figure, axes = plt.subplots(2, 4, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.13, wspace=0.5, hspace=0.35
    )
    notes = []
    for column, (frame, branch) in enumerate(ROWS):
        _, receipt, spatial, _ = load(root, frame, branch)
        residuals = np.asarray(receipt["residual_history"])
        residuals = residuals[np.isfinite(residuals)]
        trips = np.arange(1, len(residuals) + 1)
        axes[0, column].semilogy(trips, residuals, color=SOLVED, lw=3, marker=".")
        support = [trip["returned"]["support"] for trip in spatial["trips"]]
        axes[1, column].plot(
            np.arange(1, len(support) + 1), support, color=SOLVED, lw=3, marker="."
        )
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.grid(False)
        axes[1, column].set_xlabel("inner solve")
        if column == 0:
            axes[0, column].set_ylabel("live residual")
            axes[1, column].set_ylabel("support cells")
        notes.append(
            figure.text(0.17 + column * 0.233, 0.95, f"{frame} {branch}", ha="center")
        )
    save(
        figure,
        list(axes.ravel()),
        notes,
        output / "trip-traces.svg",
        previews / "trip-traces.png",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    rows = [
        panel(args.input, frame, branch, args.output, args.previews)
        for frame, branch in ROWS
    ]
    traces(args.input, args.output, args.previews)
    (args.previews / "render-receipt.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
