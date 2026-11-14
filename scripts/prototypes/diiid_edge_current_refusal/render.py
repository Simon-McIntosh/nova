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


def case_folder(root, frame, branch):
    return (
        root
        if (root / "terminal.npz").is_file()
        else root / "edge" / f"frame-{frame}-{branch}"
    )


def load(root, frame, branch):
    folder = case_folder(root, frame, branch)
    return (
        dict(np.load(folder / "terminal.npz")),
        json.loads((folder / "receipt.json").read_text()),
        json.loads((folder / "spatial.json").read_text()),
        json.loads((folder / "core-analysis.json").read_text()),
    )


def points(entries):
    nonempty = [np.asarray(entry) for entry in entries if len(entry)]
    return np.unique(np.vstack(nonempty), axis=0) if nonempty else np.empty((0, 2))


def panel(root, frame, branch, output, previews, name=None):
    data, receipt, spatial, census = load(root, frame, branch)
    changed = points(
        [
            trip[key]
            for trip in spatial["trips"]
            for key in ("entering_rz_m", "leaving_rz_m")
        ]
    )
    last_inner = len(spatial["trips"]) - 1
    folder = case_folder(root, frame, branch)
    cutoff = json.loads((folder / "cutoff-jumps.json").read_text())[
        "smallest_crossing_pair"
    ]
    cutoff_points = np.asarray(cutoff["crossing_rz_m"]) if cutoff else np.empty((0, 2))
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
            admitted = reference or spatial["terminal"]["boundary_class"] == "diverted"
            other = [
                point["rz_m"]
                for point in census[owner]["points"]
                if point["kind"] == "saddle"
                and (
                    not admitted
                    or np.linalg.norm(np.asarray(point["rz_m"]) - null) > 1e-5
                )
            ]
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
        if ax is axes[1] and len(cutoff_points):
            ax.plot(
                *cutoff_points.T, ls="none", marker="*", ms=13, color=REFUSED, zorder=15
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
    handles.append(
        Line2D([], [], color=REFUSED, marker="*", ls="none", label="source cutoff")
    )
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
    name = name or f"frame-{frame}-{branch}"
    save(figure, axes, notes, output / f"{name}.svg", previews / f"{name}.png")
    plt.close(figure)
    return dict(
        case=name,
        changed_cells=len(changed),
        terminal_refused_cells=len(refused),
        residual=receipt["residual"],
        converged=receipt["converged"],
        levels_wb=LEVELS.tolist(),
        source_cutoff_cells=len(cutoff_points),
    )


def traces(root, output, previews):
    figure, axes = plt.subplots(2, 4, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(
        left=0.08, right=0.98, top=0.80, bottom=0.13, wspace=0.5, hspace=0.35
    )
    notes = []
    for column, (frame, branch) in enumerate(ROWS):
        _, receipt, spatial, _ = load(root, frame, branch)
        residuals = np.asarray(receipt["residual_history"])
        residuals = residuals[np.isfinite(residuals)]
        trips = np.arange(1, len(residuals) + 1)
        frozen = [trip["inner"]["residual"] for trip in spatial["trips"]]
        axes[0, column].semilogy(trips, frozen, color=WALL, lw=2.6, ls="--", zorder=2)
        axes[0, column].semilogy(trips, residuals, color=SOLVED, lw=3, marker=".")
        incoming = [trip["initial"]["support"] for trip in spatial["trips"]]
        support = [trip["returned"]["support"] for trip in spatial["trips"]]
        axes[1, column].plot(trips, incoming, color=WALL, lw=2.6, ls="--")
        axes[1, column].plot(
            np.arange(1, len(support) + 1), support, color=SOLVED, lw=3, marker="."
        )
        axes[1, column].plot(
            trips[-1],
            receipt["support"],
            color=REFERENCE,
            marker="o",
            mfc="none",
            ms=9,
            mew=1.5,
            zorder=5,
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
            figure.text(0.17 + column * 0.233, 0.86, f"{frame} {branch}", ha="center")
        )
    figure.legend(
        handles=[
            Line2D(
                [], [], color=SOLVED, lw=3, label="live residual / returned support"
            ),
            Line2D(
                [],
                [],
                color=WALL,
                lw=2.6,
                ls="--",
                label="frozen residual / incoming support",
            ),
            Line2D(
                [],
                [],
                color=REFERENCE,
                marker="o",
                mfc="none",
                ls="none",
                label="terminal support",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=12,
    )
    save(
        figure,
        list(axes.ravel()),
        notes,
        output / "trip-traces.svg",
        previews / "trip-traces.png",
    )
    plt.close(figure)


def limited_trajectories(root, output, previews):
    """Expose axis loss beside the otherwise inactive radial-field unknown."""
    figure, axes = plt.subplots(3, 2, figsize=(CANVAS[0], 9.4), dpi=100)
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.91, bottom=0.09, hspace=0.36, wspace=0.30
    )
    notes = []
    for column, frame in enumerate((146, 179)):
        folder = root / "edge" / f"frame-{frame}-limited"
        audit = json.loads((folder / "iterate-audit.json").read_text())
        stages = audit["stages"]
        series = (
            [row["physical"]["axis_rz_m"][1] for row in stages],
            [row["physical"]["support"] for row in stages],
            [1000 * row["inactive_unknown_t"] for row in stages],
        )
        for ax, values, label in zip(
            axes[:, column],
            series,
            ("axis Z [m]", "support cells", "radial field [mT]"),
            strict=True,
        ):
            ax.plot(np.arange(len(stages)), values, color=SOLVED, lw=3)
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
        axes[-1, column].set_xlabel("captured warmup / Newton state")
        notes.append(
            figure.text(0.30 + column * 0.47, 0.96, f"{frame} limited", ha="center")
        )
    save(
        figure,
        list(axes.ravel()),
        notes,
        output / "limited-trajectories.svg",
        previews / "limited-trajectories.png",
    )
    plt.close(figure)


def source_changes(root, output, previews):
    """Compare current increments with the actually computed tangent remainder."""
    figure, axes = plt.subplots(2, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(
        left=0.09, right=0.98, top=0.84, bottom=0.11, hspace=0.7, wspace=0.25
    )
    notes = []
    for index, (frame, branch) in enumerate(ROWS):
        ax = axes.ravel()[index]
        _, _, spatial, _ = load(root, frame, branch)
        sweep = spatial["directional_discriminator"]
        factors = np.asarray([row["factor"] for row in sweep])
        for key, color, dash in (
            ("current_delta_l2_a", SOLVED, "-"),
            ("tangent_remainder_l2_a", WALL, "--"),
        ):
            values = np.asarray([row[key] for row in sweep]) / 1000
            positive = values > 0
            ax.loglog(
                factors[positive],
                values[positive],
                color=color,
                ls=dash,
                lw=3 if key == "current_delta_l2_a" else 2.6,
                marker=".",
            )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.set_xlabel("fraction of proposed step")
        ax.set_ylabel("current increment [kA]")
        notes.append(
            figure.text(
                0.30 if index % 2 == 0 else 0.77,
                0.88 if index < 2 else 0.44,
                f"{frame} {branch}",
                ha="center",
            )
        )
    figure.legend(
        handles=[
            Line2D([], [], color=SOLVED, lw=3, label="measured increment"),
            Line2D([], [], color=WALL, lw=2.6, ls="--", label="tangent remainder"),
        ],
        ncol=2,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    save(
        figure,
        list(axes.ravel()),
        notes,
        output / "source-cutoff-traces.svg",
        previews / "source-cutoff-traces.png",
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
    rows.append(
        panel(
            args.input / "inactive-field-suppressed",
            146,
            "limited",
            args.output,
            args.previews,
            name="inactive-field-suppressed",
        )
    )
    traces(args.input, args.output, args.previews)
    limited_trajectories(args.input, args.output, args.previews)
    source_changes(args.input, args.output, args.previews)
    (args.previews / "render-receipt.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
