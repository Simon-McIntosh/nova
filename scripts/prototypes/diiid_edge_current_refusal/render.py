"""Draw the recorded support transitions and residual histories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    LogFormatterMathtext,
    MaxNLocator,
    NullLocator,
)
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from scripts.prototypes.diiid_edge_current_refusal.marker_visibility import (
    check as check_markers,
)
from scripts.prototypes.diiid_xpoint_attribution.attribute import field
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save as checked_save,
)

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
LEVELS = np.array([-0.02, 0.0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0])
REFUSED = "#bf3446"


def linear_scale(ax, dimension, values, integer=False):
    values = np.asarray(values)
    low, high = float(values.min()), float(values.max())
    padding = max(0.05 * (high - low), 1.0 if integer else 0.01)
    limits = (low - padding, high + padding)
    axis = getattr(ax, dimension + "axis")
    getattr(ax, "set_" + dimension + "lim")(*limits)
    ticks = MaxNLocator(4, integer=integer).tick_values(*limits)
    axis.set_major_locator(
        FixedLocator(ticks[(ticks >= limits[0]) & (ticks <= limits[1])])
    )
    axis.set_minor_locator(NullLocator())
    ax.ticklabel_format(axis=dimension, style="plain", useOffset=False)


def log_scale(ax, dimension, values, margin=0.0):
    positive = np.asarray(values)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    low = int(np.floor(np.log10(positive.min())))
    high = max(low + 1, int(np.ceil(np.log10(positive.max()))))
    stride = max(1, int(np.ceil((high - low) / 4)))
    powers = np.unique(np.r_[np.arange(low, high + 1, stride), high])
    axis = getattr(ax, dimension + "axis")
    getattr(ax, "set_" + dimension + "lim")(
        10.0 ** (low - margin), 10.0 ** (high + margin)
    )
    axis.set_major_locator(FixedLocator(10.0**powers))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_locator(NullLocator())


def save(figure, axes, notes, output, preview):
    """Keep the common physical width when a stacked trace needs more height."""
    checked_save(figure, axes, notes, output, preview)
    _, height = figure.get_size_inches()
    text = re.sub(
        r'(<svg\b[^>]*\bwidth=")[^"]+', r"\g<1>31.1cm", output.read_text(), count=1
    )
    text = re.sub(
        r'(<svg\b[^>]*\bheight=")[^"]+',
        rf"\g<1>{16 * height / CANVAS[1]:.5f}cm",
        text,
        count=1,
    )
    output.write_text(text)


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
    saddles = np.asarray(
        [
            point["rz_m"]
            for owner in ("census", "reference_census")
            for point in census[owner]["points"]
            if point["kind"] == "saddle"
        ]
    )
    r_limits = (
        min(0.75, saddles[:, 0].min() - 0.05),
        max(2.65, saddles[:, 0].max() + 0.05),
    )
    z_limits = (
        min(-1.7, saddles[:, 1].min() - 0.05),
        max(1.7, saddles[:, 1].max() + 0.05),
    )
    changed = points(
        [
            trip[key]
            for trip in spatial["trips"]
            for key in ("entering_rz_m", "leaving_rz_m")
        ]
    )
    last_inner = len(spatial["trips"]) - 1
    folder = case_folder(root, frame, branch)
    audit = json.loads((folder / "iterate-audit.json").read_text())
    shadow_changed = points(
        [
            trip["shadow_transition"].get(key, [])
            for trip in audit["trips"]
            for key in ("entering_rz_m", "leaving_rz_m")
        ]
    )
    mask_changed = points([changed, shadow_changed])
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
        axes, (mask_changed, refused), (WALL, REFUSED), ("s", "o"), strict=True
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
        if len(marks) and ax is axes[0]:
            ax.plot(
                *marks.T,
                ls="none",
                marker=marker,
                ms=4,
                mfc="none",
                mec=color,
                mew=0.8,
                zorder=1,
            )
        if ax is axes[1]:
            coincident = np.any(
                np.all(refused[:, None, :] == cutoff_points[None, :, :], axis=2),
                axis=1,
            )
            for shared, size in ((False, 6), (True, 16)):
                group = refused[coincident == shared]
                if len(group):
                    ax.plot(
                        *group.T,
                        ls="none",
                        marker="o",
                        ms=size,
                        mfc="none",
                        mec=REFUSED,
                        mew=1.3,
                        zorder=14,
                    )
        if ax is axes[1] and len(cutoff_points):
            ax.plot(
                *cutoff_points.T, ls="none", marker="*", ms=9, color=REFUSED, zorder=15
            )
        ax.set_xlim(*r_limits)
        ax.set_ylim(*z_limits)
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
        Line2D(
            [],
            [],
            color=WALL,
            marker="s",
            mfc="none",
            ls="none",
            label="mask change",
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
            0.26,
            0.08,
            f"{len(changed)} support cells; {len(shadow_changed)} shadow nodes change",
            ha="center",
        ),
        figure.text(
            0.74,
            0.08,
            f"{len(refused)} cells change in terminal refused proposals",
            ha="center",
        ),
    ]
    if np.any(coincident):
        notes.append(
            figure.text(
                0.5,
                0.025,
                "A circled star is both a refused cell and a source cutoff.",
                ha="center",
                fontsize=12,
            )
        )
    name = name or f"frame-{frame}-{branch}"
    save(figure, axes, notes, output / f"{name}.svg", previews / f"{name}.png")
    visibility = check_markers(output, root, name, folder=folder)
    print("MARKER_VISIBILITY " + json.dumps(visibility))
    if visibility["errors"]:
        raise ValueError(f"{name}: {visibility['errors']}")
    plt.close(figure)
    return dict(
        case=name,
        changed_cells=len(changed),
        terminal_refused_cells=len(refused),
        residual=receipt["residual"],
        converged=receipt["converged"],
        levels_wb=LEVELS.tolist(),
        source_cutoff_cells=len(cutoff_points),
        source_cutoff_refused_cells=int(np.count_nonzero(coincident)),
        changed_shadow_nodes=len(shadow_changed),
        r_limits_m=list(r_limits),
        z_limits_m=list(z_limits),
        qualified_saddles=len(saddles),
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
        axes[0, column].semilogy(
            trips, residuals, color=SOLVED, lw=3, marker=".", zorder=3
        )
        log_scale(axes[0, column], "y", np.r_[residuals, frozen])
        incoming = [trip["initial"]["support"] for trip in spatial["trips"]]
        support = [trip["returned"]["support"] for trip in spatial["trips"]]
        axes[1, column].plot(trips, incoming, color=WALL, lw=2.6, ls="--")
        axes[1, column].plot(
            np.arange(1, len(support) + 1),
            support,
            color=SOLVED,
            lw=3,
            marker=".",
            zorder=3,
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
        linear_scale(
            axes[1, column], "y", incoming + support + [receipt["support"]], True
        )
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0.7, len(trips) + 0.3)
            ax.xaxis.set_major_locator(FixedLocator(trips))
            ax.xaxis.set_minor_locator(NullLocator())
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
            ("axis Z [m]", "support cells", "inactive field [mT]"),
            strict=True,
        ):
            ax.plot(np.arange(len(stages)), values, color=SOLVED, lw=3)
            linear_scale(ax, "x", np.arange(len(stages)), True)
            linear_scale(ax, "y", values, label == "support cells")
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
        all_values = []
        for key, color, dash in (
            ("current_delta_l2_a", SOLVED, "-"),
            ("tangent_remainder_l2_a", WALL, "--"),
        ):
            values = np.asarray([row[key] for row in sweep]) / 1000
            all_values.extend(values.tolist())
            positive = values > 0
            ax.loglog(
                factors[positive],
                values[positive],
                color=color,
                ls=dash,
                lw=3 if key == "current_delta_l2_a" else 2.6,
                marker=".",
                zorder=3 if key == "current_delta_l2_a" else 2,
            )
        log_scale(ax, "x", factors, margin=0.5)
        log_scale(ax, "y", all_values)
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
    cases = {f"frame-{frame}-{branch}": (None, frame, branch) for frame, branch in ROWS}
    cases.update(
        {
            name: (name, 146, "limited")
            for name in ("inactive-field-suppressed", "inactive-field-reference")
        }
    )
    trace_writers = {
        "trip-traces": traces,
        "limited-trajectories": limited_trajectories,
        "source-cutoff-traces": source_changes,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=[*cases, *trace_writers])
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    selected = args.only or [*cases, *trace_writers]
    rows = []
    for name in selected:
        if name in cases:
            subfolder, frame, branch = cases[name]
            rows.append(
                panel(
                    args.input / subfolder if subfolder else args.input,
                    frame,
                    branch,
                    args.output,
                    args.previews,
                    name=name,
                )
            )
        else:
            trace_writers[name](args.input, args.output, args.previews)
    (args.previews / "render-receipt.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
