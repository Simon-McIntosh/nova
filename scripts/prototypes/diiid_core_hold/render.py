"""Draw paired flux topology and coincident midplane-current comparisons."""

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
from nova.media.poloidal import draw_flux_contours, draw_wall
from scripts.prototypes.diiid_xpoint_offset.render import draw_critical_points
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)


def read(root, arm, case):
    folder = root / arm / case
    return (
        dict(np.load(folder / "terminal.npz")),
        json.loads((folder / "receipt.json").read_text()),
        json.loads((folder / "core-analysis.json").read_text()),
    )


def poloidal(root, case, output, previews):
    pair = [read(root, arm, case) for arm in ("base", "head")]
    low = min(
        float(data[name].min()) for data, _, _ in pair for name in ("flux", "label")
    )
    high = max(
        float(data[name].max()) for data, _, _ in pair for name in ("flux", "label")
    )
    levels = list(np.linspace(low, high, 12)[1:-1])
    for data, _, analysis in pair:
        levels.append(float(data["boundary"][2]))
        for key in ("census", "reference_census"):
            core = analysis[key]
            width = max(core["maxima_flux_spread_mwb"] / 1000, 0.002)
            levels.extend(core["max_flux_wb"] - width * np.array([0.25, 0.75, 1.5]))
    levels = np.unique(levels)
    points = np.array(
        [
            point["rz_m"]
            for _, _, analysis in pair
            for key in ("census", "reference_census")
            for point in analysis[key]["points"]
            if point["in_plasma"]
        ]
    )
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    span = max(float(np.max(np.ptp(points, axis=0))) + 0.06, 0.12)
    figure = plt.figure(figsize=CANVAS, dpi=100)
    axes = [
        figure.add_axes(box)
        for box in (
            (0.02, 0.17, 0.26, 0.56),
            (0.29, 0.25, 0.19, 0.38),
            (0.52, 0.17, 0.26, 0.56),
            (0.79, 0.25, 0.19, 0.38),
        )
    ]
    solved_style = replace(
        DEFAULT_INK,
        axis_color=SOLVED,
        xpoint_color=SOLVED,
        axis_marker="^",
        xpoint_marker="X",
        zorder_markers=12,
    )
    reference_style = replace(
        DEFAULT_INK,
        axis_color=REFERENCE,
        xpoint_color=REFERENCE,
        axis_marker="v",
        xpoint_marker="P",
        zorder_markers=15,
    )
    notes = []
    for i, (arm, (data, receipt, analysis)) in enumerate(
        zip(("base", "head"), pair, strict=True)
    ):
        for j, ax in enumerate(axes[2 * i : 2 * i + 2]):
            poloidal_axes(ax)
            for key, color, dash, layer in (
                ("label", REFERENCE, "--", 2),
                ("flux", SOLVED, "-", 3),
            ):
                contour = draw_flux_contours(
                    ax,
                    data["radius"],
                    data["height"],
                    data[key].T,
                    levels,
                    color=color,
                    linewidth=2.6,
                    wall=data["limiter"],
                    zorder=layer,
                )
                contour.set_linestyle(dash)
            ax.plot(
                *data["label_boundary"].T, color=REFERENCE, ls=":", lw=1.2, zorder=4
            )
            draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=5)
            draw_critical_points(
                ax, analysis["census"], np.zeros(2), solved_style, "o", 7
            )
            draw_critical_points(
                ax, analysis["reference_census"], np.zeros(2), reference_style, "s", 7
            )
            if j:
                ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
                ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
            else:
                ax.set_xlim(data["radius"][[0, -1]])
                ax.set_ylim(data["height"][[0, -1]])
        topology = receipt["terminal_topology"]
        distance = topology["lower_saddle_distance_m"]
        for x, y, label in (
            (0.22, 0.76, arm),
            (0.385, 0.67, "core zoom"),
            (
                0.25,
                0.12,
                f"residual {receipt['residual']:.5g}; converged {receipt['converged']}",
            ),
            (
                0.25,
                0.075,
                f"maxima {analysis['maxima_count']}; support {receipt['support']}",
            ),
            (
                0.25,
                0.03,
                "lower X gap unavailable"
                if distance is None
                else f"lower X gap {100 * distance:.3f} cm",
            ),
        ):
            notes.append(figure.text(x + 0.5 * i, y, label, ha="center", fontsize=14))
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved flux"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT flux"),
        Line2D([], [], color=REFERENCE, lw=1.2, ls=":", label="EFIT boundary"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
    ]
    for color, marker, label in (
        (SOLVED, "^", "solved maxima"),
        (REFERENCE, "v", "EFIT maxima"),
        (SOLVED, "X", "solved saddles"),
        (REFERENCE, "P", "EFIT saddles"),
        (SOLVED, "o", "solved minima"),
        (REFERENCE, "s", "EFIT minima"),
    ):
        handles.append(
            Line2D([], [], color=color, marker=marker, ls="none", label=label)
        )
    figure.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=13,
    )
    save(
        figure,
        axes,
        notes,
        output / (case + "-poloidal.svg"),
        previews / (case + "-poloidal.png"),
    )
    (output / (case + "-levels.json")).write_text(json.dumps(levels.tolist()) + "\n")
    plt.close(figure)


def midplane(root, case, output, previews):
    base, head = [read(root, arm, case) for arm in ("base", "head")]
    figure, axes = plt.subplots(1, 3, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.77, wspace=0.38)
    b, h = base[2], head[2]
    x = np.asarray(h["midplane"]["radius"])
    support = np.asarray(h["midplane"]["efit_support"], bool)
    for i, (ax, key, name) in enumerate(
        zip(
            axes,
            ("map_density", "pressure_term", "field_term"),
            ("map current", "pressure contribution", "field contribution"),
            strict=True,
        )
    ):
        for arm, analysis, color, marker, start in (
            ("base", b, WALL, "o", 0),
            ("head", h, SOLVED, "s", 15),
        ):
            cut = analysis["midplane"]
            values = np.asarray(cut[key] if i == 0 else cut["source"][key]) * (
                1 if i == 0 else analysis["amplitude"]
            )
            selected = np.asarray(cut["source"]["support"], bool)
            values = np.where(selected, values, np.nan) / 1e6
            ax.plot(
                x,
                values,
                color=color,
                lw=3 if arm == "head" else 2.6,
                ls="-" if arm == "head" else "--",
                zorder=3 if arm == "head" else 2,
            )
            ax.plot(
                x,
                values,
                color=color,
                marker=marker,
                mfc="white",
                ls="none",
                ms=5,
                markevery=(start, 30),
                zorder=7,
            )
        if i == 0:
            values = np.asarray(h["midplane"]["efit_density"]) / 1e6
            ax.plot(
                x,
                np.where(support, values, np.nan),
                color=REFERENCE,
                lw=2.6,
                ls=(0, (3, 3)),
                zorder=4,
            )
            ax.plot(
                x,
                np.where(support, values, np.nan),
                color=REFERENCE,
                marker="x",
                ls="none",
                ms=5,
                markevery=(7, 30),
                zorder=8,
            )
        ax.set_xlabel("radius [m]")
        ax.set_ylabel("current density [MA/m²]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.text(0.5, 1.04, name, transform=ax.transAxes, ha="center", fontsize=16)
    handles = [
        Line2D(
            [], [], color=WALL, lw=2.6, ls="--", marker="o", mfc="white", label="base"
        ),
        Line2D([], [], color=SOLVED, lw=3, marker="s", mfc="white", label="head"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", marker="x", label="EFIT map"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        handlelength=4,
    )
    save(
        figure,
        axes,
        [],
        output / (case + "-midplane.svg"),
        previews / (case + "-midplane.png"),
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
    for case in ("frame-137-diverted", "frame-146-diverted"):
        poloidal(args.input, case, args.output, args.previews)
    for case in ("frame-137-diverted", "frame-146-diverted", "mast-21978-35"):
        midplane(args.input, case, args.output, args.previews)


if __name__ == "__main__":
    main()
