"""Draw paired physical separatrices and mark residual-shadow locations."""

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


def read(root, arm, frame):
    folder = root / arm / f"frame-{frame}-diverted"
    return (
        dict(np.load(folder / "terminal.npz")),
        json.loads((folder / "receipt.json").read_text()),
        json.loads((folder / "core-analysis.json").read_text()),
        json.loads((folder / "shadow-image.json").read_text()),
    )


def render(root, frame, output, previews):
    pair = [read(root, arm, frame) for arm in ("base", "head")]
    points = np.array(
        [
            point
            for _, _, _, summary in pair
            for point in (summary["lower_x_rz_m"], summary["efit_x_rz_m"])
        ]
    )
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    zoom = dict(
        radius=(centre[0] - 0.20, centre[0] + 0.20),
        height=(centre[1] - 0.20, centre[1] + 0.20),
    )
    figure = plt.figure(figsize=CANVAS, dpi=100)
    axes = [
        figure.add_axes(box)
        for box in (
            (0.02, 0.16, 0.23, 0.63),
            (0.26, 0.16, 0.23, 0.63),
            (0.52, 0.16, 0.23, 0.63),
            (0.76, 0.16, 0.23, 0.63),
        )
    ]
    levels = np.array([-0.02, 0.0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0])
    notes = []
    for number, (arm, (data, receipt, census, summary)) in enumerate(
        zip(("base", "head"), pair, strict=True)
    ):
        for index, ax in enumerate(axes[2 * number : 2 * number + 2]):
            poloidal_axes(ax)
            r_limits = zoom["radius"] if index else (0.9, 2.4)
            z_limits = zoom["height"] if index else (-1.45, 1.35)
            radius = np.linspace(*r_limits, 101)
            height = np.linspace(*z_limits, 141)
            for key, xpoint, magnetic_axis, color, dash, marker, owner in (
                (
                    "label",
                    summary["efit_x_rz_m"],
                    data["reference_axis"],
                    REFERENCE,
                    "--",
                    "P",
                    "reference_census",
                ),
                (
                    "flux",
                    summary["lower_x_rz_m"],
                    data["axis"][:2],
                    SOLVED,
                    "-",
                    "X",
                    "census",
                ),
            ):
                interpolant = field(data["radius"], data["height"], data[key])
                values = interpolant(radius, height) - float(interpolant.ev(*xpoint))
                for contours, width in ((levels, 1.2), ([0.0], 3.0)):
                    drawn = draw_flux_contours(
                        ax,
                        radius,
                        height,
                        values.T,
                        contours,
                        color=color,
                        linewidth=width,
                        wall=data["limiter"],
                        zorder=3 if key == "flux" else 2,
                    )
                    drawn.set_linestyle(dash)
                style = replace(
                    DEFAULT_INK,
                    axis_color=color,
                    xpoint_color=color,
                    axis_marker="^" if key == "flux" else "v",
                    xpoint_marker=marker,
                    zorder_markers=12,
                )
                other = [
                    point["rz_m"]
                    for point in census[owner]["points"]
                    if point["kind"] == "saddle"
                    and np.linalg.norm(np.asarray(point["rz_m"]) - xpoint) > 1e-5
                ]
                start = len(ax.lines)
                draw_nulls(
                    ax,
                    magnetic_axis=magnetic_axis,
                    x_points=np.array([xpoint]),
                    other_x_points=np.asarray(other),
                    style=style,
                    contain=data["limiter"],
                )
                if key == "label":
                    for line in ax.lines[start:]:
                        line.set_markerfacecolor("none")
            draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=5)
            shadow = np.asarray(summary["shadow_rz_m"])
            if len(shadow):
                ax.plot(
                    *shadow.T,
                    ls="none",
                    marker="o",
                    ms=9,
                    mfc="none",
                    mec=SOLVED,
                    mew=1.4,
                    zorder=15,
                )
            ax.set_xlim(r_limits)
            ax.set_ylim(z_limits)
        for x, y, label in (
            (0.25, 0.82, arm),
            (
                0.25,
                0.11,
                f"residual {receipt['residual']:.3g}; converged {receipt['converged']}",
            ),
            (0.25, 0.06, f"lower X gap {100 * summary['lower_x_distance_m']:.3f} cm"),
        ):
            notes.append(
                figure.text(x + number * 0.5, y, label, ha="center", fontsize=14)
            )
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
        Line2D(
            [],
            [],
            color=SOLVED,
            marker="o",
            mfc="none",
            ls="none",
            label="shadow nodes",
        ),
    ]
    figure.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    name = f"frame-{frame}-lower-xpoint"
    save(figure, axes, notes, output / (name + ".svg"), previews / (name + ".png"))
    (output / (name + "-window.json")).write_text(
        json.dumps(dict(frame=frame, zoom_window_m=zoom, levels_wb=levels.tolist()))
        + "\n"
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
    for frame in (137, 146):
        render(args.input, frame, args.output, args.previews)


if __name__ == "__main__":
    main()
