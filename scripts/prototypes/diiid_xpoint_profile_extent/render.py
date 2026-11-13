"""Draw the paired lower-X zooms and the edge profile comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)
from nova.equilibrium.solve_request import SampledFluxFunction  # noqa: E402
from scripts.prototypes.diiid_xpoint_attribution.attribute import field  # noqa: E402
from scripts.prototypes.diiid_xpoint_offset.style import (  # noqa: E402
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)

LEVELS = np.array([-0.02, 0.0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0])
ZOOM_HALF_WIDTH_M = 0.05
SHOTS = (137, 146, 179)
ARMS = ("base", "edge")
NAMES = {"base": "base tables", "edge": "EFIT edge tables"}
STEPS = {
    (137, "diverted"): "137D",
    (146, "diverted"): "146D",
    (146, "limited"): "146L",
    (179, "limited"): "179L",
}
SHOT_LABELS = {137: "137D", 146: "146L", 179: "179L"}


def read(root, arm, frame, branch):
    """Read one row's terminal field, census and paired null positions.

    The solve receipt and the census carry everything the panels draw. An
    imaging receipt is used when a row has one, and otherwise the panel simply
    carries no shadow-node markers.
    """

    folder = root / arm / f"frame-{frame}-{branch}"
    receipt = json.loads((folder / "receipt.json").read_text())
    image = folder / "shadow-image.json"
    summary = dict(
        lower_x_rz_m=receipt["terminal_topology"]["lower_saddle_rz_m"],
        efit_x_rz_m=receipt["efit_x_point_rz_m"],
        residual=receipt["residual"],
        converged=receipt["converged"],
        shadow_rz_m=[],
    )
    if image.is_file():
        measured = json.loads(image.read_text())
        summary.update(
            lower_x_rz_m=measured["lower_x_rz_m"],
            residual=measured["residual"],
            converged=measured["converged"],
            shadow_rz_m=measured["shadow_rz_m"],
        )
    return (
        dict(np.load(folder / "terminal.npz")),
        receipt,
        json.loads((folder / "core-analysis.json").read_text()),
        summary,
    )


def xpoint_panels(root, frame, branch, output, previews):
    """Draw the full field and the lower-X zoom for both arms."""

    pair = [read(root, arm, frame, branch) for arm in ARMS]
    points = np.array(
        [
            point
            for _, _, _, summary in pair
            for point in (summary["lower_x_rz_m"], summary["efit_x_rz_m"])
        ]
    )
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    zoom = dict(
        radius=(centre[0] - ZOOM_HALF_WIDTH_M, centre[0] + ZOOM_HALF_WIDTH_M),
        height=(centre[1] - ZOOM_HALF_WIDTH_M, centre[1] + ZOOM_HALF_WIDTH_M),
    )
    figure = plt.figure(figsize=CANVAS, dpi=100)
    axes = [
        figure.add_axes(box)
        for box in (
            (0.03, 0.18, 0.22, 0.60),
            (0.26, 0.18, 0.22, 0.60),
            (0.53, 0.18, 0.22, 0.60),
            (0.76, 0.18, 0.22, 0.60),
        )
    ]
    notes = []
    for number, (arm, (data, receipt, census, summary)) in enumerate(
        zip(ARMS, pair, strict=True)
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
                for contours, width in ((LEVELS[LEVELS != 0.0], 1.2), ([0.0], 3.0)):
                    drawn = draw_flux_contours(
                        ax,
                        radius,
                        height,
                        values.T,  # noqa: E501
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
            (0.14 + 0.5 * number, 0.13, f"{STEPS[(frame, branch)]} {NAMES[arm]}"),
            (
                0.14 + 0.5 * number,
                0.08,
                f"residual {receipt['residual']:.3g}; "
                f"converged {receipt['converged']:.0f}",
            ),
        ):
            notes.append(figure.text(x, y, label, ha="center", fontsize=14))
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved"),
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
    ]
    if any(summary["shadow_rz_m"] for _, _, _, summary in pair):
        handles.append(
            Line2D(
                [],
                [],
                color=SOLVED,
                marker="o",
                mfc="none",
                ls="none",
                label="shadow nodes",
            )
        )
    figure.legend(
        handles=handles,
        ncol=len(handles),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    name = f"frame-{frame}-{branch}-lower-xpoint"
    save(figure, axes, notes, output / (name + ".svg"), previews / (name + ".png"))
    (output / (name + "-window.json")).write_text(
        json.dumps(
            dict(
                frame=frame,
                branch=branch,
                zoom_window_m=zoom,
                levels_wb=LEVELS.tolist(),
            )
        )
        + "\n"
    )
    plt.close(figure)


def curve(table, name):
    """Evaluate one arm's prescribed table exactly as the solve receives it."""

    return SampledFluxFunction(table["coordinate"], table[name])


def edge_profiles(tables, output, previews):
    """Draw both arms against the extraction they prescribe, near the edge."""

    figure, grid = plt.subplots(3, 2, figsize=CANVAS, dpi=100, sharex=True)
    coordinates = np.linspace(0.94, 1.0, 241)
    labels = (
        ("p_prime", "p' [MPa Wb$^{-1}$]", 1e-6),
        ("ff_prime", "FF' [T$^2$ m$^2$ Wb$^{-1}$]", 1.0),
    )
    notes = []
    for row_index, frame in enumerate(SHOTS):
        record = tables["rows"][str(frame)]
        for column, (name, ylabel, scale) in enumerate(labels):
            ax = grid[row_index][column]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            base = np.asarray(curve(record["base_table"], name)(coordinates)) * scale
            edge = np.asarray(curve(record["edge_table"], name)(coordinates)) * scale
            knots = np.asarray(record["edge_table"]["coordinate"])
            values = np.asarray(record["edge_table"][name]) * scale
            drawn = knots >= coordinates[0]
            knots, values = knots[drawn], values[drawn]
            ax.plot(coordinates, base, color="#929292", lw=3.0, ls="--", zorder=2)
            ax.plot(coordinates, edge, color=SOLVED, lw=3.0, zorder=4)
            ax.plot(
                knots,
                values,
                ls="none",
                marker="o",
                ms=7,
                mfc="white",
                mec=REFERENCE,
                mew=1.6,
                zorder=5,
            )
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xticks((0.95, 1.00))
            if column:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(SHOT_LABELS[frame], fontsize=16, labelpad=8)
            if not row_index:
                ax.set_title(ylabel, fontsize=16, pad=10)
            if row_index == len(SHOTS) - 1:
                ax.set_xlabel("normalised flux ψ$_N$")
            ax.set_xlim(coordinates[0], coordinates[-1])
    handles = [
        Line2D([], [], color="#929292", ls="--", lw=3.0, label="base tables"),
        Line2D([], [], color=SOLVED, lw=3.0, label="EFIT edge tables"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="o",
            mfc="white",
            ls="none",
            label="EFIT map extraction knots",
        ),
    ]
    figure.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    figure.subplots_adjust(
        left=0.092, right=0.955, top=0.82, bottom=0.12, hspace=0.62, wspace=0.24
    )
    name = "edge-profiles"
    save(
        figure,
        list(grid.ravel()),
        notes,
        output / (name + ".svg"),
        previews / (name + ".png"),
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    tables = json.loads(args.tables.read_text())
    for frame, branch in ((137, "diverted"), (146, "limited")):
        xpoint_panels(args.input, frame, branch, args.output, args.previews)
    edge_profiles(tables, args.output, args.previews)


if __name__ == "__main__":
    main()
