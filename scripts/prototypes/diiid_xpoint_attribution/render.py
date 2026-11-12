"""Draw inspected saved-field geometry, source profiles and displacement shares."""

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
from scipy.optimize import root

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from scripts.prototypes.diiid_xpoint_attribution.attribute import field, gradient
from scripts.prototypes.diiid_xpoint_offset.style import (
    CANVAS,
    REFERENCE,
    SOLVED,
    WALL,
    save,
)


def read(folder, frame):
    return (
        dict(np.load(folder / f"fields-{frame}.npz")),
        json.loads((folder / f"result-{frame}.json").read_text()),
        json.loads((folder / f"sensitivity-{frame}.json").read_text()),
    )


def geometry(data, result, output, previews):
    figure = plt.figure(figsize=CANVAS, dpi=100)
    axes = [
        figure.add_axes(box)
        for box in ((0.03, 0.12, 0.43, 0.70), (0.54, 0.12, 0.43, 0.70))
    ]
    radius, height = data["radius"], data["height"]
    point = np.asarray(result["reference_saddle_m"])
    other = np.asarray(result["solved_saddle_m"])
    notes = []
    physical_style = replace(
        DEFAULT_INK,
        axis_color=WALL,
        xpoint_color=WALL,
        axis_marker="s",
        xpoint_marker="D",
        zorder_markers=14,
    )
    physical_interpolant = field(radius, height, data["physical_image"])
    physical_axis = root(
        lambda point: gradient(physical_interpolant, point), data["solved_axis"]
    ).x
    styles = [
        replace(
            DEFAULT_INK,
            axis_color=REFERENCE,
            xpoint_color=REFERENCE,
            axis_marker="v",
            xpoint_marker="P",
            zorder_markers=12,
        ),
        replace(
            DEFAULT_INK,
            axis_color=SOLVED,
            xpoint_color=SOLVED,
            axis_marker="^",
            xpoint_marker="X",
            zorder_markers=15,
        ),
    ]
    # Constant gauges align each saddle to zero; both maps use one Wb array.
    levels = np.array([-0.02, 0, 0.02, 0.2, 0.6, 1.0, 1.6, 2.0])
    for index, ax in enumerate(axes):
        poloidal_axes(ax)
        center = (point + other) / 2
        r_limits = (center[0] - 0.18, center[0] + 0.18) if index else (0.9, 2.4)
        z_limits = (center[1] - 0.18, center[1] + 0.18) if index else (-1.45, 1.35)
        for key, axis, xpoint, style, dash, color in [
            ("reference", data["reference_axis"], point, styles[0], "--", REFERENCE),
            (
                "physical_image",
                physical_axis,
                result["physical_image_saddle_m"],
                physical_style,
                "-.",
                WALL,
            ),
            ("solved", data["solved_axis"], other, styles[1], "-", SOLVED),
        ]:
            interpolant = field(radius, height, data[key])
            dense_r = np.linspace(*r_limits, 151)
            dense_z = np.linspace(*z_limits, 201)
            values = interpolant(dense_r, dense_z) - float(interpolant.ev(*xpoint))
            contour = draw_flux_contours(
                ax,
                dense_r,
                dense_z,
                values.T,
                levels,
                color=color,
                linewidth=1.2,
                wall=data["limiter"],
                zorder=2 if key == "reference" else 3,
            )
            contour.set_linestyle(dash)
            separatrix = draw_flux_contours(
                ax,
                dense_r,
                dense_z,
                values.T,
                [0.0],
                color=color,
                linewidth=2.6,
                wall=data["limiter"],
                zorder=4 if key == "reference" else 5,
            )
            separatrix.set_linestyle(dash)
            first = len(ax.lines)
            draw_nulls(
                ax,
                magnetic_axis=axis,
                x_points=np.array([xpoint]),
                style=style,
                contain=data["limiter"],
            )
            if key in ("reference", "physical_image"):
                for line in ax.lines[first:]:
                    line.set_markerfacecolor("none")
                    line.set_markersize(11)
        draw_wall(ax, units=data["limiter"], color=WALL, linewidth=1.5, zorder=8)
        if index:
            center = (point + other) / 2
            ax.set_xlim(center[0] - 0.18, center[0] + 0.18)
            ax.set_ylim(center[1] - 0.18, center[1] + 0.18)
        else:
            ax.set_xlim(0.9, 2.4)
            ax.set_ylim(-1.45, 1.35)
        notes.append(
            figure.text(
                0.245 + index * 0.51,
                0.85,
                "poloidal field" if not index else "lower X-point",
                ha="center",
                fontsize=14,
            )
        )
    figure.legend(
        handles=[
            Line2D([], [], color=SOLVED, lw=3, marker="X", label="solved"),
            Line2D(
                [],
                [],
                color=REFERENCE,
                lw=2.6,
                ls="--",
                marker="P",
                markerfacecolor="none",
                label="EFIT",
            ),
            Line2D(
                [],
                [],
                color=WALL,
                lw=2.6,
                ls="-.",
                marker="D",
                markerfacecolor="none",
                label="physical image",
            ),
            Line2D([], [], color=WALL, lw=1.5, label="wall"),
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    notes.append(
        figure.text(
            0.5,
            0.035,
            f"{result['frame']}D   |   residual {result['residual']:.3g}   |   "
            f"converged {result['converged']}   |   "
            f"X-point miss {result['measured_distance_cm']:.3f} cm",
            ha="center",
            fontsize=14,
        )
    )
    name = f"frame-{result['frame']}-xpoint"
    save(figure, axes, notes, output / (name + ".svg"), previews / (name + ".png"))
    plt.close(figure)


def profiles(rows, output, previews):
    figure, axes = plt.subplots(2, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(
        left=0.095, right=0.97, bottom=0.12, top=0.84, hspace=0.65, wspace=0.3
    )
    for row, (_, result, _) in enumerate(rows):
        for col, (name, scale, label) in enumerate(
            (("p_prime", 1000, "p′ [kPa/Wb]"), ("ff_prime", 1, "FF′ [T² m²/Wb]"))
        ):
            ax = axes[row, col]
            for key, color, dash, marker, layer in [
                ("reference_profiles", REFERENCE, "--", "v", 2),
                ("solved_profiles", SOLVED, "-", "^", 4),
            ]:
                table = result[key]
                valid = np.asarray(table["reliable"], bool)
                ax.plot(
                    np.asarray(table["psi_norm"])[valid],
                    np.asarray(table[name])[valid] / scale,
                    color=color,
                    ls=dash,
                    lw=2.6,
                    marker=marker,
                    ms=4,
                    zorder=layer,
                )
            prescribed = result["input_profiles"][name]
            ax.plot(
                prescribed["coordinate"],
                np.asarray(prescribed["values"]) / scale,
                ls="none",
                marker="o",
                ms=6,
                mfc="none",
                mec=WALL,
                mew=1.2,
                zorder=6,
            )
            ax.set_xlabel("normalized flux")
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylabel(label)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
            ax.set_title(f"{result['frame']}D", fontsize=14, pad=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
    handles = [
        Line2D([], [], color=SOLVED, lw=3, marker="^", label="solved extraction"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            lw=2.6,
            ls="--",
            marker="v",
            label="EFIT extraction",
        ),
        Line2D(
            [],
            [],
            color=WALL,
            ls="none",
            marker="o",
            mfc="none",
            label="prescribed knots",
        ),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    save(
        figure,
        list(axes.ravel()),
        [],
        output / "profiles.svg",
        previews / "profiles.png",
    )
    plt.close(figure)


def attribution(rows, output, previews):
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, dpi=100, sharey=True)
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.84, wspace=0.25)
    names = [
        "profiles",
        "total_current",
        "external",
        "plasma_response",
        "boundary_shadow",
    ]
    labels = [
        "profiles",
        "current",
        "external",
        "response",
        "held nodes",
        "sum",
        "observed",
    ]
    for index, ax in enumerate(axes):
        for offset, (_, result, sensitivity), color, marker in zip(
            (-0.08, 0.08), rows, (SOLVED, WALL), ("o", "^"), strict=True
        ):
            shifts = sensitivity["path"]["shifts_m"]
            values = [100 * shifts[name][index] for name in names] + [
                100 * sensitivity["path"]["total_m"][index],
                result["measured_offset_cm"][index],
            ]
            ax.plot(
                values,
                np.arange(len(labels)) + offset,
                ls="none",
                marker=marker,
                ms=9,
                color=color,
            )
        ax.axvline(0, color=WALL, lw=1.2, ls=":", zorder=1)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlabel(
            "radial displacement [cm]" if not index else "vertical displacement [cm]"
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(6, prune="both"))
    figure.legend(
        handles=[
            Line2D([], [], color=SOLVED, marker="o", ls="none", ms=9, label="137D"),
            Line2D([], [], color=WALL, marker="^", ls="none", ms=9, label="146D"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    save(
        figure, list(axes), [], output / "attribution.svg", previews / "attribution.png"
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use("data-ink")
    rows = [read(args.measurement, frame) for frame in (137, 146)]
    for data, result, _ in rows:
        geometry(data, result, args.output, args.previews)
    profiles(rows, args.output, args.previews)
    attribution(rows, args.output, args.previews)


if __name__ == "__main__":
    main()
