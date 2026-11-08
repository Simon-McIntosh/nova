"""Draw paired terminal fields with nulls, limiter and support outlines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
from render_panels import _check_text_artists  # noqa: E402

SOLVED = DEFAULT_INK.flux_color
REFERENCE = "#929292"
SUPPORT = DEFAULT_INK.plasma_edgecolor


def save(figure, output, preview):
    """Check all text and margin labels before writing the slide canvas."""
    figure.set_size_inches(14.0, 7.2)
    figure.set_dpi(100)
    if _check_text_artists(figure, output):
        raise ValueError("figure text clips or overlaps")
    for text in figure.texts:
        point = text.get_position()
        if any(ax.get_position().contains(*point) for ax in figure.axes):
            raise ValueError("a margin label sits over the plotted field")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    tree = ET.parse(output)
    tree.getroot().set("width", "31.111cm")
    tree.getroot().set("height", "16cm")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview, dpi=100)
    print(f"FIGURE {output} text_check=passed preview={preview}", flush=True)


def render(root, branch, output, preview):
    data = [
        dict(np.load(root / arm / f"frame-146-{branch}.npz"))
        for arm in ("pseudo", "physical")
    ]
    ranges = []
    for row in data:
        ranges.extend([float(row["axis_flux"]), float(row["boundary_flux"])])
        label_spline = RectBivariateSpline(row["radius"], row["height"], row["label"])
        ranges.extend(
            [
                float(label_spline(*row["reference_axis"], grid=False)),
                float(
                    np.median(
                        label_spline(
                            row["reference_boundary"][:, 0],
                            row["reference_boundary"][:, 1],
                            grid=False,
                        )
                    )
                ),
            ]
        )
    finite = np.asarray(ranges)[np.isfinite(ranges)]
    levels = np.linspace(float(finite.min()), float(finite.max()), 11)[1:-1]
    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    for index, (arm, row) in enumerate(
        zip(("pseudo-wall control", "governed limiter"), data, strict=True)
    ):
        offset = index * 0.5
        ax = figure.add_axes([0.035 + offset, 0.22, 0.23, 0.66])
        poloidal_axes(ax)
        reference_contours = draw_flux_contours(
            ax,
            row["radius"],
            row["height"],
            row["label"].T,
            levels,
            color=REFERENCE,
            linewidth=1.2,
            wall=row["limiter"],
        )
        reference_contours.set_linestyle("--")
        draw_flux_contours(
            ax,
            row["radius"],
            row["height"],
            row["flux"].T,
            levels,
            color=SOLVED,
            linewidth=2.6,
            wall=None,
        )
        draw_wall(
            ax, row["limiter"][:, 0], row["limiter"][:, 1], color="0.2", linewidth=2.4
        )
        support = row["connected"].reshape((len(row["radius"]), len(row["height"])))
        ax.contour(
            row["radius"],
            row["height"],
            support.T.astype(float),
            levels=[0.5],
            colors=[SUPPORT],
            linewidths=1.5,
            zorder=5,
        )
        boundary = row["reference_boundary"]
        ax.plot(
            boundary[:, 0], boundary[:, 1], color=REFERENCE, lw=2.0, ls=":", zorder=6
        )
        draw_nulls(
            ax,
            row["axis"],
            np.asarray(row["x_point"]).reshape(-1, 2),
            style=DEFAULT_INK.variant(
                axis_color=SOLVED,
                xpoint_color=SOLVED,
                axis_markersize=7,
                xpoint_markersize=8,
            ),
        )
        reference_axis = row["reference_axis"]
        reference_saddle = row["reference_saddle"]
        ax.plot(
            *reference_axis,
            marker="^",
            mfc="none",
            mec=REFERENCE,
            ms=8,
            mew=1.5,
            zorder=8,
        )
        ax.plot(
            *reference_saddle,
            marker="X",
            mfc="none",
            mec=REFERENCE,
            ms=9,
            mew=1.5,
            zorder=8,
        )
        ax.set_xlim(0.8, 2.55)
        ax.set_ylim(-1.7, 1.85)
        figure.text(0.06 + offset, 0.94, arm, fontsize=14)
        note = (
            f"axis R = {row['axis'][0]:.3f} m\nZ = {row['axis'][1]:+.3f} m\n\n"
            f"support: {int(support.sum())} cells\n"
            f"material: {int(row['inside_material'].sum())} cells"
        )
        figure.text(0.28 + offset, 0.79, note, fontsize=12, va="top", linespacing=1.5)
    handles = [
        Line2D([], [], color=SOLVED, lw=2.6, label="solved flux"),
        Line2D([], [], color=REFERENCE, lw=1.2, ls="--", label="EFIT flux"),
        Line2D([], [], color="0.2", lw=2.4, label="limiter"),
        Line2D([], [], color=SUPPORT, lw=1.5, label="support outline"),
        Line2D([], [], color=REFERENCE, lw=2, ls=":", label="EFIT boundary"),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="solved axis"),
        Line2D([], [], color=SOLVED, marker="X", ls="none", label="solved saddle"),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="^",
            mfc="none",
            ls="none",
            label="EFIT axis",
        ),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="X",
            mfc="none",
            ls="none",
            label="EFIT X-point",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=12,
    )
    save(figure, output, preview)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    for branch in ("limited", "diverted"):
        render(
            args.input,
            branch,
            args.output / f"frame-146-{branch}.svg",
            args.previews / f"frame-146-{branch}.png",
        )


if __name__ == "__main__":
    main()
