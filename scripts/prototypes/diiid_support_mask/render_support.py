"""Draw captured cell masks over their native radius-height flux fields."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath

from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import _lattice_cells
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
from render_panels import _check_text_artists  # noqa: E402


def render(archive, output, previews):
    record = json.loads(archive.with_suffix(".json").read_text())
    with np.load(archive) as data:
        arrays = {key: data[key] for key in data.files}
    radius, height = arrays["radius"], arrays["height"]
    flux = arrays["flux"]
    assert flux.shape == (radius.size, height.size)
    cells = np.asarray(_lattice_cells(FluxLattice(radius, height)))
    material, closed = arrays["inside_material"], arrays["closed"]
    stages = (
        ("material", material),
        ("flux comparison", closed),
        ("material ∩ closed", material & closed),
        ("axis component", arrays["connected"]),
    )
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 4, figsize=(14, 7.2), dpi=100)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.80, wspace=0.08)
    style = DEFAULT_INK
    reference = replace(style, axis_color="#929292", xpoint_color="#929292")
    low, high = sorted((record["axis_flux_wb"], record["boundary_flux_wb"]))
    levels = np.linspace(low, high, 9)[1:-1]
    for ax, (name, mask) in zip(axes, stages, strict=True):
        poloidal_axes(ax)
        ax.set_xlim(radius[0], radius[-1])
        ax.set_ylim(height[0], height[-1])
        if np.any(mask):
            ax.add_patch(
                PathPatch(
                    PlotPath.make_compound_path_from_polys(cells[mask]),
                    facecolor="none",
                    edgecolor=style.plasma_edgecolor,
                    linewidth=0.45,
                    zorder=1,
                )
            )
        ax.contour(
            radius,
            height,
            arrays["label"].T,
            levels=levels,
            colors="#929292",
            linewidths=0.7,
            linestyles="dashed",
            zorder=2,
        )
        ax.contour(
            radius,
            height,
            flux.T,
            levels=levels,
            colors=style.flux_color,
            linestyles="solid",
            linewidths=0.9,
            zorder=3,
        )
        ax.contour(
            radius,
            height,
            flux.T,
            levels=[record["boundary_flux_wb"]],
            colors=style.separatrix_color,
            linestyles="solid",
            linewidths=1.5,
            zorder=4,
        )
        draw_wall(ax, units=arrays["limiter"], linewidth=1.2)
        before = len(ax.lines)
        draw_nulls(
            ax,
            magnetic_axis=arrays["reference_axis"],
            other_x_points=arrays["reference_saddle"][None],
            style=reference,
        )
        for line in ax.lines[before:]:
            line.set_markerfacecolor("none")
        draw_nulls(
            ax,
            magnetic_axis=arrays["data_o"][:2],
            x_points=(
                arrays["data_x"][:2][None] if record["boundary_is_xpoint"] else None
            ),
            other_x_points=(
                None if record["boundary_is_xpoint"] else arrays["data_x"][:2][None]
            ),
        )
        box = ax.get_position()
        figure.text(
            (box.x0 + box.x1) / 2,
            0.12,
            f"{name}\n{int(mask.sum())} cells",
            ha="center",
            va="top",
            fontsize=14,
        )
    handles = [
        Line2D([], [], color=style.plasma_edgecolor, lw=1.5, label="cell outline"),
        Line2D([], [], color=style.flux_color, lw=2.6, label="solved flux"),
        Line2D([], [], color="#929292", lw=2.6, ls="--", label="EFIT flux"),
        Line2D([], [], color=style.separatrix_color, lw=2.4, label="boundary level"),
        Line2D([], [], color="black", lw=1.2, label="limiter"),
        Line2D(
            [], [], color=style.axis_color, marker="^", ls="none", label="solved axis"
        ),
        Line2D(
            [],
            [],
            color=style.xpoint_color,
            marker="X",
            mfc=(style.xpoint_color if record["boundary_is_xpoint"] else "none"),
            ls="none",
            label=(
                "admitted saddle" if record["boundary_is_xpoint"] else "recorded saddle"
            ),
        ),
        Line2D(
            [],
            [],
            color="#929292",
            marker="^",
            mfc="none",
            ls="none",
            label="EFIT axis",
        ),
        Line2D(
            [],
            [],
            color="#929292",
            marker="X",
            mfc="none",
            ls="none",
            label="EFIT boundary low point",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.99),
    )
    assert not _check_text_artists(figure, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    text = output.read_text()
    import re

    text = re.sub(
        r'width="[^"]+" height="[^"]+"', 'width="31.111cm" height="16cm"', text, count=1
    )

    def compact(match):
        coordinates = re.sub(
            r"-?\d+\.\d+", lambda value: f"{float(value[0]):.2f}", match[1]
        )
        return 'd="' + re.sub(r"\s+", " ", coordinates).strip() + '"'

    text = re.sub(r'(?<!\w)d="([^"]+)"', compact, text)
    text = re.sub(r">\s+<", "><", text)
    output.write_text(text)
    previews.mkdir(parents=True, exist_ok=True)
    figure.savefig(previews / output.with_suffix(".png").name)
    plt.close(figure)
    print(f"RENDERED {output.name}; text bounds and overlap check passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    for archive in sorted(args.input.glob("frame-*.npz")):
        render(archive, args.output / archive.with_suffix(".svg").name, args.previews)


if __name__ == "__main__":
    main()
