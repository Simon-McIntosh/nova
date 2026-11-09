"""Draw paired terminal fields with nulls, limiter and support outlines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.jax.config import configure_dtypes
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    inside_wall_units,
)

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))
from render_panels import _check_text_artists  # noqa: E402

SOLVED = DEFAULT_INK.flux_color
REFERENCE = "#929292"
SUPPORT = DEFAULT_INK.plasma_edgecolor


def reference_nulls(row):
    """Polish the stored reference seeds on the reference field's own spline."""
    configure_dtypes()
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.linalg.tensor_spline import fit_tensor_spline

    assert jax.config.jax_enable_x64
    surface = fit_tensor_spline(
        jnp.asarray(row["radius"]),
        jnp.asarray(row["height"]),
        jnp.asarray(row["label"].T),
    )
    seeds = np.vstack((row["reference_axis"], row["reference_saddle"]))
    result = polish_stationary_points(surface, jnp.asarray(seeds), jnp.ones(2, bool))
    assert np.asarray(result["converged"]).all()
    np.testing.assert_array_equal(np.asarray(result["hessian_type"]), [1, -1])
    positions = np.asarray(result["position_rz"])
    print("REFERENCE_NULLS", positions.tolist(), flush=True)
    return positions


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
        receipt = json.loads(
            (
                root
                / ("pseudo" if index == 0 else "physical")
                / f"frame-146-{branch}.json"
            ).read_text()
        )
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
        from nova.equilibrium.forward import FluxLattice, _lattice_cells

        support = row["connected"].astype(bool)
        lattice = FluxLattice(row["radius"], row["height"])
        cells = [
            np.vstack((cell, cell[:1]))
            for cell, keep in zip(_lattice_cells(lattice), support, strict=True)
            if keep
        ]
        ax.add_collection(
            LineCollection(cells, colors=SUPPORT, linewidths=0.6, zorder=5)
        )
        boundary = row["reference_boundary"]
        ax.plot(
            boundary[:, 0], boundary[:, 1], color=REFERENCE, lw=2.0, ls=":", zorder=6
        )
        terminal_boundary = row["surface"]
        if len(terminal_boundary):
            ax.plot(
                terminal_boundary[:, 0],
                terminal_boundary[:, 1],
                color=SOLVED,
                linewidth=3.0,
                zorder=7,
            )
        axis_inside = bool(inside_wall_units(row["axis"][None], row["limiter"])[0])
        draw_nulls(
            ax,
            row["axis"] if axis_inside else None,
            other_x_points=np.asarray(row["x_point"]).reshape(-1, 2),
            style=DEFAULT_INK.variant(
                axis_color=SOLVED,
                xpoint_color=SOLVED,
                axis_markersize=7,
                xpoint_markersize=12,
            ),
        )
        if not axis_inside:
            ax.plot(*row["axis"], marker="^", mfc="none", mec=SOLVED, ms=8, zorder=9)
        reference_axis, reference_saddle = reference_nulls(row)
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
            ms=7,
            mew=1.5,
            zorder=8,
        )
        ax.set_xlim(0.8, 2.55)
        ax.set_ylim(-1.7, 1.85)
        figure.text(0.06 + offset, 0.94, arm, fontsize=14)
        note = (
            f"axis R = {row['axis'][0]:.3f} m\nZ = {row['axis'][1]:+.3f} m\n\n"
            f"support: {int(support.sum())} cells\n"
            f"material: {int(row['inside_material'].sum())} cells\n\n"
            f"residual: {receipt['residual']:.3g}\n"
            f"converged: {str(receipt['converged']).lower()}\n"
            f"host class: {receipt['host_boundary_read_class']}"
        )
        figure.text(0.28 + offset, 0.79, note, fontsize=12, va="top", linespacing=1.5)
    handles = [
        Line2D([], [], color=SOLVED, lw=2.6, label="terminal flux"),
        Line2D([], [], color=REFERENCE, lw=1.2, ls="--", label="EFIT flux"),
        Line2D([], [], color="0.2", lw=2.4, label="limiter"),
        Line2D([], [], color=SUPPORT, lw=0.6, label="support cells"),
        Line2D([], [], color=REFERENCE, lw=2, ls=":", label="EFIT boundary"),
        Line2D([], [], color=SOLVED, marker="^", ls="none", label="terminal axis"),
        Line2D(
            [],
            [],
            color=SOLVED,
            marker="X",
            mfc="none",
            ls="none",
            label="terminal saddle",
        ),
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
