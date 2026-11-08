"""Draw native flux and region-restricted residual line contours."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from style import BOUNDARY, FLUX, NEGATIVE, POSITIVE, validate_palette

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_forward_convergence"))
from render_panels import _check_text_artists  # noqa: E402


def save(figure, axes, notes, output, preview):
    """Refuse clipped, colliding or in-field labels before writing one SVG."""
    problems = _check_text_artists(figure, output)
    renderer = figure.canvas.get_renderer()
    for note in notes:
        box = note.get_window_extent(renderer)
        if any(box.overlaps(ax.get_window_extent(renderer)) for ax in axes):
            problems.append(f"label over data: {note.get_text()}")
    if problems:
        raise ValueError("; ".join(problems))
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    text = output.read_text().replace(
        'width="1008pt" height="518.4pt"', 'width="31.1cm" height="16cm"'
    )
    output.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    figure.savefig(preview, dpi=100)
    print(
        f"FIGURE {output.name}: text collisions=0; labels over data=0; "
        f"bytes={output.stat().st_size}"
    )
    assert output.stat().st_size <= 300_000


def render(source, output, preview):
    record = json.loads((source / "receipt.json").read_text())
    with np.load(source / "terminal.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    assert record["storage_order"] == "radius_height"
    radius, height = data["radius"], data["height"]
    flux, label = data["flux"], data["label"]
    assert flux.shape == label.shape == (radius.size, height.size)
    residual = data["residual"][: flux.size].reshape(flux.shape)
    support, band = data["connected"], data["boundary_band"]
    regions = (
        ("support interior", support & ~band),
        ("boundary band", band),
        ("outside plasma", ~support & ~band),
    )
    reference_surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(label.T)
    )
    starts = np.vstack((data["reference_axis"], data["reference_saddle"]))
    roots = polish_stationary_points(
        reference_surface, jnp.asarray(starts), jnp.ones(2, bool)
    )
    admitted = np.asarray(roots["converged"], bool)
    points = np.where(admitted[:, None], np.asarray(roots["position_rz"]), starts)
    print(
        f"REFERENCE {source.name}: polished={admitted.tolist()} "
        f"positions={points.tolist()}"
    )
    low = min(float(flux.min()), float(label.min()))
    high = max(float(flux.max()), float(label.max()))
    flux_levels = np.linspace(low, high, 11)[1:-1]
    extent = float(np.max(np.abs(residual)))
    display_scale, display_unit = (1e3, "mWb") if extent > 1e-6 else (1e18, "aWb")
    residual_levels = extent * np.array([-0.8, -0.5, -0.2, 0.2, 0.5, 0.8])
    axis_separation = float(np.linalg.norm(data["axis"][:2] - points[0]))
    print(
        f"DISPLAY {source.name}: max|r|={extent:.12g} Wb; "
        f"levels={residual_levels.tolist()}; axis separation={axis_separation:.9g} m"
    )
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 3, figsize=(14, 7.2), dpi=100)
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.17, top=0.79, wspace=0.15)
    notes = []
    reference_style = replace(
        DEFAULT_INK, axis_color=FLUX, xpoint_color=FLUX, axis_markersize=9
    )
    for ax, (name, region) in zip(axes, regions, strict=True):
        poloidal_axes(ax)
        ax.set_aspect("equal")
        ax.set_xlim(radius[0], radius[-1])
        ax.set_ylim(height[0], height[-1])
        ax.add_patch(
            Rectangle(
                (radius[0], height[0]),
                radius[-1] - radius[0],
                height[-1] - height[0],
                fill=False,
                edgecolor=FLUX,
                linewidth=0.7,
                clip_on=False,
                zorder=0,
            )
        )
        ax.contour(
            radius,
            height,
            label.T,
            levels=flux_levels,
            colors=FLUX,
            linewidths=0.75,
            linestyles="dashed",
            zorder=1,
        )
        ax.contour(
            radius,
            height,
            flux.T,
            levels=flux_levels,
            colors=FLUX,
            linewidths=1.0,
            linestyles="solid",
            zorder=2,
        )
        ax.contour(
            radius,
            height,
            np.ma.masked_where(~region, residual).T,
            levels=residual_levels,
            colors=[NEGATIVE] * 3 + [POSITIVE] * 3,
            linewidths=2.0,
            linestyles="solid",
            zorder=4,
        )
        ax.contour(
            radius,
            height,
            region.T.astype(float),
            levels=[0.5],
            colors=BOUNDARY,
            linewidths=0.7,
            linestyles="dotted",
            zorder=3,
        )
        draw_wall(ax, units=data["limiter"], linewidth=1.4)
        boundary = data["label_boundary"]
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=BOUNDARY,
            linestyle="--",
            linewidth=1.8,
        )
        before = len(ax.lines)
        draw_nulls(
            ax,
            magnetic_axis=points[0],
            other_x_points=points[1][None],
            style=reference_style,
        )
        for line in ax.lines[before:]:
            line.set_markerfacecolor("none")
        draw_nulls(
            ax, magnetic_axis=data["axis"][:2], other_x_points=data["saddle"][:2][None]
        )
        box = ax.get_position()
        notes.append(
            figure.text((box.x0 + box.x1) / 2, 0.82, name, ha="center", fontsize=16)
        )
        maximum = np.max(np.abs(residual[region]), initial=0)
        notes.append(
            figure.text(
                (box.x0 + box.x1) / 2,
                0.105,
                f"{int(region.sum())} nodes; "
                f"max |r| {maximum * display_scale:.2f} {display_unit}",
                ha="center",
                fontsize=14,
            )
        )
    handles = [
        Line2D([], [], color=POSITIVE, lw=2, label="residual > 0"),
        Line2D([], [], color=NEGATIVE, lw=2, label="residual < 0"),
        Line2D([], [], color=FLUX, lw=1, label="solved flux"),
        Line2D([], [], color=FLUX, lw=0.75, ls="--", label="EFIT flux"),
        Line2D([], [], color=BOUNDARY, lw=1.8, ls="--", label="EFIT boundary"),
        Line2D([], [], color=DEFAULT_INK.wall_color, lw=1.4, label="limiter"),
        Line2D([], [], color=BOUNDARY, lw=0.7, ls=":", label="region edge"),
        Line2D([], [], color=FLUX, lw=0.7, label="grid boundary"),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.axis_color,
            marker="^",
            ls="none",
            label="solved axis",
        ),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.xpoint_color,
            marker="X",
            mfc="none",
            ls="none",
            label="recorded saddle",
        ),
        Line2D(
            [],
            [],
            color=FLUX,
            marker="^",
            mfc="none",
            ls="none",
            label="EFIT axis",
            markersize=9,
        ),
        Line2D(
            [],
            [],
            color=FLUX,
            marker="X",
            mfc="none",
            ls="none",
            label="EFIT X-point",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    notes.extend(
        [
            figure.text(
                0.5,
                0.055,
                "residual levels: ±{0.2, 0.5, 0.8} × max|r|; "
                f"max|r| = {extent:.6g} Wb",
                ha="center",
                fontsize=14,
            ),
            figure.text(
                0.5,
                0.02,
                f"solved–EFIT axis separation = {axis_separation:.6g} m",
                ha="center",
                fontsize=14,
            ),
        ]
    )
    save(figure, axes, notes, output, preview)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    validate_palette()
    for source in sorted(args.input.glob("*/receipt.json")):
        name = source.parent.name
        render(
            source.parent, args.output / f"{name}.svg", args.previews / f"{name}.png"
        )


if __name__ == "__main__":
    main()
