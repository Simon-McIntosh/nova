"""Draw paired flux contours and enlarged X-point neighborhoods."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

ROOT = Path(__file__).resolve().parents[3]
STYLE_DIR = ROOT / "scripts/prototypes/diiid_residual_plateau"
sys.path.insert(0, str(STYLE_DIR))
from style import FLUX, POSITIVE, validate_palette  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "capture_renderer", STYLE_DIR / "render.py"
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def load(path):
    with np.load(path / "terminal.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    record = json.loads((path / "receipt.json").read_text())
    assert record["storage_order"] == "radius_height"
    radius, height = arrays["radius"], arrays["height"]
    assert arrays["flux"].shape == (radius.size, height.size)
    return arrays, record


def reference_nulls(data):
    field = fit_tensor_spline(
        jnp.asarray(data["radius"]),
        jnp.asarray(data["height"]),
        jnp.asarray(data["label"].T),
    )
    starts = np.vstack((data["reference_axis"], data["reference_saddle"]))
    roots = polish_stationary_points(field, jnp.asarray(starts), jnp.ones(2, bool))
    admitted = np.asarray(roots["converged"], bool)
    return np.where(admitted[:, None], np.asarray(roots["position_rz"]), starts)


def paint(ax, data, levels, reference, xlim=None, ylim=None):
    poloidal_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(xlim or (data["radius"][0], data["radius"][-1]))
    ax.set_ylim(ylim or (data["height"][0], data["height"][-1]))
    ax.contour(
        data["radius"],
        data["height"],
        data["label"].T,
        levels=levels,
        colors=FLUX,
        linewidths=1.2,
        linestyles="--",
        zorder=1,
    )
    boundary = data["label_boundary"]
    ax.plot(*boundary.T, color=FLUX, ls=":", lw=1.4, zorder=2)
    draw_wall(ax, units=data["limiter"], linewidth=1.4)
    ax.contour(
        data["radius"],
        data["height"],
        data["flux"].T,
        levels=levels,
        colors=POSITIVE,
        linewidths=1.8,
        zorder=4,
    )
    reference_style = replace(
        DEFAULT_INK, axis_color=FLUX, xpoint_color=FLUX, axis_markersize=10
    )
    start = len(ax.lines)
    draw_nulls(
        ax,
        magnetic_axis=reference[0],
        other_x_points=reference[1][None],
        style=reference_style,
    )
    for line in ax.lines[start:]:
        line.set_markerfacecolor("none")
        line.set_zorder(5)
    admitted = np.allclose(data["saddle"][:3], data["boundary"][:3], atol=1e-10)
    kwargs = {"x_points" if admitted else "other_x_points": data["saddle"][:2][None]}
    start = len(ax.lines)
    draw_nulls(
        ax,
        magnetic_axis=data["axis"][:2],
        style=replace(DEFAULT_INK, axis_color=POSITIVE, xpoint_color=POSITIVE),
        **kwargs,
    )
    for line in ax.lines[start:]:
        line.set_zorder(6)


def render(source, name, output, previews):
    pair = [load(source / arm / name) for arm in ("base", "head")]
    levels = np.linspace(
        min(float(d[key].min()) for d, _ in pair for key in ("flux", "label")),
        max(float(d[key].max()) for d, _ in pair for key in ("flux", "label")),
        11,
    )[1:-1]
    reference = reference_nulls(pair[0][0])
    feature = reference[1]
    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    # Separate inset axes leave an actual margin for labels and the common key.
    axes = [
        figure.add_axes(rect)
        for rect in (
            (0.02, 0.12, 0.28, 0.64),
            (0.31, 0.25, 0.16, 0.38),
            (0.52, 0.12, 0.28, 0.64),
            (0.81, 0.25, 0.16, 0.38),
        )
    ]
    notes = []
    for offset, (arm, (data, record)) in enumerate(
        zip(("before", "after"), pair, strict=True)
    ):
        main, inset = axes[2 * offset : 2 * offset + 2]
        paint(main, data, levels, reference)
        paint(
            inset,
            data,
            levels,
            reference,
            xlim=(feature[0] - 0.25, feature[0] + 0.25),
            ylim=(feature[1] - 0.25, feature[1] + 0.25),
        )
        notes.append(
            figure.text(0.25 + 0.5 * offset, 0.81, arm, ha="center", fontsize=16)
        )
        notes.append(
            figure.text(
                0.39 + 0.5 * offset, 0.68, "X-point zoom", ha="center", fontsize=14
            )
        )
        notes.append(
            figure.text(
                0.25 + 0.5 * offset,
                0.075,
                f"residual {record['residual']:.5g}; "
                f"converged {str(record['converged']).lower()}",
                ha="center",
                fontsize=14,
            )
        )
        notes.append(
            figure.text(
                0.25 + 0.5 * offset,
                0.025,
                f"X-point distance {record['x_point_distance_m']:.3f} m",
                ha="center",
                fontsize=14,
            )
        )
    handles = [
        Line2D([], [], color=POSITIVE, lw=1.8, label="solved flux"),
        Line2D([], [], color=FLUX, ls="--", lw=1.2, label="EFIT flux"),
        Line2D([], [], color=FLUX, ls=":", lw=1.4, label="EFIT boundary"),
        Line2D([], [], color=DEFAULT_INK.wall_color, lw=1.4, label="limiter"),
        Line2D([], [], color=POSITIVE, marker="^", ls="none", label="solved axis"),
        Line2D(
            [],
            [],
            color=POSITIVE,
            marker="X",
            ls="none",
            mfc="none",
            label="solved saddle",
        ),
        Line2D(
            [], [], color=FLUX, marker="^", ls="none", mfc="none", label="EFIT axis"
        ),
        Line2D(
            [], [], color=FLUX, marker="X", ls="none", mfc="none", label="EFIT X-point"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    shared.save(
        figure, axes, notes, output / (name + ".svg"), previews / (name + ".png")
    )
    print(
        json.dumps(
            dict(
                case=name, levels_wb=levels.tolist(), reference_nulls=reference.tolist()
            )
        )
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    validate_palette()
    plt.rcParams["svg.fonttype"] = "none"
    for name in ("frame-146-diverted", "frame-137-diverted"):
        render(args.input, name, args.output, args.previews)


if __name__ == "__main__":
    main()
