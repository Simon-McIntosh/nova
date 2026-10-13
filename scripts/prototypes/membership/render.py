"""Draw measured curved legs and cell-area errors with a vessel context inset."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from tests.equilibrium import test_topology_read as fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    data = np.load(args.receipt)
    _, analytic, _, _, _ = fixture._analytic_inputs("diverted")
    fixed, truth = data["fixed"], data["truth_nulls"]
    represented = data["represented"]
    selected = represented | data["near"]
    vertices, counts = data["vertices"], data["counts"]
    patch = np.concatenate(
        [v[:n] for v, n in zip(vertices[selected], counts[selected], strict=True)]
    )
    extent = np.ptp(patch, axis=0)
    lo, hi = patch.min(axis=0) - 0.25 * extent, patch.max(axis=0) + 0.25 * extent
    plt.style.use("data-ink")
    figure = plt.figure(figsize=(14, 10))
    axes = figure.add_axes((0.28, 0.08, 0.7, 0.84))
    poloidal_axes(axes)
    reference_style = replace(
        DEFAULT_INK,
        axis_color="0.55",
        xpoint_color="0.55",
        axis_markersize=15,
        xpoint_markersize=16,
    )
    measured_style = replace(
        DEFAULT_INK,
        axis_color="0.1",
        xpoint_color="0.1",
        axis_markersize=9,
        xpoint_markersize=10,
    )
    r, z = np.linspace(lo[0], hi[0], 301), np.linspace(lo[1], hi[1], 301)
    rr, zz = np.meshgrid(r, z)
    points = jnp.asarray(np.stack((rr.ravel(), zz.ravel()), axis=-1))
    flux = np.asarray(jax.jit(jax.vmap(analytic.value))(points)).reshape(rr.shape)
    contours = draw_flux_contours(axes, r, z, flux, [0], color="0.55", linewidth=2.6)
    contours.set_linestyle("--")
    for index in np.flatnonzero(selected):
        polygon = vertices[index, : counts[index]]
        polygon = np.vstack((polygon, polygon[0]))
        axes.plot(
            *polygon.T,
            color="0.65",
            linewidth=1.2,
            linestyle="-" if represented[index] else ":",
        )
        axes.text(
            *data["centre"][index],
            f"{data['error'][index]:+.2e}",
            ha="center",
            va="center",
            fontsize=16,
            color="0.15",
        )
    parameter = np.linspace(0, 2 * np.linalg.norm(extent), 401)
    for direction, curvature, cubic in zip(
        data["direction"], data["curvature"], data["cubic"], strict=True
    ):
        curve = (
            fixed[-1]
            + parameter[:, None] * direction
            + parameter[:, None] ** 2 * curvature
            + parameter[:, None] ** 3 * cubic
        )
        axes.plot(*curve.T, color="0.1", linewidth=3)
    draw_wall(axes, *data["wall"].T, color="0.4", linewidth=2.4)
    draw_nulls(axes, truth[0], truth[1:], style=reference_style)
    draw_nulls(axes, fixed[0], fixed[1:], style=measured_style)
    axes.set_xlim(lo[0], hi[0])
    axes.set_ylim(lo[1], hi[1])
    axes.text(
        0.5,
        -0.02,
        "Signed area error / median cell area",
        transform=axes.transAxes,
        ha="center",
        fontsize=20,
    )
    axes.text(
        0.02,
        1.08,
        "Curved legs: solid\nClosed-form separatrix: dashed",
        transform=axes.transAxes,
        va="top",
        fontsize=20,
    )
    context = figure.add_axes((0.01, 0.23, 0.24, 0.6))
    poloidal_axes(context)
    draw_wall(context, *data["wall"].T, color="0.4", linewidth=2.4)
    oracle, _, _, _, _ = fixture._analytic_inputs("diverted")
    boundary = oracle.separatrix(1441)
    context.plot(*boundary.T, color="0.55", linestyle="--", linewidth=2.6)
    draw_nulls(context, truth[0], truth[1:], style=reference_style)
    draw_nulls(context, fixed[0], fixed[1:], style=measured_style)
    context.plot(
        [lo[0], hi[0], hi[0], lo[0], lo[0]],
        [lo[1], lo[1], hi[1], hi[1], lo[1]],
        color="0.5",
        linewidth=1.2,
        linestyle=":",
    )
    context.text(
        0.5,
        -0.18,
        "Wall and both null sets\nGrey: closed form\nBlack: cached Bell",
        transform=context.transAxes,
        ha="center",
        fontsize=18,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
