"""Render a poloidal flux-contour panel for one active-set trip.

The trip's own solved field (saved by the per-trip driver as
``frame-<n>-trip-<k>.npz``) is drawn as line contours against the EFIT label at
the same physical levels, with both null sets, the governed limiter ring, the
axis-connected bounding surface the connectivity owner reads, and the measured
axis position.  A residual strip below the panel carries the relative residual
at every trip the frame reached.

The geometry is not derived here.  The bounding surface comes from
``nova.equilibrium.connectivity_boundary.host_boundary_read`` on the trip's flux
grid and limiter units, and the stationary points come from
``nova.equilibrium.flux_surface_connectivity.polish_stationary_points`` on the
field's own tensor spline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import jax.numpy as jnp

from nova.equilibrium.connectivity_boundary import host_boundary_read
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    inside_wall_units,
)

LEVEL_COUNT = 9


def _oriented(flux: np.ndarray, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    array = np.asarray(flux, dtype=float)
    if array.shape == (len(height), len(radius)):
        return array
    return array.T


def _point_flux(radius, height, flux) -> RegularGridInterpolator:
    """Return genuine bilinear interpolation of the map at arbitrary points."""

    return RegularGridInterpolator(
        (np.asarray(height, dtype=float), np.asarray(radius, dtype=float)),
        np.asarray(flux, dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _stationary(surface, seeds, hessian_type):
    """Polish stationary candidates on a field's own tensor spline."""

    seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)
    if not seeds.size:
        return np.empty((0, 2)), np.empty(0)
    polished = polish_stationary_points(
        surface,
        jnp.asarray(seeds),
        jnp.ones((seeds.shape[0],), dtype=bool),
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2)
    value = np.asarray(polished["value"], dtype=float).reshape(-1)
    converged = np.asarray(polished["converged"], dtype=bool).reshape(-1)
    kind = np.asarray(polished["hessian_type"]).reshape(-1)
    keep = converged & (kind == hessian_type) & np.all(np.isfinite(position), axis=1)
    return position[keep], value[keep]


def _candidates(magnitude, radius, height, count=6):
    """Return grid-scale stationary seeds for the polish to refine.

    These are seeds only: every returned coordinate is moved onto the field's
    true stationary point by :func:`_stationary` before it is drawn.
    """

    margin = 2
    seeds = []
    for i in range(margin, len(height) - margin):
        for j in range(margin, len(radius) - margin):
            block = magnitude[i - 1 : i + 2, j - 1 : j + 2]
            if magnitude[i, j] <= block.min():
                seeds.append((magnitude[i, j], radius[j], height[i]))
    seeds.sort()
    return np.asarray([[r, z] for _, r, z in seeds[:count]], dtype=float)


def _gradient_magnitude(radius, height, flux):
    dz, dr = np.gradient(flux, height, radius)
    return np.abs(dr) + np.abs(dz)


def _connected_surface(axes, radius, height, flux, level, axis, colour, width):
    """Draw only the level ring that encloses the axis.

    The full level set also carries lobes that bound no plasma; drawing those
    reads as a boundary the state does not have, so each ring is tested for
    containment of the measured axis and non-enclosing rings are dropped.
    """

    if not np.isfinite(level) or axis is None:
        return False
    contour = axes.contour(radius, height, flux, levels=[level], colors="none")
    kept = False
    for collection in contour.allsegs[0]:
        points = np.asarray(collection, dtype=float)
        if points.shape[0] < 4:
            continue
        ring = plt.matplotlib.path.Path(points)
        if not ring.contains_point((axis[0], axis[1])):
            continue
        axes.plot(
            points[:, 0],
            points[:, 1],
            color=colour,
            linewidth=width,
            zorder=DEFAULT_INK.zorder_separatrix,
        )
        kept = True
    contour.remove()
    return kept


def render_trip(frame, trip, archive, rows, out) -> dict:
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = _oriented(np.asarray(data["flux"], dtype=float), radius, height)
        label = _oriented(np.asarray(data["label"], dtype=float), radius, height)
        limiter = np.asarray(data["limiter"], dtype=float)
        wall = np.asarray(data["wall"], dtype=float)
        label_boundary = np.asarray(data["label_boundary"], dtype=float)

    row = next(r for r in rows if r["frame"] == frame and r["trip"] == trip)
    axis = np.asarray(row["axis_rz_m"], dtype=float) if row.get("axis_rz_m") else None

    mesh_r, mesh_z = np.meshgrid(radius, height)
    points = np.c_[mesh_r.ravel(), mesh_z.ravel()]
    inside = inside_wall_units(points, limiter).reshape(len(height), len(radius))
    grid = SimpleNamespace(
        rg=radius,
        zg=height,
        inside_limiter=inside,
        limiter_r=limiter[:, 0],
        limiter_z=limiter[:, 1],
    )

    boundary = None
    if axis is not None and inside[
        int(np.argmin(np.abs(height - axis[1]))),
        int(np.argmin(np.abs(radius - axis[0]))),
    ]:
        boundary = host_boundary_read(flux, grid, tuple(axis))
    bounding = float(boundary.psi_bnd) if boundary is not None and boundary.found else np.nan

    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    seeds = _candidates(_gradient_magnitude(radius, height, flux), radius, height)
    saddles, saddle_values = _stationary(surface, seeds, -1)
    extremum, _ = _stationary(surface, [axis] if axis is not None else [], 1)
    label_surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(label)
    )
    label_seeds = _candidates(_gradient_magnitude(radius, height, label), radius, height)
    label_saddles, _ = _stationary(label_surface, label_seeds, -1)
    label_axis = _extremum(label, radius, height, label_boundary)
    label_axis_points, _ = (
        _stationary(label_surface, [label_axis], 1)
        if label_axis is not None
        else (np.empty((0, 2)), None)
    )

    solved_value = _point_flux(radius, height, flux)
    label_value = _point_flux(radius, height, label)
    axis_value = float(solved_value(axis)[0]) if axis is not None else np.nan
    label_axis_value = (
        float(label_value(label_axis)[0]) if label_axis is not None else np.nan
    )
    label_limit = float(np.nanmin(label_value(limiter))) if len(limiter) else np.nan
    edge = bounding if np.isfinite(bounding) else label_limit
    levels = _shared_levels([(axis_value, edge), (label_axis_value, label_limit)])
    if not len(levels):
        levels = np.linspace(
            float(np.nanmin(flux)), float(np.nanmax(flux)), LEVEL_COUNT + 2
        )[1:-1]

    figure = plt.figure(figsize=(14.0, 7.2))
    layout = figure.add_gridspec(2, 1, height_ratios=[6.4, 0.9], hspace=0.10)
    axes = figure.add_subplot(layout[0])
    strip = figure.add_subplot(layout[1])
    figure.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.11)
    poloidal_axes(axes)

    draw_flux_contours(
        axes, radius, height, flux, levels, color=DEFAULT_INK.flux_color, wall=wall
    )
    draw_flux_contours(
        axes, radius, height, label, levels, color="#9a9a9a", linewidth=1.0, wall=None
    )
    _connected_surface(
        axes,
        radius,
        height,
        flux,
        bounding,
        axis,
        DEFAULT_INK.separatrix_color,
        DEFAULT_INK.separatrix_linewidth,
    )
    draw_wall(axes, radius=limiter[:, 0], height=limiter[:, 1], close=True)

    hollow = dataclasses.replace(
        DEFAULT_INK,
        axis_color="#606060",
        xpoint_color="#606060",
        axis_marker="o",
        xpoint_marker="o",
    )
    draw_nulls(
        axes,
        magnetic_axis=axis,
        x_points=saddles if len(saddles) else None,
    )
    if len(label_axis_points):
        draw_nulls(
            axes,
            magnetic_axis=None,
            x_points=None,
            style=hollow,
        )
        for point in label_axis_points:
            axes.plot(
                point[0],
                point[1],
                marker="^",
                markersize=DEFAULT_INK.axis_markersize,
                markerfacecolor="none",
                markeredgecolor="#606060",
                linestyle="none",
            )
        for point in label_saddles:
            axes.plot(
                point[0],
                point[1],
                marker="X",
                markersize=DEFAULT_INK.xpoint_markersize,
                markerfacecolor="none",
                markeredgecolor="#606060",
                linestyle="none",
            )
    if len(label_boundary):
        axes.plot(
            label_boundary[:, 0],
            label_boundary[:, 1],
            color="#555555",
            linewidth=1.0,
            linestyle="--",
            zorder=DEFAULT_INK.zorder_separatrix,
        )

    handles = [
        plt.Line2D([], [], color=DEFAULT_INK.flux_color, linewidth=2.6, label="solved flux"),
        plt.Line2D([], [], color="#9a9a9a", linewidth=2.0, label="EFIT flux"),
        plt.Line2D([], [], color=DEFAULT_INK.separatrix_color, linewidth=2.6, label="bounding surface"),
        plt.Line2D([], [], color="black", linewidth=2.0, label="limiter"),
        plt.Line2D([], [], color="#606060", linewidth=1.6, linestyle="--", label="EFIT boundary"),
        plt.Line2D([], [], marker="^", linestyle="none", color=DEFAULT_INK.axis_color, markersize=7, label="solved axis"),
        plt.Line2D([], [], marker="X", linestyle="none", color=DEFAULT_INK.xpoint_color, markersize=7, label="solved X-point"),
        plt.Line2D([], [], marker="^", linestyle="none", markerfacecolor="none", markeredgecolor="#606060", markersize=7, label="EFIT axis"),
        plt.Line2D([], [], marker="X", linestyle="none", markerfacecolor="none", markeredgecolor="#606060", markersize=7, label="EFIT X-point"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.5, 0.0, 0.5),
        ncol=5,
        frameon=False,
        fontsize=12,
    )

    frame_rows = sorted([r for r in rows if r["frame"] == frame], key=lambda r: r["trip"])
    trips = [r["trip"] for r in frame_rows]
    residuals = [r.get("residual", np.nan) for r in frame_rows]
    strip.plot(trips, [r if r is not None else np.nan for r in residuals], marker="o", color=DEFAULT_INK.flux_color, linewidth=1.6)
    strip.plot([trip], [row.get("residual", np.nan)], marker="o", markersize=8, color=DEFAULT_INK.separatrix_color)
    strip.set_xlim(min(trips) - 0.3, max(trips) + 0.3)
    strip.set_xticks(trips)
    strip.set_xlabel("active-set trip", fontsize=14)
    strip.set_ylabel("relative residual", fontsize=14)
    strip.tick_params(labelsize=12)
    strip.spines[["top", "right"]].set_visible(False)
    finite = [v for v in residuals if v is not None and np.isfinite(v)]
    strip.set_ylim(0, max(finite + [0.1]) * 1.2)

    axes.set_aspect("equal")
    axes.set_xlim(float(np.nanmin(radius)), float(np.nanmax(radius)))
    axes.set_ylim(float(np.nanmin(height)), float(np.nanmax(height)))

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, format="svg")
    figure.savefig(out.with_suffix(".check.png"), dpi=110)
    plt.close(figure)

    return {
        "frame": frame,
        "trip": trip,
        "axis_rz_m": None if axis is None else axis.tolist(),
        "axis_inside_limiter": None
        if axis is None
        else bool(inside_wall_units(axis[None, :], limiter)[0]),
        "bounding_flux": None if not np.isfinite(bounding) else bounding,
        "boundary_found": None if boundary is None else bool(boundary.found),
        "boundary_class": None
        if boundary is None
        else ("diverted" if boundary.is_diverted else "limited"),
        "saddles_rz_m": saddles.tolist(),
        "saddle_flux": saddle_values.tolist(),
        "residual": row.get("residual"),
        "svg": str(out),
    }


def _extremum(flux, radius, height, boundary):
    mesh = np.meshgrid(radius, height)
    points = np.c_[mesh[0].ravel(), mesh[1].ravel()]
    inside = inside_wall_units(points, np.asarray(boundary, dtype=float))
    flat = np.asarray(flux, dtype=float).ravel().copy()
    flat[~inside] = np.nan
    if not np.isfinite(flat).any():
        return None
    upper = abs(np.nanmax(flat)) > abs(np.nanmin(flat))
    index = int(np.nanargmax(flat)) if upper else int(np.nanargmin(flat))
    return points[index]


def _shared_levels(ranges):
    finite = [
        (float(a), float(b)) for a, b in ranges if np.isfinite(a) and np.isfinite(b)
    ]
    if not finite:
        return np.empty(0)
    low = min(min(pair) for pair in finite)
    high = max(max(pair) for pair in finite)
    if np.isclose(low, high):
        return np.empty(0)
    return np.linspace(low, high, LEVEL_COUNT + 2)[1:-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--figures", dest="figures", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["146:1", "146:2", "137:2"])
    arguments = parser.parse_args()
    rows = json.loads(arguments.rows.read_text())["rows"]
    records = []
    for item in arguments.panels:
        frame, trip = (int(part) for part in item.split(":"))
        archive = arguments.run_dir / f"frame-{frame}-trip-{trip}.npz"
        out = arguments.figures / f"frame-{frame}-trip-{trip}-flux-contours.svg"
        records.append(render_trip(frame, trip, archive, rows, out))
        print(json.dumps(records[-1]))
    (arguments.figures / "per-trip-panels.json").write_text(
        json.dumps({"panels": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()