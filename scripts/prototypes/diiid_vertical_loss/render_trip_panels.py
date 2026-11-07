"""Render a poloidal flux-contour panel for one active-set trip.

Each panel draws the trip's own solved field (saved by the per-trip driver as
``frame-<n>-trip-<k>.npz``) as line contours, the EFIT label field at the same
physical levels, both null sets, the governed limiter ring, the bounding surface
read on that trip's grid, and the measured axis position.  A residual strip
below the panel carries the residual at every trip the frame reached.

The bounding surface is derived here from the trip's own flux grid: the saddle
of the solved field and the flux the governed limiter carries, with the lower of
the two as the bounding flux, which is the same competition ``host_boundary_read``
applies.  That routine takes an equilibrium grid object (it evaluates
``wall_flux`` on the wall nodes) and the per-trip archives carry only the flux
array, the wall and the limiter, so it cannot be called on them; the derivation
here is the reduced form, and the fragment says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _bilinear(radius, height, flux, point) -> float:
    r = np.interp(point[0], radius, np.arange(len(radius)), left=np.nan, right=np.nan)
    z = np.interp(point[1], height, np.arange(len(height)), left=np.nan, right=np.nan)
    if not np.isfinite(r) or not np.isfinite(z):
        return float("nan")
    ri, zi = int(round(r)), int(round(z))
    return float(flux[zi, ri])


def _edge_flux(radius, height, flux, boundary) -> float:
    values = np.asarray(
        [_bilinear(radius, height, flux, point) for point in np.asarray(boundary)],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _saddles(radius, height, flux, count=3) -> np.ndarray:
    """Return interior saddle points of the field, strongest Hessian first."""

    dz, dr = np.gradient(flux, height, radius)
    magnitude = np.abs(dr) + np.abs(dz)
    scale = float(np.median(magnitude)) or 1.0
    found = []
    margin = 3
    for i in range(margin, len(height) - margin):
        for j in range(margin, len(radius) - margin):
            if magnitude[i, j] > 0.25 * scale:
                continue
            neighbours = magnitude[i - 1 : i + 2, j - 1 : j + 2]
            if magnitude[i, j] > neighbours.min():
                continue
            d2r = flux[i, j + 1] - 2 * flux[i, j] + flux[i, j - 1]
            d2z = flux[i + 1, j] - 2 * flux[i, j] + flux[i - 1, j]
            if d2r * d2z >= 0:
                continue
            found.append((magnitude[i, j], radius[j], height[i]))
    found.sort()
    return np.asarray([[r, z] for _, r, z in found[:count]], dtype=float).reshape(-1, 2)


def _extremum(radius, height, flux, boundary):
    """Return the field extremum inside a boundary polygon."""

    mesh = np.meshgrid(radius, height)
    points = np.c_[mesh[0].ravel(), mesh[1].ravel()]
    inside = inside_wall_units(points, np.asarray(boundary, dtype=float))
    flat = flux.ravel().copy()
    flat[~inside] = np.nan
    if not np.isfinite(flat).any():
        return None
    index = (
        int(np.nanargmax(flat))
        if abs(np.nanmax(flat)) > abs(np.nanmin(flat))
        else int(np.nanargmin(flat))
    )
    return points[index]


def _shared_levels(ranges) -> np.ndarray:
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


def render_trip(
    frame: int,
    trip: int,
    archive: Path,
    rows: list[dict],
    out: Path,
) -> dict:
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

    saddles = _saddles(radius, height, flux)
    limiter_flux = np.asarray(
        [_bilinear(radius, height, flux, p) for p in limiter], dtype=float
    )
    limiter_flux = float(np.nanmin(limiter_flux))
    saddle_flux = (
        float(np.min([_bilinear(radius, height, flux, p) for p in saddles]))
        if len(saddles)
        else float("nan")
    )
    if np.isfinite(saddle_flux) and saddle_flux > limiter_flux:
        bounding, bounds = saddle_flux, "saddle"
    else:
        bounding, bounds = limiter_flux, "limiter"

    label_axis = _extremum(radius, height, label, label_boundary)
    label_saddles = _saddles(radius, height, label)
    label_limiter = np.asarray(
        [_bilinear(radius, height, label, p) for p in limiter], dtype=float
    )
    label_limiter = float(np.nanmin(label_limiter))
    label_saddle = (
        float(np.min([_bilinear(radius, height, label, p) for p in label_saddles]))
        if len(label_saddles)
        else float("nan")
    )
    label_bounding = max(label_limiter, label_saddle)

    axis_value = (
        _bilinear(radius, height, flux, axis) if axis is not None else float("nan")
    )
    levels = _shared_levels(
        [
            (axis_value, bounding),
            (float(_bilinear(radius, height, label, label_axis)), label_bounding),
        ]
        if label_axis is not None
        else [(axis_value, bounding)]
    )

    frame_rows = sorted(
        [r for r in rows if r["frame"] == frame], key=lambda r: r["trip"]
    )

    figure, (axes, strip) = plt.subplots(
        2,
        1,
        figsize=(14.0, 7.2),
        gridspec_kw={"height_ratios": [5.4, 1.0], "hspace": 0.12},
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.14)
    poloidal_axes(axes)
    draw_flux_contours(
        axes, radius, height, flux, levels, color=DEFAULT_INK.flux_color, wall=wall
    )
    draw_flux_contours(
        axes,
        radius,
        height,
        label,
        levels,
        color="#9a9a9a",
        linewidth=1.0,
        wall=None,
    )
    draw_flux_contours(
        axes,
        radius,
        height,
        flux,
        [bounding],
        color=DEFAULT_INK.separatrix_color,
        linewidth=DEFAULT_INK.separatrix_linewidth,
    )
    if np.isfinite(label_bounding):
        draw_flux_contours(
            axes,
            radius,
            height,
            label,
            [label_bounding],
            color="#606060",
            linewidth=1.0,
        )
    draw_wall(axes, radius=limiter[:, 0], height=limiter[:, 1], close=True)
    draw_nulls(
        axes,
        magnetic_axis=axis,
        x_points=saddles[:1] if len(saddles) else None,
    )
    if label_axis is not None:
        draw_nulls(
            axes,
            magnetic_axis=label_axis,
            x_points=label_saddles[:1] if len(label_saddles) else None,
            contain=None,
        )
    if len(label_boundary):
        axes.plot(
            label_boundary[:, 0],
            label_boundary[:, 1],
            color="#555555",
            linewidth=1.1,
            linestyle="--",
            zorder=DEFAULT_INK.zorder_separatrix,
        )

    trips = [r["trip"] for r in frame_rows]
    residuals = [
        r["residual"] if r.get("residual") is not None else np.nan for r in frame_rows
    ]
    strip.plot(
        trips, residuals, marker="o", color=DEFAULT_INK.flux_color, linewidth=1.6
    )
    strip.plot(
        [trip],
        [row.get("residual", np.nan)],
        marker="o",
        markersize=8,
        color=DEFAULT_INK.separatrix_color,
    )
    strip.set_xlim(min(trips) - 0.3, max(trips) + 0.3)
    strip.set_xticks(trips)
    strip.set_xlabel("active-set trip", fontsize=16)
    strip.set_ylabel("residual", fontsize=16)
    strip.tick_params(labelsize=14)
    strip.spines[["top", "right"]].set_visible(False)
    strip.set_ylim(0, max([v for v in residuals if np.isfinite(v)] + [0.1]) * 1.15)

    axes.set_aspect("equal")
    axes.set_xlim(
        float(np.min(limiter[:, 0])) - 0.05, float(np.max(limiter[:, 0])) + 0.15
    )
    z_low = float(np.min(limiter[:, 1])) - 0.05
    z_high = float(np.max(limiter[:, 1])) + 0.05
    if axis is not None and np.isfinite(axis[1]):
        z_low = min(z_low, float(axis[1]) - 0.08)
        z_high = max(z_high, float(axis[1]) + 0.08)
    axes.set_ylim(z_low, z_high)

    figure.suptitle(
        f"DIII-D frame {frame}, trip {trip}: solved contours in blue, "
        f"EFIT label in grey; "
        f"bounding flux {bounding:.3f} Wb from the {bounds}",
        fontsize=14,
        y=0.965,
    )
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
        "saddles_rz_m": saddles.tolist(),
        "bounding_flux": bounding,
        "bounds": bounds,
        "limiter_flux": limiter_flux,
        "saddle_flux": saddle_flux,
        "label_bounding_flux": label_bounding,
        "residual": row.get("residual"),
        "svg": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["146:1", "146:2", "137:2"],
    )
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
