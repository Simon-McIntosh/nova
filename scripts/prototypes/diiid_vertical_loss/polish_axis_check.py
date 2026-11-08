"""Polish the candidate maxima and the solver's reported axis on cached grids.

Three seeds are polished per grid: the argmax of the flux inside the limiter, the
axis the solve reported, and the union of grid-scale stationary seeds.  Each
result is reported with its converged position, Hessian type and local winding
index, and points that converge to the same location are drawn as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.poloidal import inside_wall_units

from core_stationary_probe import _seeds, _winding


def _oriented(flux, radius, height):
    array = np.asarray(flux, dtype=float)
    return array if array.shape == (len(height), len(radius)) else array.T


def _polish(surface, seeds):
    polished = polish_stationary_points(
        surface, jnp.asarray(seeds), jnp.ones((len(seeds),), dtype=bool)
    )
    return (
        np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2),
        np.asarray(polished["value"], dtype=float).reshape(-1),
        np.asarray(polished["converged"], dtype=bool).reshape(-1),
        np.asarray(polished["hessian_type"]).reshape(-1),
    )


def check(archive: Path, axis, label: str) -> dict:
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = _oriented(data["flux"], radius, height)
        limiter = np.asarray(data["limiter"], dtype=float)
    interp = RegularGridInterpolator(
        (radius, height), flux.T, method="cubic", bounds_error=False, fill_value=np.nan
    )
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    mesh_r, mesh_z = np.meshgrid(radius, height)
    points = np.c_[mesh_r.ravel(), mesh_z.ravel()]
    inside = inside_wall_units(points, limiter)
    masked = flux.copy()
    masked.ravel()[~inside] = np.nan
    argmax = points[int(np.nanargmax(masked))]

    seeds = np.vstack([argmax[None, :], np.asarray(axis)[None, :], _seeds(radius, height, flux, 32)])
    position, value, converged, kind = _polish(surface, seeds)
    rows = []
    seen = []
    for index in range(len(position)):
        if not converged[index] or not np.all(np.isfinite(position[index])):
            continue
        point = position[index]
        if any(np.hypot(*(point - other)) < 0.02 for other in seen):
            continue
        seen.append(point)
        in_grid = bool(radius[0] < point[0] < radius[-1] and height[0] < point[1] < height[-1])
        rows.append(
            {
                "position_rz": point.tolist(),
                "flux": float(value[index]),
                "hessian_type": int(kind[index]),
                "inside_grid": in_grid,
                "inside_limiter": bool(inside_wall_units(point[None, :], limiter)[0]),
                "winding_index": _winding(interp, point) if in_grid else None,
            }
        )
    return {
        "label": label,
        "grid_argmax_rz_m": argmax.tolist(),
        "grid_argmax_flux": float(np.nanmax(masked)),
        "reported_axis_rz_m": list(map(float, axis)),
        "polished": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    rows = json.loads(arguments.rows.read_text())["rows"]
    records = []
    for frame, trip in ((146, 1), (146, 2), (137, 1), (137, 2)):
        axis = next(
            r["axis_rz_m"] for r in rows if r["frame"] == frame and r["axis_rz_m"]
            and r["trip"] == trip
        )
        records.append(
            check(
                arguments.run_dir / f"frame-{frame}-trip-{trip}.npz",
                axis,
                f"frame {frame} trip {trip}",
            )
        )
    arguments.out.write_text(json.dumps({"fields": records}, indent=2) + "\n")
    for record in records:
        print("==", record["label"])
        print("   argmax", record["grid_argmax_rz_m"], record["grid_argmax_flux"])
        print("   axis", record["reported_axis_rz_m"])
        for point in record["polished"]:
            print(
                "   O R=%.3f Z=%+.3f type=%2d idx=%s flux=%+.3f in_lim=%s"
                % (
                    point["position_rz"][0],
                    point["position_rz"][1],
                    point["hessian_type"],
                    point["winding_index"],
                    point["flux"],
                    point["inside_limiter"],
                )
            )


if __name__ == "__main__":
    main()