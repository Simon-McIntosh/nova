"""Census the extrema of each trip's solved field against the solver's axis.

Every extremum ``polish_stationary_points`` returns is reported with its flux,
position, whether it lies inside the grid and inside the limiter, and its local
winding index.  The global flux maximum inside the limiter is reported
separately, and the flux at the axis the solve reported is placed beside them.
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


def census(archive: Path, axis, limiter, label: str) -> dict:
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = _oriented(data["flux"], radius, height)
    interp = RegularGridInterpolator(
        (radius, height), flux.T, method="cubic", bounds_error=False, fill_value=np.nan
    )
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    seeds = _seeds(radius, height, flux, count=32)
    polished = polish_stationary_points(
        surface, jnp.asarray(seeds), jnp.ones((len(seeds),), dtype=bool)
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2)
    value = np.asarray(polished["value"], dtype=float).reshape(-1)
    converged = np.asarray(polished["converged"], dtype=bool).reshape(-1)
    kind = np.asarray(polished["hessian_type"]).reshape(-1)

    mesh_r, mesh_z = np.meshgrid(radius, height)
    points = np.c_[mesh_r.ravel(), mesh_z.ravel()]
    inside_limiter = inside_wall_units(points, limiter)
    masked = flux.copy()
    masked.ravel()[~inside_limiter] = np.nan
    flat_index = int(np.nanargmax(masked))
    global_max = {
        "flux": float(masked.ravel()[flat_index]),
        "position_rz": points[flat_index].tolist(),
    }

    extrema = []
    for index in range(len(position)):
        if not converged[index]:
            continue
        r, z = position[index]
        inside_grid = bool(radius[0] < r < radius[-1] and height[0] < z < height[-1])
        extrema.append(
            {
                "position_rz": position[index].tolist(),
                "flux": float(value[index]),
                "hessian_type": int(kind[index]),
                "inside_grid": inside_grid,
                "inside_limiter": bool(inside_wall_units(position[index][None, :], limiter)[0]),
                "winding_index": _winding(interp, position[index]) if inside_grid else None,
            }
        )
    return {
        "label": label,
        "global_maximum_inside_limiter": global_max,
        "reported_axis_rz_m": None if axis is None else list(map(float, axis)),
        "reported_axis_flux": None
        if axis is None
        else float(np.asarray(interp(tuple(axis)))),
        "extrema": extrema,
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
            r["axis_rz_m"] for r in rows if r["frame"] == frame and r["trip"] == trip
        )
        with np.load(arguments.run_dir / f"frame-{frame}-trip-1.npz") as data:
            limiter = np.asarray(data["limiter"], dtype=float)
        records.append(
            census(
                arguments.run_dir / f"frame-{frame}-trip-{trip}.npz",
                axis,
                limiter,
                f"frame {frame} trip {trip}",
            )
        )
    arguments.out.write_text(json.dumps({"fields": records}, indent=2) + "\n")
    for record in records:
        print("==", record["label"])
        print("   global max inside limiter", record["global_maximum_inside_limiter"])
        print("   reported axis", record["reported_axis_rz_m"], "flux", record["reported_axis_flux"])
        for point in record["extrema"]:
            print(
                "   O R=%.3f Z=%+.3f type=%2d flux=%+.3f inside_grid=%s in_lim=%s idx=%s"
                % (
                    point["position_rz"][0],
                    point["position_rz"][1],
                    point["hessian_type"],
                    point["flux"],
                    point["inside_grid"],
                    point["inside_limiter"],
                    point["winding_index"],
                )
            )


if __name__ == "__main__":
    main()