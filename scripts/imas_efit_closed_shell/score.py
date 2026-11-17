#!/usr/bin/env bash
# Read one imas-efit run's equilibrium profile map and the poloidal flux at the
# case's field points.
#
#   score.py <run-dir> [<R> <Z> ...]
#
# Prints JSON: convergence/status fields, the two-dimensional flux map's grid
# extent, and psi bilinearly interpolated at each requested point.

from __future__ import annotations

import json
import sys
from pathlib import Path

import imas
import numpy as np

DD_VERSION = "4.1.0"
DEFAULT_POINTS = ((2.0, 0.0), (3.0, 1.0), (1.2, 0.0), (0.5, 0.0))


def read_run(path: Path) -> dict:
    entry = imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=DD_VERSION)
    eq = entry.get("equilibrium")
    if len(eq.time_slice) == 0:
        return {"error": "no equilibrium time_slice written"}
    slice0 = eq.time_slice[0]
    result: dict = {
        "time": float(eq.time[0]) if len(eq.time) else None,
        "output_flag": (
            int(slice0.code.output_flag) if len(slice0.code.output_flag) else None
        ),
        "psi_axis": float(slice0.global_quantities.psi_axis),
        "psi_boundary": float(slice0.global_quantities.psi_boundary),
        "ip": float(slice0.global_quantities.ip),
    }
    if len(slice0.profiles_2d) == 0:
        result["error"] = "no profiles_2d"
        return result
    p2 = slice0.profiles_2d[0]
    grid_r = np.asarray(p2.grid.dim1, dtype=float)
    grid_z = np.asarray(p2.grid.dim2, dtype=float)
    psi = np.asarray(p2.psi, dtype=float)
    if psi.ndim != 2 or psi.shape != (grid_r.size, grid_z.size):
        result["error"] = (
            f"psi shape {psi.shape} against grid {grid_r.size}x{grid_z.size}"
        )
        return result
    result["grid"] = {
        "r": [float(grid_r[0]), float(grid_r[-1]), grid_r.size],
        "z": [float(grid_z[0]), float(grid_z[-1]), grid_z.size],
        "psi_max": float(np.nanmax(psi)) if np.isfinite(psi).any() else None,
        "psi_min": float(np.nanmin(psi)) if np.isfinite(psi).any() else None,
    }
    points = sys.argv[2:]
    coords = (
        [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
        if points
        else list(DEFAULT_POINTS)
    )
    values = []
    for r, z in coords:
        if not (grid_r[0] <= r <= grid_r[-1] and grid_z[0] <= z <= grid_z[-1]):
            values.append({"r": r, "z": z, "psi": None, "reason": "off grid"})
            continue
        i = int(np.clip(np.searchsorted(grid_r, r) - 1, 0, grid_r.size - 2))
        j = int(np.clip(np.searchsorted(grid_z, z) - 1, 0, grid_z.size - 2))
        tr = (r - grid_r[i]) / (grid_r[i + 1] - grid_r[i])
        tz = (z - grid_z[j]) / (grid_z[j + 1] - grid_z[j])
        value = (
            (1 - tr) * (1 - tz) * psi[i, j]
            + tr * (1 - tz) * psi[i + 1, j]
            + (1 - tr) * tz * psi[i, j + 1]
            + tr * tz * psi[i + 1, j + 1]
        )
        values.append({"r": r, "z": z, "psi": float(value)})
    result["psi"] = values
    return result


def main() -> None:
    run = Path(sys.argv[1])
    try:
        payload = read_run(run)
    except Exception as error:  # noqa: BLE001 - report, never mask
        payload = {"error": f"{type(error).__name__}: {error}"}
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
