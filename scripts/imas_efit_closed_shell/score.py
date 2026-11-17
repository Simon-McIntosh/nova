#!/usr/bin/env python3
# Read one imas-efit run's equilibrium profiles_2d and the poloidal flux at the
# case's field points.
#
#   score.py <run-dir> [--out <path>] [<R> <Z> ...]
#
# JSON with the run's convergence fields, the flux map's extent, and psi
# bilinearly interpolated at each requested point.  imas-python's emitters
# write to stdout, so a caller that needs the JSON alone passes --out.

from __future__ import annotations

import json
import sys
from pathlib import Path

import imas
import numpy as np

DD_VERSION = "4.1.0"
DEFAULT_POINTS = ((2.0, 0.0), (3.0, 1.0), (1.2, 0.0), (0.5, 0.0))


def parse_args(argv: list[str]) -> tuple[Path, Path | None, list[tuple[float, float]]]:
    run = Path(argv[0])
    rest = argv[1:]
    out = None
    if "--out" in rest:
        index = rest.index("--out")
        out = Path(rest[index + 1])
        del rest[index : index + 2]
    if len(rest) % 2:
        raise SystemExit(f"points must be R Z pairs, got {rest}")
    points = [
        (float(rest[i]), float(rest[i + 1])) for i in range(0, len(rest), 2)
    ]
    return run, out, points or list(DEFAULT_POINTS)


def interpolate(
    grid_r: np.ndarray, grid_z: np.ndarray, psi: np.ndarray, r: float, z: float
) -> float | None:
    if not (grid_r[0] <= r <= grid_r[-1] and grid_z[0] <= z <= grid_z[-1]):
        return None
    i = int(np.clip(np.searchsorted(grid_r, r) - 1, 0, grid_r.size - 2))
    j = int(np.clip(np.searchsorted(grid_z, z) - 1, 0, grid_z.size - 2))
    tr = (r - grid_r[i]) / (grid_r[i + 1] - grid_r[i])
    tz = (z - grid_z[j]) / (grid_z[j + 1] - grid_z[j])
    return float(
        (1 - tr) * (1 - tz) * psi[i, j]
        + tr * (1 - tz) * psi[i + 1, j]
        + (1 - tr) * tz * psi[i, j + 1]
        + tr * tz * psi[i + 1, j + 1]
    )


def read_run(run: Path, points: list[tuple[float, float]]) -> dict:
    entry = imas.DBEntry(f"imas:hdf5?path={run}", "r", dd_version=DD_VERSION)
    eq = entry.get("equilibrium")
    if len(eq.time_slice) == 0:
        return {"error": "no equilibrium time_slice written"}
    s = eq.time_slice[0]
    result: dict = {
        "time": float(eq.time[0]) if len(eq.time) else None,
        "output_flag": int(s.code.output_flag[0])
        if len(s.code.output_flag)
        else None,
        "psi_axis": float(s.global_quantities.psi_axis),
        "psi_boundary": float(s.global_quantities.psi_boundary),
        "ip": float(s.global_quantities.ip),
    }
    if len(s.profiles_2d) == 0:
        result["error"] = "no profiles_2d"
        return result
    p2 = s.profiles_2d[0]
    grid_r = np.asarray(p2.grid.dim1, dtype=float)
    grid_z = np.asarray(p2.grid.dim2, dtype=float)
    psi = np.asarray(p2.psi, dtype=float)
    if psi.ndim != 2 or psi.shape != (grid_r.size, grid_z.size):
        result["error"] = f"psi shape {psi.shape} against grid {grid_r.size}x{grid_z.size}"
        return result
    result["grid"] = {
        "r": [float(grid_r[0]), float(grid_r[-1]), int(grid_r.size)],
        "z": [float(grid_z[0]), float(grid_z[-1]), int(grid_z.size)],
        "psi_max": float(np.nanmax(psi)),
        "psi_min": float(np.nanmin(psi)),
    }
    result["psi"] = [
        {"r": r, "z": z, "psi": interpolate(grid_r, grid_z, psi, r, z)}
        for r, z in points
    ]
    return result


def main() -> None:
    run, out, points = parse_args(sys.argv[1:])
    try:
        payload = read_run(run, points)
    except Exception as error:  # noqa: BLE001 - report, never mask
        payload = {"error": f"{type(error).__name__}: {error}"}
    text = json.dumps(payload, indent=1)
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
