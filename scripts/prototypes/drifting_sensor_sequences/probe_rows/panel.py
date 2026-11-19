"""One reconstructed slice of each code, on shared levels, with both null sets.

The nova map comes from the per-slice receipt written by ``study.py nova``
(``maps/nova-<sequence>-<factor>-<slice>.npz`` beside the receipt), the
imas-efit map from that arm's own output equilibrium, resampled onto the same
uniform lattice the nova map carries.  Both maps are drawn as unfilled line
contours on one level array taken from the generating state's span, the wall is
drawn from the machine's limiter ring, and each code's nulls are drawn in that
code's own colour.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

STUDY = Path(os.environ.get("DRIFT_STUDY", str(Path(__file__).resolve().parent)))
BLUE = "#1f5fa8"
ORANGE = "#c06a1b"


def wall_polyline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the limiter ring as a NaN-separated polyline."""
    stored = np.load(path)
    ring = np.asarray(stored["wall"], dtype=float)
    return ring[:, 0], ring[:, 1]


def resample(source, target_radius, target_height) -> np.ndarray:
    """Piecewise-linear resample of one rectangular ψ(R, Z) onto the nova grid."""
    from scipy.interpolate import griddata

    radius = np.asarray(source.radius, dtype=float)
    height = np.asarray(source.height, dtype=float)
    values = np.asarray(source.flux, dtype=float)
    points = np.column_stack(
        [
            np.repeat(radius, height.size),
            np.tile(height, radius.size),
        ]
    )
    rad, hei = np.meshgrid(target_radius, target_height, indexing="ij")
    grid = np.asarray(griddata(points, values.reshape(-1), (rad, hei), method="linear"))
    holes = ~np.isfinite(grid)
    if np.any(holes):
        nearest = np.asarray(
            griddata(points, values.reshape(-1), (rad, hei), method="nearest")
        )
        grid[holes] = nearest[holes]
    return grid


def efit_state(output: Path, index: int, wall_source: str | Path):
    from nova.imas.mast_efit_referee import read_equilibrium_state

    return read_equilibrium_state(output, time_index=index, wall_source=wall_source)


def efit_x_points(output: Path, index: int) -> np.ndarray:
    """Read the solver's own X-point positions from the output equilibrium."""
    import imas

    entry = imas.DBEntry(f"imas:hdf5?path={output}", "r", dd_version="4.1.0")
    try:
        boundary = entry.get("equilibrium").time_slice[index].boundary
        radius = np.asarray(boundary.x_point.r, dtype=float)
        height = np.asarray(boundary.x_point.z, dtype=float)
    finally:
        entry.close()
    return np.column_stack([radius, height])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--factor", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    maps = STUDY / "maps"
    stem = "nova-%s-%.4f-%05d" % (args.sequence, args.factor, args.slice)
    nova = np.load(maps / (stem + ".npz"))
    receipt = json.loads((STUDY / "receipts" / (stem + ".json")).read_text())
    radius = np.asarray(nova["radius"], float)
    height = np.asarray(nova["height"], float)
    psi_nova = np.asarray(nova["flux"], float)

    output = STUDY / ("efit-%s-200-%.4f" % (args.sequence, args.factor))
    wall_file = STUDY / "conf" / "machine-wall.npz"
    state = efit_state(output, args.slice, wall_file)
    psi_efit = resample(state.flux_map, radius, height)
    x_efit = efit_x_points(output, args.slice)

    wall_r, wall_z = wall_polyline(wall_file)
    generating = json.loads((STUDY / "generating.json").read_text())
    span_axis = float(generating["flux_axis"])
    span = float(generating["flux_span"])
    levels = np.linspace(span_axis + 0.08 * span, span_axis + 0.92 * span, 12)
    # One additive constant puts imas-efit's map in the generating gauge, so a
    # single physical level array (Wb) carries both contours.  A sign flip is
    # applied first when the output stores the opposite flux orientation.
    efit_axis = float(state.flux_axis)
    efit_span = float(state.flux_boundary) - efit_axis
    if np.sign(efit_span) != np.sign(span):
        psi_efit = -psi_efit
        efit_axis = -efit_axis
        efit_span = -efit_span
    offset = span_axis - efit_axis
    psi_efit = psi_efit + offset

    payload = dict(
        radius=radius,
        height=height,
        psi_nova=psi_nova,
        psi_efit=psi_efit,
        levels=levels,
        axis_nova=np.asarray(receipt["axis"], float),
        x_points_nova=(np.asarray(receipt.get("x_points") or [], float).reshape(-1, 2)),
        axis_efit=np.asarray(state.magnetic_axis_m, float),
        x_points_efit=np.asarray(x_efit, float).reshape(-1, 2),
        wall_r=wall_r,
        wall_z=wall_z,
        flux_axis_efit=float(state.flux_axis),
        gauge_offset=float(offset),
        slice=np.array(args.slice),
    )
    stored = Path(args.output)
    stored.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stored, **payload)
    print(
        json.dumps(
            {
                k: (v if np.ndim(v) == 0 else np.asarray(v).shape)
                for k, v in payload.items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
