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

import analyse

STUDY = Path(os.environ.get("DRIFT_STUDY", str(Path(__file__).resolve().parent)))


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
        constraints = entry.get("equilibrium").time_slice[index].constraints
        points = [
            (
                float(point.position_reconstructed.r),
                float(point.position_reconstructed.z),
            )
            for point in constraints.x_point
        ]
    finally:
        entry.close()
    rows = [pair for pair in points if np.all(np.isfinite(pair))]
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def nova_slice(sequence: str, factor: float, index: int) -> tuple:
    """One slice's map and its lattice from the arm's stored map stack.

    A sequence is written as one map stack per shard, so the slice is looked up
    where its index appears; the stacks of one arm share one lattice.
    """
    pattern = "nova-%s-*-%.4f-%s"
    for path in sorted(
        (STUDY / "maps").glob(
            pattern % (sequence, factor, analyse.SELECTED_ROWS)
            + "*.npz"
        )
    ):
        stored = np.load(path)
        matches = np.flatnonzero(np.asarray(stored["slice"], dtype=int) == index)
        if matches.size:
            return (
                np.asarray(stored["flux"], dtype=float)[matches[0]],
                np.asarray(stored["radius"], dtype=float),
                np.asarray(stored["height"], dtype=float),
            )
    raise SystemExit("slice %d of %s is in no map stack" % (index, sequence))


def efit_output(sequence: str, factor: float) -> Path:
    """The arm's output directory, whichever row-set tag its stem carries."""
    matches = sorted(STUDY.glob("efit-%s-*-%.4f*" % (sequence, factor)))
    directories = [path for path in matches if path.is_dir()]
    if len(directories) != 1:
        raise SystemExit(
            "%s carries %d imas-efit outputs" % (sequence, len(directories))
        )
    return directories[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--factor", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stem = "nova-%s-%.4f-%05d" % (args.sequence, args.factor, args.slice)
    receipt = json.loads((STUDY / "receipts" / (stem + ".json")).read_text())
    psi_nova, radius, height = nova_slice(args.sequence, args.factor, args.slice)

    output = efit_output(args.sequence, args.factor)
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

    # The generating state's own field on this lattice, so the panel shows what
    # both reconstructions are set against.  It is already in the generating
    # gauge and orientation, being the state the rows came from.
    reference_file = STUDY / "reference-grid.npz"
    if not reference_file.exists():
        raise SystemExit("%s is missing: run reference.py" % reference_file)
    reference = np.load(reference_file)
    psi_ref = np.asarray(reference["psi"], dtype=float)
    axis_ref = np.asarray(reference["axis"], dtype=float)
    x_points_ref = np.asarray(reference["x_points"], dtype=float).reshape(-1, 2)

    payload = dict(
        radius=radius,
        height=height,
        psi_nova=psi_nova,
        psi_efit=psi_efit,
        psi_ref=psi_ref,
        levels=levels,
        axis_nova=np.asarray(receipt["axis"], float),
        x_points_nova=(np.asarray(receipt.get("x_points") or [], float).reshape(-1, 2)),
        axis_efit=np.asarray(state.magnetic_axis_m, float),
        x_points_efit=np.asarray(x_efit, float).reshape(-1, 2),
        axis_ref=axis_ref,
        x_points_ref=x_points_ref,
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
                k: (
                    float(v)
                    if np.ndim(v) == 0 and np.asarray(v).dtype.kind == "f"
                    else int(v)
                    if np.ndim(v) == 0
                    else np.asarray(v).shape
                )
                for k, v in payload.items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
