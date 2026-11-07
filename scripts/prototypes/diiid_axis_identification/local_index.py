"""Classify each census candidate by the winding of a small loop around it.

The global census locates candidates on a refined sampling; that sampling can
also manufacture zeros. This reads the seed's own lattice gradient -- central
differences between nodes, interpolated bilinearly -- and walks a small circle
around each candidate. The winding of the gradient direction on that circle is
the point's local index: +1 for an extremum, -1 for a saddle, 0 for a zero the
refinement invented.

The same bilinear lattice gradient is used for the boundary winding, so the
sum of the kept indices and the boundary winding are measured against one
field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
for extra in (
    REPOSITORY,
    REPOSITORY / "scripts" / "prototypes" / "diiid_axis_identification",
    REPOSITORY / "scripts" / "prototypes" / "diiid_forward_convergence",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def sample(field, radius, height, points):
    """Sample a lattice field bilinearly at arbitrary positions."""
    dr = radius[1] - radius[0]
    ri = np.clip((points[:, 0] - radius[0]) / dr, 0.0, radius.size - 1.000001)
    zi = np.clip((points[:, 1] - height[0]) / (height[1] - height[0]), 0.0, height.size - 1.000001)
    i = np.floor(ri).astype(int)
    j = np.floor(zi).astype(int)
    di = ri - i
    dj = zi - j
    return (
        field[i, j] * (1 - di) * (1 - dj)
        + field[i + 1, j] * di * (1 - dj)
        + field[i, j + 1] * (1 - di) * dj
        + field[i + 1, j + 1] * di * dj
    )


def winding(points, pairs, radius, height):
    """Return the winding number of the gradient along a closed sample path."""
    gr = sample(pairs[0], radius, height, points)
    gz = sample(pairs[1], radius, height, points)
    theta = np.unwrap(np.arctan2(gz, gr))
    return float((theta[-1] - theta[0]) / (2 * np.pi))


def boundary_path(radius, height, samples=2000):
    """Return a closed path around the lattice rectangle."""
    r0, r1 = radius[0], radius[-1]
    z0, z1 = height[0], height[-1]
    return np.vstack(
        [
            np.c_[np.linspace(r0, r1, samples, endpoint=False), np.full(samples, z0)],
            np.c_[np.full(samples, r1), np.linspace(z0, z1, samples, endpoint=False)],
            np.c_[np.linspace(r1, r0, samples, endpoint=False), np.full(samples, z1)],
            np.c_[np.full(samples, r0), np.linspace(z1, z0, samples, endpoint=False)],
        ]
    )


def load_seed(cache: Path, frame: int):
    """Return the lattice seed, reading the cache when it exists."""
    if cache.exists():
        held = np.load(cache)
        return held["radius"], held["height"], held["seed"]
    import measure_branches

    import benchmarks.diiid_forward_gs_match as bench

    expansion = float(bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    prepared = measure_branches._prepare_frame(frame, expansion)
    lattice = prepared["profile"].lattice
    radius = np.asarray(lattice.radius, dtype=float)
    height = np.asarray(lattice.height, dtype=float)
    seed = np.asarray(
        prepared["cold"].branches.flux[0], dtype=float
    )[: int(lattice.node_count)].reshape(lattice.shape)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, radius=radius, height=height, seed=seed)
    return radius, height, seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop-radius", type=float, default=None)
    arguments = parser.parse_args()

    radius, height, seed = load_seed(arguments.cache, 144)
    dr = np.gradient(seed, radius, axis=0)
    dz = np.gradient(seed, height, axis=1)
    pairs = (dr, dz)

    candidates = json.loads(arguments.census.read_text())["critical_points"]
    spacing = float(np.hypot(np.diff(radius).mean(), np.diff(height).mean()))
    loop_radius = spacing if arguments.loop_radius is None else arguments.loop_radius

    kept, dropped = [], []
    for point in candidates:
        theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=True)
        ring = np.c_[
            point["r"] + loop_radius * np.cos(theta),
            point["z"] + loop_radius * np.sin(theta),
        ]
        index = int(round(winding(ring, pairs, radius, height)))
        entry = {
            "r": round(point["r"], 4),
            "z": round(point["z"], 4),
            "psi": round(point["psi"], 4),
            "local_index": index,
        }
        if index:
            kept.append(entry)
        else:
            dropped.append(entry)

    boundary = winding(boundary_path(radius, height), pairs, radius, height)
    total = int(sum(entry["local_index"] for entry in kept))
    residual = int(round(boundary)) - total
    record = {
        "gradient_representation": "lattice central differences, bilinearly interpolated between nodes",
        "loop_radius_m": loop_radius,
        "samples_per_loop": 64,
        "kept": kept,
        "dropped_as_spurious": dropped,
        "kept_index_sum": total,
        "boundary_winding": boundary,
        "winding_minus_index_sum": residual,
        "coverage_ok": residual == 0,
    }
    arguments.output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: record[k] for k in ("kept_index_sum", "boundary_winding", "winding_minus_index_sum", "coverage_ok")}, indent=2))
    for entry in kept:
        print("kept", entry["local_index"], entry["r"], entry["z"])
    for entry in dropped:
        print("dropped", entry["local_index"], entry["r"], entry["z"])
    return 0 if residual == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())