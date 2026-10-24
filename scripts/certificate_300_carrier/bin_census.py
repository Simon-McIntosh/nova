"""Census the bin indices around the sampling coordinates two devices disagree on.

``MomentGeometry.from_cells`` merges sampling vertices by quantising them:

    key = tuple(np.rint(vertex / atomic.tolerance).astype(np.int64))

so two vertices merge when they land in the same bin, and two devices agree on
the merge only when their jitter stays inside one bin.  This script reports, at
revision, the atomic tolerance each device used, the extra coordinates the H200
arm carries, each one's nearest CPU coordinate and separation, and the bin index
that coordinate receives under each device's own tolerance and under the other
device's tolerance.

Usage: bin_census.py <cpu-stages> <gpu-stages> <out.json>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


def bin_index(point: np.ndarray, tolerance: float) -> list[int]:
    return [
        int(index) for index in np.rint(np.asarray(point) / tolerance).astype(np.int64)
    ]


def main() -> int:
    cpu_dir, gpu_dir, out_path = (Path(item).resolve() for item in sys.argv[1:4])

    cpu = np.load(cpu_dir / "s7_moment_geometry.npz")
    gpu = np.load(gpu_dir / "s7_moment_geometry.npz")
    cpu_coordinates = np.asarray(cpu["sample_coordinates"], dtype=np.float64)
    gpu_coordinates = np.asarray(gpu["sample_coordinates"], dtype=np.float64)
    cpu_tolerance = float(cpu["atomic_tolerance"][0])
    gpu_tolerance = float(gpu["atomic_tolerance"][0])

    tree = cKDTree(cpu_coordinates)
    distance, nearest = tree.query(gpu_coordinates, k=1)

    # A coordinate with no CPU counterpart within a micron is one the CPU arm
    # merged into a neighbour that the H200 arm kept separate.
    unmatched = np.flatnonzero(distance > 1.0e-6)

    rows = []
    for index in unmatched:
        point = gpu_coordinates[index]
        partner = cpu_coordinates[nearest[index]]
        rows.append(
            {
                "gpu_index": int(index),
                "gpu_coordinate": [float(value) for value in point],
                "nearest_cpu_index": int(nearest[index]),
                "nearest_cpu_coordinate": [float(value) for value in partner],
                "separation_m": float(distance[index]),
                "bin_gpu_tolerance_gpu_point": bin_index(point, gpu_tolerance),
                "bin_cpu_tolerance_gpu_point": bin_index(point, cpu_tolerance),
                "bin_cpu_tolerance_cpu_point": bin_index(partner, cpu_tolerance),
                "bin_gpu_tolerance_cpu_point": bin_index(partner, gpu_tolerance),
            }
        )

    # The 11 are split pairs: one lattice vertex whose two reconstructions
    # (centre_i + offset_j and centre_k + offset_l) landed in different bins on
    # the H200 arm and in one bin on the CPU arm.  Report every pair each arm
    # keeps closer than a micron, with each endpoint's bin under both
    # tolerances, so the bin-edge crossing is visible rather than inferred.
    def split_pairs(
        coordinates: np.ndarray, tolerance: float, other: float
    ) -> list[dict[str, object]]:  # noqa: ANN001
        tree = cKDTree(coordinates)
        pairs = sorted(tree.query_pairs(1.0e-9))
        rows = []
        for first, second in pairs:
            one, two = coordinates[first], coordinates[second]
            rows.append(
                {
                    "indices": [int(first), int(second)],
                    "separation_m": float(np.linalg.norm(one - two)),
                    "bin_own_tolerance": [
                        bin_index(one, tolerance),
                        bin_index(two, tolerance),
                    ],
                    "same_bin_own_tolerance": bin_index(one, tolerance)
                    == bin_index(two, tolerance),
                    "bin_other_tolerance": [
                        bin_index(one, other),
                        bin_index(two, other),
                    ],
                    "same_bin_other_tolerance": bin_index(one, other)
                    == bin_index(two, other),
                }
            )
        return rows

    # How far each device's own two reconstructions of the same lattice vertex
    # sit apart is the quantity the bin has to absorb.  Report it through the
    # duplicated-bin census: the number of distinct coordinates each arm keeps
    # per bin, which is exactly what a merge decision turns on.
    def bin_collisions(coordinates: np.ndarray, tolerance: float) -> dict[str, object]:
        keys: dict[tuple[int, ...], list[int]] = {}
        for index, point in enumerate(coordinates):
            keys.setdefault(tuple(bin_index(point, tolerance)), []).append(index)
        occupied = len(keys)
        return {
            "coordinates": int(len(coordinates)),
            "occupied_bins": int(occupied),
            "distinct_per_bin": float(len(coordinates) / occupied),
        }

    report = {
        "cpu_tolerance_m": cpu_tolerance,
        "gpu_tolerance_m": gpu_tolerance,
        "tolerance_ratio": cpu_tolerance / gpu_tolerance,
        "cpu_sample_coordinates": int(len(cpu_coordinates)),
        "gpu_sample_coordinates": int(len(gpu_coordinates)),
        "unmatched_gpu_count": int(len(unmatched)),
        "gpu_bin_census": bin_collisions(gpu_coordinates, gpu_tolerance),
        "cpu_bin_census": bin_collisions(cpu_coordinates, cpu_tolerance),
        "gpu_split_pairs": split_pairs(gpu_coordinates, gpu_tolerance, cpu_tolerance),
        "cpu_split_pairs": split_pairs(cpu_coordinates, cpu_tolerance, gpu_tolerance),
        "unmatched": rows,
    }

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
