"""Compare two carrier stage dumps and name the first stage that differs.

Both dumps are produced by ``dump_carrier_stages.py``.  A stage is reported as
the first disagreement when any array it wrote differs by value or by shape.
The ``--plant`` modes inject a known defect into the first dump's copy of one
stage, which is the literal negative control: the same comparison must flag the
staged defect at the stage it was planted in and no earlier.

Usage: compare_carrier_stages.py <dump-a> <dump-b> [--plant reorder|duplicate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

STAGES = (
    "s0_wall",
    "s1_coilset",
    "s2_polygons",
    "s3_delaunay",
    "s4_boundary_cells",
    "s5_stencil",
    "s6_sampling",
    "s7_moment_geometry",
    "s8_assembled",
)

# Arrays that carry coordinates, compared as a set and by nearest neighbour as
# well as slot by slot, so a permutation is distinguishable from a value shift.
COORDINATE_ARRAYS = {
    "s0_wall": ["wall"],
    "s1_coilset": ["centres", "x", "z"],
    "s3_delaunay": ["centres"],
    "s6_sampling": ["sampling"],
    "s7_moment_geometry": [
        "sample_coordinates",
        "atomic_node_coordinates",
        "atomic_centroids",
    ],
}


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    return np.ascontiguousarray(array.reshape(len(array), -1))


def nearest_row_max(first: np.ndarray, second: np.ndarray) -> float:
    """Return the largest nearest-row distance from the first set into the second."""
    from scipy.spatial import cKDTree

    distance, _index = cKDTree(second).query(first, k=1)
    return float(np.max(distance))


def compare_stage(first: Path, second: Path, stage: str) -> dict[str, object]:
    left = np.load(first / f"{stage}.npz", allow_pickle=True)
    right = np.load(second / f"{stage}.npz", allow_pickle=True)
    result: dict[str, object] = {"stage": stage, "differences": [], "agree": True}
    for key in sorted(set(left.files) | set(right.files)):
        if key not in left.files or key not in right.files:
            result["differences"].append({"array": key, "reason": "missing_key"})
            result["agree"] = False
        else:
            a, b = np.asarray(left[key]), np.asarray(right[key])
            item: dict[str, object] = {
                "array": key,
                "shape_a": list(a.shape),
                "shape_b": list(b.shape),
                "digest_a": digest(a),
                "digest_b": digest(b),
            }
            if a.shape != b.shape:
                item["reason"] = "shape"
                result["differences"].append(item)
                result["agree"] = False
                continue
            numeric = a.dtype.kind not in "OU"
            if np.array_equal(a, b):
                item["reason"] = None
                continue
            elif numeric and np.all(
                np.isclose(
                    np.asarray(a, dtype=float),
                    np.asarray(b, dtype=float),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            ):
                # NaN==NaN is not equality to numpy, but the same NaN in the
                # same slot is the same datum for this comparison.
                item["reason"] = None
                item["nan_matched"] = True
                continue
            elif numeric:
                differing = ~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
                item["reason"] = "value"
                item["mismatch_count"] = int(np.count_nonzero(differing))
                item["max_abs_delta"] = float(
                    np.nanmax(
                        np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
                    )
                )
            else:
                item["reason"] = "value"
                item["mismatch_count"] = int(np.count_nonzero(a != b))
            if key in COORDINATE_ARRAYS.get(stage, ()) and numeric:
                rows_a, rows_b = rows(a), rows(b)
                order_a = np.lexsort(rows_a.T[::-1])
                order_b = np.lexsort(rows_b.T[::-1])
                item["same_order"] = bool(np.array_equal(a, b))
                item["sorted_rows_equal"] = bool(
                    np.array_equal(rows_a[order_a], rows_b[order_b])
                )
                item["max_nearest_row_distance"] = nearest_row_max(rows_a, rows_b)
                if not item["sorted_rows_equal"] and (
                    item["max_nearest_row_distance"] > 1.0e-9
                ):
                    item["reason"] = "coordinate_set"
            result["differences"].append(item)
            result["agree"] = False
    return result


def plant(dump: Path, mode: str) -> str:
    """Infect a copy of one stage with a known defect; return the stage name."""
    target = dump / "s6_sampling.npz"
    payload = dict(np.load(target, allow_pickle=True))
    sampling = np.array(payload["sampling"], copy=True)
    if mode == "reorder":
        payload["sampling"] = sampling[:, ::-1, :]
        story = "reversed the per-cell sampling-vertex order"
    elif mode == "duplicate":
        payload["sampling"] = np.concatenate([sampling, sampling[:1]], axis=0)
        story = "appended one duplicated sampling coordinate row"
    else:
        raise ValueError(f"unknown plant mode {mode!r}")
    np.savez(target, **payload)
    return f"s6_sampling ({story})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--plant", choices=("reorder", "duplicate"), default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    first = args.first.resolve()
    second = args.second.resolve()
    planted = None
    scratch = None
    if args.plant is not None:
        scratch = Path(tempfile.mkdtemp(prefix="carrier-negative-control-"))
        copy = scratch / "planted"
        shutil.copytree(first, copy)
        planted = plant(copy, args.plant)
        first = copy

    report: dict[str, object] = {
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "planted": planted,
        "stages": [],
    }
    first_failing = None
    for stage in STAGES:
        result = compare_stage(first, second, stage)
        report["stages"].append(result)
        status = "agree" if result["agree"] else "DIFFER"
        print(f"{stage}: {status}", flush=True)
        for item in result["differences"]:
            detail = {
                k: v for k, v in item.items() if k not in ("digest_a", "digest_b")
            }
            print(f"    {detail}", flush=True)
        if not result["agree"] and first_failing is None:
            first_failing = stage
    report["first_differing_stage"] = first_failing
    print(f"FIRST_DIFFERING_STAGE={first_failing}", flush=True)

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    # A comparison exits non-zero when any stage failed to be read or compared;
    # a stage that merely differs from its partner is the measurement, not a
    # failure, and keeps the exit at zero.
    broken = [
        item
        for stage in report["stages"]
        for item in stage["differences"]
        if item.get("reason") == "missing_key"
    ]
    status = 1 if broken else 0
    if broken:
        print(f"COMPARISON_FAILED {len(broken)} missing-key difference(s)", flush=True)
    print(f"EXIT={status}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
