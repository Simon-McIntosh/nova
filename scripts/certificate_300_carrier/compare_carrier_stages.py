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
import sys
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

# Arrays that carry a coordinate ordering, compared as an unordered set too.
COORDINATE_ARRAYS = {
    "s0_wall": ["wall"],
    "s1_coilset": ["centres", "x", "z"],
    "s3_delaunay": ["centres"],
    "s6_sampling": ["sampling"],
    "s7_moment_geometry": ["sample_coordinates", "atomic_node_coordinates"],
}


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    return np.ascontiguousarray(array.reshape(len(array), -1))


def compare_stage(first: Path, second: Path, stage: str) -> dict[str, object]:
    left = np.load(first / f"{stage}.npz", allow_pickle=True)
    right = np.load(second / f"{stage}.npz", allow_pickle=True)
    result: dict[str, object] = {"stage": stage, "differences": [], "agree": True}
    for key in sorted(set(left.files) | set(right.files)):
        if key not in left.files or key not in right.files:
            result["differences"].append({"array": key, "reason": "missing_key",})
            result["agree"] = False
            continue
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
        if np.array_equal(a, b):
            item["reason"] = None
        elif a.dtype.kind in "OU":
            item["reason"] = "value"
            item["mismatch_count"] = int(np.count_nonzero(a != b))
            result["differences"].append(item)
            result["agree"] = False
            continue
        else:
            differing = ~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
            item["reason"] = "value"
            item["mismatch_count"] = int(np.count_nonzero(differing))
            item["max_abs_delta"] = float(
                np.nanmax(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
            )
            result["differences"].append(item)
            result["agree"] = False
            continue
        if key in COORDINATE_ARRAYS.get(stage, ()):
            same_order = np.array_equal(a, b)
            set_a = np.unique(rows(a), axis=0)
            set_b = np.unique(rows(b), axis=0)
            item["same_order"] = bool(same_order)
            item["unique_rows_a"] = int(len(set_a))
            item["unique_rows_b"] = int(len(set_b))
            item["row_set_equal"] = bool(
                set_a.shape == set_b.shape and np.array_equal(set_a, set_b)
            )
            if not item["row_set_equal"]:
                result["agree"] = False
    return result


def plant(dump: Path, mode: str) -> str:
    """Infect a copy of one stage with a known defect; return the stage name."""
    target = dump / "s6_sampling.npz"
    payload = dict(np.load(target, allow_pickle=True))
    sampling = np.array(payload["sampling"], copy=True)
    if mode == "reorder":
        sampling = sampling[:, ::-1, :]
        story = "reversed the per-cell sampling-vertex order"
    elif mode == "duplicate":
        sampling = np.concatenate([sampling, sampling[:1]], axis=0)
        story = "appended one duplicated sampling coordinate row"
    else:
        raise ValueError(f"unknown plant mode {mode!r}")
    payload["sampling"] = sampling
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

    report = {
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
            detail = {k: v for k, v in item.items() if k not in ("digest_a", "digest_b")}
            print(f"    {detail}", flush=True)
        if not result["agree"] and first_failing is None:
            first_failing = stage
    report["first_differing_stage"] = first_failing
    print(f"FIRST_DIFFERING_STAGE={first_failing}", flush=True)

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    if scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    print("EXIT=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())