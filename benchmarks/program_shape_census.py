"""Census the distinct compiled solve-program shapes, without compiling anything.

A compiled reduced-Newton slice program is keyed on a small set of
capacity-shaped quantities: the mesh geometry digest, the LENGTH of the support
cell index list, and the (shape, dtype) layout of every traced argument.  Two
adapters compile to one program only when all of those agree, so the census a
bucketing lever needs is a table of those capacities per identity, not a
wall-clock measurement of any single compile.

Nothing here compiles.  The bank rows are read from the persisted operand cache
(array shapes only); the certificate rows come from the certificate route's own
compile-free problem builder; the cache inventory is a directory stat.

Usage::

    python benchmarks/program_shape_census.py --out <receipt.json>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BANK_CACHE = ROOT / "logs/exact-operand-cache.npz"
DEFAULT_CACHE_ROOT = Path.home() / ".cache/nova/jax-compilation"
# The certificate route selects its machine with a NEGATIVE sentinel whose
# magnitude is the requested cell count (benchmarks/solovev_certificate.py
# REQUESTED_CELLS = (-110, -300, -500, -1000)), so a positive count is not a
# valid argument to it.
DEFAULT_REQUESTED_CELLS = -300

PROGRAM_BANK_ARRAYS = (
    "radius",
    "height",
    "flux",
    "wall",
    "axis",
    "selected_saddle",
    "binding_flux",
)

RECEIPT_BANK_ARRAYS = (
    "efit_lcfs",
    "efit_x_points",
    "efit_axis",
    "active_set_residuals",
    "active_set_mask_differences",
    "active_set_cycle_damping_activations",
)

CAPACITY_AXES = (
    "mesh_cell_capacity",
    "mesh_axis_nodes",
    "state_size",
    "wall_node_capacity",
    "sample_node_capacity",
    "support_capacity",
    "arc_capacity",
)


def _shape_dtype(array: np.ndarray) -> list:
    return [list(array.shape), str(array.dtype.str)]


def _digest(layout: dict) -> str:
    import hashlib

    payload = json.dumps(layout, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def bank_rows(path: Path) -> list[dict]:
    """One row per bank arm, from the persisted operand cache's array shapes."""
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if int(metadata.get("arm_count", -1)) != 12:
            raise RuntimeError("the operand cache must carry twelve arms")
        rows = []
        for index, record in enumerate(metadata["rows"]):
            prefix = f"arm_{index:02d}_"
            arrays = {
                key[len(prefix) :]: stored[key]
                for key in stored.keys()
                if key.startswith(prefix)
            }
            program = {
                name: _shape_dtype(arrays[name])
                for name in PROGRAM_BANK_ARRAYS
                if name in arrays
            }
            receipt = {
                name: _shape_dtype(arrays[name])
                for name in RECEIPT_BANK_ARRAYS
                if name in arrays
            }
            radius = int(np.size(arrays["radius"]))
            height = int(np.size(arrays["height"]))
            flux = arrays["flux"]
            wall = arrays["wall"]
            rows.append(
                {
                    "identity": str(record["identity"]),
                    "arm": str(record["arm"]),
                    "name": f"{record['identity']}/{record['arm']}",
                    "indices": index,
                    "capacities": {
                        "mesh_axis_nodes": radius * height,
                        "state_size": int(np.size(flux)),
                        "wall_node_capacity": int(wall.shape[0]),
                    },
                    "program_arrays": program,
                    "receipt_arrays": receipt,
                    "program_signature": _digest(program),
                    "receipt_signature": _digest(receipt),
                }
            )
        return rows


GEOMETRY_ARRAYS = ("radius", "height", "wall")
OPERAND_ONLY_ARRAYS = ("flux",)


def geometry_value_agreement(path: Path, arrays: tuple[str, ...]) -> dict:
    """Max absolute difference of the persisted geometry arrays across bank arms.

    The operator program identity hashes mesh VALUES, so two arms share a
    compiled program only if those values agree.  Comparing the persisted
    arrays measures the inputs the identity hashes without building an operator.
    """
    worst = {}
    with np.load(path, allow_pickle=False) as stored:
        count = int(json.loads(str(stored["metadata"].item()))["arm_count"])
        for name in arrays:
            reference = np.asarray(stored["arm_00_" + name], dtype=float)
            worst[name] = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(stored["arm_%02d_" % index + name], dtype=float)
                            - reference
                        )
                    )
                )
                for index in range(1, count)
            )
    return {
        "arrays": list(arrays),
        "max_abs_difference": worst,
        "shared_program": all(value == 0.0 for value in worst.values()),
    }


def certificate_rows(
    requested_cells: int, cases: tuple[str, ...] | None = None
) -> list[dict]:
    """One row per certificate case, from the route's compile-free problem builder."""
    import sys

    if str(ROOT / "benchmarks") not in sys.path:
        sys.path.insert(0, str(ROOT / "benchmarks"))
    import solovev_certificate

    names = cases if cases is not None else tuple(solovev_certificate.CASE_NAMES)
    rows = []
    for name in names:
        started = time.time()
        _profile, _seed, _request, dimensions = (
            solovev_certificate._certificate_compile_problem(name, requested_cells)
        )
        rows.append(
            {
                "case": name,
                "name": name,
                "requested_cells": requested_cells,
                "seconds": round(time.time() - started, 3),
                "capacities": certificate_capacities(dimensions),
                "dimensions": {
                    k: v for k, v in dimensions.items() if not isinstance(v, list)
                },
                "dimension_shapes": {
                    k: _shape_dtype(np.asarray(v))
                    for k, v in dimensions.items()
                    if isinstance(v, list)
                },
            }
        )
    return rows


BUCKET_NAMES = (
    "support_capacity",
    "mesh_axis_nodes",
    "mesh_cell_capacity",
    "arc_capacity",
)

DEFAULT_BUCKETS = {
    "support_capacity": (256, 512, 1024, 2048, 4096),
    "mesh_axis_nodes": (900, 1089, 2500, 4096, 10000),
    "mesh_cell_capacity": (300, 1000, 2500, 5000),
    "arc_capacity": (256, 512, 1024, 2048),
}


def _ceiling(value: int, floors: tuple[int, ...]) -> int:
    for floor in floors:
        if value <= floor:
            return floor
    return int(2 ** math.ceil(math.log2(max(value, 1))))


def bucketed(capacities: dict, buckets: dict) -> dict:
    """Round every present capacity up; an undeclared axis falls back to a power of two.

    Every program-reaching axis must appear in the bucketed signature, so an
    axis with no declared floors is rounded to the next power of two rather than
    dropped: dropping it would understate the program count by making two
    identities look alike on the axes that remain.
    """
    rounded = {}
    for name, value in capacities.items():
        if value is None:
            continue
        rounded[name] = _ceiling(int(value), buckets.get(name, ()))
    return rounded


def _group(rows: list[dict], key) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def cache_inventory(root: Path) -> dict:
    if not root.exists():
        return {"root": str(root), "present": False}
    now = time.time()
    directories = []
    total_entries = 0
    total_bytes = 0
    ages = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        entries = [
            path
            for path in directory.iterdir()
            if path.is_file() and not path.name.endswith("-atime")
        ]
        size = sum(path.stat().st_size for path in entries)
        directories.append(
            {
                "name": directory.name,
                "entries": len(entries),
                "bytes": size,
                "gib": round(size / 2**30, 3),
                "max_entry_bytes": max(
                    (path.stat().st_size for path in entries), default=0
                ),
            }
        )
        total_entries += len(entries)
        total_bytes += size
        ages.extend(now - path.stat().st_mtime for path in entries)
    ages.sort()

    def quantile(fraction: float) -> float | None:
        if not ages:
            return None
        return round(ages[min(len(ages) - 1, int(fraction * len(ages)))] / 86400.0, 2)

    return {
        "root": str(root),
        "present": True,
        "runtime_keys": len(directories),
        "directories": directories,
        "total_entries": total_entries,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 2**30, 3),
        "age_days": {
            "min": quantile(0.0),
            "median": quantile(0.5),
            "q90": quantile(0.9),
            "max": quantile(0.999),
            "older_than_10_days": sum(1 for age in ages if age > 10 * 86400),
            "older_than_20_days": sum(1 for age in ages if age > 20 * 86400),
        },
    }


def certificate_capacities(dimensions: dict) -> dict:
    return {
        "mesh_cell_capacity": dimensions.get("realised_cells"),
        "mesh_axis_nodes": dimensions.get("grid_nodes"),
        "wall_node_capacity": dimensions.get("wall_nodes"),
        "sample_node_capacity": dimensions.get("sample_nodes"),
        "state_size": dimensions.get("solve_state_size"),
        "support_capacity": dimensions.get("atomic_support_capacity"),
        "arc_capacity": dimensions.get("exact_support_capacity"),
    }


def bank_capacities(row: dict) -> dict:
    capacities = dict(row["capacities"])
    capacities.setdefault("mesh_cell_capacity", None)
    capacities.setdefault("support_capacity", None)
    capacities.setdefault("arc_capacity", None)
    capacities.setdefault("sample_node_capacity", None)
    return capacities


def summarise(rows: list[dict], buckets: dict, label: str) -> dict:
    raw = _group(rows, lambda row: json.dumps(row["capacities"], sort_keys=True))
    bucketed_groups = _group(
        rows,
        lambda row: json.dumps(bucketed(row["capacities"], buckets), sort_keys=True),
    )
    observed_axes = sorted(
        {
            name
            for row in rows
            for name, value in row["capacities"].items()
            if value is not None
        }
    )
    waste = {}
    for name in observed_axes:
        floors = buckets.get(name, ())
        observed = sorted(
            {
                int(row["capacities"][name])
                for row in rows
                if row["capacities"].get(name) is not None
            }
        )
        waste[name] = {
            "observed": observed,
            "floors": list(floors) if floors else "power-of-two fallback",
            "padding_ratio": {
                str(value): round(_ceiling(value, floors) / value, 4)
                for value in observed
            },
        }
    return {
        "label": label,
        "identities": len(rows),
        "distinct_capacity_vectors_today": len(raw),
        "distinct_bucketed_vectors": len(bucketed_groups),
        "padding_waste": waste,
        "today": {
            signature: sorted(row["name"] for row in group)
            for signature, group in raw.items()
        },
        "bucketed": {
            signature: sorted(row["name"] for row in group)
            for signature, group in bucketed_groups.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank-cache", type=Path, default=DEFAULT_BANK_CACHE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--requested-cells", type=int, default=DEFAULT_REQUESTED_CELLS)
    parser.add_argument("--skip-certificate", action="store_true")
    arguments = parser.parse_args(argv)

    bank = bank_rows(arguments.bank_cache)
    for row in bank:
        row["capacities"] = bank_capacities(row)
    bank_summary = summarise(bank, DEFAULT_BUCKETS, "bank")
    bank_summary["distinct_program_shape_signatures"] = len(
        {row["program_signature"] for row in bank}
    )
    bank_summary["distinct_receipt_shape_signatures"] = len(
        {row["receipt_signature"] for row in bank}
    )
    agreement = geometry_value_agreement(arguments.bank_cache, GEOMETRY_ARRAYS)
    agreement["operand_only_arrays"] = geometry_value_agreement(
        arguments.bank_cache, OPERAND_ONLY_ARRAYS
    )["max_abs_difference"]
    bank_summary["geometry_value_agreement"] = agreement
    bank_summary["program_count_today"] = (
        1
        if agreement["shared_program"]
        else "one per distinct geometry digest (the census reads shape only)"
    )

    certificates = (
        []
        if arguments.skip_certificate
        else certificate_rows(arguments.requested_cells)
    )
    certificate_summary = (
        summarise(certificates, DEFAULT_BUCKETS, "certificate")
        if certificates
        else None
    )

    combined_summary = summarise(
        bank + certificates, DEFAULT_BUCKETS, "bank-and-certificate"
    )

    proposals = []
    for floors in (
        DEFAULT_BUCKETS,
        {name: tuple(floors[:2]) for name, floors in DEFAULT_BUCKETS.items()},
    ):
        rows = bank + certificates
        groups = _group(
            rows,
            lambda row: json.dumps(bucketed(row["capacities"], floors), sort_keys=True),
        )
        proposals.append(
            {
                "floors": {name: list(values) for name, values in floors.items()},
                "programs": len(groups),
                "identities": len(rows),
            }
        )

    receipt = {
        "schema": "nova.program-shape-census",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bank_cache": str(arguments.bank_cache),
        "compiled_program_key": (
            "nova/equilibrium/reduced_newton.py:2055 _compiled_program_key"
        ),
        "certificate_source": (
            "benchmarks/solovev_certificate._certificate_compile_problem dimensions"
        ),
        "requested_cells": arguments.requested_cells,
        "bank": bank_summary,
        "certificate": certificate_summary,
        "combined": combined_summary,
        "bucketing_proposals": proposals,
        "cache": cache_inventory(arguments.cache_root),
        "rows": {"bank": bank, "certificate": certificates},
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("wrote", arguments.out)
    print(
        "bank identities",
        len(bank),
        "shape signatures",
        bank_summary["distinct_program_shape_signatures"],
        "programs today",
        bank_summary["program_count_today"],
    )
    if certificate_summary:
        print(
            "certificate identities",
            len(certificates),
            "bucketed vectors",
            certificate_summary["distinct_bucketed_vectors"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
