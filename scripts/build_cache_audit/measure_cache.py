"""Measure cold construction and warm reload of the shared fixture cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter
import sys


REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FIXTURES = (
    ("coarse", 1, 566),
    ("fine", 2, 1069),
)


def load_reference_module():
    """Load the fixture definitions without collecting their test suite."""
    spec = importlib.util.spec_from_file_location("cache_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _receipt(receipt) -> dict[str, object]:
    """Return one JSON-safe cache receipt."""
    return {
        "store": receipt.store,
        "key": receipt.key,
        "hit": receipt.hit,
        "lock_wait_seconds": receipt.lock_wait_seconds,
        "load_seconds": receipt.load_seconds,
        "build_seconds": receipt.build_seconds,
        "store_seconds": receipt.store_seconds,
        "validation_seconds": receipt.validation_seconds,
        "arrays_verified": receipt.arrays_verified,
        "bytes_verified": receipt.bytes_verified,
        "bitwise_stored_precision": receipt.bitwise_stored_precision,
    }


def _write(path: Path, report: dict[str, object]) -> None:
    """Checkpoint every completed fixture measurement."""
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build each absent key once, then measure and verify its warm reload."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/build_cache_audit/measurements.json"),
    )
    parser.add_argument(
        "--fixture",
        action="append",
        choices=[name for name, _multiplier, _cells in FIXTURES],
    )
    parser.add_argument("--require-cold", action="store_true")
    args = parser.parse_args()
    selected = set(args.fixture or [name for name, _multiplier, _cells in FIXTURES])

    reference = load_reference_module()
    reference.configure_dtypes()
    case = reference.require_reference()
    report: dict[str, object] = {
        "schema": "shared-fixture-cache-measurement",
        "fixtures": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)

    for name, multiplier, expected_cells in FIXTURES:
        if name not in selected:
            continue
        reference.WALL_NODES = 3 * multiplier
        requested = reference.SUITE_CELLS * multiplier
        print(
            f"CACHE_REQUEST fixture={name} requested={requested} "
            f"wall_nodes={reference.WALL_NODES} expectation=cold",
            flush=True,
        )
        cold_started = perf_counter()
        cold = reference.cached_machine(case, requested, passive=True)
        cold_request_seconds = perf_counter() - cold_started
        print(reference.machine_cache_summary(name, cold), flush=True)
        if len(cold.node) != expected_cells:
            raise AssertionError(
                f"expected {expected_cells} {name} cells, got {len(cold.node)}"
            )
        if args.require_cold and cold.cache_receipt.hit:
            raise AssertionError(f"{name} cache key already existed before cold build")

        print(
            f"CACHE_REQUEST fixture={name} requested={requested} "
            f"wall_nodes={reference.WALL_NODES} expectation=warm",
            flush=True,
        )
        warm_started = perf_counter()
        warm = reference.cached_machine(case, requested, passive=True)
        warm_request_seconds = perf_counter() - warm_started
        print(reference.machine_cache_summary(name, warm), flush=True)
        if not warm.cache_receipt.hit:
            raise AssertionError(f"{name} warm request rebuilt instead of loading")
        arrays_verified, bytes_verified = (
            reference.assert_machine_arrays_bitwise_identical(cold, warm)
        )
        if cold.cache_receipt.key != warm.cache_receipt.key:
            raise AssertionError(
                f"{name} cold and warm requests selected different keys"
            )
        operator = reference.forward_operator(case, warm)
        operator_shape = {
            "grid_nodes": operator.grid.node_number,
            "wall_nodes": operator.wall.node_number,
            "sample_nodes": operator.sample.node_number,
            "state_nodes": operator.node_number,
        }

        fixture = {
            "requested_cells": requested,
            "realised_cells": len(cold.node),
            "wall_nodes_per_segment": reference.WALL_NODES,
            "cold_request_seconds": cold_request_seconds,
            "warm_request_seconds": warm_request_seconds,
            "warm_fraction_of_cold": warm_request_seconds / cold_request_seconds,
            "cold": _receipt(cold.cache_receipt),
            "warm": _receipt(warm.cache_receipt),
            "identity": {
                "same_key": True,
                "arrays_verified": arrays_verified,
                "bytes_verified": bytes_verified,
                "native_dtype_shape_and_bytes_identical": True,
            },
            "warm_operator_reconstructed": operator_shape,
        }
        report["fixtures"][name] = fixture
        _write(args.output, report)
        print(
            f"CACHE_MEASURED fixture={name} cold_s={cold_request_seconds:.9g} "
            f"warm_s={warm_request_seconds:.9g} "
            f"fraction={fixture['warm_fraction_of_cold']:.9g} "
            f"arrays={arrays_verified} bytes={bytes_verified} bitwise=True",
            flush=True,
        )

    report["all_warm_hits"] = all(
        fixture["warm"]["hit"] for fixture in report["fixtures"].values()
    )
    report["all_bitwise_stored_precision"] = all(
        fixture["identity"]["native_dtype_shape_and_bytes_identical"]
        for fixture in report["fixtures"].values()
    )
    _write(args.output, report)
    print("CACHE_MEASUREMENT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
