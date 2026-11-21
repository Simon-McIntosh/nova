"""Measure the production point program's cached first call and its cold compile.

One process measures one arm, because an interpreter retains every executable it
compiles: a warm load and a cold compile of a second arm in the same process
would charge the second arm's compile to the first. The driver resolves the arm
through :mod:`arms`, configures the persistent cache the launch recipe selects
(``default_persistent_compilation_cache_root``), records one receipt per step and
writes the node rows an identity check compares across processes.

The receipt of a warm run carries the first call's wall and the process peak host
memory at the end of that call, which are the two numbers the point program's
load clause bounds. ``compare`` takes the cold and warm row files of one arm and
reports whether every value entry and every derivative entry is bit-identical.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402

from scripts.prototypes.cached_point_remeasure import arms  # noqa: E402

BYTE_BOUND = 64 * (2**30)
SECOND_BOUND = 60.0
TREE = Path(__file__).resolve().parents[3]


def revision() -> str:
    """Return the revision of the tree under test."""
    completed = subprocess.run(
        ["git", "-C", str(TREE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size so far."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def configure_cache() -> dict:
    """Select the persistent cache the launch recipe names, and require a card."""
    from nova.jax.config import (
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    devices = jax.devices()
    kind = devices[0].device_kind if devices else "none"
    if "H200" not in kind:
        raise SystemExit(
            "refusing to measure the load on a device that is not the H200: %s" % kind
        )
    selected = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    return {
        "device": kind,
        "platform": devices[0].platform,
        "cache_root": str(selected.root),
        "cache_directory": str(selected.directory),
        "version_key": selected.version_key,
        "cache_enabled": bool(jax.config.jax_compilation_cache_dir),
    }


def batch_loop(resolved: dict, batch: int):
    """Run the arm's batch loop and return the rows, walls and first-call facts."""
    nodes = resolved["nodes"]
    program = resolved["program"]
    coupling = resolved["coupling"]
    source_cells = resolved["source_cells"]
    rows = np.empty((len(nodes), source_cells, 3, 6), dtype=np.float64)
    walls = []
    first_seconds = None
    first_peak = 0
    for first, last, points in arms.batches(nodes, batch):
        started = perf_counter()
        result = program(jnp.asarray(points), coupling)
        jax.block_until_ready(result)
        walls.append(perf_counter() - started)
        rows[first:last] = np.asarray(result)[: last - first]
        if first_seconds is None:
            first_seconds = walls[-1]
            first_peak = peak_rss_bytes()
    return rows, walls, first_seconds, first_peak


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Report whether two float arrays carry the same bits, NaN payloads included."""
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.all(left.view(np.uint64) == right.view(np.uint64)))


def write_rows(
    path: Path, nodes: np.ndarray, rows: np.ndarray, digest: str, total: int
) -> None:
    """Write the compared node rows beside the digest of the whole loop's rows."""
    np.savez(
        path,
        nodes=np.asarray(nodes),
        rows=np.asarray(rows),
        digest=np.asarray(digest),
        total_nodes=np.asarray(int(total)),
    )


def measure(arguments: argparse.Namespace) -> int:
    """Measure one arm and record its receipt; enforce the cache miss budget."""
    import cache_guard

    fixture_started = perf_counter()
    resolved = arms.resolve(arguments.arm)
    fixture_seconds = perf_counter() - fixture_started
    header = configure_cache()
    cache_guard.install()
    ledger = cache_guard.ledger()
    batch = arms.ARM_BY_NAME[arguments.arm].batch
    loop_peak_before = peak_rss_bytes()
    rows, walls, first_seconds, first_peak = batch_loop(resolved, batch)
    digest = sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()
    subset = arms.desired_subset(resolved["nodes"], arguments.subset)
    if arguments.rows:
        write_rows(
            Path(arguments.rows),
            resolved["nodes"][subset],
            rows[subset],
            digest,
            len(resolved["nodes"]),
        )
    receipt = {
        "arm": arguments.arm,
        "role": arguments.role,
        "shape": resolved["shape"],
        "revision": revision(),
        "device": header["device"],
        "platform": header["platform"],
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "node": socket.gethostname(),
        "cache_root": header["cache_root"],
        "cache_directory": header["cache_directory"],
        "cache_enabled": header["cache_enabled"],
        "source_cells": resolved["source_cells"],
        "nodes": int(len(resolved["nodes"])),
        "batch": batch,
        "batches": len(walls),
        "fixture_seconds": round(fixture_seconds, 3),
        "first_call_seconds": round(float(first_seconds or 0.0), 6),
        "first_call_peak_rss_bytes": int(first_peak),
        "loop_seconds": round(float(sum(walls)), 6),
        "loop_peak_rss_bytes": peak_rss_bytes(),
        "peak_before_loop_rss_bytes": loop_peak_before,
        "batch_walls_seconds": [round(value, 6) for value in walls],
        "rows_digest": digest,
        "rows_compared": int(len(subset)),
    }
    receipt["hits"] = ledger.hit_count()
    receipt["misses"] = ledger.miss_count()
    receipt["compile_seconds"] = round(ledger.compile_seconds(), 3)
    console = cache_guard.emit_receipt(
        version_key=header["version_key"], revision=receipt["revision"]
    )
    receipt["miss_budget"] = console["miss_budget"]
    receipt["marker"] = console["marker"]
    receipt["unpersisted_misses"] = console["unpersisted_misses"]
    if arguments.receipt:
        Path(arguments.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print("MEASURED_ARM=%s" % json.dumps(receipt, sort_keys=True), flush=True)
    budget = console["miss_budget"]
    if receipt["unpersisted_misses"]:
        return 1
    if budget is not None and receipt["misses"] > budget:
        return 1
    return 0


def compare(arguments: argparse.Namespace) -> int:
    """Report bit identity between one arm's cold-compile and cached rows.

    Every derivative entry is compared, not a sample: the rows the loop returns
    carry the primal beside all five jet derivatives, so each slice is checked
    for the same bits on the same nodes.
    """
    cold = np.load(arguments.cold)
    warm = np.load(arguments.warm)
    cold_nodes, warm_nodes = cold["nodes"], warm["nodes"]
    cold_rows, warm_rows = cold["rows"], warm["rows"]
    values = bitwise_equal(cold_rows[..., 0], warm_rows[..., 0])
    derivatives = all(
        bitwise_equal(cold_rows[..., index], warm_rows[..., index])
        for index in range(1, cold_rows.shape[-1])
    )
    report = {
        "arm": arguments.arm,
        "cold": str(arguments.cold),
        "warm": str(arguments.warm),
        "nodes_compared": int(len(cold_nodes)),
        "arm_nodes": int(cold["total_nodes"]),
        "entries_compared": int(cold_rows.size),
        "node_coordinates_identical": bitwise_equal(cold_nodes, warm_nodes),
        "values_bitwise_identical": bool(values),
        "derivative_entries_bitwise_identical": bool(derivatives),
        "rows_digest_identical": bool(str(cold["digest"]) == str(warm["digest"])),
    }
    passed = (
        report["node_coordinates_identical"]
        and report["values_bitwise_identical"]
        and report["derivative_entries_bitwise_identical"]
        and report["rows_digest_identical"]
    )
    report["identity"] = "bit-identical" if passed else "differs"
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n")
    print("IDENTITY_ARM=%s" % json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure", help="measure one arm in this process")
    run.add_argument("--arm", required=True, choices=sorted(arms.ARM_BY_NAME))
    run.add_argument("--receipt", required=True, help="JSON receipt to write")
    run.add_argument("--rows", default="", help="npz rows for the identity check")
    run.add_argument(
        "--role", default="", help="name this process: control, fill or warm"
    )
    run.add_argument(
        "--subset",
        type=int,
        default=None,
        help="store this many nodes spread over the grid instead of all",
    )
    run.set_defaults(handler=measure)
    both = commands.add_parser("compare", help="compare cold and warm rows")
    both.add_argument("--arm", required=True, choices=sorted(arms.ARM_BY_NAME))
    both.add_argument("--cold", required=True)
    both.add_argument("--warm", required=True)
    both.add_argument("--output", required=True)
    both.set_defaults(handler=compare)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())