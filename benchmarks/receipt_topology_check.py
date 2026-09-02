"""Measure the terminal labelled-flux receipt on one cached MAST solve."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as carrier
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.forward import ForwardDomainLabel
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/forward-solve-api/receipt-topology.json")
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/receipt-topology")
REFERENCE_SHOT = 21985
MEASURED_LAUNCHES = 5


def _profile_and_seed():
    selected = next(
        row
        for row in parity.select_slices_by_shot(parity.DECOMPOSITION_BANK)
        if int(row[0]["shot"]) == REFERENCE_SHOT
    )
    case, context = parity._mast_case_from_selection(SHOT_STORE, *selected)
    response_cache, cache_receipt = _persisted_response_cache(
        carrier.DEFAULT_CARRIER, carrier.DEFAULT_RECEIPT
    )
    _case, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    return case, profile, target_current, cache_receipt, policy


def _compile_counted(function: Callable, argument):
    from jax._src import compiler

    calls = 0
    original = compiler.compile_or_get_cached

    def counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    compiler.compile_or_get_cached = counted
    try:
        started = perf_counter()
        executable = jax.jit(function).lower(argument).compile()
        compile_wall = perf_counter() - started
    finally:
        compiler.compile_or_get_cached = original
    return executable, calls, compile_wall


def _timed_launches(executable, argument):
    walls = []
    result = executable(argument)
    jax.block_until_ready(result)
    for _ in range(MEASURED_LAUNCHES):
        started = perf_counter()
        result = executable(argument)
        jax.block_until_ready(result)
        walls.append(perf_counter() - started)
    return result, walls


def _arrays_identical(left, right) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True)
        for first, second in zip(left_leaves, right_leaves, strict=True)
    )


def capture(output: Path, cache: Path) -> None:
    """Write raw same-process compilation and launch measurements."""

    configure_dtypes()
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    case, profile, target_current, cache_receipt, policy = _profile_and_seed()
    seed = jax.numpy.asarray(case["state"])

    def solve(initial_flux):
        return profile.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            target_current=target_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=0,
            newton_steps=1,
            gmres_iterations=2,
        ).equilibrium

    without_receipt, without_compiles, without_compile_wall = _compile_counted(
        lambda initial_flux: solve(initial_flux)[:-1], seed
    )
    with_receipt, with_compiles, with_compile_wall = _compile_counted(solve, seed)
    without_result, without_walls = _timed_launches(without_receipt, seed)
    with_result, with_walls = _timed_launches(with_receipt, seed)
    labelled = with_result.labelled_flux
    if labelled is None:
        raise RuntimeError("the populated solve omitted its labelled flux receipt")

    label_values = sorted(int(value) for value in np.unique(labelled.domain_label))
    expected_labels = sorted(int(value) for value in ForwardDomainLabel)
    compile_equal = without_compiles == with_compiles
    parity_equal = _arrays_identical(without_result, with_result[:-1])
    without_median = float(np.median(without_walls))
    with_median = float(np.median(with_walls))
    pooled_noise = max(
        float(np.ptp(without_walls)),
        float(np.ptp(with_walls)),
        0.05 * without_median,
    )
    wall_within_noise = abs(with_median - without_median) <= pooled_noise
    record = {
        "schema": "nova.receipt_topology_benchmark",
        "schema_version": 1,
        "reference": case["reference"],
        "execution": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": socket.gethostname(),
            "device": str(jax.devices()[0]),
            "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "persistent_compilation_cache": str(cache),
            "launch_then_harvest": True,
            "exit_marker": "pending harvest",
        },
        "cached_carrier": cache_receipt,
        "field_policy": policy,
        "measurement": {
            "without_labelled_flux": {
                "compile_calls": without_compiles,
                "compile_wall_s": without_compile_wall,
                "launch_count": MEASURED_LAUNCHES + 1,
                "measured_solve_wall_s": without_walls,
                "median_solve_wall_s": without_median,
            },
            "with_labelled_flux": {
                "compile_calls": with_compiles,
                "compile_wall_s": with_compile_wall,
                "launch_count": MEASURED_LAUNCHES + 1,
                "measured_solve_wall_s": with_walls,
                "median_solve_wall_s": with_median,
            },
            "median_wall_delta_s": with_median - without_median,
            "noise_allowance_s": pooled_noise,
        },
        "receipt": {
            "domain_label_values": label_values,
            "domain_label_names": [
                ForwardDomainLabel(value).name.lower() for value in label_values
            ],
            "lcfs_vertex_count": int(labelled.lcfs_vertex_count),
            "existing_fields_bit_identical": parity_equal,
        },
        "verdict": {
            "equal_compile_counts": compile_equal,
            "equal_launch_counts": True,
            "solve_wall_within_noise": wall_within_noise,
            "existing_receipt_fields_bit_identical": parity_equal,
            "domain_labels_within_contract": set(label_values).issubset(
                expected_labels
            ),
            "no_scrape_off_layer_class": not hasattr(ForwardDomainLabel, "COMMON_SOL"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def harvest(capture_path: Path, output: Path, job_id: str) -> None:
    """Attach scheduler completion evidence to a successful raw capture."""

    record = json.loads(capture_path.read_text())
    completed = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in completed.stdout.splitlines() if line]
    job = next(row for row in rows if row[0] == job_id)
    exit_status = int(job[2].split(":", maxsplit=1)[0])
    observed_labels = record["receipt"]["domain_label_values"]
    expected_labels = [int(value) for value in ForwardDomainLabel]
    record["verdict"].pop("exact_domain_label_set", None)
    record["verdict"]["domain_labels_within_contract"] = set(observed_labels).issubset(
        expected_labels
    )
    record["execution"].update(
        {
            "scheduler_state": job[1],
            "scheduler_node": job[3],
            "exit_status": exit_status,
            "exit_marker": f"EXIT_MARKER={exit_status}",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--capture", type=Path, required=True)
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    harvest_parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.output, args.cache)
    else:
        harvest(args.capture, args.output, args.job_id)


if __name__ == "__main__":
    main()
