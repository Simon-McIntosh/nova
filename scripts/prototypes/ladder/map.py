"""Measure the certificate's forward map at its closed-form state."""

# Configure precision before importing array-valued module defaults.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from nova.equilibrium.topology import TopologyClass
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def measure(kind: str, cells: int, output: Path) -> dict:
    # Keep the existing semantic cache protocol, with its store rooted in the
    # granted run directory so a cold build cannot write into another scope.
    def scoped_store(*, filename, dirname, group=None):
        return ZarrStore(
            filename=filename, dirname=output.parent / "cache", group=group
        )

    oracle_fixture.ZarrStore = scoped_store
    print(f"MAP_STAGE case={kind} cells={cells} stage=machine", flush=True)
    case_name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    started = perf_counter()
    carrier, source, exact = certificate._case(case_name, clip_mode="exact")
    requested = -500 if cells == 550 else -cells
    machine = certificate._case_machine(
        case_name, carrier, exact, requested, clip_mode="exact"
    )
    build_wall = perf_counter() - started
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    if np.ptp(analytic[: len(machine.node)]) <= 1e-10:
        raise AssertionError("the analytic-state control is uniform")
    print(f"MAP_STAGE case={kind} cells={cells} stage=exterior", flush=True)
    started = perf_counter()
    empty = oracle_fixture.forward_operator(source, machine).with_clip_mode("exact")
    physical, exterior, exterior_cache = oracle_fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(
        source, machine, exterior
    ).with_clip_mode("exact")
    target, _centroid, _receipt = certificate._closed_form_current_target(
        case_name, source, operator, physical
    )
    fixture_wall = perf_counter() - started
    requested_class = (
        TopologyClass.LIMITED if kind == "limited" else TopologyClass.DIVERTED
    )
    mapped = operator.flux_map(requested_class=requested_class, target_current=target)
    state = jnp.asarray(analytic, dtype=jnp.float64)
    print(f"MAP_STAGE case={kind} cells={cells} stage=compile", flush=True)
    started = perf_counter()
    executable = jax.jit(mapped).lower(state).compile()
    compile_wall = perf_counter() - started
    span = abs(float(oracle_fixture._analytic_axis_flux(exact)))
    if span <= 0:
        raise AssertionError("the analytic span control is zero")
    memory = executable.memory_analysis()
    try:
        executable_bytes = len(executable.runtime_executable().serialize())
    except (AttributeError, RuntimeError):
        executable_bytes = None
    row = {
        "revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "module": str(Path(certificate.__file__).resolve()),
        "cwd": str(Path.cwd().resolve()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(jax.devices()[0]),
        "case": kind,
        "requested_cells": requested,
        "realised_cells": len(machine.node),
        "pitch_m": float(np.sqrt(np.median(np.asarray(machine.area)))),
        "map_relative_sup": None,
        "map_relative_rms": None,
        "machine_build_seconds": build_wall,
        "fixture_seconds": fixture_wall,
        "cold_compile_seconds": compile_wall,
        "warm_execute_seconds": None,
        "device_temp_bytes": memory.temp_size_in_bytes if memory else None,
        "device_output_bytes": memory.output_size_in_bytes if memory else None,
        "device_argument_bytes": memory.argument_size_in_bytes if memory else None,
        "serialized_executable_bytes": executable_bytes,
        "largest_array_intermediates": None,
        "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "machine_cache": machine.cache,
        "exterior_cache": exterior_cache,
        "arm_relation": (
            "both arms share this map; the prototype changes only the topology read"
        ),
        "completed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    row["largest_array_intermediates"] = certificate._largest_hlo_arrays(
        executable.as_text(), limit=3
    )
    output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    print(f"MAP_STAGE case={kind} cells={cells} stage=execute", flush=True)
    try:
        started = perf_counter()
        value = np.asarray(jax.block_until_ready(executable(state)))
        warm_wall = perf_counter() - started
        started = perf_counter()
        value = np.asarray(jax.block_until_ready(executable(state)))
        warm_wall = min(warm_wall, perf_counter() - started)
    except Exception as error:
        row["execution_error"] = f"{type(error).__name__}: {error}"
        output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
        raise
    difference = value[: len(machine.node)] - analytic[: len(machine.node)]
    row["map_relative_sup"] = float(np.max(np.abs(difference)) / span)
    row["map_relative_rms"] = float(np.sqrt(np.mean(difference**2)) / span)
    row["warm_execute_seconds"] = warm_wall
    row["completed"] = True
    output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = measure(args.kind, args.cells, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    print("MAP_ROW " + json.dumps(row, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
