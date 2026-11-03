"""Measure typed forward solves while one MAST position coil is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import coil_edit_latency, mast_response_carrier_warm
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


def run(output: Path, carrier: Path) -> dict[str, object]:
    """Persist each warm edit immediately so an interrupted sweep is legible."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    coil_edit_latency._require_measurement_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    events = coil_edit_latency._cache_monitor()
    profile, prepared, carrier_record = coil_edit_latency._prepare_case(
        carrier, impose_vertical_centroid=False
    )
    state = prepared["initial"]
    base_current = np.asarray(prepared["prescribed_current"], dtype=np.float64)
    coil_index = int(prepared["circuit_index"])
    target = float(prepared["target_current"])
    fractions = np.linspace(-0.01, 0.01, 20)
    rows: list[dict[str, object]] = []
    document: dict[str, object] = {
        "shot": coil_edit_latency.SHOT,
        "slice_index": coil_edit_latency.SLICE_INDEX,
        "carrier": str(carrier),
        "carrier_identity": carrier_record.get("semantic_identity"),
        "coil": prepared["coil_mapping"],
        "cache_directory": str(cache.directory),
        "measurement_state": "running",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    first_program = None
    for edit_index, fraction in enumerate(fractions):
        current = base_current.copy()
        current[coil_index] *= 1.0 + fraction
        request = ForwardSolveRequest.from_defaults(
            carrier_identity=str(carrier),
            source_profile=profile.source,
            seed_policy=ExplicitSolveSeed(state),
            vertical_pin="omit",
            requested_class=TopologyClass.DIVERTED,
            prescribed_current=jnp.asarray(current),
            target_current=jnp.asarray(target),
            policy_overrides={
                "newton_steps": 12,
                "gmres_iterations": 12,
                "warmup": 0,
                "relaxation": 0.5,
                "step_cap": 10.0,
            },
        )
        hits_before = int(events["hits"])
        misses_before = int(events["misses"])
        started = time.perf_counter()
        receipt = profile.solve(request)
        jax.block_until_ready(receipt.equilibrium.flux)
        wall_ms = 1_000.0 * (time.perf_counter() - started)
        history = receipt.equilibrium.fixed_point
        state = receipt.equilibrium.flux
        program = next(iter(profile._accelerated_program_cache.values()))
        if first_program is None:
            first_program = program
        row = {
            "edit_index": edit_index,
            "fraction": float(fraction),
            "coil_current_a": float(current[coil_index]),
            "wall_milliseconds": wall_ms,
            "trip_count": int(np.asarray(history.active_set_iterations)),
            "converged": bool(np.asarray(history.converged)),
            "qualified": bool(np.asarray(receipt.qualified)),
            "terminal_residual": float(np.asarray(history.residual)),
            "process_program_reused": program is first_program,
            "request_cache_hit": bool(receipt.compilation_cache_hit),
            "persistent_hit_events": int(events["hits"]) - hits_before,
            "persistent_miss_events": int(events["misses"]) - misses_before,
        }
        rows.append(row)
        output.write_text(json.dumps(document, indent=2, default=str) + "\n")
        print("EDIT " + json.dumps(row, sort_keys=True), flush=True)
    warm_walls = [float(row["wall_milliseconds"]) for row in rows[1:]]
    document["summary"] = {
        "edit_count": len(rows),
        "median_warm_wall_milliseconds": float(np.median(warm_walls)),
        "converged_count": sum(bool(row["converged"]) for row in rows),
        "request_cache_hits_after_first": sum(
            bool(row["request_cache_hit"]) for row in rows[1:]
        ),
        "persistent_hit_events_after_first": sum(
            int(row["persistent_hit_events"]) for row in rows[1:]
        ),
        "program_reused_after_first": all(
            bool(row["process_program_reused"]) for row in rows[1:]
        ),
    }
    document["measurement_state"] = "complete"
    output.write_text(json.dumps(document, indent=2, default=str) + "\n")
    print("SUMMARY " + json.dumps(document["summary"], sort_keys=True), flush=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--carrier", type=Path, default=mast_response_carrier_warm.DEFAULT_CARRIER
    )
    arguments = parser.parse_args()
    run(arguments.output, arguments.carrier)


if __name__ == "__main__":
    main()
