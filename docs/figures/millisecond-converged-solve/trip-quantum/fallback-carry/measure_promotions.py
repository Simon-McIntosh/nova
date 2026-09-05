"""Measure promotion walls after unchanged-state fallback carry activates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import UTC, datetime
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time

import jax
import jax.numpy as jnp

from benchmarks import compiled_outer_loop_trace as outer_trace
from benchmarks import late_trip_promotion_timing as timing
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


_EVENTS: list[dict[str, object]] = []
_EVENT_LOG: Path | None = None


def _strict(value):
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if hasattr(value, "item"):
        return _strict(value.item())
    if isinstance(value, float) and not jnp.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_event(kind, trip, promotion, accepted=False):
    event = {
        "kind": kind,
        "timestamp_ns": time.perf_counter_ns(),
        "trip": int(trip) + 1,
        "promotion": int(promotion) + 1,
        "accepted": bool(accepted),
    }
    _EVENTS.append(event)
    if _EVENT_LOG is None:
        raise RuntimeError("promotion event log is not configured")
    with _EVENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_start(trip, promotion):
    _append_event("promotion_start", trip, promotion)


def _record_end(trip, promotion, accepted):
    _append_event("promotion_end", trip, promotion, accepted)


def _instrumented_inner():
    source = textwrap.dedent(inspect.getsource(fixed_point._newton_krylov_inner))
    source = timing._replace_once(
        source,
        "def _newton_krylov_inner(",
        "def _timed_newton_krylov_inner(",
    )
    source = timing._replace_once(
        source,
        "    carry_unchanged_fallback: bool = True,\n"
        "    precision: Precision | str = Precision.AUTOMATIC,\n",
        "    carry_unchanged_fallback: bool = True,\n"
        "    benchmark_trip_index: jax.Array | int = 0,\n"
        "    precision: Precision | str = Precision.AUTOMATIC,\n",
    )
    source = timing._replace_once(
        source,
        "        def attempt_step(measured):\n            def linear_action(vector):\n",
        "        def attempt_step(measured):\n"
        "            jax.debug.callback(\n"
        "                _record_start, benchmark_trip_index, measured.attempted,\n"
        "                ordered=True,\n"
        "            )\n"
        "            def linear_action(vector):\n",
    )
    source = timing._replace_once(
        source,
        "                    stream_inner_iterations=stream_inner_iterations,\n"
        "                )\n\n"
        "                full_fallback_refused =",
        "                    stream_inner_iterations=stream_inner_iterations,\n"
        "                )\n"
        "                jax.debug.callback(\n"
        "                    _record_end, benchmark_trip_index, measured.attempted,\n"
        "                    first.inner_trace.accepted[measured.attempted],\n"
        "                    ordered=True,\n"
        "                )\n\n"
        "                full_fallback_refused =",
    )
    source = timing._replace_once(
        source,
        "                    def carried_attempt(carried):\n"
        "                        carried_qualified =",
        "                    def carried_attempt(carried):\n"
        "                        jax.debug.callback(\n"
        "                            _record_start, benchmark_trip_index,\n"
        "                            carried.attempted, ordered=True,\n"
        "                        )\n"
        "                        carried_qualified =",
    )
    source = timing._replace_once(
        source,
        "                        return _complete_newton_promotion(\n"
        "                            carried,",
        "                        completed = _complete_newton_promotion(\n"
        "                            carried,",
    )
    source = timing._replace_once(
        source,
        "                            reuse_rejected_score=True,\n"
        "                        )\n\n"
        "                    return jax.lax.while_loop(",
        "                            reuse_rejected_score=True,\n"
        "                        )\n"
        "                        jax.debug.callback(\n"
        "                            _record_end, benchmark_trip_index,\n"
        "                            carried.attempted,\n"
        "                            completed.inner_trace.accepted[\n"
        "                                carried.attempted\n"
        "                            ],\n"
        "                            ordered=True,\n"
        "                        )\n"
        "                        return completed\n\n"
        "                    return jax.lax.while_loop(",
    )
    namespace = dict(vars(fixed_point))
    namespace.update({"_record_start": _record_start, "_record_end": _record_end})
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_timed_newton_krylov_inner"]


@contextmanager
def _timed_solver():
    original = fixed_point._active_set_newton_krylov
    fixed_point._active_set_newton_krylov = timing._instrumented_active_set(
        _instrumented_inner()
    )
    try:
        yield
    finally:
        fixed_point._active_set_newton_krylov = original


def _promotion_rows() -> list[dict[str, object]]:
    grouped = defaultdict(dict)
    for event in _EVENTS:
        key = (event["trip"], event["promotion"])
        grouped[key][event["kind"]] = event
    rows = []
    for (trip, promotion), events in sorted(grouped.items()):
        if set(events) != {"promotion_start", "promotion_end"}:
            raise RuntimeError(
                f"incomplete promotion boundary for {(trip, promotion)}: {set(events)}"
            )
        start = events["promotion_start"]
        end = events["promotion_end"]
        rows.append(
            {
                "trip": trip,
                "promotion": promotion,
                "wall_s": (end["timestamp_ns"] - start["timestamp_ns"]) * 1.0e-9,
                "accepted": end["accepted"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, table: Path, event_log: Path, cache_root: Path) -> None:
    global _EVENT_LOG
    configure_dtypes()
    outer_trace._require_measurement_host()
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, _carrier, _policy = _profile_and_seed()
    state = jnp.asarray(case["state"])

    production = jax.jit(timing._production_program(profile, target_current))
    executable = production.lower(state).compile()
    started = time.perf_counter()
    result = jax.block_until_ready(executable(state))
    production_wall = time.perf_counter() - started
    production_summary = outer_trace._solve_summary(result)

    with _timed_solver():
        jax.clear_caches()
        instrumented = jax.jit(timing._production_program(profile, target_current))
        instrumented_executable = instrumented.lower(state).compile()
        event_log.parent.mkdir(parents=True, exist_ok=True)
        event_log.write_text("", encoding="utf-8")
        _EVENT_LOG = event_log
        _EVENTS.clear()
        started = time.perf_counter()
        instrumented_result = jax.block_until_ready(instrumented_executable(state))
        instrumented_wall = time.perf_counter() - started
        instrumented_summary = outer_trace._solve_summary(instrumented_result)

    if production_summary != instrumented_summary:
        raise RuntimeError(
            "promotion timing changed the solve summary: "
            f"{production_summary} != {instrumented_summary}"
        )
    rows = _promotion_rows()
    if len(rows) != production_summary["attempted_newton_promotions"]:
        raise RuntimeError(
            f"promotion census {len(rows)} != "
            f"{production_summary['attempted_newton_promotions']}"
        )
    trip_rows = []
    for trip in range(1, production_summary["active_set_trips"] + 1):
        selected = [row for row in rows if row["trip"] == trip]
        trip_rows.append(
            {
                "trip": trip,
                "promotions": len(selected),
                "accepted_promotions": sum(row["accepted"] for row in selected),
                "promotion_wall_s": sum(row["wall_s"] for row in selected),
            }
        )
    _write_csv(table, rows)
    _write_json(
        output,
        {
            "schema": "nova.unchanged_state_fallback_promotion_timing",
            "captured_at": datetime.now(UTC).isoformat(),
            "measurement_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
            "persistent_compilation_cache": cache,
            "production_wall_s": production_wall,
            "instrumented_wall_s": instrumented_wall,
            "summary": production_summary,
            "per_trip": trip_rows,
            "promotion_table": str(table),
            "event_log": str(event_log),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output, arguments.table, arguments.event_log, arguments.cache_root)


if __name__ == "__main__":
    main()
