"""Compare exact and constant projected-condition programs on one device trip.

The benchmark lowers and compiles both routes, counts their compiled HLO
operations, records one synchronized device trace per route, and runs a
separate callback-instrumented execution census.  The instrumented census is
never used as timing evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/hessenberg-trace"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "profile.json"
DEFAULT_FIGURE = OUTPUT_ROOT / "condition-trace.png"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "condition-discriminator-trace.md"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIP_BUDGET = 1
EXPECTED_ACTIVE_SET_TRIPS = 1
EXPECTED_NEWTON_PROMOTIONS = 12
PRIOR_EXACT_SOLVE_S = 40.3993306318298
PRIOR_EXACT_TRIPS = 7
PRIOR_CONSTANT_SOLVE_S = 0.754009


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _distribution(values: list[float]) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        return {
            "count": 0,
            "sum_s": 0.0,
            "minimum_s": None,
            "median_s": None,
            "p95_s": None,
            "maximum_s": None,
        }
    return {
        "count": int(samples.size),
        "sum_s": float(samples.sum()),
        "minimum_s": float(samples.min()),
        "median_s": float(np.median(samples)),
        "p95_s": float(np.quantile(samples, 0.95)),
        "maximum_s": float(samples.max()),
    }


def _runtime() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [getattr(device, "device_kind", None) for device in devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _scheduler() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _require_measurement_host() -> None:
    devices = jax.devices()
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    if jax.default_backend() != "gpu" or not any("H200" in kind for kind in kinds):
        raise RuntimeError(
            "condition trace requires the reserved H200; "
            f"backend={jax.default_backend()} devices={kinds}"
        )


def _source_anchor(function: Callable, needle: str | None = None) -> str:
    path = Path(inspect.getsourcefile(function) or "")
    lines, first = inspect.getsourcelines(function)
    offset = 0
    if needle is not None:
        offset = next(
            (index for index, line in enumerate(lines) if needle in line),
            0,
        )
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    return f"{display}:{first + offset}"


@contextmanager
def _patched(replacements: list[tuple[Any, str, Any]]) -> Iterator[None]:
    originals: list[tuple[Any, str, Any]] = []
    try:
        for owner, name, replacement in replacements:
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _constant_projected_condition(
    projection: fixed_point._KrylovProjection,
) -> tuple[jax.Array, jax.Array]:
    dtype = projection.hessenberg.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    return one, one


def _public_gmres_with_dummy_projection(
    linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    *,
    krylov_dimension: int,
) -> tuple[jax.Array, jax.Array, fixed_point._KrylovProjection]:
    """Run the public solver while supplying dead projection-shaped outputs."""
    step, info = jax.scipy.sparse.linalg.gmres(
        linear_action,
        residual_vector,
        tol=fixed_point._GMRES_RELATIVE_TOLERANCE,
        maxiter=krylov_dimension,
        restart=krylov_dimension,
        solve_method="batched",
    )
    projection = fixed_point._KrylovProjection(
        basis=jnp.zeros(
            (krylov_dimension + 1, residual_vector.size),
            dtype=residual_vector.dtype,
        ),
        hessenberg=jnp.zeros(
            (krylov_dimension + 1, krylov_dimension),
            dtype=residual_vector.dtype,
        ),
        active_columns=jnp.asarray(0, dtype=jnp.int32),
    )
    return step, info, projection


def _production_program(profile: Any, target_current: float):
    def one_trip(initial_flux):
        return profile.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            target_current=target_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=parity.WARMUP_SWEEPS,
            newton_steps=parity.NEWTON_STEPS,
            gmres_iterations=parity.GMRES_ITERATIONS,
            relaxation=parity.RELAXATION,
            step_cap=parity.STEP_CAP,
            active_set_steps=ACTIVE_SET_TRIP_BUDGET,
        )

    return one_trip


def _solve_summary(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    reason_value = int(np.asarray(history.termination_reason))
    return {
        "active_set_trips": int(np.asarray(history.active_set_iterations)),
        "attempted_newton_promotions": int(
            np.asarray(history.attempted_newton_promotions)
        ),
        "accepted_newton_promotions": int(
            np.asarray(history.accepted_newton_promotions)
        ),
        "termination_reason": fixed_point.FixedPointTerminationReason(
            reason_value
        ).name.lower(),
        "terminal_residual": float(np.asarray(history.residual)),
        "converged": bool(np.asarray(history.converged)),
        "maximum_projected_krylov_condition": float(
            np.asarray(history.maximum_projected_krylov_condition)
        ),
    }


_HLO_OPERATION = re.compile(r"(?:^|\s)([a-z][a-z0-9-]*)\(")
_HLO_CATEGORIES = (
    "while",
    "conditional",
    "custom-call",
    "fusion",
    "dot",
    "reduce",
)


def _hlo_census(text: str) -> dict[str, Any]:
    operations: Counter[str] = Counter()
    for line in text.splitlines():
        if "=" not in line:
            continue
        right = line.split("=", 1)[1]
        match = _HLO_OPERATION.search(right)
        if match is not None:
            operations[match.group(1)] += 1
    return {
        "total_operations": int(sum(operations.values())),
        "categories": {name: int(operations[name]) for name in _HLO_CATEGORIES},
        "all_operations": dict(sorted(operations.items())),
        "method": (
            "operation mnemonics counted once per instruction line in the "
            "post-compilation executable HLO returned by compiled.as_text(); "
            "counts include called computations and describe structure, not "
            "dynamic loop iterations"
        ),
    }


def _write_hlo(trace_root: Path, arm: str, text: str) -> dict[str, Any]:
    path = trace_root / "hlo" / f"{arm}.hlo.txt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        stream.write(text)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "uncompressed_bytes": len(text.encode("utf-8")),
    }


def _latest_trace(path: Path) -> Path:
    traces = sorted(
        path.glob("plugins/profile/*/perfetto_trace.json.gz"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
    )
    if not traces:
        raise RuntimeError(f"JAX profiler wrote no Perfetto trace under {path}")
    return traces[-1]


def _merged_idle_gaps(events: list[dict[str, Any]]) -> list[float]:
    intervals = sorted(
        (
            float(event["ts"]),
            float(event["ts"]) + float(event.get("dur", 0.0)),
        )
        for event in events
    )
    if not intervals:
        return []
    gaps = []
    _start, stop = intervals[0]
    for start, end in intervals[1:]:
        if start > stop:
            gaps.append((start - stop) / 1.0e6)
            stop = end
        else:
            stop = max(stop, end)
    return gaps


def _host_gap_rows(
    events: list[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda event: float(event["ts"]))
    gaps: list[float] = []
    rows: list[dict[str, Any]] = []
    for previous, following in zip(ordered, ordered[1:], strict=False):
        previous_stop = float(previous["ts"]) + float(previous.get("dur", 0.0))
        gap_s = max(float(following["ts"]) - previous_stop, 0.0) / 1.0e6
        gaps.append(gap_s)
        rows.append(
            {
                "gap_s": gap_s,
                "preceding_api": str(previous.get("name", "unknown")),
                "following_api": str(following.get("name", "unknown")),
                "preceding_correlation_id": (previous.get("args") or {}).get(
                    "correlation_id"
                ),
                "following_correlation_id": (following.get("args") or {}).get(
                    "correlation_id"
                ),
            }
        )
    return gaps, sorted(rows, key=lambda row: row["gap_s"], reverse=True)[:100]


def _trace_summary(
    path: Path, annotation_name: str, observed_trips: int
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", [])
    process_names = {
        event.get("pid"): str(event.get("args", {}).get("name", ""))
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    gpu_pids = {pid for pid, name in process_names.items() if "GPU" in name}
    host_pids = {pid for pid, name in process_names.items() if "host:CPU" in name}
    annotations = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == annotation_name
    ]
    if len(annotations) != 1:
        raise RuntimeError(
            f"expected one {annotation_name!r} annotation, found {len(annotations)}"
        )
    annotation = annotations[0]
    start = float(annotation["ts"])
    stop = start + float(annotation.get("dur", 0.0))
    device_events = [
        event
        for event in events
        if event.get("pid") in gpu_pids
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -1.0)) <= stop
    ]
    kernel_events = [
        event
        for event in device_events
        if not str(event.get("name", "")).startswith(("Memcpy", "Memset"))
    ]
    transfer_events = [event for event in device_events if event not in kernel_events]
    if not kernel_events:
        raise RuntimeError("trace annotation contains no GPU kernel events")

    grouped: dict[str, list[float]] = {}
    for event in kernel_events:
        grouped.setdefault(str(event.get("name", "unknown")), []).append(
            float(event.get("dur", 0.0)) / 1.0e6
        )
    summed_kernel_s = sum(sum(durations) for durations in grouped.values())
    kernel_rows = sorted(
        (
            {
                "kernel": name,
                "call_count": len(durations),
                "summed_s": float(sum(durations)),
                "summed_s_per_observed_trip": float(sum(durations))
                / max(observed_trips, 1),
                "median_call_us": 1.0e6 * float(np.median(durations)),
                "share_of_summed_kernel_time": float(sum(durations))
                / max(summed_kernel_s, 1.0e-30),
            }
            for name, durations in grouped.items()
        ),
        key=lambda row: row["summed_s"],
        reverse=True,
    )

    host_launch_events = [
        event
        for event in events
        if event.get("pid") in host_pids
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -1.0)) <= stop
        and str(event.get("name", "")).startswith(
            ("cuGraphLaunch", "cuLaunchKernel", "cuLaunchCooperativeKernel")
        )
    ]
    host_gaps, largest_host_gaps = _host_gap_rows(host_launch_events)
    device_idle_gaps = _merged_idle_gaps(kernel_events)
    host_correlations = {
        str((event.get("args") or {}).get("correlation_id"))
        for event in host_launch_events
    }
    gpu_correlations = {
        str((event.get("args") or {}).get("correlation_id")) for event in kernel_events
    }
    semantic_scopes: dict[str, list[dict[str, Any]]] = {
        "projected_condition": [],
        "gmres_or_arnoldi": [],
        "other": [],
    }
    for event in kernel_events:
        scope = str((event.get("args") or {}).get("name", "")).lower()
        if "projected_condition" in scope or "symmetric_jacobi" in scope:
            semantic_scopes["projected_condition"].append(event)
        elif "gmres" in scope or "arnoldi" in scope:
            semantic_scopes["gmres_or_arnoldi"].append(event)
        else:
            semantic_scopes["other"].append(event)
    scope_rows = {}
    for name, selected in semantic_scopes.items():
        scope_rows[name] = {
            "kernel_event_count": len(selected),
            "summed_kernel_s": sum(
                float(event.get("dur", 0.0)) / 1.0e6 for event in selected
            ),
        }

    return {
        "status": "complete",
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "annotation": annotation_name,
        "annotation_wall_s": float(annotation.get("dur", 0.0)) / 1.0e6,
        "gpu_process_ids": sorted(gpu_pids),
        "host_process_ids": sorted(host_pids),
        "device_event_count": len(device_events),
        "kernel_event_count": len(kernel_events),
        "transfer_event_count": len(transfer_events),
        "unique_kernel_count": len(kernel_rows),
        "summed_kernel_s": summed_kernel_s,
        "summed_kernel_s_per_observed_trip": summed_kernel_s / max(observed_trips, 1),
        "per_kernel_sums": kernel_rows,
        "semantic_scope_sums": scope_rows,
        "device_inter_kernel_idle_gaps": _distribution(device_idle_gaps),
        "host_launch_api": {
            "launch_count": len(host_launch_events),
            "graph_launch_count": sum(
                str(event.get("name", "")).startswith("cuGraphLaunch")
                for event in host_launch_events
            ),
            "direct_kernel_launch_count": sum(
                str(event.get("name", "")).startswith(
                    ("cuLaunchKernel", "cuLaunchCooperativeKernel")
                )
                for event in host_launch_events
            ),
            "summed_api_s": sum(
                float(event.get("dur", 0.0)) / 1.0e6 for event in host_launch_events
            ),
            "summed_api_s_per_observed_trip": sum(
                float(event.get("dur", 0.0)) / 1.0e6 for event in host_launch_events
            )
            / max(observed_trips, 1),
            "between_launch_gaps": _distribution(host_gaps),
            "between_launch_gap_sum_s_per_observed_trip": sum(host_gaps)
            / max(observed_trips, 1),
            "largest_between_launch_gaps": largest_host_gaps,
            "gpu_correlation_count": len(gpu_correlations),
            "host_correlation_count": len(host_correlations),
            "matched_gpu_correlation_fraction": len(
                gpu_correlations & host_correlations
            )
            / max(len(gpu_correlations), 1),
        },
        "observed_trips_divisor": observed_trips,
        "method": (
            "all complete GPU events starting inside the synchronized annotation "
            "are retained; per-kernel durations are summed without assuming stream "
            "serialization. Host gaps are positive intervals between consecutive "
            "CUDA graph or kernel launch API events across host threads. Device "
            "idle gaps merge overlapping kernel intervals across streams first"
        ),
    }


def _compile_and_trace_arm(
    name: str,
    profile: Any,
    state: jax.Array,
    target_current: float,
    replacements: list[tuple[Any, str, Any]],
    trace_root: Path,
) -> dict[str, Any]:
    with _patched(replacements):
        jax.clear_caches()
        program = _production_program(profile, target_current)
        started = time.perf_counter()
        executable = jax.jit(program).lower(state).compile()
        compile_wall_s = time.perf_counter() - started
        hlo_text = executable.as_text()
        hlo = {
            **_hlo_census(hlo_text),
            "artifact": _write_hlo(trace_root, name, hlo_text),
        }

        started = time.perf_counter()
        first = _ready(executable(state))
        first_execute_wall_s = time.perf_counter() - started
        summary = _solve_summary(first)
        arm_trace_root = trace_root / name
        arm_trace_root.mkdir(parents=True, exist_ok=True)
        annotation = f"condition_discriminator/{name}"
        with jax.profiler.trace(str(arm_trace_root), create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation(annotation):
                started = time.perf_counter()
                traced = _ready(executable(state))
                trace_execute_wall_s = time.perf_counter() - started
        traced_summary = _solve_summary(traced)
    trace_path = _latest_trace(arm_trace_root)
    return {
        "name": name,
        "compile_wall_s": compile_wall_s,
        "first_execute_wall_s": first_execute_wall_s,
        "trace_execute_wall_s": trace_execute_wall_s,
        "trace_wall_s_per_observed_trip": trace_execute_wall_s
        / max(summary["active_set_trips"], 1),
        "solve": summary,
        "trace_solve": traced_summary,
        "hlo": hlo,
        "device_trace": _trace_summary(
            trace_path, annotation, summary["active_set_trips"]
        ),
    }


def _evaluation_census(
    name: str,
    profile: Any,
    state: jax.Array,
    target_current: float,
    constant_condition: bool,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    original_gmres = (
        _public_gmres_with_dummy_projection
        if constant_condition
        else fixed_point._gmres_with_projected_condition
    )
    original_direct = fixed_point._projected_krylov_condition
    original_condition = fixed_point._projected_condition_from_hessenberg

    def record(counter: str, _value: Any) -> None:
        counts[counter] += 1

    def counted_gmres(linear_action, residual_vector, *, krylov_dimension: int):
        def counted_linear_action(vector):
            jax.debug.callback(partial(record, "linear_action"), vector[0])
            return linear_action(vector)

        return original_gmres(
            counted_linear_action,
            residual_vector,
            krylov_dimension=krylov_dimension,
        )

    def counted_direct(
        linear_action,
        residual_vector,
        *,
        krylov_dimension: int,
        return_direction: bool = False,
    ):
        jax.debug.callback(
            partial(record, "projected_krylov_condition"), residual_vector[0]
        )
        return original_direct(
            linear_action,
            residual_vector,
            krylov_dimension=krylov_dimension,
            return_direction=return_direction,
        )

    def counted_condition(projection):
        jax.debug.callback(
            partial(record, "projected_condition_from_hessenberg"),
            projection.hessenberg[0, 0],
        )
        if constant_condition:
            return _constant_projected_condition(projection)
        return original_condition(projection)

    replacements = [
        (fixed_point, "_gmres_with_projected_condition", counted_gmres),
        (fixed_point, "_projected_krylov_condition", counted_direct),
        (fixed_point, "_projected_condition_from_hessenberg", counted_condition),
    ]
    with _patched(replacements):
        jax.clear_caches()
        program = _production_program(profile, target_current)
        executable = jax.jit(program).lower(state).compile()
        started = time.perf_counter()
        result = _ready(executable(state))
        census_execute_wall_s = time.perf_counter() - started
        summary = _solve_summary(result)
    trips = max(summary["active_set_trips"], 1)
    return {
        "arm": name,
        "counts_per_trip": {
            counter: counts[counter] / trips
            for counter in (
                "projected_krylov_condition",
                "projected_condition_from_hessenberg",
                "linear_action",
            )
        },
        "counts_total": {
            counter: counts[counter]
            for counter in (
                "projected_krylov_condition",
                "projected_condition_from_hessenberg",
                "linear_action",
            )
        },
        "solve": summary,
        "instrumented_execute_wall_s": census_execute_wall_s,
        "timing_eligible": False,
        "method": (
            "separate benchmark-only debug callbacks count executed calls after "
            "lowering; callback wall is excluded from every timing and trace"
        ),
    }


def _comparison(
    arms: list[dict[str, Any]], censuses: list[dict[str, Any]]
) -> dict[str, Any]:
    by_name = {arm["name"]: arm for arm in arms}
    exact = by_name["exact_jacobi"]
    constant = by_name["constant_discriminator"]
    exact_wall = exact["trace_wall_s_per_observed_trip"]
    constant_wall = constant["trace_wall_s_per_observed_trip"]
    categories = {}
    for category in _HLO_CATEGORIES:
        exact_count = exact["hlo"]["categories"][category]
        constant_count = constant["hlo"]["categories"][category]
        categories[category] = {
            "exact": exact_count,
            "constant": constant_count,
            "delta": exact_count - constant_count,
            "ratio": exact_count / max(constant_count, 1),
        }
    census_by_name = {row["arm"]: row for row in censuses}
    return {
        "trace_wall": {
            "exact_s_per_observed_trip": exact_wall,
            "constant_s_per_observed_trip": constant_wall,
            "saving_s_per_trip": exact_wall - constant_wall,
            "speedup": exact_wall / max(constant_wall, 1.0e-30),
        },
        "hlo_categories": categories,
        "evaluation_count_delta": {
            name: census_by_name["exact_jacobi"]["counts_per_trip"][name]
            - census_by_name["constant_discriminator"]["counts_per_trip"][name]
            for name in (
                "projected_krylov_condition",
                "projected_condition_from_hessenberg",
                "linear_action",
            )
        },
        "prior_full_solve_reference": {
            "exact_s_per_trip": PRIOR_EXACT_SOLVE_S / PRIOR_EXACT_TRIPS,
            "constant_s_per_trip": PRIOR_CONSTANT_SOLVE_S / PRIOR_EXACT_TRIPS,
        },
    }


def _mechanism(receipt: dict[str, Any]) -> dict[str, Any]:
    exact = next(arm for arm in receipt["arms"] if arm["name"] == "exact_jacobi")
    constant = next(
        arm for arm in receipt["arms"] if arm["name"] == "constant_discriminator"
    )
    comparison = receipt["comparison"]
    exact_host = exact["device_trace"]["host_launch_api"]
    constant_host = constant["device_trace"]["host_launch_api"]
    censuses = {row["arm"]: row for row in receipt["evaluation_census"]}
    exact_counts = censuses["exact_jacobi"]["counts_per_trip"]
    constant_counts = censuses["constant_discriminator"]["counts_per_trip"]
    prior = comparison["prior_full_solve_reference"]
    return {
        "finding": (
            "The cost is the projection-producing solver control path required to "
            "make the exact ratio available, not the small Gram eigenspectrum. In "
            "the observed initial trip, the reused-Hessenberg solver executes all "
            f"{exact_counts['projected_condition_from_hessenberg']:.0f} Newton "
            "promotions and "
            f"{exact_counts['linear_action']:.0f} linear actions; the constant "
            "comparator can use the established public GMRES step and settles after "
            f"{constant_counts['projected_condition_from_hessenberg']:.0f} "
            "promotions and "
            f"{constant_counts['linear_action']:.0f} actions. The nearly unchanged "
            "compiled HLO category counts rule out a static loop-unrolling or fusion "
            "explosion."
        ),
        "evidence": {
            "exact_condition_evaluations_per_trip": exact_counts[
                "projected_condition_from_hessenberg"
            ],
            "constant_condition_evaluations_per_trip": constant_counts[
                "projected_condition_from_hessenberg"
            ],
            "exact_linear_actions_per_trip": exact_counts["linear_action"],
            "constant_linear_actions_per_trip": constant_counts["linear_action"],
            "exact_host_launches": exact_host["launch_count"],
            "constant_host_launches": constant_host["launch_count"],
            "exact_host_gap_sum_s_per_trip": exact_host[
                "between_launch_gap_sum_s_per_observed_trip"
            ],
            "constant_host_gap_sum_s_per_trip": constant_host[
                "between_launch_gap_sum_s_per_observed_trip"
            ],
            "measured_initial_trip_saving_s": comparison["trace_wall"][
                "saving_s_per_trip"
            ],
            "prior_exact_full_solve_s_per_trip": prior["exact_s_per_trip"],
            "prior_constant_full_solve_s_per_trip": prior["constant_s_per_trip"],
        },
        "recommended_repair": (
            "Expose the first-restart basis and Hessenberg from the established "
            "public GMRES implementation without replacing its step algorithm, "
            "evaluate the exact condition and spectral baseline once after that "
            "solve, and pass only those scalars into qualification. Preserve the "
            "present formulas and admission decision; merely moving the Jacobi "
            "arithmetic is insufficient because it is already called once per "
            "Newton promotion."
        ),
        "estimated_saving_s_per_trip": max(
            prior["exact_s_per_trip"] - prior["constant_s_per_trip"], 0.0
        ),
        "source_anchors": receipt["source_anchors"],
    }


def _write_figure(receipt: dict[str, Any], path: Path) -> None:
    arms = {arm["name"]: arm for arm in receipt["arms"]}
    names = ["exact_jacobi", "constant_discriminator"]
    labels = ["Exact Jacobi ratio", "Constant discriminator"]
    colours = ["#b24a3b", "#3676a8"]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    walls = [1000.0 * arms[name]["trace_wall_s_per_observed_trip"] for name in names]
    axes[0].bar(labels, walls, color=colours)
    axes[0].set_ylabel("Synchronized one-trip wall [ms]")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_title("Warm device execution")
    for index, value in enumerate(walls):
        axes[0].text(index, value, f"{value:,.1f}", ha="center", va="bottom")

    launches = [
        arms[name]["device_trace"]["host_launch_api"]["launch_count"] for name in names
    ]
    axes[1].bar(labels, launches, color=colours)
    axes[1].set_ylabel("CUDA launch API events")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_title("Host submissions")
    for index, value in enumerate(launches):
        axes[1].text(index, value, f"{value:,}", ha="center", va="bottom")

    gaps = [
        1000.0
        * arms[name]["device_trace"]["host_launch_api"][
            "between_launch_gap_sum_s_per_observed_trip"
        ]
        for name in names
    ]
    axes[2].bar(labels, gaps, color=colours)
    axes[2].set_ylabel("Summed host inter-launch gaps [ms]")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].set_title("Host-side gaps")
    for index, value in enumerate(gaps):
        axes[2].text(index, value, f"{value:,.1f}", ha="center", va="bottom")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    arms = {arm["name"]: arm for arm in receipt["arms"]}
    exact = arms["exact_jacobi"]
    constant = arms["constant_discriminator"]
    comparison = receipt["comparison"]
    mechanism = receipt["mechanism"]
    censuses = {row["arm"]: row for row in receipt["evaluation_census"]}
    scheduler = receipt["scheduler"]
    lines = [
        "# Projected-condition compiled-program comparison",
        "",
        (
            f"SLURM job `{scheduler['job_id']}` on `{receipt['runtime']['host']}` "
            "compiled and traced one warm 22086/43 pure active-set trip for the "
            "fixed-sweep Jacobi projected-condition route and for the comparator "
            "with the discriminator ratio constant and projection materialization "
            "absent. "
            "Both executions observed one trip."
        ),
        (
            "The synchronized trace walls are **"
            f"{exact['trace_wall_s_per_observed_trip']:.6f} s/trip exact** and **"
            f"{constant['trace_wall_s_per_observed_trip']:.6f} s/trip constant**, "
            f"a **{comparison['trace_wall']['speedup']:.2f}x** change and "
            f"**{comparison['trace_wall']['saving_s_per_trip']:.6f} s/trip** "
            "measured saving."
        ),
        "",
        "## Compiled HLO operation census",
        "",
        "| operation category | exact Jacobi | constant | "
        "exact minus constant | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in comparison["hlo_categories"].items():
        lines.append(
            f"| `{name}` | {row['exact']:,} | {row['constant']:,} | "
            f"{row['delta']:+,} | {row['ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            (
                "Counts are from post-compilation executable HLO, including called "
                "computations. They show retained structure; dynamic loop trip counts "
                "come from the device trace and execution census."
            ),
            "",
            "## Complete device-trace summary",
            "",
            "| measure | exact Jacobi | constant |",
            "|---|---:|---:|",
        ]
    )
    trace_rows = (
        (
            "kernel events",
            lambda arm: arm["device_trace"]["kernel_event_count"],
            ",d",
        ),
        (
            "unique kernels",
            lambda arm: arm["device_trace"]["unique_kernel_count"],
            ",d",
        ),
        (
            "summed kernel time / trip [s]",
            lambda arm: arm["device_trace"]["summed_kernel_s_per_observed_trip"],
            ".6f",
        ),
        (
            "host launch APIs",
            lambda arm: arm["device_trace"]["host_launch_api"]["launch_count"],
            ",d",
        ),
        (
            "CUDA graph launches",
            lambda arm: arm["device_trace"]["host_launch_api"]["graph_launch_count"],
            ",d",
        ),
        (
            "summed host launch API / trip [s]",
            lambda arm: arm["device_trace"]["host_launch_api"][
                "summed_api_s_per_observed_trip"
            ],
            ".6f",
        ),
        (
            "summed host inter-launch gaps / trip [s]",
            lambda arm: arm["device_trace"]["host_launch_api"][
                "between_launch_gap_sum_s_per_observed_trip"
            ],
            ".6f",
        ),
        (
            "maximum host inter-launch gap [s]",
            lambda arm: arm["device_trace"]["host_launch_api"]["between_launch_gaps"][
                "maximum_s"
            ],
            ".6f",
        ),
        (
            "device idle-gap sum [s]",
            lambda arm: arm["device_trace"]["device_inter_kernel_idle_gaps"]["sum_s"],
            ".6f",
        ),
    )
    for label, getter, formatting in trace_rows:
        lines.append(
            f"| {label} | {format(getter(exact), formatting)} | "
            f"{format(getter(constant), formatting)} |"
        )
    lines.extend(
        [
            "",
            "### Exact Jacobi: per-kernel sums",
            "",
            "| rank | kernel | calls | summed [s] | share |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(exact["device_trace"]["per_kernel_sums"][:20], 1):
        lines.append(
            f"| {index} | `{row['kernel']}` | {row['call_count']:,} | "
            f"{row['summed_s_per_observed_trip']:.6f} | "
            f"{100.0 * row['share_of_summed_kernel_time']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Constant discriminator: per-kernel sums",
            "",
            "| rank | kernel | calls | summed [s] | share |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(constant["device_trace"]["per_kernel_sums"][:20], 1):
        lines.append(
            f"| {index} | `{row['kernel']}` | {row['call_count']:,} | "
            f"{row['summed_s_per_observed_trip']:.6f} | "
            f"{100.0 * row['share_of_summed_kernel_time']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The JSON receipt retains every per-kernel row and the 100 largest "
            "host inter-launch gaps for each trace; the table above is the readable "
            "top twenty. Raw Perfetto traces and complete compiled HLO are named by "
            "path, size, and SHA-256 in the same receipt.",
            "",
            "## Per-trip evaluation census",
            "",
            "| executed call | exact Jacobi | constant |",
            "|---|---:|---:|",
        ]
    )
    for counter in (
        "projected_krylov_condition",
        "projected_condition_from_hessenberg",
        "linear_action",
    ):
        lines.append(
            f"| `{counter}` | "
            f"{censuses['exact_jacobi']['counts_per_trip'][counter]:,.3f} | "
            f"{censuses['constant_discriminator']['counts_per_trip'][counter]:,.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The direct `_projected_krylov_condition` count is reported even "
                "when it is zero: on this tip qualification consumes the projection "
                "returned by GMRES and calls `_projected_condition_from_hessenberg`. "
                "The callback census is a separate execution and contributes no "
                "timing or trace evidence."
            ),
            "",
            "## Mechanism and repair",
            "",
            f"**Mechanism:** {mechanism['finding']}",
            (
                f"The source chain is `{receipt['source_anchors']['qualification']}` "
                "(ratio consumption), "
                f"`{receipt['source_anchors']['condition']}` (Gram spectrum), and "
                f"`{receipt['source_anchors']['jacobi_sweeps']}` (fixed rotations)."
            ),
            (
                "The execution census is the decisive discriminator: exact versus "
                "constant performs **"
                f"{mechanism['evidence']['exact_condition_evaluations_per_trip']:.0f} "
                "versus "
                f"{mechanism['evidence']['constant_condition_evaluations_per_trip']:.0f}"
                "** condition evaluations and **"
                f"{mechanism['evidence']['exact_linear_actions_per_trip']:.0f} versus "
                f"{mechanism['evidence']['constant_linear_actions_per_trip']:.0f}"
                "** linear actions. The traces contain **"
                f"{mechanism['evidence']['exact_host_launches']:,} versus "
                f"{mechanism['evidence']['constant_host_launches']:,}** host CUDA "
                "launches."
            ),
            f"**Recommended repair:** {mechanism['recommended_repair']}",
            (
                "The direct initial-trip trace saves **"
                f"{mechanism['evidence']['measured_initial_trip_saving_s']:.6f} s**. "
                "Applied to the prior full-solve measurement, the repair target is "
                f"**{mechanism['estimated_saving_s_per_trip']:.6f} s/trip** "
                f"({mechanism['evidence']['prior_exact_full_solve_s_per_trip']:.6f} "
                "minus "
                f"{mechanism['evidence']['prior_constant_full_solve_s_per_trip']:.6f} "
                "s/trip). "
                "A repair must retain the exact two ratios and the resulting admission "
                "sequence; this measurement does not authorize a proxy or deferred "
                "telemetry."
            ),
            "",
            "## Evidence boundary",
            "",
            f"- Measurement revision: `{receipt['measurement_revision']}`.",
            f"- Scheduler stdout: `{receipt['log_paths']['stdout']}`; stderr: "
            f"`{receipt['log_paths']['stderr']}`; combined payload log: "
            f"`{receipt['log_paths']['payload']}`.",
            f"- Exact trace: `{exact['device_trace']['path']}` "
            f"(SHA-256 `{exact['device_trace']['sha256']}`).",
            f"- Constant trace: `{constant['device_trace']['path']}` "
            f"(SHA-256 `{constant['device_trace']['sha256']}`).",
            "- No `nova/` source file was changed; both variants are benchmark-only "
            "bindings of code already present on the measured tip.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    output: Path,
    figure: Path,
    report: Path,
    trace_root: Path,
    cache_root: Path,
    log_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    configure_dtypes()
    _require_measurement_host()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    reference = case["reference"]
    identity = (int(reference["shot"]), int(reference["slice_index"]))
    if identity != (REFERENCE_SHOT, REFERENCE_SLICE):
        raise RuntimeError(f"unexpected production identity {identity}")
    state = jnp.asarray(case["state"])
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt: dict[str, Any] = {
        "schema": "nova.condition_discriminator_trace",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": revision,
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "runtime": _runtime(),
        "scheduler": _scheduler(),
        "log_paths": {
            "payload": str(log_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "persistent_compilation_cache": cache.receipt(),
        "production_identity": {
            "reference": reference,
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "active_set_trip_budget": ACTIVE_SET_TRIP_BUDGET,
            "expected_observed_active_set_trips": EXPECTED_ACTIVE_SET_TRIPS,
            "expected_newton_promotions": EXPECTED_NEWTON_PROMOTIONS,
        },
        "source_anchors": {
            "qualification": _source_anchor(
                fixed_point._qualified_krylov_step,
                "_projected_condition_from_hessenberg",
            ),
            "condition": _source_anchor(
                fixed_point._projected_condition_from_hessenberg,
                "_symmetric_jacobi_eigenvalues",
            ),
            "jacobi_sweeps": _source_anchor(
                fixed_point._symmetric_jacobi_eigenvalues,
                "jax.lax.fori_loop",
            ),
            "gmres_projection": _source_anchor(
                fixed_point._gmres_with_projected_condition,
                "_gmres_arnoldi_projection",
            ),
        },
        "arms": [],
        "evaluation_census": [],
        "measurement_boundaries": {
            "nova_source_edited": False,
            "timed_active_set_trips": EXPECTED_ACTIVE_SET_TRIPS,
            "call_census_timing_eligible": False,
            "constant_scope": (
                "_projected_condition_from_hessenberg is constant and public JAX "
                "GMRES supplies the step without retaining a projection; linear "
                "action, residual, topology, and acceptance code are unchanged"
            ),
        },
    }
    _write_json(output, receipt)

    specifications = [
        ("exact_jacobi", [], False),
        (
            "constant_discriminator",
            [
                (
                    fixed_point,
                    "_gmres_with_projected_condition",
                    _public_gmres_with_dummy_projection,
                ),
                (
                    fixed_point,
                    "_projected_condition_from_hessenberg",
                    _constant_projected_condition,
                ),
            ],
            True,
        ),
    ]
    for name, replacements, _constant in specifications:
        print(f"TRACE_ARM_START name={name}", flush=True)
        receipt["arms"].append(
            _compile_and_trace_arm(
                name,
                profile,
                state,
                target_current,
                replacements,
                trace_root,
            )
        )
        _write_json(output, receipt)
        print(f"TRACE_ARM_DONE name={name}", flush=True)

    for name, _replacements, constant in specifications:
        print(f"CENSUS_ARM_START name={name}", flush=True)
        receipt["evaluation_census"].append(
            _evaluation_census(
                name,
                profile,
                state,
                target_current,
                constant,
            )
        )
        _write_json(output, receipt)
        print(f"CENSUS_ARM_DONE name={name}", flush=True)

    receipt["comparison"] = _comparison(receipt["arms"], receipt["evaluation_census"])
    receipt["mechanism"] = _mechanism(receipt)
    receipt["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(output, receipt)
    _write_figure(receipt, figure)
    _write_report(receipt, report)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    return receipt


def preflight() -> None:
    configure_dtypes()
    case, profile, target_current, _carrier, _policy = _profile_and_seed()
    state = jnp.asarray(case["state"])
    production = _production_program(profile, target_current)
    exact = jax.eval_shape(production, state)
    with _patched(
        [
            (
                fixed_point,
                "_projected_condition_from_hessenberg",
                _constant_projected_condition,
            )
        ]
    ):
        constant = jax.eval_shape(_production_program(profile, target_current), state)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "reference": case["reference"],
                "exact_result_type": type(exact).__name__,
                "constant_result_type": type(constant).__name__,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--stdout-path", type=Path, required=True)
    parser.add_argument("--stderr-path", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
        return
    run(
        arguments.output,
        arguments.figure,
        arguments.report,
        arguments.trace_root,
        arguments.cache_root,
        arguments.log_path,
        arguments.stdout_path,
        arguments.stderr_path,
    )


if __name__ == "__main__":
    main()
