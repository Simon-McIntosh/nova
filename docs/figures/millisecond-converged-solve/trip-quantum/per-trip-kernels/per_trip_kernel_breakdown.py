"""Attribute device and dynamic-loop work to individual active-set trips.

The production solver is not edited.  This benchmark makes an in-memory copy
of its active-set entry point, adds ordered callbacks around each trip, and
uses those callbacks to hold named profiler ranges open while device work runs.
A separate untimed execution records data-dependent loop counts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
import textwrap
import threading
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
from nova.equilibrium import (
    connectivity_boundary,
    domain,
    fixed_point,
    flux_surface_connectivity,
    flux_surface_extraction,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/per-trip-kernels"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "profile.json"
DEFAULT_COMPONENT_TABLE = OUTPUT_ROOT / "per-trip-components.csv"
DEFAULT_DYNAMIC_TABLE = OUTPUT_ROOT / "per-trip-dynamic-counts.csv"
DEFAULT_KERNEL_TABLE = OUTPUT_ROOT / "kernel-growth.csv"
DEFAULT_FIGURE = OUTPUT_ROOT / "per-trip-components.png"
DEFAULT_TRACE_EVENTS = OUTPUT_ROOT / "trace-events.jsonl"
DEFAULT_CENSUS_EVENTS = OUTPUT_ROOT / "census-events.jsonl"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "per-trip-kernel-breakdown.md"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
DEFAULT_OUTER_PROFILE = (
    Path("/home/ITER/mcintos/Code/nova")
    / "docs/figures/millisecond-converged-solve/trip-quantum/outer-loop/profile.json"
)
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIP_BUDGET = 24
EXPECTED_ACTIVE_SET_TRIPS = 7
TRACE_PREFIX = "per_trip_kernel_breakdown"
LINE_SEARCH_LADDER_LENGTH = len(fixed_point._BACKTRACKING_FACTORS)


_EVENTS: list[dict[str, Any]] = []
_EVENT_LOCK = threading.RLock()
_EVENT_LOG: Path | None = None
_EVENT_PHASE = "unconfigured"
_ACTIVE_TRIP: int | None = None
_ACTIVE_RANGES: list[dict[str, Any]] = []
_RANGE_COUNTS: Counter[tuple[int, str]] = Counter()
_PROFILER_RANGES_ENABLED = False
_PROFILER_TRACE_ROOT: Path | None = None
_PROFILER_TRACE_PATHS: dict[int, Path] = {}


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _distribution(values: list[float]) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": int(samples.size),
        "sum": float(samples.sum()),
        "minimum": float(samples.min()),
        "median": float(np.median(samples)),
        "maximum": float(samples.max()),
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
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    }


def _scheduler(stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "temporary_directory": os.environ.get("TMPDIR"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _require_measurement_host() -> None:
    kinds = [str(getattr(device, "device_kind", "")) for device in jax.devices()]
    if jax.default_backend() != "gpu" or not any("H200" in kind for kind in kinds):
        raise RuntimeError(
            "per-trip tracing requires the reserved H200; "
            f"backend={jax.default_backend()} devices={kinds}"
        )
    if not jax.devices("cpu"):
        raise RuntimeError("ordered profiler callbacks require a visible CPU device")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")


def _source_anchor(function: Callable, needle: str | None = None) -> str:
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
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


def _append_event_locked(event: dict[str, Any]) -> None:
    event = {
        "phase": _EVENT_PHASE,
        "sequence": len(_EVENTS),
        **event,
    }
    _EVENTS.append(event)
    if _EVENT_LOG is None:
        return
    with _EVENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_strict(event), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reset_events(path: Path | None, phase: str) -> None:
    global _ACTIVE_TRIP, _EVENT_LOG, _EVENT_PHASE
    with _EVENT_LOCK:
        if _ACTIVE_RANGES:
            raise RuntimeError(
                f"cannot reset with live profiler ranges: {_ACTIVE_RANGES}"
            )
        _EVENTS.clear()
        _RANGE_COUNTS.clear()
        _ACTIVE_TRIP = None
        _EVENT_LOG = path
        _EVENT_PHASE = phase
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")


def _open_trip_range(index: Any, active: Any) -> None:
    global _ACTIVE_TRIP
    trip = int(index) + 1
    is_active = bool(active)
    with _EVENT_LOCK:
        if is_active:
            if _ACTIVE_TRIP is not None:
                raise RuntimeError(
                    f"trip {_ACTIVE_TRIP} remained open before trip {trip}"
                )
            _ACTIVE_TRIP = trip
            name = f"{TRACE_PREFIX}/trip_{trip:02d}"
            annotation = None
            if _PROFILER_RANGES_ENABLED:
                if _PROFILER_TRACE_ROOT is None:
                    raise RuntimeError("per-trip profiler root is not configured")
                trace_path = _PROFILER_TRACE_ROOT / f"trip-{trip:02d}"
                trace_path.mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(str(trace_path), create_perfetto_trace=True)
                _PROFILER_TRACE_PATHS[trip] = trace_path
                annotation = jax.profiler.TraceAnnotation(name)
                annotation.__enter__()
            _ACTIVE_RANGES.append(
                {"kind": "trip", "trip": trip, "name": name, "annotation": annotation}
            )
        _append_event_locked(
            {
                "kind": "trip_start",
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": trip,
                "active": is_active,
            }
        )


def _close_trip_range(index: Any, active_before: Any, active_after: Any) -> None:
    global _ACTIVE_TRIP
    trip = int(index) + 1
    was_active = bool(active_before)
    with _EVENT_LOCK:
        if was_active:
            if not _ACTIVE_RANGES or _ACTIVE_RANGES[-1]["kind"] != "trip":
                raise RuntimeError(f"nested range remained open at trip {trip} end")
            opened = _ACTIVE_RANGES.pop()
            if opened["trip"] != trip:
                raise RuntimeError(
                    f"closing trip {trip} while {opened['trip']} is open"
                )
            if opened["annotation"] is not None:
                opened["annotation"].__exit__(None, None, None)
                jax.profiler.stop_trace()
            _ACTIVE_TRIP = None
        _append_event_locked(
            {
                "kind": "trip_end",
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": trip,
                "active_before": was_active,
                "active_after": bool(active_after),
            }
        )


def _open_component_range(kind: str, _marker: Any) -> None:
    with _EVENT_LOCK:
        if _ACTIVE_TRIP is None:
            raise RuntimeError(f"{kind} started outside an active trip")
        key = (_ACTIVE_TRIP, kind)
        _RANGE_COUNTS[key] += 1
        name = f"{TRACE_PREFIX}/trip_{_ACTIVE_TRIP:02d}/{kind}_{_RANGE_COUNTS[key]:02d}"
        annotation = None
        if _PROFILER_RANGES_ENABLED:
            annotation = jax.profiler.TraceAnnotation(name)
            annotation.__enter__()
        _ACTIVE_RANGES.append(
            {
                "kind": kind,
                "trip": _ACTIVE_TRIP,
                "name": name,
                "annotation": annotation,
            }
        )
        _append_event_locked(
            {
                "kind": "component_start",
                "component": kind,
                "name": name,
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": _ACTIVE_TRIP,
            }
        )


def _close_component_range(kind: str, metric: Any) -> None:
    with _EVENT_LOCK:
        if not _ACTIVE_RANGES or _ACTIVE_RANGES[-1]["kind"] != kind:
            raise RuntimeError(
                f"component nesting mismatch: closing {kind}, stack={_ACTIVE_RANGES}"
            )
        opened = _ACTIVE_RANGES.pop()
        if opened["annotation"] is not None:
            opened["annotation"].__exit__(None, None, None)
        _append_event_locked(
            {
                "kind": "component_end",
                "component": kind,
                "name": opened["name"],
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": opened["trip"],
                "metric": _strict(np.asarray(metric)),
            }
        )


def _record_trip_result(
    index: Any,
    attempted: Any,
    accepted: Any,
    backtracks: Any,
    recoveries: Any,
    recovery_outcomes: Any,
    rebuilds: Any,
    descents: Any,
    decisions: Any,
    qualifications: Any,
    residual: Any,
    mask_difference: Any,
    damping_activated: Any,
    history_count: Any,
    mask_population: Any,
    active_after: Any,
) -> None:
    attempted_count = int(attempted)
    decision_values = np.asarray(decisions, dtype=np.int64)[:attempted_count]
    qualification_values = np.asarray(qualifications, dtype=np.int64)[:attempted_count]
    executed = int(
        np.count_nonzero(
            decision_values != int(fixed_point.InnerIterationDecision.NOT_EXECUTED)
        )
    )
    if executed != attempted_count:
        raise RuntimeError(
            "inner receipt row count differs from attempted promotions: "
            f"rows={executed} attempted={attempted_count}"
        )
    arrays = {
        "backtrack_counts": np.asarray(backtracks, dtype=np.int64)[:attempted_count],
        "recovery_activations": np.asarray(recoveries, dtype=np.int64)[
            :attempted_count
        ],
        "recovery_outcomes": np.asarray(recovery_outcomes, dtype=np.int64)[
            :attempted_count
        ],
        "model_rebuild_activations": np.asarray(rebuilds, dtype=np.int64)[
            :attempted_count
        ],
        "descent_activations": np.asarray(descents, dtype=np.int64)[:attempted_count],
    }
    with _EVENT_LOCK:
        _append_event_locked(
            {
                "kind": "trip_result",
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": int(index) + 1,
                "attempted_promotions": attempted_count,
                "accepted_promotions": int(accepted),
                "executed_receipt_rows": executed,
                **{name: values.tolist() for name, values in arrays.items()},
                "backtrack_sum": int(arrays["backtrack_counts"].sum()),
                "recovery_activation_count": int(
                    np.count_nonzero(arrays["recovery_activations"] == 1)
                ),
                "recovery_acceptance_count": int(
                    np.count_nonzero(
                        arrays["recovery_outcomes"]
                        == int(fixed_point.RecoveryOutcome.ACCEPTED)
                    )
                ),
                "model_rebuild_activation_count": int(
                    np.count_nonzero(arrays["model_rebuild_activations"] == 1)
                ),
                "descent_activation_count": int(
                    np.count_nonzero(arrays["descent_activations"] == 1)
                ),
                "inner_decision_histogram": {
                    str(key): value
                    for key, value in sorted(Counter(decision_values.tolist()).items())
                },
                "krylov_qualification_histogram": {
                    str(key): value
                    for key, value in sorted(
                        Counter(qualification_values.tolist()).items()
                    )
                },
                "line_search_grades": executed * LINE_SEARCH_LADDER_LENGTH,
                "residual": float(residual),
                "mask_difference": int(mask_difference),
                "cycle_damping_activated": bool(damping_activated),
                "mask_history_count": int(history_count),
                "mask_population": int(mask_population),
                "active_after": bool(active_after),
            }
        )


def _record_dynamic(category: str, value: Any) -> None:
    array = np.asarray(value)
    with _EVENT_LOCK:
        _append_event_locked(
            {
                "kind": "dynamic_count",
                "category": category,
                "timestamp_ns": time.perf_counter_ns(),
                "trip_index": _ACTIVE_TRIP,
                "values": _strict(array),
                "value_sum": int(np.asarray(array, dtype=np.int64).sum()),
                "value_count": int(array.size),
            }
        )


def _instrumented_active_set_entry() -> Callable:
    """Return an in-memory active-set entry with trip result callbacks."""
    source = textwrap.dedent(inspect.getsource(fixed_point._active_set_newton_krylov))
    replacements = (
        (
            "def _active_set_newton_krylov(",
            "def _instrumented_active_set_newton_krylov(",
        ),
        (
            "    initial = _solver_state(initial, precision)\n",
            "    initial = _solver_state(initial, precision)\n"
            "    jax.debug.callback(\n"
            "        _open_trip_range, jnp.asarray(0, dtype=jnp.int32),\n"
            "        jnp.asarray(True), ordered=True,\n"
            "    )\n",
        ),
        (
            "    first_presettlement = (\n",
            "    jax.debug.callback(\n"
            "        _record_trip_result,\n"
            "        jnp.asarray(0, dtype=jnp.int32),\n"
            "        first_result.attempted_newton_promotions,\n"
            "        first_result.accepted_newton_promotions,\n"
            "        first_result.promotion_backtrack_counts,\n"
            "        first_result.promotion_recovery_activations,\n"
            "        first_result.promotion_recovery_outcomes,\n"
            "        first_result.promotion_model_rebuild_activations,\n"
            "        first_result.promotion_descent_activations,\n"
            "        first_result.inner_iteration_decisions,\n"
            "        first_result.inner_iteration_krylov_qualifications,\n"
            "        first_residual, first_difference, first_damping,\n"
            "        history_count, jnp.sum(first_mask, dtype=jnp.int32),\n"
            "        first_active, ordered=True,\n"
            "    )\n"
            "    jax.debug.callback(\n"
            "        _close_trip_range, jnp.asarray(0, dtype=jnp.int32),\n"
            "        jnp.asarray(True), first_active, ordered=True,\n"
            "    )\n"
            "    first_presettlement = (\n",
        ),
        (
            "    def outer_body(index, carry):\n        def solve_active(carry):\n",
            "    def outer_body(index, carry):\n"
            "        jax.debug.callback(\n"
            "            _open_trip_range, index, carry.active, ordered=True,\n"
            "        )\n"
            "        def solve_active(carry):\n",
        ),
        (
            "            next_presettlement = (\n",
            "            jax.debug.callback(\n"
            "                _record_trip_result, index,\n"
            "                inner_result.attempted_newton_promotions,\n"
            "                inner_result.accepted_newton_promotions,\n"
            "                inner_result.promotion_backtrack_counts,\n"
            "                inner_result.promotion_recovery_activations,\n"
            "                inner_result.promotion_recovery_outcomes,\n"
            "                inner_result.promotion_model_rebuild_activations,\n"
            "                inner_result.promotion_descent_activations,\n"
            "                inner_result.inner_iteration_decisions,\n"
            "                inner_result.inner_iteration_krylov_qualifications,\n"
            "                live_residual, mask_difference, damping_activated,\n"
            "                history_count, jnp.sum(mask, dtype=jnp.int32), active,\n"
            "                ordered=True,\n"
            "            )\n"
            "            next_presettlement = (\n",
        ),
        (
            "        return jax.lax.cond("
            "carry.active, solve_active, lambda value: value, carry)\n",
            "        result = jax.lax.cond(\n"
            "            carry.active, solve_active, lambda value: value, carry\n"
            "        )\n"
            "        jax.debug.callback(\n"
            "            _close_trip_range, index, carry.active, result.active,\n"
            "            ordered=True,\n"
            "        )\n"
            "        return result\n",
        ),
    )
    for old, new in replacements:
        occurrences = source.count(old)
        if occurrences != 1:
            raise RuntimeError(
                "active-set instrumentation source pattern changed: "
                f"expected one occurrence, found {occurrences}: {old!r}"
            )
        source = source.replace(old, new)
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "_open_trip_range": _open_trip_range,
            "_close_trip_range": _close_trip_range,
            "_record_trip_result": _record_trip_result,
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_active_set_newton_krylov"]


@contextmanager
def _patched(replacements: list[tuple[Any, str, Any]]) -> Iterator[None]:
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in replacements]
    try:
        for owner, name, value in replacements:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


@contextmanager
def _patched_active_set_entry() -> Iterator[None]:
    original = fixed_point._active_set_newton_krylov
    fixed_point._active_set_newton_krylov = _instrumented_active_set_entry()
    try:
        yield
    finally:
        fixed_point._active_set_newton_krylov = original


_ORIGINAL_GMRES = fixed_point._gmres_with_projected_condition
_ORIGINAL_PROMOTION = fixed_point._backtracked_promotion
_ORIGINAL_REBUILD = fixed_point._rebuilt_model_promotion
_ORIGINAL_DESCENT = fixed_point._steepest_descent_promotion


def _annotated_gmres(
    linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    *,
    krylov_dimension: int,
):
    jax.debug.callback(
        partial(_open_component_range, "gmres"), residual_vector[0], ordered=True
    )
    step, info, projection = _ORIGINAL_GMRES(
        linear_action, residual_vector, krylov_dimension=krylov_dimension
    )
    jax.debug.callback(
        partial(_close_component_range, "gmres"),
        projection.active_columns,
        ordered=True,
    )
    return step, info, projection


def _annotated_promotion(*args, **kwargs):
    state = args[2]
    jax.debug.callback(
        partial(_open_component_range, "promotion"), state[0], ordered=True
    )
    result = _ORIGINAL_PROMOTION(*args, **kwargs)
    jax.debug.callback(
        partial(_close_component_range, "promotion"),
        result.backtrack_count,
        ordered=True,
    )
    return result


def _annotated_rebuild(*args, **kwargs):
    state = args[1]
    jax.debug.callback(
        partial(_open_component_range, "model_rebuild"), state[0], ordered=True
    )
    result = _ORIGINAL_REBUILD(*args, **kwargs)
    jax.debug.callback(
        partial(_close_component_range, "model_rebuild"),
        result.accepted,
        ordered=True,
    )
    return result


def _annotated_descent(*args, **kwargs):
    state = args[1]
    jax.debug.callback(
        partial(_open_component_range, "descent"), state[0], ordered=True
    )
    result = _ORIGINAL_DESCENT(*args, **kwargs)
    jax.debug.callback(
        partial(_close_component_range, "descent"), result.accepted, ordered=True
    )
    return result


def _trace_replacements() -> list[tuple[Any, str, Any]]:
    return [
        (fixed_point, "_gmres_with_projected_condition", _annotated_gmres),
        (fixed_point, "_backtracked_promotion", _annotated_promotion),
        (fixed_point, "_rebuilt_model_promotion", _annotated_rebuild),
        (fixed_point, "_steepest_descent_promotion", _annotated_descent),
    ]


def _census_replacements() -> list[tuple[Any, str, Any]]:
    original_polish = flux_surface_connectivity.polish_stationary_points

    def counted_gmres(linear_action, residual_vector, *, krylov_dimension: int):
        step, info, projection = _ORIGINAL_GMRES(
            linear_action, residual_vector, krylov_dimension=krylov_dimension
        )
        jax.debug.callback(
            partial(_record_dynamic, "gmres_iterations_used"),
            projection.active_columns,
            ordered=True,
        )
        return step, info, projection

    def measured_flood(category: str):
        def flood(confined, seed, n_iter):
            core, steps = flux_surface_connectivity.flood_fill_core_with_steps(
                confined, seed, n_iter
            )
            jax.debug.callback(partial(_record_dynamic, category), steps, ordered=True)
            return core

        return flood

    def measured_labels(category: str):
        def labels(confined, rings, links, n_iter):
            labelled, steps = (
                flux_surface_connectivity.label_saddle_aware_hex_connected_components_with_steps(
                    confined, rings, links, n_iter
                )
            )
            jax.debug.callback(partial(_record_dynamic, category), steps, ordered=True)
            return labelled

        return labels

    def measured_wall_dispatch(confined, seed, n_iter, use_doubling):
        if use_doubling:
            core, steps = flux_surface_connectivity.flood_fill_core_with_steps(
                confined, seed, n_iter
            )
            jax.debug.callback(
                partial(_record_dynamic, "wall_reachability_iterations"),
                steps,
                ordered=True,
            )
            return core
        return connectivity_boundary._linear_flood_fill_core(confined, seed, n_iter)

    def measured_polish(spline, seed_rz, valid, stationary_steps=8):
        result = original_polish(spline, seed_rz, valid, stationary_steps)
        jax.debug.callback(
            partial(_record_dynamic, "stationary_polish_iterations"),
            result["iteration_count"],
            ordered=True,
        )
        return result

    general_flood = measured_flood("flood_fill_iterations")
    wall_flood = measured_flood("wall_reachability_iterations")
    general_labels = measured_labels("flood_fill_iterations")
    wall_labels = measured_labels("wall_reachability_iterations")
    return [
        (fixed_point, "_gmres_with_projected_condition", counted_gmres),
        (flux_surface_connectivity, "flood_fill_core", general_flood),
        (connectivity_boundary, "flood_fill_core", wall_flood),
        (connectivity_boundary, "_flood_fill", measured_wall_dispatch),
        (flux_surface_extraction, "flood_fill_core", general_flood),
        (
            domain,
            "label_saddle_aware_hex_connected_components",
            general_labels,
        ),
        (
            connectivity_boundary,
            "label_saddle_aware_hex_connected_components",
            wall_labels,
        ),
        (flux_surface_connectivity, "polish_stationary_points", measured_polish),
        (connectivity_boundary, "polish_stationary_points", measured_polish),
    ]


def _production_program(profile: Any, target_current: float):
    def production(initial_flux):
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

    return production


def _state_hash(result: Any) -> str:
    state = np.asarray(result.equilibrium.fixed_point.state)
    return hashlib.sha256(state.tobytes(order="C")).hexdigest()


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
        "state_sha256": _state_hash(result),
    }


def _latest_trace(path: Path) -> Path:
    traces = sorted(
        path.glob("plugins/profile/*/perfetto_trace.json.gz"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
    )
    if not traces:
        raise RuntimeError(f"JAX profiler wrote no Perfetto trace under {path}")
    return traces[-1]


def _host_gap_rows(events: list[dict[str, Any]]) -> tuple[list[float], list[dict]]:
    ordered = sorted(events, key=lambda event: float(event["ts"]))
    gaps = []
    rows = []
    for previous, following in zip(ordered, ordered[1:], strict=False):
        previous_stop = float(previous["ts"]) + float(previous.get("dur", 0.0))
        gap_s = max(float(following["ts"]) - previous_stop, 0.0) / 1.0e6
        gaps.append(gap_s)
        rows.append(
            {
                "gap_s": gap_s,
                "preceding_api": str(previous.get("name", "unknown")),
                "following_api": str(following.get("name", "unknown")),
            }
        )
    return gaps, sorted(rows, key=lambda row: row["gap_s"], reverse=True)[:12]


def _events_inside(
    events: list[dict[str, Any]], start: float, stop: float
) -> list[dict[str, Any]]:
    return [event for event in events if start <= float(event.get("ts", -1.0)) < stop]


def _kernel_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("name", "unknown"))].append(
            float(event.get("dur", 0.0)) / 1.0e6
        )
    total = sum(sum(values) for values in grouped.values())
    return sorted(
        (
            {
                "kernel": name,
                "call_count": len(values),
                "summed_s": float(sum(values)),
                "median_call_us": 1.0e6 * float(np.median(values)),
                "share_of_summed_kernel_time": float(sum(values)) / max(total, 1.0e-30),
            }
            for name, values in grouped.items()
        ),
        key=lambda row: row["summed_s"],
        reverse=True,
    )


def _trace_summary(path: Path, expected_trips: list[int]) -> dict[str, Any]:
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
    complete = [event for event in events if event.get("ph") == "X"]
    trip_pattern = re.compile(rf"^{re.escape(TRACE_PREFIX)}/trip_(\d+)$")
    component_pattern = re.compile(rf"^{re.escape(TRACE_PREFIX)}/trip_(\d+)/(\w+)_\d+$")
    trip_annotations: dict[int, dict[str, Any]] = {}
    component_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in complete:
        name = str(event.get("name", ""))
        match = trip_pattern.match(name)
        if match is not None:
            trip = int(match.group(1))
            if trip in trip_annotations:
                raise RuntimeError(f"duplicate trip annotation {trip}")
            trip_annotations[trip] = event
            continue
        match = component_pattern.match(name)
        if match is not None:
            component_annotations[int(match.group(1))].append(event)
    if sorted(trip_annotations) != expected_trips:
        raise RuntimeError(
            "profiler trip annotations differ from the expected active trips: "
            f"{sorted(trip_annotations)}"
        )
    gpu_complete = [event for event in complete if event.get("pid") in gpu_pids]
    host_complete = [event for event in complete if event.get("pid") in host_pids]
    rows = []
    for trip, annotation in sorted(trip_annotations.items()):
        start = float(annotation["ts"])
        stop = start + float(annotation.get("dur", 0.0))
        device_events = _events_inside(gpu_complete, start, stop)
        kernels = [
            event
            for event in device_events
            if not str(event.get("name", "")).startswith(("Memcpy", "Memset"))
        ]
        launches = [
            event
            for event in _events_inside(host_complete, start, stop)
            if str(event.get("name", "")).startswith(
                ("cuGraphLaunch", "cuLaunchKernel", "cuLaunchCooperativeKernel")
            )
        ]
        host_gaps, largest_gaps = _host_gap_rows(launches)
        component_rows: dict[str, dict[str, Any]] = {}
        for component in component_annotations.get(trip, []):
            component_name = component_pattern.match(str(component["name"])).group(2)
            component_start = float(component["ts"])
            component_stop = component_start + float(component.get("dur", 0.0))
            selected = _events_inside(kernels, component_start, component_stop)
            row = component_rows.setdefault(
                component_name,
                {
                    "call_count": 0,
                    "annotation_s": 0.0,
                    "summed_kernel_s": 0.0,
                    "kernel_event_count": 0,
                },
            )
            row["call_count"] += 1
            row["annotation_s"] += float(component.get("dur", 0.0)) / 1.0e6
            row["summed_kernel_s"] += sum(
                float(event.get("dur", 0.0)) / 1.0e6 for event in selected
            )
            row["kernel_event_count"] += len(selected)
        kernel_rows = _kernel_rows(kernels)
        summed_kernel_s = sum(row["summed_s"] for row in kernel_rows)
        trip_wall_s = float(annotation.get("dur", 0.0)) / 1.0e6
        additive = sum(
            component_rows.get(name, {}).get("annotation_s", 0.0)
            for name in ("gmres", "promotion")
        )
        component_rows["other_outer_work"] = {
            "call_count": 1,
            "annotation_s": max(trip_wall_s - additive, 0.0),
            "summed_kernel_s": None,
            "kernel_event_count": None,
        }
        rows.append(
            {
                "trip_index": trip,
                "annotation_s": trip_wall_s,
                "device_event_count": len(device_events),
                "kernel_event_count": len(kernels),
                "summed_kernel_s": summed_kernel_s,
                "unique_kernel_count": len(kernel_rows),
                "host_launch_count": len(launches),
                "host_launch_api_s": sum(
                    float(event.get("dur", 0.0)) / 1.0e6 for event in launches
                ),
                "host_between_launch_gap_s": sum(host_gaps),
                "largest_host_launch_gaps": largest_gaps,
                "components": component_rows,
                "ranked_kernels": kernel_rows[:60],
            }
        )
    return {
        "status": "complete",
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "gpu_process_ids": sorted(gpu_pids),
        "host_process_ids": sorted(host_pids),
        "trip_rows": rows,
        "method": (
            "ordered device callbacks open one named profiler range per active-set "
            "trip and nested ranges around GMRES, promotion, rebuild, and descent; "
            "all complete GPU events whose start lies inside a range are retained"
        ),
    }


def _combined_trace_summary(paths: dict[int, Path]) -> dict[str, Any]:
    expected = list(range(1, EXPECTED_ACTIVE_SET_TRIPS + 1))
    if sorted(paths) != expected:
        raise RuntimeError(
            f"per-trip profiler sessions differ from expected: {sorted(paths)}"
        )
    summaries = [
        _trace_summary(_latest_trace(paths[trip]), [trip]) for trip in expected
    ]
    return {
        "status": "complete",
        "paths": [summary["path"] for summary in summaries],
        "sha256": [summary["sha256"] for summary in summaries],
        "size_bytes": sum(summary["size_bytes"] for summary in summaries),
        "gpu_process_ids": sorted(
            {pid for summary in summaries for pid in summary["gpu_process_ids"]}
        ),
        "host_process_ids": sorted(
            {pid for summary in summaries for pid in summary["host_process_ids"]}
        ),
        "trip_rows": [row for summary in summaries for row in summary["trip_rows"]],
        "method": (
            "one CUPTI session per active-set trip, opened and closed by ordered "
            "device callbacks around the unchanged fused production solve; nested "
            "ranges identify GMRES, promotion, rebuild, and descent, and all "
            "complete GPU events whose start lies inside a range are retained"
        ),
    }


def _trip_results(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows = {
        int(event["trip_index"]): event
        for event in events
        if event["kind"] == "trip_result"
    }
    if sorted(rows) != list(range(1, EXPECTED_ACTIVE_SET_TRIPS + 1)):
        raise RuntimeError(f"trip result rows differ from expected: {sorted(rows)}")
    return rows


def _dynamic_rows(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event["kind"] != "dynamic_count":
            continue
        trip = event["trip_index"]
        if trip is None:
            raise RuntimeError(f"dynamic count occurred outside a trip: {event}")
        grouped[int(trip)][event["category"]].append(int(event["value_sum"]))
    rows = {}
    categories = (
        "gmres_iterations_used",
        "flood_fill_iterations",
        "wall_reachability_iterations",
        "stationary_polish_iterations",
    )
    for trip in range(1, EXPECTED_ACTIVE_SET_TRIPS + 1):
        category_rows = {}
        for category in categories:
            values = grouped[trip].get(category, [])
            category_rows[category] = {
                "call_count": len(values),
                "sum": int(sum(values)),
                "minimum": min(values) if values else None,
                "median": float(np.median(values)) if values else None,
                "maximum": max(values) if values else None,
                "values": values,
            }
        rows[trip] = category_rows
    return rows


def _kernel_growth(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trip = {
        row["trip_index"]: {item["kernel"]: item for item in row["ranked_kernels"]}
        for row in trace_rows
    }
    names = sorted({name for rows in by_trip.values() for name in rows})
    slowest = max(trace_rows, key=lambda row: row["annotation_s"])
    slowest_index = slowest["trip_index"]
    rows = []
    for name in names:
        values = [by_trip[trip].get(name, {}).get("summed_s", 0.0) for trip in by_trip]
        call_counts = [
            by_trip[trip].get(name, {}).get("call_count", 0) for trip in by_trip
        ]
        correlation = float(np.corrcoef(np.arange(1, len(values) + 1), values)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
        rows.append(
            {
                "kernel": name,
                "summed_s_by_trip": values,
                "call_count_by_trip": call_counts,
                "first_to_slowest_growth_s": values[slowest_index - 1] - values[0],
                "slowest_trip_s": values[slowest_index - 1],
                "share_of_slowest_trip": values[slowest_index - 1]
                / max(slowest["annotation_s"], 1.0e-30),
                "trip_index_correlation": correlation,
            }
        )
    return sorted(
        rows,
        key=lambda row: row["first_to_slowest_growth_s"],
        reverse=True,
    )


def _component_growth(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slowest = max(trace_rows, key=lambda row: row["annotation_s"])
    first = trace_rows[0]
    rows = []
    for name in ("promotion", "gmres", "other_outer_work"):
        values = [row["components"][name]["annotation_s"] for row in trace_rows]
        rows.append(
            {
                "component": name,
                "annotation_s_by_trip": values,
                "first_to_slowest_growth_s": values[slowest["trip_index"] - 1]
                - values[0],
                "slowest_trip_s": values[slowest["trip_index"] - 1],
                "share_of_slowest_trip": values[slowest["trip_index"] - 1]
                / max(slowest["annotation_s"], 1.0e-30),
                "first_trip_s": first["components"][name]["annotation_s"],
            }
        )
    return sorted(rows, key=lambda row: row["first_to_slowest_growth_s"], reverse=True)


def _banked_rows(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    rows = {1: {"outer_wall_s": receipt["banked_one_trip"]["synchronized_wall_s"]}}
    for row in receipt["outer_iterations"]:
        if row["active_at_entry"]:
            rows[int(row["outer_index"]) + 1] = row
    return receipt, rows


def _banked_job_id(receipt: dict[str, Any]) -> str | None:
    paths = receipt.get("log_paths", {})
    match = re.search(r"slurm-(\d+)", str(paths.get("stdout", "")))
    return match.group(1) if match is not None else None


def _source_anchors() -> dict[str, str]:
    return {
        "promotion_recovery": _source_anchor(
            fixed_point._backtracked_promotion, "def recover_with_continuation"
        ),
        "model_rebuild": _source_anchor(
            fixed_point._rebuilt_model_promotion, "jax.scipy.sparse.linalg.gmres"
        ),
        "gmres_restart": _source_anchor(
            fixed_point._gmres_with_projected_condition, "def restart"
        ),
        "flood_fill": _source_anchor(
            flux_surface_connectivity.flood_fill_core_with_steps,
            "jax.lax.while_loop",
        ),
        "wall_reachability": _source_anchor(
            connectivity_boundary._flood_fill, "if use_doubling"
        ),
        "stationary_polish": _source_anchor(
            flux_surface_connectivity.polish_stationary_points,
            "_polish_stationary_points_in_bounds",
        ),
    }


def _mechanism(
    trace_rows: list[dict[str, Any]],
    trip_results: dict[int, dict[str, Any]],
    components: list[dict[str, Any]],
    anchors: dict[str, str],
) -> dict[str, Any]:
    slowest = max(trace_rows, key=lambda row: row["annotation_s"])
    slowest_index = int(slowest["trip_index"])
    dominant = components[0]
    result = trip_results[slowest_index]
    if dominant["component"] == "promotion":
        finding = (
            "Promotion evaluation is the growing interval. Late rejected "
            "promotions exhaust the fixed ladder, activate continuation recovery, "
            "and can re-linearize through the rebuilt-model path."
        )
        source = anchors["promotion_recovery"]
        secondary_source = anchors["model_rebuild"]
        repair = (
            "Do not run the complete recovery and rebuilt-model sequence for every "
            "settlement-proving rejection. Reuse the current linear model and "
            "carried globalization state for a cheap refusal test, entering the "
            "rebuild only when that test predicts an admissible decrease."
        )
    elif dominant["component"] == "gmres":
        finding = (
            "GMRES is the growing interval; the first-restart projection uses more "
            "active columns or more restart bodies on the late residuals."
        )
        source = anchors["gmres_restart"]
        secondary_source = anchors["gmres_restart"]
        repair = (
            "Stop the restart schedule on the achieved linear residual and carry "
            "the accepted projection across retries instead of rebuilding it."
        )
    else:
        finding = (
            "Work outside GMRES and promotion ranges grows most; the dynamic "
            "connectivity and polish census determines which topology loop expands."
        )
        source = anchors["flood_fill"]
        secondary_source = anchors["stationary_polish"]
        repair = (
            "Carry the settled topology partition and its polished null slots until "
            "the mask changes, avoiding repeated reachability and polish loops."
        )
    return {
        "finding": finding,
        "component": dominant["component"],
        "source": source,
        "secondary_source": secondary_source,
        "slowest_trip_index": slowest_index,
        "slowest_trip_wall_s": slowest["annotation_s"],
        "component_s_in_slowest_trip": dominant["slowest_trip_s"],
        "share_of_slowest_trip": dominant["share_of_slowest_trip"],
        "estimated_saving_s_per_slowest_trip": max(
            dominant["first_to_slowest_growth_s"], 0.0
        ),
        "estimated_repaired_slowest_trip_s": max(
            slowest["annotation_s"] - max(dominant["first_to_slowest_growth_s"], 0.0),
            0.0,
        ),
        "slowest_trip_control_counts": {
            "attempted_promotions": result["attempted_promotions"],
            "accepted_promotions": result["accepted_promotions"],
            "backtrack_sum": result["backtrack_sum"],
            "recovery_activations": result["recovery_activation_count"],
            "model_rebuild_activations": result["model_rebuild_activation_count"],
            "descent_activations": result["descent_activation_count"],
        },
        "repair": repair,
        "estimate_contract": (
            "upper-bound saving if the slowest trip's dominant interval returns to "
            "its first-trip duration; no repair is implemented by this benchmark"
        ),
    }


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tables(
    receipt: dict[str, Any], component_path: Path, dynamic_path: Path, kernel_path: Path
) -> None:
    component_rows = []
    dynamic_rows = []
    for row in receipt["per_trip"]:
        components = row["trace"]["components"]
        component_rows.append(
            {
                "trip_index": row["trip_index"],
                "banked_wall_s": row["banked_wall_s"],
                "trace_wall_s": row["trace"]["annotation_s"],
                "gmres_s": components["gmres"]["annotation_s"],
                "promotion_s": components["promotion"]["annotation_s"],
                "model_rebuild_s": components.get("model_rebuild", {}).get(
                    "annotation_s", 0.0
                ),
                "descent_s": components.get("descent", {}).get("annotation_s", 0.0),
                "other_outer_work_s": components["other_outer_work"]["annotation_s"],
                "summed_kernel_s": row["trace"]["summed_kernel_s"],
                "kernel_events": row["trace"]["kernel_event_count"],
                "host_launches": row["trace"]["host_launch_count"],
            }
        )
        result = row["result"]
        dynamic = row["dynamic"]
        dynamic_rows.append(
            {
                "trip_index": row["trip_index"],
                "attempted_promotions": result["attempted_promotions"],
                "accepted_promotions": result["accepted_promotions"],
                "backtrack_sum": result["backtrack_sum"],
                "recovery_activations": result["recovery_activation_count"],
                "model_rebuild_activations": result["model_rebuild_activation_count"],
                "descent_activations": result["descent_activation_count"],
                "gmres_calls": dynamic["gmres_iterations_used"]["call_count"],
                "gmres_iterations_used": dynamic["gmres_iterations_used"]["sum"],
                "flood_fill_iterations": dynamic["flood_fill_iterations"]["sum"],
                "wall_reachability_iterations": dynamic["wall_reachability_iterations"][
                    "sum"
                ],
                "stationary_polish_iterations": dynamic["stationary_polish_iterations"][
                    "sum"
                ],
                "mask_history_count": result["mask_history_count"],
                "mask_population": result["mask_population"],
            }
        )
    _write_csv(
        component_path,
        tuple(component_rows[0]),
        component_rows,
    )
    _write_csv(dynamic_path, tuple(dynamic_rows[0]), dynamic_rows)
    kernel_rows = receipt["kernel_growth"][:80]
    _write_csv(
        kernel_path,
        (
            "kernel",
            "first_to_slowest_growth_s",
            "slowest_trip_s",
            "share_of_slowest_trip",
            "trip_index_correlation",
        ),
        kernel_rows,
    )


def _write_figure(receipt: dict[str, Any], path: Path) -> None:
    per_trip = receipt["per_trip"]
    indices = [row["trip_index"] for row in per_trip]
    gmres = [row["trace"]["components"]["gmres"]["annotation_s"] for row in per_trip]
    promotion = [
        row["trace"]["components"]["promotion"]["annotation_s"] for row in per_trip
    ]
    other = [
        row["trace"]["components"]["other_outer_work"]["annotation_s"]
        for row in per_trip
    ]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0].bar(indices, gmres, label="GMRES")
    axes[0].bar(indices, promotion, bottom=gmres, label="promotion + recovery")
    base = np.asarray(gmres) + np.asarray(promotion)
    axes[0].bar(indices, other, bottom=base, label="reconcile and other")
    axes[0].plot(
        indices,
        [row["banked_wall_s"] for row in per_trip],
        color="black",
        marker="o",
        linestyle="--",
        label="banked uninstrumented wall",
    )
    axes[0].set_ylabel("Seconds per active-set trip")
    axes[0].set_title("Per-trip profiler ranges inside one fused production solve")
    axes[0].legend(ncol=2)

    backtracks = [row["result"]["backtrack_sum"] for row in per_trip]
    recoveries = [row["result"]["recovery_activation_count"] for row in per_trip]
    rebuilds = [row["result"]["model_rebuild_activation_count"] for row in per_trip]
    gmres_iterations = [
        row["dynamic"]["gmres_iterations_used"]["sum"] for row in per_trip
    ]
    axes[1].plot(indices, backtracks, marker="o", label="backtrack count sum")
    axes[1].plot(indices, recoveries, marker="s", label="recovery activations")
    axes[1].plot(indices, rebuilds, marker="^", label="model rebuilds")
    axes[1].plot(
        indices,
        gmres_iterations,
        marker="d",
        label="GMRES iterations actually used",
    )
    axes[1].set_xlabel("Active-set trip index")
    axes[1].set_ylabel("Dynamic count")
    axes[1].set_title("Static budgets conceal data-dependent recovery work")
    axes[1].legend(ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    mechanism = receipt["dominant_mechanism"]
    scheduler = receipt["scheduler"]
    lines = [
        "# Per-trip kernel and dynamic-loop breakdown",
        "",
        (
            f"SLURM job `{scheduler['job_id']}` on `{scheduler['node']}` traced "
            f"the MAST {REFERENCE_SHOT}/{REFERENCE_SLICE} pure production solve "
            "on one H200. Seven active trips were recorded individually; every "
            "trip row was appended to the event logs as its callback arrived."
        ),
        (
            "The callback-instrumented result is bit-identical to the uninstrumented "
            "production result: **"
            f"{str(receipt['semantic_identity']['passes']).lower()}** "
            "(state SHA-256 `"
            f"{receipt['semantic_identity']['reference_state_sha256']}`)."
        ),
        "",
        "## Per-trip timing and control work",
        "",
        "| trip | banked wall [s] | trace wall [s] | GMRES [s] | "
        "promotion and recovery [s] | other [s] | accepted / attempted | "
        "backtracks | recoveries | rebuilds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in receipt["per_trip"]:
        components = row["trace"]["components"]
        result = row["result"]
        lines.append(
            f"| {row['trip_index']} | {row['banked_wall_s']:.6f} | "
            f"{row['trace']['annotation_s']:.6f} | "
            f"{components['gmres']['annotation_s']:.6f} | "
            f"{components['promotion']['annotation_s']:.6f} | "
            f"{components['other_outer_work']['annotation_s']:.6f} | "
            f"{result['accepted_promotions']} / {result['attempted_promotions']} | "
            f"{result['backtrack_sum']} | {result['recovery_activation_count']} | "
            f"{result['model_rebuild_activation_count']} |"
        )
    lines.extend(
        [
            "",
            (
                f"**Growing component:** {mechanism['component'].replace('_', ' ')} "
                f"at `{mechanism['source']}` (secondary path "
                f"`{mechanism['secondary_source']}`). {mechanism['finding']} In "
                f"trip {mechanism['slowest_trip_index']} it occupies "
                f"**{mechanism['component_s_in_slowest_trip']:.6f} s**, or "
                f"**{100.0 * mechanism['share_of_slowest_trip']:.2f}%** of the "
                f"{mechanism['slowest_trip_wall_s']:.6f} s traced trip."
            ),
            (
                f"**Recommended repair:** {mechanism['repair']} Returning only this "
                "component to its first-trip duration is estimated to save "
                f"**{mechanism['estimated_saving_s_per_slowest_trip']:.6f} s** "
                f"from the slowest trip, leaving about "
                f"**{mechanism['estimated_repaired_slowest_trip_s']:.6f} s**. "
                "This is an attribution bound, not a measured repaired solve."
            ),
            "",
            "## Dynamic loop census",
            "",
            "| trip | GMRES calls | GMRES iterations used | flood-fill iterations | "
            "wall-reachability iterations | polish iterations | mask history | "
            "mask population |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in receipt["per_trip"]:
        dynamic = row["dynamic"]
        result = row["result"]
        lines.append(
            f"| {row['trip_index']} | "
            f"{dynamic['gmres_iterations_used']['call_count']} | "
            f"{dynamic['gmres_iterations_used']['sum']} | "
            f"{dynamic['flood_fill_iterations']['sum']} | "
            f"{dynamic['wall_reachability_iterations']['sum']} | "
            f"{dynamic['stationary_polish_iterations']['sum']} | "
            f"{result['mask_history_count']} | {result['mask_population']} |"
        )
    lines.extend(
        [
            "",
            "The dynamic census is a separate untimed execution. Its callbacks "
            "return the unchanged production arrays, so counts are execution "
            "evidence and are excluded from the device-time table.",
            "",
            "## Kernels with the largest first-to-slowest growth",
            "",
            "| rank | kernel | growth [s] | slowest-trip [s] | "
            "slowest-trip share | trip-index correlation |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(receipt["kernel_growth"][:15], 1):
        lines.append(
            f"| {index} | `{row['kernel']}` | "
            f"{row['first_to_slowest_growth_s']:.6f} | "
            f"{row['slowest_trip_s']:.6f} | "
            f"{100.0 * row['share_of_slowest_trip']:.2f}% | "
            f"{row['trip_index_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            f"- Raw Perfetto trace: `{receipt['device_trace']['path']}` "
            f"(SHA-256 `{receipt['device_trace']['sha256']}`).",
            f"- SLURM stdout: `{scheduler['stdout_path']}`; stderr: "
            f"`{scheduler['stderr_path']}`.",
            f"- Measurement revision: `{receipt['measurement_revision']}`.",
            f"- Banked outer-loop input: `{receipt['banked_outer_loop']['path']}` "
            f"(SHA-256 `{receipt['banked_outer_loop']['sha256']}`).",
            "- Profiler annotation walls include the deliberately small ordered "
            "range callbacks; banked walls remain the uninstrumented latency "
            "authority.",
            "- Nested rebuild and descent ranges overlap the promotion range and "
            "must not be added to it. Summed GPU kernel durations can exceed wall "
            "when streams overlap.",
            "- No `nova/` source file was changed; the benchmark restores every "
            "in-memory replacement after each execution.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_trace(
    profile: Any,
    state: jax.Array,
    target_current: float,
    trace_root: Path,
    event_log: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], float]:
    global _PROFILER_RANGES_ENABLED, _PROFILER_TRACE_ROOT
    program = _production_program(profile, target_current)
    started = time.perf_counter()
    reference_executable = jax.jit(program).lower(state).compile()
    reference_compile_s = time.perf_counter() - started
    reference = _ready(reference_executable(state))
    reference_summary = _solve_summary(reference)

    with _patched(_trace_replacements()), _patched_active_set_entry():
        jax.clear_caches()
        started = time.perf_counter()
        executable = jax.jit(program).lower(state).compile()
        instrumented_compile_s = time.perf_counter() - started
        _PROFILER_RANGES_ENABLED = False
        _reset_events(None, "warmup")
        warm = _ready(executable(state))
        warm_summary = _solve_summary(warm)
        _reset_events(event_log, "device_trace")
        job = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
        _PROFILER_TRACE_ROOT = trace_root / f"job-{job}"
        _PROFILER_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
        _PROFILER_TRACE_PATHS.clear()
        _PROFILER_RANGES_ENABLED = True
        try:
            started = time.perf_counter()
            traced = _ready(executable(state))
            traced_wall_s = time.perf_counter() - started
        finally:
            _PROFILER_RANGES_ENABLED = False
            _PROFILER_TRACE_ROOT = None
        traced_summary = _solve_summary(traced)
        events = list(_EVENTS)
    if _ACTIVE_RANGES or _ACTIVE_TRIP is not None:
        raise RuntimeError(
            f"profiler range leak: trip={_ACTIVE_TRIP}, ranges={_ACTIVE_RANGES}"
        )
    trace = _combined_trace_summary(_PROFILER_TRACE_PATHS)
    identity = {
        "passes": (
            reference_summary == warm_summary == traced_summary
            and reference_summary["active_set_trips"] == EXPECTED_ACTIVE_SET_TRIPS
        ),
        "reference": reference_summary,
        "warm_instrumented": warm_summary,
        "traced_instrumented": traced_summary,
        "traced_instrumented_wall_s": traced_wall_s,
        "reference_state_sha256": reference_summary["state_sha256"],
    }
    if not identity["passes"]:
        raise RuntimeError(
            f"instrumented solve changed production identity: {identity}"
        )
    return (
        trace,
        events,
        identity,
        reference_compile_s + instrumented_compile_s,
    )


def _run_census(
    profile: Any,
    state: jax.Array,
    target_current: float,
    event_log: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    global _PROFILER_RANGES_ENABLED
    program = _production_program(profile, target_current)
    _PROFILER_RANGES_ENABLED = False
    with _patched(_census_replacements()), _patched_active_set_entry():
        jax.clear_caches()
        started = time.perf_counter()
        executable = jax.jit(program).lower(state).compile()
        compile_s = time.perf_counter() - started
        _reset_events(event_log, "dynamic_census")
        result = _ready(executable(state))
        events = list(_EVENTS)
    summary = _solve_summary(result)
    if summary["active_set_trips"] != EXPECTED_ACTIVE_SET_TRIPS:
        raise RuntimeError(f"dynamic census changed trip count: {summary}")
    return events, summary, compile_s


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_dtypes()
    _require_measurement_host()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        args.cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    reference = case["reference"]
    identity = (int(reference["shot"]), int(reference["slice_index"]))
    if identity != (REFERENCE_SHOT, REFERENCE_SLICE):
        raise RuntimeError(f"unexpected production identity {identity}")
    state = jnp.asarray(case["state"])

    print("DEVICE_TRACE_START", flush=True)
    trace, trace_events, semantic_identity, trace_compile_s = _run_trace(
        profile, state, target_current, args.trace_root, args.trace_events
    )
    print("DEVICE_TRACE_DONE", flush=True)
    print("DYNAMIC_CENSUS_START", flush=True)
    census_events, census_summary, census_compile_s = _run_census(
        profile, state, target_current, args.census_events
    )
    print("DYNAMIC_CENSUS_DONE", flush=True)

    result_rows = _trip_results(trace_events)
    census_rows = _dynamic_rows(census_events)
    banked, banked_by_trip = _banked_rows(args.outer_profile)
    trace_by_trip = {row["trip_index"]: row for row in trace["trip_rows"]}
    per_trip = []
    for trip in range(1, EXPECTED_ACTIVE_SET_TRIPS + 1):
        if trip not in banked_by_trip:
            raise RuntimeError(f"banked outer-loop receipt lacks trip {trip}")
        per_trip.append(
            {
                "trip_index": trip,
                "banked_wall_s": float(banked_by_trip[trip]["outer_wall_s"]),
                "trace": trace_by_trip[trip],
                "result": result_rows[trip],
                "dynamic": census_rows[trip],
            }
        )
    kernel_growth = _kernel_growth(trace["trip_rows"])
    component_growth = _component_growth(trace["trip_rows"])
    anchors = _source_anchors()
    mechanism = _mechanism(trace["trip_rows"], result_rows, component_growth, anchors)
    receipt = {
        "schema": "nova.per_trip_kernel_breakdown",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "production_identity": {
            "reference": reference,
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "active_set_trip_budget": ACTIVE_SET_TRIP_BUDGET,
            "observed_active_set_trips": EXPECTED_ACTIVE_SET_TRIPS,
        },
        "runtime": _runtime(),
        "scheduler": _scheduler(args.stdout_path, args.stderr_path),
        "persistent_compilation_cache": cache.receipt(),
        "compile_wall_s": {
            "reference_plus_instrumented_trace": trace_compile_s,
            "dynamic_census": census_compile_s,
        },
        "semantic_identity": semantic_identity,
        "census_solve": census_summary,
        "banked_outer_loop": {
            "path": str(args.outer_profile),
            "sha256": _sha256(args.outer_profile),
            "job_id": _banked_job_id(banked),
            "measurement_revision": banked["measurement_revision"],
        },
        "device_trace": trace,
        "per_trip": per_trip,
        "component_growth": component_growth,
        "kernel_growth": kernel_growth,
        "dominant_mechanism": mechanism,
        "source_anchors": anchors,
        "artifacts": {
            "receipt": str(args.output),
            "component_table": str(args.component_table),
            "dynamic_table": str(args.dynamic_table),
            "kernel_table": str(args.kernel_table),
            "figure": str(args.figure),
            "trace_event_log": str(args.trace_events),
            "census_event_log": str(args.census_events),
            "report": str(args.report),
        },
        "measurement_boundaries": {
            "nova_source_edited": False,
            "trace_callbacks": (
                "one open, one result, and one close callback per trip plus nested "
                "component range boundaries; banked uninstrumented walls remain the "
                "latency authority"
            ),
            "dynamic_census_timing_eligible": False,
            "repair_implemented": False,
        },
    }
    _write_json(args.output, receipt)
    _write_tables(receipt, args.component_table, args.dynamic_table, args.kernel_table)
    _write_figure(receipt, args.figure)
    _write_report(receipt, args.report)
    for label, path in (
        ("RECEIPT", args.output),
        ("COMPONENT_TABLE", args.component_table),
        ("DYNAMIC_TABLE", args.dynamic_table),
        ("KERNEL_TABLE", args.kernel_table),
        ("FIGURE", args.figure),
        ("REPORT", args.report),
    ):
        print(f"{label}_WRITTEN={path}", flush=True)
    return receipt


def preflight(outer_profile: Path) -> None:
    configure_dtypes()
    _instrumented_active_set_entry()
    banked, rows = _banked_rows(outer_profile)
    if sorted(rows) != list(range(1, EXPECTED_ACTIVE_SET_TRIPS + 1)):
        raise RuntimeError(f"banked receipt lacks expected trips: {sorted(rows)}")
    print(
        "PER_TRIP_PREFLIGHT_PASS "
        f"revision={_revision()} banked_revision={banked['measurement_revision']} "
        f"trips={len(rows)}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--component-table", type=Path, default=DEFAULT_COMPONENT_TABLE)
    parser.add_argument("--dynamic-table", type=Path, default=DEFAULT_DYNAMIC_TABLE)
    parser.add_argument("--kernel-table", type=Path, default=DEFAULT_KERNEL_TABLE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--trace-events", type=Path, default=DEFAULT_TRACE_EVENTS)
    parser.add_argument("--census-events", type=Path, default=DEFAULT_CENSUS_EVENTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outer-profile", type=Path, default=DEFAULT_OUTER_PROFILE)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--stdout-path", type=Path)
    parser.add_argument("--stderr-path", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "preflight":
        preflight(args.outer_profile)
        return
    missing = [
        name
        for name in ("trace_root", "stdout_path", "stderr_path")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"run mode requires: {', '.join(missing)}")
    run(args)


if __name__ == "__main__":
    main()
