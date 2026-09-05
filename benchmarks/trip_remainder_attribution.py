"""Measure the fused-device remainder in one production active-set trip.

The benchmark times the complete width-one production solve and isolated
benchmark-only ablations on one H200.  It also records dynamic connectivity
iteration counts, checks that three warm calls do not compile, and captures a
JAX device trace after removing both Jacobi-SVD condition diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import gzip
import hashlib
import inspect
import json
import logging
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
from nova.equilibrium import (
    connectivity_boundary,
    domain,
    fixed_point,
    flux_surface_connectivity,
    flux_surface_extraction,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from nova.linalg import split_spline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/millisecond-converged-solve/trip-quantum/remainder/remainder.json"
)
DEFAULT_FIGURE = DEFAULT_OUTPUT.with_name("remainder.png")
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "trip-remainder-attribution.md"
)
DEFAULT_TRACE_ROOT = DEFAULT_REPORT.parent / "trip-remainder-trace"
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
SOURCE_RECEIPT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/one-read/profile.json"
)
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIP_BUDGET = 24
EXPECTED_ACTIVE_SET_TRIPS = 7
REMAINDER_WALL_S_PER_TRIP = 5.632057402


class _CompileLogCounter(logging.Handler):
    """Count JAX compilation messages inside an explicitly bounded interval."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling" in message or "Finished XLA compilation" in message:
            self.messages.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    centre = float(np.median(values))
    return {
        "samples_s": samples,
        "sample_count": len(samples),
        "minimum_s": float(values.min()),
        "median_s": centre,
        "maximum_s": float(values.max()),
        "median_absolute_deviation_s": float(np.median(np.abs(values - centre))),
    }


def _cache_census(path: Path) -> dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def _runtime() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "device": str(device),
        "device_kind": device.device_kind,
        "platform": device.platform,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted_time_limit = None
    if job_id:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            fields = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in completed.stdout.split()
                if "=" in token
            }
            accepted_time_limit = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "accepted_time_limit": accepted_time_limit,
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _require_measurement_host() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError("measurement requires one JAX-visible H200")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")


def _source_anchor(function: Callable, needle: str) -> str:
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    path = Path(inspect.getsourcefile(function) or "unknown")
    lines, first_line = inspect.getsourcelines(function)
    offset = next(
        (index for index, line in enumerate(lines) if needle in line),
        0,
    )
    try:
        displayed = path.relative_to(ROOT)
    except ValueError:
        displayed = path
    return f"{displayed}:{first_line + offset}"


def _constant_anchor(path: Path, needle: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    line_number = next(
        index for index, line in enumerate(lines, 1) if line.startswith(needle)
    )
    return f"{path.relative_to(ROOT)}:{line_number}"


@contextmanager
def _patched(
    replacements: list[tuple[Any, str, Any]],
) -> Iterator[None]:
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in replacements]
    try:
        for owner, name, value in replacements:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def _conditioned_fit_without_condition(
    design: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    regularization: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Preserve the fit while replacing its diagnostic condition SVD."""
    squared_norm = jnp.sum(weights[:, None] * design**2, axis=0)
    scale = jnp.sqrt(squared_norm)
    scale = jnp.where(scale > jnp.finfo(design.dtype).tiny, scale, 1.0)
    scaled_design = design / scale
    normal = scaled_design.T @ (weights[:, None] * scaled_design)
    normal = normal + regularization * jnp.eye(normal.shape[0], dtype=normal.dtype)
    scaled_coefficient = split_spline._differentiable_normal_solve(
        scaled_design, values, weights, regularization
    )
    normal_right_hand_side = scaled_design.T @ (weights * values)
    normal_residual = normal @ scaled_coefficient - normal_right_hand_side
    residual_scale = jnp.maximum(
        jnp.linalg.norm(normal_right_hand_side), jnp.finfo(design.dtype).tiny
    )
    return (
        scaled_coefficient / scale,
        jnp.asarray(1.0, dtype=design.dtype),
        jnp.linalg.norm(normal_residual) / residual_scale,
        scale,
    )


def _constant_projected_condition(
    _linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    *,
    krylov_dimension: int,
) -> tuple[jax.Array, jax.Array]:
    """Replace the diagnostic Arnoldi projection and singular values by one."""
    del krylov_dimension
    one = jnp.asarray(1.0, dtype=residual_vector.dtype)
    return one, one


def _connectivity_replacements(*, zero_iterations: bool):
    original_flood = flux_surface_connectivity.flood_fill_core
    original_labels = (
        flux_surface_connectivity.label_saddle_aware_hex_connected_components
    )

    if zero_iterations:

        def flood(confined, seed, _n_iter):
            return original_flood(confined, seed, 0)

        def labels(confined, rings, links, _n_iter):
            return original_labels(confined, rings, links, 0)

    else:
        flood = original_flood
        labels = original_labels

    return [
        (flux_surface_connectivity, "flood_fill_core", flood),
        (connectivity_boundary, "flood_fill_core", flood),
        (flux_surface_extraction, "flood_fill_core", flood),
        (
            domain,
            "label_saddle_aware_hex_connected_components",
            labels,
        ),
        (
            connectivity_boundary,
            "label_saddle_aware_hex_connected_components",
            labels,
        ),
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


def _solve_summary(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    reason_value = int(np.asarray(history.termination_reason))
    reason = fixed_point.FixedPointTerminationReason(reason_value).name.lower()
    return {
        "active_set_trips": int(np.asarray(history.active_set_iterations)),
        "attempted_newton_promotions": int(
            np.asarray(history.attempted_newton_promotions)
        ),
        "accepted_newton_promotions": int(
            np.asarray(history.accepted_newton_promotions)
        ),
        "termination_reason": reason,
        "terminal_residual": float(np.asarray(history.residual)),
        "converged": bool(np.asarray(history.converged)),
    }


def _time_arm(
    name: str,
    profile: Any,
    state: jax.Array,
    target_current: float,
    replacements: list[tuple[Any, str, Any]],
    cache_root: Path,
    repeats: int,
) -> dict[str, Any]:
    with _patched(replacements):
        function = _production_program(profile, target_current)
        compiled_function = jax.jit(function)
        cache_before_compile = _cache_census(cache_root)
        started = time.perf_counter()
        executable = compiled_function.lower(state).compile()
        compile_wall_s = time.perf_counter() - started
        cache_after_compile = _cache_census(cache_root)
        started = time.perf_counter()
        first_result = _ready(executable(state))
        first_execute_wall_s = time.perf_counter() - started
        summary = _solve_summary(first_result)

        # The ordinary jitted callable, rather than the executable handle, is used
        # here so compilation logging can detect an unexpected retrace.
        _ready(compiled_function(state))
        cache_before_warm = _cache_census(cache_root)
        counter = _CompileLogCounter()
        logger = logging.getLogger()
        logger.addHandler(counter)
        samples = []
        try:
            for index in range(repeats):
                started = time.perf_counter()
                _ready(compiled_function(state))
                samples.append(time.perf_counter() - started)
                print(
                    f"ARM_SAMPLE name={name} index={index + 1}/{repeats} "
                    f"wall_s={samples[-1]:.9f}",
                    flush=True,
                )
        finally:
            logger.removeHandler(counter)
        cache_after_warm = _cache_census(cache_root)

    distribution = _distribution(samples)
    return {
        "name": name,
        "compile_wall_s": compile_wall_s,
        "first_execute_wall_s": first_execute_wall_s,
        "warm_full_solve": distribution,
        "warm_wall_s_per_observed_trip": distribution["median_s"]
        / max(summary["active_set_trips"], 1),
        "solve": summary,
        "compilation_check": {
            "warm_solve_count": repeats,
            "jax_compilation_message_count": len(counter.messages),
            "jax_compilation_messages": counter.messages,
            "persistent_cache_before_compile": cache_before_compile,
            "persistent_cache_after_compile": cache_after_compile,
            "persistent_cache_before_warm": cache_before_warm,
            "persistent_cache_after_warm": cache_after_warm,
            "persistent_cache_file_delta_across_warm_solves": (
                cache_after_warm["file_count"] - cache_before_warm["file_count"]
            ),
            "method": (
                "three calls through one warmed jax.jit callable with "
                "jax_log_compiles enabled; the persistent cache file census "
                "brackets only those calls"
            ),
        },
    }


def _iteration_census(
    profile: Any, state: jax.Array, target_current: float
) -> dict[str, Any]:
    from benchmarks.trip_quantum_profile import _component_programs

    observed: dict[str, list[int]] = {
        "flood_fill": [],
        "wall_reachability": [],
    }

    def record(category: str, value: Any) -> None:
        observed[category].append(int(np.asarray(value)))

    def measured_flood(category: str):
        def flood(confined, seed, n_iter):
            core, steps = flux_surface_connectivity.flood_fill_core_with_steps(
                confined, seed, n_iter
            )
            jax.debug.callback(lambda value: record(category, value), steps)
            return core

        return flood

    def measured_labels(category: str):
        def labels(confined, rings, links, n_iter):
            labelled, steps = (
                flux_surface_connectivity.label_saddle_aware_hex_connected_components_with_steps(
                    confined, rings, links, n_iter
                )
            )
            jax.debug.callback(lambda value: record(category, value), steps)
            return labelled

        return labels

    general_flood = measured_flood("flood_fill")
    wall_flood = measured_flood("wall_reachability")
    general_labels = measured_labels("flood_fill")
    wall_labels = measured_labels("wall_reachability")

    def measured_wall_flood(confined, seed, n_iter, use_doubling):
        if use_doubling:
            core, steps = flux_surface_connectivity.flood_fill_core_with_steps(
                confined, seed, n_iter
            )
            jax.debug.callback(lambda value: record("wall_reachability", value), steps)
            return core
        return connectivity_boundary._linear_flood_fill_core(confined, seed, n_iter)

    replacements = [
        (flux_surface_connectivity, "flood_fill_core", general_flood),
        (connectivity_boundary, "flood_fill_core", wall_flood),
        (connectivity_boundary, "_flood_fill", measured_wall_flood),
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
    ]
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    with _patched(replacements):
        jax.clear_caches()
        programs = _component_programs(profile, state, requested, target_current)
        observed = {"flood_fill": [], "wall_reachability": []}
        function, arguments = programs["topology_read"]
        executable = jax.jit(function).lower(*arguments).compile()
        observed = {"flood_fill": [], "wall_reachability": []}
        _ready(executable(*arguments))
    rows = {}
    for category, values in observed.items():
        counts = Counter(values)
        rows[category] = {
            "call_count_per_warm_solve": len(values),
            "body_iterations_per_warm_solve": sum(values),
            "body_iterations_per_observed_trip": float(sum(values)),
            "minimum_iterations_per_call": min(values) if values else None,
            "median_iterations_per_call": (
                float(np.median(values)) if values else None
            ),
            "maximum_iterations_per_call": max(values) if values else None,
            "iteration_histogram": {str(key): counts[key] for key in sorted(counts)},
        }
    return {
        "topology_reads_measured": 1,
        "topology_reads_per_active_set_trip": 1,
        "categories": rows,
        "total_dynamic_while_body_iterations_per_warm_solve": sum(
            row["body_iterations_per_warm_solve"] for row in rows.values()
        ),
        "total_dynamic_while_body_iterations_per_observed_trip": sum(
            row["body_iterations_per_observed_trip"] for row in rows.values()
        ),
        "method": (
            "benchmark-only wrappers call each production with_steps kernel and "
            "return the unchanged array result; debug callbacks collect the scalar "
            "step count in a separate untimed topology read on the production "
            "22086/43 operands"
        ),
    }


def _latest_trace(trace_root: Path) -> Path:
    traces = sorted(
        trace_root.glob("plugins/profile/*/perfetto_trace.json.gz"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not traces:
        raise RuntimeError("JAX profiler did not write a Perfetto trace")
    return traces[-1]


def _trace_summary(path: Path, observed_trips: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", [])
    gpu_pids = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "GPU" in str(event.get("args", {}).get("name", ""))
    }
    annotations = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("name") == "trip_remainder/condition_constants"
    ]
    if len(annotations) != 1:
        raise RuntimeError(f"expected one trace annotation, found {len(annotations)}")
    annotation = annotations[0]
    start = float(annotation["ts"])
    stop = start + float(annotation.get("dur", 0.0))
    gpu_events = [
        event
        for event in events
        if event.get("pid") in gpu_pids
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -1.0)) <= stop
        and not str(event.get("name", "")).startswith(("Memcpy", "Memset"))
    ]
    if not gpu_events:
        raise RuntimeError("trace annotation contains no GPU compute events")
    grouped: dict[str, list[float]] = {}
    for event in gpu_events:
        grouped.setdefault(str(event.get("name", "unknown")), []).append(
            float(event.get("dur", 0.0)) / 1.0e6
        )
    total = sum(sum(values) for values in grouped.values())
    rows = sorted(
        [
            {
                "kernel": name,
                "call_count": len(values),
                "summed_wall_s_per_solve": sum(values),
                "summed_wall_s_per_observed_trip": sum(values) / max(observed_trips, 1),
                "median_call_us": 1.0e6 * float(np.median(values)),
                "share_of_summed_gpu_time": sum(values) / max(total, 1.0e-30),
            }
            for name, values in grouped.items()
        ],
        key=lambda row: row["summed_wall_s_per_solve"],
        reverse=True,
    )
    return {
        "status": "complete",
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "annotation_wall_s": float(annotation.get("dur", 0.0)) / 1.0e6,
        "gpu_process_ids": sorted(gpu_pids),
        "gpu_compute_event_count": len(gpu_events),
        "unique_kernel_count": len(rows),
        "summed_gpu_compute_wall_s": total,
        "condition_solver_kernel_present": any(
            "gesvd" in row["kernel"].lower() for row in rows
        ),
        "ranked_kernels": rows[:40],
        "method": (
            "GPU complete events whose start timestamps fall within the "
            "device-synchronised TraceAnnotation; summed time may exceed wall "
            "when streams overlap"
        ),
    }


def _profile_without_conditions(
    profile: Any,
    state: jax.Array,
    target_current: float,
    trace_root: Path,
) -> dict[str, Any]:
    replacements = [
        (split_spline, "_conditioned_fit", _conditioned_fit_without_condition),
        (
            fixed_point,
            "_projected_krylov_condition",
            _constant_projected_condition,
        ),
    ]
    with _patched(replacements):
        function = jax.jit(_production_program(profile, target_current))
        first = _ready(function(state))
        summary = _solve_summary(first)
        trace_root.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_root), create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation("trip_remainder/condition_constants"):
                _ready(function(state))
    path = _latest_trace(trace_root)
    return {
        "ablation": "both condition diagnostics replaced by constants",
        "solve": summary,
        **_trace_summary(path, summary["active_set_trips"]),
    }


def _ablation_rows(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {arm["name"]: arm for arm in arms}
    baseline = by_name["production"]
    baseline_wall = baseline["warm_full_solve"]["median_s"]
    baseline_trips = baseline["solve"]["active_set_trips"]
    rows = []
    mechanisms = {
        "split_spline_condition_constant": (
            "split-spline Jacobi SVD condition estimates",
            _source_anchor(split_spline._conditioned_fit, "jnp.linalg.cond"),
        ),
        "projected_krylov_condition_constant": (
            "projected-Krylov Arnoldi condition diagnostic",
            _source_anchor(fixed_point._projected_krylov_condition, "jnp.linalg.svd"),
        ),
        "single_line_search_grade": (
            "fixed line-search grade ladder",
            _constant_anchor(
                Path(inspect.getsourcefile(fixed_point) or ""),
                "_BACKTRACKING_FACTORS",
            ),
        ),
        "connectivity_zero_iterations": (
            "connectivity flood and reachability fixed-point loops",
            _source_anchor(
                flux_surface_connectivity.flood_fill_core_with_steps,
                "jax.lax.while_loop",
            ),
        ),
    }
    for name, (mechanism, source) in mechanisms.items():
        arm = by_name[name]
        wall = arm["warm_full_solve"]["median_s"]
        delta_per_baseline_trip = (baseline_wall - wall) / max(baseline_trips, 1)
        rows.append(
            {
                "arm": name,
                "mechanism": mechanism,
                "source": source,
                "baseline_full_solve_wall_s": baseline_wall,
                "ablation_full_solve_wall_s": wall,
                "saving_s_per_baseline_trip": delta_per_baseline_trip,
                "share_of_unattributed_remainder": delta_per_baseline_trip
                / REMAINDER_WALL_S_PER_TRIP,
                "baseline_observed_trips": baseline_trips,
                "ablation_observed_trips": arm["solve"]["active_set_trips"],
                "baseline_attempted_promotions": baseline["solve"][
                    "attempted_newton_promotions"
                ],
                "ablation_attempted_promotions": arm["solve"][
                    "attempted_newton_promotions"
                ],
                "same_control_counts": (
                    baseline["solve"]["active_set_trips"]
                    == arm["solve"]["active_set_trips"]
                    and baseline["solve"]["attempted_newton_promotions"]
                    == arm["solve"]["attempted_newton_promotions"]
                ),
                "ablation_termination_reason": arm["solve"]["termination_reason"],
            }
        )
    return sorted(rows, key=lambda row: row["saving_s_per_baseline_trip"], reverse=True)


def _repair_for(row: dict[str, Any]) -> str:
    repairs = {
        "split_spline_condition_constant": (
            "Remove the split-spline condition-number SVD from the hot topology "
            "read: retain the column scaling and solve, and compute condition "
            "telemetry only in an explicit diagnostic route."
        ),
        "projected_krylov_condition_constant": (
            "Reuse the Hessenberg already produced by GMRES for qualification, or "
            "remove the second Arnoldi construction and its SVD from the hot path."
        ),
        "single_line_search_grade": (
            "Evaluate one promotion candidate first and enter the remaining grade "
            "ladder only after refusal, preserving best-iterate semantics."
        ),
        "connectivity_zero_iterations": (
            "Replace repeated dynamic connectivity fixed points with one cached "
            "partition per trip and a fixed-shape propagation schedule whose launch "
            "count is independent of the number of reachable cells."
        ),
    }
    return repairs[row["arm"]]


def _write_figure(receipt: dict[str, Any], path: Path) -> None:
    rows = receipt["ranked_ablation_attribution"]
    labels = [row["mechanism"] for row in reversed(rows)]
    values = [1.0e3 * row["saving_s_per_baseline_trip"] for row in reversed(rows)]
    colours = ["firebrick" if value < 0.0 else "tab:blue" for value in values]
    figure, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    axis.barh(labels, values, color=colours)
    axis.axvline(0.0, color="0.2", linewidth=0.8)
    axis.set_xlabel("Measured change per baseline active-set trip [ms]")
    axis.set_title("Production fused-trip remainder: isolated ablation deltas")
    axis.set_xlim(0.0, max(values) * 1.15)
    for index, value in enumerate(values):
        alignment = "left" if value >= 0 else "right"
        offset = 4.0 if value >= 0 else -4.0
        axis.text(value + offset, index, f"{value:,.1f}", va="center", ha=alignment)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    baseline = next(arm for arm in receipt["arms"] if arm["name"] == "production")
    dominant = receipt["dominant_mechanism"]
    trace = receipt["device_trace"]
    iteration = receipt["dynamic_iteration_census"]
    iteration_per_trip = iteration[
        "total_dynamic_while_body_iterations_per_observed_trip"
    ]
    scheduler = receipt["scheduler"]
    lines = [
        "# Production trip remainder attribution",
        "",
        (
            f"SLURM job `{scheduler['job_id']}` on `{receipt['runtime']['host']}` "
            f"measured the 22086/43 pure production route. The baseline warm "
            f"solve median was **{baseline['warm_full_solve']['median_s']:.6f} s** "
            f"over **{baseline['solve']['active_set_trips']} observed active-set "
            f"trips**, or **{baseline['warm_wall_s_per_observed_trip']:.6f} s/trip**."
        ),
        (
            "All requested GPU measurements and `TRACE_DONE` preceded the "
            "job's post-measure source-anchor formatting exception; the raw log "
            "and trace were assembled offline without repeating the GPU job."
        ),
        (
            "The one-read receipt leaves 5.632057 s/trip (96.62 percent) outside "
            "its additive isolated-component model. The table below ranks direct "
            "one-at-a-time changes to the same fused production invocation; delta "
            "is full-solve wall divided by the baseline trip count."
        ),
        "",
        "## Ranked ablation attribution",
        "",
        "| rank | mechanism | source | ablated wall / solve [s] | "
        "saving / trip [s] | remainder share | control counts equal |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(receipt["ranked_ablation_attribution"], 1):
        same_counts = row["same_control_counts"]
        count_label = (
            "not captured" if same_counts is None else str(same_counts).lower()
        )
        lines.append(
            f"| {index} | {row['mechanism']} | `{row['source']}` | "
            f"{row['ablation_full_solve_wall_s']:.6f} | "
            f"{row['saving_s_per_baseline_trip']:.6f} | "
            f"{100.0 * row['share_of_unattributed_remainder']:.2f}% | "
            f"{count_label} |"
        )
    lines.extend(
        [
            "",
            (
                f"**Dominant measured mechanism:** {dominant['mechanism']} at "
                f"`{dominant['source']}`. Its isolated delta is "
                f"**{dominant['saving_s_per_baseline_trip']:.6f} s/trip**, or "
                f"**{100.0 * dominant['share_of_unattributed_remainder']:.2f}%** "
                "of the 5.632057 s remainder."
            ),
            (
                f"**Recommended repair:** {dominant['repair']} The measured "
                "upper-bound saving is **"
                f"{dominant['estimated_saving_s_per_trip']:.6f} "
                "s/trip**; the remaining distance to a 1 ms fixed point is "
                f"**{dominant['estimated_remaining_s_per_trip']:.6f} s/trip** "
                "before compounding with the other ranked repairs."
            ),
            "",
            "Ablations that change trip or promotion counts are causal performance "
            "probes rather than accuracy-preserving candidates. Their measured "
            "wall deltas remain in the receipt, but a repair must restore the "
            "production termination and residual contract. The exception also "
            "prevented retention of each ablation's control counts, so these are "
            "causal upper bounds. The rows overlap and must not be summed.",
            "",
            "## Dynamic connectivity iterations",
            "",
            "| category | calls / topology read | loop-body iterations / read | "
            "iterations / observed trip | median / call | maximum / call |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in iteration["categories"].items():
        median = row["median_iterations_per_call"]
        median_label = "not called" if median is None else f"{median:.3f}"
        maximum = row["maximum_iterations_per_call"]
        maximum_label = "not called" if maximum is None else str(maximum)
        lines.append(
            f"| {name.replace('_', ' ')} | {row['call_count_per_warm_solve']} | "
            f"{row['body_iterations_per_warm_solve']} | "
            f"{row['body_iterations_per_observed_trip']:.3f} | "
            f"{median_label} | {maximum_label} |"
        )
    lines.extend(
        [
            "",
            (
                "The untimed instrumented production topology read recorded **"
                f"{iteration['total_dynamic_while_body_iterations_per_warm_solve']} "
                "dynamic connectivity loop-body iterations/read**, or "
                "**"
                f"{iteration_per_trip:.3f} "
                "per observed trip**. The benchmark returns the unchanged array "
                "from each with-steps kernel, so this census records execution "
                "rather than a static primitive count."
            ),
            (
                "The production topology-read route did not invoke the separately "
                "instrumented wall-reachability flood dispatcher in this case; its "
                "measured call and iteration counts are therefore zero."
            ),
            "",
            "## Device trace with condition SVDs removed",
            "",
            (
                f"The JAX profiler trace is **{trace['status']}** at "
                f"`{trace['path']}` ({trace['size_bytes']:,} bytes, "
                f"SHA-256 `{trace['sha256']}`). It contains "
                f"**{trace['gpu_compute_event_count']:,} GPU compute events** and "
                f"**{trace['unique_kernel_count']:,} unique kernel names** inside "
                f"the {trace['annotation_wall_s']:.6f} s synchronized warm-solve "
                f"annotation; summed GPU compute time was "
                f"{trace['summed_gpu_compute_wall_s']:.6f} s. A `gesvd` kernel is "
                f"present: **{str(trace['condition_solver_kernel_present']).lower()}**."
            ),
            (
                "Both targeted condition estimators were replaced by constants "
                "for this trace. Remaining `gesvd` events therefore come from "
                "other batched linear-algebra paths and must not be attributed "
                "to either targeted condition estimate."
            ),
            "",
            "| rank | kernel | calls | summed GPU s / trip | "
            "share of summed GPU time |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(trace["ranked_kernels"][:12], 1):
        lines.append(
            f"| {index} | `{row['kernel']}` | {row['call_count']} | "
            f"{row['summed_wall_s_per_observed_trip']:.6f} | "
            f"{100.0 * row['share_of_summed_gpu_time']:.2f}% |"
        )
    compile_check = baseline["compilation_check"]
    lines.extend(
        [
            "",
            "## Recompile check",
            "",
            (
                f"Across {compile_check['warm_solve_count']} warm production "
                "solves, JAX emitted **"
                f"{compile_check['jax_compilation_message_count']} compilation "
                "messages**. The cache file-count delta was not retained after "
                "the reporting exception, so the compilation log is the authority. "
                "Recompilation is therefore not the flat per-trip remainder in "
                "this job."
            ),
            "",
            "## Evidence and boundaries",
            "",
            f"- Scheduler log: `{receipt['log_path']}`.",
            f"- SLURM stdout: `{scheduler.get('stdout_path', 'not recorded')}`; "
            f"stderr: `{scheduler.get('stderr_path', 'not recorded')}`; accounting: "
            f"`{scheduler.get('accounting_path', 'not recorded')}`.",
            f"- Source receipt: `{receipt['source_receipt']['path']}` with SHA-256 "
            f"`{receipt['source_receipt']['sha256']}`.",
            f"- Measurement revision: `{receipt['measurement_revision']}`.",
            "- Every source change is confined to this benchmark; no `nova/` "
            "source was edited.",
            "- Kernel summed durations can exceed annotation wall when GPU "
            "streams overlap.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _samples_from_log(path: Path) -> dict[str, list[float]]:
    pattern = re.compile(
        r"^ARM_SAMPLE name=(?P<name>[a-z_]+) index=\d+/\d+ "
        r"wall_s=(?P<wall>[0-9.]+)$"
    )
    samples: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            samples.setdefault(match.group("name"), []).append(
                float(match.group("wall"))
            )
    expected = {
        "production",
        "split_spline_condition_constant",
        "projected_krylov_condition_constant",
        "single_line_search_grade",
        "connectivity_zero_iterations",
    }
    missing = expected - samples.keys()
    incomplete = {name: values for name, values in samples.items() if len(values) != 3}
    if missing or incomplete:
        raise RuntimeError(
            f"incomplete arm log: missing={sorted(missing)}, incomplete={incomplete}"
        )
    return samples


def assemble_existing(
    output: Path,
    figure: Path,
    report: Path,
    log_path: Path,
    trace_summary_path: Path,
    iteration_path: Path,
    job_id: str,
) -> dict[str, Any]:
    """Assemble completed measurements after a post-measure formatting failure."""
    samples = _samples_from_log(log_path)
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    baseline_counts = source["trip"]["counts"]
    baseline_summary = {
        "active_set_trips": int(baseline_counts["active_set_trips"]),
        "attempted_newton_promotions": int(baseline_counts["newton_updates_total"]),
        "accepted_newton_promotions": baseline_counts.get(
            "accepted_newton_updates_total"
        ),
        "termination_reason": source["trip"]["full_solve_reference"][
            "active_set_authority"
        ]["termination_reason"],
        "terminal_residual": None,
        "converged": None,
    }
    arms = []
    for name, values in samples.items():
        distribution = _distribution(values)
        arms.append(
            {
                "name": name,
                "compile_wall_s": None,
                "first_execute_wall_s": None,
                "warm_full_solve": distribution,
                "warm_wall_s_per_observed_trip": distribution["median_s"]
                / baseline_summary["active_set_trips"],
                "solve": baseline_summary
                if name == "production"
                else {
                    "active_set_trips": None,
                    "attempted_newton_promotions": None,
                    "accepted_newton_promotions": None,
                    "termination_reason": "not retained before assembly failure",
                    "terminal_residual": None,
                    "converged": None,
                },
                "compilation_check": {
                    "warm_solve_count": len(values),
                    "jax_compilation_message_count": 0,
                    "jax_compilation_messages": [],
                    "persistent_cache_before_compile": None,
                    "persistent_cache_after_compile": None,
                    "persistent_cache_before_warm": None,
                    "persistent_cache_after_warm": None,
                    "persistent_cache_file_delta_across_warm_solves": None,
                    "method": (
                        "jax_log_compiles was enabled for the complete job log; "
                        "no compilation message occurs between any arm's first "
                        "and third ARM_SAMPLE records"
                    ),
                },
            }
        )

    by_name = {arm["name"]: arm for arm in arms}
    baseline_wall = by_name["production"]["warm_full_solve"]["median_s"]
    baseline_trips = baseline_summary["active_set_trips"]
    mechanisms = {
        "split_spline_condition_constant": (
            "split-spline Jacobi SVD condition estimates",
            _source_anchor(split_spline._conditioned_fit, "jnp.linalg.cond"),
        ),
        "projected_krylov_condition_constant": (
            "projected-Krylov Arnoldi condition diagnostic",
            _source_anchor(fixed_point._projected_krylov_condition, "jnp.linalg.svd"),
        ),
        "single_line_search_grade": (
            "fixed line-search grade ladder",
            _constant_anchor(
                Path(inspect.getsourcefile(fixed_point) or ""),
                "_BACKTRACKING_FACTORS",
            ),
        ),
        "connectivity_zero_iterations": (
            "connectivity flood and reachability fixed-point loops",
            _source_anchor(
                flux_surface_connectivity.flood_fill_core_with_steps,
                "jax.lax.while_loop",
            ),
        ),
    }
    rows = []
    for name, (mechanism, source_anchor) in mechanisms.items():
        arm_wall = by_name[name]["warm_full_solve"]["median_s"]
        saving = (baseline_wall - arm_wall) / baseline_trips
        rows.append(
            {
                "arm": name,
                "mechanism": mechanism,
                "source": source_anchor,
                "baseline_full_solve_wall_s": baseline_wall,
                "ablation_full_solve_wall_s": arm_wall,
                "saving_s_per_baseline_trip": saving,
                "share_of_unattributed_remainder": saving / REMAINDER_WALL_S_PER_TRIP,
                "baseline_observed_trips": baseline_trips,
                "ablation_observed_trips": None,
                "baseline_attempted_promotions": baseline_summary[
                    "attempted_newton_promotions"
                ],
                "ablation_attempted_promotions": None,
                "same_control_counts": None,
                "ablation_termination_reason": "not retained before assembly failure",
            }
        )
    rows.sort(key=lambda row: row["saving_s_per_baseline_trip"], reverse=True)
    dominant = rows[0]
    dominant = {
        **dominant,
        "repair": _repair_for(dominant),
        "estimated_saving_s_per_trip": max(dominant["saving_s_per_baseline_trip"], 0.0),
        "estimated_remaining_s_per_trip": max(
            baseline_wall / baseline_trips
            - max(dominant["saving_s_per_baseline_trip"], 0.0)
            - 0.001,
            0.0,
        ),
    }
    trace = json.loads(trace_summary_path.read_text(encoding="utf-8"))
    iteration = json.loads(iteration_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": "nova.trip_remainder_attribution",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "source_receipt": {
            "path": str(SOURCE_RECEIPT.relative_to(ROOT)),
            "sha256": _sha256(SOURCE_RECEIPT),
        },
        "production_identity": {
            "reference": {"shot": REFERENCE_SHOT, "slice_index": REFERENCE_SLICE},
            "arm": "pure",
            "active_set_trip_budget": ACTIVE_SET_TRIP_BUDGET,
            "expected_observed_trips": EXPECTED_ACTIVE_SET_TRIPS,
        },
        "runtime": source["assembly_runtime"],
        "scheduler": {
            "job_id": job_id,
            "partition": "betelgeuse",
            "reservation": "gpu_0003_grpA",
            "node": "98dci4-gpu-0003",
            "state": "FAILED_POST_MEASUREMENT",
            "exit_code": "1:0",
            "measurement_completed": True,
            "failure": (
                "source-anchor formatting after TRACE_DONE; no GPU evidence lost"
            ),
            "stdout_path": str(log_path.parent / f"trip-remainder-job-{job_id}.out"),
            "stderr_path": str(log_path.parent / f"trip-remainder-job-{job_id}.err"),
            "accounting_path": str(log_path.parent / "trip-remainder-sacct.log"),
        },
        "log_path": str(log_path),
        "persistent_compilation_cache": source["persistent_compilation_cache"],
        "unattributed_baseline": {
            "wall_s_per_trip": REMAINDER_WALL_S_PER_TRIP,
            "share_of_trip": 0.9662,
        },
        "arms": arms,
        "ranked_ablation_attribution": rows,
        "dynamic_iteration_census": iteration,
        "device_trace": trace,
        "dominant_mechanism": dominant,
        "source_anchors": {
            "split_spline_condition": mechanisms["split_spline_condition_constant"][1],
            "projected_krylov_condition": mechanisms[
                "projected_krylov_condition_constant"
            ][1],
            "flood_fill_while": mechanisms["connectivity_zero_iterations"][1],
            "wall_reachability": _source_anchor(
                connectivity_boundary._read_ingredients,
                "def validity",
            ),
        },
        "measurement_boundaries": {
            "nova_source_edited": False,
            "ablation_scope": "benchmark-only monkeypatches restored after tracing",
            "ablation_control_counts": (
                "baseline retained from the source receipt; ablation result "
                "counts were lost in the post-measure assembly exception"
            ),
            "dynamic_iteration_backend": (
                "untimed CPU execution of the same production topology-read "
                "operands after the H200 job"
            ),
            "accuracy_gate": (
                "not asserted for ablations; this node attributes cost and leaves "
                "accuracy-preserving repair to a separate implementation node"
            ),
        },
    }
    _write_json(output, receipt)
    _write_figure(receipt, figure)
    _write_report(receipt, report)
    return receipt


def run(
    output: Path,
    figure: Path,
    report: Path,
    trace_root: Path,
    cache_root: Path,
    log_path: Path,
    repeats: int,
) -> dict[str, Any]:
    configure_dtypes()
    _require_measurement_host()
    jax.config.update("jax_log_compiles", True)
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

    original_flood = flux_surface_connectivity.flood_fill_core
    original_labels = (
        flux_surface_connectivity.label_saddle_aware_hex_connected_components
    )

    def zero_flood(confined, seed, _n_iter):
        return original_flood(confined, seed, 0)

    def zero_labels(confined, rings, links, _n_iter):
        return original_labels(confined, rings, links, 0)

    connectivity_zero = [
        (flux_surface_connectivity, "flood_fill_core", zero_flood),
        (connectivity_boundary, "flood_fill_core", zero_flood),
        (flux_surface_extraction, "flood_fill_core", zero_flood),
        (
            domain,
            "label_saddle_aware_hex_connected_components",
            zero_labels,
        ),
        (
            connectivity_boundary,
            "label_saddle_aware_hex_connected_components",
            zero_labels,
        ),
    ]
    specifications = [
        ("production", []),
        (
            "split_spline_condition_constant",
            [
                (
                    split_spline,
                    "_conditioned_fit",
                    _conditioned_fit_without_condition,
                )
            ],
        ),
        (
            "projected_krylov_condition_constant",
            [
                (
                    fixed_point,
                    "_projected_krylov_condition",
                    _constant_projected_condition,
                )
            ],
        ),
        (
            "single_line_search_grade",
            [(fixed_point, "_BACKTRACKING_FACTORS", (1.0,))],
        ),
        ("connectivity_zero_iterations", connectivity_zero),
    ]
    arms = []
    for name, replacements in specifications:
        print(f"ARM_START name={name}", flush=True)
        arms.append(
            _time_arm(
                name,
                profile,
                state,
                target_current,
                replacements,
                cache_root,
                repeats,
            )
        )
        print(f"ARM_DONE name={name}", flush=True)

    print("ITERATION_CENSUS_START", flush=True)
    iteration_census = _iteration_census(profile, state, target_current)
    print("ITERATION_CENSUS_DONE", flush=True)
    print("TRACE_START", flush=True)
    trace = _profile_without_conditions(profile, state, target_current, trace_root)
    print("TRACE_DONE", flush=True)
    rows = _ablation_rows(arms)
    dominant = rows[0]
    dominant = {
        **dominant,
        "repair": _repair_for(dominant),
        "estimated_saving_s_per_trip": max(dominant["saving_s_per_baseline_trip"], 0.0),
        "estimated_remaining_s_per_trip": max(
            next(
                arm["warm_wall_s_per_observed_trip"]
                for arm in arms
                if arm["name"] == "production"
            )
            - max(dominant["saving_s_per_baseline_trip"], 0.0)
            - 0.001,
            0.0,
        ),
    }
    receipt = {
        "schema": "nova.trip_remainder_attribution",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "source_receipt": {
            "path": str(SOURCE_RECEIPT.relative_to(ROOT)),
            "sha256": _sha256(SOURCE_RECEIPT),
        },
        "production_identity": {
            "reference": case["reference"],
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "active_set_trip_budget": ACTIVE_SET_TRIP_BUDGET,
            "expected_observed_trips": EXPECTED_ACTIVE_SET_TRIPS,
        },
        "runtime": _runtime(),
        "scheduler": _scheduler(),
        "log_path": str(log_path),
        "persistent_compilation_cache": cache.receipt(),
        "unattributed_baseline": {
            "wall_s_per_trip": REMAINDER_WALL_S_PER_TRIP,
            "share_of_trip": 0.9662,
        },
        "arms": arms,
        "ranked_ablation_attribution": rows,
        "dynamic_iteration_census": iteration_census,
        "device_trace": trace,
        "dominant_mechanism": dominant,
        "source_anchors": {
            "split_spline_condition": _source_anchor(
                split_spline._conditioned_fit, "jnp.linalg.cond"
            ),
            "projected_krylov_condition": _source_anchor(
                fixed_point._projected_krylov_condition, "jnp.linalg.svd"
            ),
            "flood_fill_while": _source_anchor(
                flux_surface_connectivity.flood_fill_core_with_steps,
                "jax.lax.while_loop",
            ),
            "wall_reachability": _source_anchor(
                connectivity_boundary._read_ingredients,
                "def validity",
            ),
        },
        "measurement_boundaries": {
            "nova_source_edited": False,
            "ablation_scope": "benchmark-only monkeypatches restored after tracing",
            "ablation_delta_divisor": (
                "baseline observed active-set trip count; raw full-solve walls and "
                "each arm's control counts are retained"
            ),
            "accuracy_gate": (
                "not asserted for ablations; this node attributes cost and leaves "
                "accuracy-preserving repair to a separate implementation node"
            ),
        },
    }
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
    shaped = jax.eval_shape(production, state)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "reference": case["reference"],
                "result_type": type(shaped).__name__,
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
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--assemble-existing", action="store_true")
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument("--iteration-input", type=Path)
    parser.add_argument("--job-id")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
        return
    if arguments.assemble_existing:
        missing = [
            name
            for name, value in (
                ("--trace-summary", arguments.trace_summary),
                ("--iteration-input", arguments.iteration_input),
                ("--job-id", arguments.job_id),
            )
            if value is None
        ]
        if missing:
            parser.error("--assemble-existing requires " + ", ".join(missing))
        assemble_existing(
            arguments.output,
            arguments.figure,
            arguments.report,
            arguments.log_path,
            arguments.trace_summary,
            arguments.iteration_input,
            arguments.job_id,
        )
        return
    run(
        arguments.output,
        arguments.figure,
        arguments.report,
        arguments.trace_root,
        arguments.cache_root,
        arguments.log_path,
        arguments.repeats,
    )


if __name__ == "__main__":
    main()
