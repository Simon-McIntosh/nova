"""Profile one production active-set trip for the frozen MAST pure arm.

The driver rebuilds the persisted-response operator used by the production
solve, compiles the ordinary solve with a one-trip outer budget, and profiles a
warm invocation on one H200. Independent device-synchronised probes preserve
the distinction between directly observed component latency and the
count-scaled attribution of the complete trip wall.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
import gzip
import hashlib
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
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.biot.null import NullBase
from nova.equilibrium import fixed_point
from nova.equilibrium.connectivity_boundary import _read_ingredients
from nova.equilibrium.flux_surface_connectivity import (
    polish_census_stationary_points,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(os.environ.get("NOVA_PROFILE_ROOT", Path(__file__).resolve().parents[4]))
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/profile.json"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/attribution/"
    "trip-quantum-profile.md"
)
PARTIAL_TRACE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/attribution/trip-quantum-trace"
    "/plugins/profile/2026_09_04_14_30_40/perfetto_trace.json.gz"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
DEFAULT_TIMER_LOGS = (
    (
        "1262560",
        DEFAULT_REPORT.parent / "trip-quantum-profile-3-slurm.log",
    ),
    (
        "1262571",
        DEFAULT_REPORT.parent / "trip-quantum-profile-4-slurm.log",
    ),
)
PRIOR_FULL_SOLVE_LOG = DEFAULT_REPORT.parent / "trip-quantum-slurm.log"
PRIOR_FULL_SOLVE_JOB = "1262501"
SOURCE_RECEIPT = ROOT / "docs/figures/polish-support-performance/raw-main.json"
ACTIVE_SET_AUTHORITY = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIP_BUDGET = 24
EXPECTED_ACTIVE_SET_TRIPS = 7
LINE_SEARCH_GRADES_PER_UPDATE = 6
SCAN_ITERATION_SECONDS = 7.6e-6
SPLINE_FITS_PER_TOPOLOGY_READ = 63
TOPOLOGY_READS_PER_OUTER_RESIDUAL = 4
SPLINE_CONDITION_SVDS_PER_FIT = 2


_CACHE_EVENTS: dict[str, int] = {}


def _cache_event_listener(event: str, **_metadata: str | int) -> None:
    if event.startswith("/jax/compilation_cache/"):
        _CACHE_EVENTS[event] = _CACHE_EVENTS.get(event, 0) + 1


jax.monitoring.register_event_listener(_cache_event_listener)


_ORIGINAL_NULL_POST_INIT = NullBase.__post_init__


def _sentinel_safe_null_post_init(self) -> None:
    """Skip derived-size construction for structure-only abstract leaves."""
    if not hasattr(self.coordinate, "shape"):
        return
    _ORIGINAL_NULL_POST_INIT(self)


NullBase.__post_init__ = _sentinel_safe_null_post_init


def _cache_event_snapshot() -> dict[str, int]:
    return dict(_CACHE_EVENTS)


def _cache_event_delta(before: dict[str, int]) -> dict[str, Any]:
    events = {
        name.rsplit("/", 1)[-1]: count - before.get(name, 0)
        for name, count in _CACHE_EVENTS.items()
        if count - before.get(name, 0)
    }
    hits = events.get("cache_hits", 0)
    misses = events.get("cache_misses", 0)
    if hits and misses:
        status = "mixed"
    elif hits:
        status = "hit"
    elif misses:
        status = "miss"
    else:
        status = "not_observed"
    return {"status": status, "events": events}


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


def _active_set_authority() -> dict[str, Any]:
    payload = json.loads(ACTIVE_SET_AUTHORITY.read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload["rows"]
        if row.get("identity") == f"{REFERENCE_SHOT}/{REFERENCE_SLICE}"
        and row.get("arm") == "pure"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one active-set authority row, found {len(matches)}"
        )
    row = matches[0]
    trips = int(row["active_set_iterations"])
    if trips != EXPECTED_ACTIVE_SET_TRIPS:
        raise RuntimeError(f"active-set authority changed from 7 to {trips}")
    return {
        "path": str(ACTIVE_SET_AUTHORITY.relative_to(ROOT)),
        "sha256": _sha256(ACTIVE_SET_AUTHORITY),
        "identity": row["identity"],
        "arm": row["arm"],
        "active_set_iterations": trips,
        "termination_reason": row["termination_reason"],
    }


def _trip_count_observation(
    counts: dict[str, Any], active_set_authority: dict[str, Any]
) -> dict[str, Any]:
    """Record a trip-count change without accepting or rejecting it implicitly."""
    observed = int(counts["active_set_trips"])
    authority = int(active_set_authority["active_set_iterations"])
    return {
        "authority_active_set_trips": authority,
        "observed_active_set_trips": observed,
        "difference": observed - authority,
        "ratio": observed / authority,
        "matches": observed == authority,
        "gate_applied": False,
        "interpretation": (
            "a mismatch is a measured behavior change requiring commit attribution; "
            "it is neither rejected as invalid nor accepted as characteristic"
        ),
    }


def _prior_full_solve_evidence() -> dict[str, Any]:
    text = PRIOR_FULL_SOLVE_LOG.read_text(encoding="utf-8")
    samples = [
        float(value)
        for value in re.findall(
            r"SAMPLE_DONE name=production_trip index=\d+/\d+ wall_s=([0-9.]+)",
            text,
        )
    ]
    if len(samples) != 3:
        raise RuntimeError(f"expected three prior full-solve samples, found {samples}")
    return {
        "job_id": PRIOR_FULL_SOLVE_JOB,
        "log_path": str(PRIOR_FULL_SOLVE_LOG),
        "log_sha256": _sha256(PRIOR_FULL_SOLVE_LOG),
        "warm_solve_wall_s": samples,
        "median_warm_solve_wall_s": float(np.median(samples)),
        "minimum_warm_solve_wall_s": min(samples),
        "maximum_warm_solve_wall_s": max(samples),
    }


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
    return {
        "samples_s": samples,
        "sample_count": len(samples),
        "minimum_s": float(values.min()),
        "median_s": float(np.median(values)),
        "maximum_s": float(values.max()),
        "range_s": float(np.ptp(values)),
        "median_absolute_deviation_s": float(
            np.median(np.abs(values - np.median(values)))
        ),
    }


def _timer_log(job_id: str, path: Path) -> dict[str, Any]:
    samples: dict[str, list[float]] = {}
    pattern = re.compile(
        r"^SAMPLE_DONE name=(?P<name>[a-z_]+) index=\d+/\d+ "
        r"wall_s=(?P<wall>[0-9.]+)$"
    )
    failure = None
    dropped_activity_buffers = False
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            samples.setdefault(match.group("name"), []).append(
                float(match.group("wall"))
            )
        if "cusolverDnDgesvdj" in line:
            failure = "cusolverDnDgesvdj failed during jit_one_trip profiler replay"
        if "drop the buffer" in line:
            dropped_activity_buffers = True
    required = {
        "production_trip",
        "forward_evaluation",
        "jacobian_vector_product",
        "gmres_orthogonalisation",
        "line_search",
        "topology_read",
        "candidate_census",
        "spline_fits",
        "flood_fills",
        "wall_reachability",
        "separatrix",
    }
    missing = sorted(required - samples.keys())
    incomplete = sorted(name for name, values in samples.items() if len(values) != 3)
    if missing or incomplete:
        detail = f"missing={missing}, incomplete={incomplete}"
        raise RuntimeError(f"timer log {path} is incomplete: {detail}")
    if failure is None:
        raise RuntimeError(f"timer log {path} does not retain the profiler failure")
    return {
        "job_id": job_id,
        "log_path": str(path),
        "log_sha256": _sha256(path),
        "scheduler": {
            "state": "FAILED",
            "exit_code": "1:0",
            "partition": "betelgeuse",
            "node": "98dci4-gpu-0003",
            "reservation": "gpu_0003_grpA",
            "time_limit": "01:00:00",
        },
        "timings": {name: _distribution(values) for name, values in samples.items()},
        "timing_phase_complete": True,
        "profiler_replay_failure": failure,
        "cupti_dropped_activity_buffers": dropped_activity_buffers,
    }


def _central_probes(
    timer_runs: list[dict[str, Any]],
    control_flow: dict[str, dict[str, int]],
) -> dict[str, Any]:
    names = timer_runs[0]["timings"].keys()
    return {
        name: {
            "steady": _distribution(
                [run["timings"][name]["median_s"] for run in timer_runs]
            ),
            "run_medians_s": {
                run["job_id"]: run["timings"][name]["median_s"] for run in timer_runs
            },
            "control_flow": control_flow[name],
        }
        for name in names
    }


def _nested_jaxprs(value: Any):
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
    elif hasattr(value, "eqns"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_jaxprs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_jaxprs(item)


def _control_flow_census(
    function: Callable, arguments: tuple[Any, ...]
) -> dict[str, int]:
    """Bound fixed scan trips while retaining dynamic whiles separately."""
    closed = jax.make_jaxpr(function)(*arguments)
    while_primitives = 0

    def scan_cost(jaxpr, *, branch_minimum: bool) -> int:
        nonlocal while_primitives
        total = 0
        for equation in jaxpr.eqns:
            name = equation.primitive.name
            if name == "while":
                while_primitives += 1
                continue
            if name == "cond":
                branch_costs = [
                    scan_cost(branch.jaxpr, branch_minimum=branch_minimum)
                    for branch in equation.params.get("branches", ())
                ]
                if branch_costs:
                    total += min(branch_costs) if branch_minimum else sum(branch_costs)
                continue
            nested = sum(
                scan_cost(child, branch_minimum=branch_minimum)
                for child in _nested_jaxprs(equation.params)
            )
            if name == "scan":
                length = int(equation.params["length"])
                total += length * (1 + nested)
            else:
                total += nested
        return total

    lower = scan_cost(closed.jaxpr, branch_minimum=True)
    compiled = scan_cost(closed.jaxpr, branch_minimum=False)
    return {
        "fixed_scan_trip_lower_bound": lower,
        "fixed_scan_trip_compiled_branch_sum": compiled,
        "dynamic_while_primitives": while_primitives // 2,
    }


def _callback_census(function: Callable, arguments: tuple[Any, ...]) -> dict[str, int]:
    closed = jax.make_jaxpr(function)(*arguments)
    counts: dict[str, int] = {}

    def visit(jaxpr) -> None:
        for equation in jaxpr.eqns:
            name = equation.primitive.name
            if "callback" in name:
                counts[name] = counts.get(name, 0) + 1
            for child in _nested_jaxprs(equation.params):
                visit(child)

    visit(closed.jaxpr)
    return counts


def _compile_probe(
    name: str,
    function: Callable,
    arguments: tuple[Any, ...],
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    print(f"COMPILE_START name={name}", flush=True)
    cache_before = _cache_event_snapshot()
    started = time.perf_counter()
    executable = jax.jit(function).lower(*arguments).compile()
    compile_wall = time.perf_counter() - started
    started = time.perf_counter()
    _ready(executable(*arguments))
    first_wall = time.perf_counter() - started
    samples = []
    for index in range(repeats):
        started = time.perf_counter()
        _ready(executable(*arguments))
        samples.append(time.perf_counter() - started)
        print(
            f"SAMPLE_DONE name={name} index={index + 1}/{repeats} "
            f"wall_s={samples[-1]:.9f}",
            flush=True,
        )
    result = {
        "compile_wall_s": compile_wall,
        "first_execute_wall_s": first_wall,
        "steady": _distribution(samples),
        "control_flow": _control_flow_census(function, arguments),
        "persistent_compilation_cache": _cache_event_delta(cache_before),
    }
    return executable, result


def _profile_trace(
    trace_root: Path,
    executables: dict[str, tuple[Any, tuple[Any, ...]]],
) -> Path:
    trace_root.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_root), create_perfetto_trace=True):
        for name, (executable, arguments) in executables.items():
            with jax.profiler.TraceAnnotation(f"trip_quantum/{name}"):
                _ready(executable(*arguments))
    traces = sorted(
        trace_root.glob("plugins/profile/*/perfetto_trace.json.gz"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not traces:
        raise RuntimeError("JAX profiler did not write a Perfetto trace")
    return traces[-1]


def _trace_summary(path: Path) -> dict[str, Any]:
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
    annotations = {
        str(event.get("name")).split("trip_quantum/", 1)[1]: event
        for event in events
        if event.get("ph") == "X" and "trip_quantum/" in str(event.get("name"))
    }
    rows = {}
    for name, annotation in annotations.items():
        start = float(annotation["ts"])
        stop = start + float(annotation.get("dur", 0.0))
        inside = [
            event
            for event in events
            if event.get("pid") in gpu_pids
            and event.get("ph") == "X"
            and start <= float(event.get("ts", -1.0)) <= stop
        ]
        transfers = [
            event for event in inside if str(event.get("name", "")).startswith("Memcpy")
        ]
        compute = [
            event
            for event in inside
            if not str(event.get("name", "")).startswith(("Memcpy", "Memset"))
        ]
        rows[name] = {
            "annotation_wall_s": float(annotation.get("dur", 0.0)) / 1.0e6,
            "gpu_compute_launch_count": len(compute),
            "gpu_transfer_count": len(transfers),
            "summed_gpu_compute_wall_s": sum(
                float(event.get("dur", 0.0)) for event in compute
            )
            / 1.0e6,
            "summed_gpu_transfer_wall_s": sum(
                float(event.get("dur", 0.0)) for event in transfers
            )
            / 1.0e6,
        }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "gpu_process_ids": sorted(gpu_pids),
        "regions": rows,
        "method": (
            "compute launches are GPU complete events within synchronized "
            "TraceAnnotation intervals; transfers are reported separately"
        ),
    }


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted = None
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
            accepted = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "accepted_time_limit": accepted,
        "temporary_directory": os.environ.get("TMPDIR"),
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


def _require_h200() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError("measurement requires one JAX-visible H200")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")


def _production_program(profile: Any, target_current: float):
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)

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

    return requested, one_trip


def _component_programs(
    profile: Any, state: jax.Array, requested: jax.Array, target_current: float
):
    operator = profile.operator
    physical = state[: operator.physical_node_number]
    fixed_topology = operator._fixed_design_topology
    mask = operator.residual_shadow_mask(state, requested)
    shadowed_map = operator.flux_map_with_shadow(
        requested_class=requested,
        target_current=target_current,
    )
    mapped, _tangent = jax.linearize(lambda value: shadowed_map(value, mask), state)
    residual = mapped - state
    direction = residual / jnp.maximum(jnp.max(jnp.abs(residual)), 1.0e-30)
    _masks, topology, _connected, _admitted = operator._fixed_design_read(
        physical, requested
    )
    grid_flux, _wall_flux = fixed_topology.split_flux_map(physical)
    selected_axis = jnp.concatenate((topology.axis, topology.axis_flux[None]))
    selected_x = jnp.concatenate((topology.x_point, topology.x_point_flux[None]))
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape

    def forward_evaluation(candidate):
        return shadowed_map(candidate, mask)

    def jacobian_vector_product(candidate, vector):
        return jax.jvp(forward_evaluation, (candidate,), (vector,))[1]

    diagonal = jnp.linspace(0.5, 1.5, state.size, dtype=state.dtype)

    def gmres_orthogonalisation(right_hand_side):
        return jax.scipy.sparse.linalg.gmres(
            lambda vector: diagonal * vector,
            right_hand_side,
            maxiter=1,
            restart=parity.GMRES_ITERATIONS,
            solve_method="batched",
        )

    def line_search(candidate):
        candidate_mapped, candidate_tangent = jax.linearize(
            forward_evaluation, candidate
        )
        candidate_residual = candidate_mapped - candidate

        def model_map(value):
            return candidate_mapped + candidate_tangent(value - candidate)

        return fixed_point._backtracked_promotion(
            forward_evaluation,
            model_map,
            candidate,
            direction,
            parity.RELAXATION * candidate_residual,
            fixed_point._relative_residual(candidate_mapped, candidate),
            fixed_point._smooth_relative_sup_merit(candidate_mapped, candidate),
            jnp.asarray(1.0, dtype=candidate.dtype),
            True,
            acceptance_map_fn=forward_evaluation,
            own_mask_acceptance=True,
        )

    def candidate_census(candidate_physical):
        candidate_grid, _candidate_wall = fixed_topology.split_flux_map(
            candidate_physical
        )
        return fixed_topology.grid(candidate_grid)

    def spline_fits(candidate_physical):
        candidate_grid, _candidate_wall = fixed_topology.split_flux_map(
            candidate_physical
        )
        values = candidate_grid.reshape((radial_count, vertical_count)).T
        return polish_census_stationary_points(
            values,
            radius,
            height,
            topology.boundary_flux,
            operator.polarity,
            selected_axis,
            selected_x,
        )

    closed = fixed_topology.psi_mask(
        operator.polarity, grid_flux, topology.boundary_flux
    )

    def flood_fill(candidate_grid):
        return fixed_topology.axis_component(
            candidate_grid,
            topology.boundary_flux,
            topology.axis_flux,
            topology.axis,
            closed,
            operator.inside_material,
            jnp.asarray(True),
            topology.x_point,
        )

    def topology_read(candidate):
        return operator.residual_shadow_mask(candidate, requested)

    def wall_reachability(candidate_physical):
        candidate_grid, candidate_wall = fixed_topology.split_flux_map(
            candidate_physical
        )
        candidate_x = fixed_topology.grid(candidate_grid)[1]
        _seed, material = operator.connectivity_axis_seed(topology.axis)
        classification_wall = jnp.concatenate(
            (topology.wall_point, topology.wall_point_flux[None])
        )
        return _read_ingredients(
            candidate_grid.reshape((radial_count, vertical_count)).T,
            radius,
            height,
            material.reshape((radial_count, vertical_count)).T,
            topology.axis[0],
            topology.axis[1],
            96,
            18,
            operator.wall.coordinate[:, 0],
            operator.wall.coordinate[:, 1],
            candidate_wall,
            jnp.asarray(jnp.nan, dtype=candidate_grid.dtype),
            True,
            candidate_x,
            classification_wall,
        )

    def separatrix(candidate_physical):
        return operator._connectivity_read(candidate_physical, topology, classify=True)

    return {
        "forward_evaluation": (forward_evaluation, (state,)),
        "jacobian_vector_product": (jacobian_vector_product, (state, direction)),
        "gmres_orthogonalisation": (gmres_orthogonalisation, (residual,)),
        "line_search": (line_search, (state,)),
        "topology_read": (topology_read, (state,)),
        "candidate_census": (candidate_census, (physical,)),
        "spline_fits": (spline_fits, (physical,)),
        "flood_fills": (flood_fill, (grid_flux,)),
        "wall_reachability": (wall_reachability, (physical,)),
        "separatrix": (separatrix, (physical,)),
    }


def _count_summary(
    trips: int,
    updates: int,
    *,
    accepted: int | None = None,
    backtracks: list[int] | None = None,
    recovery_activations: int | None = None,
    model_rebuild_activations: int | None = None,
    descent_activations: int | None = None,
) -> dict[str, Any]:
    line_grades = updates * LINE_SEARCH_GRADES_PER_UPDATE
    jvp_count = updates * parity.GMRES_ITERATIONS
    residual_evaluations = updates + line_grades
    if trips <= 0:
        raise RuntimeError("production solve reported no active-set trips")
    return {
        "active_set_trips": trips,
        "newton_updates_total": updates,
        "newton_updates_per_trip": updates / trips,
        "accepted_newton_updates_total": accepted,
        "accepted_newton_updates_per_trip": (
            accepted / trips if accepted is not None else None
        ),
        "configured_gmres_iterations_per_update": parity.GMRES_ITERATIONS,
        "gmres_jacobian_vector_products_total": jvp_count,
        "gmres_jacobian_vector_products_per_trip": jvp_count / trips,
        "fixed_line_search_grades_per_update": LINE_SEARCH_GRADES_PER_UPDATE,
        "line_search_grades_total": line_grades,
        "line_search_grades_per_trip": line_grades / trips,
        "primal_residual_evaluations_total": updates,
        "primal_residual_evaluations_per_trip": updates / trips,
        "residual_evaluations_total": residual_evaluations,
        "residual_evaluations_per_trip": residual_evaluations / trips,
        "residual_evaluation_equivalents_total": updates + jvp_count + line_grades,
        "residual_evaluation_equivalents_per_trip": (updates + jvp_count + line_grades)
        / trips,
        "promotion_backtrack_counts": backtracks,
        "recovery_activations": recovery_activations,
        "model_rebuild_activations": model_rebuild_activations,
        "descent_activations": descent_activations,
        "counting_contract": (
            "residual evaluations count one primal residual for each Newton "
            "update plus all six fixed line-search grades; configured GMRES "
            "Jacobian-vector products are counted separately, and their sum is "
            "reported only as residual-evaluation equivalents"
        ),
    }


def _counts(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    trips = int(np.asarray(history.active_set_iterations))
    updates = int(np.asarray(history.attempted_newton_promotions))
    accepted = int(np.asarray(history.accepted_newton_promotions))
    backtracks = np.asarray(history.promotion_backtrack_counts, dtype=np.int64)
    recovery = np.asarray(history.promotion_recovery_activations, dtype=np.int64)[
        :updates
    ]
    rebuild = np.asarray(history.promotion_model_rebuild_activations, dtype=np.int64)[
        :updates
    ]
    descent = np.asarray(history.promotion_descent_activations, dtype=np.int64)[
        :updates
    ]
    return _count_summary(
        trips,
        updates,
        accepted=accepted,
        backtracks=backtracks[:updates].tolist(),
        recovery_activations=int(np.count_nonzero(recovery == 1)),
        model_rebuild_activations=int(np.count_nonzero(rebuild == 1)),
        descent_activations=int(np.count_nonzero(descent == 1)),
    )


def _attribution(
    trip_wall: float,
    counts: dict[str, Any],
    probes: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    median = {name: row["steady"]["median_s"] for name, row in probes.items()}
    updates = counts["newton_updates_per_trip"]
    jvp_count = counts["gmres_jacobian_vector_products_per_trip"]
    residual_evaluations = counts["residual_evaluations_per_trip"]
    topology_reads = residual_evaluations * TOPOLOGY_READS_PER_OUTER_RESIDUAL
    line_search_exclusive = max(
        median["line_search"]
        - LINE_SEARCH_GRADES_PER_UPDATE * median["forward_evaluation"],
        0.0,
    )
    attributed = {
        "topology_read": topology_reads * median["topology_read"],
        "forward_evaluation": residual_evaluations * median["forward_evaluation"],
        "jacobian_vector_product": jvp_count * median["jacobian_vector_product"],
        "gmres_orthogonalisation": updates * median["gmres_orthogonalisation"],
        "line_search": updates * line_search_exclusive,
    }
    directly_attributed = sum(attributed.values())
    attributed["host_dispatch_or_device_sync"] = max(
        trip_wall - directly_attributed, 0.0
    )
    divisor = {
        "forward_evaluation": residual_evaluations,
        "topology_read": topology_reads,
        "jacobian_vector_product": jvp_count,
        "gmres_orthogonalisation": updates,
        "line_search": updates,
        "host_dispatch_or_device_sync": 1,
    }
    rows = [
        {
            "component": name,
            "wall_s_per_trip": wall,
            "wall_s_per_evaluation": (
                line_search_exclusive
                if name == "line_search"
                else median.get(name, wall / max(divisor.get(name, 1), 1))
            ),
            "raw_direct_probe_wall_s_per_evaluation": median.get(name),
            "share_of_trip": wall / trip_wall,
        }
        for name, wall in attributed.items()
    ]
    return sorted(rows, key=lambda row: row["wall_s_per_trip"], reverse=True), {
        "method": (
            "direct synchronized probe medians multiplied by exact trip counts; "
            "the line-search row subtracts its six separately counted forward "
            "grades, topology reads use the measured four-per-residual reuse-map "
            "count, and the nonnegative timer remainder retains fused device work, "
            "host dispatch, and final synchronization because CUPTI evidence failed"
        ),
        "topology_reads_per_trip": topology_reads,
        "directly_attributed_wall_s_per_trip": directly_attributed,
        "unattributed_timer_remainder_s_per_trip": attributed[
            "host_dispatch_or_device_sync"
        ],
        "remainder_scope": (
            "upper bound on host dispatch plus final synchronization; without a "
            "complete profiler trace it also contains fused device work absent "
            "from the isolated component model"
        ),
    }


def _topology_breakdown(
    topology_trip_wall: float, probes: dict[str, Any]
) -> list[dict[str, Any]]:
    direct = {
        name: probes[name]["steady"]["median_s"]
        for name in (
            "candidate_census",
            "spline_fits",
            "flood_fills",
            "wall_reachability",
        )
    }
    direct["separatrix"] = max(
        probes["separatrix"]["steady"]["median_s"]
        - probes["wall_reachability"]["steady"]["median_s"],
        0.0,
    )
    denominator = sum(direct.values())
    return sorted(
        [
            {
                "component": name,
                "direct_probe_wall_s": value,
                "attributed_wall_s_per_trip": topology_trip_wall
                * value
                / max(denominator, 1.0e-30),
                "share_of_topology": value / max(denominator, 1.0e-30),
            }
            for name, value in direct.items()
        ],
        key=lambda row: row["attributed_wall_s_per_trip"],
        reverse=True,
    )


def _evaluation_census(probes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for name, probe in probes.items():
        control_flow = probe["control_flow"]
        rows[name] = {
            "gpu_compute_launches_per_evaluation": None,
            "gpu_transfers_per_evaluation": None,
            "launch_count_status": (
                "not captured: both CUPTI replays failed in cusolverDnDgesvdj "
                "and the partial trace is unpromoted"
            ),
            "fixed_scan_trip_lower_bound_per_evaluation": control_flow[
                "fixed_scan_trip_lower_bound"
            ],
            "fixed_scan_trip_compiled_branch_sum_per_evaluation": control_flow[
                "fixed_scan_trip_compiled_branch_sum"
            ],
            "dynamic_while_primitives": control_flow["dynamic_while_primitives"],
        }
    return rows


def _ranked_bottlenecks(
    components: list[dict[str, Any]], topology: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    component = {row["component"]: row for row in components}
    topology_row = topology[0]
    candidates = [
        {
            "bottleneck": "topology read: "
            + topology_row["component"].replace("_", " "),
            "wall_s_per_trip": component["topology_read"]["wall_s_per_trip"],
            "repair": (
                "land the shared TensorBSpline authority so each topology read fits "
                "once rather than 63 times, then reuse one topology partition across "
                "residual grades"
            ),
            "owner_plan": "null-identification-authority",
        },
        {
            "bottleneck": "host, synchronization, or unmodelled fused-device remainder",
            "wall_s_per_trip": component["host_dispatch_or_device_sync"][
                "wall_s_per_trip"
            ],
            "repair": (
                "move reconciliation under one compiled boundary and add a profiler "
                "route that isolates solver-library replay from component timing"
            ),
            "owner_plan": "millisecond-converged-solve",
        },
        {
            "bottleneck": "Jacobian-vector products",
            "wall_s_per_trip": component["jacobian_vector_product"]["wall_s_per_trip"],
            "repair": (
                "reuse or compress the linearized operator and reduce the Krylov "
                "action budget before changing nonlinear acceptance"
            ),
            "owner_plan": "solver-convergence-regression",
        },
        {
            "bottleneck": "line-search residual grades",
            "wall_s_per_trip": component["line_search"]["wall_s_per_trip"],
            "repair": (
                "batch or prune the fixed promotion ladder while retaining own-mask "
                "acceptance and best-iterate semantics"
            ),
            "owner_plan": "millisecond-converged-solve",
        },
        {
            "bottleneck": "GMRES orthogonalisation and qualification",
            "wall_s_per_trip": component["gmres_orthogonalisation"]["wall_s_per_trip"],
            "repair": "fuse Arnoldi reductions and avoid duplicated projections",
            "owner_plan": "solver-convergence-regression",
        },
    ]
    return sorted(candidates, key=lambda row: row["wall_s_per_trip"], reverse=True)[:3]


def _write_report(receipt: dict[str, Any], output: Path) -> None:
    counts = receipt["trip"]["counts"]
    prior = receipt["trip"]["prior_full_solve"]
    prior_trip_wall = prior["median_warm_solve_wall_s"] / counts["active_set_trips"]
    callback_sync = receipt["host_callback_or_sync"]
    host_sync_row = next(
        row
        for row in receipt["trip"]["component_breakdown"]
        if row["component"] == "host_dispatch_or_device_sync"
    )
    runs = receipt["timer_runs"]
    run_trip_walls = [
        run["timings"]["production_trip"]["median_s"] / counts["active_set_trips"]
        for run in runs
    ]
    if len(runs) == 2:
        timing_summary = (
            "Jobs `1262560` and `1262571` independently completed every "
            "device-synchronised timer for 22086/43 pure. Their production medians "
            f"were **{runs[0]['timings']['production_trip']['median_s']:.6f} s** and "
            f"**{runs[1]['timings']['production_trip']['median_s']:.6f} s**, or "
            f"{run_trip_walls[0]:.6f} and {run_trip_walls[1]:.6f} s per active-set "
            "trip. The profile uses the median of the two run medians: "
            f"**{receipt['trip']['wall_s']:.6f} s/trip**."
        )
    else:
        timing_summary = (
            f"Job `{runs[0]['job_id']}` completed every device-synchronised timer "
            "for 22086/43 pure. Its production median was "
            f"**{runs[0]['timings']['production_trip']['median_s']:.6f} s**, or "
            f"**{run_trip_walls[0]:.6f} s per active-set trip**."
        )
    lines = [
        "# H200 active-set trip quantum",
        "",
        timing_summary,
        (
            f"The solve executes **{counts['active_set_trips']} active-set trips**, "
            f"**{counts['residual_evaluations_total']} residual evaluations** "
            f"({counts['residual_evaluations_per_trip']:.3f}/trip: "
            f"{counts['primal_residual_evaluations_total']} primal plus "
            f"{counts['line_search_grades_total']} line-search grades), and "
            f"**{counts['gmres_jacobian_vector_products_total']} Jacobian-vector "
            "products** "
            f"({counts['gmres_jacobian_vector_products_per_trip']:.3f}/trip). The "
            "source receipt's banked count of 24 predates the current step-cap "
            f"history count of {counts['newton_updates_total']} attempted Newton "
            "promotions. The divergence is retained as an observation rather than "
            "used to reject synchronized timings."
        ),
        (
            f"Independent job `{PRIOR_FULL_SOLVE_JOB}` spans "
            f"{prior['minimum_warm_solve_wall_s']:.6f} to "
            f"{prior['maximum_warm_solve_wall_s']:.6f} s over three warm solves, "
            f"or {prior_trip_wall:.6f} s/trip at its median."
        ),
    ]
    if len(runs) == 2:
        lines.extend(
            [
                "",
                "## Repeated device-synchronised timer medians",
                "",
                "| program | job 1262560 [ms] | job 1262571 [ms] | "
                "profile centre [ms] |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, probe in {
            "production_trip": receipt["production_trip_probe"],
            **receipt["direct_probes"],
        }.items():
            run_medians = probe["run_medians_s"]
            lines.append(
                f"| {name.replace('_', ' ')} | "
                f"{1.0e3 * run_medians['1262560']:.6f} | "
                f"{1.0e3 * run_medians['1262571']:.6f} | "
                f"{1.0e3 * probe['steady']['median_s']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Additive timer attribution",
            "",
            "| rank | component | wall / trip [s] | share | "
            "direct wall / evaluation [ms] |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(receipt["trip"]["component_breakdown"], 1):
        lines.append(
            f"| {index} | {row['component'].replace('_', ' ')} | "
            f"{row['wall_s_per_trip']:.6f} | {100.0 * row['share_of_trip']:.2f}% | "
            f"{1.0e3 * row['wall_s_per_evaluation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Topology attribution",
            "",
            "| rank | sub-part | direct probe [ms] | attributed / trip [s] | share |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(receipt["trip"]["topology_breakdown"], 1):
        lines.append(
            f"| {index} | {row['component'].replace('_', ' ')} | "
            f"{1.0e3 * row['direct_probe_wall_s']:.6f} | "
            f"{row['attributed_wall_s_per_trip']:.6f} | "
            f"{100.0 * row['share_of_topology']:.2f}% |"
        )
    lines.extend(["", receipt["trip"]["topology_breakdown_method"] + "."])
    floor = receipt["trip"]["scan_latency_floor"]
    forward_census = receipt["evaluation_census"]["forward_evaluation"]
    forward_branch_scan_sum = forward_census[
        "fixed_scan_trip_compiled_branch_sum_per_evaluation"
    ]
    lines.extend(
        [
            "",
            "## Launch and scan floor",
            "",
            (
                "GPU kernel-launch and transfer counts are **not available**: both "
                "CUPTI replays failed before a complete interval was written. The "
                "partial trace is unpromoted. The branch-minimum static census has "
                f"**{floor['fixed_scan_trip_lower_bound_per_trip']:.1f} fixed scan "
                "iterations/trip**; at 7.6 µs each, their arithmetic floor is "
                f"{floor['arithmetic_floor_s']:.6f} s and measured trip wall is "
                f"**{floor['measured_to_floor_ratio']:.2f}×** that floor. Dynamic "
                "while-loop iterations are not guessed into it."
            ),
            (
                "One forward evaluation contains at least "
                f"**{forward_census['fixed_scan_trip_lower_bound_per_evaluation']} "
                f"fixed scan iterations** ({forward_branch_scan_sum} when all "
                "compiled branches are summed). The reuse-map census assigns "
                f"{TOPOLOGY_READS_PER_OUTER_RESIDUAL} topology reads/residual and "
                f"{SPLINE_FITS_PER_TOPOLOGY_READ} split-spline fits/read."
            ),
            "",
            "| evaluation | GPU launches | transfers | fixed scan lower bound | "
            "compiled-branch scan sum |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in receipt["evaluation_census"].items():
        launches = row["gpu_compute_launches_per_evaluation"] or "not captured"
        transfers = row["gpu_transfers_per_evaluation"] or "not captured"
        lines.append(
            f"| {name.replace('_', ' ')} | {launches} | {transfers} | "
            f"{row['fixed_scan_trip_lower_bound_per_evaluation']} | "
            f"{row['fixed_scan_trip_compiled_branch_sum_per_evaluation']} |"
        )
    svd = receipt["svd_census"]
    replay = receipt["profiler_replay"]
    if replay["failed_jobs"]:
        profiler_summary = (
            f"Both attempts failed in `{replay['failure_call']}` while replaying "
            "`jit_one_trip`; job 1262571 also reported a dropped CUPTI buffer. "
            "The retained Perfetto file is "
            f"{replay['partial_trace']['size_bytes']:,} "
            f"bytes at `{replay['partial_trace']['path']}` and has "
            "`promoted: false`."
        )
    else:
        profiler_summary = (
            "The timer-only route did not request CUPTI replay, so kernel-launch "
            "counts remain unavailable by construction."
        )
    lines.extend(
        [
            "",
            "## Jacobi SVD replay finding",
            "",
            (
                "The `cusolverDnDgesvdj` failure exposes production work, not "
                "profiler setup alone. The dominant authored call site is "
                "`nova.linalg.split_spline._conditioned_fit`: "
                "`jnp.linalg.cond(normal)` runs once for each of the level and field "
                "normal equations. At 63 fits/topology read and four reads/residual, "
                "that is **"
                f"{svd['split_spline_condition_svds_per_residual_evaluation']} "
                "Jacobi SVDs/residual evaluation**. "
                "`nova.equilibrium.fixed_point._projected_krylov_condition` owns one "
                "additional singular-value-only SVD per Newton residual "
                "linearisation."
            ),
            profiler_summary,
            "",
            "## Ranked bottlenecks and implied repairs",
            "",
        ]
    )
    for index, row in enumerate(receipt["ranked_bottlenecks"], 1):
        lines.append(
            f"{index}. **{row['bottleneck']}** — {row['wall_s_per_trip']:.6f} s/trip. "
            f"Repair: {row['repair']}. Owner: `{row['owner_plan']}`."
        )
    lines.extend(
        [
            "",
            "## Host callbacks and synchronization",
            "",
            (
                "The closed production jaxpr contains "
                f"**{callback_sync['closed_jaxpr_callback_primitive_count']} host "
                "callback primitives**. Every timed invocation ends in one "
                "`jax.block_until_ready` device synchronization. Without a complete "
                "device trace, the timer remainder is "
                f"{host_sync_row['wall_s_per_trip']:.6f} s/trip "
                f"({100.0 * host_sync_row['share_of_trip']:.2f}%). It is an upper "
                "bound on host plus synchronization, not a pure-host measurement, "
                "because it retains fused device work absent from the isolated model."
            ),
            "",
            "## Measurement boundaries",
            "",
            "The additive table scales direct synchronized component medians by "
            "exact evaluation counts. Isolated probes do not prove that the fused "
            "program adds identically. The receipt preserves both raw timer "
            "distributions, terminal-log hashes, the unpromoted trace identity, and "
            "the static control-flow census. No `nova/` source was changed.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _build_receipt(
    *,
    reference: dict[str, Any],
    carrier: dict[str, Any],
    policy: dict[str, Any],
    active_set_authority: dict[str, Any],
    counts: dict[str, Any],
    probes: dict[str, Any],
    production_probe: dict[str, Any],
    callback_census: dict[str, int],
    timer_runs: list[dict[str, Any]],
    runtime: dict[str, Any],
    cache_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    prior_full_solve = _prior_full_solve_evidence()
    full_solve_wall = production_probe["steady"]["median_s"]
    trip_wall = full_solve_wall / counts["active_set_trips"]
    components, attribution = _attribution(trip_wall, counts, probes)
    topology_wall = next(
        row["wall_s_per_trip"]
        for row in components
        if row["component"] == "topology_read"
    )
    topology = _topology_breakdown(topology_wall, probes)
    scan_lower = production_probe["control_flow"]["fixed_scan_trip_lower_bound"]
    scan_lower_per_trip = scan_lower / counts["active_set_trips"]
    scan_floor = scan_lower_per_trip * SCAN_ITERATION_SECONDS
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    source_wall = float(source["solve"]["median_warm_wall_s"])
    performance_receipt_count = int(source["solve"]["trips"])
    promotion_count_observation = {
        "banked_performance_receipt_count": performance_receipt_count,
        "banked_field": "solve.trips",
        "banked_revision": source["revision"],
        "observed_attempted_newton_promotions": counts["newton_updates_total"],
        "divergence": counts["newton_updates_total"] - performance_receipt_count,
        "matches": performance_receipt_count == counts["newton_updates_total"],
        "interpretation": (
            "the banked count predates the step-cap merge and is retained as an "
            "observation; it is not a validity gate on synchronized raw timings"
        ),
    }
    failed_profiler_jobs = [
        run["job_id"] for run in timer_runs if run["profiler_replay_failure"]
    ]
    partial_trace = {
        "path": str(PARTIAL_TRACE),
        "exists": PARTIAL_TRACE.exists(),
        "size_bytes": PARTIAL_TRACE.stat().st_size if PARTIAL_TRACE.exists() else None,
        "sha256": _sha256(PARTIAL_TRACE) if PARTIAL_TRACE.exists() else None,
        "promoted": False,
        "reason": (
            "production annotation ended in a CUDA failure and CUPTI reported "
            "dropped activity events"
        ),
    }
    return {
        "schema": "nova.trip_quantum_profile",
        "schema_version": 2,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_base_revision": _revision(),
        "driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
            "mode": "device-synchronised timers with offline structural census",
        },
        "timer_runs": timer_runs,
        "assembly_runtime": runtime,
        "persistent_compilation_cache": cache_receipt,
        "promotion_count_observation": promotion_count_observation,
        "reuse_map_inputs": {
            "topology_reads_per_residual_evaluation": (
                TOPOLOGY_READS_PER_OUTER_RESIDUAL
            ),
            "split_spline_fits_per_topology_read": SPLINE_FITS_PER_TOPOLOGY_READ,
            "scan_iteration_latency_s": SCAN_ITERATION_SECONDS,
            "persistent_compilation_cache_wiring_revision": "e04970ad",
        },
        "production_identity": {
            "reference": reference,
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "solver": {
                "route": "ForwardProfile.solve_branch newton_krylov",
                "newton_steps": parity.NEWTON_STEPS,
                "gmres_iterations": parity.GMRES_ITERATIONS,
                "warmup_sweeps": parity.WARMUP_SWEEPS,
                "active_set_step_budget": ACTIVE_SET_TRIP_BUDGET,
                "tolerance": parity.FIXED_POINT_CRITERION,
            },
        },
        "trip": {
            "wall_s": trip_wall,
            "measured_full_solve_wall_s": full_solve_wall,
            "run_full_solve_medians_s": {
                run["job_id"]: run["timings"]["production_trip"]["median_s"]
                for run in timer_runs
            },
            "prior_full_solve": prior_full_solve,
            "full_solve_reference": {
                "source": str(SOURCE_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(SOURCE_RECEIPT),
                "revision": source["revision"],
                "active_set_authority": active_set_authority,
                "performance_receipt_count": performance_receipt_count,
                "performance_receipt_count_unit": "attempted Newton promotions",
                "performance_receipt_count_is_not_active_set_trips": True,
                "median_warm_wall_s": source_wall,
                "arithmetic_wall_s_per_active_set_trip": source_wall
                / active_set_authority["active_set_iterations"],
            },
            "counts": counts,
            "component_breakdown": components,
            "component_attribution": attribution,
            "topology_breakdown": topology,
            "topology_breakdown_method": (
                "the directly timed candidate census, spline fit, flood fill, wall "
                "reachability, and separatrix-exclusive medians are normalized to "
                "partition the separately timed topology-read wall; these overlapping "
                "isolated probes are relative attribution weights, not additive wall"
            ),
            "scan_latency_floor": {
                **production_probe["control_flow"],
                "fixed_scan_trip_lower_bound_per_trip": scan_lower_per_trip,
                "scan_iteration_s": SCAN_ITERATION_SECONDS,
                "arithmetic_floor_s": scan_floor,
                "measured_to_floor_ratio": trip_wall / max(scan_floor, 1.0e-30),
                "scope": (
                    "minimum statically fixed scan-body count across compiled "
                    "conditional branches, divided by executed active-set trips; "
                    "dynamic while-loop iteration counts excluded"
                ),
            },
        },
        "direct_probes": probes,
        "evaluation_census": _evaluation_census(probes),
        "production_trip_probe": production_probe,
        "svd_census": {
            "split_spline_condition_call_site": (
                "nova/linalg/split_spline.py:_conditioned_fit jnp.linalg.cond(normal)"
            ),
            "condition_svds_per_split_spline_fit": SPLINE_CONDITION_SVDS_PER_FIT,
            "split_spline_condition_svds_per_topology_read": (
                SPLINE_FITS_PER_TOPOLOGY_READ * SPLINE_CONDITION_SVDS_PER_FIT
            ),
            "split_spline_condition_svds_per_residual_evaluation": (
                SPLINE_FITS_PER_TOPOLOGY_READ
                * SPLINE_CONDITION_SVDS_PER_FIT
                * TOPOLOGY_READS_PER_OUTER_RESIDUAL
            ),
            "explicit_solver_svd_call_site": (
                "nova/equilibrium/fixed_point.py:_projected_krylov_condition "
                "jnp.linalg.svd(hessenberg, compute_uv=False)"
            ),
            "explicit_solver_svds_per_newton_residual_linearisation": 1,
        },
        "profiler_replay": {
            "status": (
                "failed_unpromoted" if failed_profiler_jobs else "not_requested"
            ),
            "failure_call": "cusolverDnDgesvdj",
            "failed_jobs": failed_profiler_jobs,
            "kernel_launch_counts_available": False,
            "partial_trace": partial_trace,
            "finding": (
                "timer measurements completed before profiler replay failed inside the "
                "production Jacobi-SVD path"
                if failed_profiler_jobs
                else "profiling was omitted from the timer-only measurement"
            ),
        },
        "ranked_bottlenecks": _ranked_bottlenecks(components, topology),
        "host_callback_or_sync": {
            "closed_jaxpr_callback_primitive_count": sum(callback_census.values()),
            "closed_jaxpr_callback_primitives": callback_census,
            "final_sync": "jax.block_until_ready on every timed result pytree",
            "device_syncs_per_timed_invocation": 1,
            "note": (
                "the production route disables streaming callbacks; the timer "
                "remainder bounds host dispatch and final synchronization but also "
                "retains fused device work absent from the isolated timer model"
            ),
        },
    }


def _write_outputs(receipt: dict[str, Any], output: Path, report: Path) -> None:
    _write_json(output, receipt)
    _write_report(receipt, report)
    print(f"PROFILE_WRITTEN={output}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)


def _write_raw_checkpoint(
    output: Path,
    *,
    active_set_authority: dict[str, Any],
    counts: dict[str, Any] | None,
    probes: dict[str, Any],
    fine_probes: dict[str, Any],
    cache_receipt: dict[str, Any] | None,
    resume_provenance: dict[str, Any] | None = None,
) -> None:
    """Persist synchronized probe evidence before applying receipt checks."""
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    observed = counts["newton_updates_total"] if counts is not None else None
    banked = int(source["solve"]["trips"])
    _write_json(
        output,
        {
            "schema": "nova.trip_quantum_profile.raw_checkpoint",
            "complete": False,
            "captured_at": datetime.now(UTC).isoformat(),
            "arm": os.environ.get("NOVA_PROFILE_ARM"),
            "revision": _revision(),
            "scheduler": _scheduler(),
            "runtime": _runtime(),
            "persistent_compilation_cache": cache_receipt,
            "active_set_authority": active_set_authority,
            "counts": counts,
            "trip_count_observation": (
                _trip_count_observation(counts, active_set_authority)
                if counts is not None
                else None
            ),
            "promotion_count_observation": {
                "banked_performance_receipt_count": banked,
                "observed_attempted_newton_promotions": observed,
                "divergence": observed - banked if observed is not None else None,
                "matches": observed == banked if observed is not None else None,
                "gate_applied": False,
            },
            "direct_probes": probes,
            "fine_direct_probes": fine_probes,
            "raw_probe_names_serialized": sorted({*probes, *fine_probes}),
            "resume_provenance": resume_provenance,
        },
    )
    print(
        f"RAW_CHECKPOINT_WRITTEN={output} probes={len(probes) + len(fine_probes)}",
        flush=True,
    )


def _per_trip_components(
    receipt: dict[str, Any], fine_probes: dict[str, Any]
) -> list[dict[str, Any]]:
    counts = receipt["trip"]["counts"]
    probes = receipt["direct_probes"]
    attributed = {
        row["component"]: row for row in receipt["trip"]["component_breakdown"]
    }
    topology_calls = receipt["trip"]["component_attribution"]["topology_reads_per_trip"]
    residual_calls = counts["residual_evaluations_per_trip"]
    update_calls = counts["newton_updates_per_trip"]
    jvp_calls = counts["gmres_jacobian_vector_products_per_trip"]
    specifications = [
        ("wall_reachability", probes["wall_reachability"], topology_calls),
        ("flood_fills", probes["flood_fills"], topology_calls),
        ("separatrix", probes["separatrix"], topology_calls),
        ("spline_fits", fine_probes["tensor_spline_fit"], topology_calls),
        ("census", probes["candidate_census"], topology_calls),
        (
            "limiter_tangency_along_spline",
            fine_probes["limiter_tangency"],
            topology_calls,
        ),
        (
            "census_values_by_spline",
            probes["spline_fits"],
            topology_calls,
        ),
        ("line_of_sight_rule", fine_probes["line_of_sight"], topology_calls),
        ("forward_evaluation", probes["forward_evaluation"], residual_calls),
        (
            "jacobian_vector_product",
            probes["jacobian_vector_product"],
            jvp_calls,
        ),
        ("line_search", probes["line_search"], update_calls),
        (
            "gmres_orthogonalisation",
            probes["gmres_orthogonalisation"],
            update_calls,
        ),
    ]
    rows = [
        {
            "component": name,
            "direct_wall_s_per_call": probe["steady"]["median_s"],
            "direct_wall_samples_s": probe["steady"]["samples_s"],
            "calls_per_trip": calls,
            "direct_product_s_per_trip": probe["steady"]["median_s"] * calls,
            "persistent_compilation_cache": probe["persistent_compilation_cache"],
            "counting_note": (
                "one aggregate component call per modeled topology read"
                if calls == topology_calls
                else "solve-history count"
            ),
        }
        for name, probe, calls in specifications
    ]
    remainder = attributed["host_dispatch_or_device_sync"]
    rows.append(
        {
            "component": "host_sync_remainder",
            "direct_wall_s_per_call": remainder["wall_s_per_trip"],
            "direct_wall_samples_s": [remainder["wall_s_per_trip"]],
            "calls_per_trip": 1.0,
            "direct_product_s_per_trip": remainder["wall_s_per_trip"],
            "persistent_compilation_cache": {
                "status": "not_applicable",
                "events": {},
            },
            "counting_note": receipt["trip"]["component_attribution"][
                "remainder_scope"
            ],
        }
    )
    return rows


def assemble_existing(output: Path, report: Path) -> dict[str, Any]:
    configure_dtypes()
    timer_runs = [_timer_log(job_id, path) for job_id, path in DEFAULT_TIMER_LOGS]
    case, profile, target_current, carrier, policy = _profile_and_seed()
    active_set_authority = _active_set_authority()
    reference = case["reference"]
    state = jnp.asarray(case["state"])
    requested, production = _production_program(profile, target_current)
    component_programs = _component_programs(profile, state, requested, target_current)
    programs = {"production_trip": (production, (state,)), **component_programs}
    control_flow = {
        name: _control_flow_census(function, arguments)
        for name, (function, arguments) in programs.items()
    }
    all_probes = _central_probes(timer_runs, control_flow)
    production_probe = all_probes.pop("production_trip")
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    counts = _count_summary(
        active_set_authority["active_set_iterations"],
        int(source["solve"]["trips"]),
    )
    receipt = _build_receipt(
        reference=reference,
        carrier=carrier,
        policy=policy,
        active_set_authority=active_set_authority,
        counts=counts,
        probes=all_probes,
        production_probe=production_probe,
        callback_census=_callback_census(production, (state,)),
        timer_runs=timer_runs,
        runtime=_runtime(),
        cache_receipt=source.get("persistent_compilation_cache"),
    )
    _write_outputs(receipt, output, report)
    return receipt


def _validate_raw_production_checkpoint(
    path: Path,
    *,
    active_set_authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the durable production timing before component-only resumption."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_arm = os.environ.get("NOVA_PROFILE_ARM")
    expected_revision = _revision()
    production = payload.get("direct_probes", {}).get("production_trip", {})
    steady = production.get("steady", {})
    runtime = payload.get("runtime", {})
    scheduler = payload.get("scheduler", {})
    authority = payload.get("active_set_authority", {})
    checks = {
        "raw_schema": payload.get("schema")
        == "nova.trip_quantum_profile.raw_checkpoint",
        "incomplete_component_profile": payload.get("complete") is False,
        "arm": payload.get("arm") == expected_arm,
        "revision": payload.get("revision") == expected_revision,
        "production_probe_present": bool(production),
        "three_synchronised_samples": steady.get("sample_count") == 3
        and len(steady.get("samples_s", [])) == 3,
        "h200_runtime": runtime.get("platform") == "gpu"
        and "H200" in str(runtime.get("device_kind")),
        "x64_runtime": runtime.get("jax_enable_x64") is True,
        "scheduler_partition": scheduler.get("partition") == "betelgeuse",
        "scheduler_reservation": scheduler.get("reservation") == "gpu_0003_grpA",
        "scheduler_job_identity": bool(scheduler.get("job_id")),
        "scheduler_tmp": scheduler.get("temporary_directory") == "/tmp",
        "authority_identity": authority.get("identity")
        == active_set_authority["identity"],
        "authority_arm": authority.get("arm") == active_set_authority["arm"],
        "authority_digest": authority.get("sha256") == active_set_authority["sha256"],
        "counts_present": int(payload.get("counts", {}).get("active_set_trips", 0)) > 0,
        "only_production_serialised": payload.get("raw_probe_names_serialized")
        == ["production_trip"],
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(
            f"raw production checkpoint {path} failed validation: {failures}"
        )
    validation = {
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "source_job_id": str(scheduler["job_id"]),
        "resume_job_id": _scheduler()["job_id"],
        "policy": (
            "reuse a complete three-sample production timing from the same arm and "
            "revision when it was measured on one H200 in betelgeuse under the "
            "gpu_0003_grpA reservation; component probes run in the resume job"
        ),
        "checks": checks,
        "valid": True,
    }
    return payload, validation


def resume_components(
    output: Path,
    report: Path,
    cache_root: Path,
    repeats: int,
) -> dict[str, Any]:
    """Reuse a validated production checkpoint and run only component probes."""
    configure_dtypes()
    _require_h200()
    active_set_authority = _active_set_authority()
    prior, validation = _validate_raw_production_checkpoint(
        output, active_set_authority=active_set_authority
    )
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    reference = case["reference"]
    state = jnp.asarray(case["state"])
    requested, production = _production_program(profile, target_current)
    production_probe = prior["direct_probes"]["production_trip"]
    counts = prior["counts"]
    probes: dict[str, Any] = {}
    fine_probes: dict[str, Any] = {}
    checkpoint_probes = {"production_trip": production_probe}
    for name, (function, arguments) in _component_programs(
        profile, state, requested, target_current
    ).items():
        _executable, probes[name] = _compile_probe(name, function, arguments, repeats)
        probes[name]["scheduler"] = _scheduler()
        checkpoint_probes[name] = probes[name]
        _write_raw_checkpoint(
            output,
            active_set_authority=active_set_authority,
            counts=counts,
            probes=checkpoint_probes,
            fine_probes=fine_probes,
            cache_receipt=cache.receipt(),
            resume_provenance=validation,
        )
    from fine_component_profile import _limiter_program

    fine_programs, fine_contract = _limiter_program(profile, state, requested)
    for name, (function, arguments) in fine_programs.items():
        _executable, fine_probes[name] = _compile_probe(
            name, function, arguments, repeats
        )
        fine_probes[name]["scheduler"] = _scheduler()
        _write_raw_checkpoint(
            output,
            active_set_authority=active_set_authority,
            counts=counts,
            probes=checkpoint_probes,
            fine_probes=fine_probes,
            cache_receipt=cache.receipt(),
            resume_provenance=validation,
        )
    source_job = str(prior["scheduler"]["job_id"])
    timer_run = {
        "job_id": source_job,
        "log_path": None,
        "log_sha256": None,
        "scheduler": prior["scheduler"],
        "timings": {"production_trip": production_probe["steady"]},
        "timing_phase_complete": True,
        "profiler_replay_failure": None,
        "cupti_dropped_activity_buffers": False,
        "scope": "production timing only; direct component probes name the resume job",
    }
    receipt = _build_receipt(
        reference=reference,
        carrier=carrier,
        policy=policy,
        active_set_authority=active_set_authority,
        counts=counts,
        probes=probes,
        production_probe=production_probe,
        callback_census=_callback_census(production, (state,)),
        timer_runs=[timer_run],
        runtime=_runtime(),
        cache_receipt=cache.receipt(),
    )
    receipt["complete"] = True
    receipt["arm"] = os.environ.get("NOVA_PROFILE_ARM")
    receipt["revision"] = _revision()
    receipt["fine_direct_probes"] = fine_probes
    receipt["fine_component_contract"] = fine_contract
    receipt["per_trip_components"] = _per_trip_components(receipt, fine_probes)
    receipt["trip_count_observation"] = _trip_count_observation(
        counts, active_set_authority
    )
    receipt["resume_provenance"] = validation
    receipt["raw_serialization"] = {
        "policy": "output rewritten after each synchronized component probe",
        "production_probe_reused": True,
        "production_source_job_id": source_job,
        "component_probe_job_id": _scheduler()["job_id"],
        "trip_count_retained_as_observation": True,
    }
    _write_outputs(receipt, output, report)
    return receipt


def _compile_production_with_result(
    function: Callable,
    arguments: tuple[Any, ...],
) -> tuple[Any, dict[str, Any]]:
    """Compile and execute one warm plus one measured production solve."""
    print("COMPILE_START name=production_trip", flush=True)
    cache_before = _cache_event_snapshot()
    started = time.perf_counter()
    executable = jax.jit(function).lower(*arguments).compile()
    compile_wall = time.perf_counter() - started
    started = time.perf_counter()
    _ready(executable(*arguments))
    first_wall = time.perf_counter() - started
    started = time.perf_counter()
    result = _ready(executable(*arguments))
    sample = time.perf_counter() - started
    print(f"SAMPLE_DONE name=production_trip index=1/1 wall_s={sample:.9f}", flush=True)
    probe = {
        "compile_wall_s": compile_wall,
        "first_execute_wall_s": first_wall,
        "steady": _distribution([sample]),
        "control_flow": _control_flow_census(function, arguments),
        "persistent_compilation_cache": _cache_event_delta(cache_before),
    }
    return result, probe


def run_production_only(
    output: Path,
    cache_root: Path,
    label: str,
    base_revision: str,
    held_commit: str,
) -> dict[str, Any]:
    """Measure the production trip count and terminal state at one held tip."""
    configure_dtypes()
    _require_h200()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    active_set_authority = _active_set_authority()
    state = jnp.asarray(case["state"])
    _requested, production = _production_program(profile, target_current)
    result, probe = _compile_production_with_result(production, (state,))
    history = result.equilibrium.fixed_point
    termination_code = int(np.asarray(history.termination_reason))
    termination = fixed_point.FixedPointTerminationReason(termination_code).name.lower()
    achieved_code = int(np.asarray(result.achieved_class))
    achieved_class = TopologyClass(achieved_code).name.lower()
    counts = _counts(result)
    ancestry = (
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                held_commit,
                "HEAD",
            ],
            check=False,
        ).returncode
        == 0
    )
    if not ancestry:
        raise RuntimeError(f"held commit {held_commit} is not present at {ROOT}")
    payload = {
        "schema": "nova.production_trip_tip_attribution",
        "complete": True,
        "captured_at": datetime.now(UTC).isoformat(),
        "label": label,
        "base_revision": base_revision,
        "held_commit": held_commit,
        "revision": _revision(),
        "reference": case["reference"],
        "carrier": carrier,
        "field_policy": policy,
        "scheduler": _scheduler(),
        "runtime": _runtime(),
        "persistent_compilation_cache": cache.receipt(),
        "production_probe": probe,
        "counts": counts,
        "trip_count_observation": _trip_count_observation(counts, active_set_authority),
        "terminal": {
            "residual": float(np.asarray(result.residual)),
            "termination_code": termination_code,
            "termination_reason": termination,
            "achieved_class_code": achieved_code,
            "achieved_class": achieved_class,
            "requested_class": TopologyClass(
                int(np.asarray(result.requested_class))
            ).name.lower(),
            "topology_consistent": bool(np.asarray(result.topology_consistent)),
            "converged": bool(np.asarray(result.converged)),
        },
        "execution_contract": (
            "one compile execution followed by one synchronized measured production "
            "solve; the terminal receipt is captured from the measured solve"
        ),
    }
    _write_json(output, payload)
    print(f"TIP_ATTRIBUTION_WRITTEN={output}", flush=True)
    return payload


def run(
    output: Path,
    report: Path,
    cache_root: Path,
    repeats: int,
) -> dict[str, Any]:
    configure_dtypes()
    _require_h200()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    active_set_authority = _active_set_authority()
    reference = case["reference"]
    state = jnp.asarray(case["state"])
    requested, production = _production_program(profile, target_current)
    production_executable, production_probe = _compile_probe(
        "production_trip", production, (state,), repeats
    )
    counts = _counts(_ready(production_executable(state)))
    checkpoint_probes = {"production_trip": production_probe}
    fine_probes: dict[str, Any] = {}
    _write_raw_checkpoint(
        output,
        active_set_authority=active_set_authority,
        counts=counts,
        probes=checkpoint_probes,
        fine_probes=fine_probes,
        cache_receipt=cache.receipt(),
    )
    probes = {}
    for name, (function, arguments) in _component_programs(
        profile, state, requested, target_current
    ).items():
        _executable, probes[name] = _compile_probe(name, function, arguments, repeats)
        checkpoint_probes[name] = probes[name]
        _write_raw_checkpoint(
            output,
            active_set_authority=active_set_authority,
            counts=counts,
            probes=checkpoint_probes,
            fine_probes=fine_probes,
            cache_receipt=cache.receipt(),
        )
    from fine_component_profile import _limiter_program

    fine_programs, fine_contract = _limiter_program(profile, state, requested)
    for name, (function, arguments) in fine_programs.items():
        _executable, fine_probes[name] = _compile_probe(
            name, function, arguments, repeats
        )
        _write_raw_checkpoint(
            output,
            active_set_authority=active_set_authority,
            counts=counts,
            probes=checkpoint_probes,
            fine_probes=fine_probes,
            cache_receipt=cache.receipt(),
        )
    job_id = _scheduler()["job_id"] or "direct"
    all_probes = {"production_trip": production_probe, **probes}
    timer_run = {
        "job_id": job_id,
        "log_path": None,
        "log_sha256": None,
        "scheduler": _scheduler(),
        "timings": {name: probe["steady"] for name, probe in all_probes.items()},
        "timing_phase_complete": True,
        "profiler_replay_failure": None,
        "cupti_dropped_activity_buffers": False,
    }
    receipt = _build_receipt(
        reference=reference,
        carrier=carrier,
        policy=policy,
        active_set_authority=active_set_authority,
        counts=counts,
        probes=probes,
        production_probe=production_probe,
        callback_census=_callback_census(production, (state,)),
        timer_runs=[timer_run],
        runtime=_runtime(),
        cache_receipt=cache.receipt(),
    )
    receipt["complete"] = True
    receipt["arm"] = os.environ.get("NOVA_PROFILE_ARM")
    receipt["revision"] = _revision()
    receipt["fine_direct_probes"] = fine_probes
    receipt["fine_component_contract"] = fine_contract
    receipt["per_trip_components"] = _per_trip_components(receipt, fine_probes)
    receipt["trip_count_observation"] = _trip_count_observation(
        counts, active_set_authority
    )
    receipt["raw_serialization"] = {
        "policy": "output rewritten after each synchronized direct probe",
        "gate_applied_to_promotion_count": False,
        "promotion_divergence_retained_as_observation": True,
    }
    _write_outputs(receipt, output, report)
    return receipt


def preflight() -> None:
    configure_dtypes()
    active_set_authority = _active_set_authority()
    case, profile, target_current, _carrier, _policy = _profile_and_seed()
    state = jnp.asarray(case["state"])
    requested, production = _production_program(profile, target_current)
    programs = _component_programs(profile, state, requested, target_current)
    shaped = {
        name: jax.eval_shape(function, *arguments)
        for name, (function, arguments) in programs.items()
    }
    shaped["production_trip"] = jax.eval_shape(production, state)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "reference": case["reference"],
                "active_set_authority": active_set_authority,
                "programs": sorted(shaped),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--resume-components",
        action="store_true",
        help="reuse the validated raw production checkpoint and run components only",
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="measure one warm and one synchronized production solve",
    )
    parser.add_argument("--label", default="production-tip")
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--held-commit", default="")
    parser.add_argument(
        "--assemble-existing",
        action="store_true",
        help="assemble the two completed timer logs without a GPU rerun",
    )
    arguments = parser.parse_args()
    selected_modes = sum(
        (
            arguments.preflight,
            arguments.assemble_existing,
            arguments.resume_components,
            arguments.production_only,
        )
    )
    if selected_modes > 1:
        parser.error("select at most one execution mode")
    if arguments.preflight:
        preflight()
    elif arguments.assemble_existing:
        assemble_existing(arguments.output, arguments.report)
    elif arguments.resume_components:
        resume_components(
            arguments.output,
            arguments.report,
            arguments.cache_root,
            arguments.repeats,
        )
    elif arguments.production_only:
        if not arguments.base_revision or not arguments.held_commit:
            parser.error("--production-only requires --base-revision and --held-commit")
        run_production_only(
            arguments.output,
            arguments.cache_root,
            arguments.label,
            arguments.base_revision,
            arguments.held_commit,
        )
    else:
        run(
            arguments.output,
            arguments.report,
            arguments.cache_root,
            arguments.repeats,
        )


if __name__ == "__main__":
    main()
