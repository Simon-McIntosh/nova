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
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.equilibrium.connectivity_boundary import _read_ingredients
from nova.equilibrium.flux_surface_connectivity import (
    polish_census_stationary_points,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/profile.json"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/attribution/"
    "trip-quantum-profile.md"
)
DEFAULT_TRACE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/attribution/trip-quantum-trace"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
SOURCE_RECEIPT = ROOT / "docs/figures/polish-support-performance/raw-main.json"
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIPS = 24
LINE_SEARCH_GRADES_PER_UPDATE = 6
SCAN_ITERATION_SECONDS = 7.6e-6
PRODUCTION_FULL_TRIPS = 24


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
    leaves = jax.tree.leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])
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


def _compile_probe(
    name: str,
    function: Callable,
    arguments: tuple[Any, ...],
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    print(f"COMPILE_START name={name}", flush=True)
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
            active_set_steps=ACTIVE_SET_TRIPS,
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


def _counts(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    trips = int(np.asarray(history.active_set_iterations))
    updates = int(np.asarray(history.attempted_newton_promotions))
    accepted = int(np.asarray(history.accepted_newton_promotions))
    backtracks = np.asarray(history.promotion_backtrack_counts, dtype=np.int64)
    used_backtracks = backtracks[:updates]
    recovery = np.asarray(history.promotion_recovery_activations, dtype=np.int64)[
        :updates
    ]
    rebuild = np.asarray(history.promotion_model_rebuild_activations, dtype=np.int64)[
        :updates
    ]
    descent = np.asarray(history.promotion_descent_activations, dtype=np.int64)[
        :updates
    ]
    line_grades = updates * LINE_SEARCH_GRADES_PER_UPDATE
    jvp_count = updates * parity.GMRES_ITERATIONS
    if trips <= 0:
        raise RuntimeError("production solve reported no active-set trips")
    return {
        "active_set_trips": trips,
        "newton_updates_total": updates,
        "newton_updates_per_trip": updates / trips,
        "accepted_newton_updates_total": accepted,
        "accepted_newton_updates_per_trip": accepted / trips,
        "configured_gmres_iterations_per_update": parity.GMRES_ITERATIONS,
        "gmres_jacobian_vector_products_total": jvp_count,
        "gmres_jacobian_vector_products_per_trip": jvp_count / trips,
        "fixed_line_search_grades_per_update": LINE_SEARCH_GRADES_PER_UPDATE,
        "line_search_grades_total": line_grades,
        "line_search_grades_per_trip": line_grades / trips,
        "residual_evaluation_equivalents_total": updates + jvp_count + line_grades,
        "residual_evaluation_equivalents_per_trip": (updates + jvp_count + line_grades)
        / trips,
        "promotion_backtrack_counts": used_backtracks.tolist(),
        "recovery_activations": int(np.count_nonzero(recovery == 1)),
        "model_rebuild_activations": int(np.count_nonzero(rebuild == 1)),
        "descent_activations": int(np.count_nonzero(descent == 1)),
        "counting_contract": (
            "one primal residual for each Newton update, configured GMRES "
            "actions, and all six fixed line-search grades; recovery-model "
            "evaluations are reported separately by their activation counts"
        ),
    }


def _attribution(
    trip_wall: float,
    counts: dict[str, Any],
    probes: dict[str, Any],
    trace_region: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    median = {name: row["steady"]["median_s"] for name, row in probes.items()}
    updates = counts["newton_updates_per_trip"]
    jvp_count = counts["gmres_jacobian_vector_products_per_trip"]
    grades = counts["line_search_grades_per_trip"]
    topology_fraction = min(
        median["topology_read"] / max(median["forward_evaluation"], 1.0e-30),
        0.95,
    )
    raw = {
        "forward_evaluation": updates
        * median["forward_evaluation"]
        * (1.0 - topology_fraction),
        "topology_read": (
            updates * median["forward_evaluation"]
            + jvp_count * median["jacobian_vector_product"]
            + grades * median["forward_evaluation"]
        )
        * topology_fraction,
        "jacobian_vector_product": jvp_count
        * median["jacobian_vector_product"]
        * (1.0 - topology_fraction),
        "gmres_orthogonalisation": updates * median["gmres_orthogonalisation"],
        "line_search": grades
        * median["forward_evaluation"]
        * (1.0 - topology_fraction),
    }
    device_sum = sum(raw.values())
    traced_compute = (
        trace_region["summed_gpu_compute_wall_s"] / counts["active_set_trips"]
    )
    traced_transfer = (
        trace_region["summed_gpu_transfer_wall_s"] / counts["active_set_trips"]
    )
    scale = traced_compute / max(device_sum, 1.0e-30)
    attributed = {name: value * scale for name, value in raw.items()}
    attributed["device_transfer"] = traced_transfer
    attributed["host_dispatch_or_device_sync"] = max(
        trip_wall - traced_compute - traced_transfer, 0.0
    )
    divisor = {
        "forward_evaluation": updates,
        "topology_read": updates + jvp_count + grades,
        "jacobian_vector_product": jvp_count,
        "gmres_orthogonalisation": updates,
        "line_search": grades,
        "device_transfer": 1,
        "host_dispatch_or_device_sync": 1,
    }
    rows = [
        {
            "component": name,
            "wall_s_per_trip": wall,
            "wall_s_per_evaluation": median.get(
                name, wall / max(divisor.get(name, 1), 1)
            ),
            "share_of_trip": wall / trip_wall,
        }
        for name, wall in attributed.items()
    ]
    return sorted(rows, key=lambda row: row["wall_s_per_trip"], reverse=True), {
        "method": (
            "direct synchronized probe medians multiplied by exact trip counts; "
            "overlapping topology work is split from forward/JVP latency, device "
            "semantic device terms are normalized to profiler-observed GPU compute "
            "wall, transfers retain their profiler wall, and the nonnegative timer "
            "remainder is host dispatch or final synchronization"
        ),
        "topology_fraction_of_forward_probe": topology_fraction,
        "raw_device_component_sum_s": device_sum,
        "profiler_gpu_compute_wall_s_per_trip": traced_compute,
        "profiler_gpu_transfer_wall_s_per_trip": traced_transfer,
        "normalization_scale": scale,
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


def _ranked_bottlenecks(
    components: list[dict[str, Any]], topology: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    component = {row["component"]: row for row in components}
    topology_row = topology[0]
    candidates = [
        {
            "bottleneck": "topology read: "
            + topology_row["component"].replace("_", " "),
            "wall_s_per_trip": component["topology_read"]["wall_s_per_trip"]
            * topology_row["share_of_topology"],
            "repair": (
                "replace serial fixed-shape scans with fused or logarithmic "
                "connectivity and reuse one topology partition across residual grades"
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
            "owner_plan": "projected-conditioning-repair",
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
            "owner_plan": "projected-conditioning-repair",
        },
    ]
    return sorted(candidates, key=lambda row: row["wall_s_per_trip"], reverse=True)[:3]


def _write_report(receipt: dict[str, Any], output: Path) -> None:
    counts = receipt["trip"]["counts"]
    lines = [
        "# H200 active-set trip quantum",
        "",
        (
            f"Job `{receipt['scheduler']['job_id']}` measured one warm production "
            f"trip for 22086/43 pure at **{receipt['trip']['wall_s']:.6f} s**. "
            f"It executed {counts['newton_updates_per_trip']:.0f} Newton updates, "
            f"{counts['gmres_jacobian_vector_products_per_trip']:.0f} configured "
            f"GMRES JVPs, and {counts['line_search_grades_per_trip']:.0f} fixed "
            "line-search grades "
            f"({counts['residual_evaluation_equivalents_per_trip']:.0f} "
            "residual-evaluation "
            "equivalents)."
        ),
        "",
        "## Additive trip attribution",
        "",
        "| rank | component | wall / trip [s] | share | "
        "direct wall / evaluation [ms] |",
        "|---:|---|---:|---:|---:|",
    ]
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
    floor = receipt["trip"]["scan_latency_floor"]
    trace = receipt["profiler_trace"]["regions"]["production_trip"]
    compute_launches_per_trip = (
        trace["gpu_compute_launch_count"] / counts["active_set_trips"]
    )
    transfers_per_trip = trace["gpu_transfer_count"] / counts["active_set_trips"]
    lines.extend(
        [
            "",
            "## Launch and scan floor",
            "",
            (
                "The synchronized full-solve profiler interval contains "
                f"**{compute_launches_per_trip:.1f} "
                "GPU compute launches/trip** and "
                f"{transfers_per_trip:.1f} "
                "transfers/trip. The branch-minimum census contains "
                f"**{floor['fixed_scan_trip_lower_bound_per_trip']:.1f} "
                "fixed scan trips/trip as a lower bound**; at 7.6 µs each "
                "their arithmetic floor is "
                f"{floor['arithmetic_floor_s']:.6f} s, so measured trip wall is "
                f"**{floor['measured_to_floor_ratio']:.2f}×** that floor. Dynamic "
                "while-loop trips are listed separately and are not guessed into "
                "the floor."
            ),
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
            "## Measurement boundaries",
            "",
            "The component table is a count-scaled attribution of direct synchronized "
            "probe medians to the exact fused trip wall; it is not a claim that "
            "isolated probes add without fusion. The raw receipt retains the "
            "normalization scale, every probe distribution, the trace digest, cache "
            "identity, and control-flow "
            "census. No `nova/` source was changed.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run(
    output: Path,
    report: Path,
    trace_root: Path,
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
    reference = case["reference"]
    if (
        int(reference["shot"]) != REFERENCE_SHOT
        or int(reference["slice_index"]) != REFERENCE_SLICE
    ):
        raise RuntimeError(f"unexpected production arm {reference}")
    state = jnp.asarray(case["state"])
    requested, production = _production_program(profile, target_current)
    production_executable, production_probe = _compile_probe(
        "production_trip", production, (state,), repeats
    )
    result = _ready(production_executable(state))
    counts = _counts(result)
    if counts["active_set_trips"] != ACTIVE_SET_TRIPS:
        raise RuntimeError(
            f"production program executed {counts['active_set_trips']} trips"
        )
    component_programs = _component_programs(profile, state, requested, target_current)
    executables: dict[str, tuple[Any, tuple[Any, ...]]] = {
        "production_trip": (production_executable, (state,))
    }
    probes = {}
    for name, (function, arguments) in component_programs.items():
        executable, probe = _compile_probe(name, function, arguments, repeats)
        executables[name] = (executable, arguments)
        probes[name] = probe
    trace_path = _profile_trace(trace_root, executables)
    trace = _trace_summary(trace_path)
    full_solve_wall = production_probe["steady"]["median_s"]
    trip_wall = full_solve_wall / counts["active_set_trips"]
    production_trace = trace["regions"]["production_trip"]
    components, attribution = _attribution(trip_wall, counts, probes, production_trace)
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
    full_wall = float(source["solve"]["median_warm_wall_s"])
    receipt = {
        "schema": "nova.trip_quantum_profile",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "scheduler": _scheduler(),
        "runtime": _runtime(),
        "persistent_compilation_cache": cache.receipt(),
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
                "active_set_steps": ACTIVE_SET_TRIPS,
                "tolerance": parity.FIXED_POINT_CRITERION,
            },
        },
        "trip": {
            "wall_s": trip_wall,
            "measured_full_solve_wall_s": full_solve_wall,
            "full_solve_reference": {
                "source": str(SOURCE_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(SOURCE_RECEIPT),
                "revision": source["revision"],
                "active_set_trips": source["solve"]["trips"],
                "median_warm_wall_s": full_wall,
                "arithmetic_wall_s_per_trip": full_wall / PRODUCTION_FULL_TRIPS,
            },
            "counts": counts,
            "component_breakdown": components,
            "component_attribution": attribution,
            "topology_breakdown": topology,
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
        "production_trip_probe": production_probe,
        "profiler_trace": trace,
        "ranked_bottlenecks": _ranked_bottlenecks(components, topology),
        "host_callback_or_sync": {
            "closed_jaxpr_callback_primitive_count": 0,
            "final_sync": "jax.block_until_ready on each timed and traced result",
            "note": (
                "the production route disables streaming callbacks; profiler GPU "
                "compute and transfer wall are subtracted from the synchronized "
                "timer to bound host dispatch and final synchronization"
            ),
        },
    }
    _write_json(output, receipt)
    _write_report(receipt, report)
    print(f"PROFILE_WRITTEN={output}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    return receipt


def preflight() -> None:
    configure_dtypes()
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
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
    else:
        run(
            arguments.output,
            arguments.report,
            arguments.trace_root,
            arguments.cache_root,
            arguments.repeats,
        )


if __name__ == "__main__":
    main()
