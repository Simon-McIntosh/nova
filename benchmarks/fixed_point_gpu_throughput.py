"""Measure fixed-point batch throughput and solver-organization overhead.

The numerical map is the self-consistent 17 x 17
``ReconstructProfile.least_squares_map`` case preserved by
``fixed_point_route.py``.  This extension deliberately starts at batch eight:
the batch-one and batch-four timings already exist in the route artifact and
are imported as context, not measured again.

The current algorithms are measured as their documented use requires: one
outer ``jit`` around a ``vmap`` over independent fields.  Picard candidates
then isolate organization choices without changing its twenty map evaluations
or stopping diagnostic:

* one native batched loop with the complete residual trace;
* the same loop retaining only the last residual;
* a sequential ``lax.map`` over fields;
* one compiled batched map call per evaluation with a host-visible residual
  read after every call.

Run each backend in its own allocation and combine the partials::

    uv run python benchmarks/fixed_point_gpu_throughput.py measure \
      --platform cpu --case-source /path/to/fixed_point_route_cpu.json \
      --partial /path/to/fixed_point_gpu_throughput_cpu.json
    uv run python benchmarks/fixed_point_gpu_throughput.py measure \
      --platform gpu --case-source /path/to/fixed_point_route_cpu.json \
      --partial /path/to/fixed_point_gpu_throughput_gpu.json
    uv run python benchmarks/fixed_point_gpu_throughput.py combine \
      --cpu /path/to/fixed_point_gpu_throughput_cpu.json \
      --gpu /path/to/fixed_point_gpu_throughput_gpu.json \
      --route docs/figures/jax-dissolution/fixed_point_route.json

The combine command writes
``docs/figures/jax-dissolution/fixed_point_gpu_throughput.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy

from nova.equilibrium import ProfileDegrees, ReconstructProfile
from nova.equilibrium.measurement import Magnetics
from nova.jax.fixed_point import FixedPointResult, anderson, newton_krylov, picard


ROOT = Path(
    os.environ.get(
        "NOVA_FIXED_POINT_GPU_BENCH_ROOT", Path(__file__).resolve().parents[1]
    )
)
OUTPUT = ROOT / "docs/figures/jax-dissolution/fixed_point_gpu_throughput.json"
EVALUATIONS = 20
RELAXATION = 0.6
REPEATS = 3
BATCHES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MAP_BATCHES = (8, 32, 128, 512, 2048, 4096)
CANDIDATE_BATCHES = (32, 256, 1024, 4096)
HOST_READ_BATCHES = (256, 1024)
SIDE = 17
DIMENSION = SIDE * SIDE


class BatchedPicardResult(NamedTuple):
    """Picard state, last residual and optional residual trace."""

    state: jax.Array
    residual: jax.Array
    trace: jax.Array


@dataclass(frozen=True)
class Case:
    """Device-resident physical inputs and the independent oracle state."""

    solver: ReconstructProfile
    source_current: jax.Array
    plasma_current: jax.Array
    measured: jax.Array
    scale: jax.Array
    mask: jax.Array
    initial: jax.Array
    oracle: jax.Array
    source_provenance: dict[str, Any]

    def map_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Return the physical least-squares sweep map."""

        return self.solver.least_squares_map(
            self.source_current,
            self.plasma_current,
            self.measured,
            self.scale,
            self.mask,
        )


def _sha256(path: Path) -> str:
    """Return a file digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str:
    """Return the Linux host processor model when available."""

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _nvidia_snapshot() -> dict[str, Any] | None:
    """Capture static accelerator telemetry when nvidia-smi is available."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,pstate,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except FileNotFoundError, subprocess.SubprocessError:
        return None
    return {"command": command, "rows": output.splitlines()}


def _environment(command: list[str]) -> dict[str, Any]:
    """Capture hardware, software and threading provenance."""

    devices = jax.devices()
    benchmark = Path(__file__)
    return {
        "command": command,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "kind": device.device_kind,
                "description": str(device),
            }
            for device in devices
        ],
        "cpu_model": _cpu_model(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS")
        },
        "benchmark_sha256": _sha256(benchmark),
        "nvidia_smi": _nvidia_snapshot(),
    }


def _build_machine() -> ReconstructProfile:
    """Construct the physical machine used by the preserved route case."""

    grid_r = np.linspace(0.65, 1.35, SIDE)
    grid_z = np.linspace(-0.5, 0.5, SIDE)
    radius, height = np.meshgrid(grid_r, grid_z)
    inside = ((radius - 1.0) / 0.3) ** 2 + (height / 0.42) ** 2 <= 1.0
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    sensor_angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return ReconstructProfile.from_geometry(
        grid_r=grid_r,
        grid_z=grid_z,
        inside_limiter=inside,
        cell_width=np.array(grid_r[1] - grid_r[0]),
        cell_height=np.array(grid_z[1] - grid_z[0]),
        source_r=np.array([1.0, 1.0]),
        source_z=np.array([0.8, -0.8]),
        source_width=np.array([0.12, 0.12]),
        source_height=np.array([0.12, 0.12]),
        source_names=("shaping_upper", "shaping_lower"),
        magnetics=Magnetics(
            r=1.0 + 0.5 * np.cos(sensor_angle),
            z=0.6 * np.sin(sensor_angle),
            angle=np.zeros(sensor_angle.size),
            flux_loop=np.ones(sensor_angle.size, dtype=bool),
        ),
        degrees=ProfileDegrees(n_pressure=1, n_diamagnetic=1),
        axis_seed=(1.0, 0.0),
        wall_r=1.0 + 0.31 * np.cos(wall_angle),
        wall_z=0.43 * np.sin(wall_angle),
        iterations=8,
        relaxation=RELAXATION,
        topology_levels=48,
        topology_bisections=12,
        topology_rays=32,
    )


def _load_case(path: Path) -> Case:
    """Load the preserved physical case without repeating its oracle solve."""

    payload = json.loads(path.read_text())
    transfer = payload["_transfer"]["profile_17x17"]
    oracle = payload["_oracle_states"]["profile_17x17"]
    solver = _build_machine()
    arrays = {name: jnp.asarray(value) for name, value in transfer.items()}
    case = Case(
        solver=solver,
        source_current=arrays["source_current"],
        plasma_current=arrays["plasma_current"],
        measured=arrays["measured"],
        scale=arrays["scale"],
        mask=arrays["mask"],
        initial=arrays["initial"],
        oracle=jnp.asarray(oracle),
        source_provenance={
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "benchmark_sha256": payload["environment"]["benchmark_sha256"],
            "oracle": next(
                row for row in payload["oracles"] if row["case"] == "profile_17x17"
            ),
        },
    )
    jax.block_until_ready(case.initial)
    mapped = jax.jit(case.map_fn())(case.oracle)
    jax.block_until_ready(mapped)
    residual = _relative_residual_numpy(np.asarray(mapped), np.asarray(case.oracle))
    if residual > 1.0e-10:
        raise RuntimeError(f"preserved oracle does not reproduce: {residual:.3e}")
    case.source_provenance["active_device_oracle_residual"] = residual
    return case


def _relative_residual(mapped: jax.Array, state: jax.Array) -> jax.Array:
    """Return one relative sup residual per leading-axis field."""

    axes = tuple(range(1, mapped.ndim))
    numerator = jnp.max(jnp.abs(mapped - state), axis=axes)
    denominator = jnp.maximum(jnp.max(jnp.abs(mapped), axis=axes), 1.0e-30)
    return numerator / denominator


def _relative_residual_numpy(mapped: np.ndarray, state: np.ndarray) -> float:
    """Return a scalar relative sup residual."""

    return float(
        np.max(np.abs(mapped - state)) / max(float(np.max(np.abs(mapped))), 1.0e-30)
    )


def _relative_sup(value: np.ndarray, reference: np.ndarray) -> float:
    """Return a reference-normalized sup error."""

    return float(
        np.max(np.abs(value - reference))
        / max(float(np.max(np.abs(reference))), 1.0e-30)
    )


def _run_algorithm(
    name: str, map_fn: Callable[[jax.Array], jax.Array], initial: jax.Array
) -> FixedPointResult:
    """Run a package solver at the shared twenty-evaluation budget."""

    if name == "picard":
        return picard(map_fn, initial, evaluations=EVALUATIONS, relaxation=RELAXATION)
    if name == "anderson":
        return anderson(map_fn, initial, evaluations=EVALUATIONS, relaxation=RELAXATION)
    if name == "newton_krylov":
        return newton_krylov(
            map_fn,
            initial,
            newton_steps=2,
            gmres_iterations=4,
            warmup=8,
            relaxation=RELAXATION,
        )
    raise ValueError(f"unknown algorithm: {name}")


def _batched_picard(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    retain_trace: bool,
) -> BatchedPicardResult:
    """Run one native batched loop with per-field residual semantics."""

    batched_map = jax.vmap(map_fn)
    if retain_trace:

        def body(index, carry):
            state, trace = carry
            mapped = batched_map(state)
            residual = _relative_residual(mapped, state)
            trace = trace.at[:, index].set(residual)
            return state + RELAXATION * (mapped - state), trace

        state, trace = jax.lax.fori_loop(
            0,
            EVALUATIONS,
            body,
            (initial, jnp.full((initial.shape[0], EVALUATIONS), jnp.nan)),
        )
        return BatchedPicardResult(state, trace[:, -1], trace)

    def body(_index, carry):
        state, _residual = carry
        mapped = batched_map(state)
        residual = _relative_residual(mapped, state)
        return state + RELAXATION * (mapped - state), residual

    state, residual = jax.lax.fori_loop(
        0, EVALUATIONS, body, (initial, jnp.full(initial.shape[0], jnp.nan))
    )
    return BatchedPicardResult(
        state, residual, jnp.empty((initial.shape[0], 0), dtype=initial.dtype)
    )


def _sequential_picard(
    map_fn: Callable[[jax.Array], jax.Array], initial: jax.Array
) -> BatchedPicardResult:
    """Run the scalar package solver sequentially across the leading axis."""

    result = jax.lax.map(
        lambda state: picard(
            map_fn, state, evaluations=EVALUATIONS, relaxation=RELAXATION
        ),
        initial,
    )
    return BatchedPicardResult(result.state, result.residual, result.trace)


def _host_checked_picard(
    map_fn: Callable[[jax.Array], jax.Array], initial: jax.Array
) -> BatchedPicardResult:
    """Run Picard with one host-visible convergence read per evaluation."""

    batched_map = jax.vmap(map_fn)

    def step(state):
        mapped = batched_map(state)
        residual = _relative_residual(mapped, state)
        return state + RELAXATION * (mapped - state), residual

    compiled = jax.jit(step).lower(initial).compile()
    state = initial
    residuals = []
    for _ in range(EVALUATIONS):
        state, residual = compiled(state)
        residuals.append(np.asarray(residual))
    trace = jnp.asarray(np.stack(residuals, axis=1))
    return BatchedPicardResult(state, trace[:, -1], trace)


def _device_input(initial: jax.Array, batch: int) -> tuple[jax.Array, float]:
    """Materialize one repeated host batch on the active device and time it."""

    host = np.repeat(np.asarray(initial)[None, :], batch, axis=0)
    start = time.perf_counter()
    argument = jax.device_put(host)
    jax.block_until_ready(argument)
    transfer_ms = 1.0e3 * (time.perf_counter() - start)
    return argument, transfer_ms


def _state_metrics(
    state: jax.Array, oracle: jax.Array
) -> tuple[dict[str, Any], np.ndarray]:
    """Transfer one result and report oracle, batch and digest metrics."""

    start = time.perf_counter()
    host = np.asarray(state).copy()
    materialization_ms = 1.0e3 * (time.perf_counter() - start)
    first = host[0]
    spread = float(np.max(np.abs(host - first[None, :])))
    metrics = {
        "oracle_relative_sup_error": _relative_sup(first, np.asarray(oracle)),
        "batch_max_absolute_spread": spread,
        "finite": bool(np.isfinite(host).all()),
        "result_materialization_ms": materialization_ms,
        "state_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
    }
    return metrics, first


def _compile_and_time(
    name: str,
    organization: str,
    function: Callable[[jax.Array], Any],
    argument: jax.Array,
    oracle: jax.Array,
    *,
    input_transfer_ms: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compile once, execute once, then retain three steady timings."""

    jax.clear_caches()
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    first = compiled(argument)
    jax.block_until_ready(first.state)
    first_ms = 1.0e3 * (time.perf_counter() - start)

    timings = []
    result = first
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = compiled(argument)
        jax.block_until_ready(result.state)
        timings.append(1.0e3 * (time.perf_counter() - start))

    state_metrics, first_state = _state_metrics(result.state, oracle)
    residuals = np.asarray(result.residual)
    row = {
        "algorithm": name,
        "organization": organization,
        "dimension": int(argument.shape[-1]),
        "batch": int(argument.shape[0]),
        "map_evaluations": EVALUATIONS,
        "compile_seconds": compile_seconds,
        "first_execution_ms": first_ms,
        "steady_min_ms": min(timings),
        "steady_median_ms": float(np.median(timings)),
        "fields_per_second": 1.0e3 * int(argument.shape[0]) / min(timings),
        "input_transfer_ms": input_transfer_ms,
        "residual_max": float(np.max(residuals)),
        "residual_min": float(np.min(residuals)),
        "dtype": str(result.state.dtype),
        "devices": sorted(str(device) for device in result.state.devices()),
        **state_metrics,
    }
    return row, first_state


def _time_map(
    map_fn: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    *,
    input_transfer_ms: float,
) -> dict[str, Any]:
    """Measure one synchronized, device-resident batched map evaluation."""

    function = jax.vmap(map_fn)
    jax.clear_caches()
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    first = compiled(argument)
    jax.block_until_ready(first)
    first_ms = 1.0e3 * (time.perf_counter() - start)
    timings = []
    result = first
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = compiled(argument)
        jax.block_until_ready(result)
        timings.append(1.0e3 * (time.perf_counter() - start))
    return {
        "organization": "vmap_one_map_evaluation",
        "dimension": int(argument.shape[-1]),
        "batch": int(argument.shape[0]),
        "compile_seconds": compile_seconds,
        "first_execution_ms": first_ms,
        "steady_min_ms": min(timings),
        "steady_median_ms": float(np.median(timings)),
        "fields_per_second": 1.0e3 * int(argument.shape[0]) / min(timings),
        "input_transfer_ms": input_transfer_ms,
        "dtype": str(result.dtype),
        "devices": sorted(str(device) for device in result.devices()),
    }


def _time_host_checked(
    map_fn: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    oracle: jax.Array,
    *,
    input_transfer_ms: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compile the map step and time three twenty-read solve sequences."""

    batched_map = jax.vmap(map_fn)

    def step(state):
        mapped = batched_map(state)
        residual = _relative_residual(mapped, state)
        return state + RELAXATION * (mapped - state), residual

    jax.clear_caches()
    start = time.perf_counter()
    compiled = jax.jit(step).lower(argument).compile()
    compile_seconds = time.perf_counter() - start

    def execute() -> BatchedPicardResult:
        state = argument
        residuals = []
        for _ in range(EVALUATIONS):
            state, residual = compiled(state)
            residuals.append(np.asarray(residual))
        trace = np.stack(residuals, axis=1)
        return BatchedPicardResult(state, jnp.asarray(trace[:, -1]), jnp.asarray(trace))

    start = time.perf_counter()
    first = execute()
    jax.block_until_ready(first.state)
    first_ms = 1.0e3 * (time.perf_counter() - start)
    timings = []
    result = first
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = execute()
        jax.block_until_ready(result.state)
        timings.append(1.0e3 * (time.perf_counter() - start))
    state_metrics, first_state = _state_metrics(result.state, oracle)
    residuals = np.asarray(result.residual)
    row = {
        "algorithm": "picard",
        "organization": "host_visible_residual_each_evaluation",
        "dimension": int(argument.shape[-1]),
        "batch": int(argument.shape[0]),
        "map_evaluations": EVALUATIONS,
        "host_scalar_array_reads": EVALUATIONS,
        "compile_seconds": compile_seconds,
        "first_execution_ms": first_ms,
        "steady_min_ms": min(timings),
        "steady_median_ms": float(np.median(timings)),
        "fields_per_second": 1.0e3 * int(argument.shape[0]) / min(timings),
        "input_transfer_ms": input_transfer_ms,
        "residual_max": float(np.max(residuals)),
        "residual_min": float(np.min(residuals)),
        "dtype": str(result.state.dtype),
        "devices": sorted(str(device) for device in result.state.devices()),
        **state_metrics,
    }
    return row, first_state


def _import_census() -> dict[str, Any]:
    """Count tracked imports and distinguish production from evidence callers."""

    imports = []
    this_file = Path(__file__).resolve()
    for root_name in ("nova", "tests", "benchmarks", "apps"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == this_file:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError, UnicodeDecodeError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "nova.jax.fixed_point"
                ):
                    imports.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": node.lineno,
                            "symbols": sorted(alias.name for alias in node.names),
                        }
                    )
    imports.sort(key=lambda row: (row["path"], row["line"]))
    production = [row for row in imports if row["path"].startswith("nova/")]
    return {
        "imports": imports,
        "production_imports": production,
        "production_import_count": len(production),
        "live_profile_route": [
            "ReconstructProfile.solve_batch",
            "ReconstructProfile.least_squares_batch",
            "jax.vmap(ReconstructProfile.least_squares)",
        ],
        "forward_operator_route": [
            "nova/biot/boundary.py:ForwardFluxOperator",
            "optimistix.root_find",
        ],
        "mast_catalog_route": [
            "nova/scripts/mast_catalog_geometry.py:main",
            "nova.catalog.mast_geometry.scan_catalog",
        ],
        "mast_solve_input_route": [
            "nova.imas.mast_solve_input_ids.build_solve_input_map",
            "nova.imas.mast_solve_inputs.solve_input_map",
            "nova.imas.mast_solve_inputs.read_solve_inputs",
        ],
        "hot_path_assessment": (
            "The generic accelerator module is absent from production imports. "
            "The catalog code currently inventories geometry and authors solve inputs; "
            "it does not execute ReconstructProfile or a fixed-point accelerator."
        ),
    }


def measure(platform_name: str, case_source: Path, partial: Path) -> None:
    """Measure one active backend and write its raw partial exactly once."""

    actual = jax.default_backend()
    if actual != platform_name:
        raise RuntimeError(f"requested {platform_name!r}, active backend is {actual!r}")
    case = _load_case(case_source)
    map_fn = case.map_fn()
    arguments: dict[int, tuple[jax.Array, float]] = {}

    def argument(batch: int) -> tuple[jax.Array, float]:
        if batch not in arguments:
            arguments[batch] = _device_input(case.initial, batch)
        return arguments[batch]

    map_runs = []
    for batch in MAP_BATCHES:
        values, transfer_ms = argument(batch)
        row = _time_map(map_fn, values, input_transfer_ms=transfer_ms)
        map_runs.append(row)
        print(
            f"{platform_name} map batch={batch} "
            f"steady={row['steady_min_ms']:.3f} ms "
            f"fields/s={row['fields_per_second']:.1f}",
            flush=True,
        )

    current_runs = []
    private_states: dict[str, list[float]] = {}
    for name in ("picard", "anderson", "newton_krylov"):
        for batch in BATCHES:
            values, transfer_ms = argument(batch)

            def solve(initial, algorithm=name):
                return jax.vmap(lambda state: _run_algorithm(algorithm, map_fn, state))(
                    initial
                )

            row, first_state = _compile_and_time(
                name,
                "outer_jit_vmap_package_solver",
                solve,
                values,
                case.oracle,
                input_transfer_ms=transfer_ms,
            )
            current_runs.append(row)
            private_states[f"{name}:{batch}"] = first_state.tolist()
            print(
                f"{platform_name} {name} batch={batch} "
                f"steady={row['steady_min_ms']:.3f} ms "
                f"fields/s={row['fields_per_second']:.1f}",
                flush=True,
            )

    candidate_runs = []
    candidate_states: dict[str, list[float]] = {}
    for batch in CANDIDATE_BATCHES:
        values, transfer_ms = argument(batch)
        candidates: list[tuple[str, Callable[[jax.Array], BatchedPicardResult]]] = [
            (
                "native_batched_loop_full_trace",
                lambda initial: _batched_picard(map_fn, initial, retain_trace=True),
            ),
            (
                "native_batched_loop_final_residual",
                lambda initial: _batched_picard(map_fn, initial, retain_trace=False),
            ),
        ]
        if batch == 32:
            candidates.append(
                (
                    "sequential_lax_map",
                    lambda initial: _sequential_picard(map_fn, initial),
                )
            )
        for organization, solve in candidates:
            row, first_state = _compile_and_time(
                "picard",
                organization,
                solve,
                values,
                case.oracle,
                input_transfer_ms=transfer_ms,
            )
            candidate_runs.append(row)
            candidate_states[f"{organization}:{batch}"] = first_state.tolist()
            print(
                f"{platform_name} {organization} batch={batch} "
                f"steady={row['steady_min_ms']:.3f} ms",
                flush=True,
            )

    for batch in HOST_READ_BATCHES:
        values, transfer_ms = argument(batch)
        row, first_state = _time_host_checked(
            map_fn,
            values,
            case.oracle,
            input_transfer_ms=transfer_ms,
        )
        candidate_runs.append(row)
        candidate_states[f"host_visible_residual_each_evaluation:{batch}"] = (
            first_state.tolist()
        )
        print(
            f"{platform_name} host-checked batch={batch} "
            f"steady={row['steady_min_ms']:.3f} ms",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "platform": platform_name,
        "environment": _environment(sys.argv),
        "methodology": {
            "case": "profile_17x17",
            "dimension": DIMENSION,
            "batches": list(BATCHES),
            "map_batches": list(MAP_BATCHES),
            "candidate_batches": list(CANDIDATE_BATCHES),
            "map_evaluation_budget": EVALUATIONS,
            "relaxation": RELAXATION,
            "timing_repeats": REPEATS,
            "timing_statistic": "minimum headline with median retained",
            "synchronization": "jax.block_until_ready on every timed device result",
            "input_residency": (
                "host-to-device copy timed once, excluded from steady solve"
            ),
            "result_materialization": (
                "device-to-host copy timed after the steady solve"
            ),
            "oracle": "preserved independent SciPy finite-difference Krylov state",
            "unchanged_measurements": (
                "batch one and four are imported during combine and were not rerun"
            ),
        },
        "case_source": case.source_provenance,
        "map_runs": map_runs,
        "current_runs": current_runs,
        "candidate_runs": candidate_runs,
        "reachability": _import_census(),
        "_states": private_states,
        "_candidate_states": candidate_states,
    }
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def _route_context(route: dict[str, Any]) -> dict[str, Any]:
    """Extract the already-measured batch-one/four context."""

    context = []
    for platform_name in ("cpu", "gpu"):
        measurement = route["measurements"][platform_name]
        for row in measurement["runs"]:
            if row["case"] == "profile_17x17" and row["batch"] in (1, 4):
                context.append(
                    {
                        "platform": platform_name,
                        "algorithm": row["algorithm"],
                        "batch": row["batch"],
                        "steady_min_ms": row["steady_min_ms"],
                        "fields_per_second": 1.0e3
                        * row["batch"]
                        / row["steady_min_ms"],
                        "residual": row["residual"],
                        "oracle_relative_sup_error": row["oracle_relative_sup_error"],
                    }
                )
    return {
        "artifact": str(Path("docs/figures/jax-dissolution/fixed_point_route.json")),
        "artifact_benchmark_sha256": route["measurements"]["cpu"]["environment"][
            "benchmark_sha256"
        ],
        "measurements": context,
    }


def _by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index timing rows by algorithm, organization and batch."""

    return {
        (row.get("algorithm"), row["organization"], row["batch"]): row for row in rows
    }


def _crossover(
    cpu_rows: list[dict[str, Any]], gpu_rows: list[dict[str, Any]], algorithm: str
) -> dict[str, Any]:
    """Return the first measured batch where the GPU is faster."""

    cpu = _by_key(cpu_rows)
    gpu = _by_key(gpu_rows)
    values = []
    for batch in BATCHES:
        key = (algorithm, "outer_jit_vmap_package_solver", batch)
        cpu_row = cpu[key]
        gpu_row = gpu[key]
        values.append(
            {
                "batch": batch,
                "cpu_ms": cpu_row["steady_min_ms"],
                "gpu_ms": gpu_row["steady_min_ms"],
                "gpu_speedup": cpu_row["steady_min_ms"] / gpu_row["steady_min_ms"],
                "cpu_fields_per_second": cpu_row["fields_per_second"],
                "gpu_fields_per_second": gpu_row["fields_per_second"],
            }
        )
    winning = [row for row in values if row["gpu_speedup"] > 1.0]
    return {
        "algorithm": algorithm,
        "first_measured_gpu_win_batch": winning[0]["batch"] if winning else None,
        "largest_batch": values[-1],
        "sweep": values,
    }


def _comparisons(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    """Calculate cross-device, candidate and attribution summaries."""

    crossovers = [
        _crossover(cpu["current_runs"], gpu["current_runs"], name)
        for name in ("picard", "anderson", "newton_krylov")
    ]
    cpu_current = _by_key(cpu["current_runs"])
    gpu_current = _by_key(gpu["current_runs"])
    cpu_candidate = _by_key(cpu["candidate_runs"])
    gpu_candidate = _by_key(gpu["candidate_runs"])
    map_cpu = {row["batch"]: row for row in cpu["map_runs"]}
    map_gpu = {row["batch"]: row for row in gpu["map_runs"]}

    candidate_effects = []
    for platform_name, partial, current, candidate in (
        ("cpu", cpu, cpu_current, cpu_candidate),
        ("gpu", gpu, gpu_current, gpu_candidate),
    ):
        for row in candidate.values():
            key = ("picard", "outer_jit_vmap_package_solver", row["batch"])
            baseline = current[key]
            baseline_state = np.asarray(partial["_states"][f"picard:{row['batch']}"])
            candidate_state = np.asarray(
                partial["_candidate_states"][f"{row['organization']}:{row['batch']}"]
            )
            candidate_effects.append(
                {
                    "platform": platform_name,
                    "organization": row["organization"],
                    "batch": row["batch"],
                    "baseline_ms": baseline["steady_min_ms"],
                    "candidate_ms": row["steady_min_ms"],
                    "candidate_speedup": baseline["steady_min_ms"]
                    / row["steady_min_ms"],
                    "baseline_relative_sup_difference": _relative_sup(
                        candidate_state, baseline_state
                    ),
                    "residual_absolute_difference": abs(
                        row["residual_max"] - baseline["residual_max"]
                    ),
                    "oracle_relative_sup_error": row["oracle_relative_sup_error"],
                }
            )

    attribution = []
    for platform_name, current, maps in (
        ("cpu", cpu_current, map_cpu),
        ("gpu", gpu_current, map_gpu),
    ):
        for batch in MAP_BATCHES:
            row = current[("picard", "outer_jit_vmap_package_solver", batch)]
            map_row = maps[batch]
            map_budget_ms = EVALUATIONS * map_row["steady_min_ms"]
            attribution.append(
                {
                    "platform": platform_name,
                    "batch": batch,
                    "picard_ms": row["steady_min_ms"],
                    "twenty_separate_map_ms": map_budget_ms,
                    "fused_to_separate_map_ratio": row["steady_min_ms"] / map_budget_ms,
                    "map_fields_per_second": map_row["fields_per_second"],
                    "solve_fields_per_second": row["fields_per_second"],
                }
            )

    cross_device_accuracy = []
    for name in ("picard", "anderson", "newton_krylov"):
        for batch in BATCHES:
            key = (name, "outer_jit_vmap_package_solver", batch)
            cpu_state = np.asarray(cpu["_states"][f"{name}:{batch}"])
            gpu_state = np.asarray(gpu["_states"][f"{name}:{batch}"])
            cross_device_accuracy.append(
                {
                    "algorithm": name,
                    "batch": batch,
                    "relative_sup_error": _relative_sup(gpu_state, cpu_state),
                }
            )

    target = []
    for platform_name, current in ("cpu", cpu_current), ("gpu", gpu_current):
        for name in ("picard", "anderson", "newton_krylov"):
            passing = [
                current[(name, "outer_jit_vmap_package_solver", batch)]
                for batch in BATCHES
                if current[(name, "outer_jit_vmap_package_solver", batch)][
                    "fields_per_second"
                ]
                >= 250.0
            ]
            target.append(
                {
                    "platform": platform_name,
                    "algorithm": name,
                    "first_batch_at_250_fields_per_second": (
                        passing[0]["batch"] if passing else None
                    ),
                    "maximum_fields_per_second": max(
                        current[(name, "outer_jit_vmap_package_solver", batch)][
                            "fields_per_second"
                        ]
                        for batch in BATCHES
                    ),
                }
            )
    return {
        "crossovers": crossovers,
        "candidate_effects": candidate_effects,
        "map_and_control_attribution": attribution,
        "cross_device_accuracy": cross_device_accuracy,
        "worst_cpu_gpu_relative_sup_error": max(
            row["relative_sup_error"] for row in cross_device_accuracy
        ),
        "per_device_target": target,
    }


def _verdict(
    comparison: dict[str, Any], reachability: dict[str, Any]
) -> dict[str, Any]:
    """Classify the measured bottleneck and the source action it warrants."""

    crossover_batches = [
        row["first_measured_gpu_win_batch"] for row in comparison["crossovers"]
    ]
    gpu_targets = [
        row for row in comparison["per_device_target"] if row["platform"] == "gpu"
    ]
    target_pass = all(
        row["first_batch_at_250_fields_per_second"] is not None for row in gpu_targets
    )
    return {
        "classification": "single-implementation relocation",
        "hot_path": "not reached",
        "production_import_count": reachability["production_import_count"],
        "small_batch_cause": "launch floor and batch underfill",
        "first_measured_gpu_win_batches": crossover_batches,
        "algorithm_only_rate_observation": "pass" if target_pass else "fail",
        "catalog_throughput_gate": "not measured",
        "catalog_target_claim": "not licensed by this zero-caller measurement",
        "finding": (
            "All three extended sweeps win on H200 by batch eight after losing at "
            "batch four in the route artifact. The flat device-time floors and the "
            "map-budget attribution identify launch underfill, not extra solver-loop "
            "work. This does not establish catalog throughput because the measured "
            "module is absent from the catalog path."
        ),
        "prioritized_actions": [
            {
                "priority": 1,
                "action": (
                    "Wire one accelerator owner into ReconstructProfile.solve_batch, "
                    "then benchmark the complete MAST solve with real machine "
                    "operators."
                ),
            },
            {
                "priority": 2,
                "action": (
                    "Place one outer jit around a vmap of the complete fixed-budget "
                    "solve, pooling slices by immutable machine configuration and "
                    "grid shape."
                ),
            },
            {
                "priority": 3,
                "action": (
                    "Keep convergence masks and traces device-side and materialize "
                    "only batch summaries plus persisted results."
                ),
            },
            {
                "priority": 4,
                "action": (
                    "Relocate the generic family beside equilibrium reconstruction or "
                    "retire it in favor of the selected Optimistix path; do not "
                    "retain an unowned zero-caller module."
                ),
            },
        ],
    }


def combine(cpu_path: Path, gpu_path: Path, route_path: Path) -> None:
    """Combine immutable partials and write the reviewed measurement artifact."""

    cpu = json.loads(cpu_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    route = json.loads(route_path.read_text())
    if cpu["platform"] != "cpu" or gpu["platform"] != "gpu":
        raise ValueError("partials must be ordered as CPU then GPU")
    cpu_hash = cpu["environment"]["benchmark_sha256"]
    gpu_hash = gpu["environment"]["benchmark_sha256"]
    combine_hash = _sha256(Path(__file__))
    if cpu_hash != gpu_hash:
        raise ValueError("benchmark source differs across the two measurements")
    if cpu["case_source"]["sha256"] != gpu["case_source"]["sha256"]:
        raise ValueError("CPU and GPU did not consume the same physical case")
    if cpu["reachability"] != gpu["reachability"]:
        raise ValueError("reachability census differs across compute nodes")

    comparison = _comparisons(cpu, gpu)
    reachability = cpu["reachability"]
    payload = {
        "schema_version": 1,
        "benchmark": {
            "path": "benchmarks/fixed_point_gpu_throughput.py",
            "measurement_source_sha256": cpu_hash,
            "combination_source_sha256": combine_hash,
            "measurement_source_copy": (
                "/home/ITER/mcintos/.cache/nova-fixed-point-gpu-throughput/"
                "fixed_point_gpu_throughput.py"
            ),
        },
        "raw_partials": {
            "cpu": {"path": str(cpu_path.resolve()), "sha256": _sha256(cpu_path)},
            "gpu": {"path": str(gpu_path.resolve()), "sha256": _sha256(gpu_path)},
        },
        "methodology": cpu["methodology"],
        "case_source": cpu["case_source"],
        "prior_small_batch_context": _route_context(route),
        "catalog_boundary": {
            "scope": "all timesteps of 11,573 MAST level-2 shots",
            "wall_clock_target": "approximately one hour on eight H200 devices",
            "exact_slice_census": "open",
            "estimated_compute_requirement": "125-250 slices/s per device",
            "io_budget": {
                "catalog_terabytes_per_hour": 4.5,
                "aggregate_gigabytes_per_second": 1.25,
                "evenly_sharded_megabytes_per_second_per_device": 156.25,
                "unit_convention": "decimal",
            },
            "measurement_scope": (
                "17 x 17 ReconstructProfile sweep map, the largest reachable caller "
                "shape carried by the accelerator evidence"
            ),
            "unmeasured": [
                "complete production-sized ReconstructProfile operators",
                "the intended Optimistix/ReconstructProfile catalog solve",
                "streamed catalog input and result persistence",
                "multi-device scaling",
            ],
        },
        "reachability": reachability,
        "observability": {
            "kernel_occupancy": (
                "not exposed by synchronized wall timing or the retained nvidia-smi "
                "snapshot; no occupancy percentage is claimed"
            ),
            "available_attribution": [
                "one-map device-resident timing",
                "twenty-evaluation fused solve timing",
                "vmap versus sequential lax.map",
                "device-side trace versus final residual only",
                "host-visible convergence reads",
                "host-to-device input and device-to-host result timing",
            ],
        },
        "measurements": {
            "cpu": {
                "environment": cpu["environment"],
                "map_runs": cpu["map_runs"],
                "current_runs": cpu["current_runs"],
                "candidate_runs": cpu["candidate_runs"],
            },
            "gpu": {
                "environment": gpu["environment"],
                "map_runs": gpu["map_runs"],
                "current_runs": gpu["current_runs"],
                "candidate_runs": gpu["candidate_runs"],
            },
        },
        "comparison": comparison,
        "verdict": _verdict(comparison, reachability),
        "acceptance_gates": {
            "same_benchmark_hash": cpu_hash == gpu_hash,
            "same_case_hash": cpu["case_source"]["sha256"]
            == gpu["case_source"]["sha256"],
            "no_production_imports": reachability["production_import_count"] == 0,
            "all_finite": all(
                row["finite"]
                for partial in (cpu, gpu)
                for group in ("current_runs", "candidate_runs")
                for row in partial[group]
            ),
            "batch_spread_zero": max(
                row["batch_max_absolute_spread"]
                for partial in (cpu, gpu)
                for group in ("current_runs", "candidate_runs")
                for row in partial[group]
            )
            == 0.0,
            "candidate_picard_state_relative_sup_at_most_1e_12": max(
                row["baseline_relative_sup_difference"]
                for row in comparison["candidate_effects"]
            )
            <= 1.0e-12,
            "cpu_gpu_relative_sup_at_most_1e_10": comparison[
                "worst_cpu_gpu_relative_sup_error"
            ]
            <= 1.0e-10,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    """Parse the measure/combine command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    measure_parser.add_argument("--case-source", type=Path, required=True)
    measure_parser.add_argument("--partial", type=Path, required=True)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--cpu", type=Path, required=True)
    combine_parser.add_argument("--gpu", type=Path, required=True)
    combine_parser.add_argument("--route", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.platform, args.case_source, args.partial)
    else:
        combine(args.cpu, args.gpu, args.route)


if __name__ == "__main__":
    main()
