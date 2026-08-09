"""Route evidence for the traced fixed-point accelerator module.

This module first answers the structural question: ``nova.jax.fixed_point``
does not have a host twin.  ``ReconstructProfile.picard`` is a traced,
physics-bound method with a different result contract; the SciPy
Newton--Krylov calls drive mutable legacy plasma objects; and the traced
``ForwardFluxOperator`` is driven directly by Optimistix.  None implements the
``FixedPointResult`` API or the three-algorithm family in this module.

The numerical evidence therefore compares the three retained algorithms, not
an invented host/traced pair.  It uses the only integrated map protocol named
by the accelerator module, ``ReconstructProfile.least_squares_map``, at both
grid shapes constructed by the tests (9 x 9 and 17 x 17).  Each physical case
is made self-consistent by a bootstrap-and-remeasure construction.  A SciPy
finite-difference Krylov root solve, starting near that branch, is the
independent fixed-point oracle.  The measured solvers all start from the cold
physical seed and receive the same budget of 20 map evaluations.

Run the two device measurements in separate allocations, then combine them::

    uv run python benchmarks/fixed_point_route.py measure \
      --platform cpu --partial /tmp/fixed_point_cpu.json
    uv run python benchmarks/fixed_point_route.py measure \
      --platform gpu --cases-from /tmp/fixed_point_cpu.json \
      --partial /tmp/fixed_point_gpu.json
    uv run python benchmarks/fixed_point_route.py combine \
      --cpu /tmp/fixed_point_cpu.json --gpu /tmp/fixed_point_gpu.json

The combine command writes ``fixed_point_route.json`` and
``fixed_point_route.svg`` under ``docs/figures/jax-dissolution``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy
import scipy.optimize

from nova.equilibrium import ProfileDegrees, ReconstructProfile
from nova.equilibrium.measurement import Magnetics
from nova.jax.fixed_point import FixedPointResult, anderson, newton_krylov, picard


ROOT = Path(
    os.environ.get("NOVA_FIXED_POINT_BENCH_ROOT", Path(__file__).resolve().parents[1])
)
OUTPUT = ROOT / "docs/figures/jax-dissolution"
EVALUATIONS = 20
RELAXATION = 0.6
REPEATS = 5


@dataclass(frozen=True)
class CaseSpec:
    """One physical reconstruction shape reached by Nova's test callers."""

    name: str
    side: int
    topology_rays: int

    @property
    def dimension(self) -> int:
        """Number of flux unknowns in the flattened state."""
        return self.side * self.side


CASES = (
    CaseSpec("profile_9x9", side=9, topology_rays=32),
    CaseSpec("profile_17x17", side=17, topology_rays=32),
)


def _block(result: FixedPointResult) -> FixedPointResult:
    """Synchronise a result without transferring it away from its device."""
    jax.block_until_ready(result.state)
    return result


def _cpu_model() -> str:
    """Return the host model name when Linux exposes it."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _environment(command: list[str]) -> dict[str, Any]:
    """Capture the execution environment needed to interpret timings."""
    devices = jax.devices()
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
        "benchmark_sha256": os.environ.get("NOVA_FIXED_POINT_BENCH_SHA256")
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _build_machine(spec: CaseSpec) -> ReconstructProfile:
    """Construct the same compact physical machine used by integrated tests."""
    grid_r = np.linspace(0.65, 1.35, spec.side)
    grid_z = np.linspace(-0.5, 0.5, spec.side)
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
        topology_rays=spec.topology_rays,
    )


def _new_case(spec: CaseSpec) -> tuple[ReconstructProfile, dict[str, jax.Array]]:
    """Bootstrap and remeasure one self-consistent reconstruction case."""
    solver = _build_machine(spec)
    source_current = solver.pack_source_currents(
        {"shaping_upper": -1.0e4, "shaping_lower": -1.0e4}
    )
    plasma_current = jnp.asarray(5.0e4)
    initial = solver.initial_flux(source_current, plasma_current)
    seed_cell = jnp.linalg.lstsq(
        solver.plasma_to_grid, initial - solver.source_to_grid @ source_current
    )[0]
    bootstrap_measurement = solver.source_to_sensor @ source_current + (
        solver.plasma_to_sensor @ seed_cell
    )
    scale = jnp.full(bootstrap_measurement.size, 1.0e-3)
    mask = jnp.ones(bootstrap_measurement.size, dtype=bool)
    bootstrap_map = solver.least_squares_map(
        source_current, plasma_current, bootstrap_measurement, scale, mask
    )

    def bootstrap(initial_state):
        return picard(
            bootstrap_map,
            initial_state,
            evaluations=240,
            relaxation=RELAXATION,
        ).state

    branch_seed = jax.jit(bootstrap)(initial)
    jax.block_until_ready(branch_seed)
    basis, _topology = solver._profile_basis(branch_seed)
    coefficients = solver._least_squares_coefficients(
        basis,
        source_current,
        plasma_current,
        bootstrap_measurement,
        scale,
        mask,
    )
    measured = solver.source_to_sensor @ source_current + solver.plasma_to_sensor @ (
        basis @ coefficients
    )
    case = {
        "source_current": source_current,
        "plasma_current": plasma_current,
        "measured": measured,
        "scale": scale,
        "mask": mask,
        "initial": initial,
        "branch_seed": branch_seed,
    }
    jax.block_until_ready(measured)
    return solver, case


def _case_from_transfer(
    spec: CaseSpec, transfer: dict[str, Any]
) -> tuple[ReconstructProfile, dict[str, jax.Array]]:
    """Reconstruct one case from CPU-authored inputs on the active device."""
    solver = _build_machine(spec)
    case = {name: jnp.asarray(value) for name, value in transfer.items()}
    jax.block_until_ready(case["initial"])
    return solver, case


def _transfer(case: dict[str, jax.Array]) -> dict[str, Any]:
    """Serialise the physical inputs shared by CPU and GPU runs."""
    return {name: np.asarray(value).tolist() for name, value in case.items()}


def _map_fn(
    solver: ReconstructProfile, case: dict[str, jax.Array], source_scale=1.0
) -> Callable[[jax.Array], jax.Array]:
    """Return the reconstruction sweep map for one case."""
    return solver.least_squares_map(
        case["source_current"] * source_scale,
        case["plasma_current"],
        case["measured"],
        case["scale"],
        case["mask"],
    )


def _run_algorithm(
    name: str, map_fn: Callable[[jax.Array], jax.Array], initial: jax.Array
) -> FixedPointResult:
    """Run one solver under the common map-evaluation budget."""
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


def _relative_sup(value: np.ndarray, reference: np.ndarray) -> float:
    """Return set-normalised sup error."""
    scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    return float(np.max(np.abs(value - reference)) / scale)


def _fixed_point_residual(mapped: np.ndarray, state: np.ndarray) -> float:
    """Return ``max|g(x)-x| / max|g(x)|`` like the package contract."""
    scale = max(float(np.max(np.abs(mapped))), 1.0e-300)
    return float(np.max(np.abs(mapped - state)) / scale)


def _oracle(
    solver: ReconstructProfile, case: dict[str, jax.Array]
) -> tuple[dict[str, Any], np.ndarray]:
    """Refine the physical branch with SciPy's finite-difference Krylov root."""
    map_fn = _map_fn(solver, case)
    compiled_map = jax.jit(map_fn)
    compiled_map(case["branch_seed"]).block_until_ready()
    evaluations = 0

    def residual(state):
        nonlocal evaluations
        evaluations += 1
        state_array = jnp.asarray(state)
        return np.asarray(compiled_map(state_array)) - np.asarray(state)

    start = time.perf_counter()
    result = scipy.optimize.root(
        residual,
        np.asarray(case["branch_seed"]),
        method="krylov",
        options={"fatol": 1.0e-12, "maxiter": 300},
    )
    elapsed = time.perf_counter() - start
    state = np.asarray(result.x)
    mapped = np.asarray(compiled_map(jnp.asarray(state)))
    relative_residual = _fixed_point_residual(mapped, state)
    oracle = {
        "method": "scipy.optimize.root(method='krylov')",
        "derivative": "host finite differences",
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": evaluations,
        "wall_seconds": elapsed,
        "relative_residual": relative_residual,
    }
    if not np.isfinite(relative_residual) or relative_residual > 1.0e-10:
        raise RuntimeError(f"independent oracle did not converge: {oracle}")
    return oracle, state


def _timed_run(
    name: str,
    solver: ReconstructProfile,
    case: dict[str, jax.Array],
    oracle_state: np.ndarray,
    *,
    batch: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure tracing/compilation, first execution and steady executions."""
    map_fn = _map_fn(solver, case)
    if batch == 1:
        argument = case["initial"]

        def solve(initial):
            return _run_algorithm(name, map_fn, initial)

    else:
        argument = jnp.repeat(case["initial"][None, :], batch, axis=0)

        def solve(initial):
            return jax.vmap(lambda item: _run_algorithm(name, map_fn, item))(initial)

    jax.block_until_ready(argument)
    jax.clear_caches()
    start = time.perf_counter()
    compiled = jax.jit(solve).lower(argument).compile()
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    first = _block(compiled(argument))
    first_seconds = time.perf_counter() - start

    timings = []
    final = first
    for _ in range(REPEATS):
        start = time.perf_counter()
        final = _block(compiled(argument))
        timings.append(time.perf_counter() - start)

    state = np.asarray(final.state)
    expected = (
        oracle_state if batch == 1 else np.repeat(oracle_state[None, :], batch, axis=0)
    )
    row = {
        "algorithm": name,
        "dimension": int(case["initial"].size),
        "batch": batch,
        "map_evaluations": EVALUATIONS,
        "compile_seconds": compile_seconds,
        "first_execution_ms": 1.0e3 * first_seconds,
        "steady_min_ms": 1.0e3 * min(timings),
        "steady_median_ms": 1.0e3 * float(np.median(timings)),
        "residual": float(np.max(np.asarray(final.residual))),
        "oracle_relative_sup_error": _relative_sup(state, expected),
        "dtype": str(final.state.dtype),
        "devices": sorted(str(device) for device in final.state.devices()),
        "finite": bool(np.isfinite(state).all()),
    }
    return row, state


def _differentiation(
    name: str, solver: ReconstructProfile, case: dict[str, jax.Array]
) -> dict[str, Any]:
    """Check a physical source-current JVP against a central difference."""
    initial = case["initial"]

    def state_at(source_scale):
        return _run_algorithm(name, _map_fn(solver, case, source_scale), initial).state

    def value_and_tangent(source_scale):
        return jax.jvp(state_at, (source_scale,), (jnp.ones_like(source_scale),))

    compiled_jvp = jax.jit(value_and_tangent).lower(jnp.asarray(1.0)).compile()
    state, tangent = compiled_jvp(jnp.asarray(1.0))
    jax.block_until_ready(tangent)
    epsilon = 1.0e-4
    compiled_state = jax.jit(state_at).lower(jnp.asarray(1.0)).compile()
    upper = compiled_state(jnp.asarray(1.0 + epsilon))
    lower = compiled_state(jnp.asarray(1.0 - epsilon))
    jax.block_until_ready(upper)
    finite_difference = (np.asarray(upper) - np.asarray(lower)) / (2.0 * epsilon)
    tangent_array = np.asarray(tangent)
    return {
        "algorithm": name,
        "parameter": "multiplicative source-current scale",
        "mode": "jax.jvp",
        "finite_difference_epsilon": epsilon,
        "relative_sup_error": _relative_sup(tangent_array, finite_difference),
        "tangent_finite": bool(np.isfinite(tangent_array).all()),
        "state_finite": bool(np.isfinite(np.asarray(state)).all()),
        "dtype": str(tangent.dtype),
        "devices": sorted(str(device) for device in tangent.devices()),
    }


def _import_census() -> dict[str, Any]:
    """Find every tracked Python import of the accelerator module."""
    imports = []
    for root_name in ("nova", "tests", "benchmarks", "apps"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path == ROOT / "benchmarks/fixed_point_route.py":
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
                            "symbols": [alias.name for alias in node.names],
                        }
                    )
    imports.sort(key=lambda row: (row["path"], row["line"]))
    production = [row for row in imports if row["path"].startswith("nova/")]
    return {
        "imports": imports,
        "production_imports": production,
        "production_import_count": len(production),
        "documented_map_protocol": (
            "nova/equilibrium/profile.py:ReconstructProfile.least_squares_map"
        ),
        "candidate_peer_audit": [
            {
                "path": "nova/equilibrium/profile.py",
                "symbol": "ReconstructProfile.picard",
                "peer": False,
                "reason": (
                    "traced physics-bound scan; prescribed coefficients; "
                    "ProfileResult contract"
                ),
            },
            {
                "path": "nova/equilibrium/forward.py",
                "symbol": "scipy.optimize.newton_krylov",
                "peer": False,
                "reason": (
                    "finite-difference host root solver over a mutable Plasma "
                    "adapter; no shared result API"
                ),
            },
            {
                "path": "nova/equilibrium/inverse.py",
                "symbol": "scipy.optimize.newton_krylov",
                "peer": False,
                "reason": (
                    "legacy mutating inverse solve; only one algorithm and a "
                    "different residual protocol"
                ),
            },
            {
                "path": "nova/equilibrium/forward_operator.py",
                "symbol": "ForwardFluxOperator",
                "peer": False,
                "reason": (
                    "a map consumed directly by Optimistix; it is an "
                    "accelerator input, not another accelerator"
                ),
            },
        ],
    }


def measure(platform_name: str, partial: Path, cases_from: Path | None) -> None:
    """Measure one active JAX backend and write a portable partial result."""
    actual = jax.default_backend()
    if actual != platform_name:
        raise RuntimeError(f"requested {platform_name!r}, active backend is {actual!r}")
    source_payload = json.loads(cases_from.read_text()) if cases_from else None
    runs = []
    differentiability = []
    oracles = []
    transfers = {}
    private_states = {}
    oracle_states = {}

    for spec in CASES:
        if source_payload is None:
            solver, case = _new_case(spec)
            transfers[spec.name] = _transfer(case)
            oracle, oracle_state = _oracle(solver, case)
            oracle["case"] = spec.name
            oracle["dimension"] = spec.dimension
            oracles.append(oracle)
            oracle_states[spec.name] = oracle_state.tolist()
        else:
            solver, case = _case_from_transfer(
                spec, source_payload["_transfer"][spec.name]
            )
            transfers[spec.name] = source_payload["_transfer"][spec.name]
            oracle_state = np.asarray(source_payload["_oracle_states"][spec.name])
            oracle_states[spec.name] = oracle_state.tolist()

        map_fn = _map_fn(solver, case)
        oracle_mapped = np.asarray(jax.jit(map_fn)(jnp.asarray(oracle_state)))
        if _fixed_point_residual(oracle_mapped, oracle_state) > 1.0e-10:
            raise RuntimeError(f"oracle state did not reproduce for {spec.name}")

        for name in ("picard", "anderson", "newton_krylov"):
            row, state = _timed_run(name, solver, case, oracle_state, batch=1)
            row["case"] = spec.name
            runs.append(row)
            private_states[f"{spec.name}:{name}:1"] = state.tolist()

        if spec.side == 17:
            for name in ("picard", "anderson", "newton_krylov"):
                row, state = _timed_run(name, solver, case, oracle_state, batch=4)
                row["case"] = spec.name
                runs.append(row)
                private_states[f"{spec.name}:{name}:4"] = state.tolist()

        if spec.side == 9:
            for name in ("picard", "anderson", "newton_krylov"):
                differentiability.append(_differentiation(name, solver, case))

    payload = {
        "platform": platform_name,
        "environment": _environment(sys.argv),
        "methodology": {
            "case_shapes": [spec.dimension for spec in CASES],
            "batch_shape": 4,
            "map_evaluation_budget": EVALUATIONS,
            "relaxation": RELAXATION,
            "timing_repeats": REPEATS,
            "timing_statistic": "minimum for headline; median also retained",
            "compile_measurement": "trace, lower and compile after jax.clear_caches",
            "oracle": "SciPy finite-difference Krylov root on CPU, residual <= 1e-10",
        },
        "oracles": oracles,
        "runs": runs,
        "differentiation": differentiability,
        "reachability": _import_census(),
        "_transfer": transfers,
        "_oracle_states": oracle_states,
        "_states": private_states,
    }
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if not key.startswith("_")},
            indent=2,
        )
    )
    print("wrote", partial)


def _summary(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    """Compute cross-device cost, convergence and agreement summaries."""
    cpu_rows = {
        (row["case"], row["algorithm"], row["batch"]): row for row in cpu["runs"]
    }
    gpu_rows = {
        (row["case"], row["algorithm"], row["batch"]): row for row in gpu["runs"]
    }
    comparisons = []
    for key in sorted(cpu_rows):
        cpu_row = cpu_rows[key]
        gpu_row = gpu_rows[key]
        cpu_state = np.asarray(cpu["_states"][":".join(map(str, key))])
        gpu_state = np.asarray(gpu["_states"][":".join(map(str, key))])
        comparisons.append(
            {
                "case": key[0],
                "algorithm": key[1],
                "batch": key[2],
                "gpu_speedup": cpu_row["steady_min_ms"] / gpu_row["steady_min_ms"],
                "gpu_compile_over_cpu": gpu_row["compile_seconds"]
                / cpu_row["compile_seconds"],
                "cpu_gpu_relative_sup": _relative_sup(gpu_state, cpu_state),
            }
        )
    single_cpu = [row for row in cpu["runs"] if row["batch"] == 1]
    single_gpu = [row for row in gpu["runs"] if row["batch"] == 1]
    return {
        "cross_device": comparisons,
        "worst_oracle_relative_sup": {
            "cpu": max(row["oracle_relative_sup_error"] for row in single_cpu),
            "gpu": max(row["oracle_relative_sup_error"] for row in single_gpu),
        },
        "worst_cpu_gpu_relative_sup": max(
            row["cpu_gpu_relative_sup"] for row in comparisons
        ),
        "residual_range": {
            platform_name: {
                name: [
                    min(
                        row["residual"]
                        for row in rows
                        if row["algorithm"] == name and row["batch"] == 1
                    ),
                    max(
                        row["residual"]
                        for row in rows
                        if row["algorithm"] == name and row["batch"] == 1
                    ),
                ]
                for name in ("picard", "anderson", "newton_krylov")
            }
            for platform_name, rows in (("cpu", single_cpu), ("gpu", single_gpu))
        },
    }


def _figure(payload: dict[str, Any], target: Path) -> None:
    """Render compile, steady-cost and convergence evidence as SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "picard": "#2a78d6",
        "anderson": "#eb6834",
        "newton_krylov": "#1baf7a",
    }
    markers = {"cpu": "o", "gpu": "s"}
    labels = {
        "picard": "Picard",
        "anderson": "Anderson",
        "newton_krylov": "Newton–Krylov",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for platform_name in ("cpu", "gpu"):
        rows = [
            row
            for row in payload["measurements"][platform_name]["runs"]
            if row["batch"] == 1
        ]
        for algorithm in colors:
            selected = sorted(
                (row for row in rows if row["algorithm"] == algorithm),
                key=lambda row: row["dimension"],
            )
            dimensions = [row["dimension"] for row in selected]
            legend = f"{labels[algorithm]} · {platform_name.upper()}"
            style = {
                "color": colors[algorithm],
                "marker": markers[platform_name],
                "ls": "-" if platform_name == "cpu" else "--",
                "lw": 1.8,
                "ms": 5,
                "label": legend,
            }
            axes[0].plot(
                dimensions, [row["compile_seconds"] for row in selected], **style
            )
            axes[1].plot(
                dimensions, [row["steady_min_ms"] for row in selected], **style
            )
            axes[2].plot(dimensions, [row["residual"] for row in selected], **style)

    titles = ("Trace + compile", "Steady solve", "Residual after 20 map reads")
    ylabels = ("seconds", "milliseconds", "relative residual")
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("flux-state dimension")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#eceae4", lw=0.8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle(
        "Fixed-point accelerators: single traced implementation → relocate intact",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(target, format="svg")
    plt.close(fig)


def combine(cpu_path: Path, gpu_path: Path) -> None:
    """Combine device partials into the committed JSON and SVG evidence."""
    cpu = json.loads(cpu_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    if cpu["platform"] != "cpu" or gpu["platform"] != "gpu":
        raise ValueError("combine expects one CPU partial and one GPU partial")
    if cpu["_transfer"] != gpu["_transfer"]:
        raise ValueError("CPU and GPU cases differ")
    if cpu["reachability"] != gpu["reachability"]:
        raise ValueError("source census changed between device runs")

    clean_cpu = {key: value for key, value in cpu.items() if not key.startswith("_")}
    clean_gpu = {key: value for key, value in gpu.items() if not key.startswith("_")}
    payload = {
        "schema_version": 1,
        "benchmark": "fixed_point_route",
        "verdict": {
            "classification": "RELOCATE",
            "pair_decision": "NOT_A_PAIR",
            "destination": "nova/equilibrium/fixed_point.py",
            "retain": ["picard", "anderson", "newton_krylov"],
            "host_peer": False,
            "reachability": "opt-in integration seam; no production import currently",
            "reason": (
                "one JAX-native fixed-point family serves equilibrium maps; "
                "the apparent host candidates have different maps, mutation "
                "semantics and result APIs"
            ),
        },
        "methodology": cpu["methodology"],
        "reachability": cpu["reachability"],
        "oracles": cpu["oracles"],
        "measurements": {"cpu": clean_cpu, "gpu": clean_gpu},
        "summary": _summary(cpu, gpu),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_target = OUTPUT / "fixed_point_route.json"
    svg_target = OUTPUT / "fixed_point_route.svg"
    json_target.write_text(json.dumps(payload, indent=2) + "\n")
    _figure(payload, svg_target)
    print(json.dumps(payload["summary"], indent=2))
    print("wrote", json_target)
    print("wrote", svg_target)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    measure_parser.add_argument("--partial", type=Path, required=True)
    measure_parser.add_argument("--cases-from", type=Path)
    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("--cpu", type=Path, required=True)
    combine_parser.add_argument("--gpu", type=Path, required=True)
    return parser


def main() -> int:
    """Run a device measurement or combine completed partials."""
    args = _parser().parse_args()
    if args.command == "measure":
        if args.platform == "gpu" and args.cases_from is None:
            raise ValueError("GPU measurement requires --cases-from CPU inputs")
        measure(args.platform, args.partial, args.cases_from)
    else:
        combine(args.cpu, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
