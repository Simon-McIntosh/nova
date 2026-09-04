"""Compare integrated polish support with its parent on one H200.

The benchmark has three deliberately separate operations. ``materialize`` exports
the parent revision without changing the active worktree, ``capture`` runs one
revision in a fresh Python process, and ``combine`` writes the paired receipt
and figure only after both captures have completed.  Fresh processes prevent
JAX's in-memory compilation state from leaking between the two revisions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
from time import perf_counter
from typing import Any

import numpy as np


BASELINE_REVISION = "e2aaf78853a70b51b339981e6b6f65c97fa84614"
CANDIDATE_REVISION = "HEAD"
REFERENCE_SHOT = 22086
MEASURED_SOLVE_LAUNCHES = 5
MEASURED_BATCH_LAUNCHES = 5
MEASURED_COMPONENT_LAUNCHES = 5
DEFAULT_OPERANDS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "bank-regeneration-raw-20260902/current-operands.npz"
)
DEFAULT_CACHE = Path(
    "/work/projects/imas_gpu/sophelio/jax-cache/polish-support-performance"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/polish-support-performance/shared-spline/main-vs-shared.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/polish-support-performance/shared-spline/main-vs-shared.png"
)
FIGURE_IDENTITY = (
    "/nova/figures/polish-support-performance/shared-spline/main-vs-shared.png"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    """Map non-finite evidence to explicit nulls before strict serialization."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def materialize(repository: Path, output: Path, revision: str) -> None:
    """Export one revision as a branchless source tree below the worktree."""
    if output.exists():
        raise FileExistsError(f"refusing to replace existing source export: {output}")
    output.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", revision],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(output, filter="data")
    resolved = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (output / ".measurement-revision").write_text(resolved + "\n", encoding="utf-8")


def remove_materialized_source(output: Path, revision: str) -> None:
    """Remove only the guarded branchless source export made by this driver."""
    resolved = output.resolve()
    expected_parent = (
        Path.cwd() / "docs/figures/polish-support-performance/shared-spline"
    ).resolve()
    marker = resolved / ".measurement-revision"
    if resolved.name != ".baseline-source" or resolved.parent != expected_parent:
        raise ValueError(
            "refusing to remove a path outside the benchmark artifact root"
        )
    resolved_revision = subprocess.run(
        ["git", "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if marker.read_text(encoding="utf-8").strip() != resolved_revision:
        raise ValueError("refusing to remove a source export with another revision")
    shutil.rmtree(resolved)


def _activate_source(source_root: Path) -> None:
    source = str(source_root.resolve())
    sys.path.insert(0, source)
    os.chdir(source)


def _compile_counted(function: Callable, argument: Any):
    import jax
    from jax._src import compiler

    calls = 0
    original = compiler.compile_or_get_cached

    def counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    compiler.compile_or_get_cached = counted
    try:
        started = perf_counter()
        executable = jax.jit(function).lower(argument).compile()
        compile_wall = perf_counter() - started
    finally:
        compiler.compile_or_get_cached = original
    return executable, calls, compile_wall


@contextmanager
def _compile_counter():
    from jax._src import compiler

    state = {"calls": 0}
    original = compiler.compile_or_get_cached

    def counted(*args: Any, **kwargs: Any):
        state["calls"] += 1
        return original(*args, **kwargs)

    compiler.compile_or_get_cached = counted
    try:
        yield state
    finally:
        compiler.compile_or_get_cached = original


def _block(value: Any) -> Any:
    import jax

    return jax.block_until_ready(value)


def _timed(executable: Callable, argument: Any) -> tuple[Any, float]:
    started = perf_counter()
    result = executable(argument)
    _block(result)
    return result, perf_counter() - started


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum_s": float(np.min(array)),
        "maximum_s": float(np.max(array)),
        "range_s": float(np.ptp(array)),
        "standard_deviation_s": float(np.std(array)),
        "median_absolute_deviation_s": float(
            np.median(np.abs(array - np.median(array)))
        ),
    }


def _cache_watch() -> tuple[
    dict[str, float | int], Callable[[], dict[str, float | int]]
]:
    import jax.monitoring as monitoring

    events: dict[str, float | int] = {"hits": 0, "saved_seconds": 0.0}

    def hit(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_hits":
            events["hits"] = int(events["hits"]) + 1

    def saved(event: str, duration_secs: float, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            events["saved_seconds"] = float(events["saved_seconds"]) + duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_duration_secs_listener(saved)

    def snapshot() -> dict[str, float | int]:
        return dict(events)

    return events, snapshot


def _cache_delta(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, float | int]:
    return {
        "hits": int(after["hits"]) - int(before["hits"]),
        "saved_seconds": float(after["saved_seconds"]) - float(before["saved_seconds"]),
    }


def _gpu_identity() -> dict[str, Any]:
    import jax

    query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    device = jax.devices()[0]
    return {
        "jax_device": str(device),
        "jax_platform": device.platform,
        "visible_devices": query,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _solve_measurement(setup=None) -> dict[str, Any]:
    import jax.numpy as jnp

    from benchmarks import efit_forward_parity_slice as parity
    from benchmarks.receipt_raster_check import _profile_and_seed
    from nova.equilibrium.topology import TopologyClass

    if setup is None:
        setup = _profile_and_seed()
    case, profile, target_current, carrier_receipt, policy = setup
    if int(case["reference"]["shot"]) != REFERENCE_SHOT:
        raise RuntimeError("the frozen-six solve arm changed identity")
    seed = jnp.asarray(case["state"])

    def solve(initial_flux):
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
        )

    executable, compile_calls, compile_wall = _compile_counted(solve, seed)
    first_result, first_wall = _timed(executable, seed)
    warm_walls: list[float] = []
    result = first_result
    for _ in range(MEASURED_SOLVE_LAUNCHES):
        result, wall = _timed(executable, seed)
        warm_walls.append(wall)
    equilibrium = result.equilibrium
    trace = np.asarray(equilibrium.fixed_point.trace, dtype=np.float64)
    trips = int(np.count_nonzero(np.isfinite(trace)))
    residual = float(equilibrium.fixed_point.residual)
    physical = equilibrium.flux[: profile.operator.physical_node_number]
    _masks, topology = profile.operator.read(equilibrium.flux)
    class_read = profile.operator._connectivity_read(physical, topology, classify=True)
    achieved_class = TopologyClass(int(result.achieved_class))
    return {
        "reference": case["reference"],
        "carrier": carrier_receipt,
        "field_policy": policy,
        "solver": {
            "route": "ForwardProfile.solve_branch newton_krylov",
            "requested_class": "diverted",
            "target_current_a": target_current,
            "newton_promotions": parity.NEWTON_STEPS,
            "gmres_iterations": parity.GMRES_ITERATIONS,
            "registered_tolerance": parity.FIXED_POINT_CRITERION,
        },
        "compile_count": compile_calls,
        "compile_wall_s": compile_wall,
        "first_call_wall_s": first_wall,
        "warm_launch_wall_s": warm_walls,
        "median_warm_wall_s": float(statistics.median(warm_walls)),
        "warm_launch_spread": _spread(warm_walls),
        "trips": trips,
        "terminal_residual": residual,
        "converged": bool(residual <= parity.FIXED_POINT_CRITERION),
        "achieved_class": int(achieved_class),
        "achieved_class_name": achieved_class.name.lower(),
        "topology_consistent": bool(result.topology_consistent),
        "terminal_class_operands": {
            "class_margin": float(class_read["class_margin"]),
            "normalized_wall_flux": float(class_read["u_wall"]),
            "normalized_saddle_flux": float(class_read["u_xpoint"]),
            "axis_flux": float(class_read["psi_axis"]),
            "wall_candidate_valid": bool(
                class_read["limiter_flux_from_global_surface"]
            ),
            "wall_candidate_shadowed": bool(class_read["wall_shadowed"]),
            "wall_candidate_rz": [
                float(class_read["limiter_r"]),
                float(class_read["limiter_z"]),
            ],
            "saddle_rz": np.asarray(topology.x_point, dtype=np.float64).tolist(),
        },
    }


def _component_measurement(setup) -> dict[str, Any]:
    """Profile one production Newton trip and its topology components."""
    import inspect

    import jax.numpy as jnp

    from benchmarks import efit_forward_parity_slice as parity
    from nova.equilibrium.flux_surface_connectivity import (
        fit_tensor_spline,
        polish_census_stationary_points,
    )
    from nova.equilibrium.topology import TopologyClass

    case, profile, target_current, _carrier_receipt, _policy = setup
    seed = jnp.asarray(case["state"])
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    operator = profile.operator
    physical_number = operator.physical_node_number
    connectivity_radius, connectivity_height, connectivity_shape = (
        operator.connectivity_grid_axes()
    )

    def tensor_fit(state):
        physical = state[:physical_number]
        grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
        radial_count, vertical_count = connectivity_shape
        field = grid_flux.reshape((radial_count, vertical_count)).T
        return fit_tensor_spline(
            connectivity_radius, connectivity_height, field
        ).coefficients

    def topology_qualification(state):
        physical = state[:physical_number]
        return operator._fixed_design_topology.read_qualification(
            physical,
            operator.polarity,
            operator.inside_material,
            requested,
        )

    def fixed_design_read(state):
        return operator._fixed_design_read(state[:physical_number], requested)

    def published_domain_read(state):
        return operator.read(state, requested)

    def shadow_read(state):
        return operator.residual_shadow_components(state, requested)

    residual_map = profile.flux_map(
        requested_class=requested,
        target_current=target_current,
    )

    def one_newton_trip(state):
        return profile.solve_branch(
            state,
            requested,
            target_current=target_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=0,
            newton_steps=1,
            gmres_iterations=parity.GMRES_ITERATIONS,
            relaxation=parity.RELAXATION,
            step_cap=parity.STEP_CAP,
        )

    def measure(function: Callable) -> dict[str, Any]:
        executable, compile_calls, compile_wall = _compile_counted(function, seed)
        _first, first_wall = _timed(executable, seed)
        warm_walls = []
        for _ in range(MEASURED_COMPONENT_LAUNCHES):
            _result, wall = _timed(executable, seed)
            warm_walls.append(wall)
        return {
            "compile_count": compile_calls,
            "compile_wall_s": compile_wall,
            "first_call_wall_s": first_wall,
            "warm_wall_s": warm_walls,
            "median_warm_wall_s": float(statistics.median(warm_walls)),
            "warm_spread": _spread(warm_walls),
        }

    grid_flux, _wall_flux = operator.topology.split_flux_map(seed[:physical_number])
    candidate_capacity = int(
        operator._fixed_design_topology.grid(grid_flux)[0].shape[0]
    )
    accepts_shared_surface = (
        "surface" in inspect.signature(polish_census_stationary_points).parameters
    )
    fits_per_topology_read = 1 if accepts_shared_surface else 2 * candidate_capacity + 3
    topology_reads_per_fixed_design_read = 2
    fixed_design_reads_per_residual_evaluation = 2
    fits_per_residual_evaluation = (
        fits_per_topology_read
        * topology_reads_per_fixed_design_read
        * fixed_design_reads_per_residual_evaluation
    )
    return {
        "reference": case["reference"],
        "fit_census": {
            "candidate_capacity": candidate_capacity,
            "fits_per_candidate_component": 2,
            "fits_after_candidate_census": 3,
            "fits_per_topology_read": fits_per_topology_read,
            "topology_reads_per_fixed_design_read": (
                topology_reads_per_fixed_design_read
            ),
            "fixed_design_reads_per_residual_evaluation": (
                fixed_design_reads_per_residual_evaluation
            ),
            "tensor_spline_fits_per_residual_evaluation": (
                fits_per_residual_evaluation
            ),
            "shared_surface_api": accepts_shared_surface,
        },
        "components": {
            "tensor_spline_fit": measure(tensor_fit),
            "topology_qualification": measure(topology_qualification),
            "fixed_design_read": measure(fixed_design_read),
            "published_domain_read": measure(published_domain_read),
            "residual_shadow_read": measure(shadow_read),
            "residual_evaluation": measure(residual_map),
            "one_newton_trip": measure(one_newton_trip),
        },
    }


def _inside_polygon(
    radius: np.ndarray, height: np.ndarray, wall: np.ndarray
) -> np.ndarray:
    mesh_r, mesh_z = np.meshgrid(radius, height)
    x = mesh_r.ravel()
    y = mesh_z.ravel()
    inside = np.zeros(x.shape, dtype=bool)
    x0 = wall[:, 0]
    y0 = wall[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for first_x, first_y, second_x, second_y in zip(x0, y0, x1, y1, strict=True):
        crosses = (first_y > y) != (second_y > y)
        denominator = second_y - first_y
        intersection = first_x + (y - first_y) * (second_x - first_x) / np.where(
            denominator == 0.0, 1.0, denominator
        )
        inside ^= crosses & (x < intersection)
    return inside.reshape(mesh_r.shape)


def _bank_operands(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(np.asarray(stored["metadata"]).item()))
        valid = [
            index
            for index in range(int(metadata["arm_count"]))
            if np.asarray(stored[f"arm_{index:02d}_flux"]).shape == (33, 33)
        ]
        if len(valid) != 11:
            raise RuntimeError(
                f"expected eleven banked 33 by 33 maps, found {len(valid)}"
            )
        first = valid[0]
        radius = np.asarray(stored[f"arm_{first:02d}_radius"], dtype=np.float64)
        height = np.asarray(stored[f"arm_{first:02d}_height"], dtype=np.float64)
        wall = np.asarray(stored[f"arm_{first:02d}_wall"], dtype=np.float64)
        arrays = {
            "flux": np.stack(
                [
                    np.asarray(stored[f"arm_{i:02d}_flux"], dtype=np.float64)
                    for i in valid
                ]
            ),
            "axis": np.stack(
                [
                    np.asarray(stored[f"arm_{i:02d}_axis"], dtype=np.float64)
                    for i in valid
                ]
            ),
            "saddle": np.stack(
                [
                    np.asarray(stored[f"arm_{i:02d}_selected_saddle"], dtype=np.float64)
                    for i in valid
                ]
            ),
            "interface": np.asarray(
                [float(np.asarray(stored[f"arm_{i:02d}_binding_flux"])) for i in valid],
                dtype=np.float64,
            ),
            "radius": radius,
            "height": height,
            "wall": wall,
            "inside": _inside_polygon(radius, height, wall),
        }
        for index in valid[1:]:
            if not np.array_equal(radius, stored[f"arm_{index:02d}_radius"]):
                raise RuntimeError("banked radius axes differ")
            if not np.array_equal(height, stored[f"arm_{index:02d}_height"]):
                raise RuntimeError("banked height axes differ")
            if not np.array_equal(wall, stored[f"arm_{index:02d}_wall"]):
                raise RuntimeError("banked walls differ")
    return {
        "source": str(path),
        "sha256": _sha256(path),
        "valid_arm_indices": valid,
        "arm_count": len(valid),
    }, arrays


def _tree_numpy(value: Any) -> dict[str, np.ndarray]:
    import jax

    host = jax.device_get(value)
    return {key: np.asarray(item) for key, item in host.items()}


def _agreement(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    all_equal = True
    for key in left:
        first = np.asarray(left[key])
        second = np.asarray(right[key])
        nonfinite_equal = np.array_equal(np.isfinite(first), np.isfinite(second))
        finite = np.isfinite(first) & np.isfinite(second)
        maximum = (
            float(np.max(np.abs(first[finite] - second[finite])))
            if np.any(finite)
            else 0.0
        )
        equal = bool(nonfinite_equal and maximum <= 2.0e-12)
        fields[key] = {
            "maximum_absolute_difference": maximum,
            "nonfinite_pattern_equal": nonfinite_equal,
            "passes_2e_12": equal,
        }
        all_equal &= equal
    return {"fields": fields, "all_fields_pass": all_equal}


def _topology_measurement(operand_path: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.connectivity_boundary import traced_boundary_read
    from nova.equilibrium.flux_surface_connectivity import (
        fit_tensor_spline,
        polish_census_stationary_points,
    )

    operand_receipt, arrays = _bank_operands(operand_path)
    radius = jnp.asarray(arrays["radius"])
    height = jnp.asarray(arrays["height"])
    wall = jnp.asarray(arrays["wall"])
    inside = jnp.asarray(arrays["inside"])
    fields = jnp.asarray(arrays["flux"])
    axes = jnp.asarray(arrays["axis"])
    saddles = jnp.asarray(arrays["saddle"])
    interfaces = jnp.asarray(arrays["interface"])

    def read_one(field, axis, saddle, interface):
        surface = fit_tensor_spline(radius, height, field)
        selected_axis = jnp.r_[axis, surface(axis[0], axis[1])]
        selected_saddle = jnp.r_[saddle, surface(saddle[0], saddle[1])]
        admitted_axis, admitted_saddle, _receipt = polish_census_stationary_points(
            field,
            radius,
            height,
            interface,
            jnp.asarray(1.0, dtype=field.dtype),
            selected_axis,
            selected_saddle,
            surface=surface,
        )
        wall_flux = surface(wall[:, 0], wall[:, 1])
        read = traced_boundary_read(
            field,
            radius,
            height,
            inside,
            admitted_axis[0],
            admitted_axis[1],
            96,
            18,
            2,
            jnp.empty((0,), dtype=field.dtype),
            jnp.asarray(1.0, dtype=field.dtype),
            wall[:, 0],
            wall[:, 1],
            wall_flux,
            surface=surface,
        )
        return {
            "axis_rz": admitted_axis[:2],
            "saddle_rz": admitted_saddle[:2],
            "binding_point_rz": jnp.stack((read["limiter_r"], read["limiter_z"])),
            "class_margin": read["class_margin"],
        }

    batch_arguments = (fields, axes, saddles, interfaces)
    eager_rows = [
        read_one(field, axis, saddle, interface)
        for field, axis, saddle, interface in zip(*batch_arguments, strict=True)
    ]
    eager = jax.tree.map(lambda *items: jnp.stack(items), *eager_rows)
    _block(eager)

    single = jax.jit(read_one)
    single_results = []
    single_walls: list[float] = []
    with _compile_counter() as single_counter:
        for arguments in zip(*batch_arguments, strict=True):
            started = perf_counter()
            result = single(*arguments)
            _block(result)
            single_walls.append(perf_counter() - started)
            single_results.append(result)
    compiled = jax.tree.map(lambda *items: jnp.stack(items), *single_results)

    batched_function = jax.jit(jax.vmap(read_one))
    with _compile_counter() as batch_counter:
        started = perf_counter()
        batched = batched_function(*batch_arguments)
        _block(batched)
        batch_first_wall = perf_counter() - started
    batch_walls: list[float] = []
    for _ in range(MEASURED_BATCH_LAUNCHES):
        started = perf_counter()
        batched = batched_function(*batch_arguments)
        _block(batched)
        batch_walls.append(perf_counter() - started)

    eager_host = _tree_numpy(eager)
    compiled_host = _tree_numpy(compiled)
    batched_host = _tree_numpy(batched)
    return {
        "operand": operand_receipt,
        "kernel": (
            "polish_census_stationary_points followed by traced_boundary_read "
            "with the spline-restricted reachable-wall minimum"
        ),
        "jit": {
            "compile_count": int(single_counter["calls"]),
            "first_call_wall_s": single_walls[0],
            "wall_s_per_map": single_walls,
            "warm_wall_s_per_map": single_walls[1:],
            "median_wall_s_per_map": float(np.median(single_walls[1:])),
            "spread": _spread(single_walls[1:]),
        },
        "vmap": {
            "compile_count": int(batch_counter["calls"]),
            "first_batch_wall_s": batch_first_wall,
            "warm_batch_wall_s": batch_walls,
            "warm_wall_s_per_map": [value / len(fields) for value in batch_walls],
            "median_warm_wall_s_per_map": float(np.median(batch_walls) / len(fields)),
            "spread_per_map": {
                key: value / len(fields) for key, value in _spread(batch_walls).items()
            },
        },
        "agreement": {
            "eager_vs_jit": _agreement(eager_host, compiled_host),
            "eager_vs_vmap": _agreement(eager_host, batched_host),
            "jit_vs_vmap": _agreement(compiled_host, batched_host),
        },
        "selected_outputs": {
            key: value.tolist() for key, value in batched_host.items()
        },
    }


def capture(
    source_root: Path,
    revision: str,
    output: Path,
    operands: Path,
    cache: Path,
) -> None:
    """Measure one revision in its own process and write the raw receipt."""
    _activate_source(source_root)
    import jax
    import jaxlib

    from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
    from benchmarks.receipt_raster_check import _profile_and_seed

    configure_dtypes()
    _events, snapshot = _cache_watch()
    cache.mkdir(parents=True, exist_ok=True)
    cache_configuration = configure_persistent_compilation_cache(
        cache, minimum_compile_seconds=0.0
    )
    setup = _profile_and_seed()
    before_components = snapshot()
    components = _component_measurement(setup)
    after_components = snapshot()
    before_solve = snapshot()
    solve = _solve_measurement(setup)
    after_solve = snapshot()
    topology = _topology_measurement(operands)
    after_topology = snapshot()
    payload = {
        "schema": "nova.polish_support_performance.raw",
        "schema_version": 1,
        "revision": revision,
        "source_root": str(source_root.resolve()),
        "captured_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "gpu": _gpu_identity(),
            "tmpdir": os.environ.get("TMPDIR"),
        },
        "persistent_compilation_cache": cache_configuration.receipt(),
        "cache_events": {
            "component_profile": _cache_delta(before_components, after_components),
            "solve": _cache_delta(before_solve, after_solve),
            "topology": _cache_delta(after_solve, after_topology),
            "total": after_topology,
        },
        "per_trip_profile": components,
        "solve": solve,
        "topology_read": topology,
    }
    _strict_write(output, payload)


def _relative_change(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / baseline


def _wall_verdict(
    baseline: float, candidate: float, pooled_spread: float
) -> dict[str, Any]:
    change = candidate - baseline
    return {
        "absolute_change_s": change,
        "relative_change": _relative_change(baseline, candidate),
        "pooled_launch_spread_s": pooled_spread,
        "inside_launch_spread": bool(abs(change) <= pooled_spread),
    }


def _measure_verdicts(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    base_solve = baseline["solve"]
    cand_solve = candidate["solve"]
    solve_spread = max(
        base_solve["warm_launch_spread"]["range_s"],
        cand_solve["warm_launch_spread"]["range_s"],
    )
    base_topology = baseline["topology_read"]
    cand_topology = candidate["topology_read"]
    base_jit_warm = base_topology["jit"]["wall_s_per_map"][1:]
    cand_jit_warm = cand_topology["jit"]["wall_s_per_map"][1:]
    jit_spread = max(
        max(base_jit_warm) - min(base_jit_warm),
        max(cand_jit_warm) - min(cand_jit_warm),
    )
    vmap_spread = max(
        base_topology["vmap"]["spread_per_map"]["range_s"],
        cand_topology["vmap"]["spread_per_map"]["range_s"],
    )
    residual_scale = max(abs(base_solve["terminal_residual"]), np.finfo(float).tiny)
    class_names = {0: "limited", 1: "diverted"}
    base_class = base_solve.get(
        "achieved_class_name", class_names[int(base_solve["achieved_class"])]
    )
    candidate_class = cand_solve.get(
        "achieved_class_name", class_names[int(cand_solve["achieved_class"])]
    )
    base_profile = baseline["per_trip_profile"]
    candidate_profile = candidate["per_trip_profile"]
    base_residual = base_profile["components"]["residual_evaluation"]
    candidate_residual = candidate_profile["components"]["residual_evaluation"]
    residual_spread = max(
        base_residual["warm_spread"]["range_s"],
        candidate_residual["warm_spread"]["range_s"],
    )
    base_trip = base_profile["components"]["one_newton_trip"]
    candidate_trip = candidate_profile["components"]["one_newton_trip"]
    trip_spread = max(
        base_trip["warm_spread"]["range_s"],
        candidate_trip["warm_spread"]["range_s"],
    )
    return {
        "solve_compile_count": {
            "baseline": base_solve["compile_count"],
            "candidate": cand_solve["compile_count"],
            "relative_change": (
                cand_solve["compile_count"] - base_solve["compile_count"]
            )
            / base_solve["compile_count"],
            "inside_launch_spread": base_solve["compile_count"]
            == cand_solve["compile_count"],
        },
        "solve_first_call_wall": _wall_verdict(
            base_solve["first_call_wall_s"],
            cand_solve["first_call_wall_s"],
            solve_spread,
        ),
        "solve_median_warm_wall": _wall_verdict(
            base_solve["median_warm_wall_s"],
            cand_solve["median_warm_wall_s"],
            solve_spread,
        ),
        "solve_trips": {
            "baseline": base_solve["trips"],
            "candidate": cand_solve["trips"],
            "relative_change": (cand_solve["trips"] - base_solve["trips"])
            / max(base_solve["trips"], 1),
            "inside_launch_spread": base_solve["trips"] == cand_solve["trips"],
        },
        "solve_terminal_residual": {
            "baseline": base_solve["terminal_residual"],
            "candidate": cand_solve["terminal_residual"],
            "relative_change": (
                cand_solve["terminal_residual"] - base_solve["terminal_residual"]
            )
            / residual_scale,
            "inside_launch_spread": bool(
                np.isclose(
                    base_solve["terminal_residual"],
                    cand_solve["terminal_residual"],
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            ),
        },
        "solve_achieved_class": {
            "baseline": base_class,
            "candidate": candidate_class,
            "required": "diverted",
            "both_diverted": base_class == candidate_class == "diverted",
            "inside_launch_spread": base_class == candidate_class,
        },
        "topology_jit_wall_per_map": _wall_verdict(
            float(statistics.median(base_jit_warm)),
            float(statistics.median(cand_jit_warm)),
            jit_spread,
        ),
        "topology_vmap_wall_per_map": _wall_verdict(
            base_topology["vmap"]["median_warm_wall_s_per_map"],
            cand_topology["vmap"]["median_warm_wall_s_per_map"],
            vmap_spread,
        ),
        "topology_batch_compile_count": {
            "baseline": base_topology["vmap"]["compile_count"],
            "candidate": cand_topology["vmap"]["compile_count"],
            "required": 1,
            "relative_change": (
                cand_topology["vmap"]["compile_count"]
                - base_topology["vmap"]["compile_count"]
            )
            / max(base_topology["vmap"]["compile_count"], 1),
            "inside_launch_spread": base_topology["vmap"]["compile_count"]
            == cand_topology["vmap"]["compile_count"]
            == 1,
        },
        "tensor_spline_fits_per_residual_evaluation": {
            "baseline": base_profile["fit_census"][
                "tensor_spline_fits_per_residual_evaluation"
            ],
            "candidate": candidate_profile["fit_census"][
                "tensor_spline_fits_per_residual_evaluation"
            ],
            "candidate_fits_per_topology_read": candidate_profile["fit_census"][
                "fits_per_topology_read"
            ],
            "required_fits_per_topology_read": 1,
            "inside_launch_spread": candidate_profile["fit_census"][
                "fits_per_topology_read"
            ]
            == 1,
        },
        "residual_evaluation_wall": _wall_verdict(
            base_residual["median_warm_wall_s"],
            candidate_residual["median_warm_wall_s"],
            residual_spread,
        ),
        "one_newton_trip_wall": _wall_verdict(
            base_trip["median_warm_wall_s"],
            candidate_trip["median_warm_wall_s"],
            trip_spread,
        ),
    }


def _figure(receipt: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = receipt["measurements"]["main"]
    candidate = receipt["measurements"]["candidate"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    solve = [
        baseline["solve"]["warm_launch_wall_s"],
        candidate["solve"]["warm_launch_wall_s"],
    ]
    axes[0].boxplot(solve, tick_labels=["main", "shared"], showmeans=True)
    axes[0].set_ylabel("warm solve wall [s]")
    axes[0].set_title("Frozen-six production solve")

    labels = ["jit", "vmap"]
    x = np.arange(len(labels))
    width = 0.36
    base_values = [
        baseline["topology_read"]["jit"]["median_wall_s_per_map"],
        baseline["topology_read"]["vmap"]["median_warm_wall_s_per_map"],
    ]
    candidate_values = [
        candidate["topology_read"]["jit"]["median_wall_s_per_map"],
        candidate["topology_read"]["vmap"]["median_warm_wall_s_per_map"],
    ]
    axes[1].bar(x - width / 2, base_values, width, label="main")
    axes[1].bar(x + width / 2, candidate_values, width, label="shared")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("topology wall per map [s]")
    axes[1].set_title("Eleven-map topology read")
    axes[1].legend()
    profile_labels = ["residual", "one trip"]
    base_profile = baseline["per_trip_profile"]["components"]
    candidate_profile = candidate["per_trip_profile"]["components"]
    base_profile_values = [
        base_profile["residual_evaluation"]["median_warm_wall_s"],
        base_profile["one_newton_trip"]["median_warm_wall_s"],
    ]
    candidate_profile_values = [
        candidate_profile["residual_evaluation"]["median_warm_wall_s"],
        candidate_profile["one_newton_trip"]["median_warm_wall_s"],
    ]
    axes[2].bar(x - width / 2, base_profile_values, width, label="main")
    axes[2].bar(x + width / 2, candidate_profile_values, width, label="shared")
    axes[2].set_xticks(x, profile_labels)
    axes[2].set_ylabel("warm wall [s]")
    axes[2].set_title("One-trip component profile")
    axes[2].legend()
    gate = receipt["gate"]
    scheduler = receipt["scheduler"]
    figure.suptitle(
        f"H200 job {scheduler['job_id']} · solve convergence "
        f"{gate['both_solve_arms_converged']} · diverted both "
        f"{gate['both_solve_arms_diverted']} · vmap compile count 1 / 1",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _scheduler(job_id: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList,Elapsed,Start,End",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line]
    row = next(item for item in rows if item[0] == job_id)
    return {
        "job_id": job_id,
        "state": row[1],
        "exit_code": row[2],
        "node": row[3],
        "elapsed": row[4],
        "started_at": row[5],
        "finished_at": row[6],
        "partition": "betelgeuse",
        "reservation": "gpu_0003_grpA",
        "gpu_count": 1,
    }


def combine(
    baseline_path: Path,
    candidate_path: Path,
    output: Path,
    figure: Path,
    job_id: str,
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    verdicts = _measure_verdicts(baseline, candidate)
    agreement = all(
        candidate["topology_read"]["agreement"][name]["all_fields_pass"]
        for name in ("eager_vs_jit", "eager_vs_vmap", "jit_vs_vmap")
    ) and all(
        baseline["topology_read"]["agreement"][name]["all_fields_pass"]
        for name in ("eager_vs_jit", "eager_vs_vmap", "jit_vs_vmap")
    )
    receipt = {
        "schema": "nova.polish_support_performance",
        "schema_version": 1,
        "comparison": {
            "main_revision": baseline["revision"],
            "candidate_revision": candidate["revision"],
            "same_h200_job": True,
            "same_frozen_six_arm": baseline["solve"]["reference"]
            == candidate["solve"]["reference"],
            "same_persisted_carrier": (
                baseline["solve"]["carrier"]["carrier"]["semantic_response_identity"]
                == candidate["solve"]["carrier"]["carrier"][
                    "semantic_response_identity"
                ]
            ),
            "same_eleven_map_operand": baseline["topology_read"]["operand"]["sha256"]
            == candidate["topology_read"]["operand"]["sha256"],
        },
        "scheduler": _scheduler(job_id),
        "measurements": {"main": baseline, "candidate": candidate},
        "verdicts": verdicts,
        "gate": {
            "production_solve_remains_fast": verdicts["solve_median_warm_wall"][
                "inside_launch_spread"
            ],
            "topology_read_remains_gpu_batchable": verdicts[
                "topology_batch_compile_count"
            ]["inside_launch_spread"],
            "eager_jit_vmap_agreement": agreement,
            "both_solve_arms_converged": bool(
                baseline["solve"]["converged"] and candidate["solve"]["converged"]
            ),
            "both_solve_arms_diverted": verdicts["solve_achieved_class"][
                "both_diverted"
            ],
            "one_tensor_spline_fit_per_topology_read": verdicts[
                "tensor_spline_fits_per_residual_evaluation"
            ]["inside_launch_spread"],
            "passes": bool(
                verdicts["solve_median_warm_wall"]["inside_launch_spread"]
                and verdicts["topology_batch_compile_count"]["inside_launch_spread"]
                and agreement
                and baseline["solve"]["converged"]
                and candidate["solve"]["converged"]
                and verdicts["solve_achieved_class"]["both_diverted"]
                and verdicts["tensor_spline_fits_per_residual_evaluation"][
                    "inside_launch_spread"
                ]
            ),
        },
        "figure": FIGURE_IDENTITY,
        "raw_capture_sha256": {
            "main": _sha256(baseline_path),
            "candidate": _sha256(candidate_path),
        },
    }
    _strict_write(output, receipt)
    _figure(receipt, figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--repository", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--revision", default=BASELINE_REVISION)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--output", type=Path, required=True)
    cleanup_parser.add_argument("--revision", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--revision", required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    capture_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--baseline", type=Path, required=True)
    combine_parser.add_argument("--candidate", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    combine_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    combine_parser.add_argument("--job-id", required=True)
    arguments = parser.parse_args()
    if arguments.command == "materialize":
        materialize(arguments.repository, arguments.output, arguments.revision)
    elif arguments.command == "cleanup":
        remove_materialized_source(arguments.output, arguments.revision)
    elif arguments.command == "capture":
        capture(
            arguments.source_root,
            arguments.revision,
            arguments.output,
            arguments.operands,
            arguments.cache,
        )
    else:
        combine(
            arguments.baseline,
            arguments.candidate,
            arguments.output,
            arguments.figure,
            arguments.job_id,
        )


if __name__ == "__main__":
    main()
