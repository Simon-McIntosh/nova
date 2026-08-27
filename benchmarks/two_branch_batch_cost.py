"""Measure single-branch and dual-branch converged solve cost on one H200.

The benchmark pairs independent, non-empty limited and diverted analytic
fixtures. Every batch member is a distinct near-root state and every root and
seed carries topology, map-residual, flux-span, and physical-observable
qualification. Input-file read, host-to-device transfer, compilation, warm-up,
resident execution, and the read-inclusive total are timed and reported
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Callable

import jax
import numpy as np

from benchmarks.portfolio_warm_start import _problem
from nova.equilibrium import FluxLattice, ForwardProfile
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    OracleMachine,
    WALL_POINT_COUNT,
    _flux_blocks,
    _square_stencils,
    analytic_case,
    exact_current_moments,
    exact_state,
    forward_operator,
    limiter_contour,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/dual-branch-selection/two-branch-batch-cost.json"
BATCH_WIDTHS = (4, 16)
NEWTON_PROMOTIONS = 2
GMRES_ITERATIONS = 30
CONVERGENCE_TOLERANCE = 1.0e-10
TIMING_REPEATS = 5
CENSUS_SLICE_COUNT = 1_341_435
SEED_COHORTS = {
    "nearest_neighbour": (1.0e-6, 1.0e-5),
    "close_neighbour": (1.0e-5, 5.0e-5),
    "outer_qualified_neighbour": (5.0e-5, 1.0e-4),
}
LIMITED_LATTICE_SHAPE = (13, 17)


def _strict(value: Any) -> Any:
    """Return a strict JSON-compatible tree."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the receipt atomically as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(values: np.ndarray) -> str:
    """Return the binary digest of one float64 input state."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    """Return the digest of one serialized input bank."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    """Return the exact source revision executed by the benchmark."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _limited_tensor_machine() -> OracleMachine:
    """Construct the limited analytic carrier on a tensor-product lattice."""
    case = analytic_case()
    inboard, outboard = case.boundary_midplane_radii()
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    lattice = FluxLattice(
        np.linspace(inboard - 0.12, outboard + 0.12, LIMITED_LATTICE_SHAPE[0]),
        np.linspace(
            -1.18 * half_height,
            1.18 * half_height,
            LIMITED_LATTICE_SHAPE[1],
        ),
    )
    node = lattice.coordinate
    radial_half_step = 0.5 * lattice.radial_step
    vertical_half_step = 0.5 * lattice.vertical_step
    offsets = np.asarray(
        [
            [-radial_half_step, -vertical_half_step],
            [radial_half_step, -vertical_half_step],
            [radial_half_step, vertical_half_step],
            [-radial_half_step, vertical_half_step],
        ]
    )
    polygons = tuple(centre + offsets for centre in node)
    sampling = np.asarray(polygons)
    stencil = _square_stencils(*LIMITED_LATTICE_SHAPE)
    area = lattice.cell_area
    mesh = StencilMesh(node, stencil, area)
    moment_geometry = MomentGeometry.from_cells(
        mesh, polygons, sampling_vertices=sampling
    )
    sample = moment_geometry.sample_node_coordinates
    wall = limiter_contour(case, points=WALL_POINT_COUNT)
    grid_blocks = _flux_blocks(node, polygons, moment_geometry.atomic_mesh.centroids)
    wall_blocks = _flux_blocks(wall, polygons, moment_geometry.atomic_mesh.centroids)
    sample_blocks = _flux_blocks(
        sample, polygons, moment_geometry.atomic_mesh.centroids
    )
    return OracleMachine(
        node=node,
        area=area,
        cell_polygons=polygons,
        stencil=stencil,
        interior_stencil=stencil,
        wall_node=wall,
        sampling_vertices=sampling,
        sample_coordinates=sample,
        plasma_to_grid=grid_blocks[0],
        plasma_to_grid_r=grid_blocks[1],
        plasma_to_grid_z=grid_blocks[2],
        plasma_to_wall=wall_blocks[0],
        plasma_to_wall_r=wall_blocks[1],
        plasma_to_wall_z=wall_blocks[2],
        plasma_to_sample=sample_blocks[0],
        plasma_to_sample_r=sample_blocks[1],
        plasma_to_sample_z=sample_blocks[2],
    )


def _limited_fine_problem() -> tuple[ForwardProfile, Any, np.ndarray]:
    """Construct the non-empty fine limited fixture and its cold seed."""
    case = analytic_case()
    machine = _limited_tensor_machine()
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle = exact_state(case, coordinates)
    empty = forward_operator(case, machine)
    physical = exact_current_moments(case, empty, oracle)
    coefficients = empty.coupling_current_moments(physical)
    exterior = oracle - np.asarray(empty.current_moment_image(coefficients))
    profile = ForwardProfile(
        forward_operator(case, machine, exterior),
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=10,
    )
    cell_current = np.asarray(physical.cell_current)
    declared_current = float(np.sum(cell_current))
    current_centroid = (
        np.sum(machine.node * cell_current[:, None], axis=0) / declared_current
    )
    cold = profile.cold_seed_portfolio(
        declared_current,
        current_centroid,
    )
    return profile, cold, oracle


def _state_qualification(
    profile: ForwardProfile,
    state: np.ndarray | jax.Array,
    requested_class: TopologyClass,
) -> dict[str, Any]:
    """Read topology, direct map residual, and physical observables."""
    values = jax.numpy.asarray(state)
    mapped = profile.flux_map(requested_class=requested_class)(values)
    delta = mapped - values
    _masks, topology = profile.operator.read(values)
    moments = profile.integral_observation(values)
    scale = max(float(np.max(np.abs(np.asarray(mapped)))), 1.0e-30)
    return {
        "requested_class": int(requested_class),
        "achieved_class": int(np.asarray(topology.diverted)),
        "topology_consistent": bool(
            int(np.asarray(topology.diverted)) == int(requested_class)
        ),
        "relative_map_residual": float(np.max(np.abs(np.asarray(delta))) / scale),
        "absolute_map_residual_wb": float(np.max(np.abs(np.asarray(delta)))),
        "axis_flux_wb": float(np.asarray(topology.axis_flux)),
        "boundary_flux_wb": float(np.asarray(topology.boundary_flux)),
        "flux_span_wb": float(np.asarray(topology.flux_span)),
        "plasma_current_a": float(np.asarray(moments.plasma_current)),
        "poloidal_beta": float(np.asarray(moments.poloidal_beta)),
        "internal_inductance": float(np.asarray(moments.internal_inductance)),
        "volume_m3": float(np.asarray(moments.volume)),
        "major_radius_m": float(np.asarray(moments.major_radius)),
        "finite_state": bool(np.all(np.isfinite(np.asarray(values)))),
        "state_sha256": _digest(np.asarray(values)),
    }


def _summary(samples: list[float]) -> dict[str, Any]:
    """Return all samples and robust headline statistics."""
    return {
        "samples_seconds": samples,
        "minimum_seconds": float(np.min(samples)),
        "median_seconds": float(np.median(samples)),
        "maximum_seconds": float(np.max(samples)),
        "repeat_count": len(samples),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    """Return the median and full range of a seed-cohort measurement."""
    return {
        "values": values,
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "cohort_count": len(values),
    }


def _distinct_inputs(
    profiles: tuple[ForwardProfile, ForwardProfile],
    cold_flux: tuple[np.ndarray, np.ndarray],
    roots: tuple[np.ndarray, np.ndarray],
    count: int,
    distance_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct and qualify distinct limited and diverted input seeds."""
    classes = (TopologyClass.LIMITED, TopologyClass.DIVERTED)
    spans = []
    directions = []
    for profile, cold, root, branch_class in zip(
        profiles, cold_flux, roots, classes, strict=True
    ):
        qualification = _state_qualification(profile, root, branch_class)
        span = abs(qualification["flux_span_wb"])
        direction = cold - root
        scale = float(np.max(np.abs(direction)))
        if not np.isfinite(span) or span <= 0.0 or scale <= 0.0:
            raise RuntimeError("a branch span or seed direction is degenerate")
        spans.append(span)
        directions.append(direction / scale)

    distances = np.linspace(*distance_range, count, dtype=np.float64)
    limited = roots[0][None, :] + (
        distances[:, None] * spans[0] * directions[0][None, :]
    )
    diverted = roots[1][None, :] + (
        distances[:, None] * spans[1] * directions[1][None, :]
    )
    limited_digests = [_digest(row) for row in limited]
    diverted_digests = [_digest(row) for row in diverted]
    paired_digests = [
        _digest(np.concatenate((limited[index], diverted[index])))
        for index in range(count)
    ]
    if any(
        len(set(digests)) != count
        for digests in (limited_digests, diverted_digests, paired_digests)
    ):
        raise RuntimeError("the input constructor produced duplicate slice states")
    qualification = []
    for index, distance in enumerate(distances):
        qualification.append(
            {
                "slice_index": index,
                "distance_relative_to_branch_flux_span": float(distance),
                "limited": _state_qualification(
                    profiles[0], limited[index], TopologyClass.LIMITED
                ),
                "diverted": _state_qualification(
                    profiles[1], diverted[index], TopologyClass.DIVERTED
                ),
                "pair_sha256": paired_digests[index],
            }
        )
    if not all(
        item[branch]["topology_consistent"]
        and item[branch]["finite_state"]
        and np.isfinite(item[branch]["relative_map_residual"])
        and np.isfinite(item[branch]["flux_span_wb"])
        and abs(item[branch]["flux_span_wb"]) > 0.0
        and abs(item[branch]["plasma_current_a"]) > 0.0
        and item[branch]["volume_m3"] > 0.0
        for item in qualification
        for branch in ("limited", "diverted")
    ):
        raise RuntimeError(
            "an input seed is not a finite physical class-qualified state"
        )
    return (
        limited,
        diverted,
        {
            "construction": (
                "distinct deterministic distances along each production cold-seed "
                "direction from an independently qualified physical root"
            ),
            "distance_range_relative_to_branch_flux_span": distance_range,
            "distances_relative_to_branch_flux_span": distances,
            "limited_slice_sha256": limited_digests,
            "diverted_slice_sha256": diverted_digests,
            "paired_slice_sha256": paired_digests,
            "unique_limited_slice_count": len(set(limited_digests)),
            "unique_diverted_slice_count": len(set(diverted_digests)),
            "unique_slice_pair_count": len(set(paired_digests)),
            "broadcast_or_tile_used": False,
            "per_slice_qualification": qualification,
        },
    )


def _read_input(
    path: Path, keys: tuple[str, ...]
) -> tuple[tuple[np.ndarray, ...], float]:
    """Read and materialize one arm's inputs from its serialized bank."""
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as bank:
        values = tuple(np.array(bank[key], dtype=np.float64, copy=True) for key in keys)
    return values, time.perf_counter() - started


def _transfer(values: tuple[np.ndarray, ...]) -> tuple[tuple[jax.Array, ...], float]:
    """Place every input batch on the H200 and synchronize the transfer."""
    started = time.perf_counter()
    device_values = tuple(jax.device_put(value) for value in values)
    jax.block_until_ready(device_values)
    return device_values, time.perf_counter() - started


def _execute(
    compiled: Callable[..., Any], values: tuple[jax.Array, ...]
) -> tuple[Any, float]:
    """Run one complete solve and synchronize every result leaf."""
    started = time.perf_counter()
    result = compiled(*values)
    jax.block_until_ready(result)
    return result, time.perf_counter() - started


def _terminal_summary(
    arm: str,
    result: Any,
    roots: tuple[np.ndarray, np.ndarray],
    batch_width: int,
) -> dict[str, Any]:
    """Retain convergence, failure, residual, topology, and root-error counts."""
    if arm == "single_pinned_limited":
        converged = np.asarray(result.converged, dtype=bool)
        consistent = np.asarray(result.topology_consistent, dtype=bool)
        residual = np.asarray(result.residual, dtype=np.float64)
        flux = np.asarray(result.equilibrium.flux, dtype=np.float64)
        scale = max(float(np.max(np.abs(roots[0]))), np.finfo(float).tiny)
        root_error = np.max(np.abs(flux - roots[0]), axis=1) / scale
        slice_converged = converged
        branches_attempted = batch_width
        branches_converged = int(np.sum(converged))
        per_branch = {"limited": branches_converged}
    else:
        converged = np.stack(
            tuple(np.asarray(branch.converged, dtype=bool) for branch in result),
            axis=1,
        )
        consistent = np.stack(
            tuple(
                np.asarray(branch.topology_consistent, dtype=bool) for branch in result
            ),
            axis=1,
        )
        residual = np.stack(
            tuple(np.asarray(branch.residual, dtype=np.float64) for branch in result),
            axis=1,
        )
        root_error = np.stack(
            tuple(
                np.max(
                    np.abs(
                        np.asarray(branch.equilibrium.flux, dtype=np.float64) - root
                    ),
                    axis=1,
                )
                / max(float(np.max(np.abs(root))), np.finfo(float).tiny)
                for branch, root in zip(result, roots, strict=True)
            ),
            axis=1,
        )
        slice_converged = np.all(converged, axis=1)
        branches_attempted = 2 * batch_width
        branches_converged = int(np.sum(converged))
        per_branch = {
            "limited": int(np.sum(converged[:, int(TopologyClass.LIMITED)])),
            "diverted": int(np.sum(converged[:, int(TopologyClass.DIVERTED)])),
        }
    return {
        "input_slices_solved": batch_width,
        "input_slices_converged": int(np.sum(slice_converged)),
        "input_slice_failure_or_nonconvergence_count": int(
            batch_width - np.sum(slice_converged)
        ),
        "branches_attempted": branches_attempted,
        "branches_converged": branches_converged,
        "branch_failure_or_nonconvergence_count": branches_attempted
        - branches_converged,
        "converged_by_branch": per_branch,
        "topology_consistent_count": int(np.sum(consistent)),
        "maximum_relative_residual": float(np.max(residual)),
        "maximum_root_relative_error": float(np.max(root_error)),
        "all_finite_residuals": bool(np.all(np.isfinite(residual))),
    }


def _measure_arm(
    arm: str,
    solve: Callable[..., Any],
    bank_path: Path,
    bank_keys: tuple[str, ...],
    roots: tuple[np.ndarray, np.ndarray],
    batch_width: int,
) -> dict[str, Any]:
    """Measure compile, warm-up, execute-only, and read-inclusive solve cost."""
    compile_host, compile_read_seconds = _read_input(bank_path, bank_keys)
    compile_input, compile_transfer_seconds = _transfer(compile_host)
    compile_started = time.perf_counter()
    compiled = jax.jit(solve).lower(*compile_input).compile()
    compile_seconds = time.perf_counter() - compile_started

    warm_host, warm_read_seconds = _read_input(bank_path, bank_keys)
    warm_input, warm_transfer_seconds = _transfer(warm_host)
    warm_result, warm_execute_seconds = _execute(compiled, warm_input)
    warm_terminal = _terminal_summary(arm, warm_result, roots, batch_width)

    execute_samples = []
    steady_terminal_rows = []
    for _ in range(TIMING_REPEATS):
        result, elapsed = _execute(compiled, warm_input)
        execute_samples.append(elapsed)
        steady_terminal_rows.append(_terminal_summary(arm, result, roots, batch_width))

    total_samples = []
    read_samples = []
    transfer_samples = []
    execute_with_input_samples = []
    for _ in range(TIMING_REPEATS):
        total_started = time.perf_counter()
        host_values, read_seconds = _read_input(bank_path, bank_keys)
        device_values, transfer_seconds = _transfer(host_values)
        result, execute_seconds = _execute(compiled, device_values)
        total_samples.append(time.perf_counter() - total_started)
        read_samples.append(read_seconds)
        transfer_samples.append(transfer_seconds)
        execute_with_input_samples.append(execute_seconds)
        steady_terminal_rows.append(_terminal_summary(arm, result, roots, batch_width))

    execute = _summary(execute_samples)
    total = _summary(total_samples)
    median_execute = execute["median_seconds"]
    median_total = total["median_seconds"]
    return {
        "arm": arm,
        "batch_width": batch_width,
        "input_bank_keys": bank_keys,
        "input_shapes": [list(values.shape) for values in compile_host],
        "input_bytes": int(sum(values.nbytes for values in compile_host)),
        "compile": {
            "input_read_seconds": compile_read_seconds,
            "host_to_device_seconds": compile_transfer_seconds,
            "lower_and_compile_seconds": compile_seconds,
        },
        "warm_up": {
            "input_read_seconds": warm_read_seconds,
            "host_to_device_seconds": warm_transfer_seconds,
            "execute_seconds": warm_execute_seconds,
            "total_seconds": (
                warm_read_seconds + warm_transfer_seconds + warm_execute_seconds
            ),
            "terminal": warm_terminal,
        },
        "steady_state_execute_only": {
            **execute,
            "seconds_per_input_slice": median_execute / batch_width,
            "input_slices_per_second": batch_width / median_execute,
            "input_residency": "device resident before the timed execute call",
        },
        "steady_state_total_including_input_read_and_transfer": {
            **total,
            "seconds_per_input_slice": median_total / batch_width,
            "input_slices_per_second": batch_width / median_total,
            "input_read_samples_seconds": read_samples,
            "host_to_device_samples_seconds": transfer_samples,
            "execute_samples_seconds": execute_with_input_samples,
            "input_source": "node-local uncompressed NPZ",
            "page_cache_policy": (
                "not flushed; samples include ordinary node-local reads after "
                "the bank was staged"
            ),
        },
        "steady_state_accounting": {
            "solve_invocations": len(steady_terminal_rows),
            "input_slice_solves": batch_width * len(steady_terminal_rows),
            "input_slice_convergences": sum(
                row["input_slices_converged"] for row in steady_terminal_rows
            ),
            "input_slice_failure_or_nonconvergence_count": sum(
                row["input_slice_failure_or_nonconvergence_count"]
                for row in steady_terminal_rows
            ),
            "branch_solves": sum(
                row["branches_attempted"] for row in steady_terminal_rows
            ),
            "branch_convergences": sum(
                row["branches_converged"] for row in steady_terminal_rows
            ),
            "branch_failure_or_nonconvergence_count": sum(
                row["branch_failure_or_nonconvergence_count"]
                for row in steady_terminal_rows
            ),
            "per_invocation": steady_terminal_rows,
        },
    }


def _run(output: Path) -> dict[str, Any]:
    """Run both arms at every declared width and write the banked receipt."""
    configure_dtypes()
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"reserved H200 GPU required, got {device}")
    if not platform.node().startswith("98dci4-gpu-0003"):
        raise RuntimeError(f"reserved H200 host required, got {platform.node()}")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")
    if not jax.config.x64_enabled:
        raise RuntimeError("float64 support is disabled")

    setup_started = time.perf_counter()
    limited_profile, limited_cold, limited_root = _limited_fine_problem()
    diverted_profile, diverted_cold, diverted_root = _problem()
    profiles = (limited_profile, diverted_profile)
    roots = (limited_root, diverted_root)
    cold_flux = (
        np.asarray(
            limited_cold.branches.flux[int(TopologyClass.LIMITED)],
            dtype=np.float64,
        ),
        np.asarray(
            diverted_cold.branches.flux[int(TopologyClass.DIVERTED)],
            dtype=np.float64,
        ),
    )
    root_qualification = {
        "limited": _state_qualification(
            limited_profile, limited_root, TopologyClass.LIMITED
        ),
        "diverted": _state_qualification(
            diverted_profile, diverted_root, TopologyClass.DIVERTED
        ),
    }
    if not all(
        row["topology_consistent"]
        and row["finite_state"]
        and row["relative_map_residual"] <= CONVERGENCE_TOLERANCE
        and np.isfinite(row["flux_span_wb"])
        and abs(row["flux_span_wb"]) > 0.0
        and abs(row["plasma_current_a"]) > 0.0
        and row["volume_m3"] > 0.0
        for row in root_qualification.values()
    ):
        raise RuntimeError("a branch root is not a converged non-empty equilibrium")
    setup_seconds = time.perf_counter() - setup_started

    def solve_single(states: jax.Array) -> Any:
        return jax.vmap(
            lambda state: limited_profile.solve_branch(
                state,
                TopologyClass.LIMITED,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=NEWTON_PROMOTIONS,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(states)

    def solve_dual_branch(limited_states: jax.Array, diverted_states: jax.Array) -> Any:
        limited_results = jax.vmap(
            lambda state: limited_profile.solve_branch(
                state,
                TopologyClass.LIMITED,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=NEWTON_PROMOTIONS,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(limited_states)
        diverted_results = jax.vmap(
            lambda state: diverted_profile.solve_branch(
                state,
                TopologyClass.DIVERTED,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=NEWTON_PROMOTIONS,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(diverted_states)
        return limited_results, diverted_results

    rows = []
    input_policies = []
    scratch_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    with tempfile.TemporaryDirectory(
        prefix="nova-two-branch-cost-", dir=scratch_root
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        for width in BATCH_WIDTHS:
            width_rows = []
            for cohort, distance_range in SEED_COHORTS.items():
                limited_inputs, diverted_inputs, input_policy = _distinct_inputs(
                    profiles, cold_flux, roots, width, distance_range
                )
                input_policies.append(
                    {
                        "batch_width": width,
                        "seed_cohort": cohort,
                        **input_policy,
                    }
                )
                bank_path = temporary / f"inputs-{width}-{cohort}.npz"
                np.savez(
                    bank_path,
                    limited=limited_inputs,
                    diverted=diverted_inputs,
                )
                single = _measure_arm(
                    "single_pinned_limited",
                    solve_single,
                    bank_path,
                    ("limited",),
                    roots,
                    width,
                )
                dual = _measure_arm(
                    "two_branch_portfolio",
                    solve_dual_branch,
                    bank_path,
                    ("limited", "diverted"),
                    roots,
                    width,
                )
                execute_ratio = (
                    dual["steady_state_execute_only"]["seconds_per_input_slice"]
                    / single["steady_state_execute_only"]["seconds_per_input_slice"]
                )
                total_ratio = (
                    dual["steady_state_total_including_input_read_and_transfer"][
                        "seconds_per_input_slice"
                    ]
                    / single["steady_state_total_including_input_read_and_transfer"][
                        "seconds_per_input_slice"
                    ]
                )
                width_rows.append(
                    {
                        "seed_cohort": cohort,
                        "distance_range_relative_to_branch_flux_span": distance_range,
                        "input_bank": {
                            "format": "uncompressed NPZ staged on node-local scratch",
                            "file_bytes": bank_path.stat().st_size,
                            "sha256": _file_digest(bank_path),
                            "retained_after_run": False,
                        },
                        "single_pinned_branch": single,
                        "two_branch_portfolio": dual,
                        "cost_ratio_portfolio_over_single": {
                            "execute_only": execute_ratio,
                            "total_including_input_read_and_transfer": total_ratio,
                        },
                    }
                )
                del single, dual
                jax.clear_caches()

            single_execute = [
                row["single_pinned_branch"]["steady_state_execute_only"][
                    "seconds_per_input_slice"
                ]
                for row in width_rows
            ]
            dual_execute = [
                row["two_branch_portfolio"]["steady_state_execute_only"][
                    "seconds_per_input_slice"
                ]
                for row in width_rows
            ]
            single_total = [
                row["single_pinned_branch"][
                    "steady_state_total_including_input_read_and_transfer"
                ]["seconds_per_input_slice"]
                for row in width_rows
            ]
            dual_total = [
                row["two_branch_portfolio"][
                    "steady_state_total_including_input_read_and_transfer"
                ]["seconds_per_input_slice"]
                for row in width_rows
            ]
            execute_ratios = [
                row["cost_ratio_portfolio_over_single"]["execute_only"]
                for row in width_rows
            ]
            total_ratios = [
                row["cost_ratio_portfolio_over_single"][
                    "total_including_input_read_and_transfer"
                ]
                for row in width_rows
            ]
            rows.append(
                {
                    "batch_width": width,
                    "seed_cohorts": width_rows,
                    "cost_distribution_across_seed_cohorts": {
                        "single_execute_seconds_per_slice": _distribution(
                            single_execute
                        ),
                        "dual_execute_seconds_per_slice": _distribution(dual_execute),
                        "single_total_seconds_per_slice": _distribution(single_total),
                        "dual_total_seconds_per_slice": _distribution(dual_total),
                        "execute_cost_ratio_dual_over_single": _distribution(
                            execute_ratios
                        ),
                        "total_cost_ratio_dual_over_single": _distribution(
                            total_ratios
                        ),
                    },
                    "campaign_extrapolation": {
                        "label": "extrapolation_not_measured_campaign_throughput",
                        "slice_count": CENSUS_SLICE_COUNT,
                        "assumptions": (
                            "one H200 repeats the median seed-cohort cost at this "
                            "batch width with the same convergence and node-local "
                            "input staging across every catalog slice; compile, "
                            "warm-up, fixture setup, and cohort range are excluded "
                            "from the multiplied term"
                        ),
                        "single_median_total_seconds": (
                            CENSUS_SLICE_COUNT * float(np.median(single_total))
                        ),
                        "dual_median_total_seconds": (
                            CENSUS_SLICE_COUNT * float(np.median(dual_total))
                        ),
                        "single_total_seconds_range": [
                            CENSUS_SLICE_COUNT * min(single_total),
                            CENSUS_SLICE_COUNT * max(single_total),
                        ],
                        "dual_total_seconds_range": [
                            CENSUS_SLICE_COUNT * min(dual_total),
                            CENSUS_SLICE_COUNT * max(dual_total),
                        ],
                    },
                }
            )

    receipt = {
        "schema": "nova.two-branch-batch-cost",
        "source_revision": _source_revision(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "device_kind": device.device_kind,
            "device_count": len(jax.devices()),
            "jax_version": jax.__version__,
            "production_dtype": "float64",
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "measurement_contract": {
            "batch_widths": BATCH_WIDTHS,
            "seed_cohorts": SEED_COHORTS,
            "solver_entry_points": {
                "single": "ForwardProfile.solve_branch pinned LIMITED",
                "dual_branch": (
                    "one compiled callable containing independent production "
                    "ForwardProfile.solve_branch LIMITED and DIVERTED lanes"
                ),
            },
            "branch_pairing_arrangement": (
                "independent qualified limited and diverted fixtures are paired in "
                "one executable because the current bank contains no operator with "
                "two nondegenerate qualified roots; this measures both real solve "
                "lanes and does not claim shared-operator constant fusion"
            ),
            "newton_promotions": NEWTON_PROMOTIONS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "timing_repeats": TIMING_REPEATS,
            "complete_converged_solves_timed": all(
                cohort[arm]["steady_state_accounting"][
                    "input_slice_failure_or_nonconvergence_count"
                ]
                == 0
                for row in rows
                for cohort in row["seed_cohorts"]
                for arm in ("single_pinned_branch", "two_branch_portfolio")
            ),
            "one_map_application_benchmark": False,
            "input_read_in_total_timed_region": True,
            "compile_and_warm_up_excluded_from_steady_state": True,
            "failure_policy": (
                "every input slice and branch is counted; failed or non-converged "
                "results remain in terminal counts and are never dropped"
            ),
        },
        "fixture": {
            "definition": (
                "independent analytic production profiles: a tensor-product limited "
                "carrier and the qualified diverted root"
            ),
            "root_qualification": root_qualification,
            "root_shapes": [list(root.shape) for root in roots],
            "setup_seconds_excluded_from_measurement": setup_seconds,
        },
        "input_policy": {
            "cohort_count_per_batch_width": len(SEED_COHORTS),
            "qualification_fields": (
                "topology class, direct pinned-map residual, flux span, plasma "
                "current, poloidal beta, internal inductance, volume, major radius, "
                "finiteness, and state digest"
            ),
            "cohorts": input_policies,
        },
        "conditioning_preflight": {
            "status": "identity_rejected",
            "source_slurm_jobs": [1254531, 1254536, 1254545],
            "unconditioned_terminal": {
                "reported_relative_residual": 0.0,
                "direct_relative_residual": 0.0,
                "plasma_current_a": 0.0,
                "volume_m3": 0.0,
                "axis_flux_wb": None,
                "boundary_flux_wb": None,
                "flux_span_wb": None,
                "interpretation": (
                    "empty zero-current identity, not a physical converged root; "
                    "controlled span perturbations are undefined because its span "
                    "is non-finite"
                ),
            },
            "conditioned_trajectory": {
                "steps": [30, 100],
                "relative_residual": [0.20885528789784424, 0.08863953295475595],
                "plasma_current_a": [86612.72175390508, 38833.74784340524],
                "volume_m3": [2.4815866694993, 1.1684091821883185],
                "absolute_flux_span_wb": [0.18181015335366602, 0.07455748909722243],
                "conditioning_count": [30, 100],
                "interpretation": (
                    "conditioning slows the same collapse toward the empty identity; "
                    "it is not converging to a different qualified physical root"
                ),
            },
            "timing_exclusion": (
                "neither the empty unconditioned identity nor the collapsing "
                "conditioned trajectory appears in any timed input"
            ),
        },
        "seed_range_preflight": {
            "source_slurm_job": 1254602,
            "status": "nonconverged_ranges_excluded_from_cost_distribution",
            "qualified_range_relative_to_branch_flux_span": [1.0e-6, 1.0e-4],
            "excluded_ranges": [
                {
                    "range_relative_to_branch_flux_span": [1.0e-4, 5.0e-4],
                    "limited_converged_per_invocation": {
                        "batch_width_4": "1/4",
                        "batch_width_16": "1/16",
                    },
                    "diverted_converged_per_invocation": {
                        "batch_width_4": "4/4",
                        "batch_width_16": "16/16",
                    },
                    "maximum_terminal_relative_residual": 4.561941905915858e-4,
                },
                {
                    "range_relative_to_branch_flux_span": [5.0e-4, 1.0e-3],
                    "limited_converged_per_invocation": {
                        "batch_width_4": "0/4",
                        "batch_width_16": "0/16",
                    },
                    "diverted_converged_per_invocation": {
                        "batch_width_4": "4/4",
                        "batch_width_16": "16/16",
                    },
                    "maximum_terminal_relative_residual": 9.124068905329867e-4,
                },
            ],
            "interpretation": (
                "the two-promotion limited workload is seed-range sensitive; "
                "nonconverged slices are retained here and their timings are not "
                "used as converged-solve cost evidence"
            ),
        },
        "measurements": rows,
        "decision_scope": {
            "provides": (
                "measured accelerator cost multiple for a two-branch portfolio "
                "relative to one pinned branch"
            ),
            "does_not_provide": (
                "catalog margin distribution or a policy decision by itself"
            ),
        },
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    """Run the H200 measurement from a reserved SLURM allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = _run(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "source_revision": receipt["source_revision"],
                "rows": [
                    {
                        "batch_width": row["batch_width"],
                        "median_execute_ratio": row[
                            "cost_distribution_across_seed_cohorts"
                        ]["execute_cost_ratio_dual_over_single"]["median"],
                        "median_total_ratio": row[
                            "cost_distribution_across_seed_cohorts"
                        ]["total_cost_ratio_dual_over_single"]["median"],
                    }
                    for row in receipt["measurements"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
