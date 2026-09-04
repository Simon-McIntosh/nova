"""Measure the exact rectangular receipt on one cached MAST solve."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as carrier
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.forward import ForwardDomainLabel
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/forward-solve-api/raster-receipt.json")
DEFAULT_FIGURE_DIR = Path("docs/figures/forward-solve-api/raster-receipt")
FIGURE_IDENTITY = "/nova/figures/forward-solve-api/raster-receipt/flux-and-labels.png"
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/raster-receipt")
REFERENCE_SHOT = 22086
MEASURED_LAUNCHES = 5


def _profile_and_seed():
    selected = next(
        row
        for row in parity.select_slices_by_shot(parity.DECOMPOSITION_BANK)
        if int(row[0]["shot"]) == REFERENCE_SHOT
    )
    case, context = parity._mast_case_from_selection(SHOT_STORE, *selected)
    response_cache, cache_receipt = _persisted_response_cache(
        carrier.DEFAULT_CARRIER, carrier.DEFAULT_RECEIPT
    )
    _case, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    return case, profile, target_current, cache_receipt, policy


def _compile_counted(function: Callable, argument):
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


def _timed_launches(executable, argument):
    walls = []
    result = executable(argument)
    jax.block_until_ready(result)
    for _ in range(MEASURED_LAUNCHES):
        started = perf_counter()
        result = executable(argument)
        jax.block_until_ready(result)
        walls.append(perf_counter() - started)
    return result, walls


def _arrays_identical(left, right) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True)
        for first, second in zip(left_leaves, right_leaves, strict=True)
    )


def _write_figure(raster, output: Path) -> None:
    """Render normalized flux, labels and the traced separatrix."""
    radius = np.asarray(raster.radius)
    height = np.asarray(raster.height)
    shape = tuple(int(value) for value in np.asarray(raster.shape))
    normalized = np.asarray(raster.psi_norm).reshape(shape).T
    labels = np.asarray(raster.domain_label).reshape(shape).T
    count = int(raster.separatrix_vertex_count)
    separatrix = np.asarray(raster.separatrix)[:count]

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    flux_image = axes[0].pcolormesh(radius, height, normalized, shading="auto")
    figure.colorbar(flux_image, ax=axes[0], label=r"$\psi_N$")
    if count:
        closed = np.vstack((separatrix, separatrix[0]))
        axes[0].plot(closed[:, 0], closed[:, 1], color="white", linewidth=1.2)
    label_image = axes[1].pcolormesh(radius, height, labels, shading="nearest")
    figure.colorbar(label_image, ax=axes[1], label="domain label")
    axes[0].set_title("Exact current image")
    axes[1].set_title("Nova domain labels")
    for axis in axes:
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_aspect("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def capture(output: Path, figure_dir: Path, cache: Path) -> None:
    """Write raw compilation, launch, direct-image and topology evidence."""
    configure_dtypes()
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    case, profile, target_current, cache_receipt, policy = _profile_and_seed()
    seed = jax.numpy.asarray(case["state"])
    prescribed_current = jax.numpy.asarray(profile.operator.prescribed_field.current)

    def solve(initial_flux):
        return profile.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            target_current=target_current,
            prescribed_current=prescribed_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=0,
            newton_steps=1,
            gmres_iterations=2,
        ).equilibrium

    def solve_without_raster(initial_flux):
        return solve(initial_flux)._replace(raster_flux=None)

    without_executable, without_compiles, without_compile_wall = _compile_counted(
        solve_without_raster, seed
    )
    with_executable, with_compiles, with_compile_wall = _compile_counted(solve, seed)
    without_result, without_walls = _timed_launches(without_executable, seed)
    with_result, with_walls = _timed_launches(with_executable, seed)
    raster = with_result.raster_flux
    if raster is None:
        raise RuntimeError("the populated solve omitted its rectangular receipt")

    zero = jax.numpy.zeros_like(with_result.cell_current)
    direct = profile.operator.raster_image(
        CellCurrentMoments(with_result.cell_current, zero, zero),
        prescribed_current=prescribed_current,
    )
    psi = np.asarray(raster.psi)
    psi_norm = np.asarray(raster.psi_norm)
    direct = np.asarray(direct)
    direct_norm = (direct - float(with_result.topology.axis_flux)) / float(
        with_result.topology.flux_span
    )
    flux_scale = max(float(np.max(np.abs(direct))), np.finfo(np.float64).tiny)
    norm_scale = max(float(np.max(np.abs(direct_norm))), np.finfo(np.float64).tiny)
    flux_relative_error = float(np.max(np.abs(psi - direct)) / flux_scale)
    norm_relative_error = float(np.max(np.abs(psi_norm - direct_norm)) / norm_scale)
    existing_identical = _arrays_identical(
        without_result, with_result._replace(raster_flux=None)
    )
    without_median = float(np.median(without_walls))
    with_median = float(np.median(with_walls))
    figure_path = figure_dir / "flux-and-labels.png"
    _write_figure(raster, figure_path)
    count = int(raster.separatrix_vertex_count)
    label_values = sorted(int(value) for value in np.unique(raster.domain_label))

    record = {
        "schema": "nova.raster_flux_receipt_benchmark",
        "schema_version": 1,
        "reference": case["reference"],
        "execution": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": socket.gethostname(),
            "device": str(jax.devices()[0]),
            "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "persistent_compilation_cache": str(cache),
            "launch_then_harvest": True,
            "exit_marker": "pending harvest",
        },
        "cached_carrier": cache_receipt,
        "field_policy": policy,
        "raster": {
            "shape": [int(value) for value in np.asarray(raster.shape)],
            "radius_m": np.asarray(raster.radius).tolist(),
            "height_m": np.asarray(raster.height).tolist(),
            "node_count": int(psi.size),
            "domain_label_values": label_values,
            "domain_label_names": [
                ForwardDomainLabel(value).name.lower() for value in label_values
            ],
            "separatrix_vertex_count": count,
            "geometry_built_once_with_operator": True,
            "interpolation_used": False,
            "figure": FIGURE_IDENTITY,
        },
        "measurement": {
            "without_raster": {
                "compile_calls": without_compiles,
                "compile_wall_s": without_compile_wall,
                "launch_count": MEASURED_LAUNCHES + 1,
                "measured_solve_wall_s": without_walls,
                "median_solve_wall_s": without_median,
            },
            "with_raster": {
                "compile_calls": with_compiles,
                "compile_wall_s": with_compile_wall,
                "launch_count": MEASURED_LAUNCHES + 1,
                "measured_solve_wall_s": with_walls,
                "median_solve_wall_s": with_median,
            },
            "median_wall_delta_s": with_median - without_median,
            "raster_psi_direct_relative_error": flux_relative_error,
            "raster_psi_norm_direct_relative_error": norm_relative_error,
        },
        "receipt": {
            "all_psi_finite": bool(np.all(np.isfinite(psi))),
            "all_psi_norm_finite": bool(np.all(np.isfinite(psi_norm))),
            "existing_fields_bit_identical": existing_identical,
            "solve_residual": float(with_result.fixed_point.residual),
        },
        "verdict": {
            "compile_count_unchanged_at_one": without_compiles == with_compiles == 1,
            "existing_receipt_fields_bit_identical": existing_identical,
            "persisted_shape_is_33_by_33": tuple(np.asarray(raster.shape)) == (33, 33),
            "direct_green_agreement_at_1e_12": (
                flux_relative_error <= 1.0e-12 and norm_relative_error <= 1.0e-12
            ),
            "nonzero_spline_separatrix": count > 0,
            "converging_mast_arm": (
                float(with_result.fixed_point.residual) <= parity.FIXED_POINT_CRITERION
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def harvest(capture_path: Path, output: Path, job_id: str) -> None:
    """Attach scheduler completion evidence to a successful raw capture."""
    record = json.loads(capture_path.read_text())
    completed = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in completed.stdout.splitlines() if line]
    job = next(row for row in rows if row[0] == job_id)
    exit_status = int(job[2].split(":", maxsplit=1)[0])
    record["execution"].update(
        {
            "scheduler_state": job[1],
            "scheduler_node": job[3],
            "exit_status": exit_status,
            "exit_marker": f"EXIT_MARKER={exit_status}",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    capture_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--capture", type=Path, required=True)
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    harvest_parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.output, args.figure_dir, args.cache)
    else:
        harvest(args.capture, args.output, args.job_id)


if __name__ == "__main__":
    main()
