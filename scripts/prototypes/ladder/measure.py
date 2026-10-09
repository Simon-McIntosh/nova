"""Measure closed-form topology reads with durable per-row receipts."""

# Precision must be configured before importing array-valued module defaults.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from tests.equilibrium import test_topology_read as oracle_fixture


assert jax.config.jax_enable_x64 is True

ROOT = Path(__file__).resolve().parents[3]
BASE_ROWS = ROOT / "docs/figures/converged-forward-solve/cfs-topology-read"


def _identity():
    return {
        "revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cwd": str(Path.cwd().resolve()),
        "module": str(Path(topology.__file__).resolve()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(jax.devices()[0]),
    }


def _measure(kind: str, cells: int, arm: str) -> dict:
    oracle, field, wall, axis, saddle = oracle_fixture._analytic_inputs(kind)
    geometry = oracle_fixture._realised_hex_geometry(wall, cells)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996
        if arm == "A" and saddle is not None
        else 0.0,
        normal_form_pitch_floor=1.5 if arm == "A" else 0.0,
    )
    if arm == "B":
        # The read always uses straight-ray support in the saddle's owner cell.
        # Zero radius and zero higher derivatives are its simplest exposed form.
        original = topology._curvature_derivatives

        def zero_curvature(candidate, point, local_pitch):
            third, fourth = original(candidate, point, local_pitch)
            return jax.numpy.zeros_like(third), jax.numpy.zeros_like(fourth)

        topology._curvature_derivatives = zero_curvature
    try:
        convention = topology.TopologyConvention.from_cocos(17, 1.0)
        evaluate = jax.jit(topology.read)
        started = perf_counter()
        executable = evaluate.lower(field, geometry, convention, policy).compile()
        compile_wall = perf_counter() - started
        started = perf_counter()
        result = executable(field, geometry, convention, policy)
        jax.block_until_ready(result)
        warm_wall = perf_counter() - started
        started = perf_counter()
        result = executable(field, geometry, convention, policy)
        jax.block_until_ready(result)
        warm_wall = min(warm_wall, perf_counter() - started)
    finally:
        if arm == "B":
            topology._curvature_derivatives = original
    reference, uncertainty = oracle_fixture._reference_cell_fractions(
        kind, oracle, geometry
    )
    area = np.asarray(geometry.full_area)
    error = np.abs(np.asarray(result.membership) - reference) * area / np.median(area)
    near = (
        np.zeros(cells, dtype=bool)
        if saddle is None
        else np.linalg.norm(np.asarray(geometry.centre) - saddle, axis=1)
        <= max(0.054806712567833996, 1.5 * pitch)
    )
    x_points = np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
    x_error = (
        None
        if saddle is None or len(x_points) != 1
        else float(np.linalg.norm(x_points[0] - saddle))
    )
    axis_error = float(np.linalg.norm(np.asarray(result.axis) - axis))
    contact_error = (
        None
        if saddle is not None
        else float(
            np.linalg.norm(
                np.asarray(result.boundary)
                - np.asarray((oracle.boundary_midplane_radii()[1], 0.0))
            )
        )
    )
    memory = executable.memory_analysis()
    row = {
        **_identity(),
        "case": kind,
        "arm": arm,
        "cells": cells,
        "pitch_m": pitch,
        "qualified": bool(result.qualified),
        "valid": bool(result.valid),
        "reason": int(result.reason),
        "smooth_membership_error": float(np.max(error[~near])),
        "saddle_membership_error": float(np.max(error[near])) if near.any() else None,
        "reference_quadrature_error": float(np.max(uncertainty)),
        "axis_error_m": axis_error,
        "axis_error_pitches": axis_error / pitch,
        "x_error_m": x_error,
        "x_error_pitches": None if x_error is None else x_error / pitch,
        "limiter_contact_error_m": contact_error,
        "cold_compile_seconds": compile_wall,
        "warm_execute_seconds": warm_wall,
        "device_temp_bytes": memory.temp_size_in_bytes if memory else None,
        "device_output_bytes": memory.output_size_in_bytes if memory else None,
        "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "normal_form_owner_cells": int(
            np.count_nonzero(np.asarray(result.normal_form_cells))
        ),
    }
    if cells == 132 and arm == "A":
        baseline = json.loads(
            (BASE_ROWS / f"{kind}-moving-floor-current-rows.json").read_text()
        )[0]
        independent = oracle_fixture._symmetric_difference_measure(
            kind, oracle, geometry, result
        )
        baseline_near = np.asarray(result.normal_form_cells)
        measured_smooth = float(np.max(independent["normalised_error"][~baseline_near]))
        measured_saddle = (
            float(np.max(independent["normalised_error"][baseline_near]))
            if baseline_near.any()
            else None
        )
        row["baseline_receipt"] = str(
            BASE_ROWS / f"{kind}-moving-floor-current-rows.json"
        )
        row["baseline_smooth_error"] = baseline["smooth_max"]
        row["baseline_saddle_error"] = baseline["saddle_neighbourhood_max"]
        row["reproduced_smooth_error"] = measured_smooth
        row["reproduced_saddle_error"] = measured_saddle
        row["baseline_pitch_m"] = baseline["pitch"]
        assert abs(pitch - baseline["pitch"]) < 1e-12
        assert abs(measured_smooth - baseline["smooth_max"]) < 5e-7
        if measured_saddle is not None:
            assert abs(measured_saddle - baseline["saddle_neighbourhood_max"]) < 5e-7
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = _measure(args.kind, args.cells, args.arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print("LADDER_ROW " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
