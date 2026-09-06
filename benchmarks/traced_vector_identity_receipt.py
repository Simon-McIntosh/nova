"""Measure reduced-solve flux identity for closed and traced external fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import coil_edit_latency as coil_edit
from benchmarks import efit_forward_parity_slice as parity
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile, reduced_newton
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/forward-solve-api/traced-identity"
DEFAULT_OUTPUT = OUTPUT_ROOT / "traced-vector-identity.json"
SHOT = 22086
SLICE_INDEX = 43
SOLOVEV_CASE = "weak-rotation-reactor-static"
SOLOVEV_CELLS = -110
TRIP_LIMIT = 4
RELATIVE_BOUND = 1.0e-12


def _sha256(path: Path) -> str:
    """Return a file digest for receipt provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    """Convert arrays and scalar wrappers into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _comparison(closed: jax.Array, traced: jax.Array) -> dict[str, Any]:
    """Return full-vector difference and exact-element counts."""
    left = np.asarray(closed, dtype=np.float64)
    right = np.asarray(traced, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError("closed and traced flux vectors have different shapes")
    matching_nonfinite = (
        (np.isnan(left) & np.isnan(right))
        | (np.isposinf(left) & np.isposinf(right))
        | (np.isneginf(left) & np.isneginf(right))
    )
    incompatible = (~np.isfinite(left) | ~np.isfinite(right)) & ~matching_nonfinite
    if np.any(incompatible):
        maximum_absolute = float("inf")
        maximum_relative = float("inf")
    else:
        finite = ~matching_nonfinite
        difference = np.abs(left[finite] - right[finite])
        maximum_absolute = float(np.max(difference, initial=0.0))
        scale = float(
            max(
                np.max(np.abs(left[finite]), initial=0.0),
                np.max(np.abs(right[finite]), initial=0.0),
                np.finfo(np.float64).tiny,
            )
        )
        maximum_relative = maximum_absolute / scale
    exact = left.tobytes() == right.tobytes()
    return {
        "maximum_absolute_difference_wb": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "relative_definition": (
            "max(abs(closed-traced)) / max(abs(closed), abs(traced), float64.tiny)"
        ),
        "bitwise_equal_elements": int(np.count_nonzero(left == right)),
        "element_count": int(left.size),
        "whole_vector_bitwise_equal": exact,
        "closed_vector_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "traced_vector_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
    }


def _mast_fixture() -> tuple[str, Any, jax.Array, Any, float, dict[str, Any]]:
    """Return the frozen MAST carrier and its stored external field."""
    response_cache, carrier = coil_edit._response_cache(coil_edit.DEFAULT_CARRIER)
    selected_rows = parity.select_slices_by_shot(parity.DECOMPOSITION_BANK)
    selected, qualification = next(
        (row, row_qualification)
        for row, row_qualification in selected_rows
        if int(row["shot"]) == SHOT and int(row["slice_index"]) == SLICE_INDEX
    )
    case, context = parity._mast_case_from_selection(
        SHOT_STORE, selected, qualification
    )
    passive_case, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    return (
        "MAST 22086/43",
        profile.operator,
        jnp.asarray(passive_case["state"]),
        TopologyClass.DIVERTED,
        abs(float(case["reference"]["plasma_current_a"])),
        {
            "machine": "MAST",
            "shot": SHOT,
            "slice_index": SLICE_INDEX,
            "carrier": carrier,
            "selection_qualification": qualification,
            "prescribed_current_policy": policy,
            "newton_steps": parity.NEWTON_STEPS,
            "tolerance": parity.FIXED_POINT_CRITERION,
        },
    )


def _solovev_fixture() -> tuple[str, Any, jax.Array, Any, float, dict[str, Any]]:
    """Construct the reduced Solov'ev carrier used by the certificate route."""
    carrier_case, source_case, exact = certificate._case(SOLOVEV_CASE)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        SOLOVEV_CELLS,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(SOLOVEV_CASE, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    coefficients = empty_operator.coupling_current_moments(exact_physical)
    internal = oracle_fixture._internal_flux_image(empty_operator, coefficients)
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        SOLOVEV_CASE, source_case, operator, exact_physical
    )
    seed, branch, seed_receipt = certificate._production_seed(
        profile, SOLOVEV_CASE, target_current, centroid, current_receipt
    )
    return (
        "Solovev weak-rotation reactor static",
        operator,
        jnp.asarray(seed),
        TopologyClass(branch),
        target_current,
        {
            "case": SOLOVEV_CASE,
            "requested_cells": SOLOVEV_CELLS,
            "realised_cells": len(machine.node),
            "seed": seed_receipt,
            "newton_steps": recovery.NEWTON_STEPS,
            "tolerance": recovery.FIXED_POINT_RESIDUAL_TOLERANCE,
        },
    )


def _program(operator: Any, initial: jax.Array, requested: Any, target: float):
    """Build one kernel set whose field can remain closed or become traced."""
    external = operator.external()
    coordinates = reduced_newton.reduced_coordinates(
        operator,
        initial,
        requested_class=requested,
        target_current=jnp.asarray(target),
    )
    return (
        external,
        coordinates,
        reduced_newton._reduced_kernels(
            operator, coordinates, external, requested, jnp.asarray(target)
        ),
    )


def _drive(
    operator: Any,
    initial: jax.Array,
    requested: Any,
    target: float,
    *,
    active_set_steps: int,
    traced: bool,
    program: tuple[jax.Array, Any, dict[str, Any]],
    newton_steps: int,
    tolerance: float,
) -> dict[str, Any]:
    """Run one prefix of the reduced active-set loop in either field mode."""
    external, coordinates, kernels = program
    dynamic = (
        reduced_newton._bind_dynamic_arguments(
            kernels, external, jnp.asarray(target), requested
        )
        if traced
        else kernels
    )
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(initial, requested), dtype=bool)
    )
    driven = reduced_newton._drive_trips(
        dynamic,
        initial,
        dynamic["initial_gather"](initial),
        shadow,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
        fused=True,
        scoring=reduced_newton.LADDER_SCORING,
        regather=lambda state: reduced_newton._gather(
            coordinates,
            reduced_newton._current_moments(
                operator, state, requested, jnp.asarray(target)
            ),
        ),
        dispatched_boundary=lambda *_arguments: (_ for _ in ()).throw(
            RuntimeError("the identity receipt requires the fused reduced boundary")
        ),
        stream=True,
    )
    jax.block_until_ready(driven["state"])
    return driven


def _case_receipt(
    name: str,
    operator: Any,
    initial: jax.Array,
    requested: Any,
    target: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Compare the first four prefixes and the full reduced solve."""
    program = _program(operator, initial, requested, target)
    common = {
        "program": program,
        "newton_steps": int(metadata["newton_steps"]),
        "tolerance": float(metadata["tolerance"]),
    }
    rows = []
    for trip in range(1, TRIP_LIMIT + 1):
        closed = _drive(
            operator,
            initial,
            requested,
            target,
            active_set_steps=trip,
            traced=False,
            **common,
        )
        traced = _drive(
            operator,
            initial,
            requested,
            target,
            active_set_steps=trip,
            traced=True,
            **common,
        )
        row = _comparison(closed["state"], traced["state"])
        row.update(
            {
                "trip": trip,
                "closed_active_set_trips": len(closed["residuals"]),
                "traced_active_set_trips": len(traced["residuals"]),
                "closed_terminal_residual": float(closed["terminal_residual"]),
                "traced_terminal_residual": float(traced["terminal_residual"]),
            }
        )
        rows.append(row)
    closed_terminal = _drive(
        operator,
        initial,
        requested,
        target,
        active_set_steps=reduced_newton.ACTIVE_SET_STEPS,
        traced=False,
        **common,
    )
    traced_terminal = _drive(
        operator,
        initial,
        requested,
        target,
        active_set_steps=reduced_newton.ACTIVE_SET_STEPS,
        traced=True,
        **common,
    )
    terminal = _comparison(closed_terminal["state"], traced_terminal["state"])
    terminal.update(
        {
            "closed_active_set_trips": len(closed_terminal["residuals"]),
            "traced_active_set_trips": len(traced_terminal["residuals"]),
            "closed_converged": bool(closed_terminal["converged"]),
            "traced_converged": bool(traced_terminal["converged"]),
            "closed_terminal_residual": float(closed_terminal["terminal_residual"]),
            "traced_terminal_residual": float(traced_terminal["terminal_residual"]),
        }
    )
    return {
        "name": name,
        "metadata": metadata,
        "first_four_trips": rows,
        "converged_state": terminal,
    }


def run(output: Path) -> dict[str, Any]:
    """Run the H200 identity measurement and write its strict JSON receipt."""
    started = time.perf_counter()
    configure_dtypes()
    coil_edit._require_measurement_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    cases = [_case_receipt(*_mast_fixture()), _case_receipt(*_solovev_fixture())]
    rows = [
        row
        for case in cases
        for row in [*case["first_four_trips"], case["converged_state"]]
    ]
    maximum_absolute = max(row["maximum_absolute_difference_wb"] for row in rows)
    maximum_relative = max(row["maximum_relative_difference"] for row in rows)
    first_defect = next(
        (
            {"case": case["name"], "trip": row.get("trip", "converged")}
            for case in cases
            for row in [*case["first_four_trips"], case["converged_state"]]
            if row["maximum_relative_difference"] > RELATIVE_BOUND
        ),
        None,
    )
    payload = {
        "schema": "nova.traced-vector-identity.v1",
        "measurement_state": "complete",
        "verdict": "ROUND_OFF" if first_defect is None else "DEFECT",
        "bound": {
            "relative": RELATIVE_BOUND,
            "interpretation_at_or_below": (
                "roundoff; traced path may be declared default"
            ),
            "interpretation_above": "defect; report the first trip where it appears",
        },
        "first_defect": first_defect,
        "headline": {
            "maximum_absolute_difference_wb": maximum_absolute,
            "maximum_relative_difference": maximum_relative,
            "compared_vectors": len(rows),
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "runtime": {
            "host": platform.node(),
            "device": jax.devices()[0].device_kind,
            "platform": jax.devices()[0].platform,
            "jax": jax.__version__,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "persistent_compilation_cache": cache.receipt(),
        "comparison": {
            "closed_definition": (
                "the reduced kernel default external field closed over at "
                "kernel construction"
            ),
            "traced_definition": (
                "the same external field passed as reduced_newton external_value"
            ),
            "route": "reduced_newton._drive_trips with fused trip boundary",
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "TRACED_VECTOR_IDENTITY "
        f"verdict={payload['verdict']} max_relative={maximum_relative:.17g}",
        flush=True,
    )
    print(f"EXIT_MARKER={0 if first_defect is None else 2}", flush=True)
    if first_defect is not None:
        raise SystemExit(2)
    return payload


def _sbatch(arguments: argparse.Namespace) -> str:
    """Return the H200 launch script for this bounded measurement."""
    log_directory = arguments.log_directory.resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=traced-vector-identity
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output={log_directory}/traced-vector-identity-%j.log
set -uo pipefail
export JAX_PLATFORMS=cuda,cpu
export TMPDIR=/tmp
export JAX_COMPILATION_CACHE_DIR={default_persistent_compilation_cache_root()}
cd {ROOT}
/home/ITER/mcintos/Code/nova/.venv/bin/python \
  benchmarks/traced_vector_identity_receipt.py run \
  --output {arguments.output.resolve()}
result=$?
echo EXIT_MARKER=$result
exit $result
"""


def main() -> None:
    """Run or submit the identity receipt."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    submit_parser.add_argument(
        "--log-directory",
        type=Path,
        default=Path(
            "/home/ITER/mcintos/.config/reckon/crew/runs/r-20260906T130303852320-fsa-traced-vector-identity/logs"
        ),
    )
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments.output)
        return
    completed = subprocess.run(
        ["sbatch", "--parsable"],
        input=_sbatch(arguments),
        text=True,
        capture_output=True,
        check=True,
    )
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
