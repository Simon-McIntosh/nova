"""Measure constant-closed versus traced prescribed-current solve differences."""

from __future__ import annotations

import argparse
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
from benchmarks import mast_response_carrier_warm as response_carrier
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/forward-solve-api/traced-current-bitwise.json"
SHOT = 22086
SLICE_INDEX = 43
FLUX_RELATIVE_BOUND = 1.0e-12


def _path_text(path: tuple[Any, ...]) -> str:
    """Return a stable dotted name for one JAX pytree leaf."""
    return jax.tree_util.keystr(path).lstrip(".")


def _is_flux_field(path: str) -> bool:
    """Return whether a receipt field carries a flux-valued quantity."""
    tokens = path.replace("[", ".").replace("]", ".").split(".")
    return any(token == "psi" or "flux" in token for token in tokens)


def _numeric_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return field-scale absolute and relative differences, matching NaNs."""
    left_float = left.astype(np.float64, copy=False)
    right_float = right.astype(np.float64, copy=False)
    matching_nonfinite = (
        (np.isnan(left_float) & np.isnan(right_float))
        | (np.isposinf(left_float) & np.isposinf(right_float))
        | (np.isneginf(left_float) & np.isneginf(right_float))
    )
    incompatible_nonfinite = (~np.isfinite(left_float) | ~np.isfinite(right_float)) & (
        ~matching_nonfinite
    )
    if np.any(incompatible_nonfinite):
        return float("inf"), float("inf")
    finite = ~matching_nonfinite
    if not np.any(finite):
        return 0.0, 0.0
    absolute = np.abs(left_float[finite] - right_float[finite])
    maximum_absolute = float(np.max(absolute, initial=0.0))
    scale = float(
        max(
            np.max(np.abs(left_float[finite]), initial=0.0),
            np.max(np.abs(right_float[finite]), initial=0.0),
            np.finfo(np.float64).tiny,
        )
    )
    return maximum_absolute, maximum_absolute / scale


def _field_differences(left: Any, right: Any) -> list[dict[str, Any]]:
    """Compare every aligned receipt leaf without dropping negative evidence."""
    left_rows, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_rows, right_tree = jax.tree_util.tree_flatten_with_path(right)
    if left_tree != right_tree:
        raise RuntimeError("the omitted and traced receipts have different pytrees")
    records = []
    for (left_path, left_leaf), (right_path, right_leaf) in zip(
        left_rows, right_rows, strict=True
    ):
        path = _path_text(left_path)
        if path != _path_text(right_path):
            raise RuntimeError("the omitted and traced receipt paths differ")
        left_array = np.ascontiguousarray(np.asarray(left_leaf))
        right_array = np.ascontiguousarray(np.asarray(right_leaf))
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
        ):
            raise RuntimeError(f"receipt field {path} changed shape or dtype")
        numeric = np.issubdtype(left_array.dtype, np.number)
        if numeric:
            maximum_absolute, maximum_relative = _numeric_difference(
                left_array, right_array
            )
        else:
            maximum_absolute = None
            maximum_relative = None
        flux_field = numeric and _is_flux_field(path)
        records.append(
            {
                "path": path,
                "shape": list(left_array.shape),
                "dtype": left_array.dtype.str,
                "element_count": int(left_array.size),
                "omitted_sha256": coil_edit._sha256_bytes(left_array.tobytes()),
                "traced_sha256": coil_edit._sha256_bytes(right_array.tobytes()),
                "bit_identical": bool(left_array.tobytes() == right_array.tobytes()),
                "maximum_absolute_difference": maximum_absolute,
                "maximum_relative_difference": maximum_relative,
                "relative_definition": (
                    "max(abs(omitted-traced)) / "
                    "max(max(abs(omitted)), max(abs(traced)), float64.tiny)"
                    if numeric
                    else None
                ),
                "flux_relative_bound_applies": flux_field,
                "flux_relative_bound": FLUX_RELATIVE_BOUND if flux_field else None,
                "flux_relative_bound_passes": (
                    bool(maximum_relative <= FLUX_RELATIVE_BOUND)
                    if flux_field
                    else None
                ),
            }
        )
    return records


def _prepare_case(carrier_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load the persisted response and the selected frozen-six solve arm."""
    response_cache, carrier = coil_edit._response_cache(carrier_path)
    selected_rows = parity.select_slices_by_shot(parity.DECOMPOSITION_BANK)
    selected, qualification = next(
        (row, qualification)
        for row, qualification in selected_rows
        if int(row["shot"]) == SHOT and int(row["slice_index"]) == SLICE_INDEX
    )
    case, context = parity._mast_case_from_selection(
        SHOT_STORE,
        selected,
        qualification,
    )
    passive_case, profile, policy = parity._passive_inclusive_case(
        case,
        context,
        response_cache,
    )
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None or prescribed.current.shape != (101,):
        raise RuntimeError("the operator does not hold the complete current vector")
    if not policy["response_matrix_reused"]:
        raise RuntimeError("the persisted response carrier was not reused")
    return (
        profile,
        {
            "initial": jnp.asarray(passive_case["state"]),
            "prescribed_current": jnp.asarray(prescribed.current),
            "target_current": abs(float(case["reference"]["plasma_current_a"])),
            "reference": case["reference"],
            "qualification": qualification,
            "policy": policy,
        },
        carrier,
    )


def _solve(profile: Any, initial: jax.Array, target_current: float) -> Any:
    """Return the stored-current solve with the prescribed vector closed over."""
    return profile.solve_branch(
        initial,
        TopologyClass.DIVERTED,
        route="newton_krylov",
        target_current=target_current,
        tolerance=parity.FIXED_POINT_CRITERION,
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
        warmup=parity.WARMUP_SWEEPS,
        relaxation=parity.RELAXATION,
        step_cap=parity.STEP_CAP,
    )


def _solve_traced(
    profile: Any,
    initial: jax.Array,
    prescribed_current: jax.Array,
    target_current: float,
) -> Any:
    """Return the same solve with the identical vector supplied as traced data."""
    return profile.solve_branch(
        initial,
        TopologyClass.DIVERTED,
        route="newton_krylov",
        prescribed_current=prescribed_current,
        target_current=target_current,
        tolerance=parity.FIXED_POINT_CRITERION,
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
        warmup=parity.WARMUP_SWEEPS,
        relaxation=parity.RELAXATION,
        step_cap=parity.STEP_CAP,
    )


def _terminal_summary(branch: Any) -> dict[str, Any]:
    """Return the terminal solver facts required beside field differences."""
    fixed_point = branch.equilibrium.fixed_point
    return {
        "converged": bool(np.asarray(branch.converged)),
        "terminal_residual": float(np.asarray(branch.residual)),
        "trip_count": int(np.asarray(fixed_point.active_set_iterations)),
        "fixed_iteration_count": int(np.asarray(branch.iterations)),
        "termination": FixedPointTerminationReason(
            int(np.asarray(fixed_point.termination_reason))
        ).name.lower(),
    }


def run(output: Path, carrier_path: Path) -> dict[str, Any]:
    """Run both H200 solves and write the complete per-field comparison."""
    started = time.perf_counter()
    configure_dtypes()
    coil_edit._require_measurement_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    profile, prepared, carrier = _prepare_case(carrier_path)
    initial = prepared["initial"]
    prescribed_current = prepared["prescribed_current"]
    target_current = prepared["target_current"]

    omitted_jit = jax.jit(lambda state: _solve(profile, state, target_current))
    traced_jit = jax.jit(
        lambda state, current: _solve_traced(profile, state, current, target_current)
    )
    omitted_started = time.perf_counter()
    omitted = omitted_jit(initial)
    jax.block_until_ready(omitted)
    omitted_seconds = time.perf_counter() - omitted_started
    print(f"OMITTED_DONE seconds={omitted_seconds:.6f}", flush=True)
    traced_started = time.perf_counter()
    traced = traced_jit(initial, prescribed_current)
    jax.block_until_ready(traced)
    traced_seconds = time.perf_counter() - traced_started
    print(f"TRACED_DONE seconds={traced_seconds:.6f}", flush=True)

    fields = _field_differences(omitted, traced)
    flux_fields = [row for row in fields if row["flux_relative_bound_applies"]]
    if not flux_fields:
        raise RuntimeError("the receipt comparison found no flux fields")
    maximum_flux_absolute = max(
        row["maximum_absolute_difference"] for row in flux_fields
    )
    maximum_flux_relative = max(
        row["maximum_relative_difference"] for row in flux_fields
    )
    bit_identical = all(row["bit_identical"] for row in fields)
    bound_passes = bool(maximum_flux_relative <= FLUX_RELATIVE_BOUND)
    receipt = {
        "schema": "nova.traced-current-bitwise",
        "measurement_state": "complete",
        "verdict": "ROUND_OFF" if bound_passes else "DEFECT",
        "preregistered_gate": {
            "quantity": "maximum field-scale relative difference over flux fields",
            "bound": FLUX_RELATIVE_BOUND,
            "pass_interpretation": "roundoff",
            "fail_interpretation": "defect",
            "passes": bound_passes,
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": coil_edit._sha256(Path(__file__)),
        },
        "scheduler": coil_edit._scheduler(),
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "device": jax.devices()[0].device_kind,
            "platform": jax.devices()[0].platform,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "tmpdir": os.environ.get("TMPDIR"),
            "elapsed_seconds": time.perf_counter() - started,
            "omitted_wall_seconds": omitted_seconds,
            "traced_wall_seconds": traced_seconds,
            "exit_marker": 0 if bound_passes else 2,
        },
        "persistent_compilation_cache": cache.receipt(),
        "carrier": carrier,
        "case": {
            "machine": "MAST",
            "shot": SHOT,
            "slice_index": SLICE_INDEX,
            "time_s": float(prepared["reference"]["time_s"]),
            "frozen_arm": "current-constrained reference-seeded diverted branch",
            "route": "newton_krylov",
            "target_current_a": target_current,
            "stored_circuit_count": int(prescribed_current.size),
            "selection_qualification": prepared["qualification"],
            "prescribed_current_policy": prepared["policy"],
        },
        "comparison": {
            "omitted_definition": (
                "stored PrescribedCurrentField.current closed into the jitted trace"
            ),
            "traced_definition": (
                "the identical stored vector passed through prescribed_current"
            ),
            "prescribed_vector_bit_identical_to_stored": bool(
                np.asarray(prescribed_current).tobytes()
                == np.asarray(
                    profile.operator.prescribed_current_field.current
                ).tobytes()
            ),
            "omitted_receipt_sha256": coil_edit._tree_digest(omitted),
            "traced_receipt_sha256": coil_edit._tree_digest(traced),
            "whole_receipt_bit_identical": bit_identical,
            "receipt_field_count": len(fields),
            "differing_receipt_field_count": sum(
                not row["bit_identical"] for row in fields
            ),
            "flux_field_count": len(flux_fields),
            "maximum_flux_absolute_difference": maximum_flux_absolute,
            "maximum_flux_relative_difference": maximum_flux_relative,
        },
        "terminal": {
            "omitted": _terminal_summary(omitted),
            "traced": _terminal_summary(traced),
        },
        "receipt_fields": fields,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "COMPARISON_DONE "
        f"bit_identical={bit_identical} "
        f"max_flux_absolute={maximum_flux_absolute:.17g} "
        f"max_flux_relative={maximum_flux_relative:.17g} "
        f"verdict={receipt['verdict']}",
        flush=True,
    )
    print(f"EXIT_MARKER={receipt['runtime']['exit_marker']}", flush=True)
    if not bound_passes:
        raise SystemExit(2)
    return receipt


def _sbatch_script(arguments: argparse.Namespace) -> str:
    """Return the single-H200 launch body for the frozen comparison."""
    log_directory = arguments.log_directory.resolve()
    worktree = ROOT.resolve()
    environment = Path("/home/ITER/mcintos/Code/nova/.venv")
    command = (
        f"UV_PROJECT_ENVIRONMENT={environment} PYTHONPATH={worktree} "
        "uv run --no-sync python benchmarks/traced_current_bitwise.py run "
        f"--carrier {arguments.carrier.resolve()} "
        f"--output {arguments.output.resolve()}"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=traced-current-bitwise
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output={log_directory}/traced-current-bitwise-%j.log
set -uo pipefail
export JAX_PLATFORMS=cuda,cpu
export TMPDIR=/tmp
cd {worktree}
{command}
result=$?
echo EXIT_MARKER=$result
exit $result
"""


def _submit(arguments: argparse.Namespace) -> None:
    """Submit the measurement and print its scheduler identity."""
    arguments.log_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["sbatch", "--parsable"],
        input=_sbatch_script(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())


def _harvest(output: Path) -> None:
    """Print the compact quantitative result from a completed receipt."""
    receipt = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "job_id": receipt["scheduler"]["job_id"],
                "node": receipt["scheduler"]["node"],
                "whole_receipt_bit_identical": receipt["comparison"][
                    "whole_receipt_bit_identical"
                ],
                "differing_receipt_field_count": receipt["comparison"][
                    "differing_receipt_field_count"
                ],
                "maximum_flux_absolute_difference": receipt["comparison"][
                    "maximum_flux_absolute_difference"
                ],
                "maximum_flux_relative_difference": receipt["comparison"][
                    "maximum_flux_relative_difference"
                ],
                "terminal": receipt["terminal"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    for name in ("sbatch", "submit"):
        job_parser = subparsers.add_parser(name)
        job_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        job_parser.add_argument(
            "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
        )
        job_parser.add_argument(
            "--log-directory",
            type=Path,
            default=Path(
                "/home/ITER/mcintos/.config/reckon/crew/runs/"
                "r-20260902T180320449650-fsa-traced-current-bitwise/logs"
            ),
        )
    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments.output, arguments.carrier)
    elif arguments.command == "sbatch":
        print(_sbatch_script(arguments), end="")
    elif arguments.command == "submit":
        _submit(arguments)
    else:
        _harvest(arguments.output)


if __name__ == "__main__":
    main()
