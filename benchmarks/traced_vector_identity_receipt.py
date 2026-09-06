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
from scipy.constants import mu_0

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.biot.greens import hybrid_greens
from nova.equilibrium import ForwardProfile, reduced_newton
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/forward-solve-api/traced-identity"
DEFAULT_OUTPUT = OUTPUT_ROOT / "traced-vector-identity.json"
SHOT = 22086
SLICE_INDEX = 43
SOLOVEV_CASE = "weak-rotation-reactor-static"
TRIP_LIMIT = 4
RELATIVE_BOUND = 1.0e-12
P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
SOLOVEV_TOLERANCE = 1.0e-8
SOLOVEV_NEWTON_STEPS = 12


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
    response_cache, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER,
        response_carrier.DEFAULT_RECEIPT,
    )
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


def _solovev_terms() -> tuple[float, float, float]:
    """Return the analytic quartic, offset, and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return the analytic seed flux in webers."""
    alpha, offset, beta = _solovev_terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _solovev_wall(points: int = 61) -> tuple[np.ndarray, float]:
    """Return a material boundary on one analytic seed flux surface."""
    alpha, offset, beta = _solovev_terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2.0 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    wall = np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)]
    return wall, float(wall_flux)


def _green_block(
    target: np.ndarray, source: np.ndarray, section: float = 0.05
) -> np.ndarray:
    """Return total-flux coupling for one source and target set."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _flat_profile(amplitude: float):
    """Return a constant absolute source gradient."""

    def gradient(psi_norm):
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _edge_vanishing_profile(amplitude: float):
    """Return an absolute source gradient that vanishes at the boundary."""

    def gradient(psi_norm):
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


def _solovev_fixture() -> tuple[str, Any, jax.Array, Any, float | None, dict[str, Any]]:
    """Build the shared test-owned free-boundary Solov'ev fixture."""
    lattice = FluxLattice(np.linspace(0.6, 1.42, 25), np.linspace(-0.42, 0.42, 25))
    coordinate = lattice.coordinate
    wall, wall_flux = _solovev_wall()
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = seed_flux >= wall_flux

    angle = 2.0 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def build(core: DomainProfile, current: np.ndarray) -> ForwardProfile:
        return ForwardProfile.from_lattice(
            lattice,
            ForwardSource(
                core=core,
                boundary_field_function=BOUNDARY_FIELD_FUNCTION,
            ),
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = build(
        DomainProfile(
            p_prime=_flat_profile(P_PRIME),
            ff_prime=_flat_profile(FF_PRIME),
        ),
        np.zeros(CONDUCTORS),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_flux - coupling["plasma_to_grid"] @ cell_current,
        wall_seed - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    current = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]
    profile = build(
        DomainProfile(
            p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
            ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
        ),
        current,
    )
    return (
        "Solovev free-boundary fixture",
        profile.operator,
        seed,
        None,
        None,
        {
            "case": SOLOVEV_CASE,
            "construction": (
                "tests/test_steering_frames.py and "
                "tests/test_equilibrium_forward_solve.py free-boundary fixture"
            ),
            "grid_shape": [25, 25],
            "wall_points": len(wall),
            "conductor_count": CONDUCTORS,
            "newton_steps": SOLOVEV_NEWTON_STEPS,
            "tolerance": SOLOVEV_TOLERANCE,
        },
    )


def _program(
    operator: Any,
    initial: jax.Array,
    requested: Any,
    target: float | None,
):
    """Build one kernel set whose field can remain closed or become traced."""
    external = operator.external()
    target_value = None if target is None else jnp.asarray(target)
    coordinates = reduced_newton.reduced_coordinates(
        operator,
        initial,
        requested_class=requested,
        target_current=target_value,
    )
    return (
        external,
        coordinates,
        reduced_newton._reduced_kernels(
            operator, coordinates, external, requested, target_value
        ),
    )


def _drive(
    operator: Any,
    initial: jax.Array,
    requested: Any,
    target: float | None,
    *,
    active_set_steps: int,
    traced: bool,
    program: tuple[jax.Array, Any, dict[str, Any]],
    newton_steps: int,
    tolerance: float,
) -> dict[str, Any]:
    """Run one prefix of the reduced active-set loop in either field mode."""
    external, coordinates, kernels = program
    target_value = None if target is None else jnp.asarray(target)
    dynamic = (
        reduced_newton._bind_dynamic_arguments(
            kernels, external, target_value, requested
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
            reduced_newton._current_moments(operator, state, requested, target_value),
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
    target: float | None,
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


def prepare_only() -> dict[str, Any]:
    """Build both fixtures on the selected backend without entering a solve."""
    configure_dtypes()
    fixtures = [_mast_fixture(), _solovev_fixture()]
    rows = []
    for name, operator, initial, requested, target, metadata in fixtures:
        external = operator.external()
        jax.block_until_ready(external)
        rows.append(
            {
                "name": name,
                "initial_shape": list(np.shape(initial)),
                "external_shape": list(np.shape(external)),
                "requested_class": (
                    None if requested is None else int(np.asarray(requested))
                ),
                "target_current": target,
                "metadata": metadata,
            }
        )
    payload = {
        "status": "prepared",
        "platform": jax.default_backend(),
        "fixture_count": len(rows),
        "fixtures": rows,
    }
    print(json.dumps(_strict(payload), indent=2, sort_keys=True), flush=True)
    print("PREPARE_ONLY_EXIT=0", flush=True)
    return payload


def _require_measurement_host() -> None:
    """Require the declared one-H200 scheduler allocation."""
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"one H200 is required, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the betelgeuse partition is required")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the measurement requires eight requested CPUs")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("JAX_PLATFORMS=cuda,cpu must be set in the job body")


def run(output: Path) -> dict[str, Any]:
    """Run the H200 identity measurement and write its strict JSON receipt."""
    started = time.perf_counter()
    configure_dtypes()
    _require_measurement_host()
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
    parser.add_argument("--prepare-only", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
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
    if arguments.prepare_only:
        if arguments.command is not None:
            parser.error("--prepare-only does not accept a subcommand")
        prepare_only()
        return
    if arguments.command == "run":
        run(arguments.output)
        return
    if arguments.command is None:
        parser.error("choose --prepare-only, run, or submit")
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
