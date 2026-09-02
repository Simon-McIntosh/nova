"""Bank the absolute Solovev forward-solve accuracy certificate.

The driver is intentionally an isolated-rung program.  CPU and accelerator
jobs write disjoint part receipts and figures; aggregation only reads those
artifacts.  Every terminal state is retained, but fitted orders are separated
by the terminal residual qualification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from time import perf_counter
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from scipy import stats

from benchmarks.analytic_operator_ladder import _fit_order, _region_masks
from benchmarks.split_fit_jump_field import (
    BOUNDARY_BAND_PITCHES,
    _distance_to_boundary,
    _lcfs,
    _polynomial_flux,
    _polynomial_gradient,
    _polynomial_hessian,
)
from nova.equilibrium import ColdSeedConstruction, ForwardProfile, SaddleSeedGeometry
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.analytic_oracle_fixtures.reduced_oracle import measure_reduced_oracle
from scripts.dual_basin_fixtures.build_diverted_fixture import (
    AXIS_M,
    X_POINT_M,
    _solve_coefficients,
)
from scripts.oracle_rebaseline import measure as recovery
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases
from tests.test_solovev_recovery_gates import LOCKED_RECOVERY_BOUNDS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
)
SEED_CONTROL_OUTPUT = (
    ROOT / "docs/figures/gs-absolute-accuracy/certificate-seed-control.json"
)
FIGURE_ROOT = ROOT / "docs/figures/gs-absolute-accuracy/solovev"
PART_ROOT = FIGURE_ROOT / "production-route-parts"
DIAGNOSTIC_ROOT = FIGURE_ROOT / "production-route-diagnostics"
REQUESTED_CELLS = (-110, -300, -500, -1000)
CASE_NAMES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-jump-bearing",
)
TERMINAL_RESIDUAL_BOUND = float(LOCKED_RECOVERY_BOUNDS["fixed_point_residual"])
THEORETICAL_ORDER = 2.0
NORM_FIELDS = ("psi", "gradient", "hessian")
NORM_REGIONS = ("whole_domain", "two_pitch_boundary_band")
NORM_STATISTICS = ("sup", "rms")
HEARTBEAT_SECONDS = 30.0
REUSE_MAP_ROWS = (
    "tests/rotating_equilibrium_references.py::reference_cases",
    "benchmarks/split_fit_jump_field.py::_polynomial_gradient,_polynomial_hessian,_lcfs",
    "benchmarks/analytic_operator_ladder.py::_regional_norms,_fit_order",
    "scripts/analytic_oracle_fixtures/measure.py::cached_machine,forward_operator,exact_current_moments",
    "scripts/analytic_oracle_fixtures/reduced_oracle.py::measure_reduced_oracle",
    "tests/test_solovev_recovery_gates.py::LOCKED_RECOVERY_BOUNDS",
)
RECOVERY_GATE_LANE_PERFORMANCE = {
    "classification": "lane-performance qualification, not certificate accuracy",
    "registered_wall_bound_seconds": 60.0,
    "last_green_wall_seconds_approx": 35.0,
    "path_change_since_last_green": {
        "revision": "32f16b3b",
        "change": "restored the hex-carrier topology read",
        "only_change_on_reduced_oracle_read_path": True,
    },
    "attempts": [
        {
            "slurm_job_id": "1261036",
            "node": "98dci4-clu-2009",
            "cpu_count": 1,
            "threaded_settings": {"enabled": False},
            "wall_seconds": 87.43643704202259,
            "wall_bound_seconds": 60.0,
            "ratio_to_last_green": 2.4981839154863595,
            "numerical_forcing_and_fixed_point_assertions": "passed",
            "timing_assertion": "failed",
        },
        {
            "slurm_job_id": "1261037",
            "node": "98dci4-clu-2009",
            "cpu_count": 16,
            "threaded_settings": {
                "enabled": True,
                "xla_flags": (
                    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16"
                ),
                "omp_num_threads": 16,
                "openblas_num_threads": 16,
                "mkl_num_threads": 16,
                "numexpr_num_threads": 16,
            },
            "wall_seconds": 77.55223163502524,
            "wall_bound_seconds": 60.0,
            "ratio_to_last_green": 2.2157780467150067,
            "numerical_forcing_and_fixed_point_assertions": "passed",
            "timing_assertion": "failed",
        },
    ],
    "finding": (
        "the restored reduced-oracle topology read is 2.2 to 2.5 times slower "
        "than the approximately 35-second last-green lane"
    ),
    "disposition": "performance regression handed to the topology-read owner",
}


@contextmanager
def _timed_stage(
    name: str,
    timings: dict[str, float],
    *,
    case_name: str,
    requested_cells: int,
) -> Iterator[None]:
    """Emit flushed stage boundaries and heartbeats for scheduler harvesting."""

    started = perf_counter()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(HEARTBEAT_SECONDS):
            print(
                f"SOLOVEV_STAGE_HEARTBEAT case={case_name} "
                f"requested_cells={requested_cells} stage={name} "
                f"elapsed_seconds={perf_counter() - started:.3f}",
                flush=True,
            )

    print(
        f"SOLOVEV_STAGE_BEGIN case={case_name} requested_cells={requested_cells} "
        f"stage={name}",
        flush=True,
    )
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        timings[name] = elapsed
        stopped.set()
        worker.join()
        print(
            f"SOLOVEV_STAGE_END case={case_name} requested_cells={requested_cells} "
            f"stage={name} elapsed_seconds={elapsed:.6f}",
            flush=True,
        )


def _strict(value: Any) -> Any:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _slug(value: int) -> str:
    return "reduced" if value == -110 else f"cells-{abs(value)}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return PART_ROOT / f"{case_name}-production-route-{_slug(requested_cells)}.json"


def _figure_path(case_name: str, requested_cells: int) -> Path:
    return FIGURE_ROOT / f"{case_name}-production-route-{_slug(requested_cells)}.png"


def _diagnostic_path(case_name: str) -> Path:
    return DIAGNOSTIC_ROOT / f"{case_name}-reduced-nan-census.json"


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_default_backend": jax.default_backend(),
        "precision": "float64",
    }


def _thread_settings() -> dict[str, Any]:
    names = (
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name.lower(): os.environ.get(name) for name in names}


def _trip_residual_history(history: Any) -> list[dict[str, Any]]:
    trace = np.asarray(history.trace, dtype=np.float64)
    stride = recovery.KRYLOV_ITERATIONS + 2
    trips = []
    for trip_index in range(recovery.NEWTON_STEPS):
        start = trip_index * stride
        stop = start + stride
        local_indices = np.flatnonzero(np.isfinite(trace[start:stop]))
        if not len(local_indices):
            continue
        absolute_indices = start + local_indices
        trips.append(
            {
                "trip": trip_index + 1,
                "trace_indices": absolute_indices.tolist(),
                "residuals": trace[absolute_indices].tolist(),
            }
        )
    return trips


def _seed_control_result(
    *,
    name: str,
    seed: np.ndarray,
    seed_receipt: dict[str, Any],
    map_fn: Any,
    operator: Any,
    oracle_state: np.ndarray,
    axis_reference: np.ndarray,
    cell_count: int,
) -> dict[str, Any]:
    initial_residual = recovery._relative_map_residual(map_fn, seed)
    started = perf_counter()
    history = recovery._solve(map_fn, seed)
    solve_seconds = perf_counter() - started
    terminal_state = np.asarray(history.state, dtype=np.float64)
    terminal_residual = float(history.residual)
    error = terminal_state[:cell_count] - oracle_state[:cell_count]
    finite_error = error[np.isfinite(error)]
    topology = _topology(operator, terminal_state)
    qualified = bool(
        np.isfinite(terminal_residual) and terminal_residual <= TERMINAL_RESIDUAL_BOUND
    )
    if not np.all(np.isfinite(terminal_state)):
        qualification_reason = "nonfinite_terminal_state"
    elif not np.isfinite(terminal_residual):
        qualification_reason = "nonfinite_terminal_residual"
    elif qualified:
        qualification_reason = "fixed_point_residual_within_qualification_bound"
    else:
        qualification_reason = (
            "fixed_point_residual_above_qualification_bound_after_iteration_budget"
        )
    solver_termination = recovery.fixed_point.FixedPointTerminationReason(
        int(history.termination_reason)
    ).name.lower()
    return {
        "name": name,
        "seed": seed_receipt,
        "initial_relative_residual": initial_residual,
        "per_trip_residual_history": _trip_residual_history(history),
        "trip_count": len(_trip_residual_history(history)),
        "terminal_relative_residual": (
            terminal_residual if np.isfinite(terminal_residual) else None
        ),
        "qualification_bound": TERMINAL_RESIDUAL_BOUND,
        "qualification": "qualified" if qualified else "unqualified",
        "termination": solver_termination,
        "qualification_reason": qualification_reason,
        "solve_wall_seconds": solve_seconds,
        "whole_domain_psi_error": {
            "sup_wb": (
                float(np.max(np.abs(finite_error))) if len(finite_error) else None
            ),
            "rms_wb": (
                float(np.sqrt(np.mean(finite_error**2))) if len(finite_error) else None
            ),
            "finite_cells": int(len(finite_error)),
            "total_cells": cell_count,
        },
        "axis_position_error_m": (
            float(
                np.linalg.norm(
                    np.asarray(topology["axis_rz_m"], dtype=np.float64) - axis_reference
                )
            )
            if topology["axis_rz_m"] is not None
            else None
        ),
        "terminal_axis_rz_m": topology["axis_rz_m"],
    }


def _seed_control(output: Path = SEED_CONTROL_OUTPUT) -> dict[str, Any]:
    started = perf_counter()
    configure_dtypes()
    compilation_cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    case_name = "strong-rotation-compact-static"
    requested_cells = -500
    carrier_case, source_case, exact = _case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = _exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    production_seed, _moment_image, production_receipt = recovery._moment_seed(
        source_case, machine, operator
    )
    near_root_seed = np.asarray(oracle_state, dtype=np.float64)
    near_root_receipt = {
        "kind": "closed-form-near-root-control",
        "construction": (
            "the shipped analytic fixture closed-form state on the identical carrier; "
            "only the solve initial state differs from the production arm"
        ),
        "state_sha256_binary64": hashlib.sha256(near_root_seed.tobytes()).hexdigest(),
    }
    map_fn = operator.flux_map()
    arms = [
        _seed_control_result(
            name="production_moment_seed",
            seed=np.asarray(production_seed, dtype=np.float64),
            seed_receipt=production_receipt,
            map_fn=map_fn,
            operator=operator,
            oracle_state=oracle_state,
            axis_reference=np.asarray(exact.magnetic_axis, dtype=np.float64),
            cell_count=len(machine.node),
        ),
        _seed_control_result(
            name="closed_form_near_root_seed",
            seed=near_root_seed,
            seed_receipt=near_root_receipt,
            map_fn=map_fn,
            operator=operator,
            oracle_state=oracle_state,
            axis_reference=np.asarray(exact.magnetic_axis, dtype=np.float64),
            cell_count=len(machine.node),
        ),
    ]
    by_name = {arm["name"]: arm for arm in arms}
    production_qualified = by_name["production_moment_seed"]["qualification"]
    control_qualified = by_name["closed_form_near_root_seed"]["qualification"]
    if production_qualified == "unqualified" and control_qualified == "qualified":
        verdict = (
            "Driver sound: the near-root control converges while basin entry fails "
            "from the production moment seed."
        )
        classification = "driver_sound_production_seed_basin_entry_failure"
    elif production_qualified == "unqualified":
        verdict = "Driver suspect: neither seed converges to the registered bound."
        classification = "driver_suspect"
    else:
        verdict = "Certificate wrong: both seeds converge to the registered bound."
        classification = "certificate_wrong"
    receipt = {
        "schema": {
            "name": "nova-solovev-certificate-seed-control",
            "version": 1,
            "required": [
                "case",
                "requested_cells",
                "realised_cells",
                "lane",
                "solver",
                "arms",
                "verdict",
            ],
            "arm_required": [
                "initial_relative_residual",
                "per_trip_residual_history",
                "terminal_relative_residual",
                "trip_count",
                "termination",
                "whole_domain_psi_error",
                "axis_position_error_m",
            ],
        },
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "source_revision": _source_revision(),
        "solver_source_modified": False,
        "lane": {
            **_lane(),
            "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "threaded_settings": _thread_settings(),
            "persistent_compilation_cache": compilation_cache.receipt(),
            "wall_seconds": perf_counter() - started,
            "exit_marker": "SEED_CONTROL_EXIT=0",
        },
        "solver": {
            "route": "production undamped Newton-Krylov machinery",
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "qualification_bound": TERMINAL_RESIDUAL_BOUND,
        },
        "arms": arms,
        "verdict": {"classification": classification, "sentence": verdict},
    }
    _validate_seed_control(receipt)
    _write_json(output, receipt)
    return receipt


def _validate_seed_control(receipt: dict[str, Any]) -> None:
    for name in receipt["schema"]["required"]:
        if name not in receipt:
            raise RuntimeError(f"seed control is missing {name}")
    if len(receipt["arms"]) != 2:
        raise RuntimeError("seed control must contain exactly two arms")
    for arm in receipt["arms"]:
        for name in receipt["schema"]["arm_required"]:
            if name not in arm:
                raise RuntimeError(f"seed control arm is missing {name}")
    if receipt["lane"]["slurm_job_id"] is None:
        raise RuntimeError("seed control must be measured under SLURM")


def _diverted_source(coefficients: np.ndarray) -> RotatingEquilibrium:
    axis_flux = float(_polynomial_flux(AXIS_M[None, :], coefficients)[0])
    carrier = oracle_fixture.analytic_case()
    return RotatingEquilibrium(
        name="diverted-jump-bearing-source",
        major_radius=float(AXIS_M[0]),
        axis_flux=axis_flux,
        pressure_coefficient=float(-coefficients[0] / np.pi),
        field_coefficient=float(-coefficients[1] / (2.0 * np.pi)),
        rotation_parameter=0.0,
        boundary_f=carrier.boundary_f,
        axis_temperature=carrier.axis_temperature,
        boundary_temperature=carrier.boundary_temperature,
        mean_particle_mass=carrier.mean_particle_mass,
    )


def _diverted_seed(
    source_case: RotatingEquilibrium, machine: Any, operator: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the production axis-saddle cold seed for the diverted basin."""
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    seed_radius = 0.9 * float(np.linalg.norm(X_POINT_M - AXIS_M))
    supported = np.linalg.norm(machine.node - AXIS_M, axis=1) < seed_radius
    cell_current = (
        source_case.toroidal_current_density(machine.node[:, 0], machine.node[:, 1])
        * machine.area
        * supported
    )
    total_current = float(np.sum(cell_current))
    centroid = np.sum(machine.node * cell_current[:, None], axis=0) / total_current
    geometry = SaddleSeedGeometry(tuple(AXIS_M), tuple(X_POINT_M))
    portfolio = profile.cold_seed_portfolio(
        total_current,
        centroid,
        diverted_geometry=geometry,
    )
    branch = int(TopologyClass.DIVERTED)
    branches = portfolio.branches
    seed = np.asarray(branches.flux[branch], dtype=np.float64)
    receipt = {
        "kind": "production-current-moment-axis-saddle-geometry",
        "independent_of_closed_form_state": True,
        "closed_form_flux_samples_used": False,
        "closed_form_coupling_image_used": False,
        "fixture_exterior_boundary_condition_used": True,
        "declared_axis_m": AXIS_M.tolist(),
        "declared_saddle_m": X_POINT_M.tolist(),
        "plasma_current_a": total_current,
        "current_centroid_m": centroid.tolist(),
        "seed_support_radius_m": seed_radius,
        "supported_cell_count": int(np.count_nonzero(cell_current)),
        "production_seed_radius_m": float(branches.radius[branch]),
        "anchor_available": bool(branches.anchor_available[branch]),
        "anchor_m": np.asarray(branches.anchor[branch], dtype=np.float64).tolist(),
        "stored_flux_samples_used": bool(branches.stored_flux_samples_used[branch]),
        "state_sha256_binary64": hashlib.sha256(seed.tobytes()).hexdigest(),
    }
    return seed, receipt


def _closed_form_current_target(
    case_name: str,
    source_case: RotatingEquilibrium,
    operator: Any,
    exact_physical: Any,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Return the declared current and centroid used by the production seed."""

    if case_name != "diverted-jump-bearing":
        current, centroid, receipt = recovery._aggregate_current_moment(source_case)
        return float(current), np.asarray(centroid, dtype=np.float64), receipt

    cell_current = np.asarray(exact_physical.cell_current, dtype=np.float64)
    radial_moment = np.asarray(exact_physical.radial_moment, dtype=np.float64)
    vertical_moment = np.asarray(exact_physical.vertical_moment, dtype=np.float64)
    centres = np.asarray(
        operator.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    total = float(np.sum(cell_current))
    centroid = np.array(
        [
            np.sum(cell_current * centres[:, 0] + radial_moment) / total,
            np.sum(cell_current * centres[:, 1] + vertical_moment) / total,
        ],
        dtype=np.float64,
    )
    return (
        total,
        centroid,
        {
            "construction": (
                "closed-form polynomial current density integrated over the exact "
                "traced separatrix supports, including first cell moments"
            ),
            "declared_current_a": total,
            "current_centroid_m": centroid.tolist(),
            "closed_form_flux_samples_used_for_current_support_only": True,
        },
    )


def _production_seed(
    profile: ForwardProfile,
    case_name: str,
    target_current: float,
    centroid: np.ndarray,
    current_receipt: dict[str, Any],
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Select the requested production cold-seed branch and retain provenance."""

    requested_class = (
        TopologyClass.DIVERTED
        if case_name == "diverted-jump-bearing"
        else TopologyClass.LIMITED
    )
    geometry = (
        SaddleSeedGeometry(tuple(AXIS_M), tuple(X_POINT_M))
        if requested_class == TopologyClass.DIVERTED
        else None
    )
    portfolio = profile.cold_seed_portfolio(
        target_current,
        centroid,
        diverted_geometry=geometry,
    )
    branch = int(requested_class)
    branches = portfolio.branches
    seed = np.asarray(branches.flux[branch], dtype=np.float64)
    construction = ColdSeedConstruction(int(branches.construction[branch]))
    receipt = {
        "factory": "profile.cold_seed_portfolio",
        "requested_class": requested_class.name.lower(),
        "branch_index": branch,
        "construction": construction.name.lower(),
        "plasma_current_a": float(branches.plasma_current[branch]),
        "current_centroid_m": np.asarray(
            branches.centroid[branch], dtype=np.float64
        ).tolist(),
        "seed_radius_m": float(branches.radius[branch]),
        "supported_cell_count": int(branches.supported_cells[branch]),
        "anchor_available": bool(branches.anchor_available[branch]),
        "anchor_m": np.asarray(branches.anchor[branch], dtype=np.float64).tolist(),
        "declared_axis_m": np.asarray(
            branches.declared_axis[branch], dtype=np.float64
        ).tolist(),
        "declared_boundary_m": np.asarray(
            branches.declared_boundary[branch], dtype=np.float64
        ).tolist(),
        "stored_flux_samples_used": bool(branches.stored_flux_samples_used[branch]),
        "current_target_provenance": current_receipt,
        "state_sha256_binary64": hashlib.sha256(seed.tobytes()).hexdigest(),
    }
    return seed, branch, receipt


def _enum_name(enum_type: Any, value: Any) -> str:
    return enum_type(int(value)).name.lower()


def _production_solver_receipt(equilibrium: Any) -> dict[str, Any]:
    """Expose all globalisation telemetry retained by the public solve result."""

    history = equilibrium.fixed_point
    trip_count = int(history.active_set_iterations)
    active_residuals = np.asarray(history.active_set_residuals, dtype=np.float64)
    mask_differences = np.asarray(history.active_set_mask_differences, dtype=np.int64)
    cycle_damping = np.asarray(
        history.active_set_cycle_damping_activations, dtype=np.int64
    )
    trips = [
        {
            "trip": index + 1,
            "live_relative_residual": float(active_residuals[index]),
            "mask_difference_cells": int(mask_differences[index]),
            "cycle_damping_activated": bool(cycle_damping[index]),
        }
        for index in range(trip_count)
    ]

    decisions = np.asarray(history.inner_iteration_decisions, dtype=np.int64)
    executed = np.flatnonzero(
        decisions != int(recovery.fixed_point.InnerIterationDecision.NOT_EXECUTED)
    )
    inner = []
    for index in executed:
        inner.append(
            {
                "iteration": int(index + 1),
                "residual_before": float(
                    history.inner_iteration_residuals_before[index]
                ),
                "residual_after": float(history.inner_iteration_residuals_after[index]),
                "proposed_step_norm": float(
                    history.inner_iteration_proposed_step_norms[index]
                ),
                "accepted": bool(history.inner_iteration_accepted[index]),
                "decision": _enum_name(
                    recovery.fixed_point.InnerIterationDecision, decisions[index]
                ),
                "krylov_qualification": _enum_name(
                    recovery.fixed_point.KrylovActionQualification,
                    history.inner_iteration_krylov_qualifications[index],
                ),
                "applied_factor": float(history.inner_iteration_applied_factors[index]),
                "krylov_reduction": float(
                    history.inner_iteration_krylov_reductions[index]
                ),
                "krylov_tolerance": float(
                    history.inner_iteration_krylov_tolerances[index]
                ),
            }
        )

    total_promotion_count = int(history.attempted_newton_promotions)
    recovery_outcomes = np.asarray(history.promotion_recovery_outcomes, dtype=np.int64)
    promotions = []
    for index in executed:
        recovery_outcome = int(recovery_outcomes[index])
        promotions.append(
            {
                "promotion": index + 1,
                "backtrack_count": int(history.promotion_backtrack_counts[index]),
                "continuation_activated": bool(
                    history.promotion_recovery_activations[index]
                ),
                "continuation_radius": np.asarray(
                    history.promotion_recovery_radii[index], dtype=np.float64
                ).tolist(),
                "continuation_outcome": _enum_name(
                    recovery.fixed_point.RecoveryOutcome, recovery_outcome
                ),
                "model_rebuild_activated": bool(
                    history.promotion_model_rebuild_activations[index]
                ),
                "model_rebuild_damping": float(
                    history.promotion_model_rebuild_damping[index]
                ),
                "steepest_descent_activated": bool(
                    history.promotion_descent_activations[index]
                ),
                "steepest_descent_scale": float(
                    history.promotion_descent_scales[index]
                ),
            }
        )

    continuation = {}
    for name in ("common_sol", "private_flux"):
        record = getattr(equilibrium.continuation, name)
        continuation[name] = {
            "active": bool(record.active),
            "domain": record.domain_name,
            "form": record.form_name,
            "continuity": record.continuity_name,
            "support": float(record.support),
            "decay_width": float(record.decay_width),
            "truncated_fraction": float(record.truncated_fraction),
        }

    trace = np.asarray(history.trace, dtype=np.float64)
    trace_indices = np.flatnonzero(np.isfinite(trace))
    return {
        "telemetry_scope": (
            "active-set residuals cover every production trip; detailed inner "
            "globalisation arrays describe the terminal frozen-mask trip exposed "
            "by FixedPointResult"
        ),
        "trip_count": trip_count,
        "per_trip_residual_history": trips,
        "finite_trace_indices": trace_indices.tolist(),
        "finite_residual_trace": trace[trace_indices].tolist(),
        "termination": _enum_name(
            recovery.fixed_point.FixedPointTerminationReason,
            history.termination_reason,
        ),
        "converged": bool(history.converged),
        "krylov_action_qualification": _enum_name(
            recovery.fixed_point.KrylovActionQualification,
            history.krylov_action_qualification,
        ),
        "attempted_newton_promotions": total_promotion_count,
        "accepted_newton_promotions": int(history.accepted_newton_promotions),
        "exposed_terminal_trip_promotions": len(promotions),
        "globalisation_decisions": inner,
        "promotion_globalisation": promotions,
        "source_continuation": continuation,
        "configured_controls": {
            "own_mask_acceptance": "production default enabled",
            "strict_active_set_settlement": "production default enabled",
            "retain_outer_best_iterate": "production default enabled",
            "stop_on_active_set_stagnation": "production default enabled",
            "continue_newton_trajectory": "production default enabled",
            "continue_globalization_state": "production default enabled",
        },
    }


def _finite_census(
    values: Any, segments: tuple[tuple[str, int], ...] = ()
) -> dict[str, Any]:
    """Count non-finite values and locate the first one in a named segment."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    nonfinite = np.flatnonzero(~np.isfinite(array))
    first = None
    if len(nonfinite):
        index = int(nonfinite[0])
        offset = 0
        segment_name = "scalar"
        local_index = index
        for name, size in segments:
            if index < offset + size:
                segment_name = name
                local_index = index - offset
                break
            offset += size
        value = array[index]
        if np.isnan(value):
            kind = "nan"
        elif value > 0:
            kind = "positive_infinity"
        else:
            kind = "negative_infinity"
        first = {
            "flat_index": index,
            "segment": segment_name,
            "segment_index": int(local_index),
            "kind": kind,
        }
    return {
        "size": int(array.size),
        "finite": not len(nonfinite),
        "finite_count": int(np.count_nonzero(np.isfinite(array))),
        "nonfinite_count": int(len(nonfinite)),
        "nan_count": int(np.count_nonzero(np.isnan(array))),
        "positive_infinity_count": int(np.count_nonzero(np.isposinf(array))),
        "negative_infinity_count": int(np.count_nonzero(np.isneginf(array))),
        "first_nonfinite": first,
    }


def _diagnose_seed_action(
    *,
    name: str,
    seed: np.ndarray,
    seed_receipt: dict[str, Any],
    profile: ForwardProfile,
    target_current: float,
) -> dict[str, Any]:
    """Measure the first frozen-mask linear action through public operators."""

    operator = profile.operator
    state = jnp.asarray(seed, dtype=jnp.float64)
    initial_mask = operator.residual_shadow_mask(state)
    shadowed_map = operator.flux_map_with_shadow(target_current=target_current)

    def frozen_map(candidate):
        return shadowed_map(candidate, initial_mask)

    mapped = jax.jit(frozen_map)(state)
    jax.block_until_ready(mapped)
    residual = np.asarray(mapped - state, dtype=np.float64)
    mapped_values = np.asarray(mapped, dtype=np.float64)
    residual_sup = float(np.max(np.abs(residual)))
    relative_residual = residual_sup / max(
        float(np.max(np.abs(mapped_values))), np.finfo(np.float64).tiny
    )
    if residual_sup > np.finfo(np.float64).tiny:
        direction = residual / residual_sup
        direction_kind = "unit_sup_fixed_point_residual"
    else:
        direction = np.zeros_like(residual)
        direction[int(np.argmax(np.abs(np.asarray(seed))))] = 1.0
        direction_kind = "unit_coordinate_at_largest_seed_component"

    def exposed(candidate):
        unscaled = operator.cell_current_moments(candidate)
        normalised, amplitude = operator.normalised_current_moments(
            candidate, target_current
        )
        normalised_internal = operator.current_moment_image(normalised)
        return {
            "amplitude": amplitude,
            "frozen_map": frozen_map(candidate),
            "normalised_cell_current": normalised.cell_current,
            "normalised_internal_flux": normalised_internal,
            "normalised_radial_moment": normalised.radial_moment,
            "normalised_vertical_moment": normalised.vertical_moment,
            "unscaled_cell_current": unscaled.cell_current,
            "unscaled_radial_moment": unscaled.radial_moment,
            "unscaled_vertical_moment": unscaled.vertical_moment,
        }

    values, tangents = jax.jit(
        lambda candidate, vector: jax.jvp(exposed, (candidate,), (vector,))
    )(state, jnp.asarray(direction))
    jax.block_until_ready(tangents)
    state_segments = (
        ("grid", int(operator.grid.node_number)),
        ("wall", int(operator.wall.node_number)),
        (
            "sample",
            0 if operator.sample is None else int(operator.sample.node_number),
        ),
    )
    cell_segments = (("plasma_cell", int(operator.grid.node_number)),)
    intermediate_segments = {
        "amplitude": (),
        "frozen_map": state_segments,
        "normalised_cell_current": cell_segments,
        "normalised_internal_flux": state_segments,
        "normalised_radial_moment": cell_segments,
        "normalised_vertical_moment": cell_segments,
        "unscaled_cell_current": cell_segments,
        "unscaled_radial_moment": cell_segments,
        "unscaled_vertical_moment": cell_segments,
    }
    intermediates = {
        key: {
            "value": _finite_census(values[key], intermediate_segments[key]),
            "jvp": _finite_census(tangents[key], intermediate_segments[key]),
        }
        for key in values
    }
    linear_action = np.asarray(direction) - np.asarray(tangents["frozen_map"])
    action_census = _finite_census(linear_action, state_segments)
    ordered = (
        "unscaled_cell_current",
        "unscaled_radial_moment",
        "unscaled_vertical_moment",
        "amplitude",
        "normalised_cell_current",
        "normalised_radial_moment",
        "normalised_vertical_moment",
        "normalised_internal_flux",
        "frozen_map",
    )
    first_nonfinite = next(
        (
            {
                "intermediate": key,
                **intermediates[key]["jvp"]["first_nonfinite"],
            }
            for key in ordered
            if intermediates[key]["jvp"]["first_nonfinite"] is not None
        ),
        None,
    )
    unscaled_total = float(np.sum(np.asarray(values["unscaled_cell_current"])))
    attempted_amplitude = (
        target_current / unscaled_total if unscaled_total != 0.0 else float("inf")
    )
    amplitude = float(np.asarray(values["amplitude"]))
    return {
        "name": name,
        "seed": seed_receipt,
        "initial_frozen_shadow_cell_count": int(
            np.count_nonzero(np.asarray(initial_mask))
        ),
        "map_residual": {
            "relative_sup": relative_residual,
            "absolute_sup_wb": residual_sup,
            "census": _finite_census(residual, state_segments),
        },
        "probe_direction": {
            "construction": direction_kind,
            "sup_norm": float(np.max(np.abs(direction))),
            "census": _finite_census(direction, state_segments),
        },
        "lambda_amplitude_admissibility": {
            "target_current_a": target_current,
            "unscaled_current_a": unscaled_total,
            "attempted_amplitude": attempted_amplitude,
            "public_amplitude": amplitude if np.isfinite(amplitude) else None,
            "admissible": bool(np.isfinite(amplitude)),
            "amplitude_jvp": float(np.asarray(tangents["amplitude"])),
            "amplitude_jvp_finite": bool(
                np.isfinite(float(np.asarray(tangents["amplitude"])))
            ),
        },
        "public_intermediate_census": intermediates,
        "fixed_point_linear_action": action_census,
        "map_jvp_finite": intermediates["frozen_map"]["jvp"]["finite"],
        "fixed_point_linear_action_finite": action_census["finite"],
        "first_nonfinite_public_intermediate": first_nonfinite,
    }


def _nan_census(case_name: str) -> dict[str, Any]:
    """Bank the production and near-root first-action census at reduced size."""

    started = perf_counter()
    configure_dtypes()
    compilation_cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    requested_cells = -110
    timings: dict[str, float] = {}
    with _timed_stage(
        "nan_census_construction",
        timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        carrier_case, source_case, exact = _case(case_name)
        machine = oracle_fixture.cached_machine(
            carrier_case,
            requested_cells,
            wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        )
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        oracle_state = _exact_state(case_name, exact, coordinates)
        empty_operator = oracle_fixture.forward_operator(source_case, machine)
        exact_physical = oracle_fixture.exact_current_moments(
            source_case, empty_operator, oracle_state
        )
        exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
        exact_internal = oracle_fixture._internal_flux_image(
            empty_operator, exact_coefficients
        )
        operator = oracle_fixture.forward_operator(
            source_case, machine, oracle_state - exact_internal
        )
        profile = ForwardProfile(
            operator,
            StencilMesh(machine.node, machine.stencil, machine.area),
            newton_steps=recovery.NEWTON_STEPS,
        )
        target_current, centroid, current_receipt = _closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
        production_seed, requested_class, production_receipt = _production_seed(
            profile, case_name, target_current, centroid, current_receipt
        )
        near_root_seed = np.asarray(oracle_state, dtype=np.float64)
        near_root_receipt = {
            "factory": "closed_form_near_root_control",
            "requested_class": TopologyClass(requested_class).name.lower(),
            "construction": (
                "independent closed-form total flux on the identical cached carrier"
            ),
            "state_sha256_binary64": hashlib.sha256(
                near_root_seed.tobytes()
            ).hexdigest(),
        }
    arms = []
    for name, seed, seed_receipt in (
        ("production_moment_seed", production_seed, production_receipt),
        ("closed_form_near_root_seed", near_root_seed, near_root_receipt),
    ):
        with _timed_stage(
            f"nan_census_{name}",
            timings,
            case_name=case_name,
            requested_cells=requested_cells,
        ):
            arms.append(
                _diagnose_seed_action(
                    name=name,
                    seed=np.asarray(seed, dtype=np.float64),
                    seed_receipt=seed_receipt,
                    profile=profile,
                    target_current=target_current,
                )
            )
    by_name = {arm["name"]: arm for arm in arms}
    production_finite = by_name["production_moment_seed"][
        "fixed_point_linear_action_finite"
    ]
    near_root_finite = by_name["closed_form_near_root_seed"][
        "fixed_point_linear_action_finite"
    ]
    if not production_finite and near_root_finite:
        sentence = (
            "The production moment seed has a non-finite first linearised action "
            "while the closed-form near-root control has a finite action."
        )
    elif not production_finite:
        sentence = "Both seed classes have a non-finite first linearised action."
    else:
        sentence = "The production moment seed has a finite first linearised action."
    receipt = {
        "schema": {
            "$id": "nova.solovev-production-route-nan-census",
            "version": 1,
            "required": [
                "case",
                "requested_cells",
                "arms",
                "headline",
                "lane",
            ],
        },
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "route": "ForwardProfile production frozen-mask Newton linearisation",
        "seed_factory": "profile.cold_seed_portfolio",
        "target_current_a": target_current,
        "persistent_compilation_cache": compilation_cache.receipt(),
        "stage_wall_seconds": timings,
        "arms": arms,
        "headline": {"sentence": sentence},
        "lane": {
            **_lane(),
            "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "threaded_settings": _thread_settings(),
            "elapsed_seconds": perf_counter() - started,
            "exit_marker": "SOLOVEV_NAN_CENSUS_EXIT=0",
        },
    }
    _write_json(_diagnostic_path(case_name), receipt)
    return receipt


def _case(case_name: str) -> tuple[RotatingEquilibrium, RotatingEquilibrium, Any]:
    if case_name == "diverted-jump-bearing":
        coefficients = _solve_coefficients()
        return (
            oracle_fixture.analytic_case(),
            _diverted_source(coefficients),
            coefficients,
        )
    base_name = case_name.removesuffix("-static")
    static = reference_cases()[base_name].static_limit()
    return static, static, static


def _exact_state(case_name: str, exact: Any, coordinates: np.ndarray) -> np.ndarray:
    if case_name == "diverted-jump-bearing":
        return np.asarray(_polynomial_flux(coordinates, exact), dtype=np.float64)
    return oracle_fixture.exact_state(exact, coordinates)


def _exact_derivatives(
    case_name: str, exact: Any, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if case_name == "diverted-jump-bearing":
        return (
            np.asarray(_polynomial_gradient(coordinates, exact), dtype=np.float64),
            np.asarray(_polynomial_hessian(coordinates, exact), dtype=np.float64),
        )
    radial, vertical = exact.flux_gradient(coordinates[:, 0], coordinates[:, 1])
    label = coordinates[:, 0] ** 2 - exact.major_radius**2
    offset_gradient = exact._flux_offset_derivative(label)
    offset_curvature = exact.pressure_coefficient * np.exp(
        exact.rotation_parameter * label
    )
    hessian = np.zeros((len(coordinates), 2, 2), dtype=np.float64)
    hessian[:, 0, 0] = (
        -2.0 * offset_gradient - 4.0 * coordinates[:, 0] ** 2 * offset_curvature
    )
    hessian[:, 1, 1] = -2.0 * exact.field_coefficient
    return (
        2.0 * np.pi * np.column_stack((radial, vertical)),
        2.0 * np.pi * hessian,
    )


def _boundary(case_name: str, exact: Any) -> np.ndarray:
    if case_name == "diverted-jump-bearing":
        return _lcfs(exact)
    radius, half_height, _weight, _offset = exact._surface_nodes(0.0, 721)
    return np.vstack(
        (
            np.column_stack((radius, half_height)),
            np.column_stack((radius[::-1], -half_height[::-1])),
        )
    )


def _quadratic_derivatives(
    mesh: StencilMesh, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = mesh.coordinate[mesh.centre]
    cluster = mesh.coordinate[mesh.stencil]
    offset = cluster - centre[:, None, :]
    scale = np.max(np.abs(offset), axis=1)
    local = offset / scale[:, None, :]
    radial = local[..., 0]
    vertical = local[..., 1]
    design = np.stack(
        (
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ),
        axis=-1,
    )
    coefficient = np.einsum(
        "rij,rj->ri", np.linalg.pinv(design), np.asarray(field)[mesh.stencil]
    )
    gradient = np.column_stack(
        (coefficient[:, 1] / scale[:, 0], coefficient[:, 2] / scale[:, 1])
    )
    hessian = np.empty((len(centre), 2, 2), dtype=np.float64)
    hessian[:, 0, 0] = 2.0 * coefficient[:, 3] / scale[:, 0] ** 2
    hessian[:, 0, 1] = coefficient[:, 4] / (scale[:, 0] * scale[:, 1])
    hessian[:, 1, 0] = hessian[:, 0, 1]
    hessian[:, 1, 1] = 2.0 * coefficient[:, 5] / scale[:, 1] ** 2
    return centre, gradient, hessian


def _norm(field: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.abs(np.asarray(field)[mask])
    if selected.size == 0:
        raise RuntimeError("an accuracy norm has empty support")
    finite = np.isfinite(selected)
    if not np.any(finite):
        return {
            "support_cells": int(np.count_nonzero(mask)),
            "component_count": int(selected.size),
            "finite_component_count": 0,
            "measurement_status": "nonfinite_terminal_error",
            "sup": None,
            "rms": None,
        }
    finite_selected = selected[finite]
    return {
        "support_cells": int(np.count_nonzero(mask)),
        "component_count": int(selected.size),
        "finite_component_count": int(np.count_nonzero(finite)),
        "measurement_status": "finite" if np.all(finite) else "finite_subset",
        "sup": float(np.max(finite_selected)),
        "rms": float(np.sqrt(np.mean(finite_selected**2))),
    }


def _unavailable_norm(reason: str) -> dict[str, Any]:
    return {
        "support_cells": 0,
        "component_count": 0,
        "finite_component_count": 0,
        "measurement_status": "unavailable",
        "unavailable_reason": reason,
        "sup": None,
        "rms": None,
    }


def _field_norms(
    field: np.ndarray,
    boundary_band: np.ndarray,
    *,
    band_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "whole_domain": _norm(field, np.ones(len(field), dtype=bool)),
        "two_pitch_boundary_band": (
            _norm(field, boundary_band)
            if band_unavailable_reason is None
            else _unavailable_norm(band_unavailable_reason)
        ),
    }


def _analytic_region_norms(
    field: np.ndarray, psi_norm: np.ndarray, span_wb: float
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    absolute = np.abs(np.asarray(field, dtype=np.float64))
    for name, mask in _region_masks(psi_norm).items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            reason = (
                "no separatrix band at this resolution"
                if name == "separatrix_band"
                else f"no {name} cells at this resolution"
            )
            result[name] = {
                "cell_count": 0,
                "finite_cell_count": 0,
                "measurement_status": "unavailable",
                "unavailable_reason": reason,
                "absolute_sup_wb": None,
                "absolute_rms_wb": None,
                "relative_sup": None,
                "relative_rms": None,
            }
            continue
        finite = np.isfinite(absolute[mask])
        selected = absolute[mask][finite]
        if selected.size == 0:
            result[name] = {
                "cell_count": count,
                "finite_cell_count": 0,
                "measurement_status": "unavailable",
                "unavailable_reason": "no finite terminal error in this region",
                "absolute_sup_wb": None,
                "absolute_rms_wb": None,
                "relative_sup": None,
                "relative_rms": None,
            }
            continue
        sup = float(np.max(selected))
        rms = float(np.sqrt(np.mean(selected**2)))
        result[name] = {
            "cell_count": count,
            "finite_cell_count": int(selected.size),
            "measurement_status": "finite" if np.all(finite) else "finite_subset",
            "absolute_sup_wb": sup,
            "absolute_rms_wb": rms,
            "relative_sup": sup / span_wb,
            "relative_rms": rms / span_wb,
        }
    return result


def _topology(operator, state: np.ndarray) -> dict[str, Any]:
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError:
        return {
            "read_status": "no_qualified_axis",
            "class": None,
            "axis_rz_m": None,
            "x_point_rz_m": None,
            "boundary_flux_wb": None,
            "axis_flux_wb": None,
            "flux_span_wb": None,
        }
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    return {
        "read_status": "qualified_axis",
        "class": "diverted" if bool(topology.diverted) else "limited",
        "axis_rz_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        "boundary_flux_wb": float(topology.boundary_flux),
        "axis_flux_wb": float(topology.axis_flux),
        "flux_span_wb": float(topology.flux_span),
    }


def _analytic_diverted_topology(coefficients: np.ndarray) -> dict[str, Any]:
    axis_flux = float(_polynomial_flux(AXIS_M[None, :], coefficients)[0])
    boundary_flux = float(_polynomial_flux(X_POINT_M[None, :], coefficients)[0])
    return {
        "read_status": "analytic_stationary_point_fallback",
        "class": "diverted",
        "axis_rz_m": AXIS_M.tolist(),
        "x_point_rz_m": X_POINT_M.tolist(),
        "boundary_flux_wb": boundary_flux,
        "axis_flux_wb": axis_flux,
        "flux_span_wb": axis_flux - boundary_flux,
    }


def _plot(
    coordinates: np.ndarray,
    derivative_coordinates: np.ndarray,
    errors: dict[str, np.ndarray],
    boundary: np.ndarray,
    path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for axis, name, points in zip(
        axes,
        NORM_FIELDS,
        (coordinates, derivative_coordinates, derivative_coordinates),
        strict=True,
    ):
        magnitude = np.asarray(errors[name], dtype=np.float64)
        if magnitude.ndim > 1:
            components = magnitude.reshape(len(magnitude), -1)
            scale = np.max(np.abs(components), axis=1)
            normalized = np.divide(
                components,
                scale[:, None],
                out=np.zeros_like(components),
                where=np.isfinite(scale[:, None]) & (scale[:, None] > 0.0),
            )
            magnitude = scale * np.sqrt(np.mean(normalized**2, axis=1))
        finite = np.isfinite(magnitude)
        finite_values = magnitude[finite]
        if finite_values.size == 0:
            axis.scatter(points[:, 0], points[:, 1], s=10, c="0.72", linewidths=0.0)
            axis.text(
                0.5,
                0.5,
                "no finite field, unqualified",
                ha="center",
                va="center",
                transform=axis.transAxes,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.4"},
            )
        else:
            minimum = float(np.nanmin(finite_values))
            maximum = float(np.nanmax(finite_values))
            positive = finite_values[finite_values > 0.0]
            if positive.size and float(np.min(positive)) < maximum:
                norm = LogNorm(vmin=float(np.min(positive)), vmax=maximum)
            else:
                width = max(abs(minimum), 1.0) * np.finfo(float).eps
                norm = Normalize(vmin=minimum - width, vmax=maximum + width)
            artist = axis.scatter(
                points[finite, 0],
                points[finite, 1],
                c=finite_values,
                s=10,
                cmap="magma",
                norm=norm,
                linewidths=0.0,
            )
            figure.colorbar(artist, ax=axis, label=f"absolute {name} error")
        if not np.all(finite):
            axis.scatter(
                points[~finite, 0],
                points[~finite, 1],
                marker="x",
                s=16,
                c="#35b9c8",
                linewidths=0.7,
                label=f"non-finite: {np.count_nonzero(~finite)}",
            )
            axis.legend(loc="best", fontsize="x-small")
        axis.plot(boundary[:, 0], boundary[:, 1], color="#35b9c8", lw=0.9)
        axis.set_title(name)
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_aspect("equal")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _measure(case_name: str, requested_cells: int) -> dict[str, Any]:
    row_started = perf_counter()
    configure_dtypes()
    compilation_cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    print(
        f"SOLOVEV_COMPILATION_CACHE directory={compilation_cache.directory} "
        f"version={compilation_cache.version_key}",
        flush=True,
    )
    stage_timings: dict[str, float] = {}
    with _timed_stage(
        "case_and_machine",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        carrier_case, source_case, exact = _case(case_name)
        machine = oracle_fixture.cached_machine(
            carrier_case,
            requested_cells,
            wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        )

    with _timed_stage(
        "operator_and_seed",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        oracle_state = _exact_state(case_name, exact, coordinates)
        empty_operator = oracle_fixture.forward_operator(source_case, machine)
        exact_physical = oracle_fixture.exact_current_moments(
            source_case, empty_operator, oracle_state
        )
        exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
        exact_internal = oracle_fixture._internal_flux_image(
            empty_operator, exact_coefficients
        )
        operator = oracle_fixture.forward_operator(
            source_case, machine, oracle_state - exact_internal
        )
        mesh = StencilMesh(machine.node, machine.stencil, machine.area)
        profile = ForwardProfile(
            operator,
            mesh,
            newton_steps=recovery.NEWTON_STEPS,
        )
        target_current, current_centroid, current_receipt = _closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
        seed, requested_class, seed_receipt = _production_seed(
            profile,
            case_name,
            target_current,
            current_centroid,
            current_receipt,
        )
        seed_moments = operator.cell_current_moments(seed)
        seed_amplitude = float(
            operator.current_normalisation_amplitude(
                target_current, jnp.sum(seed_moments.cell_current)
            )
        )

    with _timed_stage(
        "production_solve",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        equilibrium = profile.solve(
            seed,
            route="newton_krylov",
            target_current=target_current,
            newton_steps=recovery.NEWTON_STEPS,
            gmres_iterations=recovery.KRYLOV_ITERATIONS,
            warmup=0,
            convergence_tolerance=TERMINAL_RESIDUAL_BOUND,
            stream_active_set=True,
            stream_inner_iterations=True,
        )
        jax.block_until_ready(equilibrium.flux)
    solve_seconds = stage_timings["production_solve"]
    terminal = equilibrium.fixed_point
    terminal_state = np.asarray(equilibrium.flux, dtype=np.float64)
    terminal_residual = float(terminal.residual)
    production_solver = _production_solver_receipt(equilibrium)
    qualification = (
        "qualified"
        if np.isfinite(terminal_residual)
        and terminal_residual <= TERMINAL_RESIDUAL_BOUND
        else "unqualified"
    )
    if not np.all(np.isfinite(terminal_state)):
        termination_reason = "nonfinite_terminal_state"
    elif not np.isfinite(terminal_residual):
        termination_reason = "nonfinite_terminal_residual"
    elif terminal_residual > TERMINAL_RESIDUAL_BOUND:
        termination_reason = (
            "fixed_point_residual_above_qualification_bound_after_iteration_budget"
        )
    else:
        termination_reason = "fixed_point_residual_within_qualification_bound"

    with _timed_stage(
        "accuracy_scoring",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        cell_count = len(machine.node)
        root_grid = terminal_state[:cell_count]
        exact_grid = oracle_state[:cell_count]
        derivative_coordinates, root_gradient, root_hessian = _quadratic_derivatives(
            mesh, root_grid
        )
        exact_gradient, exact_hessian = _exact_derivatives(
            case_name, exact, derivative_coordinates
        )
        psi_error = root_grid - exact_grid
        gradient_error = root_gradient - exact_gradient
        hessian_error = root_hessian - exact_hessian
        boundary = _boundary(case_name, exact)
        pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
        psi_boundary_band = _distance_to_boundary(machine.node, boundary) <= (
            BOUNDARY_BAND_PITCHES * pitch
        )
        derivative_boundary_band = _distance_to_boundary(
            derivative_coordinates, boundary
        ) <= (BOUNDARY_BAND_PITCHES * pitch)
        exact_topology = _topology(operator, oracle_state)
        reference_topology_fallback = False
        if (
            case_name == "diverted-jump-bearing"
            and exact_topology["read_status"] == "no_qualified_axis"
        ):
            exact_topology = _analytic_diverted_topology(exact)
            reference_topology_fallback = True
        root_topology = _topology(operator, terminal_state)
        axis_reference = (
            AXIS_M
            if case_name == "diverted-jump-bearing"
            else np.asarray(exact.magnetic_axis)
        )
        x_reference = X_POINT_M if case_name == "diverted-jump-bearing" else None
        x_error = None
        if x_reference is not None and root_topology["x_point_rz_m"] is not None:
            x_error = float(
                np.linalg.norm(np.asarray(root_topology["x_point_rz_m"]) - x_reference)
            )
        span = max(abs(float(exact_topology["flux_span_wb"])), np.finfo(float).tiny)
        exact_psi_norm = (exact_grid - float(exact_topology["axis_flux_wb"])) / float(
            exact_topology["flux_span_wb"]
        )
        analytic_regions = _analytic_region_norms(psi_error, exact_psi_norm, span)
        band_unavailable_reason = (
            "no separatrix band at this resolution"
            if case_name == "diverted-jump-bearing"
            and requested_cells == -110
            and reference_topology_fallback
            else None
        )
    figure = _figure_path(case_name, requested_cells)
    with _timed_stage(
        "figure_render",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        _plot(
            machine.node,
            derivative_coordinates,
            {
                "psi": psi_error,
                "gradient": gradient_error,
                "hessian": hessian_error,
            },
            boundary,
            figure,
            f"{case_name} · {_slug(requested_cells)} · {qualification}",
        )
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": cell_count,
        "characteristic_pitch_m": pitch,
        "derivative_support_cells": len(derivative_coordinates),
        "cache": machine.cache,
        "persistent_compilation_cache": compilation_cache.receipt(),
        "stage_wall_seconds": stage_timings,
        "lane": {
            **_lane(),
            "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "threaded_settings": _thread_settings(),
            "elapsed_seconds": perf_counter() - row_started,
            "exit_marker": "SOLOVEV_ROW_EXIT=0",
        },
        "solver": {
            "route": "profile.solve newton_krylov",
            "call": (
                "profile.solve(seed, route='newton_krylov', "
                "target_current=Ip_closed_form, oracle budgets as keywords)"
            ),
            "target_current_a": target_current,
            "target_current_argument": "Ip_closed_form",
            "requested_seed_class": TopologyClass(requested_class).name.lower(),
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "terminal_fixed_point_residual": (
                terminal_residual if np.isfinite(terminal_residual) else None
            ),
            "terminal_fixed_point_residual_status": (
                "finite" if np.isfinite(terminal_residual) else "nonfinite"
            ),
            "termination": production_solver["termination"],
            "qualification_reason": termination_reason,
            "qualification_bound": TERMINAL_RESIDUAL_BOUND,
            "qualification": qualification,
            "solve_wall_seconds": solve_seconds,
            "seed": seed_receipt,
            "production_telemetry": production_solver,
            "lambda_amplitude_history": {
                "sampling": (
                    "seed and terminal production-seam states; intermediate "
                    "amplitudes are not exposed by FixedPointResult"
                ),
                "samples": [
                    {"state": "seed", "amplitude": seed_amplitude},
                    {
                        "state": "terminal",
                        "amplitude": float(equilibrium.normalisation.amplitude),
                    },
                ],
                "terminal_policy": equilibrium.normalisation.policy_name,
                "terminal_rescaled": bool(equilibrium.normalisation.rescaled),
            },
        },
        "norms": {
            "psi": _field_norms(
                psi_error,
                psi_boundary_band,
                band_unavailable_reason=band_unavailable_reason,
            ),
            "gradient": _field_norms(
                gradient_error,
                derivative_boundary_band,
                band_unavailable_reason=band_unavailable_reason,
            ),
            "hessian": _field_norms(
                hessian_error,
                derivative_boundary_band,
                band_unavailable_reason=band_unavailable_reason,
            ),
        },
        "analytic_flux_regions": analytic_regions,
        "geometry": {
            "magnetic_axis_position_error_m": (
                float(
                    np.linalg.norm(
                        np.asarray(root_topology["axis_rz_m"]) - axis_reference
                    )
                )
                if root_topology["axis_rz_m"] is not None
                else None
            ),
            "x_point_position_error_m": x_error,
            "boundary_flux_error_wb": (
                abs(
                    float(root_topology["boundary_flux_wb"])
                    - float(exact_topology["boundary_flux_wb"])
                )
                if root_topology["boundary_flux_wb"] is not None
                else None
            ),
            "root_topology": root_topology,
            "exact_topology": exact_topology,
        },
        "figure": {
            "filesystem_path": str(figure.relative_to(ROOT)),
            "project_absolute_src": (
                f"/nova/figures/gs-absolute-accuracy/solovev/{figure.name}"
            ),
            "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
        },
    }
    _validate_row(row)
    _write_json(_part_path(case_name, requested_cells), row)
    return row


def _validate_row(row: dict[str, Any]) -> None:
    if row["case"] not in CASE_NAMES:
        raise RuntimeError("unknown certificate case")
    if row["requested_cells"] not in REQUESTED_CELLS:
        raise RuntimeError("unknown certificate resolution")
    if row["solver"]["qualification"] not in {"qualified", "unqualified"}:
        raise RuntimeError("terminal qualification is missing")
    if row["solver"]["route"] != "profile.solve newton_krylov":
        raise RuntimeError("row did not use the production solve seam")
    if row["solver"]["seed"]["factory"] != "profile.cold_seed_portfolio":
        raise RuntimeError("row did not use the production cold-seed portfolio")
    telemetry = row["solver"]["production_telemetry"]
    if telemetry["trip_count"] != len(telemetry["per_trip_residual_history"]):
        raise RuntimeError("per-trip residual history is incomplete")
    for name in (
        "termination",
        "globalisation_decisions",
        "promotion_globalisation",
        "source_continuation",
    ):
        if name not in telemetry:
            raise RuntimeError(f"production telemetry is missing {name}")
    if len(row["solver"]["lambda_amplitude_history"]["samples"]) != 2:
        raise RuntimeError("seed and terminal lambda amplitudes are required")
    for field in NORM_FIELDS:
        for region in NORM_REGIONS:
            measured = row["norms"][field][region]
            for name in NORM_STATISTICS:
                if measured[name] is None and not (
                    measured["measurement_status"] == "unavailable"
                    or (
                        row["solver"]["qualification"] == "unqualified"
                        and measured["measurement_status"] == "nonfinite_terminal_error"
                    )
                ):
                    raise RuntimeError("a named accuracy norm is missing")
    src = row["figure"]["project_absolute_src"]
    if not src.startswith("/nova/figures/gs-absolute-accuracy/solovev/"):
        raise RuntimeError("figure src is not project absolute")


def _fit_quantity(
    rows: list[dict[str, Any]], field: str, region: str, statistic: str
) -> dict[str, Any]:
    by_qualification: dict[str, Any] = {}
    for qualification in ("qualified", "unqualified"):
        selected = [
            row
            for row in rows
            if row["solver"]["qualification"] == qualification
            and row["norms"][field][region][statistic] is not None
        ]
        if len(selected) < 3:
            by_qualification[qualification] = {
                "status": "insufficient_same_qualification_rungs",
                "rung_count": len(selected),
                "order": None,
                "standard_error": None,
                "confidence_interval": None,
                "theoretical_order": THEORETICAL_ORDER,
            }
            continue
        adapted = []
        for row in selected:
            error = max(
                float(row["norms"][field][region][statistic]),
                np.finfo(np.float64).tiny,
            )
            adapted.append(
                {
                    "characteristic_pitch_m": row["characteristic_pitch_m"],
                    "one_application_residual": {
                        "regions": {"all_carrier_cells": {"relative_sup": error}}
                    },
                }
            )
        if len(adapted) >= 4:
            fitted = _fit_order(adapted)
        else:
            pitch = np.asarray(
                [row["characteristic_pitch_m"] for row in adapted], dtype=np.float64
            )
            error = np.asarray(
                [
                    row["one_application_residual"]["regions"]["all_carrier_cells"][
                        "relative_sup"
                    ]
                    for row in adapted
                ],
                dtype=np.float64,
            )
            design = np.column_stack((np.ones(len(adapted)), np.log(pitch)))
            coefficients = np.linalg.lstsq(design, np.log(error), rcond=None)[0]
            fitted_log = design @ coefficients
            residual = np.log(error) - fitted_log
            degrees_of_freedom = len(adapted) - 2
            variance = float(np.sum(residual**2) / degrees_of_freedom)
            covariance = variance * np.linalg.inv(design.T @ design)
            order = float(coefficients[1])
            standard_error = float(np.sqrt(covariance[1, 1]))
            critical = float(stats.t.ppf(0.975, degrees_of_freedom))
            fitted = {
                "model": ("log(error) = intercept + order*log(characteristic_pitch)"),
                "characteristic_pitch": "square root of median carrier-cell area",
                "rung_count": len(adapted),
                "order": order,
                "standard_error": standard_error,
                "confidence_level": 0.95,
                "confidence_interval": [
                    order - critical * standard_error,
                    order + critical * standard_error,
                ],
                "degrees_of_freedom": degrees_of_freedom,
                "fitted_error": np.exp(fitted_log).tolist(),
            }
        fitted["error_quantity"] = f"{field}.{region}.{statistic}"
        fitted["qualification"] = qualification
        fitted["theoretical_order"] = THEORETICAL_ORDER
        by_qualification[qualification] = fitted
    return by_qualification


def _registry_reproduction() -> dict[str, Any]:
    reduced = measure_reduced_oracle()
    banked = json.loads(
        (ROOT / "scripts/oracle_rebaseline/results.json").read_text(encoding="utf-8")
    )["gate_registry"]
    rows = {}
    for name, bound in LOCKED_RECOVERY_BOUNDS.items():
        registry_bound = banked[name]["proposed_bound"]
        if registry_bound != bound:
            raise RuntimeError(f"locked recovery bound drifted for {name}")
        measured = None
        if name == "standing_forcing_sup_wb":
            measured = reduced["forcing_sup_wb"]
        elif name == "fixed_point_residual":
            measured = reduced["fixed_point_residual"]
        rows[name] = {
            "locked_bound": bound,
            "banked_registry_bound": registry_bound,
            "bound_reproduced": True,
            "reduced_measurement": measured,
            "reduced_measurement_within_bound": (
                measured <= bound if measured is not None else None
            ),
            "banked_measured_floor": banked[name]["measured_floor"],
        }
    return {
        "construction": reduced["construction"],
        "requested_cells": reduced["requested_cells"],
        "realised_cells": reduced["realised_cells"],
        "map_evaluations": reduced["map_evaluations"],
        "wall_seconds": reduced["wall_seconds"],
        "registry_entry_count": len(rows),
        "all_bounds_reproduced": len(rows) == 14
        and all(row["bound_reproduced"] for row in rows.values()),
        "directly_measured_entries_within_bound": all(
            row["reduced_measurement_within_bound"] is not False
            for row in rows.values()
        ),
        "lane_performance_qualification": RECOVERY_GATE_LANE_PERFORMANCE,
        "entries": rows,
    }


def _schema() -> dict[str, Any]:
    return {
        "$id": "nova.solovev-production-route-absolute-accuracy-certificate",
        "version": 2,
        "required": [
            "schema",
            "preregistration",
            "reduced_oracle_registry_reproduction",
            "cases",
            "verdict",
        ],
        "case_required": ["rows", "convergence_order_fits"],
        "row_required": [
            "case",
            "requested_cells",
            "norms",
            "geometry",
            "solver",
            "persistent_compilation_cache",
            "stage_wall_seconds",
            "lane",
            "figure",
        ],
        "norm_fields": list(NORM_FIELDS),
        "norm_regions": list(NORM_REGIONS),
        "norm_statistics": list(NORM_STATISTICS),
        "minimum_cases": 4,
        "minimum_resolutions_per_case": 3,
        "terminal_qualifications": ["qualified", "unqualified"],
        "solver_route": "profile.solve newton_krylov",
        "seed_factory": "profile.cold_seed_portfolio",
        "required_solver_telemetry": [
            "termination",
            "per_trip_residual_history",
            "globalisation_decisions",
            "promotion_globalisation",
            "source_continuation",
            "lambda_amplitude_history",
        ],
    }


def _scheduler_receipt(job_ids: list[str]) -> list[dict[str, Any]]:
    """Return terminal scheduler rows for explicitly named evidence jobs."""

    receipts = []
    for job_id in job_ids:
        output = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-j",
                job_id,
                "--format=JobIDRaw,State,Elapsed,ExitCode,NodeList",
                "-n",
                "-P",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            if not line.strip():
                continue
            raw_id, state, elapsed, exit_code, node = line.split("|", maxsplit=4)
            rows.append(
                {
                    "job_id": raw_id,
                    "state": state,
                    "elapsed": elapsed,
                    "exit_code": exit_code,
                    "node": node,
                }
            )
        receipts.append({"requested_job_id": job_id, "rows": rows})
    return receipts


def _aggregate_partial(
    output: Path = OUTPUT, *, scheduler_job_ids: list[str] | None = None
) -> dict[str, Any]:
    """Bank every terminal production row plus the reduced action census."""

    case_payload = {}
    missing_rows = []
    for case_name in CASE_NAMES:
        rows = []
        for requested in REQUESTED_CELLS:
            path = _part_path(case_name, requested)
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                _validate_row(row)
                rows.append(row)
            else:
                missing_rows.append({"case": case_name, "requested_cells": requested})
        fits = {}
        for field in NORM_FIELDS:
            fits[field] = {}
            for region in NORM_REGIONS:
                fits[field][region] = {
                    statistic: _fit_quantity(rows, field, region, statistic)
                    for statistic in NORM_STATISTICS
                }
        case_payload[case_name] = {
            "rows": rows,
            "convergence_order_fits": fits,
            "verdict": {
                "sentence": (
                    f"Banked {len(rows)} terminal production-route rows for "
                    f"{case_name}; the complete ladder is deferred pending the "
                    "non-finite linear-action diagnosis."
                )
            },
        }

    diagnostics = {
        case_name: json.loads(_diagnostic_path(case_name).read_text(encoding="utf-8"))
        for case_name in CASE_NAMES
    }
    production_nonfinite = []
    near_root_finite = []
    production_first_nonfinite = []
    near_root_first_nonfinite = []
    admissible_amplitudes = 0
    for case_name, diagnostic in diagnostics.items():
        arms = {arm["name"]: arm for arm in diagnostic["arms"]}
        if not arms["production_moment_seed"]["fixed_point_linear_action_finite"]:
            production_nonfinite.append(case_name)
        production_first_nonfinite.append(
            arms["production_moment_seed"]["first_nonfinite_public_intermediate"][
                "intermediate"
            ]
        )
        if arms["closed_form_near_root_seed"]["fixed_point_linear_action_finite"]:
            near_root_finite.append(case_name)
        near_root_first_nonfinite.append(
            arms["closed_form_near_root_seed"]["first_nonfinite_public_intermediate"][
                "intermediate"
            ]
        )
        admissible_amplitudes += sum(
            arm["lambda_amplitude_admissibility"]["admissible"] for arm in arms.values()
        )
    production_first = sorted(set(production_first_nonfinite))
    near_root_first = sorted(set(near_root_first_nonfinite))
    headline = (
        f"The first (I-J)v action is non-finite in "
        f"{len(production_nonfinite)} of {len(CASE_NAMES)} production moment seeds "
        f"and {len(CASE_NAMES) - len(near_root_finite)} of {len(CASE_NAMES)} "
        f"closed-form controls; all {admissible_amplitudes} lambda amplitudes are "
        f"admissible, and the earliest exposed non-finite JVP is "
        f"{', '.join(production_first)} for the production seeds and "
        f"{', '.join(near_root_first)} for the controls."
    )
    receipt = {
        "schema": {
            "$id": "nova.solovev-production-route-partial-certificate",
            "version": 1,
            "required": [
                "schema",
                "preregistration",
                "reduced_oracle_registry_reproduction",
                "cases",
                "nan_census",
                "scheduler_evidence",
                "verdict",
            ],
            "terminal_qualifications": ["qualified", "unqualified"],
            "row_required": _schema()["row_required"],
            "norm_fields": list(NORM_FIELDS),
            "norm_regions": list(NORM_REGIONS),
            "norm_statistics": list(NORM_STATISTICS),
        },
        "preregistration": {
            "measurement": (
                "ForwardProfile production-route terminal rows and a reduced-rung "
                "public-intermediate census of the first frozen-mask linear action"
            ),
            "cases": list(CASE_NAMES),
            "requested_cells": list(REQUESTED_CELLS),
            "terminal_residual_bound": TERMINAL_RESIDUAL_BOUND,
            "solver_route": (
                "profile.solve(seed, route='newton_krylov', "
                "target_current=Ip_closed_form)"
            ),
            "seed_factory": "profile.cold_seed_portfolio",
            "source_revision": _source_revision(),
            "solver_source_modified": False,
            "full_ladder_policy": (
                "deferred until the non-finite first linear action is understood"
            ),
        },
        "reduced_oracle_registry_reproduction": _registry_reproduction(),
        "cases": case_payload,
        "nan_census": {
            "requested_cells": -110,
            "cases": diagnostics,
            "headline": headline,
        },
        "scheduler_evidence": _scheduler_receipt(scheduler_job_ids or []),
        "verdict": {
            "schema_valid": True,
            "headline": headline,
            "banked_row_count": sum(
                len(case["rows"]) for case in case_payload.values()
            ),
            "missing_rows": missing_rows,
            "full_sixteen_row_ladder_complete": not missing_rows,
            "full_ladder_deferred": True,
            "production_nonfinite_action_cases": production_nonfinite,
            "near_root_finite_action_cases": near_root_finite,
            "production_first_nonfinite_intermediates": production_first,
            "near_root_first_nonfinite_intermediates": near_root_first,
            "admissible_lambda_amplitude_count": admissible_amplitudes,
        },
    }
    _validate_partial(receipt)
    _write_json(output, receipt)
    return receipt


def _validate_partial(receipt: dict[str, Any]) -> None:
    """Validate the banked partial receipt and its complete reduced census."""

    for name in receipt["schema"]["required"]:
        if name not in receipt:
            raise RuntimeError(f"partial certificate is missing {name}")
    row_count = 0
    for case_name in CASE_NAMES:
        if case_name not in receipt["cases"]:
            raise RuntimeError(f"partial certificate is missing case {case_name}")
        for row in receipt["cases"][case_name]["rows"]:
            _validate_row(row)
            row_count += 1
        diagnostic = receipt["nan_census"]["cases"].get(case_name)
        if diagnostic is None or len(diagnostic.get("arms", [])) != 2:
            raise RuntimeError(f"NaN census is incomplete for {case_name}")
    if row_count != receipt["verdict"]["banked_row_count"]:
        raise RuntimeError("banked row count does not match the receipt")
    registry = receipt["reduced_oracle_registry_reproduction"]
    if registry["registry_entry_count"] != 14 or not registry["all_bounds_reproduced"]:
        raise RuntimeError("the locked recovery registry was not reproduced")


def _aggregate(output: Path = OUTPUT) -> dict[str, Any]:
    case_payload = {}
    for case_name in CASE_NAMES:
        rows = [
            json.loads(_part_path(case_name, requested).read_text(encoding="utf-8"))
            for requested in REQUESTED_CELLS
        ]
        for row in rows:
            _validate_row(row)
        fits = {}
        for field in NORM_FIELDS:
            fits[field] = {}
            for region in NORM_REGIONS:
                fits[field][region] = {
                    statistic: _fit_quantity(rows, field, region, statistic)
                    for statistic in NORM_STATISTICS
                }
        qualified_count = sum(
            row["solver"]["qualification"] == "qualified" for row in rows
        )
        case_payload[case_name] = {
            "rows": rows,
            "convergence_order_fits": fits,
            "verdict": {
                "sentence": (
                    f"The production route qualified {qualified_count} "
                    f"of {len(rows)} {case_name} resolution rows against the "
                    f"{TERMINAL_RESIDUAL_BOUND:.0e} terminal residual bound."
                )
            },
        }
    receipt = {
        "schema": _schema(),
        "preregistration": {
            "measurement": (
                "ForwardProfile production solves against independently evaluated "
                "closed-form total flux"
            ),
            "cases": list(CASE_NAMES),
            "requested_cells": list(REQUESTED_CELLS),
            "terminal_residual_bound": TERMINAL_RESIDUAL_BOUND,
            "qualification_policy": (
                "retain every terminal row and fit each qualification cohort separately"
            ),
            "boundary_band": "Euclidean distance at most two characteristic pitches",
            "theoretical_discrete_operator_order": THEORETICAL_ORDER,
            "gauge": "shared exact exterior; no post-solve re-zeroing",
            "reuse_map": (
                "docs/research/forward-accuracy-reuse-map.html#absolute-reference"
            ),
            "cited_reuse_rows": list(REUSE_MAP_ROWS),
            "lane_policy": {
                "fine": (
                    "H200 betelgeuse reservation gpu_0003_grpA with "
                    "JAX_PLATFORMS=cuda,cpu"
                ),
                "other": "CPU lane used by the closed-form recovery gates",
            },
            "source_revision": _source_revision(),
            "solver_source_modified": False,
            "solver_route": (
                "profile.solve(seed, route='newton_krylov', "
                "target_current=Ip_closed_form)"
            ),
            "seed_factory": "profile.cold_seed_portfolio",
            "machine_factory": "cached_machine at every rung including reduced",
        },
        "aggregation_lane": {
            **_lane(),
            "exit_marker": "SOLOVEV_AGGREGATE_EXIT=0",
        },
        "reduced_oracle_registry_reproduction": _registry_reproduction(),
        "cases": case_payload,
        "verdict": {},
    }
    qualifications = [
        row["solver"]["qualification"]
        for case in case_payload.values()
        for row in case["rows"]
    ]
    receipt["verdict"] = {
        "schema_valid": True,
        "case_count": len(case_payload),
        "resolution_rows": len(qualifications),
        "qualified_rows": qualifications.count("qualified"),
        "unqualified_rows": qualifications.count("unqualified"),
        "all_rows_retained": len(qualifications)
        == len(CASE_NAMES) * len(REQUESTED_CELLS),
        "all_locked_recovery_bounds_reproduced": receipt[
            "reduced_oracle_registry_reproduction"
        ]["all_bounds_reproduced"],
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def _validate(receipt: dict[str, Any]) -> None:
    schema = receipt["schema"]
    for name in schema["required"]:
        if name not in receipt:
            raise RuntimeError(f"certificate is missing {name}")
    if len(receipt["cases"]) < schema["minimum_cases"]:
        raise RuntimeError("certificate has too few cases")
    expected_rows = set(REQUESTED_CELLS)
    for case_name, case in receipt["cases"].items():
        rows = case["rows"]
        if len(rows) < schema["minimum_resolutions_per_case"]:
            raise RuntimeError(f"{case_name} has too few resolutions")
        if {row["requested_cells"] for row in rows} != expected_rows:
            raise RuntimeError(f"{case_name} has an incomplete resolution ladder")
        for row in rows:
            _validate_row(row)
        for field in NORM_FIELDS:
            for region in NORM_REGIONS:
                for statistic in NORM_STATISTICS:
                    if statistic not in case["convergence_order_fits"][field][region]:
                        raise RuntimeError("a per-norm convergence fit is missing")
    registry = receipt["reduced_oracle_registry_reproduction"]
    if registry["registry_entry_count"] != 14 or not registry["all_bounds_reproduced"]:
        raise RuntimeError("the locked recovery registry was not reproduced")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--requested-cells", type=int, choices=REQUESTED_CELLS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--aggregate-partial", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-partial", action="store_true")
    parser.add_argument("--seed-control", action="store_true")
    parser.add_argument("--validate-seed-control", action="store_true")
    parser.add_argument("--nan-census", action="store_true")
    parser.add_argument("--scheduler-job-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.nan_census:
        if arguments.case is None:
            raise SystemExit("--nan-census requires one --case")
        receipt = _nan_census(arguments.case)
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        return
    if arguments.seed_control:
        output = SEED_CONTROL_OUTPUT if arguments.output == OUTPUT else arguments.output
        receipt = _seed_control(output)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.validate_seed_control:
        output = SEED_CONTROL_OUTPUT if arguments.output == OUTPUT else arguments.output
        receipt = json.loads(output.read_text(encoding="utf-8"))
        _validate_seed_control(receipt)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.aggregate:
        receipt = _aggregate(arguments.output)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.aggregate_partial:
        receipt = _aggregate_partial(
            arguments.output,
            scheduler_job_ids=arguments.scheduler_job_id,
        )
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.validate:
        receipt = json.loads(arguments.output.read_text(encoding="utf-8"))
        _validate(receipt)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.validate_partial:
        receipt = json.loads(arguments.output.read_text(encoding="utf-8"))
        _validate_partial(receipt)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.case is None or arguments.requested_cells is None:
        raise SystemExit("one --case and --requested-cells pair is required")
    row = _measure(arguments.case, arguments.requested_cells)
    print(
        json.dumps(
            {
                "case": row["case"],
                "requested_cells": row["requested_cells"],
                "qualification": row["solver"]["qualification"],
                "terminal_residual": row["solver"]["terminal_fixed_point_residual"],
                "part": str(_part_path(row["case"], row["requested_cells"])),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
