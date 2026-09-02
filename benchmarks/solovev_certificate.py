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
from nova.equilibrium import ForwardProfile, SaddleSeedGeometry
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
OUTPUT = ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate.json"
SEED_CONTROL_OUTPUT = (
    ROOT / "docs/figures/gs-absolute-accuracy/certificate-seed-control.json"
)
FIGURE_ROOT = ROOT / "docs/figures/gs-absolute-accuracy/solovev"
PART_ROOT = FIGURE_ROOT / "parts"
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
    return PART_ROOT / f"{case_name}-{_slug(requested_cells)}.json"


def _figure_path(case_name: str, requested_cells: int) -> Path:
    return FIGURE_ROOT / f"{case_name}-{_slug(requested_cells)}.png"


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
        termination = "nonfinite_terminal_state"
    elif not np.isfinite(terminal_residual):
        termination = "nonfinite_terminal_residual"
    elif qualified:
        termination = "fixed_point_residual_within_qualification_bound"
    else:
        termination = (
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
        "termination": termination,
        "solver_termination": solver_termination,
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
        machine_builder = (
            oracle_fixture.build_machine
            if requested_cells == -110
            else oracle_fixture.cached_machine
        )
        if requested_cells == -110:
            machine = machine_builder(
                carrier_case,
                requested_cells,
                wall_nodes=oracle_fixture.WALL_POINT_COUNT,
            )
            machine.cache.update(
                {
                    "hit": False,
                    "semantic_key": "reduced-oracle-direct-build",
                    "build_seconds": None,
                }
            )
        else:
            machine = machine_builder(
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
        if case_name == "diverted-jump-bearing":
            seed, seed_receipt = _diverted_seed(source_case, machine, operator)
            solver_route = (
                "production axis-saddle cold seed with undamped Newton-Krylov"
            )
        else:
            seed, _moment_image, seed_receipt = recovery._moment_seed(
                source_case, machine, operator
            )
            solver_route = "production moment seed with undamped Newton-Krylov"

    with _timed_stage(
        "compile_and_first_solve",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        compiled = recovery._solve(operator.flux_map(), seed)
    compile_and_solve_seconds = stage_timings["compile_and_first_solve"]
    with _timed_stage(
        "compile_warm_solve",
        stage_timings,
        case_name=case_name,
        requested_cells=requested_cells,
    ):
        terminal = recovery._solve(operator.flux_map(), seed)
    warm_solve_seconds = stage_timings["compile_warm_solve"]
    terminal_state = np.asarray(terminal.state, dtype=np.float64)
    terminal_residual = float(terminal.residual)
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
        mesh = StencilMesh(machine.node, machine.stencil, machine.area)
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
        "lane": _lane(),
        "solver": {
            "route": solver_route,
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "terminal_fixed_point_residual": (
                terminal_residual if np.isfinite(terminal_residual) else None
            ),
            "terminal_fixed_point_residual_status": (
                "finite" if np.isfinite(terminal_residual) else "nonfinite"
            ),
            "termination_reason": termination_reason,
            "qualification_bound": TERMINAL_RESIDUAL_BOUND,
            "qualification": qualification,
            "compile_and_solve_wall_seconds": compile_and_solve_seconds,
            "compile_warm_wall_seconds": warm_solve_seconds,
            "first_run_terminal_residual": (
                float(compiled.residual)
                if np.isfinite(float(compiled.residual))
                else None
            ),
            "seed": seed_receipt,
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
    for field in NORM_FIELDS:
        for region in NORM_REGIONS:
            measured = row["norms"][field][region]
            for name in NORM_STATISTICS:
                if measured[name] is None and not (
                    row["solver"]["qualification"] == "unqualified"
                    and measured["measurement_status"]
                    in {"nonfinite_terminal_error", "unavailable"}
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
        "$id": "nova.solovev-absolute-accuracy-certificate",
        "version": 1,
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
    }


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
        case_payload[case_name] = {
            "rows": rows,
            "convergence_order_fits": fits,
        }
    receipt = {
        "schema": _schema(),
        "preregistration": {
            "measurement": (
                "unchanged production forward solves against independently "
                "evaluated closed-form total flux"
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
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed-control", action="store_true")
    parser.add_argument("--validate-seed-control", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
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
    if arguments.validate:
        receipt = json.loads(arguments.output.read_text(encoding="utf-8"))
        _validate(receipt)
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
