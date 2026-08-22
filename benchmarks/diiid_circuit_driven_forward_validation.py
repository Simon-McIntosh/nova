"""Compare circuit-complete and shipped-only DIII-D forward solves.

The cohort is selected without consulting a solve score.  It contains the first
five lexicographic, polarity-screened shots with a finite diverted median label
and at least 50 kA absolute recorded plasma current, and excludes every shot
used by the fixed-wiring calibration.  Label flux is used for the prescribed
source functions, branch seed, and scoring only.  The conductor-current path
receives competition magnetics channels and geometry, so no label-derived
current can enter either arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_current_pinned_forward as current_pinned
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _eligible_frame,
    _plasma_mask,
    _read,
    _separatrix,
    build_profile,
    canonical_axes,
    contour_separation,
    gauge_metrics,
)
from benchmarks.diiid_state_of_play_figures import boundary_gradient_minimum
from nova.equilibrium import (
    BranchAdmissibility,
    SelectionHistory,
    SelectionPolicy,
    select_forward_branch,
)
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_current import (
    complete_profile_current_adapter,
    shipped_current_at,
)
from nova.imas.diiid_description import (
    PF_ACTIVE_CIRCUIT,
    POLOIDAL_CONDUCTORS,
    dataset_machine_description,
    geometry_digest,
)
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(
    "docs/figures/coil-circuit-discovery/circuit-driven-forward-validation"
)
CALIBRATION_RECEIPT = Path(
    "docs/figures/coil-circuit-discovery/grid_residual_current_regression_receipt.json"
)
POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
RECEIPT_NAME = "circuit_driven_forward_validation_receipt.json"
FIGURE_NAME = "circuit_driven_forward_validation_overlay.png"
FRAME_COUNT = 5
RECORDED_PLASMA_CURRENT_FLOOR_A = 50_000.0
LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS = 0.0429
GLOBALIZED_RESIDUAL_TOLERANCE = 1.0e-8
NEWTON_OPTIONS = {
    "newton_steps": 12,
    "gmres_iterations": 12,
    "warmup": 0,
    "relaxation": 0.5,
    "step_cap": 10.0,
}
ANDERSON_OPTIONS = {
    "evaluations": 168,
    "relaxation": 0.5,
    "depth": 3,
    "warmup": 6,
    "step_cap": 2.0,
    "ridge": 1.0e-10,
}
ROUTE_NAMES = (
    "current_pinned_eliminated",
    "anderson_portfolio",
    "newton_krylov_portfolio",
)


@dataclass(frozen=True)
class SelectedFrame:
    """One score-independent out-of-cohort diverted frame."""

    path: Path
    frame: int
    time_ms: float
    recorded_plasma_current_a: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_population(
    path: Path = CALIBRATION_RECEIPT,
) -> tuple[set[tuple[str, int]], set[str], dict[str, Any]]:
    """Return the exact fixed-wiring calibration population and its proof."""

    receipt = json.loads(path.read_text())
    records = receipt["records"]
    exact = {(str(item["shot"]), int(item["frame"])) for item in records}
    shots = {shot for shot, _frame in exact}
    selection = receipt["selection"]
    if len(records) != 60 or selection["frames"] != 60:
        raise RuntimeError("the fixed-wiring calibration bank no longer has 60 frames")
    if len(shots) != 20 or selection["shots"] != 20:
        raise RuntimeError("the fixed-wiring calibration bank no longer has 20 shots")
    return exact, shots, receipt


def polarity_population(path: Path = POLARITY_RECEIPT) -> set[str]:
    """Return the complete banked current-polarity exclusion population."""

    receipt = json.loads(path.read_text())
    census = receipt["full_corpus_census"]
    affected = {str(name) for name in census["affected_shots"]}
    if census["shot_count"] != 7_041 or len(affected) != 603:
        raise RuntimeError("the polarity census no longer carries 7,041/603 shots")
    return affected


def select_frames(
    paths: list[Path],
    calibration_shots: set[str],
    polarity_affected: set[str],
    count: int = FRAME_COUNT,
) -> tuple[list[SelectedFrame], list[SelectedFrame]]:
    """Take qualified frames and retain earlier low-current exclusions."""

    selected: list[SelectedFrame] = []
    low_current: list[SelectedFrame] = []
    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *current_pinned.PLASMA_CURRENT_COLUMNS))
    )
    for path in sorted(paths):
        if path.name in calibration_shots or path.name in polarity_affected:
            continue
        row = _read(path, columns)
        frame = _eligible_frame(row)
        if frame is None:
            continue
        time_ms = float(row["efit_times"][frame])
        target_current_a = current_pinned._target_current(row, time_ms)
        candidate = SelectedFrame(
            path,
            frame,
            time_ms,
            target_current_a,
        )
        if abs(target_current_a) < RECORDED_PLASMA_CURRENT_FLOOR_A:
            low_current.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} admissible frames were found")
    return selected, low_current


def _initial_amplitude_guard_row(selected: SelectedFrame) -> dict[str, Any]:
    """Measure the landed amplitude guard at one excluded label seed."""

    columns = tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *_CURRENT_COLUMNS,
                *current_pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    row = _read(selected.path, columns)
    row["_source_path"] = str(selected.path)
    profile, seed, _label, _wall, _reliable, _statement = build_profile(
        row,
        selected.frame,
        REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    )
    unscaled_current_a = float(
        np.sum(
            np.asarray(
                profile.operator.cell_current(jnp.asarray(seed), TopologyClass.DIVERTED)
            )
        )
    )
    amplitude = selected.recorded_plasma_current_a / unscaled_current_a
    try:
        current_pinned._lambda_value(
            selected.recorded_plasma_current_a,
            unscaled_current_a,
        )
        guard_triggered = False
        termination = None
    except current_pinned.LambdaOutOfBand as error:
        guard_triggered = True
        termination = str(error)
        if error.value != amplitude:
            raise RuntimeError("the measured guard value differs from the ratio")
    return {
        "shot": selected.path.name,
        "frame": selected.frame,
        "time_ms": selected.time_ms,
        "recorded_plasma_current_a": selected.recorded_plasma_current_a,
        "absolute_current_a": abs(selected.recorded_plasma_current_a),
        "qualification_floor_a": RECORDED_PLASMA_CURRENT_FLOOR_A,
        "qualified": False,
        "seed_unscaled_plasma_current_a": unscaled_current_a,
        "requested_profile_amplitude": amplitude,
        "amplitude_band": list(current_pinned.LAMBDA_BAND),
        "amplitude_guard_triggered": guard_triggered,
        "termination": termination,
        "iterations": 0,
        "fixed_point_relative_residual": None,
        "residual_status": (
            "not evaluated because the initial amplitude guard fired"
            if guard_triggered
            else "not evaluated because the frame failed current qualification"
        ),
    }


def _strict_float(value: Any) -> float | None:
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _distribution(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if finite.size == 0:
        return {"minimum": None, "median": None, "maximum": None, "mean": None}
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _current_receipt(adapter, current_a: np.ndarray) -> dict[str, Any]:
    rows = []
    for declaration, value, uncertainty in zip(
        adapter.resolution.declarations,
        current_a,
        adapter.resolution.prescribed_standard_deviation_a,
        strict=True,
    ):
        relation = declaration.relation
        rows.append(
            {
                "name": declaration.name,
                "value_a_turn": float(value),
                "tier": declaration.tier.value,
                "authority": declaration.provenance,
                "relation_source": None if relation is None else relation.source,
                "relation_scale": None if relation is None else relation.scale,
                "uncertainty_a_turn": float(uncertainty),
                "relation_provenance": (
                    None if relation is None else relation.provenance
                ),
            }
        )
    return {
        "response_order": list(adapter.resolution.names),
        "complete_count": len(current_a),
        "unknown_parameter_count": len(adapter.resolution.unknown_indices),
        "all_finite": bool(np.all(np.isfinite(current_a))),
        "conductors": rows,
        "response": adapter.response_receipt,
    }


def _label_boundary(row: dict[str, Any], frame: int) -> np.ndarray:
    count = int(row["efit_lcfs_n"][frame])
    return np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        )
    )


def _score_state(
    profile,
    state: np.ndarray,
    label: np.ndarray,
    row: dict[str, Any],
    frame: int,
    converged: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Score one terminal state, qualifying geometry only after convergence."""

    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    state = np.asarray(state, dtype=float)
    predicted = state[: profile.lattice.node_count].reshape(profile.lattice.shape)
    _masks, topology = profile.operator.read(jnp.asarray(state))
    interior = _plasma_mask(row, frame, radius, height)
    r_squared, fractional_rms, gauge, aligned = gauge_metrics(
        label, predicted, interior
    )
    label_boundary = _label_boundary(row, frame)
    predicted_boundary = _separatrix(
        radius,
        height,
        predicted,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    boundary_mean, boundary_maximum = contour_separation(
        predicted_boundary, label_boundary
    )
    full_radius, full_height = canonical_axes(row)
    label_x = boundary_gradient_minimum(
        full_radius,
        full_height,
        np.asarray(row["efit_psirz"][frame], dtype=float),
        label_boundary,
    )
    solved_x = np.asarray(topology.x_point, dtype=float)
    x_separation = float(np.linalg.norm(solved_x - label_x))
    metrics = {
        "interior_r_squared": _strict_float(r_squared),
        "fractional_flux_rms": _strict_float(fractional_rms),
        "additive_gauge_wb": _strict_float(gauge),
        "boundary_mean_separation_m": (
            _strict_float(boundary_mean / 1000.0) if converged else None
        ),
        "boundary_maximum_separation_m": (
            _strict_float(boundary_maximum / 1000.0) if converged else None
        ),
        "x_point_separation_m": _strict_float(x_separation) if converged else None,
        "within_label_representability_ceiling": bool(
            converged
            and np.isfinite(fractional_rms)
            and fractional_rms <= LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS
        ),
    }
    fields = {
        "radius": radius,
        "height": height,
        "aligned": aligned,
        "boundary": predicted_boundary,
        "x_point": solved_x,
    }
    terminal = {
        "finite": bool(np.all(np.isfinite(state))),
        "diverted": bool(topology.diverted),
        "axis_rz_m": [float(value) for value in np.asarray(topology.axis)],
        "x_point_rz_m": (
            [float(value) for value in solved_x]
            if np.all(np.isfinite(solved_x))
            else None
        ),
        "label_x_point_rz_m": [float(value) for value in label_x],
        "unscaled_profile_plasma_current_a": _strict_float(
            np.sum(
                np.asarray(
                    profile.operator.cell_current(
                        jnp.asarray(state), TopologyClass.DIVERTED
                    )
                )
            )
        ),
    }
    return metrics, fields, terminal


def _branch_record(branch) -> dict[str, Any]:
    equilibrium = branch.equilibrium
    trace = np.asarray(equilibrium.fixed_point.trace, dtype=float)
    return {
        "requested_class": ("diverted" if int(branch.requested_class) else "limited"),
        "achieved_class": "diverted" if int(branch.achieved_class) else "limited",
        "topology_consistent": bool(branch.topology_consistent),
        "converged": bool(branch.converged),
        "residual": _strict_float(branch.residual),
        "iterations": int(branch.iterations),
        "finite": bool(equilibrium.finite.passed),
        "residual_trajectory": [float(value) for value in trace if np.isfinite(value)],
    }


def _globalized_route(
    profile,
    seed: np.ndarray,
    current: np.ndarray,
    label: np.ndarray,
    row: dict[str, Any],
    frame: int,
    route: str,
    options: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one landed two-class portfolio and apply pure branch selection."""

    portfolio = profile.solve_portfolio(
        jnp.stack((jnp.asarray(seed), jnp.asarray(seed))),
        route=route,
        current=jnp.asarray(current),
        tolerance=GLOBALIZED_RESIDUAL_TOLERANCE,
        **options,
    )
    policy = SelectionPolicy(
        cold_start_class=TopologyClass.DIVERTED,
        persistence_threshold=3,
    )
    selection = select_forward_branch(
        portfolio,
        SelectionHistory(),
        policy,
        BranchAdmissibility(limited=False, diverted=True),
    )
    branches = {}
    for topology_class in (TopologyClass.LIMITED, TopologyClass.DIVERTED):
        branch = jax.tree.map(
            lambda value: value[int(topology_class)], portfolio.branches
        )
        branches[topology_class.name.lower()] = _branch_record(branch)
    diverted = jax.tree.map(
        lambda value: value[int(TopologyClass.DIVERTED)], portfolio.branches
    )
    converged = bool(diverted.converged)
    metrics, fields, terminal = _score_state(
        profile,
        np.asarray(diverted.equilibrium.flux),
        label,
        row,
        frame,
        converged,
    )
    selection_record = selection.as_dict()
    selection_record["residuals"] = {
        name: _strict_float(value)
        for name, value in selection_record["residuals"].items()
    }
    route_id = f"{route}_portfolio"
    return (
        {
            "route_id": route_id,
            "entry_point": "ForwardProfile.solve_portfolio",
            "route": route,
            "options": options,
            "residual_tolerance": GLOBALIZED_RESIDUAL_TOLERANCE,
            "converged": converged,
            "fixed_point_relative_residual": _strict_float(diverted.residual),
            "iterations": int(diverted.iterations),
            "residual_trajectory": branches["diverted"]["residual_trajectory"],
            "requested_class": "diverted",
            "achieved_class": branches["diverted"]["achieved_class"],
            "topology_consistent": branches["diverted"]["topology_consistent"],
            "portfolio_branches": branches,
            "branch_selection": selection_record,
            "terminal_state": terminal,
            "metrics": metrics,
        },
        fields,
    )


def _current_pinned_route(
    profile,
    seed: np.ndarray,
    current: np.ndarray,
    target_current_a: float,
    label: np.ndarray,
    row: dict[str, Any],
    frame: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the landed plasma-current elimination on a diverted map."""

    try:
        solved = current_pinned.solve_eliminated(
            profile, seed, current, target_current_a
        )
    except current_pinned.LambdaOutOfBand as error:
        metrics, fields, terminal = _score_state(
            profile,
            seed,
            label,
            row,
            frame,
            False,
        )
        terminal["target_plasma_current_a"] = target_current_a
        terminal["profile_amplitude"] = error.value
        return (
            {
                "route_id": "current_pinned_eliminated",
                "entry_point": "diiid_current_pinned_forward.solve_eliminated",
                "route": "host_newton_krylov_with_exact_current_elimination",
                "options": {
                    "outer_iterations": current_pinned.HOST_OUTER_ITERATIONS,
                    "inner_iterations": current_pinned.HOST_INNER_ITERATIONS,
                    "line_search": "armijo",
                },
                "residual_tolerance": current_pinned.RELATIVE_RESIDUAL_CRITERION,
                "current_tolerance": current_pinned.CURRENT_CONSTRAINT_CRITERION,
                "converged": False,
                "fixed_point_relative_residual": None,
                "residual_status": (
                    "not evaluated because the initial amplitude guard fired"
                ),
                "iterations": 0,
                "map_evaluations": 0,
                "rejected_trial_evaluations": 1,
                "residual_trajectory": [],
                "requested_class": "diverted",
                "achieved_class": "diverted" if terminal["diverted"] else "limited",
                "topology_consistent": bool(terminal["diverted"]),
                "branch_selection": {
                    "selected_class": None,
                    "reason": "initial_amplitude_guard",
                },
                "termination": str(error),
                "lambda_guard_triggered": True,
                "lambda_guard_value": error.value,
                "terminal_state": terminal,
                "metrics": metrics,
            },
            fields,
        )
    residual = float(solved["relative_residual"])
    current_error = float(solved["current_relative_error"])
    finite = bool(np.all(np.isfinite(solved["state"])))
    converged = bool(
        finite
        and solved["topology"] == "diverted"
        and np.isfinite(residual)
        and residual <= current_pinned.RELATIVE_RESIDUAL_CRITERION
        and np.isfinite(current_error)
        and current_error <= current_pinned.CURRENT_CONSTRAINT_CRITERION
        and not solved["lambda_guard_triggered"]
    )
    metrics, fields, terminal = _score_state(
        profile,
        solved["state"],
        label,
        row,
        frame,
        converged,
    )
    terminal["target_plasma_current_a"] = target_current_a
    terminal["current_relative_error"] = _strict_float(current_error)
    terminal["profile_amplitude"] = _strict_float(solved["amplitude"])
    return (
        {
            "route_id": "current_pinned_eliminated",
            "entry_point": "diiid_current_pinned_forward.solve_eliminated",
            "route": "host_newton_krylov_with_exact_current_elimination",
            "options": {
                "outer_iterations": current_pinned.HOST_OUTER_ITERATIONS,
                "inner_iterations": current_pinned.HOST_INNER_ITERATIONS,
                "line_search": "armijo",
            },
            "residual_tolerance": current_pinned.RELATIVE_RESIDUAL_CRITERION,
            "current_tolerance": current_pinned.CURRENT_CONSTRAINT_CRITERION,
            "converged": converged,
            "fixed_point_relative_residual": _strict_float(residual),
            "residual_status": "evaluated",
            "iterations": int(solved["iterations"]),
            "map_evaluations": int(solved["map_evaluations"]),
            "rejected_trial_evaluations": int(solved["rejected_trial_evaluations"]),
            "residual_trajectory": [
                float(value) for value in solved["residual_history"]
            ],
            "requested_class": "diverted",
            "achieved_class": solved["topology"],
            "topology_consistent": solved["topology"] == "diverted",
            "branch_selection": {
                "selected_class": "diverted" if converged else None,
                "reason": "exact_current_elimination_diverted_pin",
            },
            "termination": solved["termination"],
            "lambda_guard_triggered": bool(solved["lambda_guard_triggered"]),
            "lambda_guard_value": (
                None
                if solved["lambda_guard_value"] is None
                else _strict_float(solved["lambda_guard_value"])
            ),
            "terminal_state": terminal,
            "metrics": metrics,
        },
        fields,
    )


def _solve_repertoire(
    profile,
    seed: np.ndarray,
    current: np.ndarray,
    target_current_a: float,
    label: np.ndarray,
    row: dict[str, Any],
    frame: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """Compose the current-pinned and globalized routes on one arm."""

    routes = {}
    fields = {}
    route_record, route_fields = _current_pinned_route(
        profile,
        seed,
        current,
        target_current_a,
        label,
        row,
        frame,
    )
    routes[route_record["route_id"]] = route_record
    fields[route_record["route_id"]] = route_fields
    for name, options in (
        ("anderson", ANDERSON_OPTIONS),
        ("newton_krylov", NEWTON_OPTIONS),
    ):
        route_record, route_fields = _globalized_route(
            profile,
            seed,
            current,
            label,
            row,
            frame,
            name,
            options,
        )
        routes[route_record["route_id"]] = route_record
        fields[route_record["route_id"]] = route_fields
    if tuple(routes) != ROUTE_NAMES:
        raise RuntimeError("the landed solve repertoire is incomplete")
    return routes, fields


def solve_frame(
    selected: SelectedFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the shared labelled source and run both route repertoires."""

    columns = tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *_CURRENT_COLUMNS,
                *current_pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    row = _read(selected.path, columns)
    row["_source_path"] = str(selected.path)
    profile, seed, label, wall, reliable, wall_statement = build_profile(
        row, selected.frame, REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )

    current_row = {
        name: row[name]
        for name in (
            *_GEOMETRY_COLUMNS,
            *_CURRENT_COLUMNS,
            *current_pinned.PLASMA_CURRENT_COLUMNS,
            "efit_grid_R",
            "efit_grid_Z",
        )
    }
    current_row["_source_path"] = str(selected.path)
    description = dataset_machine_description(
        current_row, source_row=str(selected.path)
    ).physical
    shipped = shipped_current_at(
        current_row,
        description,
        POLOIDAL_CONDUCTORS,
        selected.time_ms,
    )
    shipped_vector = np.asarray([shipped[name] for name in POLOIDAL_CONDUCTORS])
    np.testing.assert_allclose(
        np.asarray(profile.operator.external_current),
        shipped_vector,
        rtol=0.0,
        atol=1.0e-9,
    )
    adapter = complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
        use_circuit=True,
    )
    circuit_vector = adapter.resolution.current(())
    if len(circuit_vector) != 24 or adapter.resolution.unknown_names:
        raise RuntimeError("the circuit did not prescribe all 24 conductor currents")

    target_current_a = current_pinned._target_current(current_row, selected.time_ms)
    if target_current_a != selected.recorded_plasma_current_a:
        raise RuntimeError("the selected plasma-current target changed before solving")
    shipped_routes, shipped_fields = _solve_repertoire(
        profile,
        seed,
        shipped_vector,
        target_current_a,
        label,
        row,
        selected.frame,
    )
    circuit_routes, circuit_fields = _solve_repertoire(
        adapter.profile,
        seed,
        circuit_vector,
        target_current_a,
        label,
        row,
        selected.frame,
    )
    converged_circuit = [
        route for route in circuit_routes.values() if route["converged"]
    ]
    best_circuit = (
        min(
            converged_circuit,
            key=lambda route: route["metrics"]["fractional_flux_rms"],
        )["route_id"]
        if converged_circuit
        else None
    )
    record = {
        "shot": selected.path.name,
        "frame": selected.frame,
        "time_ms": selected.time_ms,
        "source_parquet": str(selected.path),
        "source_parquet_sha256": _sha256(selected.path),
        "geometry_digest": geometry_digest(row),
        "qualification": {
            "finite_diverted_label": True,
            "polarity_screened": True,
            "calibration_frame_member": False,
            "calibration_shot_member": False,
            "reliable_flux_function_surfaces": reliable,
        },
        "source_and_seed": {
            "profile_functions": "extracted from the EFIT label",
            "branch_seed": "EFIT label map in Nova convention",
            "pseudo_wall": wall_statement,
            "pseudo_wall_expansion": REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        },
        "target_plasma_current": {
            "value_a": target_current_a,
            "authority": "same-frame shipped magnetics_plasma_current channel",
            "role": "prescribed current-elimination target, never a label fit",
        },
        "circuit_driven": {
            "current_receipt": _current_receipt(adapter, circuit_vector),
            "routes": circuit_routes,
            "best_converged_route": best_circuit,
        },
        "shipped_only": {
            "current_receipt": {
                "response_order": list(POLOIDAL_CONDUCTORS),
                "complete_count": len(shipped_vector),
                "unknown_parameter_count": 0,
                "all_finite": bool(np.all(np.isfinite(shipped_vector))),
                "conductors": [
                    {
                        "name": name,
                        "value_a_turn": float(shipped[name]),
                        "authority": f"same-frame shipped magnetics_{name} channel",
                    }
                    for name in POLOIDAL_CONDUCTORS
                ],
            },
            "routes": shipped_routes,
        },
    }
    fields = {
        "label": label,
        "label_boundary": _label_boundary(row, selected.frame),
        "wall": wall,
        "shipped": shipped_fields,
        "circuit": circuit_fields,
        "best_circuit_route": best_circuit,
    }
    return record, fields


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-route and best-converged cohort evidence."""

    def arm(name: str) -> dict[str, Any]:
        route_table = {}
        for route_name in ROUTE_NAMES:
            routes = [record[name]["routes"][route_name] for record in records]
            converged = [route for route in routes if route["converged"]]
            route_table[route_name] = {
                "converged_frames": len(converged),
                "failed_frames": len(routes) - len(converged),
                "terminal_residual": _distribution(
                    [route["fixed_point_relative_residual"] for route in routes]
                ),
                "terminal_fractional_flux_rms": _distribution(
                    [route["metrics"]["fractional_flux_rms"] for route in routes]
                ),
                "fractional_flux_rms_on_converged_frames": _distribution(
                    [route["metrics"]["fractional_flux_rms"] for route in converged]
                ),
                "boundary_mean_separation_m_on_converged_frames": _distribution(
                    [
                        route["metrics"]["boundary_mean_separation_m"]
                        for route in converged
                    ]
                ),
                "x_point_separation_m_on_converged_frames": _distribution(
                    [route["metrics"]["x_point_separation_m"] for route in converged]
                ),
                "frames_within_label_representability_ceiling": sum(
                    route["metrics"]["within_label_representability_ceiling"]
                    for route in routes
                ),
            }
        frames_with_any = sum(
            any(route["converged"] for route in record[name]["routes"].values())
            for record in records
        )
        return {
            "routes": route_table,
            "frames_with_any_converged_route": frames_with_any,
            "maximum_frames_converged_by_one_route": max(
                item["converged_frames"] for item in route_table.values()
            ),
        }

    circuit = arm("circuit_driven")
    shipped = arm("shipped_only")
    best_rows = []
    for record in records:
        route_name = record["circuit_driven"]["best_converged_route"]
        if route_name is None:
            best_rows.append(
                {
                    "shot": record["shot"],
                    "frame": record["frame"],
                    "route": None,
                    "converged": False,
                    "fractional_flux_rms": None,
                    "boundary_mean_separation_m": None,
                    "x_point_separation_m": None,
                    "within_label_representability_ceiling": False,
                }
            )
            continue
        route = record["circuit_driven"]["routes"][route_name]
        best_rows.append(
            {
                "shot": record["shot"],
                "frame": record["frame"],
                "route": route_name,
                "converged": True,
                "fractional_flux_rms": route["metrics"]["fractional_flux_rms"],
                "boundary_mean_separation_m": route["metrics"][
                    "boundary_mean_separation_m"
                ],
                "x_point_separation_m": route["metrics"]["x_point_separation_m"],
                "within_label_representability_ceiling": route["metrics"][
                    "within_label_representability_ceiling"
                ],
            }
        )
    converged_best = [item for item in best_rows if item["converged"]]
    return {
        "frame_count": len(records),
        "shot_count": len({record["shot"] for record in records}),
        "circuit_driven": circuit,
        "shipped_only": shipped,
        "best_converged_circuit_per_frame": best_rows,
        "best_converged_circuit_summary": {
            "converged_frames": len(converged_best),
            "fractional_flux_rms": _distribution(
                [item["fractional_flux_rms"] for item in converged_best]
            ),
            "boundary_mean_separation_m": _distribution(
                [item["boundary_mean_separation_m"] for item in converged_best]
            ),
            "x_point_separation_m": _distribution(
                [item["x_point_separation_m"] for item in converged_best]
            ),
            "frames_within_label_representability_ceiling": sum(
                item["within_label_representability_ceiling"] for item in best_rows
            ),
        },
        "qualified_route_score_table": [
            {
                "arm": arm_name,
                "route": route_name,
                **arm_summary["routes"][route_name],
            }
            for arm_name, arm_summary in (
                ("circuit_driven", circuit),
                ("shipped_only", shipped),
            )
            for route_name in ROUTE_NAMES
        ],
    }


def render_overlay(
    records: list[dict[str, Any]], fields: list[dict[str, Any]], path: Path
) -> None:
    """Plot each label beside its best convergence-qualified circuit map."""

    figure, axes = plt.subplots(
        len(records), 2, figsize=(8.0, 2.6 * len(records)), constrained_layout=True
    )
    for row_axes, record, frame_fields in zip(axes, records, fields, strict=True):
        label = frame_fields["label"]
        best_route = frame_fields["best_circuit_route"]
        representative = next(iter(frame_fields["circuit"].values()))
        radius = representative["radius"]
        height = representative["height"]
        best_fields = (
            None if best_route is None else frame_fields["circuit"][best_route]
        )
        finite_parts = [label[np.isfinite(label)]]
        if best_fields is not None:
            finite_parts.append(
                best_fields["aligned"][np.isfinite(best_fields["aligned"])]
            )
        finite = np.concatenate(finite_parts)
        low, high = np.quantile(finite, [0.01, 0.99])
        panels = (
            (label, "EFIT label", None, None),
            (
                None if best_fields is None else best_fields["aligned"],
                (
                    "No converged circuit route"
                    if best_route is None
                    else f"Best circuit: {best_route.replace('_', ' ')}"
                ),
                None if best_fields is None else best_fields["boundary"],
                None if best_fields is None else best_fields["x_point"],
            ),
        )
        for axis, (flux, title, boundary, x_point) in zip(
            row_axes, panels, strict=True
        ):
            if flux is None:
                axis.set_facecolor("#eeeeee")
                axis.text(
                    0.5,
                    0.5,
                    "No convergence-qualified\ncircuit state",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_aspect("equal")
                axis.set_xlabel("R [m]")
                axis.set_ylabel("Z [m]")
                axis.set_title(title, fontsize=9)
                continue
            image = axis.pcolormesh(
                radius,
                height,
                flux.T,
                shading="auto",
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            label_boundary = frame_fields["label_boundary"]
            axis.plot(
                label_boundary[:, 0],
                label_boundary[:, 1],
                color="white",
                linewidth=1.0,
                label="label LCFS",
            )
            if boundary is not None and len(boundary):
                axis.plot(
                    boundary[:, 0],
                    boundary[:, 1],
                    color="tab:red",
                    linestyle="--",
                    linewidth=0.9,
                    label="solve separatrix",
                )
            if x_point is not None and np.all(np.isfinite(x_point)):
                axis.plot(*x_point, marker="x", color="tab:red", markersize=5)
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            axis.set_ylabel("Z [m]")
            axis.set_title(title, fontsize=9)
            figure.colorbar(image, ax=axis, label="total poloidal flux [Wb]")
        row_axes[0].text(
            0.02,
            0.98,
            f"{Path(record['shot']).stem[-8:]} frame {record['frame']}",
            transform=row_axes[0].transAxes,
            va="top",
            fontsize=7,
            color="white",
        )
        if best_route is not None:
            solve = record["circuit_driven"]["routes"][best_route]
            axis = row_axes[1]
            axis.text(
                0.02,
                0.02,
                f"res={solve['fixed_point_relative_residual']:.3e}\n"
                f"flux RMS={solve['metrics']['fractional_flux_rms']:.3%}",
                transform=axis.transAxes,
                fontsize=7,
                color="white",
                va="bottom",
            )
    axes[0, 0].legend(loc="lower left", fontsize=6)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path, frame_count: int = FRAME_COUNT) -> dict[str, Any]:
    """Execute the out-of-cohort paired forward validation."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    calibration_frames, calibration_shots, calibration = calibration_population()
    affected = polarity_population()
    selected, low_current = select_frames(
        list(data.glob("*.parquet")), calibration_shots, affected, frame_count
    )
    low_current_audit = [_initial_amplitude_guard_row(item) for item in low_current]
    guard_failure_rows = [
        item for item in low_current_audit if item["amplitude_guard_triggered"]
    ]
    selected_pairs = {(item.path.name, item.frame) for item in selected}
    if selected_pairs & calibration_frames:
        raise RuntimeError("selected frame overlaps the fixed-wiring calibration bank")
    if {item.path.name for item in selected} & calibration_shots:
        raise RuntimeError("selected shot overlaps the fixed-wiring calibration bank")
    if {item.path.name for item in selected} & affected:
        raise RuntimeError("selected shot overlaps the polarity exclusion population")

    records = []
    fields = []
    for selected_frame in selected:
        record, frame_fields = solve_frame(selected_frame)
        records.append(record)
        fields.append(frame_fields)
    figure_path = output / FIGURE_NAME
    render_overlay(records, fields, figure_path)
    aggregate = summarize(records)
    circuit_summary = aggregate["circuit_driven"]
    shipped_summary = aggregate["shipped_only"]
    receipt = {
        "measurement": (
            "out-of-cohort DIII-D GS forward validation of the fixed-wiring "
            "pf_active circuit"
        ),
        "selection": {
            "rule": (
                "one median eligible diverted frame from each of the first five "
                "lexicographic shots absent from both the polarity population "
                "and every fixed-wiring calibration shot, with absolute recorded "
                "plasma current at least 50 kA; no solve score consulted"
            ),
            "recorded_plasma_current_floor_abs_a": (RECORDED_PLASMA_CURRENT_FLOOR_A),
            "selected_frames": [
                {
                    "shot": item.path.name,
                    "frame": item.frame,
                    "time_ms": item.time_ms,
                    "recorded_plasma_current_a": item.recorded_plasma_current_a,
                }
                for item in selected
            ],
            "selected_frame_count": len(selected),
            "selected_shot_count": len({item.path.name for item in selected}),
            "all_finite_diverted": True,
            "all_polarity_screened": True,
            "all_recorded_plasma_currents_qualified": all(
                abs(item.recorded_plasma_current_a) >= RECORDED_PLASMA_CURRENT_FLOOR_A
                for item in selected
            ),
            "degenerate_current_audit": low_current_audit,
            "guard_failure_rows": guard_failure_rows,
            "calibration_bank": {
                "receipt": str(CALIBRATION_RECEIPT),
                "sha256": _sha256(CALIBRATION_RECEIPT),
                "frame_count": len(calibration_frames),
                "shot_count": len(calibration_shots),
                "selection_receipt_frames": calibration["selection"]["frames"],
                "exact_selected_pair_intersection": [],
                "selected_shot_intersection": [],
                "strictly_outside": True,
            },
            "polarity_bank": {
                "receipt": str(POLARITY_RECEIPT),
                "sha256": _sha256(POLARITY_RECEIPT),
                "affected_shot_count": len(affected),
                "selected_intersection": [],
            },
        },
        "arms": {
            "circuit_driven": (
                "24 response columns and currents: 19 shipped competition "
                "channels plus 5 fixed-wiring pf_active circuit drives"
            ),
            "shipped_only": "the original 19 shipped response columns and currents",
            "shared": (
                "same prescribed EFIT-derived profile functions, label branch seed, "
                "grid, pseudo-wall, shipped plasma-current target and landed route "
                "repertoire"
            ),
        },
        "current_path_audit": {
            "competition_current_channels": [
                f"magnetics_{name}" for name in POLOIDAL_CONDUCTORS
            ],
            "circuit_source_channel": "magnetics_ECOILA",
            "plasma_current_channel": "magnetics_plasma_current",
            "plasma_current_role": (
                "same-frame shipped diagnostic target used only by exact current "
                "elimination"
            ),
            "label_derived_current_reads": 0,
            "per_frame_current_fits": 0,
            "least_squares_updates": 0,
            "unknown_current_parameters": 0,
            "label_use": (
                "prescribed source functions, branch seed and scoring only; current "
                "extraction receives a row containing geometry and magnetics fields"
            ),
        },
        "solver": {
            "route_order": list(ROUTE_NAMES),
            "current_pinned": {
                "entry_point": "diiid_current_pinned_forward.solve_eliminated",
                "relative_residual_tolerance": (
                    current_pinned.RELATIVE_RESIDUAL_CRITERION
                ),
                "current_relative_error_tolerance": (
                    current_pinned.CURRENT_CONSTRAINT_CRITERION
                ),
                "prescribed_topology": "diverted",
            },
            "globalized": {
                "entry_point": (
                    "nova.equilibrium.forward.ForwardProfile.solve_portfolio"
                ),
                "relative_residual_tolerance": GLOBALIZED_RESIDUAL_TOLERANCE,
                "anderson_options": ANDERSON_OPTIONS,
                "newton_krylov_options": NEWTON_OPTIONS,
                "branch_selection": {
                    "entry_point": (
                        "nova.equilibrium.branch_selection.select_forward_branch"
                    ),
                    "cold_start_class": "diverted",
                    "limited_admissible": False,
                    "diverted_admissible": True,
                    "persistence_threshold": 3,
                },
            },
        },
        "comparison": {
            "flux_gauge": "one additive constant over the labelled LCFS interior",
            "label_representability_ceiling_fractional_rms": (
                LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS
            ),
            "ceiling_scope": (
                "a measured label representability floor, not a guarantee that the "
                "free-boundary solver or circuit arm reaches it"
            ),
            "boundary_and_x_point_metrics": "reported only for converged solves",
        },
        "aggregate": aggregate,
        "verdict": {
            "circuit_frames_with_any_converged_route": circuit_summary[
                "frames_with_any_converged_route"
            ],
            "shipped_only_frames_with_any_converged_route": shipped_summary[
                "frames_with_any_converged_route"
            ],
            "maximum_circuit_frames_converged_by_one_route": circuit_summary[
                "maximum_frames_converged_by_one_route"
            ],
            "best_converged_circuit_frames": aggregate[
                "best_converged_circuit_summary"
            ]["converged_frames"],
            "best_converged_circuit_frames_within_representability_ceiling": (
                aggregate["best_converged_circuit_summary"][
                    "frames_within_label_representability_ceiling"
                ]
            ),
            "per_route_failure_table": {
                route_name: {
                    "circuit_converged_frames": circuit_summary["routes"][route_name][
                        "converged_frames"
                    ],
                    "circuit_failed_frames": circuit_summary["routes"][route_name][
                        "failed_frames"
                    ],
                    "shipped_only_converged_frames": shipped_summary["routes"][
                        route_name
                    ]["converged_frames"],
                    "shipped_only_failed_frames": shipped_summary["routes"][route_name][
                        "failed_frames"
                    ],
                }
                for route_name in ROUTE_NAMES
            },
            "three_frame_convergence_floor_reached": bool(
                circuit_summary["maximum_frames_converged_by_one_route"] >= 3
            ),
            "recovery_demonstrated": bool(
                circuit_summary["maximum_frames_converged_by_one_route"] >= 3
                and aggregate["best_converged_circuit_summary"][
                    "frames_within_label_representability_ceiling"
                ]
                == aggregate["best_converged_circuit_summary"]["converged_frames"]
            ),
            "statement": (
                "Recovery requires a convergence-qualified circuit state on at least "
                "three frames and every such best-per-frame state to fall within the "
                "4.29% fractional-RMS representability ceiling. If no individual "
                "route reaches three frames, the per-route failure table is the "
                "finding."
            ),
        },
        "caveats": {
            "label_representability": (
                "The EFIT label has a 4.29% fractional-RMS representability ceiling "
                "under the landed comparison, so smaller discrepancies cannot be "
                "attributed uniquely to conductor currents."
            ),
            "e89_systematic": {
                "name": "end_loop_bundle_normalisation",
                "E89UP_effective_gain_minus_integer_wiring": 0.04569475694961733,
                "E89DN_effective_gain_minus_integer_wiring": 0.04562407643237165,
                "statement": (
                    "The E89 drives retain the measured shared normalisation "
                    "systematic; this study does not reinterpret it as exact wiring."
                ),
            },
            "circuit_closure": (
                "The calibration receipt found only 1 of 60 frames passed its "
                "post-fit closure rule, so label flux retains non-conductor content."
            ),
        },
        "pf_active_circuit": PF_ACTIVE_CIRCUIT.as_record(),
        "frames": records,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "overlay_figure": str(figure_path),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output, arguments.frames)
    print(json.dumps(receipt["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
