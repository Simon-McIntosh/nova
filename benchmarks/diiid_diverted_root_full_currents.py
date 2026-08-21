"""Measure diverted DIII-D roots with shipped and recovered conductor currents.

Five polarity-screened frames and their five directly recovered conductor
currents are read from the landed recovery receipt.  Each frame is solved twice
through the same topology-pinned Newton--Krylov path: once with the nineteen
shipped poloidal conductors (the twentieth shipped channel supplies the
toroidal-field function), and once with the five recovered poloidal conductors
appended.  The recovered values are immutable inputs; this benchmark performs
no fit, current adjustment, or profile adjustment.

Distance from the labelled map is deliberately diagnostic.  A root passes only
when its fixed-point residual is at most 1e-6, every result is finite, and the
unpinned terminal topology read is diverted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from benchmarks.diiid_boundary_current_recovery import (
    NETCDF_DD_VERSION,
    NETCDF_ENTRY,
    OMITTED_COILS,
    POLARITY_RECEIPT,
    RECEIPT_NAME as RECOVERY_RECEIPT_NAME,
    DEFAULT_OUTPUT as RECOVERY_OUTPUT,
    _rectangle_vertices,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _plasma_mask,
    _read,
    build_profile,
)
from nova.biot.polygon import polygon_greens
from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/diverted-root")
PREREGISTRATION_NAME = "diverted_root_full_currents_preregistration.json"
RECEIPT_NAME = "diverted_root_full_currents_receipt.json"
CHECKPOINT_NAME = "diverted_root_full_currents_frames.jsonl"
FIGURE_NAME = "diverted_root_full_currents.png"
FRAME_COUNT = 5
POLARITY_AFFECTED_SHOT_COUNT = 603
FIXED_POINT_CRITERION = 1.0e-6
LABEL_REPRESENTABILITY_CEILING = 0.0429
CURRENT_ARM_NAMES = ("shipped_20_only", "shipped_20_plus_recovered_5")
HOST_OUTER_ITERATIONS = 1000
HOST_INNER_ITERATIONS = 400
PLATEAU_KRYLOV_DIMENSION = 64
_OMITTED_RESPONSE_CACHE: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}


@dataclass(frozen=True)
class FrameInput:
    """One immutable frame and its landed recovered-current vector."""

    shot: str
    frame: int
    recovered_currents_a: tuple[float, ...]


def preregistration() -> dict[str, Any]:
    """Return the fixed input, solve, topology, and diagnostic declaration."""

    return {
        "measurement": "diverted free-boundary root with complete current set",
        "selection": {
            "frames": FRAME_COUNT,
            "source": str(RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME),
            "rule": (
                "the five distinct-shot replacement frames already banked by the "
                "boundary-current recovery measurement"
            ),
            "polarity_screen": (
                "every shot is absent from the landed 603-shot affected population"
            ),
        },
        "current_arms": {
            "control": (
                "nineteen shipped poloidal conductors plus the shipped bcoil "
                "channel used by the toroidal-field source"
            ),
            "full": (
                "the same twenty shipped channels plus ECOILB, E567UP, E567DN, "
                "E89UP and E89DN at the directly recovered values banked per frame"
            ),
            "poloidal_conductor_count": len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS),
            "coefficients_fitted": 0,
            "current_adjustments": 0,
        },
        "solver": {
            "entry_point": "scipy.optimize.newton_krylov over ForwardProfile.flux_map",
            "route": "host_newton_krylov",
            "requested_class": "diverted",
            "relative_residual_criterion": FIXED_POINT_CRITERION,
            "maximum_outer_iterations": HOST_OUTER_ITERATIONS,
            "maximum_inner_gmres_iterations": HOST_INNER_ITERATIONS,
            "budget_multiplier_over_recorded_100_by_40_host_fixture": 10,
            "line_search": "armijo",
            "seed": (
                "the convention-clean labelled map, used only to select the "
                "diverted branch; it is not an equilibrium input"
            ),
        },
        "pass_criterion": (
            "finite terminal receipt AND relative residual <= 1e-6 AND unpinned "
            "terminal topology classified diverted"
        ),
        "label_distance": {
            "measure": (
                "gauge-aligned RMS on the labelled LCFS interior divided by the "
                "whole labelled-map range"
            ),
            "representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
            "role": "diagnostic only; never included in the pass criterion",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_preregistration(output: Path) -> Path:
    """Persist the complete declaration before any solve runs."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk diverted-root preregistration differs from policy")
    path.write_text(encoded)
    return path


def selected_inputs(
    recovery_receipt: dict[str, Any], affected_shots: set[str]
) -> list[FrameInput]:
    """Read exactly the banked, distinct-shot, polarity-screened frame inputs."""

    root = recovery_receipt["root_existence"]["replacement_polarity_screened"]
    if (
        root["frame_count"] < FRAME_COUNT
        or not root["all_shots_screened_free_of_affected_population"]
    ):
        raise RuntimeError("landed recovery receipt lacks five screened frames")
    selected = []
    for record in root["frames"][:FRAME_COUNT]:
        shot = str(record["shot"])
        if shot in affected_shots:
            raise RuntimeError(f"selected shot {shot} is polarity affected")
        currents = record["recovered_currents_a"]
        selected.append(
            FrameInput(
                shot=shot,
                frame=int(record["frame"]),
                recovered_currents_a=tuple(
                    float(currents[name]) for name in OMITTED_COILS
                ),
            )
        )
    if len({item.shot for item in selected}) != FRAME_COUNT:
        raise RuntimeError("the diverted-root cohort must use distinct shots")
    return selected


def _omitted_vertices() -> dict[str, tuple[tuple[np.ndarray, float], ...]]:
    """Read the independent geometry and signed turns for omitted conductors."""

    import imas

    result: dict[str, tuple[tuple[np.ndarray, float], ...]] = {}
    with imas.DBEntry(NETCDF_ENTRY, "r", dd_version=NETCDF_DD_VERSION) as entry:
        active = entry.get("pf_active", autoconvert=False)
        coils = {str(coil.name): coil for coil in active.coil}
        for name in OMITTED_COILS:
            elements = []
            for element in coils[name].element:
                geometry = element.geometry
                geometry_type = int(geometry.geometry_type)
                if geometry_type == 1:
                    vertices = np.column_stack(
                        [
                            np.asarray(geometry.outline.r, dtype=float),
                            np.asarray(geometry.outline.z, dtype=float),
                        ]
                    )
                elif geometry_type == 2:
                    vertices = _rectangle_vertices(geometry)
                else:
                    raise ValueError(
                        f"unsupported geometry type {geometry_type} for {name}"
                    )
                elements.append((vertices, float(element.turns_with_sign)))
            result[name] = tuple(elements)
    return result


def omitted_response(
    coordinates: np.ndarray,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> np.ndarray:
    """Return total-flux response at arbitrary targets for the five conductors."""

    target = np.asarray(coordinates, dtype=float)
    key = (target.shape, np.ascontiguousarray(target).tobytes())
    cached = _OMITTED_RESPONSE_CACHE.get(key)
    if cached is not None:
        return cached
    columns = []
    for name in OMITTED_COILS:
        response = np.zeros(len(target), dtype=float)
        for vertices, turns in geometry[name]:
            response += turns * polygon_greens(target[:, 0], target[:, 1], vertices)[0]
        columns.append(response)
    result = np.column_stack(columns)
    _OMITTED_RESPONSE_CACHE[key] = result
    return result


def append_recovered_conductors(
    profile: ForwardProfile,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> ForwardProfile:
    """Append five response columns without changing any plasma-side operator."""

    grid_response = omitted_response(profile.operator.grid.coordinate, geometry)
    wall_response = omitted_response(profile.operator.wall.coordinate, geometry)
    grid = replace(
        profile.operator.grid,
        source_target=jnp.column_stack(
            (profile.operator.grid.source_target, jnp.asarray(grid_response))
        ),
    )
    wall = replace(
        profile.operator.wall,
        source_target=jnp.column_stack(
            (profile.operator.wall.source_target, jnp.asarray(wall_response))
        ),
    )
    current = jnp.r_[
        profile.operator.external_current,
        jnp.zeros(len(OMITTED_COILS), dtype=profile.operator.external_current.dtype),
    ]
    operator = replace(
        profile.operator,
        grid=grid,
        wall=wall,
        external_current=current,
    )
    return replace(profile, operator=operator)


def current_arms(profile: ForwardProfile, recovered: tuple[float, ...]) -> np.ndarray:
    """Return paired 24-conductor control and full current vectors."""

    shipped = np.asarray(profile.operator.external_current, dtype=float)
    if shipped.size != len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS):
        raise RuntimeError("profile does not carry the complete conductor response")
    control = shipped.copy()
    control[-len(OMITTED_COILS) :] = 0.0
    full = shipped.copy()
    full[-len(OMITTED_COILS) :] = np.asarray(recovered, dtype=float)
    return np.stack((control, full))


class _HostCriterionReached(Exception):
    """Stop the host solve immediately after the registered criterion is met."""

    def __init__(self, state: np.ndarray):
        super().__init__("registered relative residual reached")
        self.state = np.asarray(state, dtype=float)


def _relative_residual(image: np.ndarray, state: np.ndarray) -> float:
    """Return the solver's registered max-norm fixed-point defect."""

    return float(np.max(np.abs(image - state)) / max(np.max(np.abs(image)), 1.0e-30))


def solve_host_pinned(
    profile: ForwardProfile, seed: np.ndarray, current: np.ndarray
) -> dict[str, Any]:
    """Run the large-budget host Newton--Krylov solve on the diverted map."""

    mapped = jax.jit(profile.flux_map(jnp.asarray(current), TopologyClass.DIVERTED))
    evaluations = 0
    accepted_history: list[float] = []
    topology_history: list[str] = []

    def image(state: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        return np.asarray(mapped(jnp.asarray(state)), dtype=float)

    def residual(state: np.ndarray) -> np.ndarray:
        return image(state) - state

    initial = np.asarray(seed, dtype=float)
    accepted_history.append(_relative_residual(image(initial), initial))
    _initial_masks, initial_topology = profile.operator.read(jnp.asarray(initial))
    topology_history.append(
        "diverted" if bool(initial_topology.diverted) else "limited"
    )

    def record(state: np.ndarray, value: np.ndarray) -> None:
        scale = max(np.max(np.abs(state + value)), 1.0e-30)
        relative = float(np.max(np.abs(value)) / scale)
        accepted_history.append(relative)
        _accepted_masks, accepted_topology = profile.operator.read(jnp.asarray(state))
        topology_class = "diverted" if bool(accepted_topology.diverted) else "limited"
        topology_history.append(topology_class)
        print(
            f"HOST_ACCEPTED {len(accepted_history) - 1} "
            f"residual={relative:.12e} topology={topology_class}",
            flush=True,
        )
        if relative <= FIXED_POINT_CRITERION:
            raise _HostCriterionReached(state)

    try:
        terminal = scipy.optimize.newton_krylov(
            residual,
            initial,
            method="gmres",
            inner_maxiter=HOST_INNER_ITERATIONS,
            maxiter=HOST_OUTER_ITERATIONS,
            f_tol=0.0,
            line_search="armijo",
            callback=record,
        )
        termination = "host solver returned before its outer ceiling"
    except _HostCriterionReached as reached:
        terminal = reached.state
        termination = "registered relative residual reached"
    except scipy.optimize.NoConvergence as error:
        terminal = np.asarray(error.args[0], dtype=float)
        termination = "host outer-iteration ceiling exhausted"
    terminal_image = image(terminal)
    relative = _relative_residual(terminal_image, terminal)
    _masks, topology = profile.operator.read(jnp.asarray(terminal))
    return {
        "state": np.asarray(terminal),
        "relative_residual": relative,
        "finite": bool(np.all(np.isfinite(terminal_image + terminal))),
        "diverted": bool(topology.diverted),
        "x_point": np.asarray(topology.x_point, dtype=float),
        "accepted_residual_history": np.asarray(accepted_history, dtype=float),
        "accepted_topology_history": topology_history,
        "accepted_iterations": len(accepted_history) - 1,
        "map_evaluations": evaluations,
        "termination": termination,
    }


def accelerated_plateau(
    profile: ForwardProfile, seed: np.ndarray, current: np.ndarray
) -> dict[str, Any]:
    """Reproduce the fixed-budget accelerator state used by the first attempt."""

    result = fixed_point.newton_krylov(
        profile.flux_map(jnp.asarray(current), TopologyClass.DIVERTED),
        jnp.asarray(seed),
        newton_steps=24,
        gmres_iterations=24,
        warmup=8,
        relaxation=0.5,
        step_cap=10.0,
    )
    _masks, topology = profile.operator.read(result.state)
    return {
        "state": np.asarray(result.state, dtype=float),
        "relative_residual": float(result.residual),
        "trace": np.asarray(result.trace, dtype=float),
        "diverted": bool(topology.diverted),
    }


def residual_jacobian_diagnostic(
    profile: ForwardProfile,
    state: np.ndarray,
    current: np.ndarray,
    *,
    krylov_dimension: int = PLATEAU_KRYLOV_DIMENSION,
) -> dict[str, Any]:
    """Measure exact-tangent Krylov rank and local smoothness at a plateau."""

    state = np.asarray(state, dtype=float)
    trial = jnp.asarray(state)
    mapped = profile.flux_map(jnp.asarray(current), TopologyClass.DIVERTED)
    image, tangent = jax.linearize(mapped, trial)
    residual = image - trial
    residual_np = np.asarray(residual, dtype=float)

    def action(vector: np.ndarray) -> np.ndarray:
        direction = jnp.asarray(vector)
        return np.asarray(direction - tangent(direction), dtype=float)

    residual_norm = float(np.linalg.norm(residual_np))
    basis = np.zeros((state.size, krylov_dimension + 1), dtype=float)
    hessenberg = np.zeros((krylov_dimension + 1, krylov_dimension), dtype=float)
    basis[:, 0] = residual_np / residual_norm
    completed = 0
    nonfinite_action_column: int | None = None
    breakdown_threshold = np.finfo(float).eps * np.sqrt(state.size)
    for column in range(krylov_dimension):
        vector = action(basis[:, column]).copy()
        if not np.all(np.isfinite(vector)):
            nonfinite_action_column = column
            break
        for row in range(column + 1):
            coefficient = float(np.dot(basis[:, row], vector))
            hessenberg[row, column] += coefficient
            vector -= coefficient * basis[:, row]
        for row in range(column + 1):
            coefficient = float(np.dot(basis[:, row], vector))
            hessenberg[row, column] += coefficient
            vector -= coefficient * basis[:, row]
        next_norm = float(np.linalg.norm(vector))
        hessenberg[column + 1, column] = next_norm
        completed = column + 1
        if next_norm <= breakdown_threshold:
            break
        basis[:, column + 1] = vector / next_norm
    projected = hessenberg[: completed + 1, :completed]
    if completed:
        singular = np.linalg.svd(projected, compute_uv=False)
        rank_threshold = singular[0] * max(projected.shape) * np.finfo(state.dtype).eps
        rank = int(np.count_nonzero(singular > rank_threshold))
    else:
        singular = np.empty(0, dtype=float)
        rank_threshold = float("nan")
        rank = 0

    def jax_action(vector):
        return vector - tangent(vector)

    step, info = jax.scipy.sparse.linalg.gmres(
        jax_action,
        residual,
        maxiter=24,
        restart=24,
        solve_method="batched",
    )
    step_np = np.asarray(step, dtype=float)
    step_finite = bool(np.all(np.isfinite(step_np)))
    if step_finite:
        linear_residual = action(step_np) - residual_np
        proposed = state + step_np
        proposed_image = np.asarray(mapped(jnp.asarray(proposed)), dtype=float)
        _proposed_masks, proposed_topology = profile.operator.read(
            jnp.asarray(proposed)
        )
        linear_action_finite = bool(np.all(np.isfinite(linear_residual)))
        linear_relative = (
            float(np.linalg.norm(linear_residual) / residual_norm)
            if linear_action_finite
            else None
        )
        proposed_relative = (
            _relative_residual(proposed_image, proposed)
            if np.all(np.isfinite(proposed_image))
            else None
        )
        proposed_class = "diverted" if bool(proposed_topology.diverted) else "limited"
        maximum_step = float(np.max(np.abs(step_np)))
    else:
        linear_action_finite = False
        linear_relative = None
        proposed_relative = None
        proposed_class = None
        maximum_step = None

    direction = basis[:, 0]
    exact_direction = action(direction)
    exact_direction_finite = bool(np.all(np.isfinite(exact_direction)))
    state_scale = max(float(np.max(np.abs(state))), 1.0e-30)
    smoothness = []
    previous_finite_difference: np.ndarray | None = None
    for fraction in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3):
        delta = fraction * state_scale
        plus = state + delta * direction
        minus = state - delta * direction
        plus_residual = plus - np.asarray(mapped(jnp.asarray(plus)), dtype=float)
        minus_residual = minus - np.asarray(mapped(jnp.asarray(minus)), dtype=float)
        finite_difference = (plus_residual - minus_residual) / (2.0 * delta)
        finite_difference_norm = float(np.linalg.norm(finite_difference))
        change = (
            None
            if previous_finite_difference is None
            else float(
                np.linalg.norm(finite_difference - previous_finite_difference)
                / max(finite_difference_norm, 1.0e-30)
            )
        )
        tangent_error = (
            float(
                np.linalg.norm(finite_difference - exact_direction)
                / max(np.linalg.norm(exact_direction), 1.0e-30)
            )
            if exact_direction_finite
            else None
        )
        _plus_masks, plus_topology = profile.operator.read(jnp.asarray(plus))
        _minus_masks, minus_topology = profile.operator.read(jnp.asarray(minus))
        smoothness.append(
            {
                "relative_state_perturbation": fraction,
                "central_difference_tangent_relative_error": tangent_error,
                "central_difference_finite": bool(
                    np.all(np.isfinite(finite_difference))
                ),
                "central_difference_l2_norm": finite_difference_norm,
                "relative_change_from_previous_scale": change,
                "minus_topology": (
                    "diverted" if bool(minus_topology.diverted) else "limited"
                ),
                "plus_topology": (
                    "diverted" if bool(plus_topology.diverted) else "limited"
                ),
            }
        )
        previous_finite_difference = finite_difference

    finite_difference_fraction = 1.0e-6

    def finite_difference_action(vector: np.ndarray) -> np.ndarray:
        vector_scale = max(float(np.max(np.abs(vector))), 1.0e-30)
        delta = finite_difference_fraction * state_scale / vector_scale
        plus = state + delta * vector
        minus = state - delta * vector
        plus_residual = plus - np.asarray(mapped(jnp.asarray(plus)), dtype=float)
        minus_residual = minus - np.asarray(mapped(jnp.asarray(minus)), dtype=float)
        return (plus_residual - minus_residual) / (2.0 * delta)

    fd_basis = np.zeros_like(basis)
    fd_hessenberg = np.zeros_like(hessenberg)
    fd_basis[:, 0] = residual_np / residual_norm
    fd_completed = 0
    fd_nonfinite_column: int | None = None
    for column in range(krylov_dimension):
        vector = finite_difference_action(fd_basis[:, column]).copy()
        if not np.all(np.isfinite(vector)):
            fd_nonfinite_column = column
            break
        for row in range(column + 1):
            coefficient = float(np.dot(fd_basis[:, row], vector))
            fd_hessenberg[row, column] += coefficient
            vector -= coefficient * fd_basis[:, row]
        for row in range(column + 1):
            coefficient = float(np.dot(fd_basis[:, row], vector))
            fd_hessenberg[row, column] += coefficient
            vector -= coefficient * fd_basis[:, row]
        next_norm = float(np.linalg.norm(vector))
        fd_hessenberg[column + 1, column] = next_norm
        fd_completed = column + 1
        if next_norm <= breakdown_threshold:
            break
        fd_basis[:, column + 1] = vector / next_norm
    fd_projected = fd_hessenberg[: fd_completed + 1, :fd_completed]
    fd_singular = np.linalg.svd(fd_projected, compute_uv=False)
    fd_rank_threshold = (
        fd_singular[0] * max(fd_projected.shape) * np.finfo(state.dtype).eps
    )
    fd_rank = int(np.count_nonzero(fd_singular > fd_rank_threshold))
    return {
        "state_dimension": int(state.size),
        "relative_residual": _relative_residual(np.asarray(image), state),
        "arnoldi": {
            "requested_dimension": krylov_dimension,
            "completed_dimension": completed,
            "breakdown": bool(
                completed < krylov_dimension and nonfinite_action_column is None
            ),
            "first_nonfinite_action_column": nonfinite_action_column,
            "exact_first_direction_finite": exact_direction_finite,
            "projected_numerical_rank": rank,
            "rank_threshold": (
                float(rank_threshold) if np.isfinite(rank_threshold) else None
            ),
            "largest_singular_value": (float(singular[0]) if singular.size else None),
            "smallest_singular_value": (float(singular[-1]) if singular.size else None),
            "projected_condition_number": (
                float(singular[0] / singular[-1]) if singular.size else None
            ),
            "singular_values": singular.tolist(),
        },
        "fixed_inner_gmres": {
            "iterations": 24,
            "info": None if info is None else int(info),
            "finite_step": step_finite,
            "finite_linear_action": linear_action_finite,
            "maximum_absolute_step_wb": maximum_step,
            "relative_linear_residual_l2": linear_relative,
            "proposed_nonlinear_relative_residual": proposed_relative,
            "proposed_topology": proposed_class,
        },
        "finite_difference_arnoldi": {
            "relative_state_perturbation": finite_difference_fraction,
            "requested_dimension": krylov_dimension,
            "completed_dimension": fd_completed,
            "first_nonfinite_action_column": fd_nonfinite_column,
            "projected_numerical_rank": fd_rank,
            "rank_threshold": float(fd_rank_threshold),
            "largest_singular_value": float(fd_singular[0]),
            "smallest_singular_value": float(fd_singular[-1]),
            "projected_condition_number": float(fd_singular[0] / fd_singular[-1]),
            "singular_values": fd_singular.tolist(),
        },
        "central_difference_smoothness": smoothness,
    }


def label_distance(
    labelled: np.ndarray, solved: np.ndarray, interior: np.ndarray
) -> dict[str, float]:
    """Return the registered gauge-aligned diagnostic label distance."""

    selected = np.asarray(interior, dtype=bool) & np.isfinite(labelled + solved)
    gauge = float(np.mean(labelled[selected] - solved[selected]))
    difference = solved[selected] + gauge - labelled[selected]
    rms = float(np.sqrt(np.mean(difference**2)))
    span = float(np.ptp(labelled))
    return {
        "fractional_rms": rms / span,
        "rms_wb": rms,
        "label_range_wb": span,
        "additive_gauge_wb": gauge,
        "representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        "used_as_pass_criterion": False,
    }


def qualify_arm(
    *,
    residual: float,
    finite: bool,
    diverted: bool,
    iterations: int,
    x_point: np.ndarray,
    diagnostic: dict[str, float],
    trace: np.ndarray,
) -> dict[str, Any]:
    """Serialize independent fixed-point, topology, and label diagnostics."""

    fixed_point_converged = bool(
        finite and np.isfinite(residual) and residual <= FIXED_POINT_CRITERION
    )
    simultaneous = bool(fixed_point_converged and diverted)
    point = np.asarray(x_point, dtype=float)
    return {
        "fixed_point": {
            "relative_residual": float(residual),
            "criterion": FIXED_POINT_CRITERION,
            "iterations": int(iterations),
            "finite": bool(finite),
            "converged": fixed_point_converged,
            "residual_trajectory": [
                float(value) if np.isfinite(value) else None for value in trace
            ],
        },
        "topology": {
            "class": "diverted" if diverted else "limited",
            "diverted": bool(diverted),
            "x_point_rz_m": point.tolist() if np.all(np.isfinite(point)) else None,
        },
        "simultaneously_converged_and_diverted": simultaneous,
        "label_map_diagnostic": diagnostic,
    }


def solve_frame(
    row: dict[str, Any],
    frame_input: FrameInput,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> dict[str, Any]:
    """Solve both current arms through one batched production branch path."""

    profile, seed, label, _wall, _reliable, _statement = build_profile(
        row, frame_input.frame, 0.02
    )
    profile = append_recovered_conductors(profile, geometry)
    arms = current_arms(profile, frame_input.recovered_currents_a)
    radius = profile.lattice.radius
    height = profile.lattice.height
    interior = _plasma_mask(row, frame_input.frame, radius, height)
    records = {}
    for index, name in enumerate(CURRENT_ARM_NAMES):
        host = solve_host_pinned(profile, seed, arms[index])
        solved = np.asarray(host["state"][: profile.lattice.node_count]).reshape(
            profile.lattice.shape
        )
        records[name] = qualify_arm(
            residual=host["relative_residual"],
            finite=host["finite"],
            diverted=host["diverted"],
            iterations=host["accepted_iterations"],
            x_point=host["x_point"],
            diagnostic=label_distance(label, solved, interior),
            trace=host["accepted_residual_history"],
        )
        records[name]["fixed_point"].update(
            {
                "map_evaluations": host["map_evaluations"],
                "termination": host["termination"],
                "topology_history": host["accepted_topology_history"],
            }
        )
    return {
        "shot": frame_input.shot,
        "frame": frame_input.frame,
        "time_ms": float(row["efit_times"][frame_input.frame]),
        "screened_out_of_affected_polarity_population": True,
        "shipped_channel_count": 20,
        "poloidal_conductor_count": len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS),
        "recovered_currents_a": dict(
            zip(OMITTED_COILS, frame_input.recovered_currents_a, strict=True)
        ),
        "coefficients_fitted": 0,
        "current_adjustments": 0,
        "seed": {
            "kind": "convention-clean labelled map used only as a branch seed",
            "stored_map_used_as_equilibrium_input": False,
        },
        "arms": records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the paired cohort verdict without using label distance as a gate."""

    full = [item["arms"][CURRENT_ARM_NAMES[1]] for item in records]
    control = [item["arms"][CURRENT_ARM_NAMES[0]] for item in records]
    full_passes = sum(item["simultaneously_converged_and_diverted"] for item in full)
    return {
        "frame_count": len(records),
        "all_shots_screened_free_of_affected_population": all(
            item["screened_out_of_affected_polarity_population"] for item in records
        ),
        "full_current_converged_diverted_frames": int(full_passes),
        "control_converged_diverted_frames": int(
            sum(item["simultaneously_converged_and_diverted"] for item in control)
        ),
        "control_fixed_point_converged_frames": int(
            sum(item["fixed_point"]["converged"] for item in control)
        ),
        "full_current_residuals": [
            item["fixed_point"]["relative_residual"] for item in full
        ],
        "control_residuals": [
            item["fixed_point"]["relative_residual"] for item in control
        ],
        "full_current_label_fractional_rms": [
            item["label_map_diagnostic"]["fractional_rms"] for item in full
        ],
        "label_representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        "label_distance_is_diagnostic_only": True,
        "passed": bool(
            len(records) >= FRAME_COUNT
            and full_passes == len(records)
            and all(
                item["screened_out_of_affected_polarity_population"] for item in records
            )
        ),
        "frames": records,
    }


def _figure(summary: dict[str, Any], path: Path) -> None:
    """Plot paired residual/topology outcomes and diagnostic label distance."""

    frames = summary["frames"]
    labels = [f"{item['shot'][9:17]}:{item['frame']}" for item in frames]
    x = np.arange(len(frames))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for name, color, marker in (
        (CURRENT_ARM_NAMES[0], "#4477aa", "o"),
        (CURRENT_ARM_NAMES[1], "#cc6677", "s"),
    ):
        values = [
            item["arms"][name]["fixed_point"]["relative_residual"] for item in frames
        ]
        axes[0].semilogy(x, values, marker=marker, color=color, label=name)
    axes[0].axhline(FIXED_POINT_CRITERION, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Relative fixed-point residual")
    axes[0].set_title("Root residual by unchanged current arm")
    axes[0].legend(frameon=False, fontsize=8)
    values = summary["full_current_label_fractional_rms"]
    axes[1].bar(x, values, color="#cc6677")
    axes[1].axhline(
        LABEL_REPRESENTABILITY_CEILING,
        color="black",
        linestyle="--",
        label="0.0429 representability ceiling",
    )
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Label-map fractional RMS (diagnostic)")
    axes[1].set_title("Not a root pass criterion")
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path) -> dict[str, Any]:
    """Run the fixed cohort and write checkpoint, receipt, and figure."""

    configure_dtypes()
    declaration = write_preregistration(output)
    recovery_path = RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME
    recovery = json.loads(recovery_path.read_text())
    polarity = json.loads(POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    selected = selected_inputs(recovery, affected)
    geometry = _omitted_vertices()
    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    records = []
    for number, frame_input in enumerate(selected, start=1):
        path = data / frame_input.shot
        row = _read(path, columns)
        row["_source_path"] = str(path)
        record = solve_frame(row, frame_input, geometry)
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        full = record["arms"][CURRENT_ARM_NAMES[1]]
        control = record["arms"][CURRENT_ARM_NAMES[0]]
        print(
            f"SOLVED {number}/{len(selected)} {frame_input.shot}:{frame_input.frame} "
            f"full={full['fixed_point']['relative_residual']:.6e}/"
            f"{full['topology']['class']} control="
            f"{control['fixed_point']['relative_residual']:.6e}/"
            f"{control['topology']['class']}",
            flush=True,
        )
    result = summarize(records)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_path": str(declaration),
        "preregistration_sha256": _sha256(declaration),
        "authorities": {
            "recovery_receipt": str(recovery_path),
            "recovery_receipt_sha256": _sha256(recovery_path),
            "polarity_receipt": str(POLARITY_RECEIPT),
            "polarity_receipt_sha256": _sha256(POLARITY_RECEIPT),
            "affected_shot_count": len(affected),
            "omitted_geometry_entry": str(NETCDF_ENTRY),
            "omitted_geometry_dd_version": NETCDF_DD_VERSION,
        },
        "result": result,
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _figure(result, output / FIGURE_NAME)
    if not result["passed"]:
        raise RuntimeError(
            "fewer than five full-current frames converged below 1e-6 as diverted"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregister-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.preregister_only:
        print(f"PREREGISTERED {write_preregistration(arguments.output)}")
        return
    receipt = run(arguments.data, arguments.output)
    headline = dict(receipt["result"])
    headline.pop("frames", None)
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
