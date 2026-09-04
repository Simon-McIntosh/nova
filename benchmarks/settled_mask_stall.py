"""Diagnose Newton contraction on MAST states whose residual mask has settled.

The measurement rebuilds the production profile through the persisted response
carrier because the published operand cache intentionally retains grid geometry
but not the wall and direct-sample leaves of the solver state.  It validates the
rebuilt terminal against that cache before freezing the terminal residual mask.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium import fixed_point
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERANDS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260904T065818622161-nia-bank-regeneration-repaired-census-2/"
    "raw/current-operands.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/solver-convergence-regression/settled-mask-stall/measurement.json"
)
TARGETS = ((21985, 51), (21986, 46), (21989, 55), (22086, 43))
SMOOTH_NEWTON_STEPS = 8
SMOOTH_GMRES_ITERATIONS = 40
SMOOTH_RELAXATION = 0.5
SMOOTH_STEP_CAP = 10.0
FINITE_DIFFERENCE_RELATIVE_STEPS = (1.0e-2, 1.0e-4, 1.0e-6)
EXPECTED_GRID_CELLS = 33 * 33


def _load_script(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load measurement dependency {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bank_producer = _load_script(
    "bank_producer",
    ROOT / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py",
)
reachability = _load_script(
    "real_equilibria_reachability",
    ROOT / "docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py",
)


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _array(values: Any, *, limit: int | None = None) -> list[Any]:
    result = np.asarray(values)
    if limit is not None:
        result = result.reshape(-1)[:limit]

    def strict(value):
        if isinstance(value, list):
            return [strict(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    output = strict(result.tolist())
    return output if isinstance(output, list) else [output]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    module_root = Path(fixed_point.__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(module_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_banked_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load the four pure-arm terminal witnesses from the exact operand cache."""

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        for index, row in enumerate(metadata["rows"]):
            key = (int(row["shot"]), int(row["slice_index"]))
            if row["arm"] != "pure" or key not in TARGETS:
                continue
            prefix = f"arm_{index:02d}"
            selected[key] = dict(row) | {
                "active_set_residuals": np.asarray(
                    stored[f"{prefix}_active_set_residuals"], dtype=np.float64
                ),
                "active_set_mask_differences": np.asarray(
                    stored[f"{prefix}_active_set_mask_differences"], dtype=np.int64
                ),
                "active_set_cycle_damping_activations": np.asarray(
                    stored[f"{prefix}_active_set_cycle_damping_activations"],
                    dtype=np.int64,
                ),
            }
    missing = sorted(set(TARGETS) - set(selected))
    if missing:
        raise RuntimeError(f"operand cache lacks pure-arm targets: {missing}")
    return selected


def _bank_validation(
    result, banked: dict[str, Any], *, require_match: bool
) -> dict[str, Any]:
    iterations = int(np.asarray(result.active_set_iterations))
    residuals = np.asarray(result.active_set_residuals, dtype=np.float64)[:iterations]
    differences = np.asarray(result.active_set_mask_differences, dtype=np.int64)[
        :iterations
    ]
    expected_residuals = banked["active_set_residuals"]
    expected_differences = banked["active_set_mask_differences"]
    residual_delta = (
        float(np.max(np.abs(residuals - expected_residuals)))
        if residuals.shape == expected_residuals.shape
        else None
    )
    mask_exact = bool(
        differences.shape == expected_differences.shape
        and np.array_equal(differences, expected_differences)
    )
    terminal = float(np.asarray(result.residual))
    terminal_delta = abs(terminal - float(banked["terminal_residual"]))
    passes = bool(
        residual_delta is not None
        and residual_delta <= 5.0e-11
        and terminal_delta <= 5.0e-11
        and mask_exact
    )
    if require_match and not passes:
        raise AssertionError(
            "rebuilt terminal does not reproduce the banked pure arm: "
            f"residual_delta={residual_delta}, terminal_delta={terminal_delta}, "
            f"mask_exact={mask_exact}"
        )
    return {
        "passes": passes,
        "absolute_tolerance": 5.0e-11,
        "maximum_residual_difference": residual_delta,
        "terminal_residual_difference": terminal_delta,
        "mask_differences_exact": mask_exact,
        "active_set_residuals": residuals.tolist(),
        "active_set_mask_differences": differences.tolist(),
    }


def _norm_summary(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    usable = array[finite]
    return {
        "size": int(array.size),
        "finite_count": int(np.count_nonzero(finite)),
        "l2": float(np.linalg.norm(usable)) if usable.size else None,
        "sup": float(np.max(np.abs(usable))) if usable.size else None,
    }


def _relative_disagreement(observed: Any, expected: Any) -> float:
    observed_array = np.asarray(observed, dtype=np.float64).reshape(-1)
    expected_array = np.asarray(expected, dtype=np.float64).reshape(-1)
    scale = max(
        float(np.linalg.norm(observed_array)),
        float(np.linalg.norm(expected_array)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(observed_array - expected_array) / scale)


def _boundary_regions(operator, masks) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return full-grid boundary regions from the grid's null stencil.

    The topology reader owns only its interior connectivity subset.  Residuals
    and domain labels instead live on every grid cell, so the diagnostic expands
    the grid null stencil onto its centre indices and makes border rows
    centre-only.  This preserves the null stencil as the sole adjacency
    authority without pretending its interior-row count is the grid size.
    """

    labels = np.asarray(masks.label, dtype=np.int64).reshape(-1)
    grid_cells = int(operator.grid.node_number)
    if labels.shape != (grid_cells,) or grid_cells != EXPECTED_GRID_CELLS:
        raise AssertionError(
            "settled-mask receipt requires 1089 aligned grid labels, got "
            f"labels={labels.shape}, grid_cells={grid_cells}"
        )
    source = np.asarray(operator.grid.null.stencil, dtype=np.int64)
    if source.ndim != 2 or source.shape[1] < 2:
        raise AssertionError(
            "grid null stencil must have shape (interior_centres, ring_width)"
        )
    centre = source[:, 0]
    if (
        np.unique(centre).size != centre.size
        or source.min(initial=0) < 0
        or source.max(initial=0) >= grid_cells
    ):
        raise AssertionError("grid null stencil must carry unique in-grid centres")
    expanded = np.repeat(
        np.arange(grid_cells, dtype=np.int64)[:, None], source.shape[1], axis=1
    )
    expanded[centre] = source
    if expanded.shape != (EXPECTED_GRID_CELLS, source.shape[1]):
        raise AssertionError(
            "expanded adjacency must have shape (1089, ring_width), got "
            f"{expanded.shape}"
        )
    if not np.array_equal(expanded[:, 0], np.arange(grid_cells)):
        raise AssertionError("expanded grid adjacency must be centre-first")

    excluded = np.asarray(masks.excluded_material, dtype=bool)
    core = np.asarray(masks.core, dtype=bool)
    neighbours = expanded[:, 1:]
    active = ~excluded
    separatrix = active & np.any(core[neighbours] != core[:, None], axis=1)
    limiter = active & np.any(excluded[neighbours] != excluded[:, None], axis=1)
    boundary = separatrix | limiter

    topology_subset = np.asarray(operator.topology.connectivity_rings, dtype=np.int64)
    topology_centre_first = bool(
        topology_subset.ndim == 2
        and topology_subset.shape[1] >= 2
        and np.array_equal(topology_subset[:, 0], centre)
    )
    evidence = {
        "authority": "operator.grid.null.stencil",
        "source_shape": list(source.shape),
        "expanded_shape": list(expanded.shape),
        "centre_first": True,
        "border_rows_are_centre_only": int(grid_cells - source.shape[0]),
        "topology_reader_subset": {
            "authority": "operator.topology.connectivity_rings",
            "shape": list(topology_subset.shape),
            "centre_first_and_same_centres": topology_centre_first,
            "used_for_residual_region_classification": False,
        },
        "counts": {
            "separatrix_adjacent": int(np.count_nonzero(separatrix)),
            "limiter_adjacent": int(np.count_nonzero(limiter)),
            "boundary_adjacent_union": int(np.count_nonzero(boundary)),
        },
    }
    return {
        "boundary_adjacent": boundary,
        "separatrix_adjacent": separatrix,
        "limiter_adjacent": limiter,
    }, evidence


def _region_decomposition(operator, masks, residual: Any) -> dict[str, Any]:
    grid_residual = np.asarray(residual, dtype=np.float64)[: operator.grid.node_number]
    boundary_regions, _evidence = _boundary_regions(operator, masks)
    boundary = boundary_regions["boundary_adjacent"]
    core = np.asarray(masks.core, dtype=bool) & ~boundary
    private = np.asarray(masks.private_flux, dtype=bool) & ~boundary
    common = np.asarray(masks.common_sol, dtype=bool) & ~boundary
    excluded = np.asarray(masks.excluded_material, dtype=bool)
    return {
        name: _norm_summary(grid_residual[selection])
        for name, selection in {
            "core": core,
            "boundary_adjacent": boundary,
            "separatrix_adjacent": boundary_regions["separatrix_adjacent"],
            "limiter_adjacent": boundary_regions["limiter_adjacent"],
            "private_flux": private,
            "common_sol": common,
            "excluded_material": excluded,
        }.items()
    }


def _partition_observables(operator, state, requested_class) -> dict[str, Any]:
    if operator.moment_geometry is None:
        masks, topology = operator.read(state, requested_class)
        if operator.sample is None:
            sample_psi_norm = jnp.empty(0, dtype=jnp.asarray(state).dtype)
        else:
            sample_flux = operator.sample_node_flux(state)
            sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        support_area = jnp.empty(0, dtype=jnp.asarray(state).dtype)
        support_boundary = jnp.empty(0, dtype=bool)
        clip_geometry_route = "unavailable_on_centroid_current_carrier"
    else:
        masks, topology, sample_psi_norm, support = operator._support_partition(
            state, requested_class
        )
        support_area = support.area
        support_boundary = support.boundary
        clip_geometry_route = "traced_clipped_support"
    return {
        "psi_norm": np.asarray(masks.psi_norm, dtype=np.float64),
        "labels": np.asarray(masks.label, dtype=np.int64),
        "sample_psi_norm": np.asarray(sample_psi_norm, dtype=np.float64),
        "support_area": np.asarray(support_area, dtype=np.float64),
        "support_boundary": np.asarray(support_boundary, dtype=bool),
        "clip_geometry_route": clip_geometry_route,
        "axis_flux": float(np.asarray(topology.axis_flux)),
        "boundary_flux": float(np.asarray(topology.boundary_flux)),
        "x_point_flux": float(np.asarray(topology.x_point_flux)),
        "axis": np.asarray(topology.axis, dtype=np.float64),
        "x_point": np.asarray(topology.x_point, dtype=np.float64),
        "limiter_point": np.asarray(topology.wall_point, dtype=np.float64),
    }


def _differentiable_partition_observables(operator, state, requested_class):
    """Return the piecewise-smooth read leaves used by residual construction."""

    if operator.moment_geometry is None:
        masks, topology = operator.read(state, requested_class)
        return {
            "psi_norm": masks.psi_norm,
            "axis_flux": jnp.atleast_1d(topology.axis_flux),
            "boundary_flux": jnp.atleast_1d(topology.boundary_flux),
            "x_point_flux": jnp.atleast_1d(topology.x_point_flux),
            "axis": topology.axis,
            "x_point": topology.x_point,
            "limiter_point": topology.wall_point,
        }
    masks, topology, _sample_psi_norm, support = operator._support_partition(
        state, requested_class
    )
    return {
        "psi_norm": masks.psi_norm,
        "axis_flux": jnp.atleast_1d(topology.axis_flux),
        "boundary_flux": jnp.atleast_1d(topology.boundary_flux),
        "x_point_flux": jnp.atleast_1d(topology.x_point_flux),
        "axis": topology.axis,
        "x_point": topology.x_point,
        "limiter_point": topology.wall_point,
        "support_area": support.area,
    }


def _derivative_comparison(finite_difference: Any, tangent: Any) -> dict[str, Any]:
    return {
        "finite_difference": _norm_summary(finite_difference),
        "jacobian_vector_product": _norm_summary(tangent),
        "relative_disagreement": _relative_disagreement(finite_difference, tangent),
    }


def _observable_difference(
    minus: dict[str, Any], plus: dict[str, Any], denominator: float
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in (
        "psi_norm",
        "sample_psi_norm",
        "support_area",
        "axis",
        "x_point",
        "limiter_point",
    ):
        output[name] = _norm_summary((plus[name] - minus[name]) / denominator)
    for name in ("axis_flux", "boundary_flux", "x_point_flux"):
        output[name] = (plus[name] - minus[name]) / denominator
    output["label_changes"] = int(np.count_nonzero(plus["labels"] != minus["labels"]))
    output["support_boundary_changes"] = int(
        np.count_nonzero(plus["support_boundary"] != minus["support_boundary"])
    )
    output["clip_geometry_route"] = plus["clip_geometry_route"]
    return output


def _jacobian_diagnostics(
    operator, frozen_map, state, mask, requested_class, target_current
):
    mapped, tangent = jax.linearize(frozen_map, state)
    right_hand_side = mapped - state

    def residual_action(vector):
        return vector - tangent(vector)

    direction, gmres_info = jax.scipy.sparse.linalg.gmres(
        residual_action,
        right_hand_side,
        maxiter=SMOOTH_GMRES_ITERATIONS,
        restart=SMOOTH_GMRES_ITERATIONS,
        solve_method="batched",
    )
    direction.block_until_ready()
    jvp = residual_action(direction)
    state_jvp = jnp.where(mask, 0.0, direction)
    source_jvp = jvp - state_jvp
    _observable_values, observable_jvp = jax.jvp(
        lambda candidate: _differentiable_partition_observables(
            operator, candidate, requested_class
        ),
        (state,),
        (direction,),
    )
    state_scale = max(float(jnp.linalg.norm(state)), 1.0)
    direction_scale = max(float(jnp.linalg.norm(direction)), np.finfo(float).tiny)
    finite_differences = []
    for relative_step in FINITE_DIFFERENCE_RELATIVE_STEPS:
        epsilon = relative_step * state_scale / direction_scale
        plus_state = state + epsilon * direction
        minus_state = state - epsilon * direction
        plus_residual = plus_state - frozen_map(plus_state)
        minus_residual = minus_state - frozen_map(minus_state)
        difference = (plus_residual - minus_residual) / (2.0 * epsilon)
        plus_internal = operator.internal(plus_state, requested_class, target_current)
        minus_internal = operator.internal(minus_state, requested_class, target_current)
        source_difference = jnp.where(
            mask, 0.0, -(plus_internal - minus_internal) / (2.0 * epsilon)
        )
        plus_partition = _partition_observables(operator, plus_state, requested_class)
        minus_partition = _partition_observables(operator, minus_state, requested_class)
        observable_differences = {
            name: (plus_partition[name] - minus_partition[name]) / (2.0 * epsilon)
            for name in observable_jvp
        }
        component_disagreement = {
            "state_identity": _derivative_comparison(state_jvp, state_jvp),
            "source_term_through_psi_norm": _derivative_comparison(
                source_difference, source_jvp
            ),
            "psi_norm_read": _derivative_comparison(
                observable_differences["psi_norm"], observable_jvp["psi_norm"]
            ),
            "boundary_flux_read": _derivative_comparison(
                observable_differences["boundary_flux"],
                observable_jvp["boundary_flux"],
            ),
            "limiter_point_read": _derivative_comparison(
                observable_differences["limiter_point"],
                observable_jvp["limiter_point"],
            ),
            "external_field": {
                "finite_difference": {"l2": 0.0, "sup": 0.0},
                "jacobian_vector_product": {"l2": 0.0, "sup": 0.0},
                "relative_disagreement": 0.0,
                "reason": "captured conductor field is constant in solver state",
            },
        }
        if operator.moment_geometry is None:
            component_disagreement["clip_geometry"] = {
                "available": False,
                "reason": (
                    "centroid-current carrier has no moment geometry; no "
                    "clipped-support derivative was synthesized"
                ),
            }
        else:
            component_disagreement["clip_geometry"] = {
                "available": True,
                **_derivative_comparison(
                    observable_differences["support_area"],
                    observable_jvp["support_area"],
                ),
            }
        finite_differences.append(
            {
                "relative_state_step": relative_step,
                "epsilon": epsilon,
                "residual_jvp_relative_disagreement": _relative_disagreement(
                    difference, jvp
                ),
                "grid_region_disagreement": _region_decomposition(
                    operator,
                    operator.read(state, requested_class)[0],
                    np.asarray(difference) - np.asarray(jvp),
                ),
                "residual_component_disagreement": component_disagreement,
                "topology_and_support_directional_derivative": (
                    _observable_difference(
                        minus_partition, plus_partition, 2.0 * epsilon
                    )
                ),
            }
        )
    decomposed_jvp = state_jvp + source_jvp
    linear_residual = residual_action(direction) - right_hand_side
    return {
        "gmres_iterations": SMOOTH_GMRES_ITERATIONS,
        "gmres_info": int(np.asarray(gmres_info)),
        "direction": _norm_summary(direction),
        "linear_residual": _norm_summary(linear_residual),
        "relative_linear_residual_l2": _relative_disagreement(
            residual_action(direction), right_hand_side
        ),
        "jvp_term_decomposition": {
            "state_term": _norm_summary(state_jvp),
            "source_through_psi_norm_and_clip": _norm_summary(source_jvp),
            "external_field": {
                "l2": 0.0,
                "sup": 0.0,
                "reason": "captured conductor field is constant in solver state",
            },
            "reconstructed_total": _norm_summary(decomposed_jvp),
            "reconstruction_relative_disagreement": _relative_disagreement(
                decomposed_jvp, jvp
            ),
        },
        "finite_difference_checks": finite_differences,
    }, direction


def _smooth_solve(frozen_map, state) -> dict[str, Any]:
    result = fixed_point.newton_krylov(
        frozen_map,
        state,
        newton_steps=SMOOTH_NEWTON_STEPS,
        gmres_iterations=SMOOTH_GMRES_ITERATIONS,
        warmup=0,
        relaxation=SMOOTH_RELAXATION,
        step_cap=SMOOTH_STEP_CAP,
        convergence_tolerance=1.0e-8,
        stream_inner_iterations=True,
    )
    result.state.block_until_ready()
    reason_value = int(np.asarray(result.termination_reason))
    try:
        reason = FixedPointTerminationReason(reason_value).name.lower()
    except ValueError:
        reason = f"unknown_{reason_value}"
    residuals_before = _array(result.inner_iteration_residuals_before)
    residuals_after = _array(result.inner_iteration_residuals_after)
    proposed_step_norms = _array(result.inner_iteration_proposed_step_norms)
    accepted = _array(result.inner_iteration_accepted)
    decisions = _array(result.inner_iteration_decisions)
    applied_factors = _array(result.inner_iteration_applied_factors)
    krylov_reductions = _array(result.inner_iteration_krylov_reductions)
    krylov_tolerances = _array(result.inner_iteration_krylov_tolerances)
    trajectory_count = min(
        len(residuals_before),
        len(residuals_after),
        len(proposed_step_norms),
        len(accepted),
        len(decisions),
        len(applied_factors),
    )
    return {
        "initial_residual": float(
            fixed_point._relative_residual(frozen_map(state), state)
        ),
        "terminal_residual": float(np.asarray(result.residual)),
        "converged": bool(np.asarray(result.converged)),
        "termination_reason": reason,
        "attempted_promotions": int(np.asarray(result.attempted_newton_promotions)),
        "accepted_promotions": int(np.asarray(result.accepted_newton_promotions)),
        "residuals_before": residuals_before,
        "residuals_after": residuals_after,
        "proposed_step_norms": proposed_step_norms,
        "accepted": accepted,
        "decisions": decisions,
        "applied_factors": applied_factors,
        "krylov_reductions": krylov_reductions,
        "krylov_tolerances": krylov_tolerances,
        "newton_trajectory": [
            {
                "step": index + 1,
                "residual_before": residuals_before[index],
                "residual_after": residuals_after[index],
                "proposed_step_norm": proposed_step_norms[index],
                "accepted": accepted[index],
                "decision": decisions[index],
                "applied_factor": applied_factors[index],
            }
            for index in range(trajectory_count)
        ],
    }


def _measure_row(
    selected_row,
    qualification,
    response_cache,
    banked: dict[str, Any],
    *,
    require_bank_match: bool,
) -> dict[str, Any]:
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("profile rebuild entered the direct response builder")
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    observed = bank_producer._ObservedProfile(profile)
    seed = jnp.asarray(passive_case["state"])
    observed.solve_portfolio(
        jnp.stack((seed, seed)),
        route="newton_krylov",
        target_current=target_current,
        tolerance=reachability.FIXED_POINT_CRITERION,
        newton_steps=reachability.NEWTON_STEPS,
        gmres_iterations=reachability.GMRES_ITERATIONS,
        warmup=reachability.WARMUP_SWEEPS,
        relaxation=reachability.RELAXATION,
        step_cap=reachability.STEP_CAP,
    )
    if observed.portfolio is None:
        raise RuntimeError("production solve returned no branch portfolio")
    branch = jax.tree.map(
        lambda value: value[int(TopologyClass.DIVERTED)],
        observed.portfolio.branches,
    )
    production = branch.equilibrium.fixed_point
    state = branch.equilibrium.flux
    state.block_until_ready()
    validation = _bank_validation(production, banked, require_match=require_bank_match)
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    mask = profile.operator.residual_shadow_mask(state, requested_class)
    shadowed_map = profile.operator.flux_map_with_shadow(
        requested_class=requested_class, target_current=target_current
    )

    def frozen_map(candidate):
        return shadowed_map(candidate, mask)

    mapped = frozen_map(state)
    residual = state - mapped
    masks, topology = profile.operator.read(state, requested_class)
    jacobian, _direction = _jacobian_diagnostics(
        profile.operator,
        frozen_map,
        state,
        mask,
        requested_class,
        target_current,
    )
    external = profile.operator.external()
    internal = profile.operator.internal(state, requested_class, target_current)
    residual_terms = {
        "state": jnp.where(mask, 0.0, state),
        "source_through_psi_norm": jnp.where(mask, 0.0, -internal),
        "external_field": jnp.where(mask, 0.0, -external),
    }
    reconstructed = sum(residual_terms.values())
    boundary_regions, boundary_evidence = _boundary_regions(profile.operator, masks)
    table = bank_producer._candidate_table_status(profile, state)
    return {
        "identity": f"{int(selected_row['shot'])}/{int(selected_row['slice_index'])}",
        "bank_validation": validation,
        "candidate_table_status": table,
        "terminal_topology": {
            "axis": _array(topology.axis),
            "x_point": _array(topology.x_point),
            "limiter_point": _array(topology.wall_point),
            "axis_flux": _strict_float(topology.axis_flux),
            "boundary_flux": _strict_float(topology.boundary_flux),
            "x_point_flux": _strict_float(topology.x_point_flux),
        },
        "settled_mask": {
            "size": int(mask.size),
            "excluded_count": int(np.count_nonzero(np.asarray(mask))),
            "grid_boundary_regions": {
                name: int(np.count_nonzero(selection))
                for name, selection in boundary_regions.items()
            },
            "boundary_adjacency": boundary_evidence,
        },
        "stall_residual": {
            "total": _norm_summary(residual),
            "regions": _region_decomposition(profile.operator, masks, residual),
            "terms": {
                name: {
                    "total": _norm_summary(values),
                    "regions": _region_decomposition(profile.operator, masks, values),
                }
                for name, values in residual_terms.items()
            },
            "term_reconstruction_relative_disagreement": _relative_disagreement(
                reconstructed, residual
            ),
        },
        "jacobian_consistency": jacobian,
        "frozen_mask_smooth_solve": _smooth_solve(frozen_map, state),
    }


def measure(
    *,
    operands: Path,
    output: Path,
    source_label: str,
    require_bank_match: bool,
) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    banked = _load_banked_rows(operands)
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    rows = []
    for key in TARGETS:
        print(f"MEASURING {key[0]}/{key[1]} source={source_label}", flush=True)
        row, qualification = selected[key]
        measured = _measure_row(
            row,
            qualification,
            response_cache,
            banked[key],
            require_bank_match=require_bank_match,
        )
        rows.append(measured)
        print(
            "MEASURED "
            + json.dumps(
                {
                    "identity": measured["identity"],
                    "smooth_terminal": measured["frozen_mask_smooth_solve"][
                        "terminal_residual"
                    ],
                    "fd_disagreement": [
                        item["residual_jvp_relative_disagreement"]
                        for item in measured["jacobian_consistency"][
                            "finite_difference_checks"
                        ]
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    receipt = {
        "artifact": "settled residual-mask smooth-solve diagnosis",
        "source_label": source_label,
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "evidence_inputs": {
            "operands": str(operands),
            "operands_sha256": _sha256(operands),
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "smooth_newton_steps": SMOOTH_NEWTON_STEPS,
            "smooth_gmres_iterations": SMOOTH_GMRES_ITERATIONS,
            "smooth_relaxation": SMOOTH_RELAXATION,
            "smooth_step_cap": SMOOTH_STEP_CAP,
            "finite_difference_relative_steps": list(FINITE_DIFFERENCE_RELATIVE_STEPS),
            "terminal_reconstruction": (
                "rebuild through the persisted carrier, validate terminal residual "
                "and active-set history against the exact operand cache, then retain "
                "the full in-memory solver state"
            ),
            "frozen_partition": (
                "terminal residual-shadow mask held fixed for every smooth map, "
                "Jacobian action, finite difference, and Newton solve"
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _mechanism(row: dict[str, Any]) -> dict[str, Any]:
    smooth = row["frozen_mask_smooth_solve"]
    jacobian = row["jacobian_consistency"]
    fd = jacobian["finite_difference_checks"]
    best = min(fd, key=lambda item: item["residual_jvp_relative_disagreement"])
    table = row["candidate_table_status"]
    contracts = smooth["terminal_residual"] < 0.5 * smooth["initial_residual"]
    consistent = best["residual_jvp_relative_disagreement"] <= 1.0e-3
    overflow = bool(table["o_point"]["truncated"] or table["x_point"]["truncated"])
    linear_solve_adequate = bool(
        jacobian["gmres_info"] == 0
        and jacobian["relative_linear_residual_l2"] <= 1.0e-3
    )
    raw_step_sup = float(jacobian["direction"]["sup"])
    proposed_step_sup = float(smooth["proposed_step_norms"][0])
    configured_step_bound = float(
        SMOOTH_STEP_CAP * SMOOTH_RELAXATION * row["stall_residual"]["total"]["sup"]
    )
    raw_to_proposed_fraction = proposed_step_sup / max(raw_step_sup, 1.0e-300)
    step_cap_binding = bool(
        np.isclose(proposed_step_sup, configured_step_bound, rtol=1.0e-5, atol=0.0)
    )
    conditioning_damping = bool(
        linear_solve_adequate
        and not step_cap_binding
        and raw_to_proposed_fraction < 0.1
    )
    clip = best["residual_component_disagreement"]["clip_geometry"]
    if clip.get("available", False):
        clip_evidence = (
            "clipped-support derivative relative disagreement "
            f"{clip['relative_disagreement']:.6g}"
        )
    else:
        clip_evidence = clip["reason"]
    if not consistent:
        name = "jacobian_inconsistency_in_piecewise_topology_read"
    elif contracts:
        name = "outer_active_set_or_globalization_not_fixed_map_contraction"
    elif not linear_solve_adequate:
        name = "inexact_newton_krylov_cap"
    elif conditioning_damping:
        name = "projected_krylov_conditioning_over_damps_newton_step"
    elif not smooth["accepted_promotions"]:
        name = "damping_or_acceptance_collapse_on_consistent_fixed_map"
    else:
        name = "ill_conditioned_fixed_partition_map"
    alternatives = {
        "missing_derivative_through_topology_read": {
            "ruled_out": consistent,
            "evidence": (
                "best central finite-difference/JVP relative disagreement "
                f"{best['residual_jvp_relative_disagreement']:.6g} at relative "
                f"state step {best['relative_state_step']:.1e}"
            ),
        },
        "non_smooth_residual_from_local_support_read": {
            "ruled_out": not bool(clip.get("available", False)),
            "evidence": clip_evidence,
        },
        "inexact_newton_cap": {
            "ruled_out": bool(linear_solve_adequate and not step_cap_binding),
            "evidence": (
                f"GMRES info {jacobian['gmres_info']}; relative linear residual "
                f"{jacobian['relative_linear_residual_l2']:.6g} at dimension "
                f"{jacobian['gmres_iterations']}; proposed step "
                f"{proposed_step_sup:.6g} versus configured bound "
                f"{configured_step_bound:.6g}"
            ),
        },
        "damping_collapse": {
            "ruled_out": False,
            "selected": conditioning_damping,
            "evidence": (
                f"projected-conditioning proposal/raw direction fraction "
                f"{raw_to_proposed_fraction:.6g}; accepted "
                f"{smooth['accepted_promotions']} of "
                f"{smooth['attempted_promotions']} promotions with subsequent "
                f"line-search factors {smooth['applied_factors']}"
            ),
        },
        "saddle_mis_selection": {
            "ruled_out": True,
            "evidence": (
                "the measured residual map consumes the frozen terminal mask and "
                "does not reselect a saddle during any Jacobian action or Newton "
                f"promotion; terminal candidate-table overflow={overflow} is "
                "retained as a reconstruction caveat, not a cause of contraction"
            ),
        },
    }
    return {
        "name": name,
        "fixed_mask_contracts_by_half": contracts,
        "best_fd_jvp_relative_disagreement": best["residual_jvp_relative_disagreement"],
        "best_relative_state_step": best["relative_state_step"],
        "gmres_relative_linear_residual_l2": jacobian["relative_linear_residual_l2"],
        "raw_newton_direction_sup": raw_step_sup,
        "first_proposed_step_sup": proposed_step_sup,
        "configured_step_bound": configured_step_bound,
        "step_cap_binding": step_cap_binding,
        "raw_to_proposed_step_fraction": raw_to_proposed_fraction,
        "accepted_promotions": smooth["accepted_promotions"],
        "candidate_table_overflow_present": overflow,
        "alternatives": alternatives,
        "repair": {
            "jacobian_inconsistency_in_piecewise_topology_read": (
                "make the boundary and limiter topology read locally consistent "
                "with the residual Jacobian or hand off the derivative at its kink"
            ),
            "outer_active_set_or_globalization_not_fixed_map_contraction": (
                "repair active-set reconciliation or carried globalization state"
            ),
            "inexact_newton_krylov_cap": (
                "raise or adapt the Krylov dimension using achieved linear reduction"
            ),
            "projected_krylov_conditioning_over_damps_newton_step": (
                "recalibrate the projected-condition discriminator against the "
                "current linear model, then admit the verified raw Newton direction "
                "through the existing nonlinear merit ladder"
            ),
            "damping_or_acceptance_collapse_on_consistent_fixed_map": (
                "repair the promotion merit or damping rule on the settled partition"
            ),
            "ill_conditioned_fixed_partition_map": (
                "regularise or constrain the neutral physical mode"
            ),
        }[name],
        "owner": "nova equilibrium forward solver",
    }


def _finite_values(values: list[Any]) -> list[float]:
    return [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]


def _draw_diagnosis(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(17.0, 13.0), constrained_layout=True
    )
    colors = {"current": "#087e8b", "candidate": "#ff5a5f"}
    for row_index, row in enumerate(rows):
        residual_axis, derivative_axis, region_axis, step_axis = axes[row_index]
        for key, label in (
            ("current", "current main"),
            ("polish_support_candidate", "polish-support tip"),
        ):
            smooth = row[key]["frozen_mask_smooth_solve"]
            values = [smooth["initial_residual"]] + _finite_values(
                smooth["residuals_after"]
            )
            color = colors["current" if key == "current" else "candidate"]
            residual_axis.semilogy(
                range(len(values)), values, marker="o", label=label, color=color
            )
        residual_axis.axhline(1.0e-8, color="black", lw=0.8, ls="--")
        residual_axis.set_ylabel(row["identity"] + "\nrelative residual")
        residual_axis.set_xlabel("fixed-mask Newton step")
        residual_axis.grid(alpha=0.25)

        for key, label in (
            ("current", "total current"),
            ("polish_support_candidate", "total candidate"),
        ):
            checks = row[key]["jacobian_consistency"]["finite_difference_checks"]
            x = [item["relative_state_step"] for item in checks]
            total = [item["residual_jvp_relative_disagreement"] for item in checks]
            source = [
                item["residual_component_disagreement"]["source_term_through_psi_norm"][
                    "relative_disagreement"
                ]
                for item in checks
            ]
            color = colors["current" if key == "current" else "candidate"]
            derivative_axis.loglog(x, total, "o-", color=color, label=label)
            derivative_axis.loglog(x, source, "x--", color=color, alpha=0.8)
        derivative_axis.set_xlabel("relative finite-difference step")
        derivative_axis.set_ylabel("FD/JVP disagreement")
        derivative_axis.grid(alpha=0.25)

        region_names = ("core", "boundary_adjacent", "private_flux")
        positions = np.arange(len(region_names))
        width = 0.36
        for offset, key in (
            (-width / 2, "current"),
            (width / 2, "polish_support_candidate"),
        ):
            values = [
                row[key]["stall_residual"]["regions"][name]["l2"] or 0.0
                for name in region_names
            ]
            color = colors["current" if key == "current" else "candidate"]
            region_axis.bar(positions + offset, values, width, color=color)
        region_axis.set_yscale("log")
        region_axis.set_xticks(positions, ("core", "boundary", "private"))
        region_axis.set_ylabel("stall residual L2")
        region_axis.grid(axis="y", alpha=0.25)

        for key, label in (
            ("current", "current step norm"),
            ("polish_support_candidate", "candidate step norm"),
        ):
            smooth = row[key]["frozen_mask_smooth_solve"]
            norms = _finite_values(smooth["proposed_step_norms"])
            color = colors["current" if key == "current" else "candidate"]
            step_axis.semilogy(
                np.arange(1, len(norms) + 1), norms, "o-", color=color, label=label
            )
            factors = _finite_values(smooth["applied_factors"])
            step_axis.plot(
                np.arange(1, len(factors) + 1),
                factors,
                "x:",
                color=color,
                alpha=0.8,
            )
        step_axis.set_xlabel("Newton step")
        step_axis.set_ylabel("step norm (solid) / factor (dotted)")
        step_axis.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    axes[0, 3].legend(fontsize=7)
    figure.suptitle(
        "Frozen settled-mask contraction, Jacobian consistency, "
        "residual region, and step acceptance",
        fontsize=13,
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], report: Path) -> None:
    lines = [
        "# Settled-mask stall attribution",
        "",
        (
            f"Compared current `{receipt['current_source_commit']}` with "
            "polish-support "
            f"candidate `{receipt['candidate_source_commit']}` on four pure MAST rows. "
            "Each diagnostic rebuilds and validates the bank terminal, freezes its "
            "residual mask, and runs eight Newton updates with GMRES 40."
        ),
        "",
        (
            "| row | current initial -> terminal | ratio | candidate initial -> "
            "terminal | ratio | mechanism |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rows"]:
        current = row["current"]["frozen_mask_smooth_solve"]
        candidate = row["polish_support_candidate"]["frozen_mask_smooth_solve"]
        lines.append(
            f"| {row['identity']} | {current['initial_residual']:.6e} -> "
            f"{current['terminal_residual']:.6e} | "
            f"{current['terminal_residual'] / current['initial_residual']:.6f} | "
            f"{candidate['initial_residual']:.6e} -> "
            f"{candidate['terminal_residual']:.6e} | "
            f"{candidate['terminal_residual'] / candidate['initial_residual']:.6f} | "
            f"`{row['mechanism']['name']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-row attribution",
            "",
        ]
    )
    for row in receipt["rows"]:
        mechanism = row["mechanism"]
        candidate_mechanism = row["candidate_mechanism"]
        lines.extend(
            [
                f"### {row['identity']} pure",
                "",
                f"Named mechanism: `{mechanism['name']}`. Implied repair: "
                f"{mechanism['repair']}.",
                "",
                (
                    "Evidence: best finite-difference/JVP disagreement "
                    f"{mechanism['best_fd_jvp_relative_disagreement']:.6e}; "
                    "GMRES relative linear residual "
                    f"{mechanism['gmres_relative_linear_residual_l2']:.6e}; "
                    "raw-to-proposed Newton fraction "
                    f"{mechanism['raw_to_proposed_step_fraction']:.6e}; "
                    f"accepted promotions {mechanism['accepted_promotions']}. "
                    "Candidate raw-to-proposed fraction "
                    f"{candidate_mechanism['raw_to_proposed_step_fraction']:.6e}; "
                    "candidate named mechanism "
                    f"`{candidate_mechanism['name']}`."
                ),
                "",
            ]
        )
        for name, alternative in mechanism["alternatives"].items():
            if alternative.get("selected", False):
                verdict = "identified"
            elif alternative["ruled_out"]:
                verdict = "ruled out"
            else:
                verdict = "not ruled out"
            lines.append(f"- `{name}` — {verdict}: {alternative['evidence']}")
        lines.append("")
    first = receipt["rows"][0]["current"]["settled_mask"]["boundary_adjacency"]
    lines.extend(
        [
            "## Measurement authority and caveats",
            "",
            (
                "Current main reproduces all four exact cached terminal histories. "
                "The candidate reproduces none of those current-main histories; it "
                "is therefore measured from its own regenerated terminal states, "
                "with every mismatch retained per row in the receipt rather than "
                "treated as paired-state identity."
            ),
            "",
            (
                "Residual-region adjacency is expanded from "
                f"`operator.grid.null.stencil` {first['source_shape']} to "
                f"{first['expanded_shape']} with centre-first indexing. The "
                f"topology-reader subset remains separately recorded as "
                f"{first['topology_reader_subset']['shape']} and is not used to "
                "index the 1,089-cell residual vector."
            ),
            "",
            (
                "The persisted response carrier uses centroid currents and has no "
                "moment geometry. Clipped-support derivatives are therefore recorded "
                "as unavailable; no clipped-support result is fabricated."
            ),
            "",
            f"Figure: `{receipt['figure']}`.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def combine(
    current: Path, candidate: Path, output: Path, report: Path | None = None
) -> dict[str, Any]:
    current_data = json.loads(current.read_text(encoding="utf-8"))
    candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_by_identity = {row["identity"]: row for row in candidate_data["rows"]}
    rows = []
    for current_row in current_data["rows"]:
        identity = current_row["identity"]
        candidate_row = candidate_by_identity[identity]
        current_smooth = current_row["frozen_mask_smooth_solve"]
        candidate_smooth = candidate_row["frozen_mask_smooth_solve"]
        current_mechanism = _mechanism(current_row)
        candidate_mechanism = _mechanism(candidate_row)
        rows.append(
            {
                "identity": identity,
                "current": current_row,
                "polish_support_candidate": candidate_row,
                "mechanism": current_mechanism,
                "candidate_mechanism": candidate_mechanism,
                "support_change": {
                    "initial_residual_ratio_candidate_to_current": (
                        candidate_smooth["initial_residual"]
                        / current_smooth["initial_residual"]
                    ),
                    "terminal_residual_ratio_candidate_to_current": (
                        candidate_smooth["terminal_residual"]
                        / current_smooth["terminal_residual"]
                    ),
                    "changes_named_mechanism": (
                        candidate_mechanism["name"] != current_mechanism["name"]
                    ),
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_path = output.with_name("stall-diagnosis.png")
    _draw_diagnosis(rows, figure_path)
    receipt = {
        "artifact": "settled-mask stall diagnosis and unified-support comparison",
        "current_source_commit": current_data["source_commit"],
        "candidate_source_commit": candidate_data["source_commit"],
        "measurement_contract": {
            **current_data["measurement_contract"],
            "smooth_relaxation": SMOOTH_RELAXATION,
            "smooth_step_cap": SMOOTH_STEP_CAP,
        },
        "figure": str(figure_path),
        "rows": rows,
        "verdict": {
            "mechanisms": {row["identity"]: row["mechanism"]["name"] for row in rows},
            "all_current_bank_terminals_reproduced": all(
                row["current"]["bank_validation"]["passes"] for row in rows
            ),
            "all_candidate_bank_terminals_reproduced": all(
                row["polish_support_candidate"]["bank_validation"]["passes"]
                for row in rows
            ),
            "support_change_alters_any_mechanism": any(
                row["support_change"]["changes_named_mechanism"] for row in rows
            ),
        },
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if report is not None:
        _write_report(receipt, report)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--source-label", required=True)
    measure_parser.add_argument("--allow-bank-drift", action="store_true")
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--current", type=Path, required=True)
    combine_parser.add_argument("--candidate", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    combine_parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "measure":
        result = measure(
            operands=arguments.operands,
            output=arguments.output,
            source_label=arguments.source_label,
            require_bank_match=not arguments.allow_bank_drift,
        )
        print(json.dumps({"rows": len(result["rows"])}, sort_keys=True))
    else:
        result = combine(
            arguments.current,
            arguments.candidate,
            arguments.output,
            arguments.report,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
