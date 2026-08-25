"""Contrast pinned diverted portfolio solves with margin-graded iteration.

The cohort is the frozen six-reference MAST forward-parity set.  Each pair
starts from the same labelled flux and uses the same current-pinned diverted
map.  The pure arm enters through ``ForwardProfile.solve_portfolio``; the
mixed arm replaces the production residual policy with the continuous merit
``relative_residual + max(-class_margin, 0)``.  Terminal margin comparisons
therefore compare two different iterations and are not a causal ablation of
the margin value alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import (
    _compare_existing_numeric_fields,
    _infinity_name,
    _margin_graded_newton_krylov,
    _terminal_xpoint_diagnostics,
)
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import (
    _persisted_response_cache,
    _source_revision,
)
from nova.equilibrium.connectivity_boundary import traced_smooth_boundary_read
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/dual-branch-selection/pinned-branch-contrast.json")
PARITY_RELATIVE_TOLERANCE = 1.0e-10
SMOOTH_CLASS_TEMPERATURE = 0.01
PARITY_ROUNDOFF_BOUND = 4.0 * np.finfo(np.float64).eps
FLOAT_TRACE_RELATIVE_BOUND = 1.0e-10
DERIVED_CONTRACTION_RELATIVE_BOUND = 1.0e-2
ROUNDOFF_CONDITIONED_RATIO_RELATIVE_BOUND = 2.0e-1
REGENERATION_IDENTITY_PATHS = frozenset(
    {
        "driver_sha256",
        "source_commit",
        "response_carrier.named_cache_only_check.command.1",
        "response_carrier.named_cache_only_check.stdout",
    }
)


def _digest(values: jax.Array | np.ndarray) -> str:
    """Return a stable identity for one float64 state."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _serial_residuals(values: jax.Array | np.ndarray) -> list[float | None]:
    """Serialize a fixed residual trace without inventing numeric sentinels."""
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=np.float64)
    ]


def _newton_residuals(values: jax.Array | np.ndarray) -> list[float | None]:
    """Extract one retained residual after each production Newton promotion."""
    trace = np.asarray(values, dtype=np.float64)
    stride = GMRES_ITERATIONS + 2
    indices = np.arange(stride - 1, stride * NEWTON_STEPS, stride)
    if trace.size < stride * NEWTON_STEPS:
        raise RuntimeError(
            "the production residual trace is shorter than its declared "
            "Newton-Krylov evaluation budget"
        )
    return _serial_residuals(trace[indices])


def _fit_contraction(values: list[float | None]) -> dict[str, Any]:
    """Fit ``log(residual) = intercept + iteration * log(rho)`` by OLS."""
    points = [
        (index, value)
        for index, value in enumerate(values)
        if value is not None and np.isfinite(value) and value > 0.0
    ]
    if len(points) < 2:
        return {
            "ratio": None,
            "finite_positive_point_count": len(points),
            "contracts": None,
            "fit_r_squared": None,
        }
    iteration = np.asarray([item[0] for item in points], dtype=np.float64)
    logged = np.log(np.asarray([item[1] for item in points], dtype=np.float64))
    slope, intercept = np.polyfit(iteration, logged, 1)
    predicted = intercept + slope * iteration
    total = float(np.sum((logged - np.mean(logged)) ** 2))
    unexplained = float(np.sum((logged - predicted) ** 2))
    ratio = float(np.exp(slope))
    return {
        "ratio": ratio,
        "finite_positive_point_count": len(points),
        "contracts": bool(ratio < 1.0),
        "fit_r_squared": 1.0 if total == 0.0 else float(1.0 - unexplained / total),
    }


def _smooth_diverted_probability(profile, state: jax.Array) -> float:
    """Read ``p_diverted`` with the exact comparator operands used by margin."""
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    _masks, topology = operator._fixed_design_topology.read(
        physical,
        operator.polarity,
        operator.inside_material,
        None,
    )
    _limited, diverted = operator._fixed_design_topology.grid(grid_flux)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    radius, height, connectivity_shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = connectivity_shape
    reading = traced_smooth_boundary_read(
        grid_flux.reshape((radial_count, vertical_count)).T,
        radius,
        height,
        operator.inside_material.reshape((radial_count, vertical_count)).T,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        2,
        jnp.empty((0,), dtype=radius.dtype),
        jnp.asarray(1.0, dtype=grid_flux.dtype),
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        jnp.asarray(SMOOTH_CLASS_TEMPERATURE, dtype=grid_flux.dtype),
        classification_x=diverted,
        classification_wall=classification_wall,
    )
    return float(reading["p_diverted"])


def _terminal_observables(profile, state: jax.Array) -> dict[str, Any]:
    """Return exact and smooth terminal topology observables."""
    _masks, topology = profile.operator.read(state)
    diverted = bool(topology.diverted)
    margin = float(profile.operator.topology_margin(state))
    nonfinite_margin = None
    if not np.isfinite(margin):
        nonfinite_margin = "positive_infinity" if margin > 0.0 else "negative_infinity"
    diagnostics = _terminal_xpoint_diagnostics(profile, state, topology)
    diagnostic_margin = diagnostics["class_margin_from_operands"]
    diagnostics["class_margin_from_operands"] = (
        diagnostic_margin if np.isfinite(diagnostic_margin) else None
    )
    diagnostics["class_margin_from_operands_nonfinite"] = _infinity_name(
        diagnostic_margin
    )
    return {
        "achieved_class": "diverted" if diverted else "limited",
        "topology_consistent": diverted,
        "class_margin": margin if np.isfinite(margin) else None,
        "class_margin_nonfinite": nonfinite_margin,
        "p_diverted": _smooth_diverted_probability(profile, state),
        "terminal_xpoint_diagnostics": diagnostics,
    }


def _diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all terminal operands and test the reachability hypothesis."""

    reference_diagnostics = []
    absent_status_counts: dict[str, int] = {}
    all_status_counts: dict[str, int] = {}
    connectivity_support_count = 0
    absent_terminal_count = 0
    for record in records:
        reference = record["reference"]
        arms = {}
        for arm_name in ("pure_arm", "mixed_arm"):
            arm = record[arm_name]
            diagnostics = arm["terminal_xpoint_diagnostics"]
            status = diagnostics["selection_status"]
            all_status_counts[status] = all_status_counts.get(status, 0) + 1
            connectivity_support_count += int(
                status == "selected_typed_saddle_with_connectivity_support"
            )
            if arm["class_margin"] is None:
                absent_terminal_count += 1
                absent_status_counts[status] = absent_status_counts.get(status, 0) + 1
            arms[arm_name] = {
                "class_margin": arm["class_margin"],
                "class_margin_nonfinite": arm["class_margin_nonfinite"],
                "selection_status": status,
                "selected_x_normalized_flux_operand": diagnostics[
                    "selected_x_normalized_flux_operand"
                ],
                "selected_x_normalized_flux_operand_nonfinite": diagnostics[
                    "selected_x_normalized_flux_operand_nonfinite"
                ],
                "wall_status": diagnostics["wall_operand"]["status"],
                "wall_normalized_flux_before_shadow": diagnostics["wall_operand"][
                    "normalized_flux_before_shadow"
                ],
                "wall_normalized_flux": diagnostics["wall_operand"]["normalized_flux"],
                "wall_normalized_flux_nonfinite": diagnostics["wall_operand"][
                    "normalized_flux_nonfinite"
                ],
            }
        reference_diagnostics.append(
            {
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
                "arms": arms,
            }
        )

    absent_reachability_count = absent_status_counts.get(
        "selected_typed_saddle_not_connectivity_reachable", 0
    )
    holds = bool(
        absent_terminal_count > 0 and absent_reachability_count == absent_terminal_count
    )
    return {
        "reference_count": len(reference_diagnostics),
        "terminal_count": 2 * len(reference_diagnostics),
        "reference_diagnostics": reference_diagnostics,
        "all_terminal_selection_status_counts": all_status_counts,
        "connectivity_support_terminal_count": connectivity_support_count,
        "absent_class_margin_terminal_count": absent_terminal_count,
        "absent_class_margin_selection_status_counts": absent_status_counts,
        "leading_hypothesis": {
            "prior_diiid_terminal_count": 10,
            "prior_diiid_not_connectivity_reachable_count": 10,
            "prior_diiid_connectivity_support_count": 0,
            "mast_absent_not_connectivity_reachable_count": (absent_reachability_count),
            "mast_absent_terminal_count": absent_terminal_count,
            "holds": holds,
            "interpretation": (
                "connectivity reachability remains the prime suspect"
                if holds
                else "the MAST categories do not isolate connectivity reachability"
            ),
        },
    }


def _compare_regeneration_semantics(
    banked: dict[str, Any], regenerated: dict[str, Any]
) -> dict[str, Any]:
    """Require exact semantics and bound every regenerated floating-point drift."""

    semantic_count = 0
    unchanged_float_count = 0
    changed_float_count = 0
    duration_count = 0
    identity_count = 0
    envelope_counts: dict[str, int] = {}
    envelope_maxima: dict[str, dict[str, Any]] = {}

    def record_float(
        path: tuple[str, ...], banked_value: float, regenerated_value: float
    ) -> None:
        nonlocal unchanged_float_count, changed_float_count, duration_count
        field = ".".join(path)
        if path[-1].endswith("seconds"):
            duration_count += 1
            return
        if banked_value == regenerated_value:
            unchanged_float_count += 1
            return
        if not (np.isfinite(banked_value) and np.isfinite(regenerated_value)):
            raise RuntimeError(
                f"regenerated float changed finiteness at {field}: "
                f"banked={banked_value!r}, regenerated={regenerated_value!r}"
            )
        absolute = abs(regenerated_value - banked_value)
        magnitude = max(abs(banked_value), abs(regenerated_value))
        relative = absolute / magnitude if magnitude > 0.0 else 0.0
        if absolute <= PARITY_ROUNDOFF_BOUND:
            envelope = "binary64_roundoff"
            bound = PARITY_ROUNDOFF_BOUND
            metric = absolute
        elif "fitted_contraction" in path:
            envelope = "derived_contraction_relative"
            bound = DERIVED_CONTRACTION_RELATIVE_BOUND
            metric = relative
        elif path[-1] == "terminal_residual_ratio_pure_over_mixed":
            envelope = "roundoff_conditioned_ratio_relative"
            bound = ROUNDOFF_CONDITIONED_RATIO_RELATIVE_BOUND
            metric = relative
        else:
            envelope = "float_trace_relative"
            bound = FLOAT_TRACE_RELATIVE_BOUND
            metric = relative
        if metric > bound:
            raise RuntimeError(
                f"regenerated float left {envelope} at {field}: "
                f"banked={banked_value!r}, regenerated={regenerated_value!r}, "
                f"metric={metric!r}, bound={bound!r}"
            )
        changed_float_count += 1
        envelope_counts[envelope] = envelope_counts.get(envelope, 0) + 1
        maximum = envelope_maxima.get(envelope)
        if maximum is None or metric > maximum["metric"]:
            envelope_maxima[envelope] = {
                "field": field,
                "banked_value": banked_value,
                "regenerated_value": regenerated_value,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "metric": metric,
                "bound": bound,
            }

    def compare(left: Any, right: Any, path: tuple[str, ...] = ()) -> None:
        nonlocal semantic_count, identity_count
        field = ".".join(path)
        if isinstance(left, dict):
            if not isinstance(right, dict):
                raise RuntimeError(f"regenerated receipt changed type at {field}")
            for key, value in left.items():
                if key not in right:
                    raise RuntimeError(
                        f"regenerated receipt dropped {'.'.join(path + (key,))}"
                    )
                compare(value, right[key], path + (key,))
            return
        if isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise RuntimeError(f"regenerated receipt changed {field} length")
            for index, (banked_item, regenerated_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(
                    banked_item,
                    regenerated_item,
                    path + (str(index),),
                )
            return
        if isinstance(left, float) and isinstance(right, float):
            record_float(path, left, right)
            return
        if field in REGENERATION_IDENTITY_PATHS:
            identity_count += 1
            return
        semantic_count += 1
        if type(left) is not type(right) or left != right:
            raise RuntimeError(
                f"regenerated semantic field changed at {field}: "
                f"banked={left!r}, regenerated={right!r}"
            )

    compare(banked, regenerated)
    return {
        "semantic_field_count_compared": semantic_count,
        "semantic_field_difference_count": 0,
        "unchanged_float_field_count": unchanged_float_count,
        "changed_float_field_count": changed_float_count,
        "duration_field_count_preserved_without_comparison": duration_count,
        "identity_field_count_preserved_without_comparison": identity_count,
        "identity_fields": sorted(REGENERATION_IDENTITY_PATHS),
        "envelope_counts": envelope_counts,
        "envelope_maxima": envelope_maxima,
        "envelopes": {
            "binary64_roundoff_absolute": PARITY_ROUNDOFF_BOUND,
            "float_trace_relative": FLOAT_TRACE_RELATIVE_BOUND,
            "derived_contraction_relative": DERIVED_CONTRACTION_RELATIVE_BOUND,
            "roundoff_conditioned_ratio_relative": (
                ROUNDOFF_CONDITIONED_RATIO_RELATIVE_BOUND
            ),
        },
    }


def _merge_terminal_diagnostics(
    banked: dict[str, Any], regenerated: dict[str, Any]
) -> dict[str, Any]:
    """Enrich the bank after exact-semantic and bounded-float checks."""

    banked_measurement = dict(banked)
    banked_measurement.pop("xpoint_diagnostic_enrichment", None)
    regeneration_drift = _compare_regeneration_semantics(
        banked_measurement, regenerated
    )
    merged = json.loads(json.dumps(banked, allow_nan=False))
    regenerated_records = {
        (int(record["reference"]["shot"]), int(record["reference"]["slice_index"])): (
            record
        )
        for record in regenerated["references"]
    }
    for banked_record in merged["references"]:
        key = (
            int(banked_record["reference"]["shot"]),
            int(banked_record["reference"]["slice_index"]),
        )
        regenerated_record = regenerated_records[key]
        for arm_name in ("pure_arm", "mixed_arm"):
            banked_record[arm_name]["terminal_xpoint_diagnostics"] = regenerated_record[
                arm_name
            ]["terminal_xpoint_diagnostics"]
    regenerated_summary = _diagnostic_summary(regenerated["references"])
    merged["xpoint_diagnostic_enrichment"] = {
        **regenerated_summary,
        "regeneration_drift": regeneration_drift,
        "source_diagnostic": (
            "traced_margin_candidate_diagnostics via the shared terminal serializer"
        ),
        "source_diagnostic_commit": "597417af",
        "preservation_policy": (
            "the bank remains authoritative; every semantic field is exact, float "
            "drift must remain inside its declared envelope, and only regenerated "
            "terminal diagnostics are added"
        ),
        "benchmark_regression_findings": {
            "forward_topology_state_pytree": {
                "introduced_by_commit": "f2665b5d",
                "regression": (
                    "ForwardTopologyState changed from an automatically registered "
                    "NamedTuple to an unregistered dataclass carrying a callable, so "
                    "ForwardProfile.solve_portfolio failed when vmap returned it"
                ),
                "production_repair": (
                    "register a lossless PyTree whose tenth array leaf is class_margin "
                    "while keeping the callable out of the leaves"
                ),
                "production_batching_consumer": "nova/transport/coupled_window.py",
            },
            "removed_structured_axis_attributes": {
                "introduced_by_commit": "f2665b5d",
                "attributes": [
                    "_connectivity_radius",
                    "_connectivity_height",
                    "_connectivity_shape",
                ],
                "benchmark_expectation": (
                    "cached tensor-product radius and height arrays plus their shape"
                ),
                "public_repair": ("consume ForwardFluxOperator.connectivity_grid_axes"),
                "migrated_python_caller_count": 2,
                "migrated_python_callers": [
                    "benchmarks/pinned_branch_contrast.py",
                    "benchmarks/traced_selection_seam.py",
                ],
                "remaining_private_attribute_caller_count": 0,
                "product_module_caller_count": 0,
            },
        },
    }
    _compare_existing_numeric_fields(banked, merged)
    merged["xpoint_diagnostic_enrichment"][
        "existing_output_numeric_field_difference_count"
    ] = 0
    return merged


def _pure_arm(profile, seed: jax.Array, target_current: float) -> tuple[dict, Any]:
    """Run the production two-branch portfolio and retain its diverted arm."""
    initial = jnp.stack((seed, seed))
    portfolio = profile.solve_portfolio(
        initial,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    portfolio.branches.equilibrium.flux.block_until_ready()
    branch = jax.tree.map(
        lambda value: value[int(TopologyClass.DIVERTED)], portfolio.branches
    )
    sequence = _newton_residuals(branch.equilibrium.fixed_point.trace)
    record = {
        "entry_point": "ForwardProfile.solve_portfolio",
        "requested_class": "diverted",
        "terminal_residual": float(branch.residual),
        "converged": bool(branch.converged),
        "finite": bool(branch.equilibrium.finite.passed),
        "iterations": int(branch.iterations),
        "residual_sequence": sequence,
        "fitted_contraction": _fit_contraction(sequence),
        **_terminal_observables(profile, branch.equilibrium.flux),
    }
    return record, portfolio


def _mixed_arm(profile, seed: jax.Array, target_current: float) -> dict[str, Any]:
    """Run the margin-penalty mixed iteration on the same diverted map and seed."""
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    result = _margin_graded_newton_krylov(
        mapped,
        profile.operator.topology_margin,
        seed,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    result.state.block_until_ready()
    sequence = _serial_residuals(result.trace)
    image = mapped(result.state)
    terminal_residual = float(
        jnp.max(jnp.abs(image - result.state))
        / jnp.maximum(jnp.max(jnp.abs(image)), jnp.asarray(1.0e-30))
    )
    observables = _terminal_observables(profile, result.state)
    return {
        "entry_point": "margin-graded fixed-ladder Newton-Krylov",
        "requested_class": "diverted",
        "terminal_residual": terminal_residual,
        "converged": bool(
            np.isfinite(terminal_residual)
            and terminal_residual <= FIXED_POINT_CRITERION
            and observables["topology_consistent"]
        ),
        "finite": bool(jnp.all(jnp.isfinite(result.state))),
        "iterations": NEWTON_STEPS,
        "residual_sequence": sequence,
        "fitted_contraction": _fit_contraction(sequence),
        "accepted_class_margins": _serial_residuals(result.accepted_class_margins),
        "accepted_topology_penalties": _serial_residuals(
            result.accepted_topology_penalties
        ),
        **observables,
    }


def _batch_two_parity(profile, seed: jax.Array, target_current: float) -> dict:
    """Assert jitted single versus jitted-vmapped batch-two portfolio parity."""
    initial = jnp.stack((seed, seed))

    def solve(branch_seeds, current_target):
        return profile.solve_portfolio(
            branch_seeds,
            route="newton_krylov",
            target_current=current_target,
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
            warmup=WARMUP_SWEEPS,
            relaxation=RELAXATION,
            step_cap=STEP_CAP,
        )

    single = jax.jit(solve)(initial, jnp.asarray(target_current))
    batch = jax.jit(jax.vmap(solve))(
        jnp.stack((initial, initial)),
        jnp.asarray((target_current, target_current)),
    )
    batch.branches.equilibrium.flux.block_until_ready()
    single_flux = np.asarray(single.branches.equilibrium.flux, dtype=np.float64)
    batch_flux = np.asarray(batch.branches.equilibrium.flux, dtype=np.float64)
    scale = max(float(np.max(np.abs(single_flux))), np.finfo(float).tiny)
    flux_relative_difference = float(
        np.max(np.abs(batch_flux - single_flux[None])) / scale
    )
    residual_difference = float(
        np.max(
            np.abs(
                np.asarray(batch.branches.residual)
                - np.asarray(single.branches.residual)[None]
            )
        )
    )
    exact_receipt_parity = all(
        np.array_equal(
            np.asarray(batch_value),
            np.broadcast_to(
                np.asarray(single_value)[None], np.asarray(batch_value).shape
            ),
        )
        for batch_value, single_value in (
            (batch.branches.requested_class, single.branches.requested_class),
            (batch.branches.achieved_class, single.branches.achieved_class),
            (
                batch.branches.topology_consistent,
                single.branches.topology_consistent,
            ),
        )
    )
    passes = bool(
        flux_relative_difference <= PARITY_RELATIVE_TOLERANCE
        and residual_difference <= PARITY_RELATIVE_TOLERANCE
        and exact_receipt_parity
    )
    if not passes:
        raise AssertionError(
            "jitted-vmapped batch-two portfolio differs from the jitted single "
            f"portfolio: flux={flux_relative_difference:.6g}, "
            f"residual={residual_difference:.6g}, receipts={exact_receipt_parity}"
        )
    return {
        "passes": True,
        "batch_size": 2,
        "batch_members": "two identical copies of the first frozen reference",
        "registered_relative_tolerance": PARITY_RELATIVE_TOLERANCE,
        "maximum_flux_relative_difference": flux_relative_difference,
        "maximum_terminal_residual_absolute_difference": residual_difference,
        "exact_requested_achieved_and_consistency_receipt_parity": (
            exact_receipt_parity
        ),
    }


def run(
    *,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output: Path = DEFAULT_OUTPUT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Run and bank the paired six-reference contrast."""
    configure_dtypes()
    banked_receipt = json.loads(output.read_text()) if output.exists() else None
    source_revision = _source_revision()
    response_cache, carrier_evidence = _persisted_response_cache(
        carrier, carrier_receipt
    )
    selected = select_slices_by_shot(bank)
    records = []
    direct_builder_entries = 0
    parity = None
    for selected_row, qualification in selected:
        case, context = _mast_case_from_selection(store, selected_row, qualification)
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        direct_builder_entries += int(policy["section_kernel_evaluations_this_shot"])
        reference = passive_case["reference"]
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(reference["plasma_current_a"]))
        pure, _portfolio = _pure_arm(profile, seed, target_current)
        mixed = _mixed_arm(profile, seed, target_current)
        if parity is None:
            parity = _batch_two_parity(profile, seed, target_current)
            parity["reference"] = {
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
            }
        records.append(
            {
                "reference": reference,
                "seed_sha256": _digest(seed),
                "same_seed_both_arms": True,
                "pure_arm": pure,
                "mixed_arm": mixed,
                "terminal_residual_ratio_pure_over_mixed": float(
                    pure["terminal_residual"]
                    / max(mixed["terminal_residual"], np.finfo(float).tiny)
                ),
            }
        )

    if len(records) != 6:
        raise RuntimeError(f"expected six frozen references, measured {len(records)}")
    if direct_builder_entries != 0:
        raise RuntimeError("persisted-carrier run entered the direct response builder")
    if parity is None or not parity["passes"]:
        raise RuntimeError("the batch-two parity assertion did not run")

    pure_contracting = sum(
        bool(record["pure_arm"]["fitted_contraction"]["contracts"])
        for record in records
    )
    pure_diverted = sum(
        record["pure_arm"]["achieved_class"] == "diverted" for record in records
    )
    mixed_diverted = sum(
        record["mixed_arm"]["achieved_class"] == "diverted" for record in records
    )
    receipt = {
        "artifact": "paired pinned-branch and margin-penalty contrast",
        "source_commit": source_revision,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_contract": {
            "cohort": "six frozen MAST diverted-label references",
            "selection": "lowest worst-fraction qualified row per frozen shot",
            "seed": "the same labelled reference flux for both arms",
            "map": "all 101 fitted circuits with declared plasma-current pinning",
            "pure_arm": (
                "DIVERTED branch of production ForwardProfile.solve_portfolio"
            ),
            "mixed_arm": (
                "the same DIVERTED map ranked by relative residual plus the "
                "unit-weight penalty max(-class_margin, 0)"
            ),
            "fixed_point_criterion": FIXED_POINT_CRITERION,
            "newton_promotions": NEWTON_STEPS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "contraction_fit": (
                "ordinary least squares of log(residual) against zero-based "
                "Newton-promotion index over the twelve retained post-promotion "
                "residuals; fitted rho is exp(slope), and rho below one is called "
                "contraction"
            ),
            "smooth_class_observable": (
                "p_diverted from traced_smooth_boundary_read at temperature 0.01 "
                "using the exact margin comparator's X-point and wall operands"
            ),
            "terminal_comparison_qualification": (
                "margin-terminal comparisons cross two different iterations: "
                "the pure production residual policy and the mixed residual-plus-"
                "margin merit policy"
            ),
            "nonfinite_margin_policy": (
                "a non-finite terminal class margin is serialized as null with "
                "class_margin_nonfinite naming its sign; no numeric sentinel is used"
            ),
        },
        "response_carrier": carrier_evidence,
        "direct_green_operator_builder_entries": direct_builder_entries,
        "jit_vmap_batch_two_parity": parity,
        "summary": {
            "reference_count": len(records),
            "pure_fitted_contraction_count": pure_contracting,
            "pure_terminal_diverted_count": pure_diverted,
            "mixed_terminal_diverted_count": mixed_diverted,
            "pure_terminal_residual_range": [
                min(record["pure_arm"]["terminal_residual"] for record in records),
                max(record["pure_arm"]["terminal_residual"] for record in records),
            ],
            "mixed_terminal_residual_range": [
                min(record["mixed_arm"]["terminal_residual"] for record in records),
                max(record["mixed_arm"]["terminal_residual"] for record in records),
            ],
        },
        "references": records,
    }
    if banked_receipt is not None:
        receipt = _merge_terminal_diagnostics(banked_receipt, receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    """Run the paired contrast from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument(
        "--carrier-receipt",
        type=Path,
        default=response_carrier.DEFAULT_RECEIPT,
    )
    args = parser.parse_args()
    result = run(
        store=args.store,
        bank=args.bank,
        output=args.output,
        carrier=args.carrier,
        carrier_receipt=args.carrier_receipt,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
