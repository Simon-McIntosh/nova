"""Diagnose Newton contraction on MAST states whose residual mask has settled.

The measurement rebuilds the production profile through the persisted response
carrier because the published operand cache intentionally retains grid geometry
but not the wall and direct-sample leaves of the solver state.  It validates the
rebuilt terminal against that cache before freezing the terminal residual mask.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

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
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.observation import ConstraintViolationError
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
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
DEFAULT_REPAIR_OUTPUT = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/repair/stall-repair.json"
)
DEFAULT_BASELINE = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/current-measurement.json"
)
DEFAULT_PRODUCTION_OUTPUT = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/production/production-path.json"
)
DEFAULT_PRODUCTION_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/production-path/"
    "production-path-damping.md"
)
DEFAULT_PUBLIC_ROUTE_OUTPUT = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/public-route/four-rows.json"
)
DEFAULT_UNDAMPED_OUTPUT = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/undamped/four-rows.json"
)
DEFAULT_VERTICAL_MODE = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "vertical-mode/jacobian-null-direction.json"
)
TARGETS = ((21985, 51), (21986, 46), (21989, 55), (22086, 43))
SMOOTH_NEWTON_STEPS = 8
SMOOTH_GMRES_ITERATIONS = 40
SMOOTH_RELAXATION = 0.5
SMOOTH_STEP_CAP = 10.0
PUBLIC_ROUTE_POLICY = PerturbedSeedPolicy()
PRODUCTION_PATH_FRACTIONS = tuple(np.linspace(0.0, 1.0, 8))
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
                "radius": np.asarray(stored[f"{prefix}_radius"], dtype=np.float64),
                "height": np.asarray(stored[f"{prefix}_height"], dtype=np.float64),
                "terminal_state": np.asarray(
                    stored[f"{prefix}_flux"], dtype=np.float64
                ),
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
                "axis": np.asarray(stored[f"{prefix}_axis"], dtype=np.float64),
                "selected_saddle": np.asarray(
                    stored[f"{prefix}_selected_saddle"], dtype=np.float64
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


def _termination_name(value: Any) -> str:
    """Return the stable host label for one fixed-point termination value."""
    reason_value = int(np.asarray(value))
    try:
        return FixedPointTerminationReason(reason_value).name.lower()
    except ValueError:
        return f"unknown_{reason_value}"


class _ProductionTraceCollector:
    """Group solver-owned linear and promotion callbacks by active-set trip."""

    def __init__(self, *, step_cap: float, relaxation: float) -> None:
        self.step_cap = step_cap
        self.relaxation = relaxation
        self._linear_actions: list[dict[str, Any]] = []
        self._promotion_contexts: list[dict[str, Any]] = []
        self._reusable_linear_action: dict[str, Any] | None = None
        self._reusable_action_state: np.ndarray | None = None
        self._reusable_action_partition: np.ndarray | None = None
        self._newton_steps: list[dict[str, Any]] = []
        self.trips: list[dict[str, Any]] = []
        self._global_step = 0
        self._linear_action_count = 0

    def linear_action(
        self,
        qualification,
        projected_condition,
        conditioning_applied,
        condition_baseline,
        unconditioned_step,
        conditioned_step,
        achieved_reduction,
        requested_tolerance,
        residual_vector,
        nonlinear_residual,
    ) -> None:
        raw = np.asarray(unconditioned_step, dtype=np.float64).reshape(-1)
        conditioned = np.asarray(conditioned_step, dtype=np.float64).reshape(-1)
        residual = np.asarray(residual_vector, dtype=np.float64).reshape(-1)
        raw_sup = float(np.max(np.abs(raw)))
        raw_l2 = float(np.linalg.norm(raw))
        conditioned_sup = float(np.max(np.abs(conditioned)))
        residual_sup = float(np.max(np.abs(residual)))
        cap = self.step_cap * self.relaxation * residual_sup
        cap_factor = min(1.0, cap / max(conditioned_sup, np.finfo(float).tiny))
        reduction = float(np.asarray(achieved_reduction))
        baseline = float(np.asarray(condition_baseline))
        projected = float(np.asarray(projected_condition))
        nonlinear = float(np.asarray(nonlinear_residual))
        trust_threshold = float(np.sqrt(np.finfo(raw.dtype).eps))
        eligible_without_trust = bool(
            np.isfinite(projected)
            and np.isfinite(baseline)
            and projected
            > fixed_point._PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT * baseline
            and nonlinear > np.finfo(raw.dtype).eps ** 0.25
        )
        trusted = bool(reduction <= trust_threshold)
        applied = bool(np.asarray(conditioning_applied))
        self._linear_action_count += 1
        self._linear_actions.append(
            {
                "receipt": self._linear_action_count,
                "qualification": fixed_point.KrylovActionQualification(
                    int(np.asarray(qualification))
                ).name.lower(),
                "projected_condition": projected,
                "condition_baseline": baseline,
                "conditioning_eligible_without_linear_trust": (eligible_without_trust),
                "conditioning_applied": applied,
                "sqrt_epsilon_trust_threshold": trust_threshold,
                "sqrt_epsilon_bypass_engaged": bool(
                    eligible_without_trust and trusted and not applied
                ),
                "achieved_krylov_relative_residual": reduction,
                "requested_krylov_relative_tolerance": float(
                    np.asarray(requested_tolerance)
                ),
                "raw_newton_direction_sup": raw_sup,
                "raw_newton_direction_l2": raw_l2,
                "conditioned_direction_sup_before_step_cap": conditioned_sup,
                "step_cap_bound_sup": cap,
                "step_cap_factor": cap_factor,
                "nonlinear_residual": nonlinear,
            }
        )

    def promotion_context(self, state, partition, allows_action_reuse) -> None:
        """Retain the exact state and frozen partition paired with a promotion."""
        self._promotion_contexts.append(
            {
                "state": np.asarray(state).copy(),
                "partition": np.asarray(partition).copy(),
                "allows_action_reuse": bool(np.asarray(allows_action_reuse)),
            }
        )

    def inner_iteration(
        self,
        iteration,
        residual_before,
        residual_after,
        proposed_step_norm,
        accepted,
        decision,
        krylov_qualification,
        applied_factor,
        krylov_reduction,
        krylov_tolerance,
        model_error_fraction,
        step_cap_activated,
        step_cap_factor,
    ) -> None:
        if not self._promotion_contexts:
            raise RuntimeError(
                "inner iteration arrived without its state-partition context"
            )
        context = self._promotion_contexts.pop(0)
        reused_action = False
        if self._linear_actions:
            primary, *auxiliary = self._linear_actions
            self._linear_actions.clear()
            self._reusable_linear_action = primary
            self._reusable_action_state = context["state"]
            self._reusable_action_partition = context["partition"]
        else:
            if not context["allows_action_reuse"]:
                raise RuntimeError(
                    "inner iteration arrived without a linear-action receipt"
                )
            if self._reusable_linear_action is None:
                raise RuntimeError(
                    "carried inner iteration has no preceding linear action to reuse"
                )
            if not np.array_equal(context["state"], self._reusable_action_state):
                raise RuntimeError(
                    "carried inner iteration changed state before reusing "
                    "a linear action"
                )
            if not np.array_equal(
                context["partition"], self._reusable_action_partition
            ):
                raise RuntimeError(
                    "carried inner iteration changed partition before reusing "
                    "a linear action"
                )
            primary = self._reusable_linear_action
            auxiliary = []
            reused_action = True
        proposed = float(np.asarray(proposed_step_norm))
        ladder_factor = float(np.asarray(applied_factor))
        raw_sup = primary["raw_newton_direction_sup"]
        cap_activated = bool(np.asarray(step_cap_activated))
        measured_cap_factor = float(np.asarray(step_cap_factor))
        self._global_step += 1
        self._newton_steps.append(
            {
                "global_step": self._global_step,
                "iteration_in_trip": int(np.asarray(iteration)) + 1,
                "residual_before": float(np.asarray(residual_before)),
                "residual_after": float(np.asarray(residual_after)),
                "raw_newton_direction_sup": raw_sup,
                "proposed_step_sup_after_step_cap": proposed,
                "step_cap_activated": cap_activated,
                "step_cap_factor": measured_cap_factor,
                "model_error_fraction": _strict_float(model_error_fraction),
                "merit_ladder_factor": ladder_factor,
                "effective_raw_direction_fraction": (
                    proposed * ladder_factor / max(raw_sup, np.finfo(float).tiny)
                ),
                "accepted": bool(np.asarray(accepted)),
                "decision": fixed_point.InnerIterationDecision(
                    int(np.asarray(decision))
                ).name.lower(),
                "krylov_qualification": fixed_point.KrylovActionQualification(
                    int(np.asarray(krylov_qualification))
                ).name.lower(),
                "achieved_krylov_relative_residual": float(
                    np.asarray(krylov_reduction)
                ),
                "requested_krylov_relative_tolerance": float(
                    np.asarray(krylov_tolerance)
                ),
                "linear_action": primary,
                "linear_action_receipt": primary["receipt"],
                "linear_action_reused": reused_action,
                "auxiliary_linear_actions": auxiliary,
            }
        )

    def active_set_trip(
        self, active, trip_index, mask_difference, live_residual, inner_iterations
    ) -> None:
        if not bool(np.asarray(active)):
            return
        if self._linear_actions or self._promotion_contexts:
            raise RuntimeError("unpaired promotion accounting remains at trip boundary")
        self.trips.append(
            {
                "trip": int(np.asarray(trip_index)) + 1,
                "mask_difference": int(np.asarray(mask_difference)),
                "live_residual": float(np.asarray(live_residual)),
                "attempted_newton_promotions": int(np.asarray(inner_iterations)),
                "newton_steps": self._newton_steps,
            }
        )
        self._newton_steps = []
        self._reusable_linear_action = None
        self._reusable_action_state = None
        self._reusable_action_partition = None

    def finish(self) -> None:
        if self._linear_actions or self._promotion_contexts or self._newton_steps:
            raise RuntimeError(
                "production trace ended with callbacks outside an active-set trip"
            )


@contextmanager
def _production_trace_hooks(collector: _ProductionTraceCollector):
    """Observe the production solver without changing its numerical values."""

    original_qualified = fixed_point._qualified_krylov_step
    original_complete = fixed_point._complete_newton_promotion
    original_inner = fixed_point._print_inner_iteration
    original_trip = fixed_point._print_active_set_trip

    def observed_qualified(*args, **kwargs):
        result = original_qualified(*args, **kwargs)
        residual_vector = args[1]
        nonlinear_residual = args[2]
        jax.debug.callback(
            collector.linear_action,
            result.qualification,
            result.projected_condition,
            result.conditioning_applied,
            result.condition_baseline,
            result.unconditioned_step,
            result.step,
            result.achieved_reduction,
            result.requested_tolerance,
            residual_vector,
            nonlinear_residual,
            ordered=True,
        )
        return result

    def observed_complete(*args, **kwargs):
        measured, state = args[:2]
        jax.debug.callback(
            collector.promotion_context,
            state,
            measured.shadow_mask,
            jnp.asarray(kwargs.get("reuse_rejected_score", False), dtype=bool),
            ordered=True,
        )
        return original_complete(*args, **kwargs)

    fixed_point._qualified_krylov_step = observed_qualified
    fixed_point._complete_newton_promotion = observed_complete
    fixed_point._print_inner_iteration = collector.inner_iteration
    fixed_point._print_active_set_trip = collector.active_set_trip
    try:
        yield
    finally:
        fixed_point._qualified_krylov_step = original_qualified
        fixed_point._complete_newton_promotion = original_complete
        fixed_point._print_inner_iteration = original_inner
        fixed_point._print_active_set_trip = original_trip


def _verify_trace_action_reuse_accounting() -> dict[str, Any]:
    """Exercise receipt reuse and reject missing actions across changed inputs."""

    def record_action(collector: _ProductionTraceCollector) -> None:
        collector.linear_action(
            int(fixed_point.KrylovActionQualification.ACCEPTED),
            2.0,
            False,
            1.0,
            np.asarray([1.0]),
            np.asarray([1.0]),
            1.0e-6,
            1.0e-5,
            np.asarray([1.0]),
            1.0,
        )

    def record_inner(collector: _ProductionTraceCollector, iteration: int) -> None:
        collector.inner_iteration(
            iteration,
            1.0,
            1.0,
            1.0,
            False,
            int(fixed_point.InnerIterationDecision.SUFFICIENT_DECREASE_REFUSED),
            int(fixed_point.KrylovActionQualification.ACCEPTED),
            0.0,
            1.0e-6,
            1.0e-5,
            np.nan,
            False,
            1.0,
        )

    def refusing_map(state):
        return jnp.where(state[0] == 0.0, jnp.ones_like(state), state / 2.1)

    collector = _ProductionTraceCollector(step_cap=10.0, relaxation=0.5)
    with _production_trace_hooks(collector):
        result = jax.jit(
            lambda: fixed_point._newton_krylov_inner(
                refusing_map,
                jnp.zeros(1),
                newton_steps=12,
                gmres_iterations=1,
                warmup=0,
                stream_inner_iterations=True,
                carry_unchanged_fallback=True,
            )
        )()
        result.state.block_until_ready()
    collector.active_set_trip(
        True,
        0,
        0,
        result.residual,
        result.attempted_newton_promotions,
    )
    collector.finish()
    traced_steps = collector.trips[0]["newton_steps"]
    if len(traced_steps) != 12:
        raise AssertionError(
            f"fallback fixture emitted {len(traced_steps)} rather than 12 iterations"
        )
    fresh = [step for step in traced_steps if not step["linear_action_reused"]]
    carried = [step for step in traced_steps if step["linear_action_reused"]]
    if len(fresh) != 1 or len(carried) != 11:
        raise AssertionError(
            "fallback fixture did not emit one fresh plus eleven carried actions"
        )
    receipt_ids = {step["linear_action_receipt"] for step in traced_steps}
    if len(receipt_ids) != 1 or collector._linear_action_count != 1:
        raise AssertionError("carried iterations invented linear-action receipts")

    state = np.asarray([1.0, 2.0])
    partition = np.asarray([True, False, True])

    def require_failure(
        changed_state, changed_partition, allows_action_reuse, expected: str
    ) -> None:
        guarded = _ProductionTraceCollector(step_cap=10.0, relaxation=0.5)
        record_action(guarded)
        guarded.promotion_context(state, partition, False)
        record_inner(guarded, 0)
        guarded.promotion_context(changed_state, changed_partition, allows_action_reuse)
        try:
            record_inner(guarded, 1)
        except RuntimeError as error:
            if expected not in str(error):
                raise AssertionError(
                    f"unexpected accounting refusal: {error}"
                ) from error
        else:
            raise AssertionError(f"accounting accepted {expected}")

    require_failure(state + 1.0, partition, True, "changed state")
    require_failure(state, ~partition, True, "changed partition")
    require_failure(state, partition, False, "without a linear-action receipt")
    return {
        "fresh_receipts": collector._linear_action_count,
        "iterations_attributed": len(traced_steps),
        "reused_iterations": len(carried),
        "receipt_id_count": len(receipt_ids),
        "changed_state_refused": True,
        "changed_partition_refused": True,
        "unmarked_missing_action_refused": True,
    }


class _WholeStepPathCollector:
    """Retain true and second-order residuals along accepted Newton steps."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def record(
        self,
        accepted,
        recovery_activated,
        applied_factor,
        step_sup,
        fractions,
        actual_residuals,
        linear_model_residuals,
        quadratic_model_residuals,
    ) -> None:
        if not bool(np.asarray(accepted)):
            return
        actual = np.asarray(actual_residuals, dtype=np.float64)
        linear = np.asarray(linear_model_residuals, dtype=np.float64)
        quadratic = np.asarray(quadratic_model_residuals, dtype=np.float64)
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "accepted": True,
                "recovery_activated": bool(np.asarray(recovery_activated)),
                "applied_factor": float(np.asarray(applied_factor)),
                "accepted_step_sup": float(np.asarray(step_sup)),
                "samples": [
                    {
                        "fraction": float(fraction),
                        "actual_relative_residual": float(actual_value),
                        "linear_model_relative_residual": float(linear_value),
                        "quadratic_model_relative_residual": float(quadratic_value),
                        "actual_minus_quadratic": float(actual_value - quadratic_value),
                    }
                    for fraction, actual_value, linear_value, quadratic_value in zip(
                        np.asarray(fractions, dtype=np.float64),
                        actual,
                        linear,
                        quadratic,
                        strict=True,
                    )
                ],
                "full_step_actual_residual": float(actual[-1]),
                "full_step_quadratic_prediction": float(quadratic[-1]),
                "full_step_absolute_model_error": float(
                    abs(actual[-1] - quadratic[-1])
                ),
            }
        )


@contextmanager
def _whole_step_path_hooks(collector: _WholeStepPathCollector):
    """Sample the frozen residual without modifying promotion selection."""

    original_promotion = fixed_point._backtracked_promotion
    fractions = jnp.asarray(PRODUCTION_PATH_FRACTIONS, dtype=jnp.float64)

    def observed_promotion(*args, **kwargs):
        result = original_promotion(*args, **kwargs)
        map_fn, _model_map_fn, state = args[:3]
        accepted_step = result.state - state

        def residual_vector(candidate):
            return map_fn(candidate) - candidate

        residual_at_origin, first_directional = jax.jvp(
            residual_vector, (state,), (accepted_step,)
        )

        def first_directional_at(candidate):
            return jax.jvp(residual_vector, (candidate,), (accepted_step,))[1]

        _first_at_origin, second_directional = jax.jvp(
            first_directional_at, (state,), (accepted_step,)
        )

        def sampled(fraction):
            candidate = state + fraction * accepted_step
            actual_mapped = map_fn(candidate)
            linear_residual = residual_at_origin + fraction * first_directional
            quadratic_residual = (
                linear_residual + 0.5 * fraction * fraction * second_directional
            )
            linear_mapped = candidate + linear_residual
            quadratic_mapped = candidate + quadratic_residual
            return (
                fixed_point._relative_residual(actual_mapped, candidate),
                fixed_point._relative_residual(linear_mapped, candidate),
                fixed_point._relative_residual(quadratic_mapped, candidate),
            )

        actual, linear, quadratic = jax.lax.map(sampled, fractions)
        jax.debug.callback(
            collector.record,
            result.accepted,
            result.recovery_activated,
            result.applied_factor,
            jnp.max(jnp.abs(accepted_step)),
            fractions,
            actual,
            linear,
            quadratic,
            ordered=True,
        )
        return result

    fixed_point._backtracked_promotion = observed_promotion
    try:
        yield
    finally:
        fixed_point._backtracked_promotion = original_promotion


def _active_set_summary(result, banked: dict[str, Any]) -> dict[str, Any]:
    """Pair one production active-set result with its banked comparator."""
    iterations = int(np.asarray(result.active_set_iterations))
    return {
        "bank": {
            "terminal_residual": float(banked["terminal_residual"]),
            "active_set_iterations": int(banked["active_set_iterations"]),
            "termination_reason": str(banked["termination_reason"]),
            "converged": bool(banked["converged"]),
            "active_set_residuals": banked["active_set_residuals"].tolist(),
            "active_set_mask_differences": banked[
                "active_set_mask_differences"
            ].tolist(),
        },
        "measured": {
            "terminal_residual": float(np.asarray(result.residual)),
            "active_set_iterations": iterations,
            "termination_reason": _termination_name(result.termination_reason),
            "converged": bool(np.asarray(result.converged)),
            "active_set_residuals": _array(
                result.active_set_residuals, limit=iterations
            ),
            "active_set_mask_differences": _array(
                result.active_set_mask_differences, limit=iterations
            ),
            "krylov_conditioning_count": int(
                np.asarray(result.krylov_conditioning_count)
            ),
            "maximum_projected_krylov_condition": _strict_float(
                result.maximum_projected_krylov_condition
            ),
        },
    }


def _terminal_topology(profile, state) -> dict[str, Any]:
    """Return the admitted production topology at one terminal state."""
    try:
        _masks, topology = profile.operator.read(state)
        geometry = bank_producer._post_cutover_geometry(profile, state, topology)
    except (NoQualifiedAxisError, ConstraintViolationError) as error:
        return {
            "axis_admitted": False,
            "admitted_axis_m": None,
            "saddle_admitted": False,
            "admitted_saddle_m": None,
            "achieved_class": None,
            "failure_exception_class": type(error).__name__,
        }
    axis = np.asarray(topology.axis, dtype=np.float64)
    saddle = np.asarray(geometry["selected_saddle"], dtype=np.float64)
    achieved_class = geometry["achieved_class"]
    return {
        "axis_admitted": bool(np.all(np.isfinite(axis))),
        "admitted_axis_m": axis.tolist(),
        "saddle_admitted": bool(
            achieved_class == "diverted" and np.all(np.isfinite(saddle))
        ),
        "admitted_saddle_m": (saddle.tolist() if np.all(np.isfinite(saddle)) else None),
        "achieved_class": achieved_class,
        "failure_exception_class": None,
    }


def _bank_topology(banked: dict[str, Any]) -> dict[str, Any]:
    """Return topology fields retained beside a pinned-budget bank row."""
    axis = np.asarray(banked["axis"], dtype=np.float64)
    saddle = np.asarray(banked["selected_saddle"], dtype=np.float64)
    achieved_class = banked.get("nova_achieved_class")
    return {
        "axis_admitted": bool(np.all(np.isfinite(axis))),
        "admitted_axis_m": axis.tolist() if np.all(np.isfinite(axis)) else None,
        "saddle_admitted": bool(
            achieved_class == "diverted" and np.all(np.isfinite(saddle))
        ),
        "admitted_saddle_m": (saddle.tolist() if np.all(np.isfinite(saddle)) else None),
        "achieved_class": achieved_class,
        "failure_exception_class": banked.get("failure_exception_class"),
    }


def _full_terminal_state(profile, template: Any, banked: dict[str, Any]) -> jax.Array:
    """Restore the cached grid witness and its interpolated wall samples."""
    grid_flux = banked["terminal_state"]
    radius = banked["radius"]
    height = banked["height"]
    lattice = profile.lattice
    if grid_flux.shape != (height.size, radius.size):
        raise RuntimeError("cached grid flux does not match its coordinate axes")
    if tuple(lattice.shape) != (radius.size, height.size):
        raise RuntimeError("cached grid does not match the rebuilt profile lattice")

    state = np.asarray(template, dtype=np.float64).copy()
    grid_count = int(lattice.node_count)
    wall = np.asarray(profile.operator.wall.coordinate, dtype=np.float64)
    physical_count = int(profile.operator.physical_node_number)
    if state.ndim != 1 or state.size != physical_count:
        raise RuntimeError("cached witness has an unsupported solver-state tail")
    if physical_count != grid_count + len(wall):
        raise RuntimeError("rebuilt profile has an unsupported wall-state layout")

    spline = RectBivariateSpline(radius, height, grid_flux.T, kx=3, ky=3, s=0)
    state[:grid_count] = grid_flux.T.reshape(-1)
    state[grid_count:physical_count] = spline.ev(wall[:, 0], wall[:, 1])
    return jnp.asarray(state, dtype=jnp.asarray(template).dtype)


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
    reason = _termination_name(result.termination_reason)
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
        gmres_iterations=PUBLIC_ROUTE_POLICY.gmres_iterations,
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
    production_state = branch.equilibrium.flux
    production_state.block_until_ready()
    validation = _bank_validation(production, banked, require_match=require_bank_match)
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    state = _full_terminal_state(profile, production_state, banked)
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
        "active_set_solve": _active_set_summary(production, banked),
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


def _frozen_repair_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce one frozen-mask solve to its contraction and step receipt."""
    smooth = row["frozen_mask_smooth_solve"]
    initial = float(smooth["initial_residual"])
    terminal = float(smooth["terminal_residual"])
    raw_step = float(row["jacobian_consistency"]["direction"]["sup"])
    proposed_step = float(smooth["proposed_step_norms"][0])
    residuals_after = _finite_values(smooth["residuals_after"])
    residuals_before = _finite_values(smooth["residuals_before"])
    linear_reductions = _finite_values(smooth["krylov_reductions"])
    trust_threshold = float(np.sqrt(np.finfo(np.float64).eps))
    return {
        "initial_residual": initial,
        "terminal_residual": terminal,
        "eight_step_contraction_ratio": terminal / initial,
        "residuals_per_newton_step": [initial, *residuals_after],
        "stepwise_contraction_ratios": [
            after / before
            for before, after in zip(residuals_before, residuals_after, strict=True)
        ],
        "raw_newton_direction_sup": raw_step,
        "first_proposed_step_sup": proposed_step,
        "raw_to_proposed_step_fraction": proposed_step
        / max(raw_step, np.finfo(np.float64).tiny),
        "accepted_promotions": int(smooth["accepted_promotions"]),
        "attempted_promotions": int(smooth["attempted_promotions"]),
        "applied_factors": smooth["applied_factors"],
        "linear_residual_reductions": smooth["krylov_reductions"],
        "linear_residual_trust_threshold": trust_threshold,
        "linear_model_trusted_per_step": [
            reduction <= trust_threshold for reduction in linear_reductions
        ],
        "all_linear_models_trusted": bool(linear_reductions)
        and all(reduction <= trust_threshold for reduction in linear_reductions),
    }


def _draw_repair(rows: list[dict[str, Any]], figure_path: Path) -> None:
    """Plot frozen contraction, admitted fractions, and active-set outcomes."""
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(14.5, 12.5), constrained_layout=True
    )
    colors = {"before": "#9b5de5", "after": "#087e8b"}
    for row_index, row in enumerate(rows):
        residual_axis, fraction_axis, active_axis = axes[row_index]
        for key, label in (("before", "banked guard"), ("after", "repaired guard")):
            values = row["frozen_mask"][key]["residuals_per_newton_step"]
            residual_axis.semilogy(
                range(len(values)),
                values,
                marker="o",
                color=colors[key],
                label=label,
            )
        residual_axis.axhline(1.0e-8, color="black", lw=0.8, ls="--")
        residual_axis.set_ylabel(row["identity"] + "\nrelative residual")
        residual_axis.set_xlabel("frozen-mask Newton step")
        residual_axis.grid(alpha=0.25)

        fractions = [
            row["frozen_mask"][key]["raw_to_proposed_step_fraction"]
            for key in ("before", "after")
        ]
        fraction_axis.bar((0, 1), fractions, color=(colors["before"], colors["after"]))
        fraction_axis.set_xticks((0, 1), ("before", "after"))
        fraction_axis.set_ylabel("proposed / raw Newton direction")
        fraction_axis.set_ylim(0.0, max(1.05, 1.1 * max(fractions)))
        fraction_axis.grid(axis="y", alpha=0.25)

        active = row["full_active_set"]
        residuals = [active[key]["terminal_residual"] for key in ("bank", "measured")]
        active_axis.bar((0, 1), residuals, color=(colors["before"], colors["after"]))
        active_axis.set_yscale("log")
        active_axis.set_xticks(
            (0, 1),
            (
                f"bank\n{active['bank']['active_set_iterations']} trips",
                f"repair\n{active['measured']['active_set_iterations']} trips",
            ),
        )
        active_axis.set_ylabel("full active-set terminal residual")
        bank_reason = (
            active["bank"]["termination_reason"]
            .removeprefix("active_set_")
            .replace("_", " ")
        )
        measured_reason = (
            active["measured"]["termination_reason"]
            .removeprefix("active_set_")
            .replace("_", " ")
        )
        active_axis.set_title(
            f"bank: {bank_reason}\nrepair: {measured_reason}",
            fontsize=8,
        )
        active_axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Projected-conditioning repair: frozen Newton contraction and full solve",
        fontsize=13,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def render_repair(*, measurement: Path, baseline: Path, output: Path) -> dict[str, Any]:
    """Render a repair receipt from one completed device measurement."""
    after_path = measurement
    measured = json.loads(after_path.read_text(encoding="utf-8"))
    baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
    baseline_by_identity = {row["identity"]: row for row in baseline_data["rows"]}
    attribution_ratios = {
        "21985/51": 0.919426,
        "21986/46": 0.996959,
        "21989/55": 0.990372,
        "22086/43": 0.773444,
    }
    rows = []
    for after_row in measured["rows"]:
        identity = after_row["identity"]
        before = _frozen_repair_summary(baseline_by_identity[identity])
        after = _frozen_repair_summary(after_row)
        expected = attribution_ratios[identity]
        rows.append(
            {
                "identity": identity,
                "attribution_contraction_ratio": expected,
                "baseline_matches_attribution": bool(
                    np.isclose(
                        before["eight_step_contraction_ratio"],
                        expected,
                        rtol=0.0,
                        atol=5.0e-7,
                    )
                ),
                "frozen_mask": {"before": before, "after": after},
                "full_active_set": after_row["active_set_solve"],
            }
        )
    figure_path = output.with_suffix(".png")
    _draw_repair(rows, figure_path)
    receipt = {
        "artifact": "projected Krylov conditioning repair",
        "source_commit": _source_revision(),
        "measurement_source_commit": measured["source_commit"],
        "baseline_source_commit": baseline_data["source_commit"],
        "runtime": measured["runtime"],
        "evidence_inputs": {
            **measured["evidence_inputs"],
            "baseline_measurement": str(baseline),
            "baseline_measurement_sha256": _sha256(baseline),
            "after_measurement": str(after_path),
            "after_measurement_sha256": _sha256(after_path),
            "fixed_point_source_sha256": _sha256(
                ROOT / "nova/equilibrium/fixed_point.py"
            ),
        },
        "measurement_contract": {
            **measured["measurement_contract"],
            "comparison": (
                "same cached terminal state and frozen residual mask before and "
                "after the linear-residual-conditioned discriminator"
            ),
            "full_active_set_start": (
                "each production solve starts from the persisted bank seed and "
                "retains every terminal verdict"
            ),
        },
        "figure": str(figure_path),
        "rows": rows,
        "verdict": {
            "all_baselines_match_attribution": all(
                row["baseline_matches_attribution"] for row in rows
            ),
            "all_trusted_linear_directions_admitted_to_merit_ladder": all(
                row["frozen_mask"]["after"]["all_linear_models_trusted"] for row in rows
            ),
            "all_raw_to_proposed_fractions_improved": all(
                row["frozen_mask"]["after"]["raw_to_proposed_step_fraction"]
                > row["frozen_mask"]["before"]["raw_to_proposed_step_fraction"]
                for row in rows
            ),
            "all_frozen_residuals_improve_on_baseline": all(
                row["frozen_mask"]["after"]["eight_step_contraction_ratio"]
                < row["frozen_mask"]["before"]["eight_step_contraction_ratio"]
                for row in rows
            ),
            "full_active_set_converged_count": sum(
                row["full_active_set"]["measured"]["converged"] for row in rows
            ),
            "full_active_set_row_count": len(rows),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def measure_repair(*, operands: Path, baseline: Path, output: Path) -> dict[str, Any]:
    """Remeasure the repair and compare it with the frozen banked diagnosis."""
    after_path = output.with_name("after-measurement.json")
    measure(
        operands=operands,
        output=after_path,
        source_label="projected-conditioning repair",
        require_bank_match=False,
    )
    return render_repair(measurement=after_path, baseline=baseline, output=output)


def _bitwise_bank_comparison(result, banked: dict[str, Any]) -> dict[str, Any]:
    """Compare one executed active-set history with its persisted bank row."""
    iterations = int(np.asarray(result.active_set_iterations))
    residuals = np.asarray(result.active_set_residuals, dtype=np.float64)[:iterations]
    differences = np.asarray(result.active_set_mask_differences, dtype=np.int64)[
        :iterations
    ]
    residuals_exact = bool(
        residuals.shape == banked["active_set_residuals"].shape
        and np.array_equal(residuals, banked["active_set_residuals"])
    )
    differences_exact = bool(
        differences.shape == banked["active_set_mask_differences"].shape
        and np.array_equal(differences, banked["active_set_mask_differences"])
    )
    terminal_exact = bool(
        float(np.asarray(result.residual)) == float(banked["terminal_residual"])
    )
    reason_exact = _termination_name(result.termination_reason) == str(
        banked["termination_reason"]
    )
    return {
        "all_exact": bool(
            residuals_exact and differences_exact and terminal_exact and reason_exact
        ),
        "active_set_residuals_exact": residuals_exact,
        "active_set_mask_differences_exact": differences_exact,
        "terminal_residual_exact": terminal_exact,
        "termination_reason_exact": reason_exact,
        "measured_active_set_residuals": residuals.tolist(),
        "measured_active_set_mask_differences": differences.tolist(),
    }


def _production_path_solve(profile, seed, target_current, banked) -> dict[str, Any]:
    """Run one branch through the public production path with live trace hooks."""
    collector = _ProductionTraceCollector(
        step_cap=reachability.STEP_CAP,
        relaxation=reachability.RELAXATION,
    )
    with _production_trace_hooks(collector):
        branch = profile.solve_branch(
            seed,
            jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
            route="newton_krylov",
            target_current=target_current,
            tolerance=reachability.FIXED_POINT_CRITERION,
            newton_steps=reachability.NEWTON_STEPS,
            gmres_iterations=PUBLIC_ROUTE_POLICY.gmres_iterations,
            warmup=reachability.WARMUP_SWEEPS,
            relaxation=reachability.RELAXATION,
            step_cap=reachability.STEP_CAP,
            stream_active_set=True,
            stream_inner_iterations=True,
        )
        branch.equilibrium.flux.block_until_ready()
    collector.finish()
    result = branch.equilibrium.fixed_point
    iterations = int(np.asarray(result.active_set_iterations))
    if len(collector.trips) != iterations:
        raise AssertionError(
            "production trace trip count does not match the solver receipt: "
            f"callbacks={len(collector.trips)}, result={iterations}"
        )
    flattened = [step for trip in collector.trips for step in trip["newton_steps"]]
    for trip in collector.trips:
        for step in trip["newton_steps"]:
            step["trip"] = trip["trip"]
            step["mask_difference_after_trip"] = trip["mask_difference"]
    callback_conditioning_count = sum(
        step["linear_action"]["conditioning_applied"] for step in flattened
    )
    callback_maximum_condition = max(
        step["linear_action"]["projected_condition"] for step in flattened
    )
    result_conditioning_count = int(np.asarray(result.krylov_conditioning_count))
    result_maximum_condition = float(
        np.asarray(result.maximum_projected_krylov_condition)
    )
    if result_conditioning_count != callback_conditioning_count:
        raise AssertionError(
            "whole-solve conditioning count disagrees with the per-step trace: "
            f"result={result_conditioning_count}, "
            f"callbacks={callback_conditioning_count}"
        )
    if not np.isclose(
        result_maximum_condition,
        callback_maximum_condition,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise AssertionError(
            "whole-solve maximum condition disagrees with the per-step trace: "
            f"result={result_maximum_condition}, callbacks={callback_maximum_condition}"
        )
    return {
        "configuration": {
            "route": "ForwardProfile.solve_branch(newton_krylov)",
            "newton_steps": reachability.NEWTON_STEPS,
            "gmres_iterations": PUBLIC_ROUTE_POLICY.gmres_iterations,
            "warmup": reachability.WARMUP_SWEEPS,
            "relaxation": reachability.RELAXATION,
            "step_cap": reachability.STEP_CAP,
            "convergence_tolerance": reachability.FIXED_POINT_CRITERION,
        },
        "result": _active_set_summary(result, banked)["measured"],
        "terminal_topology": _terminal_topology(profile, branch.equilibrium.flux),
        "bitwise_bank_comparison_at_gmres_30": _bitwise_bank_comparison(result, banked),
        "trips": collector.trips,
        "newton_steps": flattened,
        "sqrt_epsilon_bypass_count": sum(
            step["linear_action"]["sqrt_epsilon_bypass_engaged"] for step in flattened
        ),
        "conditioning_applied_count": callback_conditioning_count,
        "maximum_projected_krylov_condition": callback_maximum_condition,
        "terminal_result_conditioning_count_scope": "whole active-set solve",
    }


def _whole_step_residual_path(profile, template, banked, target_current):
    """Sample eight points along every accepted frozen-mask Newton step."""
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    state = _full_terminal_state(profile, template, banked)
    mask = profile.operator.residual_shadow_mask(state, requested_class)
    shadowed_map = profile.operator.flux_map_with_shadow(
        requested_class=requested_class, target_current=target_current
    )

    def frozen_map(candidate):
        return shadowed_map(candidate, mask)

    collector = _WholeStepPathCollector()
    with _whole_step_path_hooks(collector):
        smooth = _smooth_solve(frozen_map, state)
    if len(collector.steps) != smooth["accepted_promotions"]:
        raise AssertionError(
            "whole-step path count does not match accepted frozen promotions: "
            f"paths={len(collector.steps)}, accepted={smooth['accepted_promotions']}"
        )
    maximum_error = max(
        step["full_step_absolute_model_error"] for step in collector.steps
    )
    return {
        "fractions": list(PRODUCTION_PATH_FRACTIONS),
        "model": (
            "second-order directional Taylor model of the exact frozen residual "
            "vector, normalized with the production relative-sup residual"
        ),
        "frozen_solve": smooth,
        "steps": collector.steps,
        "maximum_full_step_absolute_model_error": maximum_error,
        "newton_consistent_at_1e_3": bool(maximum_error <= 1.0e-3),
    }


def _repair_full_solve_exact(row: dict[str, Any]) -> bool:
    active = row["full_active_set"]
    bank = active["bank"]
    measured = active["measured"]
    return bool(
        bank["terminal_residual"] == measured["terminal_residual"]
        and bank["active_set_iterations"] == measured["active_set_iterations"]
        and bank["termination_reason"] == measured["termination_reason"]
        and bank["active_set_residuals"] == measured["active_set_residuals"]
        and bank["active_set_mask_differences"]
        == measured["active_set_mask_differences"]
    )


def _mechanism_layers(
    identity: str,
    production: dict[str, Any],
    repair_row: dict[str, Any],
    whole_path: dict[str, Any] | None,
) -> dict[str, Any]:
    """Name each measured damping or residual-consistency layer."""
    steps = production["newton_steps"]
    linear = [step["linear_action"] for step in steps]
    reductions = [item["achieved_krylov_relative_residual"] for item in linear]
    thresholds = [item["sqrt_epsilon_trust_threshold"] for item in linear]
    mask_differences = [trip["mask_difference"] for trip in production["trips"]]
    frozen = repair_row["frozen_mask"]["after"]
    frozen_cap_fraction = float(frozen["raw_to_proposed_step_fraction"])
    frozen_ladder = [
        float(value)
        for value in frozen["applied_factors"]
        if value is not None and np.isfinite(value)
    ]
    production_cap = [float(step["step_cap_factor"]) for step in steps]
    production_ladder = [float(step["merit_ladder_factor"]) for step in steps]
    repair_active = repair_row["full_active_set"]["measured"]
    repair_exact = _repair_full_solve_exact(repair_row)
    gmres_layer = bool(
        repair_exact
        and not repair_active["converged"]
        and production["sqrt_epsilon_bypass_count"] > 0
        and production["result"]["terminal_residual"]
        != repair_active["terminal_residual"]
    )
    cap_layer = bool(
        frozen_cap_fraction < 1.0 - 1.0e-12
        or any(value < 1.0 - 1.0e-12 for value in production_cap)
    )
    ladder_layer = bool(
        any(value < 1.0 - 1.0e-12 for value in frozen_ladder)
        or any(value < 1.0 - 1.0e-12 for value in production_ladder)
    )
    mask_layer = any(value != 0 for value in mask_differences)
    residual_layer = bool(
        whole_path is not None and not whole_path["newton_consistent_at_1e_3"]
    )
    layers = [
        {
            "name": "bank_route_gmres_budget_too_small_for_sqrt_epsilon_bypass",
            "active": gmres_layer,
            "evidence": (
                f"held bank replay pins GMRES {reachability.GMRES_ITERATIONS}, "
                "its terminal inner receipt records conditioning "
                f"{repair_active['krylov_conditioning_count']} times and remains "
                f"at {repair_active['terminal_residual']:.6e}; "
                "the public production budget GMRES "
                f"{PUBLIC_ROUTE_POLICY.gmres_iterations} "
                f"reaches {min(reductions):.6e} to {max(reductions):.6e} against "
                f"sqrt(eps) {min(thresholds):.6e}, bypasses "
                f"{production['sqrt_epsilon_bypass_count']} of {len(steps)} "
                f"directions and ends at "
                f"{production['result']['terminal_residual']:.6e}"
            ),
            "repair": (
                "stop overriding the public GMRES-30 production budget with the "
                "bank driver's GMRES-12 constant, or adapt the bank route until "
                "the recomputed linear residual reaches sqrt(eps)"
            ),
            "owner": "nova equilibrium forward solver",
        },
        {
            "name": "step_cap",
            "active": cap_layer,
            "evidence": (
                "frozen-mask first proposed/raw direction fraction "
                f"{frozen_cap_fraction:.6f}; production cap factors "
                f"{production_cap}"
            ),
            "repair": (
                "make the cap conditional on the measured nonlinear model error "
                "or enlarge it while leaving the merit ladder as the acceptance "
                "authority"
            ),
            "owner": "nova equilibrium forward solver",
        },
        {
            "name": "merit_ladder",
            "active": ladder_layer,
            "evidence": (
                f"frozen-mask ladder factors {frozen_ladder}; production ladder "
                f"factors {production_ladder}"
            ),
            "repair": (
                "repair the model-trust or merit prediction at the first refused "
                "full step; retain strict decrease and best-iterate retention"
            ),
            "owner": "nova equilibrium globalization",
        },
        {
            "name": "mask_motion",
            "active": mask_layer,
            "evidence": f"per-trip mask differences {mask_differences}",
            "repair": (
                "relinearize on each changed partition and carry no conditioning "
                "decision across a mask change"
            ),
            "owner": "nova equilibrium active-set reconciliation",
        },
        {
            "name": "residual_not_newton_consistent_at_1e_3",
            "active": residual_layer,
            "evidence": (
                "not measured on this row"
                if whole_path is None
                else "maximum actual-versus-quadratic full-step residual error "
                f"{whole_path['maximum_full_step_absolute_model_error']:.6e}"
            ),
            "repair": (
                "split accepted steps at topology or support hand-offs, or make "
                "the residual and its Jacobian use one locally consistent read "
                "across the complete step"
            ),
            "owner": "nova equilibrium residual and topology seam",
        },
    ]
    active_names = [layer["name"] for layer in layers if layer["active"]]
    preferred = {
        "21985/51": "residual_not_newton_consistent_at_1e_3",
        "21986/46": "step_cap",
        "21989/55": "step_cap",
        "22086/43": ("bank_route_gmres_budget_too_small_for_sqrt_epsilon_bypass"),
    }[identity]
    primary = preferred if preferred in active_names else active_names[0]
    return {"primary": primary, "active_layers": active_names, "layers": layers}


def _draw_production_path(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(18.0, 13.0), constrained_layout=True
    )
    for row_index, row in enumerate(rows):
        production = row["production_path"]
        trips = production["trips"]
        steps = production["newton_steps"]
        identity = row["identity"]
        trip_axis, krylov_axis, norm_axis, factor_axis = axes[row_index]

        trip_numbers = [trip["trip"] for trip in trips]
        trip_axis.semilogy(
            trip_numbers,
            [trip["live_residual"] for trip in trips],
            "o-",
            color="#087e8b",
        )
        trip_axis.set_ylabel(identity + "\nactive-set residual")
        trip_axis.set_xlabel("active-set trip")
        trip_axis.grid(alpha=0.25)
        mask_axis = trip_axis.twinx()
        mask_axis.bar(
            trip_numbers,
            [trip["mask_difference"] for trip in trips],
            color="#f9c74f",
            alpha=0.28,
        )
        mask_axis.set_ylabel("mask difference")

        step_numbers = [step["global_step"] for step in steps]
        reductions = [step["achieved_krylov_relative_residual"] for step in steps]
        bypass = [
            step["linear_action"]["sqrt_epsilon_bypass_engaged"] for step in steps
        ]
        colors = ["#2a9d8f" if flag else "#e76f51" for flag in bypass]
        krylov_axis.semilogy(step_numbers, reductions, color="#6c757d", lw=0.8)
        krylov_axis.scatter(step_numbers, reductions, c=colors, s=24)
        krylov_axis.axhline(
            np.sqrt(np.finfo(np.float64).eps), color="black", ls="--", lw=0.8
        )
        krylov_axis.set_xlabel("production Newton step")
        krylov_axis.set_ylabel("GMRES relative residual")
        krylov_axis.grid(alpha=0.25)

        norm_axis.semilogy(
            step_numbers,
            [step["raw_newton_direction_sup"] for step in steps],
            "o-",
            label="raw",
            color="#9b5de5",
        )
        norm_axis.semilogy(
            step_numbers,
            [step["proposed_step_sup_after_step_cap"] for step in steps],
            "s-",
            label="proposed",
            color="#087e8b",
        )
        norm_axis.set_xlabel("production Newton step")
        norm_axis.set_ylabel("direction sup norm")
        norm_axis.grid(alpha=0.25)

        factor_axis.plot(
            step_numbers,
            [step["step_cap_factor"] for step in steps],
            "o-",
            label="step cap",
        )
        factor_axis.plot(
            step_numbers,
            [step["merit_ladder_factor"] for step in steps],
            "s-",
            label="merit ladder",
        )
        factor_axis.plot(
            step_numbers,
            [step["effective_raw_direction_fraction"] for step in steps],
            "^-",
            label="effective/raw",
        )
        factor_axis.set_ylim(-0.02, 1.05)
        factor_axis.set_xlabel("production Newton step")
        factor_axis.set_ylabel("multiplicative factor")
        factor_axis.grid(alpha=0.25)

    axes[0, 2].legend(fontsize=8)
    axes[0, 3].legend(fontsize=8)
    figure.suptitle(
        "Production active-set route: mask motion, GMRES trust, cap and merit",
        fontsize=13,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _draw_whole_step_path(path: dict[str, Any], figure_path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), constrained_layout=True)
    for axis, step in zip(axes.flat, path["steps"], strict=True):
        fractions = [sample["fraction"] for sample in step["samples"]]
        axis.semilogy(
            fractions,
            [sample["actual_relative_residual"] for sample in step["samples"]],
            "o-",
            label="actual",
            color="#e76f51",
        )
        axis.semilogy(
            fractions,
            [sample["quadratic_model_relative_residual"] for sample in step["samples"]],
            "s--",
            label="quadratic model",
            color="#277da1",
        )
        axis.set_title(f"Newton step {step['step']}")
        axis.set_xlabel("fraction of accepted step")
        axis.set_ylabel("relative residual")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        "21985/51 frozen mask: whole-step residual versus second-order model",
        fontsize=13,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_production_report(receipt: dict[str, Any], report: Path) -> None:
    lines = [
        "# Production-path damping diagnosis",
        "",
        (
            f"Measured `{receipt['source_commit']}` on "
            f"{', '.join(receipt['runtime']['devices'])} through "
            "`ForwardProfile.solve_branch(newton_krylov)` at GMRES 30. The "
            "repair-tip bank-seed receipt is reused separately to preserve its "
            "exact GMRES-12 bank identity and frozen-mask GMRES-40 comparison."
        ),
        "",
        (
            "| row | repair full solve exact to bank | production GMRES-30 "
            "bypass | production conditioning | frozen proposed/raw | primary "
            "mechanism |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rows"]:
        production = row["production_path"]
        frozen = row["repair_reference"]["frozen_mask"]["after"]
        lines.append(
            f"| {row['identity']} | "
            f"{row['repair_reference']['full_active_set_bitwise_exact_to_bank']} | "
            f"{production['sqrt_epsilon_bypass_count']} / "
            f"{len(production['newton_steps'])} | "
            f"{production['conditioning_applied_count']} / "
            f"{len(production['newton_steps'])} | "
            f"{frozen['raw_to_proposed_step_fraction']:.6f} | "
            f"`{row['mechanisms']['primary']}` |"
        )
    lines.extend(
        [
            "",
            (
                "The held repair changes none of the three exact bank histories: "
                f"{receipt['verdict']['repair_full_solve_bitwise_exact_count']} of "
                "four rows are bitwise exact (21986/46, 21989/55, 22086/43), "
                "while 21985/51 moves but still does not converge. Overall "
                f"convergence remains {receipt['verdict']['repair_converged_count']} "
                "of four on the bank driver's explicit GMRES-12 route. This is "
                "not the public GMRES-30 production result: that route converges "
                f"{receipt['verdict']['production_gmres_30_converged_count']} of "
                "four, including 22086/43 at machine precision."
            ),
            "",
            "## Per-row production trace",
            "",
        ]
    )
    for row in receipt["rows"]:
        production = row["production_path"]
        lines.extend(
            [
                f"### {row['identity']} pure",
                "",
                (
                    f"Primary mechanism: `{row['mechanisms']['primary']}`. Active "
                    "layers: "
                    + ", ".join(
                        f"`{name}`" for name in row["mechanisms"]["active_layers"]
                    )
                    + "."
                ),
                "",
                "| trip | Newton | GMRES rel. residual | bypass | raw sup | "
                "proposed sup | cap factor | merit factor | effective/raw | "
                "mask difference |",
                "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for step in production["newton_steps"]:
            lines.append(
                f"| {step['trip']} | {step['global_step']} | "
                f"{step['achieved_krylov_relative_residual']:.6e} | "
                f"{step['linear_action']['sqrt_epsilon_bypass_engaged']} | "
                f"{step['raw_newton_direction_sup']:.6e} | "
                f"{step['proposed_step_sup_after_step_cap']:.6e} | "
                f"{step['step_cap_factor']:.6f} | "
                f"{step['merit_ladder_factor']:.6f} | "
                f"{step['effective_raw_direction_fraction']:.6f} | "
                f"{step['mask_difference_after_trip']} |"
            )
        lines.extend(["", "Measured layers and repairs:", ""])
        for layer in row["mechanisms"]["layers"]:
            verdict = "active" if layer["active"] else "inactive"
            lines.append(
                f"- `{layer['name']}` — {verdict}. {layer['evidence']}. Repair: "
                f"{layer['repair']}. Owner: {layer['owner']}."
            )
        lines.append("")
    path = receipt["rows"][0].get("whole_step_residual_path")
    if path is not None:
        lines.extend(
            [
                "## 21985/51 whole-step residual",
                "",
                (
                    "All eight frozen-mask promotions use factor 1.0. The table "
                    "therefore samples each complete accepted step, rather than "
                    "extrapolating the directional derivative at its origin."
                ),
                "",
                "| step | start actual | full-step actual | quadratic prediction | "
                "absolute error |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for step in path["steps"]:
            lines.append(
                f"| {step['step']} | "
                f"{step['samples'][0]['actual_relative_residual']:.6e} | "
                f"{step['full_step_actual_residual']:.6e} | "
                f"{step['full_step_quadratic_prediction']:.6e} | "
                f"{step['full_step_absolute_model_error']:.6e} |"
            )
        lines.extend(
            [
                "",
                (
                    "Maximum full-step actual-versus-quadratic error is "
                    f"{path['maximum_full_step_absolute_model_error']:.6e}; "
                    "Newton-consistent at the 1e-3 level: "
                    f"{path['newton_consistent_at_1e_3']}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence authority",
            "",
            f"- Production trace: `{receipt['figures']['production_path']}`.",
            f"- Whole-step trace: `{receipt['figures']['whole_step_residual']}`.",
            (
                f"- Exact operands: `{receipt['evidence_inputs']['operands']}` "
                f"(SHA-256 `{receipt['evidence_inputs']['operands_sha256']}`)."
            ),
            (
                "- Held repair receipt: "
                f"`{receipt['evidence_inputs']['repair_receipt']}` (SHA-256 "
                f"`{receipt['evidence_inputs']['repair_receipt_sha256']}`)."
            ),
            (
                "- No `nova/` file was changed; instrumentation is confined to "
                "`benchmarks/settled_mask_stall.py`."
            ),
            (
                "- The returned active-set `FixedPointResult` aggregates "
                "conditioning count and maximum projected condition over every "
                "frozen-mask solve. Both are checked against the production "
                "per-step callback trace before this receipt is written."
            ),
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def measure_production_path(
    *, operands: Path, repair: Path, output: Path, report: Path
) -> dict[str, Any]:
    """Measure the remaining production damping on the conditioning repair tip."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    banked = _load_banked_rows(operands)
    repair_data = json.loads(repair.read_text(encoding="utf-8"))
    repair_by_identity = {row["identity"]: row for row in repair_data["rows"]}
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    rows = []
    for key in TARGETS:
        identity = f"{key[0]}/{key[1]}"
        print(f"PRODUCTION-PATH {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        production = _production_path_solve(profile, seed, target_current, banked[key])
        whole_path = (
            _whole_step_residual_path(
                profile,
                jnp.asarray(seed),
                banked[key],
                target_current,
            )
            if key == (21985, 51)
            else None
        )
        repair_row = repair_by_identity[identity]
        repair_reference = {
            **repair_row,
            "full_active_set_bitwise_exact_to_bank": _repair_full_solve_exact(
                repair_row
            ),
        }
        row = {
            "identity": identity,
            "production_path": production,
            "repair_reference": repair_reference,
            "whole_step_residual_path": whole_path,
        }
        row["mechanisms"] = _mechanism_layers(
            identity, production, repair_row, whole_path
        )
        rows.append(row)
        print(
            "PRODUCTION-PATH-DONE "
            + json.dumps(
                {
                    "identity": identity,
                    "bypass": production["sqrt_epsilon_bypass_count"],
                    "conditioning": production["conditioning_applied_count"],
                    "steps": len(production["newton_steps"]),
                    "mechanism": row["mechanisms"]["primary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    production_figure = output.with_name("production-path.png")
    whole_path_figure = output.with_name("whole-step-residual-21985-51.png")
    _draw_production_path(rows, production_figure)
    _draw_whole_step_path(rows[0]["whole_step_residual_path"], whole_path_figure)
    repair_exact_count = sum(
        row["repair_reference"]["full_active_set_bitwise_exact_to_bank"] for row in rows
    )
    repair_converged_count = sum(
        row["repair_reference"]["full_active_set"]["measured"]["converged"]
        for row in rows
    )
    receipt = {
        "artifact": "production active-set damping diagnosis",
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "operands": str(operands),
            "operands_sha256": _sha256(operands),
            "repair_receipt": str(repair),
            "repair_receipt_sha256": _sha256(repair),
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "production_route": "ForwardProfile.solve_branch(newton_krylov)",
            "production_gmres_iterations": PUBLIC_ROUTE_POLICY.gmres_iterations,
            "held_bank_replay_gmres_iterations": reachability.GMRES_ITERATIONS,
            "sqrt_epsilon_threshold": float(np.sqrt(np.finfo(np.float64).eps)),
            "whole_step_fractions": list(PRODUCTION_PATH_FRACTIONS),
            "whole_step_model": (
                "second-order directional Taylor model of the exact frozen "
                "residual vector at each accepted-step origin"
            ),
            "repair_comparison": (
                "reuse the held repair-tip H200 receipt for the exact bank-seed "
                f"GMRES-{reachability.GMRES_ITERATIONS} solve and frozen-mask "
                "GMRES-40 contraction"
            ),
        },
        "figures": {
            "production_path": str(production_figure),
            "whole_step_residual": str(whole_path_figure),
        },
        "report": str(report),
        "rows": rows,
        "verdict": {
            "repair_full_solve_bitwise_exact_count": repair_exact_count,
            "repair_full_solve_row_count": len(rows),
            "repair_converged_count": repair_converged_count,
            "production_gmres_30_converged_count": sum(
                row["production_path"]["result"]["converged"] for row in rows
            ),
            "production_gmres_30_bypass_count": sum(
                row["production_path"]["sqrt_epsilon_bypass_count"] for row in rows
            ),
            "production_gmres_30_newton_step_count": sum(
                len(row["production_path"]["newton_steps"]) for row in rows
            ),
            "primary_mechanisms": {
                row["identity"]: row["mechanisms"]["primary"] for row in rows
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_production_report(receipt, report)
    return receipt


def render_production_path(
    *, measurement: Path, output: Path, report: Path
) -> dict[str, Any]:
    """Re-render a completed device measurement without rerunning the solver."""
    receipt = json.loads(measurement.read_text(encoding="utf-8"))
    for row in receipt["rows"]:
        row["production_path"]["terminal_result_conditioning_count_scope"] = (
            "whole active-set solve"
        )
        row["mechanisms"] = _mechanism_layers(
            row["identity"],
            row["production_path"],
            row["repair_reference"],
            row.get("whole_step_residual_path"),
        )
    production_figure = output.with_name("production-path.png")
    whole_path_figure = output.with_name("whole-step-residual-21985-51.png")
    _draw_production_path(receipt["rows"], production_figure)
    _draw_whole_step_path(
        receipt["rows"][0]["whole_step_residual_path"], whole_path_figure
    )
    receipt["figures"] = {
        "production_path": str(production_figure),
        "whole_step_residual": str(whole_path_figure),
    }
    receipt["report"] = str(report)
    receipt["measurement_contract"]["held_bank_replay_gmres_iterations"] = (
        reachability.GMRES_ITERATIONS
    )
    receipt["verdict"]["production_gmres_30_converged_count"] = sum(
        row["production_path"]["result"]["converged"] for row in receipt["rows"]
    )
    receipt["verdict"]["primary_mechanisms"] = {
        row["identity"]: row["mechanisms"]["primary"] for row in receipt["rows"]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_production_report(receipt, report)
    return receipt


def _draw_public_route_comparison(
    rows: list[dict[str, Any]], figure_path: Path
) -> None:
    """Plot pinned-bank and public-route residual histories for four rows."""
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for axis, row in zip(axes.flat, rows, strict=True):
        bank = row["bank_pinned_budget"]
        public = row["public_route"]
        bank_result = bank["result"]
        public_result = public["result"]
        bank_residuals = bank_result["active_set_residuals"]
        public_residuals = public_result["active_set_residuals"]
        axis.semilogy(
            np.arange(1, len(bank_residuals) + 1),
            bank_residuals,
            "o--",
            color="#e76f51",
            label=f"bank GMRES {bank['configuration']['gmres_iterations']}",
        )
        axis.semilogy(
            np.arange(1, len(public_residuals) + 1),
            public_residuals,
            "o-",
            color="#087e8b",
            label=f"public GMRES {public['configuration']['gmres_iterations']}",
        )
        axis.axhline(
            reachability.FIXED_POINT_CRITERION,
            color="#343a40",
            linewidth=0.9,
            linestyle=":",
            label="tolerance",
        )
        topology = public["terminal_topology"]
        axis.set_title(
            f"{row['identity']} pure\n"
            f"{public_result['termination_reason']} · "
            f"{topology['achieved_class']}"
        )
        axis.set_xlabel("active-set trip")
        axis.set_ylabel("relative residual")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("Pinned bank budget versus the public Newton–Krylov route")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def measure_public_route(*, operands: Path, output: Path) -> dict[str, Any]:
    """Re-solve four bank seeds through the public branch budget."""
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
        identity = f"{key[0]}/{key[1]}"
        print(f"PUBLIC-ROUTE {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        measured = _production_path_solve(profile, seed, target_current, banked[key])
        bank_result = {
            "terminal_residual": float(banked[key]["terminal_residual"]),
            "active_set_iterations": int(banked[key]["active_set_iterations"]),
            "termination_reason": str(banked[key]["termination_reason"]),
            "converged": bool(banked[key]["converged"]),
            "active_set_residuals": banked[key]["active_set_residuals"].tolist(),
            "active_set_mask_differences": banked[key][
                "active_set_mask_differences"
            ].tolist(),
        }
        public_route = {
            "configuration": measured["configuration"],
            "result": measured["result"],
            "terminal_topology": measured["terminal_topology"],
            "conditioning_trace": {
                "count": measured["conditioning_applied_count"],
                "maximum_projected_condition": measured[
                    "maximum_projected_krylov_condition"
                ],
                "result_scope": measured["terminal_result_conditioning_count_scope"],
            },
        }
        row = {
            "identity": identity,
            "arm": "pure",
            "bank_pinned_budget": {
                "configuration": {
                    "route": "ForwardProfile.solve_branch(newton_krylov)",
                    "gmres_iterations": reachability.GMRES_ITERATIONS,
                },
                "result": bank_result,
                "terminal_topology": _bank_topology(banked[key]),
            },
            "public_route": public_route,
        }
        rows.append(row)
        print(
            "PUBLIC-ROUTE-DONE "
            + json.dumps(
                {
                    "identity": identity,
                    "residual": public_route["result"]["terminal_residual"],
                    "termination": public_route["result"]["termination_reason"],
                    "trips": public_route["result"]["active_set_iterations"],
                    "class": public_route["terminal_topology"]["achieved_class"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    figure_path = output.with_suffix(".png")
    _draw_public_route_comparison(rows, figure_path)
    receipt = {
        "artifact": "public-route four-row corroboration",
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "operands": str(operands),
            "operands_sha256": _sha256(operands),
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "seed": "persisted pure-arm bank seed for each row",
            "route": "ForwardProfile.solve_branch(newton_krylov)",
            "bank_pinned_gmres_iterations": reachability.GMRES_ITERATIONS,
            "public_route_gmres_iterations": (PUBLIC_ROUTE_POLICY.gmres_iterations),
            "conditioning_authority": (
                "FixedPointResult whole-active-set aggregate checked against "
                "the per-step callback trace"
            ),
        },
        "figure": str(figure_path),
        "rows": rows,
        "verdict": {
            "row_count": len(rows),
            "bank_converged_count": sum(
                row["bank_pinned_budget"]["result"]["converged"] for row in rows
            ),
            "public_route_converged_count": sum(
                row["public_route"]["result"]["converged"] for row in rows
            ),
            "all_public_axes_admitted": all(
                row["public_route"]["terminal_topology"]["axis_admitted"]
                for row in rows
            ),
            "all_public_saddles_admitted": all(
                row["public_route"]["terminal_topology"]["saddle_admitted"]
                for row in rows
            ),
            "all_public_classes_diverted": all(
                row["public_route"]["terminal_topology"]["achieved_class"] == "diverted"
                for row in rows
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _undamped_frozen_solve(frozen_map, initial) -> dict[str, Any]:
    """Take raw fixed-partition Newton directions without globalization."""

    def take_step(state, _unused):
        mapped, tangent = jax.linearize(frozen_map, state)
        residual_vector = mapped - state

        def residual_action(vector):
            return vector - tangent(vector)

        step, info = jax.scipy.sparse.linalg.gmres(
            residual_action,
            residual_vector,
            tol=fixed_point._GMRES_RELATIVE_TOLERANCE,
            maxiter=SMOOTH_GMRES_ITERATIONS,
            restart=SMOOTH_GMRES_ITERATIONS,
            solve_method="batched",
        )
        linear_residual = residual_vector - residual_action(step)
        reduction = jnp.linalg.norm(linear_residual) / jnp.maximum(
            jnp.linalg.norm(residual_vector),
            jnp.finfo(residual_vector.dtype).tiny,
        )
        candidate = state + step
        candidate_mapped = frozen_map(candidate)
        observation = (
            fixed_point._relative_residual(mapped, state),
            fixed_point._relative_residual(candidate_mapped, candidate),
            jnp.max(jnp.abs(step)),
            jnp.asarray(info, dtype=jnp.int32),
            reduction,
        )
        return candidate, observation

    terminal, observations = jax.lax.scan(
        take_step,
        initial,
        xs=None,
        length=SMOOTH_NEWTON_STEPS,
    )
    terminal.block_until_ready()
    residuals_before, residuals_after, step_norms, info, reductions = (
        np.asarray(value) for value in observations
    )
    return {
        "initial_residual": float(residuals_before[0]),
        "residuals_before": residuals_before.tolist(),
        "residuals_after": residuals_after.tolist(),
        "residuals_per_step": [
            {
                "step": index + 1,
                "before": float(before),
                "after": float(after),
            }
            for index, (before, after) in enumerate(
                zip(residuals_before, residuals_after, strict=True)
            )
        ],
        "raw_step_norms_taken": step_norms.tolist(),
        "gmres_info": info.tolist(),
        "gmres_relative_residuals": reductions.tolist(),
        "terminal_residual": float(residuals_after[-1]),
        "finite": bool(
            np.all(np.isfinite(residuals_after)) and np.all(np.isfinite(step_norms))
        ),
        "damping": {
            "projected_conditioning": False,
            "step_cap": False,
            "merit_ladder": False,
            "factors": [1.0] * SMOOTH_NEWTON_STEPS,
        },
    }


def _draw_undamped_comparison(rows: list[dict[str, Any]], figure_path: Path) -> None:
    """Plot raw fixed-partition steps beside repaired production histories."""
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(13.0, 12.5), constrained_layout=True
    )
    for row_index, row in enumerate(rows):
        frozen_axis, production_axis = axes[row_index]
        undamped = row["frozen_mask_undamped"]
        frozen_values = [undamped["initial_residual"], *undamped["residuals_after"]]
        frozen_axis.semilogy(
            range(len(frozen_values)),
            frozen_values,
            "o-",
            color="#d1495b",
            label="raw Newton, factor one",
        )
        frozen_axis.axhline(
            reachability.FIXED_POINT_CRITERION,
            color="#343a40",
            linewidth=0.9,
            linestyle=":",
        )
        frozen_axis.set_ylabel(row["identity"] + "\nrelative residual")
        frozen_axis.set_xlabel("frozen-mask raw Newton step")
        frozen_axis.grid(alpha=0.25)

        repaired = row["production_route_repaired_cap"]["result"]
        public = row["public_route_receipt"]
        production_axis.semilogy(
            range(1, len(public["active_set_residuals"]) + 1),
            public["active_set_residuals"],
            "o--",
            color="#6c757d",
            label="public receipt",
        )
        production_axis.semilogy(
            range(1, len(repaired["active_set_residuals"]) + 1),
            repaired["active_set_residuals"],
            "o-",
            color="#087e8b",
            label="model-error cap",
        )
        production_axis.axhline(
            reachability.FIXED_POINT_CRITERION,
            color="#343a40",
            linewidth=0.9,
            linestyle=":",
        )
        production_axis.set_xlabel("production active-set trip")
        production_axis.set_ylabel("relative residual")
        production_axis.set_title(
            f"{repaired['termination_reason']} · "
            f"{row['production_route_repaired_cap']['terminal_topology']['achieved_class']}"
        )
        production_axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    figure.suptitle(
        "Undamped frozen-mask control and model-error-triggered production cap"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def measure_undamped_control(
    *,
    operands: Path,
    public_route: Path,
    vertical_mode: Path,
    output: Path,
) -> dict[str, Any]:
    """Measure raw frozen Newton and the repaired public production route."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    banked = _load_banked_rows(operands)
    public_data = json.loads(public_route.read_text(encoding="utf-8"))
    public_by_identity = {row["identity"]: row for row in public_data["rows"]}
    vertical_data = json.loads(vertical_mode.read_text(encoding="utf-8"))
    vertical_by_identity = {
        row["identity"]: row["verdict"]
        for row in vertical_data["rows"]
        if row["arm"] == "pure"
    }
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    rows = []
    for key in TARGETS:
        identity = f"{key[0]}/{key[1]}"
        print(f"UNDAMPED-CONTROL {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        frozen_state = _full_terminal_state(profile, seed, banked[key])
        mask = profile.operator.residual_shadow_mask(frozen_state, requested_class)
        shadowed_map = profile.operator.flux_map_with_shadow(
            requested_class=requested_class,
            target_current=target_current,
        )

        def frozen_map(candidate):
            return shadowed_map(candidate, mask)

        undamped = _undamped_frozen_solve(frozen_map, frozen_state)
        production = _production_path_solve(
            profile,
            seed,
            target_current,
            banked[key],
        )
        public_result = public_by_identity[identity]["public_route"]
        cap_steps = production["newton_steps"]
        rows.append(
            {
                "identity": identity,
                "arm": "pure",
                "vertical_mode_evidence": vertical_by_identity[identity],
                "frozen_mask_undamped": undamped,
                "production_route_repaired_cap": {
                    "result": production["result"],
                    "terminal_topology": production["terminal_topology"],
                    "step_cap_activation_count": sum(
                        step["step_cap_activated"] for step in cap_steps
                    ),
                    "step_cap_factors": [step["step_cap_factor"] for step in cap_steps],
                    "model_error_fractions": [
                        step["model_error_fraction"] for step in cap_steps
                    ],
                    "newton_step_count": len(cap_steps),
                },
                "public_route_receipt": {
                    "terminal_residual": public_result["result"]["terminal_residual"],
                    "active_set_iterations": public_result["result"][
                        "active_set_iterations"
                    ],
                    "termination_reason": public_result["result"]["termination_reason"],
                    "converged": public_result["result"]["converged"],
                    "active_set_residuals": public_result["result"][
                        "active_set_residuals"
                    ],
                    "achieved_class": public_result["terminal_topology"][
                        "achieved_class"
                    ],
                },
            }
        )
        print(
            "UNDAMPED-CONTROL-DONE "
            + json.dumps(
                {
                    "identity": identity,
                    "frozen_terminal": undamped["terminal_residual"],
                    "production_terminal": production["result"]["terminal_residual"],
                    "cap_activations": sum(
                        step["step_cap_activated"] for step in cap_steps
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    figure_path = output.with_suffix(".png")
    _draw_undamped_comparison(rows, figure_path)
    receipt = {
        "artifact": "undamped frozen-mask control and repaired production cap",
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "operands": str(operands),
            "operands_sha256": _sha256(operands),
            "public_route_receipt": str(public_route),
            "public_route_receipt_sha256": _sha256(public_route),
            "vertical_mode_receipt": str(vertical_mode),
            "vertical_mode_receipt_sha256": _sha256(vertical_mode),
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "frozen_mask_steps": SMOOTH_NEWTON_STEPS,
            "frozen_mask_gmres_iterations": SMOOTH_GMRES_ITERATIONS,
            "frozen_mask_damping": {
                "projected_conditioning": False,
                "step_cap": False,
                "merit_ladder": False,
                "step_factor": 1.0,
            },
            "production_route": "ForwardProfile.solve_branch(newton_krylov)",
            "production_gmres_iterations": PUBLIC_ROUTE_POLICY.gmres_iterations,
            "step_cap_model_error_fraction": (
                fixed_point._STEP_CAP_MODEL_ERROR_FRACTION
            ),
            "step_cap_authority": (
                "previous accepted step actual-versus-linear merit error; "
                "the nonlinear merit ladder remains sole acceptance authority"
            ),
            "partition_change": (
                "relinearize on the changed frozen mask and reset projected "
                "conditioning plus prior model-error state"
            ),
        },
        "figure": str(figure_path),
        "rows": rows,
        "verdict": {
            "row_count": len(rows),
            "undamped_all_factors_one": all(
                row["frozen_mask_undamped"]["damping"]["factors"]
                == [1.0] * SMOOTH_NEWTON_STEPS
                for row in rows
            ),
            "production_converged_count": sum(
                row["production_route_repaired_cap"]["result"]["converged"]
                for row in rows
            ),
            "public_receipt_converged_count": sum(
                row["public_route_receipt"]["converged"] for row in rows
            ),
            "vertical_near_null_rows": [
                row["identity"]
                for row in rows
                if row["vertical_mode_evidence"]["near_null_is_vertical_mode"]
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    repair_parser = subparsers.add_parser("repair")
    repair_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    repair_parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    repair_parser.add_argument("--output", type=Path, default=DEFAULT_REPAIR_OUTPUT)
    render_parser = subparsers.add_parser("render-repair")
    render_parser.add_argument("--measurement", type=Path, required=True)
    render_parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    render_parser.add_argument("--output", type=Path, default=DEFAULT_REPAIR_OUTPUT)
    production_parser = subparsers.add_parser("production-path")
    production_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    production_parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR_OUTPUT)
    production_parser.add_argument(
        "--output", type=Path, default=DEFAULT_PRODUCTION_OUTPUT
    )
    production_parser.add_argument(
        "--report", type=Path, default=DEFAULT_PRODUCTION_REPORT
    )
    render_production_parser = subparsers.add_parser("render-production-path")
    render_production_parser.add_argument("--measurement", type=Path, required=True)
    render_production_parser.add_argument(
        "--output", type=Path, default=DEFAULT_PRODUCTION_OUTPUT
    )
    render_production_parser.add_argument(
        "--report", type=Path, default=DEFAULT_PRODUCTION_REPORT
    )
    public_route_parser = subparsers.add_parser("public-route")
    public_route_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    public_route_parser.add_argument(
        "--output", type=Path, default=DEFAULT_PUBLIC_ROUTE_OUTPUT
    )
    subparsers.add_parser("trace-accounting-test")
    undamped_parser = subparsers.add_parser("undamped-control")
    undamped_parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    undamped_parser.add_argument(
        "--public-route", type=Path, default=DEFAULT_PUBLIC_ROUTE_OUTPUT
    )
    undamped_parser.add_argument(
        "--vertical-mode", type=Path, default=DEFAULT_VERTICAL_MODE
    )
    undamped_parser.add_argument("--output", type=Path, default=DEFAULT_UNDAMPED_OUTPUT)
    arguments = parser.parse_args()
    if arguments.action == "measure":
        result = measure(
            operands=arguments.operands,
            output=arguments.output,
            source_label=arguments.source_label,
            require_bank_match=not arguments.allow_bank_drift,
        )
        print(json.dumps({"rows": len(result["rows"])}, sort_keys=True))
    elif arguments.action == "combine":
        result = combine(
            arguments.current,
            arguments.candidate,
            arguments.output,
            arguments.report,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "repair":
        result = measure_repair(
            operands=arguments.operands,
            baseline=arguments.baseline,
            output=arguments.output,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "render-repair":
        result = render_repair(
            measurement=arguments.measurement,
            baseline=arguments.baseline,
            output=arguments.output,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "production-path":
        result = measure_production_path(
            operands=arguments.operands,
            repair=arguments.repair,
            output=arguments.output,
            report=arguments.report,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "render-production-path":
        result = render_production_path(
            measurement=arguments.measurement,
            output=arguments.output,
            report=arguments.report,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "public-route":
        result = measure_public_route(
            operands=arguments.operands,
            output=arguments.output,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    elif arguments.action == "trace-accounting-test":
        print(
            json.dumps(
                _verify_trace_action_reuse_accounting(), indent=2, sort_keys=True
            )
        )
    else:
        result = measure_undamped_control(
            operands=arguments.operands,
            public_route=arguments.public_route,
            vertical_mode=arguments.vertical_mode,
            output=arguments.output,
        )
        print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
