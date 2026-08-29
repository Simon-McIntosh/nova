"""Fixed-point accelerators for the free-boundary equilibrium maps.

Solve ``x = g(x)`` for a fixed-shape differentiable map ``g`` — the
free-boundary flux map (``ForwardFluxOperator.__call__``, whose root of
``residual`` is the equilibrium) or the reconstruction sweep map
(``ReconstructProfile.least_squares_map``).  Three schemes share one result
contract and one per-map-evaluation residual accounting so budget ladders
compare fairly:

* :func:`picard` — relaxed fixed-point iteration,
  ``x ← x + β (g(x) − x)``;
* :func:`anderson` — safeguarded Anderson mixing (Walker–Ni type-II) of the
  relaxed iteration, in-graph and fixed-shape: the last ``depth``
  residual/iterate differences take a ridge-regularised least-squares step,
  guarded by a warmup delay, a history restart on residual growth, a step cap
  against the plain relaxed step, and a non-finite fallback — none of which
  costs an extra map evaluation;
* :func:`newton_krylov` — exact-tangent Jacobian-free Newton–Krylov: each
  step linearises the map ONCE (``jax.linearize`` — exact tangents, no finite
  differences) and solves ``(I − J) s = f`` with a fixed-shape batched GMRES
  before its step can be promoted.  The bounded outer loop stops at the first
  finite relative residual at or below the registered criterion while its
  diagnostic arrays retain their configured padded shapes.

All three are ``jit``-safe and ``vmap``-safe; map a leading batch axis with
``jax.vmap`` over the initial state and any batched map parameters.  Batched
Newton solves execute until their slowest active lane terminates while
preserving each lane's own attempted and accepted promotion counts.  The
free-boundary maps have multiple fixed points (the outboard-corner attractor),
so the BASIN is guarded by the caller seeding the topology read at the current
centroid (the axis-seed pin); the accelerators contribute step caps, never
basin logic.

The state is a flat 1-D array — the concatenated flux vector both solve maps
already carry.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from nova.jax.config import Precision, resolve_precision
from nova.equilibrium.manifold_advance import (
    ManifoldAdvanceQualification,
    normal_component,
    oriented_secant,
)

__all__ = [
    "AmplificationObservation",
    "FIXED_POINT_RESIDUAL_TOLERANCE",
    "FixedPointResult",
    "FixedPointTerminationReason",
    "KinkAwareResult",
    "KrylovActionQualification",
    "ManifoldAdvanceQualification",
    "RecoveryOutcome",
    "anderson",
    "kink_aware_newton_krylov",
    "newton_krylov",
    "picard",
]


_BACKTRACKING_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
_RECORDED_BACKTRACKING_FACTOR_COUNT = 4
_RECOVERY_RADIUS_INITIAL = 1.0
_RECOVERY_RADIUS_GROWTH = 2.0
_RECOVERY_RADIUS_SHRINKAGE = 0.5
_RECOVERY_RADIUS_UPDATE_TRIPS = 6
_MODEL_REBUILD_DAMPING_INITIAL = 1.0e-3
_MODEL_REBUILD_DAMPING_GROWTH = 10.0
_MODEL_REBUILD_DAMPING_TRIPS = 6
_STEEPEST_DESCENT_SCALES = (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
_RELATIVE_SUP_MERIT_EXPONENT = 8
_NONMONOTONE_MERIT_WINDOW = 6
_ACTIVE_SET_ITERATION_LIMIT = 16
_ACTIVE_SET_CYCLE_DAMPING = 0.5
_PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT = math.e
_SUFFICIENT_DECREASE_SLOPE = 1.0e-4
FIXED_POINT_RESIDUAL_TOLERANCE = 1.0e-8


def _print_active_set_trip(
    active, trip_index, mask_difference, live_residual, inner_iterations
) -> None:
    """Print one device-reported active-set trip immediately on the host."""

    if not bool(active):
        return
    print(
        "active-set "
        f"trip={int(trip_index)} "
        f"mask_difference={int(mask_difference)} "
        f"live_residual={float(live_residual):.17g} "
        f"inner_iterations={int(inner_iterations)}",
        flush=True,
    )


class AmplificationObservation(IntEnum):
    """Advisory shape of the increments along a qualified solve trajectory."""

    NOT_APPLICABLE = 0
    CONTRACTING = 1
    SUSTAINED_GROWTH = 2


class KrylovActionQualification(IntEnum):
    """Host-readable reason a Newton--Krylov linear action was refused."""

    NOT_APPLICABLE = 0
    ACCEPTED = 1
    NONFINITE_LINEAR_ACTION = 2
    NONSUCCESSFUL_GMRES_STATUS = 3
    NONFINITE_ACHIEVED_LINEAR_RESIDUAL = 4
    ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL = 5


class RecoveryOutcome(IntEnum):
    """Host-readable outcome of an exhausted Newton promotion's recovery."""

    NOT_APPLICABLE = 0
    ACCEPTED = 1
    REFUSED = 2


class FixedPointTerminationReason(IntEnum):
    """Host-readable reason a bounded Newton solve stopped."""

    NOT_APPLICABLE = 0
    CONVERGED = 1
    ITERATION_BUDGET_EXHAUSTED = 2
    NONFINITE_RESIDUAL = 3
    KRYLOV_ACTION_REFUSED = 4
    MANIFOLD_ADVANCE_REFUSED = 5
    SUFFICIENT_DECREASE_REFUSED = 6
    ACTIVE_SET_CYCLE_DETECTED = 7
    ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED = 8
    ACTIVE_SET_STAGNATED = 9


class FixedPointResult(NamedTuple):
    """Fixed-shape result of a fixed-point solve.

    ``trace`` holds one entry per map evaluation: the relative sup-norm
    residual where the scheme measured one, NaN where the evaluation was a
    Newton tangent pass — so ladder plots of different schemes share one
    x-axis.  Exact-tangent Newton returns the best measured ``state`` and
    ``residual`` in that relative sup metric; the other schemes retain their
    terminal measured iterate.
    ``krylov_action_qualification`` names the first refused linear-action
    condition, reports ``ACCEPTED`` when every Newton action passed, and is
    ``NOT_APPLICABLE`` for the non-Krylov schemes.
    ``amplification_observation`` independently reports whether two consecutive
    increment ratios exceeded one before a later increment contracted.  It is
    advisory: it is derived after promotion and never participates in step
    qualification or candidate admission.
    ``krylov_conditioning_count`` reports how many accepted linear actions were
    damped by the iteration-local projected Krylov condition discriminator.
    ``maximum_projected_krylov_condition`` reports the largest discriminator
    value encountered by a Krylov route and is NaN for non-Krylov schemes.
    When a caller supplies an admissibility predicate and a preceding admitted
    state, the trailing arrays record the predictor-corrector attempts in
    state-space arclength.  ``newton_step_equivalents`` is the sum of promoted
    advance length divided by the corresponding qualified Newton-step length.
    ``attempted_newton_promotions`` counts linearised Newton proposals;
    ``accepted_newton_promotions`` counts proposals promoted to ``state``.
    ``converged`` and ``termination_reason`` report why the bounded Newton loop
    stopped.  They retain not-applicable defaults for non-Newton schemes.
    ``promotion_backtrack_counts`` records how many fixed half-step reductions
    preceded each accepted promotion, the ladder length for a rejected
    promotion, and -1 in unused slots.
    ``promotion_recovery_activations`` is one when the damped-continuation
    fallback was evaluated after backtracking exhaustion, zero when the Newton
    ladder decided the promotion, and -1 in unused slots.
    ``promotion_recovery_radii`` records the carried radius before and after
    each activated recovery.  A refused recovery shrinks its radius for the
    next promotion; an accepted recovery grows from the accepted radius.
    ``promotion_recovery_outcomes`` names that result and is not-applicable for
    promotions decided by the Newton ladder.
    ``promotion_model_rebuild_activations`` is one when radius exhaustion
    triggers a fresh linearization, and ``promotion_model_rebuild_damping``
    records its accepted Levenberg damping or the last refused value.
    ``promotion_descent_activations`` is one when both recovery models exhaust
    and the smooth relative-sup gradient ladder runs.  Its selected absolute step
    scale is recorded in ``promotion_descent_scales``, with zero for a refused
    ladder and NaN where the fallback was not evaluated.
    ``active_set_iterations`` counts frozen-mask inner solves.  Their live
    residuals and re-evaluated mask differences are recorded in the two
    corresponding fixed-length arrays.  ``active_set_cycle_damping_activations``
    marks the midpoint transition attempted when a mask repeats.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array
    krylov_action_qualification: jax.Array | int = (
        KrylovActionQualification.NOT_APPLICABLE
    )
    amplification_observation: jax.Array | int = AmplificationObservation.NOT_APPLICABLE
    krylov_conditioning_count: jax.Array | int = 0
    maximum_projected_krylov_condition: jax.Array | float = float("nan")
    manifold_advance_qualification: jax.Array | int = (
        ManifoldAdvanceQualification.NOT_APPLICABLE
    )
    manifold_admissibility: jax.Array | bool = False
    predictor_lengths: jax.Array | float = float("nan")
    corrector_lengths: jax.Array | float = float("nan")
    advance_lengths: jax.Array | float = float("nan")
    newton_step_lengths: jax.Array | float = float("nan")
    newton_step_equivalents: jax.Array | float = float("nan")
    attempted_newton_promotions: jax.Array | int = 0
    accepted_newton_promotions: jax.Array | int = 0
    converged: jax.Array | bool = False
    termination_reason: jax.Array | int = FixedPointTerminationReason.NOT_APPLICABLE
    shadow_mask_changes: jax.Array | int = -1
    promotion_backtrack_counts: jax.Array | int = -1
    promotion_recovery_activations: jax.Array | int = -1
    promotion_recovery_radii: jax.Array | float = float("nan")
    promotion_recovery_outcomes: jax.Array | int = -1
    promotion_model_rebuild_activations: jax.Array | int = -1
    promotion_model_rebuild_damping: jax.Array | float = float("nan")
    promotion_descent_activations: jax.Array | int = -1
    promotion_descent_scales: jax.Array | float = float("nan")
    active_set_iterations: jax.Array | int = 0
    active_set_residuals: jax.Array | float = float("nan")
    active_set_mask_differences: jax.Array | int = -1
    active_set_cycle_damping_activations: jax.Array | int = -1


class KinkAwareResult(NamedTuple):
    """Result of an explicitly selected derivative-hand-off policy.

    ``trace`` records the residual at each relaxed warmup state and each
    accepted nonlinear state.  ``crossings`` identifies nonlinear moves whose
    unconstrained Newton proposal straddled the caller's detected surface.
    ``candidate_admissibility`` preserves the established diagnostic contract
    for the four largest nonmonotone factors.  ``accepted_factors`` records the
    selected factor from the complete fixed ladder and is zero when no trial
    was selected.
    ``krylov_action_qualification`` retains the first refused linear action.
    ``amplification_observation`` is the same independent advisory trajectory
    observation carried by :class:`FixedPointResult`.
    ``krylov_conditioning_count`` and ``maximum_projected_krylov_condition``
    carry the iteration-local Krylov conditioning receipt.  On a caller-
    qualified nonmonotone route, conditioning is a fallback after the raw
    ladder has no admissible trial and is promoted only when it does not raise
    the current residual.  ``effective_newton_fractions`` records each promoted
    step length relative to its undamped, uncapped Newton step.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array
    crossings: jax.Array
    candidate_admissibility: jax.Array
    accepted_factors: jax.Array
    krylov_action_qualification: jax.Array | int
    amplification_observation: jax.Array | int = AmplificationObservation.NOT_APPLICABLE
    krylov_conditioning_count: jax.Array | int = 0
    maximum_projected_krylov_condition: jax.Array | float = float("nan")
    effective_newton_fractions: jax.Array | float = float("nan")


class _QualifiedKrylovStep(NamedTuple):
    """One linear solve and its fail-closed qualification."""

    step: jax.Array
    qualification: jax.Array
    projected_condition: jax.Array
    conditioning_applied: jax.Array
    condition_baseline: jax.Array
    unconditioned_step: jax.Array


class _AmplificationState(NamedTuple):
    """Fixed-shape state for the advisory increment-growth observation."""

    previous_increment: jax.Array
    consecutive_growth_ratios: jax.Array
    sustained_growth: jax.Array
    contracted_after_sustained_growth: jax.Array


class _NewtonIterationState(NamedTuple):
    """Carry for one bounded exact-tangent Newton solve."""

    state: jax.Array
    residual: jax.Array
    best_state: jax.Array
    best_residual: jax.Array
    recent_merits: jax.Array
    merit_observations: jax.Array
    trace: jax.Array
    qualification: jax.Array
    amplification: _AmplificationState
    conditioning_count: jax.Array
    maximum_condition: jax.Array
    condition_baseline: jax.Array
    attempted: jax.Array
    accepted: jax.Array
    converged: jax.Array
    termination_reason: jax.Array
    active: jax.Array
    current_measured: jax.Array
    shadow_mask: jax.Array
    shadow_mask_changes: jax.Array
    promotion_backtrack_counts: jax.Array
    promotion_recovery_activations: jax.Array
    recovery_radius: jax.Array
    promotion_recovery_radii: jax.Array
    promotion_recovery_outcomes: jax.Array
    promotion_model_rebuild_activations: jax.Array
    promotion_model_rebuild_damping: jax.Array
    promotion_descent_activations: jax.Array
    promotion_descent_scales: jax.Array


class _ActiveSetIterationState(NamedTuple):
    """Carry for bounded frozen-mask solves and live-mask reconciliation."""

    state: jax.Array
    mask: jax.Array
    mask_history: jax.Array
    mask_history_count: jax.Array
    result: FixedPointResult
    live_residual: jax.Array
    residuals: jax.Array
    mask_differences: jax.Array
    cycle_damping_activations: jax.Array
    iterations: jax.Array
    attempted_promotions: jax.Array
    accepted_promotions: jax.Array
    active: jax.Array
    converged: jax.Array
    cycle_detected: jax.Array
    nonfinite: jax.Array
    stagnated: jax.Array


class _BacktrackedPromotion(NamedTuple):
    """One fixed-ladder nonlinear promotion and its sufficient-decrease receipt."""

    state: jax.Array
    residual: jax.Array
    accepted: jax.Array
    backtrack_count: jax.Array
    recovery_activated: jax.Array
    recovery_radius: jax.Array
    recovery_radius_before: jax.Array
    recovery_outcome: jax.Array


class _RebuiltModelPromotion(NamedTuple):
    """One fixed-trip Levenberg search on a freshly linearized model."""

    state: jax.Array
    residual: jax.Array
    accepted: jax.Array
    damping: jax.Array


class _SteepestDescentPromotion(NamedTuple):
    """One fixed-scale search along the smooth relative-sup gradient."""

    state: jax.Array
    residual: jax.Array
    accepted: jax.Array
    scale: jax.Array


class _ManifoldNewtonIterationState(NamedTuple):
    """Carry for one bounded predictor-corrector Newton solve."""

    state: jax.Array
    previous: jax.Array
    tangent_orientation: jax.Array
    residual: jax.Array
    trace: jax.Array
    qualification: jax.Array
    amplification: _AmplificationState
    conditioning_count: jax.Array
    maximum_condition: jax.Array
    condition_baseline: jax.Array
    advance_qualifications: jax.Array
    manifold_admissibility: jax.Array
    predictor_lengths: jax.Array
    corrector_lengths: jax.Array
    advance_lengths: jax.Array
    newton_step_lengths: jax.Array
    attempted: jax.Array
    accepted: jax.Array
    converged: jax.Array
    termination_reason: jax.Array
    active: jax.Array
    current_measured: jax.Array


def _initial_amplification_state(dtype) -> _AmplificationState:
    """Return an empty advisory observation accumulator."""
    return _AmplificationState(
        previous_increment=jnp.asarray(jnp.nan, dtype=dtype),
        consecutive_growth_ratios=jnp.asarray(0, dtype=jnp.int32),
        sustained_growth=jnp.asarray(False),
        contracted_after_sustained_growth=jnp.asarray(False),
    )


def _observe_increment(
    observation: _AmplificationState,
    state: jax.Array,
    candidate: jax.Array,
    promoted: jax.Array | bool,
) -> _AmplificationState:
    """Record one promoted increment without influencing its promotion."""
    increment = jnp.max(jnp.abs(candidate - state))
    has_previous = jnp.isfinite(observation.previous_increment)
    grew = has_previous & (increment > observation.previous_increment)
    contracted = has_previous & (increment < observation.previous_increment)
    consecutive = jnp.where(grew, observation.consecutive_growth_ratios + 1, 0)
    sustained = observation.sustained_growth | (consecutive >= 2)
    contracted_after_growth = observation.contracted_after_sustained_growth | (
        observation.sustained_growth & contracted
    )
    updated = _AmplificationState(
        previous_increment=increment,
        consecutive_growth_ratios=consecutive,
        sustained_growth=sustained,
        contracted_after_sustained_growth=contracted_after_growth,
    )
    return jax.tree.map(
        lambda new, old: jnp.where(promoted, new, old), updated, observation
    )


def _amplification_result(
    observation: _AmplificationState, qualification: jax.Array
) -> jax.Array:
    """Classify a qualified trajectory without turning the signal into a gate."""
    observed = jnp.where(
        observation.contracted_after_sustained_growth,
        AmplificationObservation.SUSTAINED_GROWTH,
        AmplificationObservation.CONTRACTING,
    )
    return jnp.where(
        qualification == KrylovActionQualification.ACCEPTED,
        observed,
        AmplificationObservation.NOT_APPLICABLE,
    )


def _solver_state(initial: jax.Array, precision: Precision | str) -> jax.Array:
    """Cast the state before tracing under the general-solver policy."""
    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    return jnp.asarray(initial, dtype=dtype)


def _projected_krylov_condition(
    linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    *,
    krylov_dimension: int,
) -> tuple[jax.Array, jax.Array]:
    """Measure the rectangular Arnoldi projection condition at one iteration.

    The fixed-shape, twice-orthogonalised Arnoldi construction matches the
    event-local discriminator used to separate amplification bursts.  Early
    Krylov breakdown is retained as a shorter active projection without
    introducing data-dependent array shapes or control flow.
    """
    residual_norm = jnp.linalg.norm(residual_vector)
    fallback = jnp.ones_like(residual_vector) / jnp.sqrt(residual_vector.size)
    first_vector = jnp.where(
        residual_norm > 0.0,
        residual_vector / jnp.maximum(residual_norm, 1.0e-300),
        fallback,
    )
    basis = jnp.zeros(
        (krylov_dimension + 1, residual_vector.size), dtype=residual_vector.dtype
    )
    basis = basis.at[0].set(first_vector)
    hessenberg = jnp.zeros(
        (krylov_dimension + 1, krylov_dimension), dtype=residual_vector.dtype
    )
    basis_valid = (
        jnp.zeros(krylov_dimension + 1, dtype=jnp.bool_).at[0].set(residual_norm > 0.0)
    )
    basis_index = jnp.arange(krylov_dimension + 1)
    breakdown_floor = 64.0 * jnp.finfo(residual_vector.dtype).eps

    def arnoldi_column(column, carry):
        basis, hessenberg, basis_valid = carry
        column_valid = basis_valid[column]
        action = linear_action(basis[column])
        active_rows = (basis_index <= column) & basis_valid

        coefficients = jnp.where(active_rows, basis @ action, 0.0)
        action = action - coefficients @ basis
        corrections = jnp.where(active_rows, basis @ action, 0.0)
        action = action - corrections @ basis
        coefficients = coefficients + corrections

        next_norm = jnp.linalg.norm(action)
        next_valid = (
            column_valid & jnp.isfinite(next_norm) & (next_norm > breakdown_floor)
        )
        next_vector = jnp.where(
            next_valid,
            action / jnp.maximum(next_norm, 1.0e-300),
            jnp.zeros_like(action),
        )
        hessenberg = hessenberg.at[:, column].set(
            jnp.where(column_valid, coefficients, 0.0)
        )
        hessenberg = hessenberg.at[column + 1, column].set(
            jnp.where(column_valid, next_norm, 0.0)
        )
        basis = basis.at[column + 1].set(next_vector)
        basis_valid = basis_valid.at[column + 1].set(next_valid)
        return basis, hessenberg, basis_valid

    _basis, hessenberg, basis_valid = jax.lax.fori_loop(
        0,
        krylov_dimension,
        arnoldi_column,
        (basis, hessenberg, basis_valid),
    )
    singular_values = jnp.linalg.svd(hessenberg, compute_uv=False)
    active_columns = jnp.sum(basis_valid[:-1], dtype=jnp.int32)
    smallest_index = jnp.maximum(active_columns - 1, 0)
    largest = singular_values[0]
    smallest = singular_values[smallest_index]
    median_index = jnp.maximum((active_columns - 1) // 2, 0)
    median = singular_values[median_index]
    condition = largest / jnp.maximum(smallest, jnp.finfo(smallest.dtype).tiny)
    spectral_baseline = largest / jnp.maximum(median, jnp.finfo(median.dtype).tiny)
    return (
        jnp.where(active_columns > 0, condition, 1.0),
        jnp.where(active_columns > 0, spectral_baseline, 1.0),
    )


def _relative_residual(mapped: jax.Array, state: jax.Array) -> jax.Array:
    """Relative sup-norm fixed-point residual ``max|g−x| / max|g|``."""
    return jnp.max(jnp.abs(mapped - state)) / jnp.maximum(
        jnp.max(jnp.abs(mapped)), 1.0e-30
    )


def _smooth_relative_sup_merit(mapped: jax.Array, state: jax.Array) -> jax.Array:
    """Smooth p-norm surrogate for the relative sup fixed-point residual.

    The denominator includes the same finite floor as :func:`_relative_residual`.
    For ``n`` residual entries and even exponent ``p``, norm equivalence gives
    ``rho / (n + 1) ** (1 / p) <= merit <= rho * (n + 1) ** (1 / p)``,
    where ``rho`` is the relative sup residual.
    """
    residual_vector = jnp.ravel(mapped - state)
    mapped_vector = jnp.ravel(mapped)
    denominator_vector = jnp.concatenate(
        (mapped_vector, jnp.asarray([1.0e-30], dtype=mapped_vector.dtype))
    )
    return jnp.linalg.vector_norm(
        residual_vector, ord=_RELATIVE_SUP_MERIT_EXPONENT
    ) / jnp.linalg.vector_norm(denominator_vector, ord=_RELATIVE_SUP_MERIT_EXPONENT)


def _record_merit(
    recent_merits: jax.Array,
    observation_count: jax.Array,
    merit: jax.Array,
    record: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Append one finite accepted merit to a fixed-shape recent window."""
    should_record = record & jnp.isfinite(merit)
    index = jnp.mod(observation_count, recent_merits.size)
    recent_merits = recent_merits.at[index].set(
        jnp.where(should_record, merit, recent_merits[index])
    )
    return recent_merits, observation_count + should_record.astype(jnp.int32)


def _recent_merit_maximum(
    recent_merits: jax.Array, current_merit: jax.Array
) -> jax.Array:
    """Return the finite nonmonotone reference including the current merit."""
    return jnp.max(jnp.where(jnp.isfinite(recent_merits), recent_merits, current_merit))


def _backtracked_promotion(
    map_fn: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    step: jax.Array,
    continuation_step: jax.Array,
    current_residual: jax.Array,
    reference_merit: jax.Array,
    recovery_radius: jax.Array,
) -> _BacktrackedPromotion:
    """Select a window-decreasing Newton trial or continuation recovery."""

    factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=state.dtype)
    candidates = state[None, :] + factors[:, None] * step[None, :]

    def score(candidate):
        mapped = map_fn(candidate)
        return (
            _smooth_relative_sup_merit(mapped, candidate),
            _relative_residual(mapped, candidate),
        )

    merits, residuals = jax.lax.map(score, candidates)
    required = reference_merit * (1.0 - _SUFFICIENT_DECREASE_SLOPE * factors)
    sufficient = jnp.isfinite(merits) & jnp.isfinite(residuals) & (merits <= required)
    ladder_accepted = jnp.any(sufficient)
    ladder_selected = jnp.argmax(sufficient)

    def accept_newton_ladder(_):
        return _BacktrackedPromotion(
            state=candidates[ladder_selected],
            residual=residuals[ladder_selected],
            accepted=jnp.asarray(True),
            backtrack_count=jnp.asarray(ladder_selected, dtype=jnp.int32),
            recovery_activated=jnp.asarray(False),
            recovery_radius=recovery_radius,
            recovery_radius_before=jnp.asarray(jnp.nan, dtype=state.dtype),
            recovery_outcome=jnp.asarray(
                RecoveryOutcome.NOT_APPLICABLE, dtype=jnp.int32
            ),
        )

    def recover_with_continuation(_):
        minimum_radius = jnp.sqrt(jnp.finfo(state.dtype).eps)
        initial_radius = jnp.clip(
            recovery_radius, minimum_radius, _RECOVERY_RADIUS_INITIAL
        )

        def recovery_trial(_index, carry):
            selected_state, selected_residual, accepted, radius = carry
            candidate = state + radius * continuation_step
            candidate_merit, candidate_residual = score(candidate)
            required_merit = reference_merit * (
                1.0 - _SUFFICIENT_DECREASE_SLOPE * radius
            )
            select = (
                ~accepted
                & jnp.isfinite(candidate_merit)
                & jnp.isfinite(candidate_residual)
                & (candidate_merit <= required_merit)
            )
            return (
                jnp.where(select, candidate, selected_state),
                jnp.where(select, candidate_residual, selected_residual),
                accepted | select,
                jnp.where(
                    accepted | select,
                    radius,
                    jnp.maximum(radius * _RECOVERY_RADIUS_SHRINKAGE, minimum_radius),
                ),
            )

        candidate, candidate_residual, recovery_accepted, trial_radius = (
            jax.lax.fori_loop(
                0,
                _RECOVERY_RADIUS_UPDATE_TRIPS,
                recovery_trial,
                (
                    state,
                    current_residual,
                    jnp.asarray(False),
                    initial_radius,
                ),
            )
        )
        next_radius = jnp.where(
            recovery_accepted,
            jnp.minimum(
                trial_radius * _RECOVERY_RADIUS_GROWTH, _RECOVERY_RADIUS_INITIAL
            ),
            trial_radius,
        )
        return _BacktrackedPromotion(
            state=candidate,
            residual=candidate_residual,
            accepted=recovery_accepted,
            backtrack_count=jnp.asarray(factors.size, dtype=jnp.int32),
            recovery_activated=jnp.asarray(True),
            recovery_radius=next_radius,
            recovery_radius_before=initial_radius,
            recovery_outcome=jnp.where(
                recovery_accepted,
                RecoveryOutcome.ACCEPTED,
                RecoveryOutcome.REFUSED,
            ).astype(jnp.int32),
        )

    return jax.lax.cond(
        ladder_accepted,
        accept_newton_ladder,
        recover_with_continuation,
        operand=None,
    )


def _rebuilt_model_promotion(
    map_fn: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    current_residual: jax.Array,
    reference_merit: jax.Array,
    *,
    gmres_iterations: int,
    maximum_step: jax.Array,
) -> _RebuiltModelPromotion:
    """Re-linearize and seek a window-decreasing Levenberg-damped step."""
    mapped, tangent = jax.linearize(map_fn, state)
    residual_vector = mapped - state

    def linear_action(vector):
        return vector - tangent(vector)

    transpose_action = jax.linear_transpose(linear_action, jnp.zeros_like(state))

    def transpose(vector):
        return transpose_action(vector)[0]

    normal_rhs = transpose(residual_vector)

    def damping_trial(_index, carry):
        selected_state, selected_residual, accepted, damping, recorded = carry

        def damped_normal_action(vector):
            return transpose(linear_action(vector)) + damping * vector

        step, info = jax.scipy.sparse.linalg.gmres(
            damped_normal_action,
            normal_rhs,
            maxiter=gmres_iterations,
            restart=gmres_iterations,
            solve_method="batched",
        )
        norm_step = jnp.max(jnp.abs(step))
        bounded_step = jnp.where(
            norm_step > maximum_step,
            step * (maximum_step / jnp.maximum(norm_step, 1.0e-300)),
            step,
        )
        candidate = state + bounded_step
        candidate_mapped = map_fn(candidate)
        candidate_merit = _smooth_relative_sup_merit(candidate_mapped, candidate)
        candidate_residual = _relative_residual(candidate_mapped, candidate)
        select = (
            ~accepted
            & (jnp.asarray(info) == 0)
            & jnp.all(jnp.isfinite(step))
            & jnp.isfinite(candidate_merit)
            & jnp.isfinite(candidate_residual)
            & (candidate_merit <= reference_merit * (1.0 - _SUFFICIENT_DECREASE_SLOPE))
        )
        return (
            jnp.where(select, candidate, selected_state),
            jnp.where(select, candidate_residual, selected_residual),
            accepted | select,
            jnp.where(
                accepted | select,
                damping,
                damping * _MODEL_REBUILD_DAMPING_GROWTH,
            ),
            jnp.where(select | ~accepted, damping, recorded),
        )

    candidate, residual, accepted, _next_damping, recorded_damping = jax.lax.fori_loop(
        0,
        _MODEL_REBUILD_DAMPING_TRIPS,
        damping_trial,
        (
            state,
            current_residual,
            jnp.asarray(False),
            jnp.asarray(_MODEL_REBUILD_DAMPING_INITIAL, dtype=state.dtype),
            jnp.asarray(jnp.nan, dtype=state.dtype),
        ),
    )
    return _RebuiltModelPromotion(
        state=candidate,
        residual=residual,
        accepted=accepted,
        damping=recorded_damping,
    )


def _steepest_descent_promotion(
    map_fn: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    current_residual: jax.Array,
    reference_merit: jax.Array,
) -> _SteepestDescentPromotion:
    """Seek Armijo decrease of the smooth relative-sup merit on a fixed ladder."""

    def merit(candidate):
        return _smooth_relative_sup_merit(map_fn(candidate), candidate)

    _current_merit, gradient = jax.value_and_grad(merit)(state)
    gradient_norm = jnp.max(jnp.abs(gradient))
    finite_gradient = jnp.all(jnp.isfinite(gradient)) & jnp.isfinite(gradient_norm)
    direction = jnp.where(
        finite_gradient & (gradient_norm > 0.0),
        -gradient / jnp.maximum(gradient_norm, 1.0e-300),
        jnp.zeros_like(gradient),
    )
    scales = jnp.asarray(_STEEPEST_DESCENT_SCALES, dtype=state.dtype)
    candidates = state[None, :] + scales[:, None] * direction[None, :]

    def score(candidate):
        mapped = map_fn(candidate)
        return (
            _smooth_relative_sup_merit(mapped, candidate),
            _relative_residual(mapped, candidate),
        )

    merits, residuals = jax.lax.map(score, candidates)
    directional_derivative = jnp.sum(gradient * direction)
    required = reference_merit + (
        _SUFFICIENT_DECREASE_SLOPE * scales * directional_derivative
    )
    sufficient = (
        finite_gradient
        & (gradient_norm > 0.0)
        & jnp.isfinite(merits)
        & jnp.isfinite(residuals)
        & (merits <= required)
    )
    accepted = jnp.any(sufficient)
    selected = jnp.argmax(sufficient)
    return _SteepestDescentPromotion(
        state=jnp.where(accepted, candidates[selected], state),
        residual=jnp.where(accepted, residuals[selected], current_residual),
        accepted=accepted,
        scale=jnp.where(
            accepted, scales[selected], jnp.asarray(0.0, dtype=state.dtype)
        ),
    )


def _qualified_krylov_step(
    linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    nonlinear_residual: jax.Array,
    *,
    gmres_iterations: int,
    condition_ratio_limit: float,
    preceding_condition_baseline: jax.Array,
) -> _QualifiedKrylovStep:
    """Solve, condition, and apply the shared linear-action qualification."""
    probe_scale = jnp.maximum(jnp.max(jnp.abs(residual_vector)), 1.0e-300)
    probe = jnp.where(
        probe_scale > 1.0e-300,
        residual_vector / probe_scale,
        jnp.ones_like(residual_vector) / jnp.sqrt(residual_vector.size),
    )
    finite_linear_action = jnp.all(jnp.isfinite(linear_action(probe)))
    projected_condition, spectral_baseline = _projected_krylov_condition(
        linear_action,
        residual_vector,
        krylov_dimension=gmres_iterations,
    )
    has_preceding_baseline = jnp.isfinite(preceding_condition_baseline) & (
        preceding_condition_baseline > 0.0
    )
    condition_baseline = jnp.where(
        has_preceding_baseline,
        jnp.sqrt(preceding_condition_baseline * spectral_baseline),
        spectral_baseline,
    )

    step, info = jax.scipy.sparse.linalg.gmres(
        linear_action,
        residual_vector,
        maxiter=gmres_iterations,
        restart=gmres_iterations,
        solve_method="batched",
    )
    unconditioned_step = step
    achieved_linear_residual = residual_vector - linear_action(step)
    finite_achieved_residual = jnp.all(jnp.isfinite(step)) & jnp.all(
        jnp.isfinite(achieved_linear_residual)
    )
    successful_status = jnp.asarray(info) == 0
    norm_step = jnp.max(jnp.abs(step))
    material_residual_floor = jnp.sqrt(jnp.finfo(residual_vector.dtype).eps)
    zero_step_at_material_residual = (norm_step == 0.0) & (
        nonlinear_residual > material_residual_floor
    )

    finite_condition = jnp.isfinite(projected_condition)
    conditioning_applied = (
        finite_condition
        & jnp.isfinite(condition_baseline)
        & (projected_condition > condition_ratio_limit * condition_baseline)
        & (nonlinear_residual > jnp.finfo(residual_vector.dtype).eps ** 0.25)
    )
    damping = jnp.where(
        conditioning_applied,
        condition_baseline / (condition_ratio_limit * projected_condition),
        1.0,
    )
    step = step * damping

    qualification = jnp.asarray(KrylovActionQualification.ACCEPTED, dtype=jnp.int32)
    qualification = jnp.where(
        zero_step_at_material_residual,
        KrylovActionQualification.ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL,
        qualification,
    )
    qualification = jnp.where(
        ~finite_achieved_residual,
        KrylovActionQualification.NONFINITE_ACHIEVED_LINEAR_RESIDUAL,
        qualification,
    )
    qualification = jnp.where(
        ~successful_status,
        KrylovActionQualification.NONSUCCESSFUL_GMRES_STATUS,
        qualification,
    )
    qualification = jnp.where(
        ~finite_linear_action,
        KrylovActionQualification.NONFINITE_LINEAR_ACTION,
        qualification,
    )
    return _QualifiedKrylovStep(
        step=step,
        qualification=qualification,
        projected_condition=projected_condition,
        conditioning_applied=conditioning_applied,
        condition_baseline=condition_baseline,
        unconditioned_step=unconditioned_step,
    )


def picard(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    evaluations: int,
    relaxation: float = 0.5,
    shadow_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Relaxed Picard iteration with per-evaluation residual accounting."""
    initial = _solver_state(initial, precision)

    observe_shadows = shadow_mask_fn is not None
    carry_shadows = promoted_shadow_mask_fn is not None
    if carry_shadows != (shadowed_map_fn is not None):
        raise ValueError("promoted shadow masks require a shadowed map")
    if carry_shadows and not observe_shadows:
        raise ValueError("promoted shadow masks require an initial shadow mask")

    def shadow_mask(state):
        if observe_shadows:
            return jnp.ravel(jnp.asarray(shadow_mask_fn(state), dtype=bool))
        return jnp.zeros(1, dtype=bool)

    def mapped_with_shadow(state, shadow):
        if carry_shadows:
            return shadowed_map_fn(state, shadow)
        return map_fn(state)

    def promoted_shadow(state, previous):
        if carry_shadows:
            return jnp.ravel(promoted_shadow_mask_fn(state, previous))
        return shadow_mask(state)

    def body(index, carry):
        state, trace, previous_shadow, shadow_changes = carry
        mapped = mapped_with_shadow(state, previous_shadow)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        candidate = state + relaxation * (mapped - state)
        current_shadow = promoted_shadow(candidate, previous_shadow)
        changed = jnp.sum(current_shadow != previous_shadow, dtype=jnp.int32)
        shadow_changes = shadow_changes.at[index].set(
            jnp.where(observe_shadows, changed, -1)
        )
        return candidate, trace, current_shadow, shadow_changes

    state, trace, _shadow, shadow_changes = jax.lax.fori_loop(
        0,
        evaluations,
        body,
        (
            initial,
            jnp.full(evaluations, jnp.nan, dtype=initial.dtype),
            shadow_mask(initial),
            jnp.full(evaluations, -1, dtype=jnp.int32),
        ),
    )
    return FixedPointResult(
        state, trace[evaluations - 1], trace, shadow_mask_changes=shadow_changes
    )


def anderson(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    evaluations: int,
    relaxation: float = 0.5,
    depth: int = 3,
    warmup: int = 6,
    step_cap: float = 2.0,
    ridge: float = 1.0e-10,
    shadow_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Safeguarded Anderson acceleration of the relaxed iteration.

    Walker–Ni type-II mixing: with ``f = g(x) − x`` and ΔX, ΔF the last
    ``depth`` iterate/residual differences,

        x⁺ = x + β f − (ΔX + β ΔF) γ,   γ = argmin ‖f − ΔF γ‖²,

    reducing to plain relaxed Picard while the history is shorter than two.
    Four guards, none costing a map evaluation: engagement is delayed
    ``warmup`` evaluations past the transient; the history restarts when the
    residual norm grows; the accepted move is capped at ``step_cap`` × the
    plain relaxed step; and a non-finite candidate falls back to the relaxed
    step.  The ridge is relative to the mean diagonal of ΔFᵀΔF.
    """
    initial = _solver_state(initial, precision)
    n_flat = initial.shape[0]
    observe_shadows = shadow_mask_fn is not None
    carry_shadows = promoted_shadow_mask_fn is not None
    if carry_shadows != (shadowed_map_fn is not None):
        raise ValueError("promoted shadow masks require a shadowed map")
    if carry_shadows and not observe_shadows:
        raise ValueError("promoted shadow masks require an initial shadow mask")

    def shadow_mask(state):
        if observe_shadows:
            return jnp.ravel(jnp.asarray(shadow_mask_fn(state), dtype=bool))
        return jnp.zeros(1, dtype=bool)

    def mapped_with_shadow(state, shadow):
        if carry_shadows:
            return shadowed_map_fn(state, shadow)
        return map_fn(state)

    def promoted_shadow(state, previous):
        if carry_shadows:
            return jnp.ravel(promoted_shadow_mask_fn(state, previous))
        return shadow_mask(state)

    def body(index, carry):
        (
            state,
            dx,
            df,
            f_prev,
            x_prev,
            norm_prev,
            trace,
            previous_shadow,
            shadow_changes,
        ) = carry
        mapped = mapped_with_shadow(state, previous_shadow)
        f = mapped - state
        trace = trace.at[index].set(_relative_residual(mapped, state))
        relaxed = state + relaxation * f

        norm_f = jnp.max(jnp.abs(f))
        grew = norm_f > norm_prev
        dx = jnp.where(grew, jnp.zeros_like(dx), dx)
        df = jnp.where(grew, jnp.zeros_like(df), df)
        column = jnp.mod(index, depth)
        update = jax.lax.dynamic_update_index_in_dim
        have_history = (index >= 1) & ~grew
        dx = jnp.where(have_history, update(dx, state - x_prev, column, axis=1), dx)
        df = jnp.where(have_history, update(df, f - f_prev, column, axis=1), df)

        gram = df.T @ df
        gram = gram + ridge * (jnp.trace(gram) / depth + 1.0e-30) * jnp.eye(
            depth, dtype=gram.dtype
        )
        gamma = jnp.linalg.solve(gram, df.T @ f)
        mixed = state + relaxation * f - (dx + relaxation * df) @ gamma

        # step cap: never move more than step_cap x the plain relaxed step —
        # the move is RESCALED to the cap (not rejected), bounding any single
        # wild step through the early transient without forfeiting the mix
        step = mixed - state
        step_norm = jnp.max(jnp.abs(step))
        allowed = step_cap * jnp.maximum(jnp.max(jnp.abs(relaxed - state)), 1.0e-300)
        mixed = jnp.where(
            step_norm > allowed, state + step * (allowed / step_norm), mixed
        )
        accept = (index >= warmup) & ~grew & jnp.all(jnp.isfinite(mixed))
        state_next = jnp.where(accept, mixed, relaxed)
        current_shadow = promoted_shadow(state_next, previous_shadow)
        changed = jnp.sum(current_shadow != previous_shadow, dtype=jnp.int32)
        shadow_changes = shadow_changes.at[index].set(
            jnp.where(observe_shadows, changed, -1)
        )
        return (
            state_next,
            dx,
            df,
            f,
            state,
            norm_f,
            trace,
            current_shadow,
            shadow_changes,
        )

    init = (
        initial,
        jnp.zeros((n_flat, depth), dtype=initial.dtype),
        jnp.zeros((n_flat, depth), dtype=initial.dtype),
        jnp.zeros(n_flat, dtype=initial.dtype),
        initial,
        jnp.asarray(jnp.inf, dtype=initial.dtype),
        jnp.full(evaluations, jnp.nan, dtype=initial.dtype),
        shadow_mask(initial),
        jnp.full(evaluations, -1, dtype=jnp.int32),
    )
    state, *middle, trace, _shadow, shadow_changes = jax.lax.fori_loop(
        0, evaluations, body, init
    )
    del middle
    return FixedPointResult(
        state, trace[evaluations - 1], trace, shadow_mask_changes=shadow_changes
    )


def _manifold_newton_krylov(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    previous_admitted_state: jax.Array,
    admissibility_fn: Callable[[jax.Array], jax.Array],
    *,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    krylov_condition_limit: float,
    convergence_tolerance: float,
    precision: Precision | str,
) -> FixedPointResult:
    """Advance along admitted-state secants and correct in their normal space."""
    initial = _solver_state(initial, precision)
    previous = _solver_state(previous_admitted_state, precision)
    if previous.shape != initial.shape:
        raise ValueError("previous admitted state must match the initial state shape")

    stride = 3
    trace_length = warmup + newton_steps * stride

    def warm_body(index, carry):
        state, prior, trace, amplification = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        candidate = state + relaxation * (mapped - state)
        admitted = jnp.all(jnp.isfinite(candidate)) & jnp.asarray(
            admissibility_fn(candidate), dtype=jnp.bool_
        )
        amplification = _observe_increment(amplification, state, candidate, admitted)
        prior = jnp.where(admitted, state, prior)
        state = jnp.where(admitted, candidate, state)
        return state, prior, trace, amplification

    state, previous, trace, amplification = jax.lax.fori_loop(
        0,
        warmup,
        warm_body,
        (
            initial,
            previous,
            jnp.full(trace_length, jnp.nan, dtype=initial.dtype),
            _initial_amplification_state(initial.dtype),
        ),
    )
    initial_secant = oriented_secant(previous, state, state - previous)
    tangent_orientation = initial_secant.tangent

    def bounded_step(step, residual_vector):
        fallback = relaxation * residual_vector
        cap = step_cap * jnp.max(jnp.abs(fallback))
        norm_step = jnp.max(jnp.abs(step))
        return jnp.where(
            norm_step > cap,
            step * (cap / jnp.maximum(norm_step, 1.0e-300)),
            step,
        )

    def continue_newton(carry):
        return (~carry.current_measured) | (
            carry.active & (carry.attempted < newton_steps)
        )

    def newton_body(carry):
        state = carry.state
        mapped, tangent_action = jax.linearize(map_fn, state)
        residual_vector = mapped - state
        current_residual = _relative_residual(mapped, state)
        base = warmup + carry.attempted * stride
        trace = carry.trace
        if newton_steps > 0:
            trace = trace.at[base].set(current_residual)
        finite_residual = jnp.isfinite(current_residual)
        current_converged = finite_residual & (
            current_residual <= convergence_tolerance
        )
        can_attempt = (
            finite_residual & ~current_converged & (carry.attempted < newton_steps)
        )
        stopped_reason = jnp.where(
            ~finite_residual,
            FixedPointTerminationReason.NONFINITE_RESIDUAL,
            jnp.where(
                current_converged,
                FixedPointTerminationReason.CONVERGED,
                FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
            ),
        )
        measured = carry._replace(
            residual=current_residual,
            trace=trace,
            converged=current_converged,
            termination_reason=jnp.asarray(stopped_reason, dtype=jnp.int32),
            active=jnp.asarray(False),
            current_measured=jnp.asarray(True),
        )

        def attempt_step(measured):
            def linear_action(vector):
                return vector - tangent_action(vector)

            predictor_action = _qualified_krylov_step(
                linear_action,
                residual_vector,
                current_residual,
                gmres_iterations=gmres_iterations,
                condition_ratio_limit=krylov_condition_limit,
                preceding_condition_baseline=measured.condition_baseline,
            )
            predictor_step = bounded_step(predictor_action.step, residual_vector)
            secant = oriented_secant(
                measured.previous, state, measured.tangent_orientation
            )
            predictor_length = jnp.linalg.norm(predictor_step)
            predictor_accepted = (
                predictor_action.qualification == KrylovActionQualification.ACCEPTED
            )
            secant_accepted = (
                secant.qualification == ManifoldAdvanceQualification.ACCEPTED
            )
            predictor_enabled = predictor_accepted & secant_accepted
            predictor = state + jnp.where(
                predictor_enabled,
                predictor_length * secant.tangent,
                jnp.zeros_like(state),
            )
            predictor_image, predictor_tangent_action = jax.linearize(map_fn, predictor)
            predictor_residual_vector = predictor_image - predictor
            normal_residual = normal_component(
                predictor_residual_vector, secant.tangent
            )
            normal_relative_residual = jnp.max(jnp.abs(normal_residual)) / jnp.maximum(
                jnp.max(jnp.abs(predictor_image)), 1.0e-30
            )

            def bordered_action(vector):
                action = vector - predictor_tangent_action(vector)
                normal_action = normal_component(action, secant.tangent)
                tangential_constraint = (
                    jnp.vdot(vector, secant.tangent) * secant.tangent
                )
                return normal_action + tangential_constraint

            corrector_action = _qualified_krylov_step(
                bordered_action,
                normal_residual,
                normal_relative_residual,
                gmres_iterations=gmres_iterations,
                condition_ratio_limit=krylov_condition_limit,
                preceding_condition_baseline=predictor_action.condition_baseline,
            )
            correction = normal_component(corrector_action.step, secant.tangent)
            corrector_accepted = (
                corrector_action.qualification == KrylovActionQualification.ACCEPTED
            )
            candidate = predictor + jnp.where(
                predictor_enabled & corrector_accepted,
                correction,
                jnp.zeros_like(correction),
            )
            candidate_image = map_fn(candidate)
            candidate_residual = _relative_residual(candidate_image, candidate)
            finite_candidate = (
                jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(candidate_image))
                & jnp.isfinite(candidate_residual)
            )
            admitted = jnp.asarray(admissibility_fn(candidate), dtype=jnp.bool_)
            advance_length = jnp.linalg.norm(candidate - state)
            material_floor = jnp.sqrt(jnp.finfo(state.dtype).eps) * jnp.maximum(
                jnp.linalg.norm(state), 1.0
            )
            material_advance = advance_length > material_floor
            actions_accepted = predictor_accepted & corrector_accepted
            promoted = (
                secant_accepted
                & actions_accepted
                & finite_candidate
                & admitted
                & material_advance
            )
            advance_qualification = jnp.asarray(
                ManifoldAdvanceQualification.ACCEPTED, dtype=jnp.int32
            )
            advance_qualification = jnp.where(
                ~material_advance,
                ManifoldAdvanceQualification.ZERO_MATERIAL_ADVANCE,
                advance_qualification,
            )
            advance_qualification = jnp.where(
                ~admitted,
                ManifoldAdvanceQualification.INADMISSIBLE_CORRECTED_STATE,
                advance_qualification,
            )
            advance_qualification = jnp.where(
                ~finite_candidate,
                ManifoldAdvanceQualification.NONFINITE_CORRECTED_STATE,
                advance_qualification,
            )
            advance_qualification = jnp.where(
                ~actions_accepted,
                ManifoldAdvanceQualification.KRYLOV_ACTION_REFUSED,
                advance_qualification,
            )
            advance_qualification = jnp.where(
                ~secant_accepted,
                ManifoldAdvanceQualification.DEGENERATE_SECANT,
                advance_qualification,
            )
            step_qualification = jnp.where(
                predictor_accepted,
                corrector_action.qualification,
                predictor_action.qualification,
            )
            prior_failed = (
                measured.qualification != KrylovActionQualification.NOT_APPLICABLE
            ) & (measured.qualification != KrylovActionQualification.ACCEPTED)
            qualification = jnp.where(
                prior_failed, measured.qualification, step_qualification
            )
            amplification = _observe_increment(
                measured.amplification, state, candidate, promoted
            )
            trace = measured.trace.at[base + 1].set(normal_relative_residual)
            trace = trace.at[base + 2].set(
                jnp.where(promoted, candidate_residual, current_residual)
            )
            previous = jnp.where(promoted, state, measured.previous)
            tangent_orientation = jnp.where(
                promoted, secant.tangent, measured.tangent_orientation
            )
            terminal_state = jnp.where(promoted, candidate, state)
            residual = jnp.where(promoted, candidate_residual, current_residual)
            conditioning_count = measured.conditioning_count + jnp.asarray(
                predictor_accepted & predictor_action.conditioning_applied,
                dtype=jnp.int32,
            )
            conditioning_count = conditioning_count + jnp.asarray(
                corrector_accepted & corrector_action.conditioning_applied,
                dtype=jnp.int32,
            )
            maximum_condition = jnp.maximum(
                measured.maximum_condition,
                jnp.maximum(
                    predictor_action.projected_condition,
                    corrector_action.projected_condition,
                ),
            )
            index = measured.attempted
            advance_qualifications = measured.advance_qualifications.at[index].set(
                advance_qualification
            )
            manifold_admissibility = measured.manifold_admissibility.at[index].set(
                finite_candidate & admitted
            )
            predictor_lengths = measured.predictor_lengths.at[index].set(
                predictor_length
            )
            corrector_lengths = measured.corrector_lengths.at[index].set(
                jnp.linalg.norm(correction)
            )
            advance_lengths = measured.advance_lengths.at[index].set(
                jnp.where(promoted, advance_length, 0.0)
            )
            newton_step_lengths = measured.newton_step_lengths.at[index].set(
                jnp.where(predictor_accepted, predictor_length, 0.0)
            )
            attempted = measured.attempted + 1
            accepted = measured.accepted + jnp.asarray(promoted, dtype=jnp.int32)
            converged = promoted & (candidate_residual <= convergence_tolerance)
            exhausted = attempted >= newton_steps
            active = promoted & ~converged & ~exhausted
            reason = jnp.where(
                ~finite_candidate,
                FixedPointTerminationReason.NONFINITE_RESIDUAL,
                jnp.where(
                    ~actions_accepted,
                    FixedPointTerminationReason.KRYLOV_ACTION_REFUSED,
                    jnp.where(
                        ~promoted,
                        FixedPointTerminationReason.MANIFOLD_ADVANCE_REFUSED,
                        jnp.where(
                            converged,
                            FixedPointTerminationReason.CONVERGED,
                            FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
                        ),
                    ),
                ),
            )
            return _ManifoldNewtonIterationState(
                terminal_state,
                previous,
                tangent_orientation,
                residual,
                trace,
                qualification,
                amplification,
                conditioning_count,
                maximum_condition,
                corrector_action.condition_baseline,
                advance_qualifications,
                manifold_admissibility,
                predictor_lengths,
                corrector_lengths,
                advance_lengths,
                newton_step_lengths,
                attempted,
                accepted,
                converged,
                jnp.asarray(reason, dtype=jnp.int32),
                active,
                jnp.asarray(True),
            )

        return jax.lax.cond(can_attempt, attempt_step, lambda value: value, measured)

    loop = jax.lax.while_loop(
        continue_newton,
        newton_body,
        _ManifoldNewtonIterationState(
            state,
            previous,
            tangent_orientation,
            jnp.asarray(jnp.inf, dtype=initial.dtype),
            trace,
            jnp.asarray(KrylovActionQualification.NOT_APPLICABLE, dtype=jnp.int32),
            amplification,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=initial.dtype),
            jnp.asarray(jnp.nan, dtype=initial.dtype),
            jnp.full(
                newton_steps,
                ManifoldAdvanceQualification.NOT_APPLICABLE,
                dtype=jnp.int32,
            ),
            jnp.zeros(newton_steps, dtype=jnp.bool_),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(
                FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
                dtype=jnp.int32,
            ),
            jnp.asarray(True),
            jnp.asarray(False),
        ),
    )
    length_floor = jnp.finfo(initial.dtype).tiny
    newton_step_equivalents = jnp.sum(
        jnp.where(
            loop.newton_step_lengths > length_floor,
            loop.advance_lengths / jnp.maximum(loop.newton_step_lengths, length_floor),
            0.0,
        )
    )
    return FixedPointResult(
        state=loop.state,
        residual=loop.residual,
        trace=loop.trace,
        krylov_action_qualification=loop.qualification,
        amplification_observation=_amplification_result(
            loop.amplification, loop.qualification
        ),
        krylov_conditioning_count=loop.conditioning_count,
        maximum_projected_krylov_condition=loop.maximum_condition,
        manifold_advance_qualification=loop.advance_qualifications,
        manifold_admissibility=loop.manifold_admissibility,
        predictor_lengths=loop.predictor_lengths,
        corrector_lengths=loop.corrector_lengths,
        advance_lengths=loop.advance_lengths,
        newton_step_lengths=loop.newton_step_lengths,
        newton_step_equivalents=newton_step_equivalents,
        attempted_newton_promotions=loop.attempted,
        accepted_newton_promotions=loop.accepted,
        converged=loop.converged,
        termination_reason=loop.termination_reason,
    )


def _newton_krylov_inner(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    newton_steps: int,
    gmres_iterations: int = 8,
    warmup: int = 8,
    relaxation: float = 0.5,
    step_cap: float = 10.0,
    krylov_condition_limit: float = _PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
    convergence_tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    admissibility_fn: Callable[[jax.Array], jax.Array] | None = None,
    previous_admitted_state: jax.Array | None = None,
    shadow_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Exact-tangent Jacobian-free Newton–Krylov on the fixed-point residual.

    After ``warmup`` relaxed Picard sweeps, each Newton attempt linearises the
    map once with the exact ``jax.linearize`` tangent and solves
    ``(I − J) s = f`` with a ``gmres_iterations``-step fixed-shape batched
    GMRES.  The bounded outer loop stops before proposing a step when the
    current finite relative residual is at or below ``convergence_tolerance``;
    it also stops immediately after the first promoted state meeting that
    criterion.  A step is refused when the initial linear action is non-finite,
    GMRES reports a non-successful status, the achieved linear residual is
    non-finite, or an exactly-zero step carries a material nonlinear residual.
    Before promotion, the same fixed Krylov projection measures the condition
    of ``I - J``.  Its largest-to-median singular-value ratio supplies a robust
    typical-spectrum baseline, combined geometrically with the preceding
    projection's baseline.  When the full condition exceeds
    ``krylov_condition_limit`` times that online baseline, the step is scaled
    by baseline divided by condition and by the default ratio.  The default
    trigger is one natural-log unit of separation (a ratio of ``e``), so its
    scale comes from the ratio coordinate rather than a measured trajectory;
    applying the same margin below the baseline supplies symmetric log-space
    hysteresis.  The first-power inverse law is the proportional normalization
    that returns condition-weighted step size to a fixed baseline multiple; it
    needs no projection-dimension calibration.  Damping releases below the
    fourth root of machine precision so local convergence can finish without
    trajectory control.  A qualified
    step is then capped at ``step_cap`` × the relaxed step, bounding excursions
    while the current-centroid pin holds the basin.  Promotion selects the
    first of six fixed half-step factors that provides sufficient nonlinear
    residual decrease.  If that ladder exhausts, a bounded trust-region update
    follows the existing relaxed Picard direction and may promote only a
    strictly decreasing residual.  Its radius is carried across promotions,
    shrinks after refused trials, and grows after an accepted re-shape.  The
    update uses a fixed-trip masked loop, and the receipts record the Newton
    reductions, recovery activation, outcome, and radius trajectory.  Once the
    carried radius reaches its numerical floor, the recovery rebuilds the
    local linearization at the unchanged iterate and solves a fixed ladder of
    Levenberg-damped normal models.  That fresh-model step is tested against
    the same bounded merit envelope; its activation and selected damping are
    retained in the promotion receipts.  If both recovery models exhaust, a
    final fixed ladder follows the negative gradient of the relative
    eighth-norm merit, normalized to unit infinity norm.  Every promotion route
    compares sufficient decrease with the maximum of the six most recently
    accepted finite merits.  The exact relative-sup residual still selects the
    best returned iterate and supplies the unchanged convergence criterion.  The
    trace retains its configured length and ``2 + gmres_iterations`` stride
    (one linearisation value, tangent slots, and one promotion read); unused
    iterations remain NaN padding.  Under ``vmap``, the batched while-loop runs
    to the slowest active lane while preserving each lane's receipt.  Supplying
    both ``admissibility_fn`` and ``previous_admitted_state`` selects the
    topology-manifold mode through this same production solver seam.  Its
    secant predictor uses state-space arclength, its bordered corrector is
    normal to that secant, and promotion still requires both shared Krylov
    qualifications plus the caller's existing physical predicate.
    """
    if newton_steps < 0:
        raise ValueError("newton_steps must be non-negative")
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be finite and positive")
    if (admissibility_fn is None) != (previous_admitted_state is None):
        raise ValueError(
            "manifold advance requires both admissibility_fn and "
            "previous_admitted_state"
        )
    if admissibility_fn is not None:
        return _manifold_newton_krylov(
            map_fn,
            initial,
            previous_admitted_state,
            admissibility_fn,
            newton_steps=newton_steps,
            gmres_iterations=gmres_iterations,
            warmup=warmup,
            relaxation=relaxation,
            step_cap=step_cap,
            krylov_condition_limit=krylov_condition_limit,
            convergence_tolerance=convergence_tolerance,
            precision=precision,
        )
    initial = _solver_state(initial, precision)
    stride = 2 + gmres_iterations
    trace_length = warmup + newton_steps * stride
    change_length = warmup + newton_steps
    observe_shadows = shadow_mask_fn is not None
    carry_shadows = promoted_shadow_mask_fn is not None
    if carry_shadows != (shadowed_map_fn is not None):
        raise ValueError("promoted shadow masks require a shadowed map")
    if carry_shadows and not observe_shadows:
        raise ValueError("promoted shadow masks require an initial shadow mask")

    def shadow_mask(state):
        if observe_shadows:
            return jnp.ravel(jnp.asarray(shadow_mask_fn(state), dtype=bool))
        return jnp.zeros(1, dtype=bool)

    def mapped_with_shadow(state, shadow):
        if carry_shadows:
            return shadowed_map_fn(state, shadow)
        return map_fn(state)

    def promoted_shadow(state, previous):
        if carry_shadows:
            return jnp.ravel(promoted_shadow_mask_fn(state, previous))
        return shadow_mask(state)

    def warm_body(index, carry):
        (
            state,
            trace,
            amplification,
            previous_shadow,
            shadow_changes,
            best_state,
            best_residual,
            recent_merits,
            merit_observations,
        ) = carry
        mapped = mapped_with_shadow(state, previous_shadow)
        residual = _relative_residual(mapped, state)
        merit = _smooth_relative_sup_merit(mapped, state)
        trace = trace.at[index].set(residual)
        is_best = jnp.isfinite(residual) & (residual < best_residual)
        best_state = jnp.where(is_best, state, best_state)
        best_residual = jnp.where(is_best, residual, best_residual)
        recent_merits, merit_observations = _record_merit(
            recent_merits,
            merit_observations,
            merit,
            jnp.asarray(True),
        )
        candidate = state + relaxation * (mapped - state)
        amplification = _observe_increment(amplification, state, candidate, True)
        current_shadow = promoted_shadow(candidate, previous_shadow)
        changed = jnp.sum(current_shadow != previous_shadow, dtype=jnp.int32)
        shadow_changes = shadow_changes.at[index].set(
            jnp.where(observe_shadows, changed, -1)
        )
        return (
            candidate,
            trace,
            amplification,
            current_shadow,
            shadow_changes,
            best_state,
            best_residual,
            recent_merits,
            merit_observations,
        )

    (
        state,
        trace,
        amplification,
        current_shadow,
        shadow_changes,
        best_state,
        best_residual,
        recent_merits,
        merit_observations,
    ) = jax.lax.fori_loop(
        0,
        warmup,
        warm_body,
        (
            initial,
            jnp.full(trace_length, jnp.nan, dtype=initial.dtype),
            _initial_amplification_state(initial.dtype),
            shadow_mask(initial),
            jnp.full(change_length, -1, dtype=jnp.int32),
            initial,
            jnp.asarray(jnp.inf, dtype=initial.dtype),
            jnp.full(_NONMONOTONE_MERIT_WINDOW, jnp.nan, dtype=initial.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )

    def continue_newton(carry):
        first_measurement = ~carry.current_measured
        has_budget = carry.attempted < newton_steps
        return first_measurement | (carry.active & has_budget)

    def newton_body(carry):
        state = carry.state

        def frozen_map(candidate):
            return mapped_with_shadow(candidate, carry.shadow_mask)

        mapped, tangent = jax.linearize(frozen_map, state)
        residual_vector = mapped - state
        nonlinear_residual = _relative_residual(mapped, state)
        current_merit = _smooth_relative_sup_merit(mapped, state)
        base = warmup + carry.attempted * stride
        trace = carry.trace
        if newton_steps > 0:
            trace = trace.at[base].set(nonlinear_residual)

        finite_residual = jnp.isfinite(nonlinear_residual)
        current_converged = finite_residual & (
            nonlinear_residual <= convergence_tolerance
        )
        can_attempt = (
            finite_residual & ~current_converged & (carry.attempted < newton_steps)
        )
        stopped_reason = jnp.where(
            ~finite_residual,
            FixedPointTerminationReason.NONFINITE_RESIDUAL,
            jnp.where(
                current_converged,
                FixedPointTerminationReason.CONVERGED,
                FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
            ),
        )
        current_is_best = finite_residual & (nonlinear_residual < carry.best_residual)
        recent_merits, merit_observations = _record_merit(
            carry.recent_merits,
            carry.merit_observations,
            current_merit,
            ~carry.current_measured,
        )
        reference_merit = _recent_merit_maximum(recent_merits, current_merit)
        measured = carry._replace(
            residual=nonlinear_residual,
            best_state=jnp.where(current_is_best, state, carry.best_state),
            best_residual=jnp.where(
                current_is_best, nonlinear_residual, carry.best_residual
            ),
            recent_merits=recent_merits,
            merit_observations=merit_observations,
            trace=trace,
            converged=current_converged,
            termination_reason=jnp.asarray(stopped_reason, dtype=jnp.int32),
            active=jnp.asarray(False),
            current_measured=jnp.asarray(True),
        )

        def attempt_step(measured):
            def linear_action(vector):
                return vector - tangent(vector)

            qualified_step = _qualified_krylov_step(
                linear_action,
                residual_vector,
                nonlinear_residual,
                gmres_iterations=gmres_iterations,
                condition_ratio_limit=krylov_condition_limit,
                preceding_condition_baseline=measured.condition_baseline,
            )
            step = qualified_step.step
            step_qualification = qualified_step.qualification
            action_accepted = step_qualification == KrylovActionQualification.ACCEPTED
            norm_step = jnp.max(jnp.abs(step))
            cap = step_cap * jnp.max(jnp.abs(relaxation * residual_vector))
            step = jnp.where(
                norm_step > cap,
                step * (cap / jnp.maximum(norm_step, 1.0e-300)),
                step,
            )
            attempted = measured.attempted + 1
            conditioning_count = measured.conditioning_count + jnp.asarray(
                action_accepted & qualified_step.conditioning_applied,
                dtype=jnp.int32,
            )
            maximum_condition = jnp.maximum(
                measured.maximum_condition, qualified_step.projected_condition
            )

            def promoted_state(_):
                promotion = _backtracked_promotion(
                    frozen_map,
                    state,
                    step,
                    relaxation * residual_vector,
                    nonlinear_residual,
                    reference_merit,
                    measured.recovery_radius,
                )
                minimum_radius = jnp.sqrt(jnp.finfo(state.dtype).eps)
                rebuild_activated = (
                    promotion.recovery_activated
                    & ~promotion.accepted
                    & (promotion.recovery_radius <= minimum_radius)
                )

                def rebuild_model(_):
                    return _rebuilt_model_promotion(
                        frozen_map,
                        state,
                        nonlinear_residual,
                        reference_merit,
                        gmres_iterations=gmres_iterations,
                        maximum_step=cap,
                    )

                def skip_rebuild(_):
                    return _RebuiltModelPromotion(
                        state=state,
                        residual=nonlinear_residual,
                        accepted=jnp.asarray(False),
                        damping=jnp.asarray(jnp.nan, dtype=state.dtype),
                    )

                rebuilt = jax.lax.cond(
                    rebuild_activated, rebuild_model, skip_rebuild, operand=None
                )
                descent_activated = rebuild_activated & ~rebuilt.accepted

                def descend(_):
                    return _steepest_descent_promotion(
                        frozen_map,
                        state,
                        nonlinear_residual,
                        reference_merit,
                    )

                def skip_descent(_):
                    return _SteepestDescentPromotion(
                        state=state,
                        residual=nonlinear_residual,
                        accepted=jnp.asarray(False),
                        scale=jnp.asarray(jnp.nan, dtype=state.dtype),
                    )

                descent = jax.lax.cond(
                    descent_activated, descend, skip_descent, operand=None
                )
                rebuilt_or_descent = jnp.where(
                    rebuilt.accepted, rebuilt.state, descent.state
                )
                rebuilt_or_descent_residual = jnp.where(
                    rebuilt.accepted, rebuilt.residual, descent.residual
                )
                candidate = jnp.where(
                    promotion.accepted, promotion.state, rebuilt_or_descent
                )
                candidate_residual = jnp.where(
                    promotion.accepted,
                    promotion.residual,
                    rebuilt_or_descent_residual,
                )
                promotion_accepted = (
                    promotion.accepted | rebuilt.accepted | descent.accepted
                )
                accepted = measured.accepted + jnp.asarray(
                    promotion_accepted, dtype=jnp.int32
                )
                amplification = _observe_increment(
                    measured.amplification, state, candidate, promotion_accepted
                )
                updated_trace = measured.trace.at[base + stride - 1].set(
                    candidate_residual
                )
                finite_candidate_residual = jnp.isfinite(candidate_residual)
                converged = (
                    promotion_accepted
                    & finite_candidate_residual
                    & (candidate_residual <= convergence_tolerance)
                )
                exhausted = attempted >= newton_steps
                recovery_retry = promotion.recovery_activated & ~promotion_accepted
                active = (
                    finite_candidate_residual
                    & ~converged
                    & ~exhausted
                    & (promotion_accepted | recovery_retry)
                )
                reason = jnp.where(
                    ~promotion_accepted,
                    FixedPointTerminationReason.SUFFICIENT_DECREASE_REFUSED,
                    jnp.where(
                        ~finite_candidate_residual,
                        FixedPointTerminationReason.NONFINITE_RESIDUAL,
                        jnp.where(
                            converged,
                            FixedPointTerminationReason.CONVERGED,
                            FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
                        ),
                    ),
                )
                proposed_shadow = promoted_shadow(candidate, measured.shadow_mask)
                candidate_shadow = jnp.where(
                    promotion_accepted, proposed_shadow, measured.shadow_mask
                )
                changed = jnp.sum(
                    candidate_shadow != measured.shadow_mask, dtype=jnp.int32
                )
                shadow_changes = measured.shadow_mask_changes.at[
                    warmup + measured.attempted
                ].set(jnp.where(observe_shadows, changed, -1))
                backtrack_counts = measured.promotion_backtrack_counts.at[
                    measured.attempted
                ].set(promotion.backtrack_count)
                recovery_activations = measured.promotion_recovery_activations.at[
                    measured.attempted
                ].set(jnp.asarray(promotion.recovery_activated, dtype=jnp.int32))
                recovery_radii = measured.promotion_recovery_radii.at[
                    measured.attempted
                ].set(
                    jnp.where(
                        promotion.recovery_activated,
                        jnp.stack(
                            (
                                promotion.recovery_radius_before,
                                promotion.recovery_radius,
                            )
                        ),
                        jnp.full(2, jnp.nan, dtype=state.dtype),
                    )
                )
                recovery_outcomes = measured.promotion_recovery_outcomes.at[
                    measured.attempted
                ].set(promotion.recovery_outcome)
                rebuild_activations = measured.promotion_model_rebuild_activations.at[
                    measured.attempted
                ].set(jnp.asarray(rebuild_activated, dtype=jnp.int32))
                rebuild_damping = measured.promotion_model_rebuild_damping.at[
                    measured.attempted
                ].set(
                    jnp.where(
                        rebuild_activated,
                        rebuilt.damping,
                        jnp.asarray(jnp.nan, dtype=state.dtype),
                    )
                )
                descent_activations = measured.promotion_descent_activations.at[
                    measured.attempted
                ].set(jnp.asarray(descent_activated, dtype=jnp.int32))
                descent_scales = measured.promotion_descent_scales.at[
                    measured.attempted
                ].set(
                    jnp.where(
                        descent_activated,
                        descent.scale,
                        jnp.asarray(jnp.nan, dtype=state.dtype),
                    )
                )
                candidate_is_best = (
                    promotion_accepted
                    & finite_candidate_residual
                    & (candidate_residual < measured.best_residual)
                )
                candidate_merit = _smooth_relative_sup_merit(
                    frozen_map(candidate), candidate
                )
                recent_merits, merit_observations = _record_merit(
                    measured.recent_merits,
                    measured.merit_observations,
                    candidate_merit,
                    promotion_accepted,
                )
                return _NewtonIterationState(
                    candidate,
                    candidate_residual,
                    jnp.where(candidate_is_best, candidate, measured.best_state),
                    jnp.where(
                        candidate_is_best,
                        candidate_residual,
                        measured.best_residual,
                    ),
                    recent_merits,
                    merit_observations,
                    updated_trace,
                    step_qualification,
                    amplification,
                    conditioning_count,
                    maximum_condition,
                    qualified_step.condition_baseline,
                    attempted,
                    accepted,
                    converged,
                    jnp.asarray(reason, dtype=jnp.int32),
                    active,
                    jnp.asarray(True),
                    candidate_shadow,
                    shadow_changes,
                    backtrack_counts,
                    recovery_activations,
                    promotion.recovery_radius,
                    recovery_radii,
                    recovery_outcomes,
                    rebuild_activations,
                    rebuild_damping,
                    descent_activations,
                    descent_scales,
                )

            def refused_state(_):
                return _NewtonIterationState(
                    state,
                    nonlinear_residual,
                    measured.best_state,
                    measured.best_residual,
                    measured.recent_merits,
                    measured.merit_observations,
                    measured.trace,
                    step_qualification,
                    measured.amplification,
                    conditioning_count,
                    maximum_condition,
                    qualified_step.condition_baseline,
                    attempted,
                    measured.accepted,
                    jnp.asarray(False),
                    jnp.asarray(
                        FixedPointTerminationReason.KRYLOV_ACTION_REFUSED,
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(False),
                    jnp.asarray(True),
                    measured.shadow_mask,
                    measured.shadow_mask_changes.at[warmup + measured.attempted].set(
                        jnp.where(observe_shadows, 0, -1)
                    ),
                    measured.promotion_backtrack_counts,
                    measured.promotion_recovery_activations.at[measured.attempted].set(
                        0
                    ),
                    measured.recovery_radius,
                    measured.promotion_recovery_radii,
                    measured.promotion_recovery_outcomes.at[measured.attempted].set(
                        RecoveryOutcome.NOT_APPLICABLE
                    ),
                    measured.promotion_model_rebuild_activations.at[
                        measured.attempted
                    ].set(0),
                    measured.promotion_model_rebuild_damping,
                    measured.promotion_descent_activations.at[measured.attempted].set(
                        0
                    ),
                    measured.promotion_descent_scales,
                )

            return jax.lax.cond(
                action_accepted, promoted_state, refused_state, operand=None
            )

        return jax.lax.cond(can_attempt, attempt_step, lambda value: value, measured)

    loop = jax.lax.while_loop(
        continue_newton,
        newton_body,
        _NewtonIterationState(
            state,
            jnp.asarray(jnp.inf, dtype=initial.dtype),
            best_state,
            best_residual,
            recent_merits,
            merit_observations,
            trace,
            jnp.asarray(KrylovActionQualification.NOT_APPLICABLE, dtype=jnp.int32),
            amplification,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=initial.dtype),
            jnp.asarray(jnp.nan, dtype=initial.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(
                FixedPointTerminationReason.ITERATION_BUDGET_EXHAUSTED,
                dtype=jnp.int32,
            ),
            jnp.asarray(True),
            jnp.asarray(False),
            current_shadow,
            shadow_changes,
            jnp.full(newton_steps, -1, dtype=jnp.int32),
            jnp.full(newton_steps, -1, dtype=jnp.int32),
            jnp.asarray(_RECOVERY_RADIUS_INITIAL, dtype=initial.dtype),
            jnp.full((newton_steps, 2), jnp.nan, dtype=initial.dtype),
            jnp.full(newton_steps, -1, dtype=jnp.int32),
            jnp.full(newton_steps, -1, dtype=jnp.int32),
            jnp.full(newton_steps, jnp.nan, dtype=initial.dtype),
            jnp.full(newton_steps, -1, dtype=jnp.int32),
            jnp.full(newton_steps, jnp.nan, dtype=initial.dtype),
        ),
    )
    return FixedPointResult(
        state=loop.best_state,
        residual=loop.best_residual,
        trace=loop.trace,
        krylov_action_qualification=loop.qualification,
        amplification_observation=_amplification_result(
            loop.amplification, loop.qualification
        ),
        krylov_conditioning_count=loop.conditioning_count,
        maximum_projected_krylov_condition=loop.maximum_condition,
        attempted_newton_promotions=loop.attempted,
        accepted_newton_promotions=loop.accepted,
        converged=loop.converged,
        termination_reason=loop.termination_reason,
        shadow_mask_changes=loop.shadow_mask_changes,
        promotion_backtrack_counts=loop.promotion_backtrack_counts,
        promotion_recovery_activations=loop.promotion_recovery_activations,
        promotion_recovery_radii=loop.promotion_recovery_radii,
        promotion_recovery_outcomes=loop.promotion_recovery_outcomes,
        promotion_model_rebuild_activations=(loop.promotion_model_rebuild_activations),
        promotion_model_rebuild_damping=loop.promotion_model_rebuild_damping,
        promotion_descent_activations=loop.promotion_descent_activations,
        promotion_descent_scales=loop.promotion_descent_scales,
    )


def _active_set_newton_krylov(
    initial: jax.Array,
    *,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    krylov_condition_limit: float,
    convergence_tolerance: float,
    shadow_mask_fn: Callable[[jax.Array], jax.Array],
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array],
    active_set_steps: int,
    stream_active_set: bool,
    stop_on_active_set_stagnation: bool,
    precision: Precision | str,
) -> FixedPointResult:
    """Reconcile bounded frozen-mask solves with their live active sets."""
    initial = _solver_state(initial, precision)
    initial_mask = jnp.ravel(jnp.asarray(shadow_mask_fn(initial), dtype=bool))
    history = jnp.zeros((active_set_steps + 1, initial_mask.size), dtype=bool)
    history = history.at[0].set(initial_mask)

    def solve_frozen(state, mask):
        def frozen_map(candidate):
            return shadowed_map_fn(candidate, mask)

        def frozen_mask(_candidate):
            return mask

        return _newton_krylov_inner(
            frozen_map,
            state,
            newton_steps=newton_steps,
            gmres_iterations=gmres_iterations,
            warmup=warmup,
            relaxation=relaxation,
            step_cap=step_cap,
            krylov_condition_limit=krylov_condition_limit,
            convergence_tolerance=convergence_tolerance,
            shadow_mask_fn=frozen_mask,
            precision=precision,
        )

    def mask_seen(mask, mask_history, history_count):
        populated = jnp.arange(mask_history.shape[0]) < history_count
        matches = jnp.all(mask_history == mask[None, :], axis=1)
        return jnp.any(populated & matches)

    def reconcile(
        index,
        state,
        mask,
        mask_history,
        history_count,
        inner_result,
        trip_active,
        previous_live_residual,
    ):
        solved_state = inner_result.state
        observed_mask = jnp.ravel(promoted_shadow_mask_fn(solved_state, mask))
        observed_difference = jnp.sum(observed_mask != mask, dtype=jnp.int32)
        observed_mapped = shadowed_map_fn(solved_state, observed_mask)
        observed_residual = _relative_residual(observed_mapped, solved_state)
        observed_finite = jnp.isfinite(observed_residual)
        converged = (
            observed_finite
            & (observed_residual <= convergence_tolerance)
            & (observed_difference == 0)
        )
        repeated = (
            (observed_difference > 0)
            & mask_seen(observed_mask, mask_history, history_count)
            & ~converged
        )

        damped_state = state + _ACTIVE_SET_CYCLE_DAMPING * (solved_state - state)
        damped_mask = jnp.ravel(promoted_shadow_mask_fn(damped_state, mask))
        damped_mapped = shadowed_map_fn(damped_state, damped_mask)
        damped_residual = _relative_residual(damped_mapped, damped_state)
        damped_finite = jnp.isfinite(damped_residual)
        damping_repeats = mask_seen(damped_mask, mask_history, history_count)
        cycle_detected = repeated & damping_repeats

        selected_state = jnp.where(repeated, damped_state, solved_state)
        selected_mask = jnp.where(repeated, damped_mask, observed_mask)
        selected_residual = jnp.where(repeated, damped_residual, observed_residual)
        selected_finite = jnp.where(repeated, damped_finite, observed_finite)
        selected_difference = jnp.sum(selected_mask != mask, dtype=jnp.int32)
        stagnated = (
            stop_on_active_set_stagnation
            & selected_finite
            & ~converged
            & ~cycle_detected
            & (selected_difference == 0)
            & (selected_residual == previous_live_residual)
        )
        if stream_active_set:
            jax.debug.callback(
                _print_active_set_trip,
                trip_active,
                jnp.asarray(index, dtype=jnp.int32),
                selected_difference,
                selected_residual,
                jnp.asarray(inner_result.attempted_newton_promotions, dtype=jnp.int32),
                ordered=True,
            )
        can_continue = selected_finite & ~converged & ~cycle_detected & ~stagnated
        record_mask = can_continue & (history_count < mask_history.shape[0])
        mask_history = mask_history.at[history_count].set(
            jnp.where(record_mask, selected_mask, mask_history[history_count])
        )
        history_count = history_count + record_mask.astype(jnp.int32)
        active = can_continue & ((index + 1) < active_set_steps)
        return (
            selected_state,
            selected_mask,
            mask_history,
            history_count,
            selected_residual,
            selected_difference,
            repeated,
            active,
            converged,
            cycle_detected,
            ~selected_finite,
            stagnated,
        )

    first_result = solve_frozen(initial, initial_mask)
    (
        first_state,
        first_mask,
        history,
        history_count,
        first_residual,
        first_difference,
        first_damping,
        first_active,
        first_converged,
        first_cycle,
        first_nonfinite,
        first_stagnated,
    ) = reconcile(
        0,
        initial,
        initial_mask,
        history,
        jnp.asarray(1, dtype=jnp.int32),
        first_result,
        jnp.asarray(True),
        jnp.asarray(jnp.nan, dtype=initial.dtype),
    )
    outer = _ActiveSetIterationState(
        state=first_state,
        mask=first_mask,
        mask_history=history,
        mask_history_count=history_count,
        result=first_result,
        live_residual=first_residual,
        residuals=jnp.full(active_set_steps, jnp.nan, dtype=initial.dtype)
        .at[0]
        .set(first_residual),
        mask_differences=jnp.full(active_set_steps, -1, dtype=jnp.int32)
        .at[0]
        .set(first_difference),
        cycle_damping_activations=jnp.full(active_set_steps, -1, dtype=jnp.int32)
        .at[0]
        .set(first_damping.astype(jnp.int32)),
        iterations=jnp.asarray(1, dtype=jnp.int32),
        attempted_promotions=jnp.asarray(
            first_result.attempted_newton_promotions, dtype=jnp.int32
        ),
        accepted_promotions=jnp.asarray(
            first_result.accepted_newton_promotions, dtype=jnp.int32
        ),
        active=first_active,
        converged=first_converged,
        cycle_detected=first_cycle,
        nonfinite=first_nonfinite,
        stagnated=first_stagnated,
    )

    def outer_body(index, carry):
        def solve_active(carry):
            inner_result = solve_frozen(carry.state, carry.mask)
            (
                state,
                mask,
                mask_history,
                history_count,
                live_residual,
                mask_difference,
                damping_activated,
                active,
                converged,
                cycle_detected,
                nonfinite,
                stagnated,
            ) = reconcile(
                index,
                carry.state,
                carry.mask,
                carry.mask_history,
                carry.mask_history_count,
                inner_result,
                carry.active,
                carry.live_residual,
            )
            return _ActiveSetIterationState(
                state=state,
                mask=mask,
                mask_history=mask_history,
                mask_history_count=history_count,
                result=inner_result,
                live_residual=live_residual,
                residuals=carry.residuals.at[index].set(live_residual),
                mask_differences=carry.mask_differences.at[index].set(mask_difference),
                cycle_damping_activations=(
                    carry.cycle_damping_activations.at[index].set(
                        damping_activated.astype(jnp.int32)
                    )
                ),
                iterations=carry.iterations + 1,
                attempted_promotions=(
                    carry.attempted_promotions
                    + jnp.asarray(
                        inner_result.attempted_newton_promotions, dtype=jnp.int32
                    )
                ),
                accepted_promotions=(
                    carry.accepted_promotions
                    + jnp.asarray(
                        inner_result.accepted_newton_promotions, dtype=jnp.int32
                    )
                ),
                active=active,
                converged=converged,
                cycle_detected=cycle_detected,
                nonfinite=nonfinite,
                stagnated=stagnated,
            )

        return jax.lax.cond(carry.active, solve_active, lambda value: value, carry)

    outer = jax.lax.fori_loop(1, active_set_steps, outer_body, outer)
    reason = jnp.where(
        outer.converged,
        FixedPointTerminationReason.CONVERGED,
        jnp.where(
            outer.cycle_detected,
            FixedPointTerminationReason.ACTIVE_SET_CYCLE_DETECTED,
            jnp.where(
                outer.nonfinite,
                FixedPointTerminationReason.NONFINITE_RESIDUAL,
                jnp.where(
                    outer.stagnated,
                    FixedPointTerminationReason.ACTIVE_SET_STAGNATED,
                    FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED,
                ),
            ),
        ),
    )
    return outer.result._replace(
        state=outer.state,
        residual=outer.live_residual,
        attempted_newton_promotions=outer.attempted_promotions,
        accepted_newton_promotions=outer.accepted_promotions,
        converged=outer.converged,
        termination_reason=jnp.asarray(reason, dtype=jnp.int32),
        active_set_iterations=outer.iterations,
        active_set_residuals=outer.residuals,
        active_set_mask_differences=outer.mask_differences,
        active_set_cycle_damping_activations=outer.cycle_damping_activations,
    )


def newton_krylov(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    newton_steps: int,
    gmres_iterations: int = 8,
    warmup: int = 8,
    relaxation: float = 0.5,
    step_cap: float = 10.0,
    krylov_condition_limit: float = _PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
    convergence_tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    admissibility_fn: Callable[[jax.Array], jax.Array] | None = None,
    previous_admitted_state: jax.Array | None = None,
    shadow_mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    active_set_steps: int = _ACTIVE_SET_ITERATION_LIMIT,
    stream_active_set: bool = False,
    stop_on_active_set_stagnation: bool = True,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Run globalized Newton and reconcile state-dependent active sets.

    Supplying the promoted-mask and shadowed-map callbacks wraps the existing
    smooth Newton solve in a fixed-trip active-set iteration.  Each inner solve
    holds its composed mask fixed, then the outer iteration re-evaluates that
    mask at the inner solution and measures the live residual.  Convergence
    requires both the registered residual tolerance and zero changed mask
    cells.  A repeated mask receives one midpoint-damped transition; another
    previously seen mask terminates with a named cycle result.
    ``stop_on_active_set_stagnation`` ends the outer iteration when consecutive
    trips have bit-identical live residuals and the later trip changes no mask
    cells.  It can be disabled to reproduce the full bounded loop for diagnostic
    comparisons.
    ``stream_active_set`` emits one flushed host line for every executed outer
    trip; it is disabled by default and does not alter any solver carry.
    """
    if active_set_steps <= 0:
        raise ValueError("active_set_steps must be positive")
    carry_shadows = promoted_shadow_mask_fn is not None
    if carry_shadows != (shadowed_map_fn is not None):
        raise ValueError("promoted shadow masks require a shadowed map")
    if carry_shadows and shadow_mask_fn is None:
        raise ValueError("promoted shadow masks require an initial shadow mask")
    options = {
        "newton_steps": newton_steps,
        "gmres_iterations": gmres_iterations,
        "warmup": warmup,
        "relaxation": relaxation,
        "step_cap": step_cap,
        "krylov_condition_limit": krylov_condition_limit,
        "convergence_tolerance": convergence_tolerance,
        "admissibility_fn": admissibility_fn,
        "previous_admitted_state": previous_admitted_state,
        "precision": precision,
    }
    if not carry_shadows or admissibility_fn is not None:
        return _newton_krylov_inner(
            map_fn,
            initial,
            shadow_mask_fn=shadow_mask_fn,
            promoted_shadow_mask_fn=promoted_shadow_mask_fn,
            shadowed_map_fn=shadowed_map_fn,
            **options,
        )
    return _active_set_newton_krylov(
        initial,
        newton_steps=newton_steps,
        gmres_iterations=gmres_iterations,
        warmup=warmup,
        relaxation=relaxation,
        step_cap=step_cap,
        krylov_condition_limit=krylov_condition_limit,
        convergence_tolerance=convergence_tolerance,
        shadow_mask_fn=shadow_mask_fn,
        promoted_shadow_mask_fn=promoted_shadow_mask_fn,
        shadowed_map_fn=shadowed_map_fn,
        active_set_steps=active_set_steps,
        stream_active_set=stream_active_set,
        stop_on_active_set_stagnation=stop_on_active_set_stagnation,
        precision=precision,
    )


def kink_aware_newton_krylov(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    strategy: Literal["clarke", "nonmonotone", "surface_restricted", "damped_hybrid"],
    newton_steps: int,
    gmres_iterations: int = 8,
    warmup: int = 8,
    relaxation: float = 0.5,
    step_cap: float = 10.0,
    krylov_condition_limit: float = _PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
    surface_fn: Callable[[jax.Array], jax.Array] | None = None,
    admissibility_fn: Callable[[jax.Array], jax.Array] | None = None,
    nonmonotone_allowance: float = 0.05,
    hybrid_weight: float = 1.0 / 1.766,
    hybrid_schedule: Literal["fixed", "residual_release"] = "fixed",
    hybrid_final_weight: float = 1.0,
    hybrid_release_residual: float = 1.0e-4,
    precision: Precision | str = Precision.AUTOMATIC,
) -> KinkAwareResult:
    """Newton--Krylov with an explicit derivative-hand-off treatment.

    The four policies alter only proposal acceptance; ``map_fn`` is never
    smoothed or modified.  ``clarke`` averages exact tangents immediately on
    either side of a detected crossing.  ``nonmonotone`` selects the longest
    fixed-ladder proposal admitted by the recent residual envelope;
    when ``admissibility_fn`` is supplied, a trial must also have a finite map
    evaluation and make that predicate true before it can be selected. The
    predicate may return a boolean directly or a qualification result carrying
    an ``admitted`` field.
    ``surface_restricted`` shortens a straddling proposal to just beyond the
    detected surface.  ``damped_hybrid`` blends the Newton and relaxed fixed-
    point proposals with an explicit weight.  Its optional residual-release
    schedule interpolates from that initial weight to ``hybrid_final_weight``
    as the accepted residual approaches ``hybrid_release_residual``.

    The surface callback returns a signed scalar whose zero set is the hand-
    off.  It is required for the two surface-aware policies.  This function is
    additive: the established accelerators and their trace accounting are not
    called or altered.
    """
    strategies = {
        "clarke",
        "nonmonotone",
        "surface_restricted",
        "damped_hybrid",
    }
    if strategy not in strategies:
        raise ValueError(f"unknown kink-aware strategy: {strategy!r}")
    if strategy in {"clarke", "surface_restricted"} and surface_fn is None:
        raise ValueError(f"{strategy!r} requires surface_fn")
    if admissibility_fn is not None and strategy != "nonmonotone":
        raise ValueError("admissibility_fn requires the nonmonotone strategy")
    if hybrid_schedule not in {"fixed", "residual_release"}:
        raise ValueError(f"unknown hybrid schedule: {hybrid_schedule!r}")

    initial = _solver_state(initial, precision)
    trace_length = warmup + newton_steps

    def warm_body(index, carry):
        state, trace, amplification = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        candidate = state + relaxation * (mapped - state)
        amplification = _observe_increment(amplification, state, candidate, True)
        return candidate, trace, amplification

    state, trace, amplification = jax.lax.fori_loop(
        0,
        warmup,
        warm_body,
        (
            initial,
            jnp.full(trace_length, jnp.nan, dtype=initial.dtype),
            _initial_amplification_state(initial.dtype),
        ),
    )

    def krylov_step(tangent, residual_vector, nonlinear_residual, condition_baseline):
        return _qualified_krylov_step(
            lambda vector: vector - tangent(vector),
            residual_vector,
            nonlinear_residual,
            gmres_iterations=gmres_iterations,
            condition_ratio_limit=krylov_condition_limit,
            preceding_condition_baseline=condition_baseline,
        )

    def bounded_step(step, residual_vector):
        fallback = relaxation * residual_vector
        step = jnp.where(jnp.all(jnp.isfinite(step)), step, fallback)
        cap = step_cap * jnp.max(jnp.abs(fallback))
        norm_step = jnp.max(jnp.abs(step))
        return jnp.where(
            norm_step > cap,
            step * (cap / jnp.maximum(norm_step, 1.0e-300)),
            step,
        )

    def crossing_fraction(state, proposal):
        start = surface_fn(state)

        def bisect(_index, bounds):
            lower, upper = bounds
            middle = 0.5 * (lower + upper)
            value = surface_fn(state + middle * (proposal - state))
            same_side = (value >= 0.0) == (start >= 0.0)
            return (
                jnp.where(same_side, middle, lower),
                jnp.where(same_side, upper, middle),
            )

        lower, upper = jax.lax.fori_loop(
            0,
            16,
            bisect,
            (jnp.asarray(0.0, initial.dtype), jnp.asarray(1.0, initial.dtype)),
        )
        return 0.5 * (lower + upper)

    def newton_body(index, carry):
        (
            state,
            residual,
            trace,
            crossings,
            recent,
            candidate_admissibility,
            accepted_factors,
            qualification,
            amplification,
            conditioning_count,
            maximum_condition,
            condition_baseline,
            effective_newton_fractions,
        ) = carry
        mapped, tangent = jax.linearize(map_fn, state)
        residual_vector = mapped - state
        current_residual = _relative_residual(mapped, state)
        qualified_step = krylov_step(
            tangent, residual_vector, current_residual, condition_baseline
        )
        step_qualification = qualified_step.qualification
        action_accepted = step_qualification == KrylovActionQualification.ACCEPTED
        step = bounded_step(qualified_step.step, residual_vector)
        unconditioned_step = bounded_step(
            qualified_step.unconditioned_step, residual_vector
        )
        proposal = state + step
        conditioning_used = action_accepted & qualified_step.conditioning_applied

        if surface_fn is None:
            crossed = jnp.asarray(False)
        else:
            start_side = surface_fn(state)
            proposal_side = surface_fn(proposal)
            crossed = (start_side * proposal_side <= 0.0) & (
                start_side != proposal_side
            )

        if strategy == "clarke":
            fraction = crossing_fraction(state, proposal)

            def clarke_step(_):
                width = jnp.asarray(1.0e-3, initial.dtype)
                left = state + jnp.maximum(0.0, fraction - width) * step
                right = state + jnp.minimum(1.0, fraction + width) * step
                _, left_tangent = jax.linearize(map_fn, left)
                _, right_tangent = jax.linearize(map_fn, right)

                def average_tangent(vector):
                    return 0.5 * (left_tangent(vector) + right_tangent(vector))

                averaged = _qualified_krylov_step(
                    lambda vector: vector - average_tangent(vector),
                    residual_vector,
                    current_residual,
                    gmres_iterations=gmres_iterations,
                    condition_ratio_limit=krylov_condition_limit,
                    preceding_condition_baseline=condition_baseline,
                )
                return bounded_step(
                    averaged.step,
                    residual_vector,
                )

            step = jax.lax.cond(crossed, clarke_step, lambda _: step, operand=None)
            proposal = state + step
            promoted = map_fn(proposal)
            accepted_residual = _relative_residual(promoted, proposal)
        elif strategy == "surface_restricted":
            fraction = crossing_fraction(state, proposal)
            restricted = fraction + 0.05 * (1.0 - fraction)
            proposal = jnp.where(crossed, state + restricted * step, proposal)
            promoted = map_fn(proposal)
            accepted_residual = _relative_residual(promoted, proposal)
        elif strategy == "nonmonotone":
            factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=initial.dtype)

            def score(candidate):
                candidate_mapped = map_fn(candidate)
                return _relative_residual(candidate_mapped, candidate)

            envelope = jnp.max(
                jnp.where(jnp.isfinite(recent), recent, current_residual)
            )

            def evaluate_ladder(trial_step):
                candidates = state[None, :] + factors[:, None] * trial_step[None, :]
                scores = jax.lax.map(score, candidates)
                if admissibility_fn is None:
                    caller_admitted = jnp.ones(factors.shape, dtype=jnp.bool_)
                else:

                    def trial_is_admitted(candidate):
                        qualification = admissibility_fn(candidate)
                        if hasattr(qualification, "admitted"):
                            qualification = qualification.admitted
                        return jnp.asarray(qualification, dtype=jnp.bool_)

                    caller_admitted = jax.lax.map(trial_is_admitted, candidates)
                finite_trials = jnp.all(
                    jnp.isfinite(candidates), axis=1
                ) & jnp.isfinite(scores)
                admitted = finite_trials & caller_admitted & action_accepted
                within_envelope = admitted & (
                    scores <= envelope * (1.0 + nonmonotone_allowance)
                )
                first = jnp.argmax(within_envelope)
                best = jnp.argmin(jnp.where(admitted, scores, jnp.inf))
                selected = jnp.where(jnp.any(within_envelope), first, best)
                return candidates, scores, admitted, selected

            if admissibility_fn is None:
                conditioned = evaluate_ladder(step)
                candidates, scores, candidate_admitted, selected = conditioned
                any_admissible = jnp.any(candidate_admitted)
            else:
                unconditioned = evaluate_ladder(unconditioned_step)
                (
                    unconditioned_candidates,
                    unconditioned_scores,
                    unconditioned_admitted,
                    unconditioned_selected,
                ) = unconditioned
                any_unconditioned = jnp.any(unconditioned_admitted)
                conditioned = jax.lax.cond(
                    qualified_step.conditioning_applied & ~any_unconditioned,
                    evaluate_ladder,
                    lambda _trial_step: unconditioned,
                    step,
                )
                (
                    conditioned_candidates,
                    conditioned_scores,
                    conditioned_admitted,
                    _conditioned_selected,
                ) = conditioned
                improving_conditioned = conditioned_admitted & (
                    conditioned_scores <= current_residual
                )
                any_conditioned = jnp.any(improving_conditioned)
                best_conditioned = jnp.argmin(
                    jnp.where(improving_conditioned, conditioned_scores, jnp.inf)
                )
                conditioning_used = (
                    action_accepted
                    & qualified_step.conditioning_applied
                    & ~any_unconditioned
                    & any_conditioned
                )
                candidates = jnp.where(
                    conditioning_used,
                    conditioned_candidates,
                    unconditioned_candidates,
                )
                scores = jnp.where(
                    conditioning_used,
                    conditioned_scores,
                    unconditioned_scores,
                )
                candidate_admitted = jnp.where(
                    conditioning_used,
                    improving_conditioned,
                    unconditioned_admitted,
                )
                selected = jnp.where(
                    conditioning_used, best_conditioned, unconditioned_selected
                )
                any_admissible = any_unconditioned | conditioning_used
            proposal = jnp.where(any_admissible, candidates[selected], state)
            accepted_residual = jnp.where(
                any_admissible, scores[selected], current_residual
            )
            candidate_admissibility = candidate_admissibility.at[index].set(
                candidate_admitted[:_RECORDED_BACKTRACKING_FACTOR_COUNT]
            )
            accepted_factors = accepted_factors.at[index].set(
                jnp.where(any_admissible, factors[selected], 0.0)
            )
            selected_step = jnp.where(conditioning_used, step, unconditioned_step)
            raw_step_norm = jnp.linalg.norm(qualified_step.unconditioned_step)
            effective_newton_fraction = jnp.where(
                any_admissible & (raw_step_norm > 0.0),
                factors[selected]
                * jnp.linalg.norm(selected_step)
                / jnp.maximum(raw_step_norm, jnp.finfo(initial.dtype).tiny),
                0.0,
            )
            effective_newton_fractions = effective_newton_fractions.at[index].set(
                effective_newton_fraction
            )
        else:
            if hybrid_schedule == "fixed":
                weight = hybrid_weight
            else:
                released = jnp.clip(
                    hybrid_release_residual / jnp.maximum(current_residual, 1.0e-300),
                    0.0,
                    1.0,
                )
                weight = hybrid_weight + released * (
                    hybrid_final_weight - hybrid_weight
                )
            proposal = (
                state + weight * step + (1.0 - weight) * relaxation * residual_vector
            )
            promoted = map_fn(proposal)
            accepted_residual = _relative_residual(promoted, proposal)

        proposal = jnp.where(action_accepted, proposal, state)
        accepted_residual = jnp.where(
            action_accepted, accepted_residual, current_residual
        )
        amplification = _observe_increment(
            amplification, state, proposal, action_accepted
        )

        trace = trace.at[warmup + index].set(accepted_residual)
        crossings = crossings.at[index].set(crossed)
        recent = recent.at[jnp.mod(index, recent.size)].set(accepted_residual)
        prior_failed = (qualification != KrylovActionQualification.NOT_APPLICABLE) & (
            qualification != KrylovActionQualification.ACCEPTED
        )
        qualification = jnp.where(prior_failed, qualification, step_qualification)
        conditioning_count = conditioning_count + jnp.asarray(
            conditioning_used, dtype=jnp.int32
        )
        maximum_condition = jnp.maximum(
            maximum_condition, qualified_step.projected_condition
        )
        return (
            proposal,
            accepted_residual,
            trace,
            crossings,
            recent,
            candidate_admissibility,
            accepted_factors,
            qualification,
            amplification,
            conditioning_count,
            maximum_condition,
            qualified_step.condition_baseline,
            effective_newton_fractions,
        )

    initial_residual = jnp.where(
        warmup > 0, trace[jnp.maximum(warmup - 1, 0)], jnp.asarray(jnp.inf)
    )
    (
        state,
        residual,
        trace,
        crossings,
        _recent,
        candidate_admissibility,
        accepted_factors,
        qualification,
        amplification,
        conditioning_count,
        maximum_condition,
        _condition_baseline,
        effective_newton_fractions,
    ) = jax.lax.fori_loop(
        0,
        newton_steps,
        newton_body,
        (
            state,
            initial_residual,
            trace,
            jnp.zeros(newton_steps, dtype=jnp.bool_),
            jnp.full(len(_BACKTRACKING_FACTORS), jnp.nan, dtype=initial.dtype),
            jnp.zeros(
                (newton_steps, _RECORDED_BACKTRACKING_FACTOR_COUNT),
                dtype=jnp.bool_,
            ),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.asarray(KrylovActionQualification.NOT_APPLICABLE, dtype=jnp.int32),
            amplification,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=initial.dtype),
            jnp.asarray(jnp.nan, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
        ),
    )
    return KinkAwareResult(
        state,
        residual,
        trace,
        crossings,
        candidate_admissibility,
        accepted_factors,
        qualification,
        _amplification_result(amplification, qualification),
        conditioning_count,
        maximum_condition,
        effective_newton_fractions,
    )
