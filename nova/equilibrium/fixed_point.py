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
  (no early exit).  Its trace stride remains ``2 + gmres_iterations``; an
  initial action probe and the achieved linear residual qualify the solve
  before its step can be promoted.

All three are ``jit``-safe and ``vmap``-safe (fixed shapes, no data-dependent
control flow); map a leading batch axis with ``jax.vmap`` over the initial
state and any batched map parameters.  The free-boundary maps have multiple
fixed points (the outboard-corner attractor), so the BASIN is guarded by the
caller seeding the topology read at the current centroid (the axis-seed pin);
the accelerators contribute step caps, never basin logic.

The state is a flat 1-D array — the concatenated flux vector both solve maps
already carry.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from nova.jax.config import Precision, resolve_precision

__all__ = [
    "FixedPointResult",
    "KinkAwareResult",
    "KrylovActionQualification",
    "anderson",
    "kink_aware_newton_krylov",
    "newton_krylov",
    "picard",
]


class KrylovActionQualification(IntEnum):
    """Host-readable reason a Newton--Krylov linear action was refused."""

    NOT_APPLICABLE = 0
    ACCEPTED = 1
    NONFINITE_LINEAR_ACTION = 2
    NONSUCCESSFUL_GMRES_STATUS = 3
    NONFINITE_ACHIEVED_LINEAR_RESIDUAL = 4
    ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL = 5


class FixedPointResult(NamedTuple):
    """Fixed-shape result of a fixed-point solve.

    ``trace`` holds one entry per map evaluation: the relative sup-norm
    residual where the scheme measured one, NaN where the evaluation was a
    Newton tangent pass — so ladder plots of different schemes share one
    x-axis.  ``residual`` is the residual at the last measured evaluation.
    ``krylov_action_qualification`` names the first refused linear-action
    condition, reports ``ACCEPTED`` when every Newton action passed, and is
    ``NOT_APPLICABLE`` for the non-Krylov schemes.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array
    krylov_action_qualification: jax.Array | int = (
        KrylovActionQualification.NOT_APPLICABLE
    )


class KinkAwareResult(NamedTuple):
    """Result of an explicitly selected derivative-hand-off policy.

    ``trace`` records the residual at each relaxed warmup state and each
    accepted nonlinear state.  ``crossings`` identifies nonlinear steps whose
    unconstrained Newton proposal straddled the caller's detected surface.
    ``candidate_admissibility`` records which of the four nonmonotone
    backtracking factors had finite evaluations and passed the caller's
    predicate.  ``accepted_factors`` is zero when no trial was selected.
    ``krylov_action_qualification`` retains the first refused linear action.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array
    crossings: jax.Array
    candidate_admissibility: jax.Array
    accepted_factors: jax.Array
    krylov_action_qualification: jax.Array | int


class _QualifiedKrylovStep(NamedTuple):
    """One linear solve and its fail-closed qualification."""

    step: jax.Array
    qualification: jax.Array


def _solver_state(initial: jax.Array, precision: Precision | str) -> jax.Array:
    """Cast the state before tracing under the general-solver policy."""
    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    return jnp.asarray(initial, dtype=dtype)


def _relative_residual(mapped: jax.Array, state: jax.Array) -> jax.Array:
    """Relative sup-norm fixed-point residual ``max|g−x| / max|g|``."""
    return jnp.max(jnp.abs(mapped - state)) / jnp.maximum(
        jnp.max(jnp.abs(mapped)), 1.0e-30
    )


def _qualified_krylov_step(
    linear_action: Callable[[jax.Array], jax.Array],
    residual_vector: jax.Array,
    nonlinear_residual: jax.Array,
    *,
    gmres_iterations: int,
) -> _QualifiedKrylovStep:
    """Solve one Krylov system and apply the shared action qualification."""
    probe_scale = jnp.maximum(jnp.max(jnp.abs(residual_vector)), 1.0e-300)
    probe = jnp.where(
        probe_scale > 1.0e-300,
        residual_vector / probe_scale,
        jnp.ones_like(residual_vector) / jnp.sqrt(residual_vector.size),
    )
    finite_linear_action = jnp.all(jnp.isfinite(linear_action(probe)))

    step, info = jax.scipy.sparse.linalg.gmres(
        linear_action,
        residual_vector,
        maxiter=gmres_iterations,
        restart=gmres_iterations,
        solve_method="batched",
    )
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
    return _QualifiedKrylovStep(step=step, qualification=qualification)


def picard(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    evaluations: int,
    relaxation: float = 0.5,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Relaxed Picard iteration with per-evaluation residual accounting."""
    initial = _solver_state(initial, precision)

    def body(index, carry):
        state, trace = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        return state + relaxation * (mapped - state), trace

    state, trace = jax.lax.fori_loop(
        0,
        evaluations,
        body,
        (initial, jnp.full(evaluations, jnp.nan, dtype=initial.dtype)),
    )
    return FixedPointResult(state, trace[evaluations - 1], trace)


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

    def body(index, carry):
        state, dx, df, f_prev, x_prev, norm_prev, trace = carry
        mapped = map_fn(state)
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
        return state_next, dx, df, f, state, norm_f, trace

    init = (
        initial,
        jnp.zeros((n_flat, depth), dtype=initial.dtype),
        jnp.zeros((n_flat, depth), dtype=initial.dtype),
        jnp.zeros(n_flat, dtype=initial.dtype),
        initial,
        jnp.asarray(jnp.inf, dtype=initial.dtype),
        jnp.full(evaluations, jnp.nan, dtype=initial.dtype),
    )
    state, *_, trace = jax.lax.fori_loop(0, evaluations, body, init)
    return FixedPointResult(state, trace[evaluations - 1], trace)


def newton_krylov(
    map_fn: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    newton_steps: int,
    gmres_iterations: int = 8,
    warmup: int = 8,
    relaxation: float = 0.5,
    step_cap: float = 10.0,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FixedPointResult:
    """Exact-tangent Jacobian-free Newton–Krylov on the fixed-point residual.

    After ``warmup`` relaxed Picard sweeps (the fixed-shape transient), each
    Newton step linearises the map once with the exact ``jax.linearize``
    tangent and solves ``(I − J) s = f`` with a ``gmres_iterations``-step
    fixed-shape batched GMRES — no early exit, so the whole solve ``vmap``s
    unchanged.  A step is refused when the initial linear action is non-finite,
    GMRES reports a non-successful status, the achieved linear residual is
    non-finite, or an exactly-zero step carries a material nonlinear residual.
    A qualified step is capped at ``step_cap`` × the relaxed step, bounding
    excursions while the current-centroid pin holds the basin.  The trace
    retains its ``2 + gmres_iterations`` stride (one linearisation value,
    tangent slots, and one promotion read); the qualification actions do not
    add measured nonlinear-map entries.
    """
    initial = _solver_state(initial, precision)
    stride = 2 + gmres_iterations
    trace_length = warmup + newton_steps * stride

    def warm_body(index, carry):
        state, trace = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        return state + relaxation * (mapped - state), trace

    state, trace = jax.lax.fori_loop(
        0,
        warmup,
        warm_body,
        (initial, jnp.full(trace_length, jnp.nan, dtype=initial.dtype)),
    )

    def newton_body(index, carry):
        state, residual, trace, qualification = carry
        mapped, tangent = jax.linearize(map_fn, state)
        f = mapped - state
        base = warmup + index * stride
        nonlinear_residual = _relative_residual(mapped, state)
        trace = trace.at[base].set(nonlinear_residual)

        def linear_action(vector):
            return vector - tangent(vector)

        qualified_step = _qualified_krylov_step(
            linear_action,
            f,
            nonlinear_residual,
            gmres_iterations=gmres_iterations,
        )
        step = qualified_step.step
        step_qualification = qualified_step.qualification
        norm_step = jnp.max(jnp.abs(step))
        accepted = step_qualification == KrylovActionQualification.ACCEPTED

        cap = step_cap * jnp.max(jnp.abs(relaxation * f))
        step = jnp.where(
            norm_step > cap, step * (cap / jnp.maximum(norm_step, 1.0e-300)), step
        )
        state = jnp.where(accepted, state + step, state)
        promoted = map_fn(state)
        residual = _relative_residual(promoted, state)
        trace = trace.at[base + stride - 1].set(residual)
        prior_failed = (qualification != KrylovActionQualification.NOT_APPLICABLE) & (
            qualification != KrylovActionQualification.ACCEPTED
        )
        qualification = jnp.where(prior_failed, qualification, step_qualification)
        return state, residual, trace, qualification

    state, residual, trace, qualification = jax.lax.fori_loop(
        0,
        newton_steps,
        newton_body,
        (
            state,
            trace[jnp.maximum(warmup - 1, 0)],
            trace,
            jnp.asarray(KrylovActionQualification.NOT_APPLICABLE, dtype=jnp.int32),
        ),
    )
    return FixedPointResult(state, residual, trace, qualification)


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
    of four backtracking proposals admitted by the recent residual envelope;
    when ``admissibility_fn`` is supplied, a trial must also have a finite map
    evaluation and make that predicate true before it can be selected.
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
        state, trace = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        return state + relaxation * (mapped - state), trace

    state, trace = jax.lax.fori_loop(
        0,
        warmup,
        warm_body,
        (initial, jnp.full(trace_length, jnp.nan, dtype=initial.dtype)),
    )

    def krylov_step(tangent, residual_vector, nonlinear_residual):
        return _qualified_krylov_step(
            lambda vector: vector - tangent(vector),
            residual_vector,
            nonlinear_residual,
            gmres_iterations=gmres_iterations,
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
        ) = carry
        mapped, tangent = jax.linearize(map_fn, state)
        residual_vector = mapped - state
        current_residual = _relative_residual(mapped, state)
        qualified_step = krylov_step(tangent, residual_vector, current_residual)
        step_qualification = qualified_step.qualification
        action_accepted = step_qualification == KrylovActionQualification.ACCEPTED
        step = bounded_step(qualified_step.step, residual_vector)
        proposal = state + step

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
            factors = jnp.asarray((1.0, 0.5, 0.25, 0.125), dtype=initial.dtype)
            trial_step = jnp.where(action_accepted, step, jnp.zeros_like(step))
            candidates = state[None, :] + factors[:, None] * trial_step[None, :]

            def score(candidate):
                candidate_mapped = map_fn(candidate)
                return _relative_residual(candidate_mapped, candidate)

            scores = jax.lax.map(score, candidates)
            if admissibility_fn is None:
                caller_admitted = jnp.ones(factors.shape, dtype=jnp.bool_)
            else:
                caller_admitted = jax.lax.map(admissibility_fn, candidates).astype(
                    jnp.bool_
                )
            finite_trials = jnp.all(jnp.isfinite(candidates), axis=1) & jnp.isfinite(
                scores
            )
            candidate_admitted = finite_trials & caller_admitted & action_accepted
            envelope = jnp.max(
                jnp.where(jnp.isfinite(recent), recent, current_residual)
            )
            within_envelope = candidate_admitted & (
                scores <= envelope * (1.0 + nonmonotone_allowance)
            )
            first = jnp.argmax(within_envelope)
            best_admissible = jnp.argmin(jnp.where(candidate_admitted, scores, jnp.inf))
            selected = jnp.where(jnp.any(within_envelope), first, best_admissible)
            any_admissible = jnp.any(candidate_admitted)
            proposal = jnp.where(any_admissible, candidates[selected], state)
            accepted_residual = jnp.where(
                any_admissible, scores[selected], current_residual
            )
            candidate_admissibility = candidate_admissibility.at[index].set(
                candidate_admitted
            )
            accepted_factors = accepted_factors.at[index].set(
                jnp.where(any_admissible, factors[selected], 0.0)
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

        trace = trace.at[warmup + index].set(accepted_residual)
        crossings = crossings.at[index].set(crossed)
        recent = recent.at[jnp.mod(index, recent.size)].set(accepted_residual)
        prior_failed = (qualification != KrylovActionQualification.NOT_APPLICABLE) & (
            qualification != KrylovActionQualification.ACCEPTED
        )
        qualification = jnp.where(prior_failed, qualification, step_qualification)
        return (
            proposal,
            accepted_residual,
            trace,
            crossings,
            recent,
            candidate_admissibility,
            accepted_factors,
            qualification,
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
    ) = jax.lax.fori_loop(
        0,
        newton_steps,
        newton_body,
        (
            state,
            initial_residual,
            trace,
            jnp.zeros(newton_steps, dtype=jnp.bool_),
            jnp.full(4, jnp.nan, dtype=initial.dtype),
            jnp.zeros((newton_steps, 4), dtype=jnp.bool_),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.asarray(KrylovActionQualification.NOT_APPLICABLE, dtype=jnp.int32),
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
    )
