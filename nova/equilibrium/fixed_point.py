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
  (no early exit), at a cost of ``2 + gmres_iterations`` map evaluations per
  step.

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
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from nova.jax.config import Precision, resolve_precision

__all__ = [
    "FixedPointResult",
    "KinkAwareResult",
    "anderson",
    "kink_aware_newton_krylov",
    "newton_krylov",
    "picard",
]


class FixedPointResult(NamedTuple):
    """Fixed-shape result of a fixed-point solve.

    ``trace`` holds one entry per map evaluation: the relative sup-norm
    residual where the scheme measured one, NaN where the evaluation was a
    Newton tangent pass — so ladder plots of different schemes share one
    x-axis.  ``residual`` is the residual at the last measured evaluation.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array


class KinkAwareResult(NamedTuple):
    """Result of an explicitly selected derivative-hand-off policy.

    ``trace`` records the residual at each relaxed warmup state and each
    accepted nonlinear state.  ``crossings`` identifies nonlinear steps whose
    unconstrained Newton proposal straddled the caller's detected surface.
    """

    state: jax.Array
    residual: jax.Array
    trace: jax.Array
    crossings: jax.Array


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
    unchanged.  A non-finite Krylov step falls back to the relaxed Picard
    step and any step is capped at ``step_cap`` × the relaxed step, bounding
    excursions while the current-centroid pin holds the basin.  Cost per step
    is ``2 + gmres_iterations`` map evaluations (one linearisation value, the
    tangent passes, one promotion read), which is exactly the trace layout.
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
        state, residual, trace = carry
        mapped, tangent = jax.linearize(map_fn, state)
        f = mapped - state
        base = warmup + index * stride
        trace = trace.at[base].set(_relative_residual(mapped, state))

        step, _info = jax.scipy.sparse.linalg.gmres(
            lambda vector: vector - tangent(vector),
            f,
            maxiter=gmres_iterations,
            restart=gmres_iterations,
            solve_method="batched",
        )
        step = jnp.where(jnp.all(jnp.isfinite(step)), step, relaxation * f)
        cap = step_cap * jnp.max(jnp.abs(relaxation * f))
        norm_step = jnp.max(jnp.abs(step))
        step = jnp.where(
            norm_step > cap, step * (cap / jnp.maximum(norm_step, 1.0e-300)), step
        )
        state = state + step
        promoted = map_fn(state)
        residual = _relative_residual(promoted, state)
        trace = trace.at[base + stride - 1].set(residual)
        return state, residual, trace

    state, residual, trace = jax.lax.fori_loop(
        0,
        newton_steps,
        newton_body,
        (state, trace[jnp.maximum(warmup - 1, 0)], trace),
    )
    return FixedPointResult(state, residual, trace)


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
    of four backtracking proposals admitted by the recent residual envelope.
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

    def krylov_step(tangent, residual_vector):
        step, _info = jax.scipy.sparse.linalg.gmres(
            lambda vector: vector - tangent(vector),
            residual_vector,
            maxiter=gmres_iterations,
            restart=gmres_iterations,
            solve_method="batched",
        )
        return step

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
        state, residual, trace, crossings, recent = carry
        mapped, tangent = jax.linearize(map_fn, state)
        residual_vector = mapped - state
        current_residual = _relative_residual(mapped, state)
        step = bounded_step(krylov_step(tangent, residual_vector), residual_vector)
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

                return bounded_step(
                    krylov_step(average_tangent, residual_vector), residual_vector
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
            candidates = state[None, :] + factors[:, None] * step[None, :]

            def score(candidate):
                candidate_mapped = map_fn(candidate)
                return _relative_residual(candidate_mapped, candidate)

            scores = jax.lax.map(score, candidates)
            envelope = jnp.max(
                jnp.where(jnp.isfinite(recent), recent, current_residual)
            )
            admitted = scores <= envelope * (1.0 + nonmonotone_allowance)
            first = jnp.argmax(admitted)
            selected = jnp.where(jnp.any(admitted), first, jnp.argmin(scores))
            proposal = candidates[selected]
            accepted_residual = scores[selected]
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

        trace = trace.at[warmup + index].set(accepted_residual)
        crossings = crossings.at[index].set(crossed)
        recent = recent.at[jnp.mod(index, recent.size)].set(accepted_residual)
        return proposal, accepted_residual, trace, crossings, recent

    initial_residual = jnp.where(
        warmup > 0, trace[jnp.maximum(warmup - 1, 0)], jnp.asarray(jnp.inf)
    )
    state, residual, trace, crossings, _recent = jax.lax.fori_loop(
        0,
        newton_steps,
        newton_body,
        (
            state,
            initial_residual,
            trace,
            jnp.zeros(newton_steps, dtype=jnp.bool_),
            jnp.full(4, jnp.nan, dtype=initial.dtype),
        ),
    )
    return KinkAwareResult(state, residual, trace, crossings)
