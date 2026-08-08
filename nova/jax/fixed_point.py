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
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.jax.config import enable_x64

enable_x64()

__all__ = ["FixedPointResult", "anderson", "newton_krylov", "picard"]


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
) -> FixedPointResult:
    """Relaxed Picard iteration with per-evaluation residual accounting."""

    def body(index, carry):
        state, trace = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        return state + relaxation * (mapped - state), trace

    state, trace = jax.lax.fori_loop(
        0, evaluations, body, (initial, jnp.full(evaluations, jnp.nan))
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
        gram = gram + ridge * (jnp.trace(gram) / depth + 1.0e-30) * jnp.eye(depth)
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
        jnp.zeros((n_flat, depth)),
        jnp.zeros((n_flat, depth)),
        jnp.zeros(n_flat),
        initial,
        jnp.asarray(jnp.inf, dtype=initial.dtype),
        jnp.full(evaluations, jnp.nan),
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
    stride = 2 + gmres_iterations
    trace_length = warmup + newton_steps * stride

    def warm_body(index, carry):
        state, trace = carry
        mapped = map_fn(state)
        trace = trace.at[index].set(_relative_residual(mapped, state))
        return state + relaxation * (mapped - state), trace

    state, trace = jax.lax.fori_loop(
        0, warmup, warm_body, (initial, jnp.full(trace_length, jnp.nan))
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
