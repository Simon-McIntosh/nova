"""Chord-Newton fixed-point solve over a pure map, batched over states.

The solver owns no physics. It is handed a pure JAX map
``residual_fn(state, operands)`` returning a flat residual and it drives that
residual to a tolerance. Two entry points:

* :func:`factor` builds the dense Jacobian of the map at a state by a vmapped
  ``jax.jvp`` over identity directions and LU-factored it;
* :func:`fixed_point` corrects a state, either cold from a fresh factor at each
  correction with a damped backtracking line search, or as a chord correction
  reusing a supplied factor.

Both are traceable under ``jax.jit`` and ``jax.vmap``: the correction loop and
the backtracking line search are ``lax.while_loop`` constructs, the chord path
unrolls its corrections at trace time, and no constant is captured that scales
with the state.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

__all__ = [
    "Factor",
    "SolveResult",
    "factor",
    "fixed_point",
]

BACKTRACKING_STEPS = 12
DEFAULT_MAX_CORRECTIONS = 25
DEFAULT_TOLERANCE = 1e-12


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Factor:
    """A dense LU factorisation of the map's Jacobian at one state."""

    lu: jax.Array
    pivots: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolveResult:
    """The terminal state of a fixed-point solve and how it was reached.

    ``residual`` is the scaled infinity norm of the map at ``state``,
    ``corrections`` the number of accepted corrections applied, and ``factor``
    the LU factorisation available for a chord reuse at a neighbouring state.
    """

    state: jax.Array
    residual: jax.Array
    converged: jax.Array
    corrections: jax.Array
    factor: Factor


def _inf_norm(vector):
    """Infinity norm of a flat residual, the solver's convergence measure."""
    return jnp.max(jnp.abs(vector))


def factor(residual_fn, state, operands):
    """Lu-factor the dense Jacobian of ``residual_fn`` at ``state``.

    ``residual_fn`` is pure and takes ``(state, operands)``; ``state`` is flat.
    Every Jacobian column is one vmapped ``jax.jvp`` over identity directions,
    so a single traced program builds the whole matrix.
    """
    flat = jnp.asarray(state).reshape(-1)
    directions = jnp.eye(flat.size, dtype=flat.dtype)

    def column(direction):
        _, tangent = jax.jvp(
            lambda probe: residual_fn(probe, operands), (flat,), (direction,)
        )
        return tangent

    columns = jax.vmap(column)(directions)
    lu, pivots = jsp_linalg.lu_factor(columns.T)
    return Factor(lu=lu, pivots=pivots)


class _DampedResult(NamedTuple):
    state: object
    auxiliary: object
    residual: jax.Array
    scale: jax.Array
    evaluations: jax.Array


def _damped_scale(
    evaluate,
    state,
    step,
    current,
    auxiliary,
    *,
    active=True,
    max_steps=BACKTRACKING_STEPS,
    tolerance=0.0,
    keep_incumbent=True,
    backtrack=True,
):
    """Search dyadic updates with one validity flag and mask per lane.

    ``evaluate(state)`` returns the norm, validity and auxiliary read together.
    State and auxiliary pytrees share the leading shape of ``current``; scalar
    norms also work under vmap. Each pending lane stops at its first valid
    reduction (or tolerance), retaining the incumbent if no trial improves it.
    With ``keep_incumbent=False`` the smallest-residual valid trial is retained
    instead, allowing escape from a non-smooth residual's local plateau.
    ``evaluations`` counts whole-batch calls, including companion lanes.
    With ``backtrack=False`` return the full trial and its read even if invalid;
    the caller owns validity and certification of that evaluation.
    """
    dtype = current.dtype
    active = jnp.broadcast_to(jnp.asarray(active, bool), current.shape)

    def choose(mask, trial, retained):
        return jax.tree.map(
            lambda value, prior: jnp.where(
                mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim)),
                value,
                prior,
            ),
            trial,
            retained,
        )

    def condition(carry):
        return jnp.any(carry[6]) & (carry[5] < max_steps)

    def body(carry):
        best, best_aux, best_norm, best_scale, scale, calls, pending, tried = carry
        proposed = jax.tree.map(
            lambda value, change: (
                value
                + scale.reshape(scale.shape + (1,) * (value.ndim - scale.ndim)) * change
            ),
            state,
            step,
        )
        trial = choose(pending, proposed, best)
        norm, valid, trial_aux = evaluate(trial)
        valid = valid & jnp.isfinite(norm)
        better = pending & (
            ~jnp.asarray(backtrack)
            | (valid & ((norm < best_norm) | (~tried & ~jnp.asarray(keep_incumbent))))
        )
        return (
            choose(better, trial, best),
            choose(better, trial_aux, best_aux),
            jnp.where(better, norm, best_norm),
            jnp.where(better, scale, best_scale),
            scale * jnp.asarray(0.5, dtype),
            calls + 1,
            pending & backtrack & (~valid | ((norm >= current) & (norm > tolerance))),
            tried | (pending & valid),
        )

    initial = (
        state,
        auxiliary,
        jnp.where(jnp.isfinite(current), current, jnp.inf),
        jnp.zeros_like(current),
        jnp.ones_like(current),
        jnp.int32(0),
        active,
        jnp.zeros_like(active),
    )
    best, best_aux, best_norm, scale, _, calls, _, _ = jax.lax.while_loop(
        condition,
        body,
        initial,
    )
    return _DampedResult(best, best_aux, best_norm, scale, calls)


def _result(state, residual, corrections, factored, tolerance):
    magnitude = _inf_norm(residual)
    return SolveResult(
        state=state,
        residual=magnitude,
        converged=magnitude <= tolerance,
        corrections=corrections,
        factor=factored,
    )


def _cold(residual_fn, state, operands, tolerance, max_corrections):
    def condition(carry):
        _, residual, corrections = carry
        return (corrections < max_corrections) & (_inf_norm(residual) > tolerance)

    def body(carry):
        state, residual, corrections = carry
        current = factor(residual_fn, state, operands)
        step = jsp_linalg.lu_solve((current.lu, current.pivots), -residual)

        def evaluate(candidate):
            value = residual_fn(candidate, operands)
            norm = _inf_norm(value)
            return norm, jnp.isfinite(norm), value

        trial = _damped_scale(evaluate, state, step, _inf_norm(residual), residual)
        return trial.state, trial.auxiliary, corrections + 1

    initial = (state, residual_fn(state, operands), jnp.int32(0))
    state, residual, corrections = jax.lax.while_loop(condition, body, initial)
    return _result(
        state, residual, corrections, factor(residual_fn, state, operands), tolerance
    )


def _chord(residual_fn, state, operands, supplied, tolerance, max_corrections):
    residual = residual_fn(state, operands)
    for _ in range(max_corrections):
        step = jsp_linalg.lu_solve((supplied.lu, supplied.pivots), -residual)
        state = state + step
        residual = residual_fn(state, operands)
    return _result(state, residual, jnp.int32(max_corrections), supplied, tolerance)


def fixed_point(
    residual_fn,
    state,
    operands,
    *,
    factor=None,
    tolerance=DEFAULT_TOLERANCE,
    max_corrections=DEFAULT_MAX_CORRECTIONS,
):
    """Drive ``residual_fn(state, operands)`` to ``tolerance``.

    With no ``factor`` the solve is cold: each correction rebuilds the Jacobian
    and applies a damped backtracking line search. With a ``factor`` it is a
    chord solve, reusing that factor for ``max_corrections`` corrections.
    """
    flat = jnp.asarray(state).reshape(-1)
    if factor is None:
        return _cold(residual_fn, flat, operands, tolerance, max_corrections)
    return _chord(residual_fn, flat, operands, factor, tolerance, max_corrections)
