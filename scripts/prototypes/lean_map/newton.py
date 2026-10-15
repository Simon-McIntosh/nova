"""Dense forward-mode Newton and reusable warm-slice factor for lean sources."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve


class NewtonResult(NamedTuple):
    moments: jax.Array
    residual: jax.Array
    converged: jax.Array
    accepted_steps: jax.Array
    history: jax.Array


def _scaled_norm(residual, scale):
    return jnp.max(jnp.abs(residual / scale))


def _jacobian(residual, moments):
    """Differentiate columns in bounded batches to cap live tangent storage."""
    _, push = jax.linearize(residual, moments)
    basis = jnp.eye(moments.size, dtype=moments.dtype).reshape(
        (moments.size,) + moments.shape
    )
    columns = jax.lax.map(push, basis, batch_size=8)
    return columns.reshape((moments.size, moments.size)).T


def dense_newton(residual, start, scale, *, steps=8, tolerance=1e-12):
    """Scan one compiled Newton site with a fixed four-trial line search."""
    alphas = jnp.asarray((1.0, 0.5, 0.25, 0.125))

    def one_step(state, _):
        moments, accepted, done = state
        value = residual(moments)
        norm = _scaled_norm(value, scale)

        def advance(_):
            matrix = _jacobian(residual, moments)
            correction = jnp.linalg.solve(matrix, -value.reshape(-1)).reshape(
                moments.shape
            )
            candidates = moments[None] + alphas[:, None, None] * correction
            candidate_values = jax.lax.map(residual, candidates)
            norms = jax.vmap(lambda row: _scaled_norm(row, scale))(candidate_values)
            admissible = jnp.isfinite(norms) & (norms < norm)
            scores = jnp.where(admissible, norms, jnp.inf)
            chosen = jnp.argmin(scores)
            improved = jnp.any(admissible)
            updated = jnp.where(improved, candidates[chosen], moments)
            terminal = _scaled_norm(residual(updated), scale)
            return updated, accepted + improved.astype(jnp.int32), terminal

        updated, count, terminal = jax.lax.cond(
            done | (norm <= tolerance) | ~jnp.isfinite(norm),
            lambda _: (moments, accepted, norm),
            advance,
            operand=None,
        )
        finished = done | (terminal <= tolerance)
        return (updated, count, finished), terminal

    (moments, count, done), history = jax.lax.scan(
        one_step,
        (start, jnp.asarray(0, jnp.int32), jnp.asarray(False)),
        None,
        length=steps,
    )
    terminal = _scaled_norm(residual(moments), scale)
    return NewtonResult(
        moments, terminal, done & (terminal <= tolerance), count, history
    )


def factor_at(residual, moments):
    """Factor the full dense derivative once for a neighbouring slice."""
    matrix = _jacobian(residual, moments)
    return lu_factor(matrix)


def cached_factor_corrections(residual, start, factor, scale, *, steps=5):
    """Use a cached factor with checked trial steps and stop at convergence."""
    alphas = jnp.asarray((1.0, 0.5, 0.25, 0.125))
    initial = _scaled_norm(residual(start), scale)

    def correct(state, _):
        moments, norm, accepted, done = state

        def advance(_):
            value = residual(moments)
            correction = lu_solve(factor, -value.reshape(-1)).reshape(moments.shape)
            candidates = moments[None] + alphas[:, None, None] * correction
            norms = jax.lax.map(
                lambda candidate: _scaled_norm(residual(candidate), scale),
                candidates,
            )
            eligible = jnp.isfinite(norms) & (norms < norm)
            chosen = jnp.argmin(jnp.where(eligible, norms, jnp.inf))
            improved = jnp.any(eligible)
            updated = jnp.where(improved, candidates[chosen], moments)
            terminal = _scaled_norm(residual(updated), scale)
            finished = (terminal <= 1e-12) | ~improved | ~jnp.isfinite(terminal)
            return updated, terminal, accepted + improved.astype(jnp.int32), finished

        next_state = jax.lax.cond(
            done | (norm <= 1e-12),
            lambda _: state,
            advance,
            operand=None,
        )
        return next_state, next_state[1]

    (moments, terminal, accepted, _), history = jax.lax.scan(
        correct,
        (start, initial, jnp.asarray(0, jnp.int32), jnp.asarray(False)),
        None,
        length=steps,
    )
    return NewtonResult(
        moments,
        terminal,
        terminal <= 1e-12,
        accepted,
        history,
    )
