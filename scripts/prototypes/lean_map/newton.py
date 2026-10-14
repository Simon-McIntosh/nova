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


def dense_newton(residual, start, scale, *, steps=8, tolerance=1e-12):
    """Scan one compiled Newton site with a fixed four-trial line search."""
    size = start.size
    alphas = jnp.asarray((1.0, 0.5, 0.25, 0.125))

    def one_step(state, _):
        moments, accepted, done = state
        value = residual(moments)
        norm = _scaled_norm(value, scale)

        def advance(_):
            matrix = jax.jacfwd(residual)(moments).reshape((size, size))
            correction = jnp.linalg.solve(matrix, -value.reshape(-1)).reshape(
                moments.shape
            )
            candidates = moments[None] + alphas[:, None, None] * correction
            candidate_values = jax.vmap(residual)(candidates)
            norms = jax.vmap(lambda row: _scaled_norm(row, scale))(candidate_values)
            admissible = jnp.isfinite(norms) & (norms < norm)
            scores = jnp.where(admissible, norms, jnp.inf)
            chosen = jnp.argmin(scores)
            improved = jnp.any(admissible)
            updated = jnp.where(improved, candidates[chosen], moments)
            terminal = jnp.where(improved, norms[chosen], norm)
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
    matrix = jax.jacfwd(residual)(moments).reshape((moments.size, moments.size))
    return lu_factor(matrix)


def cached_factor_corrections(residual, start, factor, scale, *, steps=3):
    """Apply a predecessor's factor with live residual checks each correction."""

    def correct(moments, _):
        value = residual(moments)
        correction = lu_solve(factor, -value.reshape(-1)).reshape(moments.shape)
        updated = moments + correction
        return updated, _scaled_norm(residual(updated), scale)

    moments, history = jax.lax.scan(correct, start, None, length=steps)
    terminal = _scaled_norm(residual(moments), scale)
    return NewtonResult(
        moments,
        terminal,
        terminal <= 1e-12,
        jnp.asarray(steps, jnp.int32),
        history,
    )
