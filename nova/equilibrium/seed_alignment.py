"""Outcome-blind diagnostics for choosing forward-equilibrium seeds."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def residual_action_amplification(
    map_fn: Callable[[jax.Array], jax.Array], seed: jax.Array
) -> jax.Array:
    r"""Measure how strongly the seed residual occupies a weak action direction.

    For residual ``r = F(seed) - seed`` and local action
    ``A = I - J_F(seed)``, the returned scalar is

    ``||r||_2 / ||A r||_2``.

    A larger value means the residual is more closely aligned with a direction
    on which the local fixed-point action is weak.  The diagnostic evaluates
    only the supplied seed, map, and tangent action.  It does not require a
    terminal state, reference state, or solve outcome.

    An exact fixed point returns zero.  A nonzero residual with an exactly zero
    action returns infinity, preserving the mathematically singular ordering.
    """

    seed = jnp.asarray(seed)
    mapped, tangent_action = jax.linearize(map_fn, seed)
    residual = mapped - seed
    tangent_residual = tangent_action(residual)
    action_residual = residual - tangent_residual
    residual_norm = jnp.linalg.norm(residual)
    action_norm = jnp.linalg.norm(action_residual)
    amplification = jnp.where(
        action_norm > 0,
        residual_norm / action_norm,
        jnp.asarray(jnp.inf, dtype=residual_norm.dtype),
    )
    return jnp.where(residual_norm > 0, amplification, jnp.zeros_like(amplification))
