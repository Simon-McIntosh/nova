"""One runtime-argument dense Newton correction for a moment map."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from scripts.prototypes.fine_grid_convergence.handoff import map_moments


DIRECTION_BLOCK = 16
BACKTRACKING_STEPS = 20


def residual(state, operands, scale):
    """Scaled moment residual with every carrier value passed at runtime."""
    moments = state.reshape((-1, 3)) * scale
    mapped = map_moments(moments, *operands)
    return (mapped.moment_residual / scale).reshape(-1)


def factor(state, operands, scale):
    """Build dense Jacobian columns by vmapped JVP and factor them on device."""
    count = state.size
    blocks = (count + DIRECTION_BLOCK - 1) // DIRECTION_BLOCK
    columns = jnp.zeros((blocks * DIRECTION_BLOCK, count), state.dtype)

    def body(block, packed):
        first = block * DIRECTION_BLOCK
        indices = first + jnp.arange(DIRECTION_BLOCK)
        directions = (indices[:, None] == jnp.arange(count)[None]).astype(state.dtype)

        def column(direction):
            _, tangent = jax.jvp(
                lambda probe: residual(probe, operands, scale),
                (state,),
                (direction,),
            )
            return tangent

        products = jax.vmap(column)(directions)
        return jax.lax.dynamic_update_slice(packed, products, (first, 0))

    columns = jax.lax.fori_loop(0, blocks, body, columns)
    matrix = columns[:count].T
    return jsp_linalg.lu_factor(matrix)


def advance(state, operands, scale, lu, pivots, refactor):
    """Replace the state-capturing linearize correction with one jitted step."""
    current = residual(state, operands, scale)
    magnitude = jnp.max(jnp.abs(current))
    lu, pivots = jax.lax.cond(
        refactor,
        lambda _: factor(state, operands, scale),
        lambda _: (lu, pivots),
        operand=None,
    )
    step = jsp_linalg.lu_solve((lu, pivots), -current)

    def condition(carry):
        _, accepted, attempts, _ = carry
        return (~accepted) & (attempts < BACKTRACKING_STEPS)

    def body(carry):
        trial, accepted, attempts, chosen = carry
        proposed = residual(state + trial * step, operands, scale)
        take = jnp.max(jnp.abs(proposed)) < magnitude
        return (
            trial / 2,
            accepted | take,
            attempts + 1,
            jnp.where(take, trial, chosen),
        )

    _, _, _, chosen = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(1.0, state.dtype),
            jnp.asarray(False),
            jnp.int32(0),
            jnp.asarray(0.0, state.dtype),
        ),
    )
    updated = state + chosen * step
    end = residual(updated, operands, scale)
    return updated, jnp.max(jnp.abs(end)), lu, pivots, chosen
