"""One runtime-argument dense Newton correction for a moment map."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from nova.equilibrium.constraint import CurrentCentroidConstraint


DIRECTION_BLOCK = 16
BACKTRACKING_STEPS = 20


def without_profile(operands):
    """Keep the callable profile outside the runtime array pytree."""
    return (*operands[:7], *operands[8:])


def residual(
    state,
    operands,
    scale,
    profile,
    component="halfplane",
    density_mode="fitted",
    pin=None,
):
    """Scaled moment residual with carrier arrays passed at runtime."""
    moments_state = state if pin is None else state[:-1]
    moments = moments_state.reshape((-1, 3)) * scale
    if pin is not None:
        active, target, row_scale, field_scale, values_basis, jet_basis = pin
        field_value = state[-1] * field_scale
        values = replace(
            operands[0], exterior=operands[0].exterior + field_value * values_basis
        )
        window = replace(
            operands[2], exterior=operands[2].exterior + field_value * jet_basis
        )
        operands = (values, operands[1], window, *operands[3:])
    mapped = map_moments(
        moments,
        *operands[:7],
        profile,
        *operands[7:],
        component=component,
        density_mode=density_mode,
    )
    base = (mapped.moment_residual / scale).reshape(-1)
    if pin is None:
        return base
    row = jnp.where(
        active,
        (CurrentCentroidConstraint.booked_height(moments) - target) / row_scale,
        state[-1],
    )
    return jnp.concatenate((base, jnp.reshape(row, (1,))))


def factor(
    state,
    operands,
    scale,
    profile,
    component="halfplane",
    density_mode="fitted",
    pin=None,
):
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
                lambda probe: residual(
                    probe, operands, scale, profile, component, density_mode, pin
                ),
                (state,),
                (direction,),
            )
            return tangent

        products = jax.vmap(column)(directions)
        return jax.lax.dynamic_update_slice(packed, products, (first, 0))

    columns = jax.lax.fori_loop(0, blocks, body, columns)
    matrix = columns[:count].T
    return jsp_linalg.lu_factor(matrix)


def advance(
    state,
    operands,
    scale,
    lu,
    pivots,
    refactor,
    profile,
    component="halfplane",
    density_mode="fitted",
    pin=None,
):
    """Replace the state-capturing linearize correction with one jitted step."""
    current = residual(state, operands, scale, profile, component, density_mode, pin)
    magnitude = jnp.max(jnp.abs(current))
    lu, pivots = jax.lax.cond(
        refactor,
        lambda _: factor(state, operands, scale, profile, component, density_mode, pin),
        lambda _: (lu, pivots),
        operand=None,
    )
    step = jsp_linalg.lu_solve((lu, pivots), -current)

    def condition(carry):
        _, accepted, attempts, _ = carry
        return (~accepted) & (attempts < BACKTRACKING_STEPS)

    def body(carry):
        trial, accepted, attempts, chosen = carry
        proposed = residual(
            state + trial * step, operands, scale, profile, component, density_mode, pin
        )
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
    end = residual(updated, operands, scale, profile, component, density_mode, pin)
    return updated, jnp.max(jnp.abs(end)), lu, pivots, chosen
