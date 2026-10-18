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

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

__all__ = [
    "Factor",
    "SolveResult",
    "VerticalPin",
    "factor",
    "fixed_point",
    "pin_residual",
    "pin_state",
    "pinned_fixed_point",
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
class VerticalPin:
    """One compensating unknown that holds a vertical position at a target.

    The compensating unknown is the last element of the state: the current in
    its spatial sign enters the resolved map through the runtime operands, so
    the field shape is not captured in the traced program.  ``target`` is the
    held current-centroid height in metres and ``pin_scale`` the row's declared
    scale; ``initial`` seeds the unknown.  Target, scale and seed are runtime
    leaves, so one compiled program serves any target and any pin.
    """

    target: jax.Array
    pin_scale: jax.Array
    initial: jax.Array


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


def _damped_scale(residual_fn, state, step, current, operands):
    """Largest dyadic step scale in (0, 1] that reduces the infinity norm."""
    dtype = state.dtype

    def condition(carry):
        _, found, index, _, _ = carry
        return (~found) & (index < BACKTRACKING_STEPS)

    def body(carry):
        scale, found, index, best_scale, best_residual = carry
        candidate_residual = _inf_norm(residual_fn(state + scale * step, operands))
        take = candidate_residual < current
        return (
            scale * jnp.asarray(0.5, dtype),
            found | take,
            index + 1,
            jnp.where(take, scale, best_scale),
            jnp.where(take, candidate_residual, best_residual),
        )

    initial = (
        jnp.asarray(1.0, dtype),
        jnp.asarray(False),
        jnp.int32(0),
        jnp.asarray(0.0, dtype),
        current,
    )
    _, _, _, best_scale, _ = jax.lax.while_loop(condition, body, initial)
    return best_scale


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
        scale = _damped_scale(residual_fn, state, step, _inf_norm(residual), operands)
        state = state + scale * step
        return state, residual_fn(state, operands), corrections + 1

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


def pin_residual(residual_fn, observation, pin):
    """Augment a map with one pin row held by one compensating unknown.

    The augmented state is the base state with the compensating unknown
    appended.  ``residual_fn(extended, operands)`` is the base map, which reads
    the unknown off the state's last element and adds its declared field
    through the runtime operands; ``observation(extended, operands)`` returns
    the scalar the pin holds.  ``pin.target`` and ``pin.pin_scale`` are read
    from the ``pin`` argument at runtime, so the traced program is the same for
    any pin target.
    """
    target = jnp.asarray(pin.target)
    scale = jnp.asarray(pin.pin_scale)

    def augmented(extended, operands):
        base = jnp.reshape(residual_fn(extended, operands), (-1,))
        row = jnp.reshape((observation(extended, operands) - target) / scale, (1,))
        return jnp.concatenate((base, row))

    return augmented


def pin_state(state, pin):
    """Return the base state extended by the pin's initial compensating unknown."""
    flat = jnp.reshape(jnp.asarray(state), (-1,))
    seed = jnp.reshape(jnp.asarray(pin.initial), (1,))
    return jnp.concatenate((flat, seed))


def pinned_fixed_point(
    residual_fn,
    observation,
    state,
    operands,
    pin,
    *,
    factor=None,
    tolerance=DEFAULT_TOLERANCE,
    max_corrections=DEFAULT_MAX_CORRECTIONS,
):
    """Drive a map augmented by one vertical pin to ``tolerance``.

    The pin's basis, target and scale ride in ``pin`` as runtime leaves, so a
    single compiled program pins at any target value and degenerates to the
    free map when the basis carries no leverage.  The terminal state's last
    element is the compensating unknown; ``pin_residual``'s last row is the
    pin's own residual.
    """
    augmented = pin_residual(residual_fn, observation, pin)
    return fixed_point(
        augmented,
        pin_state(state, pin),
        operands,
        factor=factor,
        tolerance=tolerance,
        max_corrections=max_corrections,
    )
