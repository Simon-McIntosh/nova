"""The traced forward Grad-Shafranov map."""

from __future__ import annotations

from typing import NamedTuple
from enum import IntEnum

import jax
import jax.numpy as jnp


def moment_residual(
    moments, operands, cache, roots, convention, policy, profile, quadrature
):
    """Return m − book(read(exterior + G m)) in physical cell moments."""
    from nova.equilibrium import support_read

    support, booking = support_read.read_and_book(
        operands,
        cache,
        tuple(moments[:, index] for index in range(3)),
        roots,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        axis_sign=-convention.sigma,
    )
    booked = jnp.stack(booking[:3], axis=-1)
    return moments - booked, (support, booking)


class MomentTermination(IntEnum):
    """Device termination of the moment equation."""

    CONVERGED = 0
    NONFINITE = 1
    LINE_SEARCH_FAILED = 2
    ITERATION_LIMIT = 3


class MomentSolution(NamedTuple):
    """Cell-moment root and fixed-capacity accepted-step telemetry."""

    moments: jax.Array
    residual: jax.Array
    converged: jax.Array
    iterations: jax.Array
    residual_history: jax.Array
    moment_history: jax.Array
    accepted: jax.Array
    linear_residual_history: jax.Array
    step_factors: jax.Array
    termination_reason: jax.Array


def dense_jacobian(equation, state, *, column_block=128):
    """Assemble a dense jacfwd matrix with bounded tangent intermediates.

    Each device-loop trip differentiates a coordinate block, retaining the
    complete Jacobian for LU while bounding membership and booking tangents
    by the block width. Out-of-range coordinates in the last block are dropped.
    """
    if state.ndim != 1 or column_block < 1:
        raise ValueError("dense Jacobian needs a vector and positive column block")
    width = min(column_block, state.size)
    groups = (state.size + width - 1) // width

    def columns(group):
        indices = group * width + jnp.arange(width)

        def displaced(displacement):
            return equation(state.at[indices].add(displacement, mode="drop"))

        return jax.jacfwd(displaced)(jnp.zeros(width, dtype=state.dtype))

    blocks = jax.lax.map(columns, jnp.arange(groups))
    return jnp.transpose(blocks, (1, 0, 2)).reshape((state.size, -1))[:, : state.size]


def newton_moments(
    initial,
    operands,
    cache,
    roots,
    convention,
    policy,
    profile,
    quadrature,
    **options,
):
    """Keep scalar Krylov arithmetic unchanged when mapping over solves.

    Closure conversion lifts every traced operand before the sequential
    batching boundary, including member-varying profile and exterior data.
    The device loop then runs exactly the same scalar solve for each member.
    """
    from jax.custom_batching import sequential_vmap

    def solve(state):
        return _newton_moments(
            state,
            operands,
            cache,
            roots,
            convention,
            policy,
            profile,
            quadrature,
            **options,
        )

    converted, arguments = jax.closure_convert(solve, initial)
    return sequential_vmap(converted)(initial, *arguments)


def _newton_moments(
    initial,
    operands,
    cache,
    roots,
    convention,
    policy,
    profile,
    quadrature,
    *,
    method="krylov",
    steps=20,
    tolerance=1e-12,
    krylov_size=30,
):
    """Solve the scaled moment equation with device LU or JVP-based GMRES.

    Geometry and coupling remain explicit operands. One scan carries Newton
    steps; a bounded backtracking loop accepts only finite residual decreases.
    Moment scales convert first moments to current times cell pitch without
    changing the physical equation. The final residual is the equation's
    scaled sup norm, including cells outside the selected support.
    """
    import jax.scipy.linalg as dense
    from jax.scipy.sparse.linalg import gmres

    if method not in {"dense", "krylov"}:
        raise ValueError("moment Newton method must be dense or krylov")
    if steps < 1 or krylov_size < 1 or tolerance <= 0:
        raise ValueError("moment Newton budgets and tolerance must be positive")
    pitch = operands.geometry.pitch
    current_scale = jnp.maximum(
        jnp.max(jnp.abs(initial[:, 0])), jnp.finfo(initial.dtype).tiny
    )
    scale = current_scale * jnp.stack((jnp.ones_like(pitch), pitch, pitch), axis=-1)

    def equation(value):
        moments = value.reshape(initial.shape) * scale
        residual, _ = moment_residual(
            moments, operands, cache, roots, convention, policy, profile, quadrature
        )
        return (residual / scale).reshape(-1)

    seed = (initial / scale).reshape(-1)
    residual = equation(seed)
    norm = jnp.max(jnp.abs(residual))

    def advance(carry):
        state, residual, norm, count = carry
        if method == "dense":
            jacobian = dense_jacobian(equation, state)
            delta = dense.lu_solve(dense.lu_factor(jacobian), -residual)
            linear_error = jnp.max(jnp.abs(jacobian @ delta + residual))
        else:

            def product(direction):
                return jax.jvp(equation, (state,), (direction,))[1]

            delta, _ = gmres(
                product,
                -residual,
                tol=1e-10,
                atol=1e-14,
                restart=min(krylov_size, state.size),
                maxiter=2,
                solve_method="batched",
            )
            linear_error = jnp.max(jnp.abs(product(delta) + residual))

        def search_condition(trial):
            index, _value, _residual, _norm, accepted = trial
            return (index < 16) & ~accepted

        def search(trial):
            index, _value, _residual, _norm, _accepted = trial
            factor = jnp.exp2(-index.astype(state.dtype))
            candidate = state + factor * delta
            result = equation(candidate)
            error = jnp.max(jnp.abs(result))
            accepted = jnp.isfinite(error) & (error <= (1 - 1e-4 * factor) * norm)
            return index + 1, candidate, result, error, accepted

        attempts, candidate, result, error, accepted = jax.lax.while_loop(
            search_condition,
            search,
            (jnp.asarray(0), state, residual, norm, jnp.asarray(False)),
        )
        state = jnp.where(accepted, candidate, state)
        residual = jnp.where(accepted, result, residual)
        norm = jnp.where(accepted, error, norm)
        factor = jnp.where(accepted, jnp.exp2(1 - attempts.astype(state.dtype)), 0.0)
        return (state, residual, norm, count + accepted), (
            accepted,
            linear_error,
            factor,
        )

    def trip(carry, index):
        state, residual, norm, count = carry
        active = jnp.isfinite(norm) & (norm > tolerance) & (count == index)
        updated, (accepted, linear_error, factor) = jax.lax.cond(
            active,
            advance,
            lambda carry: (
                carry,
                (jnp.asarray(False), jnp.zeros_like(norm), jnp.zeros_like(norm)),
            ),
            carry,
        )
        return updated, (updated[0], updated[2], accepted, linear_error, factor)

    final, history = jax.lax.scan(
        trip, (seed, residual, norm, jnp.asarray(0)), jnp.arange(steps)
    )
    states, errors, accepted, linear_errors, factors = history
    moment_history = (
        jnp.concatenate((seed[None], states)).reshape((-1, *initial.shape)) * scale
    )
    return MomentSolution(
        final[0].reshape(initial.shape) * scale,
        final[2],
        final[2] <= tolerance,
        final[3],
        jnp.concatenate((norm[None], errors)),
        moment_history,
        accepted,
        linear_errors,
        factors,
        jnp.where(
            final[2] <= tolerance,
            int(MomentTermination.CONVERGED),
            jnp.where(
                ~jnp.isfinite(final[2]),
                int(MomentTermination.NONFINITE),
                jnp.where(
                    final[3] < steps,
                    int(MomentTermination.LINE_SEARCH_FAILED),
                    int(MomentTermination.ITERATION_LIMIT),
                ),
            ),
        ),
    )


def apply(
    psi: jax.Array,
    external: jax.Array,
    operator,
    requested_class=None,
    target_current=None,
    shadow=None,
    partition=None,
    moments=None,
) -> jax.Array:
    """Image booked plasma current and exterior flux using operand geometry.

    ``moments`` are physical cell-current moments already booked on a support
    read. They are imaged directly, so the operator reads no support of its
    own and ``shadow``, the residual exclusion, must be supplied with them.
    """
    if moments is not None:
        if shadow is None:
            raise ValueError("booked moments require the residual shadow of their read")
        image = external + operator.current_moment_image(
            operator.coupling_current_moments(moments)
        )
        return operator._exclude_shadow_residual(psi, image, requested_class, shadow)
    internal = (
        operator.internal(psi, requested_class, target_current)
        if partition is None
        else operator._internal_on_partition(psi, partition, target_current)
    )
    image = external + internal
    return operator._exclude_shadow_residual(psi, image, requested_class, shadow=shadow)
