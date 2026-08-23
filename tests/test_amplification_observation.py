"""Advisory observation of geometric increment growth in fixed-point solves."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point
    from nova.equilibrium.fixed_point import (
        AmplificationObservation,
        KrylovActionQualification,
        newton_krylov,
        picard,
    )


def _solve_with_steps(monkeypatch, step_fn, *, steps, terminal_state):
    def prescribed_gmres(_operator, right_hand_side, **_options):
        return step_fn(right_hand_side), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", prescribed_gmres)

    def map_fn(state):
        return jnp.where(
            state[0] < terminal_state,
            jnp.full_like(state, 100.0),
            state,
        )

    return newton_krylov(
        map_fn,
        jnp.zeros(1),
        newton_steps=steps,
        gmres_iterations=1,
        warmup=0,
    )


def _assert_refusal(result, expected):
    assert (
        KrylovActionQualification(int(result.krylov_action_qualification)) is expected
    )
    assert (
        AmplificationObservation(int(result.amplification_observation))
        is AmplificationObservation.NOT_APPLICABLE
    )
    np.testing.assert_array_equal(np.asarray(result.state), np.zeros(1))
    assert float(result.residual) > 1.0e-2


def test_sustained_growth_is_reported_without_blocking_terminal_promotion(monkeypatch):
    def growth_then_contraction(right_hand_side):
        residual = right_hand_side[0]
        increment = jnp.select(
            (residual > 99.5, residual > 98.0, residual > 95.0),
            (1.0, 2.0, 4.0),
            default=1.0,
        )
        return jnp.full_like(right_hand_side, increment)

    result = _solve_with_steps(
        monkeypatch,
        growth_then_contraction,
        steps=4,
        terminal_state=8.0,
    )

    np.testing.assert_allclose(np.asarray(result.state), [8.0])
    assert float(result.residual) == 0.0
    assert (
        KrylovActionQualification(int(result.krylov_action_qualification))
        is KrylovActionQualification.ACCEPTED
    )
    assert (
        AmplificationObservation(result._asdict()["amplification_observation"])
        is AmplificationObservation.SUSTAINED_GROWTH
    )


def test_qualified_contracting_trajectory_has_its_own_observation(monkeypatch):
    def contracting_steps(right_hand_side):
        residual = right_hand_side[0]
        increment = jnp.select(
            (residual > 98.0, residual > 95.0),
            (4.0, 2.0),
            default=1.0,
        )
        return jnp.full_like(right_hand_side, increment)

    result = _solve_with_steps(
        monkeypatch,
        contracting_steps,
        steps=3,
        terminal_state=7.0,
    )

    np.testing.assert_allclose(np.asarray(result.state), [7.0])
    assert float(result.residual) == 0.0
    assert (
        AmplificationObservation(int(result.amplification_observation))
        is AmplificationObservation.CONTRACTING
    )


def test_non_krylov_result_marks_the_observation_not_applicable():
    result = picard(
        lambda state: 0.5 * state + jnp.ones_like(state),
        jnp.zeros(1),
        evaluations=3,
    )

    assert (
        AmplificationObservation(int(result.amplification_observation))
        is AmplificationObservation.NOT_APPLICABLE
    )


def test_nonfinite_linear_action_keeps_the_state_unpromoted():
    @jax.custom_jvp
    def finite_map(state):
        return state + jnp.ones_like(state)

    @finite_map.defjvp
    def finite_map_jvp(primals, tangents):
        (state,), (tangent,) = primals, tangents
        return finite_map(state), tangent * jnp.nan

    result = newton_krylov(
        finite_map,
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
    )

    _assert_refusal(result, KrylovActionQualification.NONFINITE_LINEAR_ACTION)


def test_unsuccessful_solver_status_keeps_the_state_unpromoted(monkeypatch):
    def unsuccessful_gmres(_operator, right_hand_side, **_options):
        return right_hand_side, jnp.asarray(1)

    monkeypatch.setattr(
        fixed_point.jax.scipy.sparse.linalg, "gmres", unsuccessful_gmres
    )
    result = newton_krylov(
        lambda state: 0.5 * state + jnp.ones_like(state),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
    )

    _assert_refusal(result, KrylovActionQualification.NONSUCCESSFUL_GMRES_STATUS)


def test_nonfinite_achieved_residual_keeps_the_state_unpromoted(monkeypatch):
    def nonfinite_gmres(_operator, right_hand_side, **_options):
        return jnp.full_like(right_hand_side, jnp.nan), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", nonfinite_gmres)
    result = newton_krylov(
        lambda state: 0.5 * state + jnp.ones_like(state),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
    )

    _assert_refusal(
        result,
        KrylovActionQualification.NONFINITE_ACHIEVED_LINEAR_RESIDUAL,
    )


def test_zero_step_at_material_residual_keeps_the_state_unpromoted(monkeypatch):
    def zero_step_gmres(_operator, right_hand_side, **_options):
        return jnp.zeros_like(right_hand_side), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", zero_step_gmres)
    result = newton_krylov(
        lambda state: 0.5 * state + jnp.ones_like(state),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
    )

    _assert_refusal(
        result,
        KrylovActionQualification.ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL,
    )


if __name__ == "__main__":
    pytest.main([__file__])
