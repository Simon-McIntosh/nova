"""Fail-closed qualification of Newton--Krylov linear actions."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point
    from nova.equilibrium.fixed_point import (
        KrylovActionQualification,
        newton_krylov,
    )


def _solve(map_fn):
    return newton_krylov(
        map_fn,
        jnp.zeros(3),
        newton_steps=1,
        gmres_iterations=3,
        warmup=0,
    )


def _assert_refused(result, expected):
    observed = KrylovActionQualification(int(result.krylov_action_qualification))
    assert observed is expected
    np.testing.assert_array_equal(np.asarray(result.state), np.zeros(3))
    assert float(result.residual) > 1.0e-2


def test_nonfinite_linear_action_is_refused_before_gmres_success_is_trusted():
    @jax.custom_jvp
    def finite_map(state):
        return state + jnp.ones_like(state)

    @finite_map.defjvp
    def finite_map_jvp(primals, tangents):
        (state,), (tangent,) = primals, tangents
        return finite_map(state), tangent * jnp.nan

    result = _solve(finite_map)

    _assert_refused(result, KrylovActionQualification.NONFINITE_LINEAR_ACTION)


def test_nonsuccessful_gmres_status_refuses_a_finite_step(monkeypatch):
    def unsuccessful_gmres(_operator, right_hand_side, **_options):
        return 2.0 * right_hand_side, jnp.asarray(1)

    monkeypatch.setattr(
        fixed_point.jax.scipy.sparse.linalg, "gmres", unsuccessful_gmres
    )
    result = _solve(lambda state: 0.5 * state + jnp.ones_like(state))

    _assert_refused(result, KrylovActionQualification.NONSUCCESSFUL_GMRES_STATUS)


def test_nonfinite_achieved_linear_residual_refuses_the_step(monkeypatch):
    def nonfinite_step(_operator, right_hand_side, **_options):
        return jnp.full_like(right_hand_side, jnp.nan), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", nonfinite_step)
    result = _solve(lambda state: 0.5 * state + jnp.ones_like(state))

    _assert_refused(
        result,
        KrylovActionQualification.NONFINITE_ACHIEVED_LINEAR_RESIDUAL,
    )


def test_zero_step_with_material_nonlinear_residual_is_refused(monkeypatch):
    def zero_step(_operator, right_hand_side, **_options):
        return jnp.zeros_like(right_hand_side), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", zero_step)
    result = _solve(lambda state: 0.5 * state + jnp.ones_like(state))

    _assert_refused(
        result,
        KrylovActionQualification.ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL,
    )


if __name__ == "__main__":
    pytest.main([__file__])
