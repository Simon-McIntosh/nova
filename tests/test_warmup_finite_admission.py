"""Warm-up proposals must leave a finite map for the Newton hand-off."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import _newton_krylov_inner, newton_krylov
from nova.jax.config import Precision, configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


def _bounded_map(state):
    return jnp.where(state > 1.5, jnp.nan, 2.0 - 0.5 * state)


def test_nonfinite_proposal_is_refused_before_newton():
    result = jax.jit(
        lambda initial: newton_krylov(
            _bounded_map,
            initial,
            newton_steps=2,
            gmres_iterations=1,
            warmup=4,
            relaxation=1.0,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1, dtype=jnp.float64))

    assert np.isfinite(result.state).all()
    assert np.isfinite(result.trajectory_state).all()
    assert int(result.attempted_newton_promotions) >= 1
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1])
    np.testing.assert_array_equal(result.warmup_first_nonfinite_index, [0])
    np.testing.assert_allclose(result.state, [4.0 / 3.0], rtol=0, atol=1e-14)
    assert bool(result.converged)


def test_refusal_retains_the_matching_shadow_partition():
    def mask(state):
        return state >= 0.5

    def shadowed(state, partition):
        consistent = jnp.all(partition == mask(state))
        return jnp.where(consistent, _bounded_map(state), jnp.nan)

    result = jax.jit(
        lambda initial: _newton_krylov_inner(
            _bounded_map,
            initial,
            newton_steps=2,
            gmres_iterations=1,
            warmup=4,
            relaxation=1.0,
            shadow_mask_fn=mask,
            promoted_shadow_mask_fn=lambda state, previous: mask(state),
            shadowed_map_fn=shadowed,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1, dtype=jnp.float64))

    assert np.isfinite(result.trajectory_state).all()
    assert int(result.attempted_newton_promotions) >= 1
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1])
    np.testing.assert_array_equal(result.warmup_first_nonfinite_index, [0])
    np.testing.assert_array_equal(result.shadow_mask_changes[:4], [0, 1, 0, 0])
    assert bool(result.converged)


def test_finite_warmup_keeps_its_trajectory_and_reports_no_refusal():
    result = jax.jit(
        lambda initial: newton_krylov(
            lambda state: 2.0 - 0.5 * state,
            initial,
            newton_steps=0,
            warmup=4,
            relaxation=0.5,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1, dtype=jnp.float64))

    np.testing.assert_array_equal(result.trajectory_state, [1.328125])
    np.testing.assert_array_equal(result.warmup_refused_proposals, [0])
    np.testing.assert_array_equal(result.warmup_first_nonfinite_index, [-1])


def test_finite_image_with_nonfinite_residual_is_refused():
    def overflowing_difference(state):
        return jnp.where(state >= 1e308, -1e308, 1e308)

    result = jax.jit(
        lambda initial: newton_krylov(
            overflowing_difference,
            initial,
            newton_steps=0,
            warmup=2,
            relaxation=1.0,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1, dtype=jnp.float64))

    np.testing.assert_array_equal(result.trajectory_state, [5e307])
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1])
    np.testing.assert_array_equal(result.warmup_first_nonfinite_index, [-1])


def test_active_set_history_keeps_each_warmup_refusal():
    def mask(state):
        return state >= 0.5

    result = jax.jit(
        lambda initial: newton_krylov(
            _bounded_map,
            initial,
            newton_steps=2,
            gmres_iterations=1,
            warmup=4,
            relaxation=1.0,
            shadow_mask_fn=mask,
            promoted_shadow_mask_fn=lambda state, previous: mask(state),
            shadowed_map_fn=lambda state, partition: _bounded_map(state),
            active_set_steps=3,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1, dtype=jnp.float64))

    assert bool(result.converged)
    assert int(result.attempted_newton_promotions) >= 1
    assert int(result.active_set_iterations) == 2
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1, 0, 0])
    np.testing.assert_array_equal(result.warmup_first_nonfinite_index, [0, -1, -1])
