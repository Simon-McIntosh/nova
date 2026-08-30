"""Retention of the promoted best state across settled active-set trips."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import newton_krylov


def _settled_mask(_state):
    return jnp.zeros(1, dtype=bool)


def _solve_monotone(retain_outer_best_iterate: bool):
    def shadowed_map(state, _mask):
        return 0.5 * state + 1.0

    return newton_krylov(
        lambda state: shadowed_map(state, _settled_mask(state)),
        jnp.asarray([0.0, 0.5]),
        newton_steps=2,
        gmres_iterations=2,
        warmup=1,
        relaxation=1.0,
        shadow_mask_fn=_settled_mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=3,
        stop_on_active_set_stagnation=False,
        retain_outer_best_iterate=retain_outer_best_iterate,
    )


def _assert_result_equal(left, right):
    for field in left._fields:
        left_value = np.asarray(getattr(left, field))
        right_value = np.asarray(getattr(right, field))
        if np.issubdtype(left_value.dtype, np.inexact):
            assert np.array_equal(left_value, right_value, equal_nan=True), field
        else:
            np.testing.assert_array_equal(left_value, right_value, err_msg=field)


def test_retention_is_bit_identical_for_a_monotone_solve_eager_and_jit():
    for transform in (lambda function: function, jax.jit):
        retained = transform(lambda: _solve_monotone(True))()
        unguarded = transform(lambda: _solve_monotone(False))()
        _assert_result_equal(retained, unguarded)


def _solve_merit_degrading_warmup(retain_outer_best_iterate: bool):
    initial = jnp.ones(8)
    warmup_target = initial.at[0].set(2.0)

    def shadowed_map(state, _mask):
        degraded = warmup_target + 1.4
        return jnp.where(state[0] < 1.5, warmup_target, degraded)

    return newton_krylov(
        lambda state: shadowed_map(state, _settled_mask(state)),
        initial,
        newton_steps=1,
        gmres_iterations=1,
        warmup=1,
        relaxation=1.0,
        shadow_mask_fn=_settled_mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=1,
        convergence_tolerance=0.45,
        stop_on_active_set_stagnation=False,
        retain_outer_best_iterate=retain_outer_best_iterate,
    )


def test_settled_trip_keeps_incoming_state_when_warmup_degrades_merit():
    initial = np.ones(8)
    warmup_endpoint = initial.copy()
    warmup_endpoint[0] = 2.0

    for transform in (lambda function: function, jax.jit):
        retained = transform(lambda: _solve_merit_degrading_warmup(True))()
        unguarded = transform(lambda: _solve_merit_degrading_warmup(False))()

        np.testing.assert_array_equal(retained.state, initial)
        np.testing.assert_array_equal(unguarded.state, warmup_endpoint)
        assert float(retained.residual) == 0.5
        assert float(unguarded.residual) < float(retained.residual)
        retained_image = warmup_endpoint
        unguarded_image = warmup_endpoint + 1.4
        retained_merit = np.linalg.norm(
            retained_image - initial, ord=8
        ) / np.linalg.norm(np.r_[retained_image, 1.0e-30], ord=8)
        unguarded_merit = np.linalg.norm(
            unguarded_image - warmup_endpoint, ord=8
        ) / np.linalg.norm(np.r_[unguarded_image, 1.0e-30], ord=8)
        assert unguarded_merit > retained_merit
