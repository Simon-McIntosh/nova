"""Exact-equality termination for a motionless active-set solve."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import (
    FixedPointTerminationReason,
    newton_krylov,
)


def _constant_mask(_state):
    return jnp.zeros(1, dtype=bool)


def _stagnating_solve(stop_on_stagnation: bool):
    def shadowed_map(state, _mask):
        return state + 1.0

    return newton_krylov(
        lambda state: shadowed_map(state, _constant_mask(state)),
        jnp.ones(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=_constant_mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=6,
        stop_on_active_set_stagnation=stop_on_stagnation,
        stop_on_active_set_settlement=False,
    )


def _assert_array_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if np.issubdtype(left.dtype, np.inexact):
        assert np.array_equal(left, right, equal_nan=True)
    else:
        np.testing.assert_array_equal(left, right)


def test_stagnation_exit_preserves_the_full_loop_result_and_executed_receipts():
    execution_specific = {
        "attempted_newton_promotions",
        "termination_reason",
        "active_set_iterations",
        "active_set_residuals",
        "active_set_mask_differences",
        "active_set_cycle_damping_activations",
    }

    for transform in (lambda function: function, jax.jit):
        stopped = transform(lambda: _stagnating_solve(True))()
        full = transform(lambda: _stagnating_solve(False))()

        np.testing.assert_array_equal(stopped.state, full.state)
        np.testing.assert_array_equal(stopped.residual, full.residual)
        np.testing.assert_array_equal(
            _constant_mask(stopped.state), _constant_mask(full.state)
        )
        for field in stopped._fields:
            if field not in execution_specific:
                _assert_array_equal(getattr(stopped, field), getattr(full, field))

        executed = int(stopped.active_set_iterations)
        assert executed == 2
        assert int(full.active_set_iterations) == 6
        np.testing.assert_array_equal(
            stopped.active_set_residuals[:executed],
            full.active_set_residuals[:executed],
        )
        np.testing.assert_array_equal(
            stopped.active_set_mask_differences[:executed],
            full.active_set_mask_differences[:executed],
        )
        np.testing.assert_array_equal(
            stopped.active_set_cycle_damping_activations[:executed],
            full.active_set_cycle_damping_activations[:executed],
        )
        assert np.all(np.isnan(stopped.active_set_residuals[executed:]))
        np.testing.assert_array_equal(
            stopped.active_set_mask_differences[executed:], -1
        )
        np.testing.assert_array_equal(
            stopped.active_set_cycle_damping_activations[executed:], -1
        )
        assert (
            int(stopped.termination_reason)
            == FixedPointTerminationReason.ACTIVE_SET_STAGNATED
        )
        assert (
            int(full.termination_reason)
            == FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
        )


def test_strictly_decreasing_residual_does_not_trigger_stagnation():
    def shadowed_map(state, _mask):
        return 0.5 * state + 1.0

    def solve():
        return newton_krylov(
            lambda state: shadowed_map(state, _constant_mask(state)),
            jnp.zeros(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            step_cap=0.1,
            shadow_mask_fn=_constant_mask,
            promoted_shadow_mask_fn=lambda _state, previous: previous,
            shadowed_map_fn=shadowed_map,
            active_set_steps=4,
        )

    for result in (solve(), jax.jit(solve)()):
        assert int(result.active_set_iterations) == 4
        assert np.all(np.diff(np.asarray(result.active_set_residuals)) < 0.0)
        np.testing.assert_array_equal(result.active_set_mask_differences, 0)
        assert (
            int(result.termination_reason)
            == FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
        )
