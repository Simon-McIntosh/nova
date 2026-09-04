"""Continuation of a contracting Newton trajectory across settled masks."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import newton_krylov


def _settled_mask(_state):
    return jnp.zeros(1, dtype=bool)


def _assert_result_equal(left, right):
    for field in left._fields:
        left_value = np.asarray(getattr(left, field))
        right_value = np.asarray(getattr(right, field))
        if np.issubdtype(left_value.dtype, np.inexact):
            assert np.array_equal(left_value, right_value, equal_nan=True), field
        else:
            np.testing.assert_array_equal(left_value, right_value, err_msg=field)


def _stalled_segment_solve(continue_newton_trajectory: bool):
    def shadowed_map(state, _mask):
        return state + jnp.exp(0.2 * (state - 10.0))

    return newton_krylov(
        lambda state: shadowed_map(state, _settled_mask(state)),
        jnp.asarray([10.0]),
        newton_steps=2,
        gmres_iterations=2,
        warmup=1,
        relaxation=1.0,
        step_cap=0.1,
        convergence_tolerance=1.0e-12,
        shadow_mask_fn=_settled_mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=6,
        continue_newton_trajectory=continue_newton_trajectory,
    )


def test_continuation_accumulates_contracting_segments_eager_and_jit():
    """A compatible partition extends one contracting trajectory across segments.

    Continuation reaches every available segment and therefore outlasts a reset
    control.  The active-set mask stays fixed while the recorded tail contracts,
    and eager and compiled execution produce identical solver receipts.
    """
    eager_continued = _stalled_segment_solve(True)
    eager_reset = _stalled_segment_solve(False)
    compiled_continued = jax.jit(lambda: _stalled_segment_solve(True))()
    compiled_reset = jax.jit(lambda: _stalled_segment_solve(False))()

    _assert_result_equal(eager_continued, compiled_continued)
    _assert_result_equal(eager_reset, compiled_reset)

    for continued, reset in (
        (eager_continued, eager_reset),
        (compiled_continued, compiled_reset),
    ):
        assert int(continued.active_set_iterations) == 6
        assert int(continued.active_set_iterations) > int(reset.active_set_iterations)
        np.testing.assert_array_equal(continued.active_set_mask_differences, 0)
        assert np.all(np.diff(np.asarray(continued.active_set_residuals)[3:]) < 0.0)


def _mask_changing_solve(continue_newton_trajectory: bool):
    def mask(state):
        return jnp.asarray([state[0] > 0.5])

    def shadowed_map(state, _mask):
        return jnp.ones_like(state)

    return newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        jnp.zeros(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, _previous: mask(state),
        shadowed_map_fn=shadowed_map,
        active_set_steps=3,
        continue_newton_trajectory=continue_newton_trajectory,
    )


def test_mask_change_is_bit_identical_with_continuation_on_and_off():
    for transform in (lambda function: function, jax.jit):
        continued = transform(lambda: _mask_changing_solve(True))()
        reset = transform(lambda: _mask_changing_solve(False))()
        _assert_result_equal(continued, reset)
        np.testing.assert_array_equal(continued.active_set_mask_differences[:2], [1, 0])


def _monotone_solve(continue_newton_trajectory: bool):
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
        continue_newton_trajectory=continue_newton_trajectory,
    )


def test_monotone_solve_is_bit_identical_with_continuation_on_and_off():
    for transform in (lambda function: function, jax.jit):
        continued = transform(lambda: _monotone_solve(True))()
        reset = transform(lambda: _monotone_solve(False))()
        _assert_result_equal(continued, reset)
