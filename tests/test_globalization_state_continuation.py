"""Continuation of decision state across compatible frozen-mask solves."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import _newton_krylov_inner, newton_krylov


def _settled_mask(_state):
    return jnp.zeros(1, dtype=bool)


def _solve_mask_change(continue_globalization_state: bool):
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
        continue_globalization_state=continue_globalization_state,
    )


def test_mask_change_resets_globalization_state_eager_and_jit():
    for transform in (lambda function: function, jax.jit):
        carried = transform(lambda: _solve_mask_change(True))()
        reset = transform(lambda: _solve_mask_change(False))()

        np.testing.assert_array_equal(carried.state, reset.state)
        np.testing.assert_array_equal(carried.residual, reset.residual)
        np.testing.assert_array_equal(carried.active_set_mask_differences[:2], [1, 0])


def _solve_monotone(continue_globalization_state: bool):
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
        continue_globalization_state=continue_globalization_state,
    )


def test_monotone_solve_is_bit_identical_eager_and_jit():
    for transform in (lambda function: function, jax.jit):
        carried = transform(lambda: _solve_monotone(True))()
        reset = transform(lambda: _solve_monotone(False))()

        np.testing.assert_array_equal(carried.state, reset.state)
        np.testing.assert_array_equal(carried.residual, reset.residual)


def test_inner_decision_state_accumulates_across_segments():
    """A resumed segment carries recovery state and then makes new decisions.

    Merit history grows and conditioning remains defined across the boundary.
    The first recovery starts from the carried radius, successive recovery rows
    form a state chain, and the final row supplies the resumed terminal radius.
    """

    def map_fn(state):
        return state + jnp.exp(0.2 * (state - 10.0))

    first, first_state = _newton_krylov_inner(
        map_fn,
        jnp.asarray([10.0]),
        newton_steps=2,
        gmres_iterations=2,
        warmup=1,
        relaxation=1.0,
        step_cap=0.1,
        convergence_tolerance=1.0e-12,
        return_globalization_state=True,
    )
    second, second_state = _newton_krylov_inner(
        map_fn,
        first.trajectory_state,
        newton_steps=2,
        gmres_iterations=2,
        warmup=1,
        relaxation=1.0,
        step_cap=0.1,
        convergence_tolerance=1.0e-12,
        run_warmup=False,
        globalization_state=first_state,
        resume_globalization=True,
        return_globalization_state=True,
    )

    assert int(second_state.merit_observations) > int(first_state.merit_observations)
    assert np.isfinite(np.asarray(second_state.condition_baseline))
    recovery_radii = np.asarray(second.promotion_recovery_radii)
    assert float(recovery_radii[0, 0]) == float(first_state.recovery_radius)
    np.testing.assert_array_equal(recovery_radii[:-1, 1], recovery_radii[1:, 0])
    assert float(second_state.recovery_radius) == float(recovery_radii[-1, 1])
