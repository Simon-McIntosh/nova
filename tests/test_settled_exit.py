"""Per-member settlement masking for the bounded active-set loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import (
    FixedPointTerminationReason,
    newton_krylov,
)


def _qualified_noop_solve(initial, change_mask, settlement):
    def mask(state):
        return jnp.asarray([change_mask & (state[0] >= 0.5)])

    def shadowed_map(state, active_mask):
        can_advance = change_mask & ~active_mask[0]
        return jnp.where(can_advance, jnp.ones_like(state), state + 1.0)

    return newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        initial,
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        convergence_tolerance=1.0e-12,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, _previous: mask(state),
        shadowed_map_fn=shadowed_map,
        active_set_steps=4,
        stop_on_active_set_stagnation=False,
        stop_on_active_set_settlement=settlement,
    )


def test_settled_members_mask_later_trips_without_shortening_receipts():
    solve = jax.jit(
        jax.vmap(_qualified_noop_solve, in_axes=(0, 0, None)),
        static_argnums=2,
    )
    initial = jnp.zeros((2, 1))
    result = solve(initial, jnp.asarray([False, True]), True)

    np.testing.assert_array_equal(result.active_set_iterations, [1, 2])
    np.testing.assert_array_equal(
        result.termination_reason,
        [
            FixedPointTerminationReason.ACTIVE_SET_SETTLED,
            FixedPointTerminationReason.ACTIVE_SET_SETTLED,
        ],
    )
    np.testing.assert_array_equal(result.active_set_mask_differences[0, 1:], -1)
    np.testing.assert_array_equal(result.active_set_mask_differences[1, 2:], -1)
    assert np.all(np.isnan(np.asarray(result.active_set_residuals[0, 1:])))
    assert np.all(np.isnan(np.asarray(result.active_set_residuals[1, 2:])))
    np.testing.assert_array_equal(result.active_set_mask_differences[0, 0], 0)
    np.testing.assert_array_equal(result.active_set_mask_differences[1, :2], [1, 0])
    np.testing.assert_array_equal(result.state, [[0.0], [1.0]])
    np.testing.assert_array_equal(result.residual, [1.0, 0.5])
    np.testing.assert_array_equal(result.accepted_newton_promotions, [0, 1])


def test_settlement_toggle_preserves_the_executed_prefix():
    settled = _qualified_noop_solve(jnp.zeros(1), jnp.asarray(False), True)
    full = _qualified_noop_solve(jnp.zeros(1), jnp.asarray(False), False)

    executed = int(settled.active_set_iterations)
    assert executed == 1
    assert int(full.active_set_iterations) == 4
    np.testing.assert_array_equal(
        settled.active_set_residuals[:executed],
        full.active_set_residuals[:executed],
    )
    np.testing.assert_array_equal(
        settled.active_set_mask_differences[:executed],
        full.active_set_mask_differences[:executed],
    )
    np.testing.assert_array_equal(
        settled.active_set_cycle_damping_activations[:executed],
        full.active_set_cycle_damping_activations[:executed],
    )
    np.testing.assert_array_equal(settled.state, full.state)
    np.testing.assert_array_equal(settled.residual, full.residual)


def test_refused_own_mask_promotion_qualifies_as_a_settled_noop():
    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _mask):
        return state + 1.0

    def solve(settlement):
        return newton_krylov(
            lambda state: shadowed_map(state, mask(state)),
            jnp.ones(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=mask,
            promoted_shadow_mask_fn=lambda _state, previous: previous,
            shadowed_map_fn=shadowed_map,
            active_set_steps=4,
            stop_on_active_set_stagnation=False,
            stop_on_active_set_settlement=settlement,
        )

    result = solve(True)
    full = solve(False)

    assert int(result.accepted_newton_promotions) == 0
    assert int(result.active_set_iterations) == 1
    np.testing.assert_array_equal(result.active_set_mask_differences, [0, -1, -1, -1])
    np.testing.assert_array_equal(result.state, [1.0])
    np.testing.assert_array_equal(result.residual, 0.5)
    np.testing.assert_array_equal(result.state, full.state)
    np.testing.assert_array_equal(result.residual, full.residual)
    assert (
        int(result.termination_reason) == FixedPointTerminationReason.ACTIVE_SET_SETTLED
    )
