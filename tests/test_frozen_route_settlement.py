"""Residual-gated settlement for frozen-partition Newton solves."""

from __future__ import annotations

import jax.numpy as jnp

from nova.equilibrium.fixed_point import FixedPointTerminationReason, newton_krylov


def test_frozen_partition_settlement_exhausts_while_the_live_residual_is_large():
    """A stable partition cannot settle an unconverged live fixed point."""

    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _partition):
        return state + 1.0

    shadowed_map._read_frozen_partition = lambda state, _previous: mask(state)
    shadowed_map._map_frozen_partition = shadowed_map
    shadowed_map._frozen_partition_shadow = lambda partition: partition

    result = newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        jnp.ones(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        convergence_tolerance=1.0e-12,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=3,
        stop_on_active_set_stagnation=False,
    )

    assert not bool(result.converged)
    assert int(result.active_set_iterations) == 3
    assert (
        int(result.termination_reason)
        == FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
    )
