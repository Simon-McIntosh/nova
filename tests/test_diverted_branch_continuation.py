"""Admitted separatrices survive finite proposals onto another branch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.topology import TopologyState
from nova.jax.config import Precision, configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


class Partition(NamedTuple):
    label: jax.Array
    topology: TopologyState
    profile_support: None
    residual_shadow: jax.Array


def _solve(initial, *, switch="limited", warmup=0, newton_steps=2):
    def mapped(state):
        return 2.0 - 0.5 * state

    def mask(state):
        return jnp.ones_like(state, dtype=bool)

    def read(state, previous=None):
        del previous
        changed = state[0] >= 0.8
        axis = jnp.array([1.5, 0.0])
        saddle = jnp.array([1.0, jnp.where(changed & (switch == "upper"), 1.0, -1.0)])
        diverted = ~changed if switch == "limited" else jnp.asarray(True)
        topology = TopologyState(
            axis, 0.0, saddle, 1.0, saddle, 1.0, axis, 2.0, diverted
        )
        return Partition(jnp.ones(1, dtype=jnp.int32), topology, None, mask(state))

    def shadowed(state, shadow):
        del shadow
        return mapped(state)

    shadowed._read_frozen_partition = read
    shadowed._map_frozen_partition = lambda state, partition: mapped(state)
    shadowed._frozen_partition_shadow = lambda partition: partition.residual_shadow
    return newton_krylov(
        mapped,
        initial,
        newton_steps=newton_steps,
        gmres_iterations=1,
        warmup=warmup,
        relaxation=1.0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, previous: mask(state),
        shadowed_map_fn=shadowed,
        active_set_steps=2,
        precision=Precision.DOUBLE,
    )


@pytest.mark.parametrize("switch", ["limited", "upper"])
def test_newton_refuses_boundary_loss_and_upper_saddle_switch(switch):
    result = jax.jit(lambda initial: _solve(initial, switch=switch))(jnp.zeros(1))
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.trajectory_state, [0.0])
    assert int(result.accepted_newton_promotions) == 0
    assert int(jnp.sum(result.newton_branch_refused_proposals)) > 0
    assert int(result.newton_first_branch_refusal_index[0]) == 0


@pytest.mark.parametrize("switch", ["limited", "upper"])
def test_warmup_refuses_boundary_loss_and_upper_saddle_switch(switch):
    result = jax.jit(lambda initial: _solve(initial, switch=switch, warmup=2))(
        jnp.zeros(1)
    )
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.trajectory_state, [0.0])
    assert int(jnp.sum(result.warmup_branch_refused_proposals)) == 2
    assert int(result.warmup_first_branch_refusal_index[0]) == 0
    assert int(jnp.sum(result.warmup_refused_proposals)) == 0


def test_limited_seed_remains_unconstrained():
    result = jax.jit(_solve)(jnp.ones(1))
    np.testing.assert_allclose(result.state, [4.0 / 3.0], rtol=0.0, atol=1e-14)
    assert bool(result.converged)
    assert int(jnp.sum(result.newton_branch_refused_proposals)) == 0


def test_warmup_relaxation_stays_damped_after_finite_refusal():
    def mapped(state):
        return jnp.where(state > 1.5, jnp.nan, 2.0 - 0.5 * state)

    result = jax.jit(
        lambda initial: newton_krylov(
            mapped,
            initial,
            warmup=3,
            newton_steps=0,
            relaxation=1.0,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1))
    # Refuse 2, then retain relaxation 1/2 for both accepted proposals.
    np.testing.assert_array_equal(result.trajectory_state, [1.25])
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1])
