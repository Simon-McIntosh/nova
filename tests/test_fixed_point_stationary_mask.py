"""Stopping at a settled mask above tolerance.

A settled active set is not a converged fixed point, and it is not a stall
until a Newton promotion at that mask has been tried.  These tests pin the
three endings apart: a settlement that ends only after a refused promotion, one
that keeps its trips because no promotion was proposed, and true convergence.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from nova.equilibrium.fixed_point import (
    FixedPointTerminationReason,
    active_set_settles_as_stall,
    active_set_stop_reason,
    newton_krylov,
)
from nova.equilibrium.solve_request import ForwardSolveReceipt


def _settled_solve():
    """A map with no fixed point, held at a frozen mask above tolerance."""

    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _mask):
        return state + 1.0

    return newton_krylov(
        lambda state: shadowed_map(state, mask(state)),
        jnp.ones(1),
        newton_steps=1,
        gmres_iterations=1,
        warmup=0,
        convergence_tolerance=1.0e-12,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda _state, previous: previous,
        shadowed_map_fn=shadowed_map,
        active_set_steps=4,
        stop_on_active_set_stagnation=False,
    )


def test_settled_mask_above_tolerance_attempts_a_promotion_before_it_ends():
    result = _settled_solve()

    assert not bool(result.converged)
    assert float(result.residual) > 1.0e-12
    assert int(result.attempted_newton_promotions) > 0
    assert int(result.accepted_newton_promotions) == 0
    assert (
        int(result.termination_reason)
        == FixedPointTerminationReason.ACTIVE_SET_STEP_REFUSED
    )
    assert (
        int(result.termination_reason) != FixedPointTerminationReason.ACTIVE_SET_SETTLED
    )


def test_a_settlement_that_proposed_no_promotion_is_not_a_stall():
    """The loop keeps its trips until a promotion at the frozen mask is tried."""

    assert not bool(active_set_settles_as_stall(jnp.asarray(True), jnp.asarray(0)))
    assert not bool(active_set_settles_as_stall(jnp.asarray(False), jnp.asarray(0)))
    assert bool(active_set_settles_as_stall(jnp.asarray(True), jnp.asarray(1)))


def test_a_settled_mask_above_tolerance_never_reports_settled():
    for stationary in (True, False):
        reason = active_set_stop_reason(
            converged=jnp.asarray(False),
            cycle_detected=jnp.asarray(False),
            nonfinite=jnp.asarray(False),
            stationary=jnp.asarray(stationary),
            step_refused=jnp.asarray(not stationary),
            settled=jnp.asarray(True),
            stagnated=jnp.asarray(False),
        )
        assert int(reason) != int(FixedPointTerminationReason.ACTIVE_SET_SETTLED)


def test_converged_map_reports_convergence():
    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _mask):
        return state

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
        active_set_steps=4,
    )

    assert bool(result.converged)
    assert int(result.termination_reason) == FixedPointTerminationReason.CONVERGED


def test_forward_receipt_publishes_the_promotion_counts():
    names = {field.name for field in dataclasses.fields(ForwardSolveReceipt)}

    assert "attempted_newton_promotions" in names
    assert "accepted_newton_promotions" in names
