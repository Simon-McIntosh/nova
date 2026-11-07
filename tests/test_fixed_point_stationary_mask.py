"""Termination naming for a mask that stops changing above tolerance.

A settled active set is not a converged fixed point.  These tests pin the
three endings apart: a settlement that proposed no Newton step (a stationary
map), one whose steps were proposed and refused, and true convergence.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from nova.equilibrium.fixed_point import (
    FixedPointTerminationReason,
    active_set_stop_reason,
    newton_krylov,
)
from nova.equilibrium.solve_request import ForwardSolveReceipt


def _refused_step_solve():
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


def test_refused_step_at_a_frozen_mask_names_the_refused_step():
    result = _refused_step_solve()

    assert not bool(result.converged)
    assert float(result.residual) > 1.0e-12
    assert int(result.attempted_newton_promotions) > 0
    assert int(result.accepted_newton_promotions) == 0
    assert (
        int(result.termination_reason)
        == FixedPointTerminationReason.ACTIVE_SET_STEP_REFUSED
    )


def test_stationary_map_is_named_as_stationary_and_not_as_converged():
    result = _refused_step_solve()

    reason = active_set_stop_reason(
        converged=jnp.asarray(False),
        cycle_detected=jnp.asarray(False),
        nonfinite=jnp.asarray(False),
        stationary=jnp.asarray(True),
        step_refused=jnp.asarray(False),
        settled=jnp.asarray(True),
        stagnated=jnp.asarray(False),
    )

    assert int(reason) == FixedPointTerminationReason.ACTIVE_SET_STATIONARY
    assert not bool(result.converged)
    assert (
        int(result.termination_reason) != FixedPointTerminationReason.ACTIVE_SET_SETTLED
    )


def test_a_stable_mask_above_tolerance_never_reports_settled():
    """Neither settlement ending may report the retired settled label."""

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
