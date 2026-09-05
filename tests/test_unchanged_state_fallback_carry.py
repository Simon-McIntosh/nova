"""Bit-identity contract for carrying an unchanged promotion fallback."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import (
        InnerIterationDecision,
        KrylovActionQualification,
        _MODEL_REBUILD_DAMPING_INITIAL,
        _NONMONOTONE_MERIT_WINDOW,
        _NewtonGlobalizationState,
        _RECOVERY_RADIUS_INITIAL,
        _initial_amplification_state,
        _newton_krylov_inner,
    )
    from nova.jax.config import configure_dtypes


def _refusing_full_fallback(*, carry_unchanged_fallback):
    def map_fn(state):
        return jnp.where(
            state[0] == 0.0,
            jnp.ones_like(state),
            state / 2.1,
        )

    return _newton_krylov_inner(
        map_fn,
        jnp.zeros(1),
        newton_steps=12,
        gmres_iterations=1,
        warmup=0,
        carry_unchanged_fallback=carry_unchanged_fallback,
    )


def test_carried_full_fallback_is_bit_identical_to_pipeline_reentry():
    configure_dtypes()
    reference = jax.jit(
        lambda: _refusing_full_fallback(carry_unchanged_fallback=False)
    )()
    carried = jax.jit(lambda: _refusing_full_fallback(carry_unchanged_fallback=True))()

    for reference_leaf, carried_leaf in zip(
        jax.tree.leaves(reference), jax.tree.leaves(carried), strict=True
    ):
        np.testing.assert_array_equal(reference_leaf, carried_leaf)

    assert int(carried.attempted_newton_promotions) == 12
    assert int(carried.accepted_newton_promotions) == 0
    np.testing.assert_array_equal(carried.state, [0.0])
    np.testing.assert_array_equal(carried.promotion_recovery_activations, 1)
    np.testing.assert_array_equal(carried.promotion_model_rebuild_activations, 1)
    np.testing.assert_array_equal(carried.promotion_descent_activations, 1)
    np.testing.assert_array_equal(
        carried.inner_iteration_decisions,
        InnerIterationDecision.SUFFICIENT_DECREASE_REFUSED,
    )
    np.testing.assert_array_equal(
        carried.inner_iteration_krylov_qualifications,
        KrylovActionQualification.ACCEPTED,
    )
    np.testing.assert_array_equal(carried.inner_iteration_accepted, 0)


def _globalization_state_with_condition_baseline(initial, baseline):
    """Resume a solve with a pre-warmed Krylov spectral baseline."""
    return _NewtonGlobalizationState(
        best_state=initial,
        best_residual=jnp.asarray(jnp.inf, dtype=initial.dtype),
        recent_merits=jnp.full(_NONMONOTONE_MERIT_WINDOW, jnp.nan, dtype=initial.dtype),
        merit_observations=jnp.asarray(0, dtype=jnp.int32),
        amplification=_initial_amplification_state(initial.dtype),
        condition_baseline=jnp.asarray(baseline, dtype=initial.dtype),
        previous_model_error_fraction=jnp.asarray(jnp.nan, dtype=initial.dtype),
        recovery_radius=jnp.asarray(_RECOVERY_RADIUS_INITIAL, dtype=initial.dtype),
        model_rebuild_damping=jnp.asarray(
            _MODEL_REBUILD_DAMPING_INITIAL, dtype=initial.dtype
        ),
    )


def _ill_conditioned_fallback(*, carry_unchanged_fallback):
    """A four-dimensional fallback whose linear action is ill-conditioned.

    The map discontinues at the origin into a fixed offset, whose diagonally
    contracted branch keeps one mode near marginal stability.  The projected
    condition of ``I - J`` then stays well above the ratio limit while the
    two-step GMRES solve at dimension four remains unresolved, so every
    requalified carried step is damped.  The pre-warmed baseline keeps the
    geometric-mean baseline relaxing from one carried attempt to the next, so
    no two carried steps coincide.
    """
    alpha = 5.0
    slow = 0.9
    offset = 0.01
    stray = 1.0e-4
    branch = jnp.asarray([1.0 / alpha] + [slow] + [0.5, 0.5], dtype=jnp.float64)

    def map_fn(state):
        jump = jnp.asarray(
            [1.0]
            + [slow * state[1] + offset]
            + [0.5 * state[2] + stray, 0.5 * state[3] + stray],
            dtype=jnp.float64,
        )
        return jnp.where(state[0] == 0.0, jump, state * branch)

    initial = jnp.zeros(4)
    return _newton_krylov_inner(
        map_fn,
        initial,
        newton_steps=12,
        gmres_iterations=2,
        warmup=0,
        carry_unchanged_fallback=carry_unchanged_fallback,
        globalization_state=_globalization_state_with_condition_baseline(initial, 2.0),
        resume_globalization=True,
    )


def test_damped_carried_step_on_ill_conditioned_action_is_bit_identical():
    configure_dtypes()
    reference = jax.jit(
        lambda: _ill_conditioned_fallback(carry_unchanged_fallback=False)
    )()
    carried = jax.jit(
        lambda: _ill_conditioned_fallback(carry_unchanged_fallback=True)
    )()

    for reference_leaf, carried_leaf in zip(
        jax.tree.leaves(reference), jax.tree.leaves(carried), strict=True
    ):
        np.testing.assert_array_equal(reference_leaf, carried_leaf)

    assert int(carried.krylov_conditioning_count) > 0
    assert float(carried.maximum_projected_krylov_condition) > jnp.e
    assert int(carried.accepted_newton_promotions) == 0
    np.testing.assert_array_equal(
        carried.inner_iteration_decisions,
        InnerIterationDecision.SUFFICIENT_DECREASE_REFUSED,
    )
    # The requalified carried steps re-scale the shared solve under the
    # relaxing baseline, so the recorded proposed steps are not all equal.
    proposed_norms = np.asarray(carried.inner_iteration_proposed_step_norms)
    assert np.unique(proposed_norms).size > 1


def _accepting_fallback(*, carry_unchanged_fallback):
    """A carried sequence that accepts a promotion on the unchanged state.

    The quadratic branch term breaks the scale-free refusal of the offset so a
    sufficiently re-scaled carried step clears the decrease ladder while the
    raw first-attempt step does not.  The carried sequence therefore spends
    several requalified attempts refusing and then promotes through the rebuilt
    normal model at the same residual.
    """
    alpha = 3.0
    slow = 0.9
    offset = 0.1
    stray = 1.0e-4
    branch = jnp.asarray([1.0 / alpha] + [slow] + [0.5, 0.5], dtype=jnp.float64)

    def map_fn(state):
        jump = jnp.asarray(
            [1.0]
            + [slow * state[1] + offset]
            + [0.5 * state[2] + stray, 0.5 * state[3] + stray],
            dtype=jnp.float64,
        )
        linear = state * branch + 1.0e-2 * state * state * branch
        return jnp.where(state[0] == 0.0, jump, linear)

    return _newton_krylov_inner(
        map_fn,
        jnp.zeros(4),
        newton_steps=16,
        gmres_iterations=2,
        warmup=0,
        carry_unchanged_fallback=carry_unchanged_fallback,
    )


def test_accepting_carry_is_bit_identical_to_pipeline_reentry():
    configure_dtypes()
    reference = jax.jit(lambda: _accepting_fallback(carry_unchanged_fallback=False))()
    carried = jax.jit(lambda: _accepting_fallback(carry_unchanged_fallback=True))()

    for reference_leaf, carried_leaf in zip(
        jax.tree.leaves(reference), jax.tree.leaves(carried), strict=True
    ):
        np.testing.assert_array_equal(reference_leaf, carried_leaf)

    accepted_decisions = (
        InnerIterationDecision.NEWTON_LADDER_ACCEPTED,
        InnerIterationDecision.CONTINUATION_ACCEPTED,
        InnerIterationDecision.REBUILT_MODEL_ACCEPTED,
        InnerIterationDecision.STEEPEST_DESCENT_ACCEPTED,
    )
    decisions = np.asarray(carried.inner_iteration_decisions)
    residuals = np.asarray(carried.inner_iteration_residuals_before)
    assert int(carried.accepted_newton_promotions) > 0
    assert int(carried.krylov_conditioning_count) > 0
    # Some acceptance follows a run of refusals at one unchanged residual:
    # the state that refused on the first attempt was carried and then
    # promoted while the stored linear solve was re-qualified.
    accepted_under_carry = False
    refusing = False
    for index, decision in enumerate(decisions):
        if decision == InnerIterationDecision.SUFFICIENT_DECREASE_REFUSED:
            refusing = True
        elif (
            decision in accepted_decisions
            and refusing
            and residuals[index] == residuals[index - 1]
        ):
            accepted_under_carry = True
            break
    assert accepted_under_carry
