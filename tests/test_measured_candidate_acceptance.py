"""Measured candidate admission when a local model is pessimistic."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point as fixed_point_module
    from nova.equilibrium.fixed_point import (
        RecoveryOutcome,
        _backtracked_promotion,
        newton_krylov,
    )
    from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _enable_float64():
    configure_dtypes()


def _mapped_with_residual(state, residual):
    return state / (1.0 - residual)


def _promotion(actual_residuals, predicted_residuals, *, own_mask_acceptance=True):
    incumbent, full, half, other = actual_residuals
    predicted_incumbent, predicted_full, predicted_half, predicted_other = (
        predicted_residuals
    )

    def actual_map(state):
        value = state[0]
        residual = jnp.where(
            value == 1.0,
            incumbent,
            jnp.where(
                value == 2.0,
                full,
                jnp.where(value == 1.5, half, other),
            ),
        )
        return _mapped_with_residual(state, residual)

    def local_model(state):
        value = state[0]
        residual = jnp.where(
            value == 1.0,
            predicted_incumbent,
            jnp.where(
                value == 2.0,
                predicted_full,
                jnp.where(value == 1.5, predicted_half, predicted_other),
            ),
        )
        return _mapped_with_residual(state, residual)

    return _backtracked_promotion(
        actual_map,
        local_model,
        jnp.ones(1),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.asarray(incumbent),
        jnp.asarray(incumbent),
        jnp.asarray(1.0),
        True,
        acceptance_map_fn=actual_map,
        own_mask_acceptance=own_mask_acceptance,
    )


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_pessimistic_model_promotes_strict_measured_decrease(compiled):
    def promote():
        return _promotion(
            (0.04, 0.001, 0.08, 0.08),
            (0.04, 0.25, 0.08, 0.08),
        )

    result = jax.jit(promote)() if compiled else promote()

    np.testing.assert_array_equal(result.state, [2.0])
    assert bool(result.accepted)
    assert not bool(result.recovery_activated)
    assert bool(result.model_distrusted)
    assert float(result.applied_factor) == 1.0


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_pessimistic_model_requires_own_mask_authority(compiled):
    def promote():
        return _promotion(
            (0.04, 0.001, 0.08, 0.08),
            (0.04, 0.25, 0.08, 0.08),
            own_mask_acceptance=False,
        )

    result = jax.jit(promote)() if compiled else promote()

    np.testing.assert_array_equal(result.state, [1.0])
    assert not bool(result.accepted)
    assert bool(result.recovery_activated)
    assert not bool(result.model_distrusted)
    assert float(result.applied_factor) == 0.0


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_trusted_half_step_remains_accepted(compiled):
    def promote():
        return _promotion(
            (0.04, 0.08, 0.039, 0.08),
            (0.04, 0.08, 0.0395, 0.08),
        )

    result = jax.jit(promote)() if compiled else promote()

    np.testing.assert_array_equal(result.state, [1.5])
    assert bool(result.accepted)
    assert not bool(result.recovery_activated)
    assert not bool(result.model_distrusted)
    assert float(result.applied_factor) == 0.5


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_worsening_candidates_remain_refused(compiled):
    def promote():
        return _promotion(
            (0.04, 0.06, 0.055, 0.05),
            (0.04, 0.001, 0.01, 0.02),
        )

    result = jax.jit(promote)() if compiled else promote()

    np.testing.assert_array_equal(result.state, [1.0])
    assert not bool(result.accepted)
    assert bool(result.recovery_activated)
    assert bool(result.model_distrusted)
    assert float(result.applied_factor) == 0.0


@pytest.mark.parametrize(
    ("model_unreliable", "expected_rebuilds"),
    ((True, 1), (False, 0)),
    ids=("pessimistic-admission", "trusted-admission"),
)
def test_model_unreliability_forces_one_rebuild_on_the_next_iteration(
    monkeypatch,
    model_unreliable,
    expected_rebuilds,
):
    def promotion_stub(*args, **_kwargs):
        state = args[2]
        recovery_radius = args[7]
        first = state[0] < 0.5
        candidate = jnp.where(first, jnp.ones_like(state), 2.0 * jnp.ones_like(state))
        candidate_mapped = 0.5 * candidate + 1.0
        candidate_residual = fixed_point_module._relative_residual(
            candidate_mapped, candidate
        )
        return fixed_point_module._BacktrackedPromotion(
            state=candidate,
            residual=candidate_residual,
            accepted=jnp.asarray(True),
            backtrack_count=jnp.asarray(0, dtype=jnp.int32),
            recovery_activated=jnp.asarray(False),
            recovery_radius=recovery_radius,
            recovery_radius_before=jnp.asarray(jnp.nan, dtype=state.dtype),
            recovery_outcome=jnp.asarray(
                RecoveryOutcome.NOT_APPLICABLE, dtype=jnp.int32
            ),
            applied_factor=jnp.asarray(1.0, dtype=state.dtype),
            model_distrusted=first & jnp.asarray(model_unreliable),
            model_error_fraction=jnp.asarray(1.0, dtype=state.dtype),
        )

    def rebuild_stub(_map_fn, state, *_args, **_kwargs):
        candidate = 2.0 * jnp.ones_like(state)
        return fixed_point_module._RebuiltModelPromotion(
            state=candidate,
            residual=jnp.asarray(0.0, dtype=state.dtype),
            accepted=jnp.asarray(True),
            damping=jnp.asarray(1.0e-3, dtype=state.dtype),
            next_damping=jnp.asarray(1.0e-3, dtype=state.dtype),
        )

    monkeypatch.setattr(fixed_point_module, "_backtracked_promotion", promotion_stub)
    monkeypatch.setattr(fixed_point_module, "_rebuilt_model_promotion", rebuild_stub)

    result = newton_krylov(
        lambda state: 0.5 * state + 1.0,
        jnp.zeros(1),
        newton_steps=2,
        gmres_iterations=1,
        warmup=0,
    )

    assert int(jnp.sum(result.promotion_model_rebuild_activations == 1)) == (
        expected_rebuilds
    )
    np.testing.assert_array_equal(
        np.asarray(result.promotion_model_rebuild_activations),
        [0, expected_rebuilds],
    )
    np.testing.assert_array_equal(result.state, [2.0])
    assert bool(result.converged)
