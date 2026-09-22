"""Measured candidate admission when a local model is pessimistic."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import _backtracked_promotion
    from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _enable_float64():
    configure_dtypes()


def _mapped_with_residual(state, residual):
    return state / (1.0 - residual)


def _promotion(actual_residuals, predicted_residuals):
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
        own_mask_acceptance=True,
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
