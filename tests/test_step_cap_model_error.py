"""Behavioural contracts for model-error-triggered Newton step capping."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point as fixed_point_module
    from nova.equilibrium.fixed_point import (
        _STEP_CAP_MODEL_ERROR_FRACTION,
        newton_krylov,
    )


@pytest.mark.parametrize(
    ("quadratic_coefficient", "expected_error", "cap_activated"),
    ((0.2, 1.0 / 6.0, False), (0.4, 2.0 / 7.0, True)),
    ids=("accurate-model", "inaccurate-model"),
)
def test_step_cap_follows_the_previous_accepted_model_error(
    quadratic_coefficient,
    expected_error,
    cap_activated,
):
    def solve():
        return newton_krylov(
            lambda state: quadratic_coefficient * state**2 + 1.0,
            jnp.zeros(1),
            newton_steps=2,
            gmres_iterations=1,
            warmup=0,
            relaxation=0.5,
            step_cap=0.1,
            model_trust_selection=False,
        )

    for result in (solve(), jax.jit(solve)()):
        model_errors = np.asarray(result.inner_iteration_model_error_fractions)
        activations = np.asarray(result.inner_iteration_step_cap_activations)
        cap_factors = np.asarray(result.inner_iteration_step_cap_factors)

        np.testing.assert_allclose(model_errors[0], expected_error)
        assert bool(model_errors[0] > _STEP_CAP_MODEL_ERROR_FRACTION) is cap_activated
        np.testing.assert_array_equal(activations, [0, int(cap_activated)])
        assert cap_factors[0] == 1.0
        if cap_activated:
            np.testing.assert_allclose(cap_factors[1], 0.01)
        else:
            assert cap_factors[1] == 1.0
        np.testing.assert_array_equal(result.inner_iteration_applied_factors, [1, 1])


def test_partition_change_clears_conditioning_and_model_error_state(monkeypatch):
    original = fixed_point_module._qualified_krylov_step

    def expose_carried_conditioning(*args, **kwargs):
        receipt = original(*args, **kwargs)
        stale_baseline = jnp.isfinite(kwargs["preceding_condition_baseline"])
        return receipt._replace(
            step=jnp.where(
                stale_baseline,
                0.5 * receipt.unconditioned_step,
                receipt.unconditioned_step,
            ),
            conditioning_applied=stale_baseline,
            condition_baseline=jnp.asarray(2.0, dtype=receipt.step.dtype),
        )

    monkeypatch.setattr(
        fixed_point_module,
        "_qualified_krylov_step",
        expose_carried_conditioning,
    )

    def mask_fn(state):
        return state >= 0.5

    def shadowed_map(state, mask):
        changing_partition = 0.4 * state**2 + 1.0
        settled_partition = jnp.full_like(state, 2.0)
        return jnp.where(mask, settled_partition, changing_partition)

    def solve():
        return newton_krylov(
            lambda state: shadowed_map(state, mask_fn(state)),
            jnp.zeros(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=mask_fn,
            promoted_shadow_mask_fn=lambda state, _previous: mask_fn(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=3,
            model_trust_selection=False,
        )

    for result in (solve(), jax.jit(solve)()):
        np.testing.assert_array_equal(result.active_set_mask_differences, [1, 0, -1])
        assert int(result.active_set_iterations) == 2
        assert int(result.krylov_conditioning_count) == 0
        np.testing.assert_array_equal(result.inner_iteration_step_cap_activations, [0])
        np.testing.assert_allclose(result.state, [2.0])
        assert bool(result.converged)
