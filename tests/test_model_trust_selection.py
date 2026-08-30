"""Model-fidelity selection for fixed-shape nonlinear promotions."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import (
        _backtracked_promotion,
        newton_krylov,
    )
    from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _enable_float64():
    configure_dtypes()


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_trust_selection_is_inert_for_a_matching_newton_model(compiled):
    matrix = jnp.asarray([[0.25, 0.0], [0.0, 0.5]])
    offset = jnp.asarray([1.0, -0.5])

    def solve(enabled):
        return newton_krylov(
            lambda state: matrix @ state + offset,
            jnp.zeros(2),
            newton_steps=2,
            gmres_iterations=2,
            warmup=0,
            model_trust_selection=enabled,
        )

    evaluate = jax.jit(solve, static_argnums=0) if compiled else solve
    trusted = evaluate(True)
    unchecked = evaluate(False)

    np.testing.assert_array_equal(trusted.state, unchecked.state)
    np.testing.assert_array_equal(trusted.residual, unchecked.residual)
    np.testing.assert_array_equal(trusted.trace, unchecked.trace)
    assert int(trusted.accepted_newton_promotions) == int(
        unchecked.accepted_newton_promotions
    )


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_mispredicting_newton_family_falls_through_to_matching_continuation(
    compiled,
):
    def actual_map(state):
        value = state[0]
        mapped = jnp.where(
            value == 0.0, 1.0, jnp.where(value > 0.0, 20.0 * value, 2.0 * value)
        )
        return jnp.asarray([mapped])

    def local_model(state):
        value = state[0]
        mapped = jnp.where(
            value == 0.0, 1.0, jnp.where(value > 0.0, value, 2.0 * value)
        )
        return jnp.asarray([mapped])

    def promote(enabled):
        return _backtracked_promotion(
            actual_map,
            local_model,
            jnp.zeros(1),
            jnp.ones(1),
            -jnp.ones(1),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            enabled,
        )

    evaluate = jax.jit(promote, static_argnums=0) if compiled else promote
    trusted = evaluate(True)
    unchecked = evaluate(False)

    np.testing.assert_array_equal(trusted.state, [-1.0])
    assert bool(trusted.accepted)
    assert bool(trusted.recovery_activated)
    np.testing.assert_array_equal(unchecked.state, [1.0])
    assert bool(unchecked.accepted)
    assert not bool(unchecked.recovery_activated)
