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
