from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.seed_alignment import residual_action_amplification


def _diagonal_map(diagonal: jax.Array):
    def apply(state: jax.Array) -> jax.Array:
        return diagonal * state

    return apply


def test_residual_action_amplification_ranks_weak_action_alignment() -> None:
    map_fn = _diagonal_map(jnp.asarray([0.9, 0.0]))

    weak_score = residual_action_amplification(map_fn, jnp.asarray([1.0, 0.0]))
    strong_score = residual_action_amplification(map_fn, jnp.asarray([0.0, 1.0]))

    np.testing.assert_allclose(weak_score, 10.0, rtol=1.0e-6)
    np.testing.assert_allclose(strong_score, 1.0, rtol=1.0e-6)
    assert float(weak_score) > float(strong_score)


def test_residual_action_amplification_is_scale_invariant_for_linear_map() -> None:
    map_fn = _diagonal_map(jnp.asarray([0.75, -0.5]))
    seed = jnp.asarray([2.0, -3.0])

    baseline = residual_action_amplification(map_fn, seed)
    scaled = residual_action_amplification(map_fn, 7.0 * seed)

    np.testing.assert_allclose(scaled, baseline, rtol=1.0e-6)


def test_residual_action_amplification_handles_exact_fixed_point() -> None:
    map_fn = _diagonal_map(jnp.asarray([0.9, 0.2]))

    score = residual_action_amplification(map_fn, jnp.zeros(2))

    np.testing.assert_array_equal(score, 0.0)


def test_residual_action_amplification_supports_jit_and_vmap() -> None:
    map_fn = _diagonal_map(jnp.asarray([0.9, 0.0]))
    seeds = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    score = jax.jit(jax.vmap(lambda seed: residual_action_amplification(map_fn, seed)))

    np.testing.assert_allclose(score(seeds), [10.0, 1.0], rtol=1.0e-6)
