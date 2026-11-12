"""Residual exclusions must not freeze the physical flux extension."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


def operator():
    """A coupled affine source makes held and imaged shadow roots distinct."""
    result = object.__new__(ForwardFluxOperator)
    result.use_linear_moments = True
    result.moment_geometry = None
    result.external = lambda *_: jnp.array([1.0, 2.0, 3.0])
    result.internal = lambda state, *_: jnp.array(
        [0.2 * state[0] + 0.1 * state[1], 0.3 * state[0], 0.1 * state[2]]
    )
    result.residual_shadow_mask = lambda *_: jnp.array([False, True, False])
    return result


def image(op, state):
    return op.external() + op.internal(state)


@pytest.mark.parametrize("solver", ["warmup", "newton"])
def test_shadow_root_follows_physical_image_instead_of_seed(solver):
    op = operator()
    seed = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(seed)
    mapped = op.flux_map_with_shadow()
    arguments = dict(
        shadow_mask_fn=op.residual_shadow_mask,
        promoted_shadow_mask_fn=lambda *_: mask,
        shadowed_map_fn=mapped,
    )

    def solve(state):
        if solver == "warmup":
            return fixed_point.picard(
                lambda value: mapped(value, mask),
                state,
                evaluations=100,
                relaxation=0.5,
                **arguments,
            )
        return fixed_point.newton_krylov(
            lambda value: mapped(value, mask),
            state,
            newton_steps=8,
            warmup=0,
            gmres_iterations=3,
            convergence_tolerance=1e-11,
            **arguments,
        )

    result = jax.jit(solve)(seed)
    np.testing.assert_allclose(result.state, image(op, result.state), atol=1e-9, rtol=0)
    assert abs(float(result.state[1] - seed[1])) > 10
    np.testing.assert_array_equal(op.residual_shadow_mask(result.state), mask)


def test_warmup_assigns_the_shadow_image_without_relaxing_it():
    op = operator()
    seed = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(seed)
    mapped = op.flux_map_with_shadow()
    result = fixed_point._newton_krylov_inner(
        lambda value: mapped(value, mask),
        seed,
        newton_steps=0,
        warmup=1,
        relaxation=0.25,
        shadow_mask_fn=op.residual_shadow_mask,
        promoted_shadow_mask_fn=lambda *_: mask,
        shadowed_map_fn=mapped,
    )
    np.testing.assert_allclose(result.trajectory_state[1], image(op, seed)[1])


def test_residual_still_excludes_shadows_and_preserves_other_entries():
    op = operator()
    state = jnp.array([0.0, -11.0, 0.0])
    expected = jnp.where(op.residual_shadow_mask(state), 0.0, state - image(op, state))
    for residual in (op.residual, jax.jit(op.residual)):
        np.testing.assert_array_equal(residual(state), expected)
    _, tangent = jax.jvp(op.residual, (state,), (jnp.array([0.0, 1.0, 0.0]),))
    assert float(tangent[1]) == 0
    assert float(tangent[0]) == -0.1


def test_frozen_partition_map_and_dynamic_exterior_share_one_compilation():
    op = operator()
    op.use_linear_moments = False
    op._internal_on_partition = lambda state, *_: op.internal(state)
    traced = op.traced_flux_map_with_shadow()
    state = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(state)
    compiled = jax.jit(lambda state, mask, external: traced(state, mask, external))
    for seed, shadow, external in (
        (state, mask, op.external()),
        (state + 1, ~mask, op.external() + 2),
    ):
        np.testing.assert_allclose(
            compiled(seed, shadow, external), external + op.internal(seed)
        )
    assert compiled._cache_size() == 1
    partition = SimpleNamespace(residual_shadow=mask)
    np.testing.assert_allclose(
        traced._map_frozen_partition(state, partition, op.external()), image(op, state)
    )
