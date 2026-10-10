"""Runtime geometry ownership in the forward flux map."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import gs
from nova.jax.config import configure_dtypes
from tests.test_forward_operator_arguments import _operator
from tests.test_forward_profile_arguments import _source


def _geometry_pair():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    first = _operator()
    second = _operator()
    first.source = _source([2.0], [0.5])
    second.source = _source([2.0], [0.5])
    second.area = np.asarray(first.area) * 1.01
    second.grid.plasma_target = first.grid.plasma_target * 1.1
    second._geometry_identity = second._compute_geometry_identity()
    assert first.geometry_identity != second.geometry_identity
    return first, second


def test_map_matches_the_existing_book_and_image():
    first, second = _geometry_pair()
    state = jnp.linspace(-1.0, 1.0, first.node_number)
    for operator in (first, second):
        external = operator.external()
        image = external + operator.internal(state)
        expected = operator._exclude_shadow_residual(state, image, None)
        np.testing.assert_allclose(
            gs.apply(state, external, operator), expected, rtol=1e-13, atol=1e-13
        )


def test_one_compilation_serves_distinct_geometry_values():
    first, second = _geometry_pair()
    assert first.program_identity == second.program_identity
    state = jnp.linspace(-1.0, 1.0, first.node_number)
    compiled = jax.jit(gs.apply)
    first_result = compiled(state, first.external(), first)
    assert compiled._cache_size() == 1
    second_result = compiled(state, second.external(), second)
    assert compiled._cache_size() == 1
    assert not np.array_equal(np.asarray(first_result), np.asarray(second_result))


def test_captured_coupling_is_a_program_constant():
    first, second = _geometry_pair()
    state = jnp.ones(first.grid.node_number)
    traces = []

    def bound(block):
        def image(current):
            traces.append(1)
            return block @ current

        return image

    first_bound = bound(first.grid.plasma_target)
    second_bound = bound(second.grid.plasma_target)
    first_jaxpr = jax.make_jaxpr(first_bound)(state)
    second_jaxpr = jax.make_jaxpr(second_bound)(state)
    assert any(
        tuple(np.shape(value)) == tuple(first.grid.plasma_target.shape)
        for value in first_jaxpr.consts
    )
    assert any(
        tuple(np.shape(value)) == tuple(second.grid.plasma_target.shape)
        for value in second_jaxpr.consts
    )
    jax.jit(first_bound)(state).block_until_ready()
    jax.jit(second_bound)(state).block_until_ready()
    assert len(traces) == 2
