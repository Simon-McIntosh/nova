"""Traced replacement contract for prescribed conductor currents."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    PrescribedCurrentField,
)


@dataclass(frozen=True)
class _LinearTarget:
    """Minimal conductor target exposing a fixed response matrix."""

    response: jax.Array

    @property
    def node_number(self) -> int:
        return self.response.shape[0]

    def external(self, current) -> jax.Array:
        return self.response @ current


def _operator() -> tuple[ForwardFluxOperator, jax.Array, jax.Array]:
    ordinary_response = jnp.asarray(((1.0, -2.0), (0.5, 3.0), (-4.0, 2.5), (1.5, 0.25)))
    prescribed_response = jnp.arange(12.0).reshape(4, 3) * 0.125
    stored_prescribed = jnp.asarray((11.0, -7.0, 2.0))
    operator = object.__new__(ForwardFluxOperator)
    operator.grid = _LinearTarget(ordinary_response[:2])
    operator.wall = _LinearTarget(ordinary_response[2:])
    operator.sample = None
    operator.external_current = jnp.asarray((3.0, -5.0))
    operator.prescribed_field = PrescribedCurrentField(
        response=prescribed_response,
        current=stored_prescribed,
    )
    operator.internal = lambda psi, requested_class=None, target_current=None: (
        jnp.zeros_like(psi)
    )
    operator._exclude_shadow_residual = (
        lambda psi, mapped, requested_class=None, shadow=None: mapped
    )
    return operator, ordinary_response, prescribed_response


def test_omitted_prescribed_current_keeps_stored_path_bit_identical():
    operator, ordinary_response, prescribed_response = _operator()
    expected = (
        ordinary_response @ operator.external_current
        + prescribed_response @ operator.prescribed_field.current
    )

    np.testing.assert_array_equal(operator.external(), expected)
    np.testing.assert_array_equal(operator.flux_map()(jnp.zeros(4)), expected)


def test_traced_prescribed_current_replaces_complete_stored_vector():
    operator, ordinary_response, prescribed_response = _operator()
    edited = jnp.asarray((-4.0, 9.0, 13.0))
    expected = (
        ordinary_response @ operator.external_current + prescribed_response @ edited
    )
    mapped = jax.jit(
        lambda prescribed: operator.flux_map(prescribed_current=prescribed)(
            jnp.zeros(4)
        )
    )

    np.testing.assert_array_equal(mapped(edited), expected)
    np.testing.assert_array_equal(
        mapped(edited + 1.0),
        ordinary_response @ operator.external_current
        + prescribed_response @ (edited + 1.0),
    )
    assert mapped._cache_size() == 1


def test_generic_and_prescribed_currents_are_independent_not_double_counted():
    operator, ordinary_response, prescribed_response = _operator()
    ordinary = jnp.asarray((17.0, -19.0))
    prescribed = jnp.asarray((23.0, 29.0, -31.0))
    expected = ordinary_response @ ordinary + prescribed_response @ prescribed

    actual = operator.external(
        current=ordinary,
        prescribed_current=prescribed,
    )

    np.testing.assert_array_equal(actual, expected)
    stored_then_added = (
        ordinary_response @ ordinary
        + prescribed_response @ operator.prescribed_field.current
        + prescribed_response @ prescribed
    )
    assert not np.array_equal(np.asarray(actual), np.asarray(stored_then_added))


def test_prescribed_current_override_requires_the_declared_policy_shape():
    operator, _ordinary_response, _prescribed_response = _operator()
    with pytest.raises(ValueError, match="stored circuit vector shape"):
        operator.external(prescribed_current=jnp.ones(2))

    operator.prescribed_field = None
    with pytest.raises(ValueError, match="requires a prescribed current field"):
        operator.external(prescribed_current=jnp.ones(3))
