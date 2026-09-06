"""Contracts for traced constraint bindings and direction selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
)


class _ScalarFunctional:
    """Minimal one-row functional for checking the static pair layout."""

    row_count = 1


def test_list_typed_binding_leaves_survive_a_jitted_tree_read() -> None:
    """Plain lists become JAX leaves before pair validation and tracing."""
    pair = ConstraintPair(
        functional=_ScalarFunctional(),
        unknown=ConstraintMultiplier([1.0]),
        binding=ConstraintBinding(
            target=[2.0],
            tolerance=[1.0e-8],
            scale=[3.0],
            initial_unknown=[0.25],
        ),
    )

    @jax.jit
    def read_leaves(value):
        binding = value.binding
        return jnp.concatenate(
            (
                binding.target,
                binding.tolerance,
                binding.scale,
                binding.initial_unknown,
            )
        )

    np.testing.assert_allclose(read_leaves(pair), [2.0, 1.0e-8, 3.0, 0.25])
