"""Contracts for traced constraint bindings and direction selection."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import (
    CompensatorRule,
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    select_compensating_directions,
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


def test_banked_solovev_direction_selection_matches_record() -> None:
    """The fallback Solovev bank preserves the recorded selection diagnostics."""
    path = Path(__file__).parent / "data" / "solovev_constraint_response.npz"
    with np.load(path) as bank:
        response = np.asarray(bank["response"])
        drivable = np.asarray(bank["drivable"])
        selection = select_compensating_directions(response, circuits=drivable)

        assert selection.rule is CompensatorRule[bank["rule"].item().upper()]
        assert selection.leading_circuits(0, count=3) == tuple(
            np.asarray(bank["leading_circuits"]).tolist()
        )
        np.testing.assert_allclose(
            selection.singular_values,
            bank["singular_values"],
            rtol=1.0e-9,
            atol=0.0,
        )


def test_exports_match_constraint_module_objects() -> None:
    from nova.equilibrium import (
        CompensatorRule,
        CompensatorSelection,
        constraint_response_matrix,
        derive_circuit_compensators,
    )
    from nova.equilibrium.constraint import (
        CompensatorRule as module_CompensatorRule,
        CompensatorSelection as module_CompensatorSelection,
        constraint_response_matrix as module_constraint_response_matrix,
        derive_circuit_compensators as module_derive_circuit_compensators,
    )

    assert CompensatorRule is module_CompensatorRule
    assert CompensatorSelection is module_CompensatorSelection
    assert constraint_response_matrix is module_constraint_response_matrix
    assert derive_circuit_compensators is module_derive_circuit_compensators
