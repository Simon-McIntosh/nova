"""Public solve propagation of traced prescribed conductor currents."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    PrescribedCurrentField,
)
from nova.equilibrium.topology import TopologyClass


@dataclass(frozen=True)
class _LinearTarget:
    """Minimal conductor target exposing one fixed response matrix."""

    response: jax.Array

    @property
    def node_number(self) -> int:
        return self.response.shape[0]

    def external(self, current) -> jax.Array:
        return self.response @ current


@dataclass(frozen=True)
class _Lattice:
    """Minimal lattice carrying the operator node count."""

    node_count: int


def _profile() -> tuple[ForwardProfile, jax.Array, jax.Array]:
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
    operator.source = SimpleNamespace(closure_degrees=0)
    operator.internal = lambda psi, requested_class=None, target_current=None: (
        jnp.zeros_like(psi)
    )
    operator._exclude_shadow_residual = (
        lambda psi, mapped, requested_class=None, shadow=None: mapped
    )
    operator.residual_shadow_mask = (
        lambda psi, requested_class=None, previous_shadow=None: jnp.zeros_like(
            psi, dtype=bool
        )
    )
    operator.read = lambda flux, requested_class=None: (
        None,
        SimpleNamespace(diverted=jnp.asarray(TopologyClass.DIVERTED)),
    )
    profile = ForwardProfile(
        operator=operator,
        lattice=_Lattice(operator.grid.node_number),
        evaluations=1,
        relaxation=1.0,
    )
    profile._receipt = lambda flux, history, *args, **kwargs: SimpleNamespace(
        flux=flux,
        fixed_point=history,
        finite=SimpleNamespace(passed=jnp.asarray(True)),
    )
    return profile, ordinary_response, prescribed_response


def test_solve_none_reproduces_the_stored_vector_bit_identically():
    profile, _ordinary_response, _prescribed_response = _profile()
    seed = jnp.zeros(4)
    options = {"route": "picard", "evaluations": 1, "relaxation": 1.0}

    omitted = profile.solve(seed, **options).flux
    explicit_none = profile.solve(seed, prescribed_current=None, **options).flux
    explicit_stored = profile.solve(
        seed,
        prescribed_current=profile.operator.prescribed_field.current,
        **options,
    ).flux

    np.testing.assert_array_equal(explicit_none, omitted)
    np.testing.assert_array_equal(explicit_stored, omitted)


def test_solve_and_solve_branch_trace_prescribed_current_replacements():
    profile, ordinary_response, prescribed_response = _profile()
    seed = jnp.zeros(4)
    edited = jnp.asarray((-4.0, 9.0, 13.0))
    expected = (
        ordinary_response @ profile.operator.external_current
        + prescribed_response @ edited
    )

    solved = jax.jit(
        lambda prescribed: (
            profile.solve(
                seed,
                route="picard",
                prescribed_current=prescribed,
                evaluations=1,
                relaxation=1.0,
            ).flux
        )
    )
    branched = jax.jit(
        lambda prescribed: (
            profile.solve_branch(
                seed,
                TopologyClass.DIVERTED,
                route="picard",
                prescribed_current=prescribed,
                evaluations=1,
                relaxation=1.0,
                tolerance=jnp.inf,
            ).equilibrium.flux
        )
    )

    np.testing.assert_array_equal(solved(edited), expected)
    np.testing.assert_array_equal(branched(edited), expected)
    np.testing.assert_array_equal(
        solved(edited + 1.0),
        ordinary_response @ profile.operator.external_current
        + prescribed_response @ (edited + 1.0),
    )
    np.testing.assert_array_equal(
        branched(edited + 1.0),
        ordinary_response @ profile.operator.external_current
        + prescribed_response @ (edited + 1.0),
    )
    assert solved._cache_size() == 1
    assert branched._cache_size() == 1
