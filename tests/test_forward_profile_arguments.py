"""Program arguments for member-varying forward flux functions."""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import pytest

from nova.equilibrium.source import (
    DomainProfile,
    ForwardSource,
    PolynomialFluxFunction,
)
from nova.jax.config import configure_dtypes
from tests.test_forward_operator_arguments import _operator


def _source(pressure, diamagnetic, *, pressure_scale=1.0, diamagnetic_scale=1.0):
    """Return one fixed-order source with independently traced physical scales."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    return ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(
                jnp.asarray(pressure), jnp.asarray(pressure_scale)
            ),
            ff_prime=PolynomialFluxFunction(
                jnp.asarray(diamagnetic), jnp.asarray(diamagnetic_scale)
            ),
        )
    )


def _operator_with_source(source):
    """Put an explicit source representation on the compact operator fixture."""
    template = _operator()
    active = object.__new__(type(template))
    active.__dict__ = template.__dict__.copy()
    active.source = source
    return active


def _digest(lowered) -> str:
    text = lowered.as_text(dialect="stablehlo")
    return hashlib.sha256(text.encode()).hexdigest()


def test_profile_coefficients_are_operator_arguments() -> None:
    """Same-order profiles share static identity and carry different leaves."""
    first = _operator_with_source(_source([2.0, -1.0], [0.5]))
    second = first.with_source(
        _source(
            [2.25, -0.75],
            [0.4],
            pressure_scale=0.9,
            diamagnetic_scale=1.1,
        )
    )

    assert first.program_identity == second.program_identity
    first_leaves = [
        *jax.tree_util.tree_leaves(first.source.core.p_prime),
        *jax.tree_util.tree_leaves(first.source.core.ff_prime),
    ]
    second_leaves = [
        *jax.tree_util.tree_leaves(second.source.core.p_prime),
        *jax.tree_util.tree_leaves(second.source.core.ff_prime),
    ]
    assert len(first_leaves) == len(second_leaves) == 4
    assert any(
        not jnp.array_equal(left, right)
        for left, right in zip(first_leaves, second_leaves, strict=True)
    )


def test_one_map_program_serves_two_profiles_and_two_current_targets() -> None:
    """Coefficient and current edits change operands without changing StableHLO."""
    first = _operator_with_source(_source([2.0], [0.5]))
    second = first.with_source(_source([1.8], [0.55]))
    state = jnp.linspace(-1.0, 1.0, first.node_number)
    target_marker = jnp.asarray(1.0)
    mapped = first.traced_flux_map(target_current=target_marker)

    @jax.jit
    def apply(flux, external, operator, target_current):
        return mapped(flux, external, operator, target_current)

    fixture = apply.lower(state, first.external(), first, jnp.asarray(2.0))
    changed_profile = apply.lower(state, second.external(), second, jnp.asarray(2.0))
    changed_target = apply.lower(state, first.external(), first, jnp.asarray(3.0))

    assert _digest(fixture) == _digest(changed_profile)
    assert _digest(fixture) == _digest(changed_target)


def test_incompatible_profile_representation_is_refused() -> None:
    """A request cannot silently exchange the static profile evaluator."""
    operator = _operator_with_source(_source([2.0], [0.5]))

    def pressure(value):
        return jnp.asarray(value)

    incompatible = ForwardSource(
        core=DomainProfile(p_prime=pressure, ff_prime=pressure)
    )
    with pytest.raises(ValueError, match="static flux-function representation"):
        operator.with_source(incompatible)
