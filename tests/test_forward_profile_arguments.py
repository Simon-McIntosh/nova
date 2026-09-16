"""Program arguments for member-varying forward flux functions."""

from __future__ import annotations

from dataclasses import replace
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
from tests.test_forward_compile_identity import CASES, _certificate_row
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


def test_profile_component_image_is_the_normalisation_tangent() -> None:
    """The pressure-amplitude JVP agrees on a qualified certificate state."""
    from benchmarks.profile_coefficients_recompile_audit import (
        _scaled_row,
        _scaled_source,
    )

    closure_row = _certificate_row(CASES[0])
    argument_source = _scaled_source(closure_row[0].source, 1.0, 1.0)
    profile, state, requested_class, target_current, _request = _scaled_row(
        closure_row, argument_source
    )
    operator = profile.operator
    amplitude = jnp.asarray(0.25)

    tangent = operator.profile_component_image(
        state,
        component="pressure_gradient",
        amplitude=amplitude,
        requested_class=requested_class,
        target_current=target_current,
    )

    def source_at(scale):
        function = argument_source.core.p_prime
        varied = PolynomialFluxFunction(
            function.coefficients,
            function.normalisation * scale,
        )
        return replace(
            argument_source, core=replace(argument_source.core, p_prime=varied)
        )

    def central_difference(step):
        upper = operator.with_source(source_at(1.0 + step)).internal(
            state, requested_class, target_current
        )
        lower = operator.with_source(source_at(1.0 - step)).internal(
            state, requested_class, target_current
        )
        return amplitude * (upper - lower) / (2.0 * step), upper, lower

    coarse, _coarse_upper, _coarse_lower = central_difference(1.0e-3)
    fine, fine_upper, fine_lower = central_difference(5.0e-4)
    error = jnp.max(jnp.abs(tangent - fine))
    truncation = jnp.max(jnp.abs(coarse - fine))
    image_scale = jnp.maximum(
        jnp.max(jnp.abs(fine_upper)), jnp.max(jnp.abs(fine_lower))
    )
    roundoff = (
        32.0 * jnp.finfo(state.dtype).eps * jnp.abs(amplitude) * image_scale / 5.0e-4
    )
    earned_tolerance = 2.0 * truncation + roundoff
    figures = {
        "maximum_tangent": float(jnp.max(jnp.abs(tangent))),
        "maximum_error": float(error),
        "truncation_estimate": float(truncation),
        "roundoff_floor": float(roundoff),
        "earned_tolerance": float(earned_tolerance),
    }
    print(f"PROFILE_COMPONENT_FIGURES {figures}")

    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.max(jnp.abs(tangent)) > 0.0
    assert error <= earned_tolerance, figures
