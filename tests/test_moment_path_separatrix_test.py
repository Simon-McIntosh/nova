"""The current-moment path tests the separatrix with no open-field-line closure."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.source import (
    DomainProfile,
    ForwardSource,
    PolynomialFluxFunction,
    _FluxSelectedProfile,
)
from nova.jax.config import configure_dtypes


def _closed_source() -> ForwardSource:
    """Return a source declaring no open-domain closure at all."""

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    return ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.asarray([2.0, -0.75])),
            ff_prime=PolynomialFluxFunction(jnp.asarray([0.5, 0.25])),
        )
    )


def _profile_handed_to_quadrature(source: ForwardSource, psi_norm) -> object:
    """Capture the profile argument ``current_moments`` passes to the quadrature."""

    seen: dict[str, object] = {}

    def quadrature(profile, flux, sample_flux, profile_support):
        seen["profile"] = profile
        zeros = jnp.zeros_like(jnp.asarray(flux))
        return (zeros, zeros, zeros)

    masks = DomainMasks(
        label=jnp.full(jnp.shape(jnp.asarray(psi_norm)), PlasmaDomain.CORE),
        psi_norm=jnp.asarray(psi_norm),
    )
    source.current_moments(masks, quadrature, None)
    return seen["profile"]


def test_moment_path_applies_the_separatrix_test_at_the_seed() -> None:
    """Every flux evaluation is clipped at psi_norm one without a closure."""

    source = _closed_source()
    assert source.common_sol is None
    profile = _profile_handed_to_quadrature(source, [0.5, 1.2])
    assert isinstance(profile, _FluxSelectedProfile)

    radius = jnp.asarray(1.0)
    above = float(profile.current_density(radius, jnp.asarray(1.2)))
    inside = float(profile.current_density(radius, jnp.asarray(0.5)))
    core = float(source.core.current_density(radius, jnp.asarray(0.5)))

    assert above == 0.0
    assert core != 0.0
    assert inside == core
