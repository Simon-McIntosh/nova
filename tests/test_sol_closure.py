import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.sol_closure import (
    EICH_FIELD_EXPONENT,
    EICH_WIDTH_COEFFICIENT_M,
    EichSolClosure,
    SolDecayVariant,
    eich_width,
)
from nova.equilibrium.source import DomainProfile, ForwardSource

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def confined_profile():
    return DomainProfile(
        p_prime=lambda psi_norm: 1.8e5 + 2.5e4 * (psi_norm - 1.0),
        ff_prime=lambda psi_norm: 0.42 - 0.08 * (psi_norm - 1.0),
    )


@pytest.fixture
def width():
    return eich_width(
        outboard_midplane_radius_m=6.2,
        radial_poloidal_field_t=0.3,
        vertical_poloidal_field_t=0.4,
        flux_span_wb=2.7,
    )


@pytest.fixture
def closure(width):
    return EichSolClosure(
        width=width,
        spreading_length_m=4.0 * width.heat_flux_width_m,
        spreading_fraction=0.35,
    )


def test_eich_width_uses_the_equilibrium_outboard_midplane_field(width):
    expected_physical = EICH_WIDTH_COEFFICIENT_M * 0.5**EICH_FIELD_EXPONENT
    expected_normalized = 2.0 * np.pi * 6.2 * 0.5 * expected_physical / 2.7

    assert width.outboard_midplane_poloidal_field_t == pytest.approx(0.5)
    assert width.heat_flux_width_m == pytest.approx(expected_physical)
    assert width.normalized_flux_width == pytest.approx(expected_normalized)


@pytest.mark.parametrize("variant", list(SolDecayVariant))
def test_sol_current_matches_confined_value_and_first_derivative(
    confined_profile, closure, variant
):
    radius = closure.width.outboard_midplane_radius_m
    continued = closure.domain_profile(confined_profile, variant)

    def confined_current(psi_norm):
        return confined_profile.current_density(radius, psi_norm)

    def sol_current(psi_norm):
        return continued.current_density(radius, psi_norm)

    assert sol_current(jnp.asarray(1.0)) == pytest.approx(
        confined_current(jnp.asarray(1.0)), rel=2.0e-14
    )
    assert jax.grad(sol_current)(jnp.asarray(1.0)) == pytest.approx(
        jax.grad(confined_current)(jnp.asarray(1.0)), rel=2.0e-13
    )


def test_material_mask_is_the_only_common_sol_support(confined_profile, closure):
    width = closure.width.normalized_flux_width
    psi_norm = jnp.asarray(
        [
            0.7,
            1.0 + 2.0 * width,
            1.0 + 20.0 * width,
            0.15,
            1.0 + 4.0 * width,
        ]
    )
    labels = jnp.asarray(
        [
            PlasmaDomain.CORE,
            PlasmaDomain.COMMON_SOL,
            PlasmaDomain.COMMON_SOL,
            PlasmaDomain.PRIVATE_FLUX,
            PlasmaDomain.EXCLUDED_MATERIAL,
        ],
        dtype=jnp.int8,
    )
    masks = DomainMasks(label=labels, psi_norm=psi_norm)
    radius = jnp.full(psi_norm.shape, closure.width.outboard_midplane_radius_m)

    profile = closure.domain_profile(confined_profile, SolDecayVariant.SINGLE_LENGTH)
    source = ForwardSource(core=confined_profile, common_sol=profile)
    total = np.asarray(source.current_density(radius, masks))
    sol_only = np.asarray(
        closure.current_density(
            confined_profile,
            radius,
            masks,
            SolDecayVariant.SINGLE_LENGTH,
        )
    )

    assert np.isinf(float(profile.continuation_record().support))
    assert float(profile.continuation_record().truncated_fraction) == 0.0
    assert sol_only[1] != 0.0
    assert sol_only[2] != 0.0
    assert sol_only[3] == 0.0
    assert sol_only[4] == 0.0
    assert total[3] == 0.0
    assert total[4] == 0.0


def test_dual_length_tail_and_measured_support_extent(confined_profile, closure):
    single_extent = closure.support_extent(
        confined_profile, SolDecayVariant.SINGLE_LENGTH
    )
    dual_extent = closure.support_extent(confined_profile, SolDecayVariant.DUAL_LENGTH)
    radius = closure.width.outboard_midplane_radius_m

    def fraction_at(variant, psi_norm):
        profile = closure.domain_profile(confined_profile, variant)
        boundary = abs(float(profile.current_density(radius, jnp.asarray(1.0))))
        current = abs(float(profile.current_density(radius, jnp.asarray(psi_norm))))
        return current / boundary

    assert single_extent > 1.0
    assert dual_extent > single_extent
    assert fraction_at(SolDecayVariant.SINGLE_LENGTH, single_extent) == pytest.approx(
        1.0e-6, rel=2.0e-10
    )
    assert fraction_at(SolDecayVariant.DUAL_LENGTH, dual_extent) == pytest.approx(
        1.0e-6, rel=2.0e-10
    )

    probe = 1.0 + 12.0 * closure.width.normalized_flux_width
    assert fraction_at(SolDecayVariant.DUAL_LENGTH, probe) > fraction_at(
        SolDecayVariant.SINGLE_LENGTH, probe
    )


def test_material_bounded_support_is_only_valid_for_common_sol(
    confined_profile, closure
):
    policy = closure.policy(SolDecayVariant.SINGLE_LENGTH)

    with pytest.raises(ValueError, match="private-flux branch"):
        policy.extend(confined_profile, PlasmaDomain.PRIVATE_FLUX)
