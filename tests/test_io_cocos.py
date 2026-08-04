"""The convention algebra, and the factors it must give for the MAST sources.

Every factor here is derived from the four digits rather than written down, so the
tests are about the algebra's properties -- self-inverse, identity, agreement with
the independently pinned convention table -- and then about the specific numbers
the MAST ledger states for its own sources.  A test asserting only the numbers
would pass with a hand-written lookup table and tell us nothing about a convention
pair nobody has tried yet.
"""

from __future__ import annotations

import math

import pytest

from nova.io.cocos import (
    B0_LIKE,
    CONVENTION_DIGITS,
    DODPSI_LIKE,
    IP_LIKE,
    ONE_LIKE,
    PSI_LIKE,
    Q_LIKE,
    TRANSFORMATIONS,
    ConventionError,
    convention,
    convention_transform,
    conventions,
    identify_convention,
    transform_factor,
)
from nova.scripts.identify_source_cocos import cocos_from_digits

MAST_SOURCE = 3
MAST_MEASURED_FLUX = 13
DICTIONARY_TARGET = 17


def test_every_convention_is_its_own_digits():
    """A convention identified from its digits is the convention it came from."""

    assert len(CONVENTION_DIGITS) == 16
    for entry in conventions():
        assert identify_convention(**dict(zip(_DIGIT_NAMES, entry.digits))) == (
            entry.identifier
        )


_DIGIT_NAMES = ("sigma_bp", "e_bp", "sigma_r_phi_z", "sigma_rho_theta_phi")


def test_the_table_agrees_with_the_pinned_source_verdict():
    """The digits here are the ones the source-convention audit was pinned against.

    Two tables of the same published constant is one too many unless they are
    checked against each other, so this is the check: the audit's reverse lookup
    and this module's forward one must name the same convention for all sixteen.
    """

    for entry in conventions():
        assert cocos_from_digits(**dict(zip(_DIGIT_NAMES, entry.digits))) == (
            entry.identifier
        )


@pytest.mark.parametrize("transformation", sorted(TRANSFORMATIONS))
def test_a_convention_to_itself_changes_nothing(transformation):
    """The identity transform is the identity, whichever quantity it is asked about."""

    for entry in conventions():
        factor = transform_factor(
            transformation, source=entry.identifier, target=entry.identifier
        )
        assert factor == pytest.approx(1.0)


@pytest.mark.parametrize("transformation", sorted(TRANSFORMATIONS))
def test_every_transform_is_undone_by_its_inverse(transformation):
    """Swapping source and target must give the reciprocal factor, for every pair."""

    for source in CONVENTION_DIGITS:
        for target in CONVENTION_DIGITS:
            forward = convention_transform(source=source, target=target)
            backward = forward.inverse()
            product = forward.factor(transformation) * backward.factor(transformation)
            assert product == pytest.approx(1.0), (source, target)


def test_the_mast_source_transform_matches_the_ledger():
    """COCOS 3 to 17: psi scales by 2*pi, q flips, the currents are untouched."""

    factors = {
        name: transform_factor(name, source=MAST_SOURCE, target=DICTIONARY_TARGET)
        for name in TRANSFORMATIONS
    }
    assert factors[PSI_LIKE] == pytest.approx(math.tau)
    assert factors[DODPSI_LIKE] == pytest.approx(1.0 / math.tau)
    assert factors[Q_LIKE] == pytest.approx(-1.0)
    assert factors[IP_LIKE] == pytest.approx(1.0)
    assert factors[B0_LIKE] == pytest.approx(1.0)
    assert factors[ONE_LIKE] == pytest.approx(1.0)


def test_a_measured_total_flux_needs_no_further_two_pi():
    """The source's senses with the 2*pi already in the flux is ten places along.

    This is what lets a measured flux loop in Weber and a reconstructed flux
    function in Weber per radian share one algebra: they are the same physics in
    two conventions differing only in that digit, so declaring which one an array
    is in gets each the right factor without either being special-cased.
    """

    source = convention(MAST_SOURCE)
    measured = convention(MAST_MEASURED_FLUX)
    assert measured.e_bp == source.e_bp + 1
    assert measured.sigma_bp == source.sigma_bp
    assert measured.sigma_r_phi_z == source.sigma_r_phi_z
    assert measured.sigma_rho_theta_phi == source.sigma_rho_theta_phi
    assert transform_factor(
        PSI_LIKE, source=MAST_MEASURED_FLUX, target=DICTIONARY_TARGET
    ) == pytest.approx(1.0)
    # the sign-bearing digits are untouched by moving the 2*pi, so q still flips
    assert transform_factor(
        Q_LIKE, source=MAST_MEASURED_FLUX, target=DICTIONARY_TARGET
    ) == pytest.approx(-1.0)


def test_the_flux_scale_is_the_reciprocal_of_the_flux_derivative_scale():
    """A derivative with respect to the flux must carry the inverse of its scale."""

    for source in CONVENTION_DIGITS:
        flux = transform_factor(PSI_LIKE, source=source, target=DICTIONARY_TARGET)
        derivative = transform_factor(
            DODPSI_LIKE, source=source, target=DICTIONARY_TARGET
        )
        assert flux * derivative == pytest.approx(1.0), source


def test_a_poloidal_angle_is_not_carried_between_conventions():
    """The refusal is deliberate: an installed orientation is measured, not converted.

    Two landed records disagree on the poloidal-angle factor for these very
    conventions, and the described probe orientations were authored from the
    source's own measured angles rather than transformed, so no solve-input row
    needs one.  Offering a factor here would let a future caller pick up the
    disagreement silently.
    """

    with pytest.raises(ConventionError, match="no factor is derived"):
        transform_factor("pol_angle_like", source=MAST_SOURCE, target=DICTIONARY_TARGET)


def test_an_unknown_convention_is_refused():
    """A convention outside the sixteen is an error, not a pass-through."""

    with pytest.raises(ConventionError, match="not one of the sixteen"):
        convention(9)
    with pytest.raises(ConventionError, match="no convention has digits"):
        identify_convention(
            sigma_bp=+1, e_bp=2, sigma_r_phi_z=+1, sigma_rho_theta_phi=+1
        )
