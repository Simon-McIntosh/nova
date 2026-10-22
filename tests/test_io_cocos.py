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

from nova_cocos import (
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


# Sauter & Medvedev CPC 184 (2013) Table I, transcribed here from the paper so
# this file is an independent source for the published constants rather than a
# second reading of the module's own table.  The 1-8 block and the 11-18 block
# carry the same (sigma_Bp, sigma_RphiZ, sigma_rho_theta_phi) in the same order
# and differ only in e_Bp: the second block is the first with the 2*pi moved
# into the flux, which is why a convention and its flux scaling differ by ten.
_TABLE_I_BASE: dict[int, tuple[int, int, int]] = {
    1: (+1, +1, +1),
    2: (+1, -1, +1),
    3: (-1, +1, -1),
    4: (-1, -1, -1),
    5: (+1, +1, -1),
    6: (+1, -1, -1),
    7: (-1, +1, +1),
    8: (-1, -1, +1),
}
_FLUX_CARRYING_BLOCK = 10


def _table_i() -> dict[int, tuple[int, int, int, int]]:
    """Return Table I as (sigma_Bp, e_Bp, sigma_RphiZ, sigma_rho_theta_phi)."""

    table = {}
    for identifier, (sigma_bp, sigma_r_phi_z, sigma_rho_phi_z) in _TABLE_I_BASE.items():
        table[identifier] = (sigma_bp, 0, sigma_r_phi_z, sigma_rho_phi_z)
        table[identifier + _FLUX_CARRYING_BLOCK] = (
            sigma_bp,
            1,
            sigma_r_phi_z,
            sigma_rho_phi_z,
        )
    return table


def test_table_i_is_transcribed_independently_of_the_module():
    """Every row of the published table identifies the convention it came from.

    The transcription above is the independent source: the module's own digits
    table is compared against it, so a digit changed on either side has to
    disagree with the other rather than with itself.
    """

    table = _table_i()
    assert len(table) == 16
    for identifier, digits in table.items():
        assert identify_convention(**dict(zip(_DIGIT_NAMES, digits))) == identifier
    assert {key: tuple(value) for key, value in CONVENTION_DIGITS.items()} == table


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
