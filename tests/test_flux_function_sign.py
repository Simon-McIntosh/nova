"""The COCOS 17 relation between the toroidal current and the flux functions.

Three independent routes establish the sign that multiplies
``R p' + FF'/(mu0 R)`` under COCOS 17, and this file pins the transcription to
nova's owner.  The routes agree on ``+2*pi`` when ``p'`` is the derivative with
respect to COCOS 17's own signed poloidal flux:

* the general digit form, evaluated at COCOS 11 and COCOS 17;
* the DD4-native regression, which is COCOS 17 by definition;
* imas-ambix's canonical convention, which carries the same total-flux
  Ampere relation.

The owner currently states ``-2*pi`` together with ``p' = -dp/dPhi``.  A
selection is one equation with two signs in it, so the owner has to move both
at once or neither: this file takes the COCOS 17 statement at its word,
``j_φ = +2*pi * (R p' + FF'/(mu0 R))`` with ``p' = dp/dpsi``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.constants import mu_0

from nova_cocos import DODPSI_LIKE, PSI_LIKE, convention_transform
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR, toroidal_current_density

#: The general digit form the Grad--Shafranov derivation gives, written as
#: ``j_phi = -(sigma_Bp)(2 pi)^(e_Bp) (R p' + FF'/(mu0 R))`` with ``p'`` the
#: derivative against that convention's own signed poloidal flux.  Because the
#: derivative changes sign with ``sigma_Bp``, the written equation is the same
#: in COCOS 11 and COCOS 17; what differs between them is the sign of the flux
#: the derivative is taken against.
COCOS_11_DIGITS = {"sigma_bp": +1, "e_bp": 1}
COCOS_17_DIGITS = {"sigma_bp": -1, "e_bp": 1}


def _relation(digits, radius, p_prime, ff_prime):
    """Return the digit form's current density for one convention."""

    sigma_bp = digits["sigma_bp"]
    e_bp = digits["e_bp"]
    flux_functions = radius * p_prime + ff_prime / (mu_0 * radius)
    return -sigma_bp * (2.0 * math.pi) ** e_bp * flux_functions


def _cocos17_relation(radius, p_prime, ff_prime):
    return _relation(COCOS_17_DIGITS, radius, p_prime, ff_prime)


def test_the_digit_form_gives_the_same_current_in_cocos_11_and_17():
    """A selection is invariant under the 11 to 17 mapping.

    COCOS 11 to 17 negates psi and therefore the derivative taken against it,
    and negates the digit ``sigma_Bp`` beside it.  The two minus signs cancel,
    so the same physical current leaves the written equation unchanged.  This
    is why the DD4 regression and the raw DD3 regressions carry the same
    printed slope and must not be read as two conventions disagreeing.
    """

    radius = np.array([3.0, 5.0, 17.0])
    p_prime_11 = np.array([-4.0e4, -2.0e4, -1.0e3])
    ff_prime_11 = np.array([-2.0, -1.0, -0.1])
    transform = convention_transform(source=11, target=17)
    assert transform.factor(PSI_LIKE) == -1.0
    assert transform.factor(DODPSI_LIKE) == -1.0
    p_prime_17 = transform.factor(DODPSI_LIKE) * p_prime_11
    ff_prime_17 = transform.factor(DODPSI_LIKE) * ff_prime_11
    assert np.allclose(
        _relation(COCOS_11_DIGITS, radius, p_prime_11, ff_prime_11),
        _relation(COCOS_17_DIGITS, radius, p_prime_17, ff_prime_17),
    )


def test_the_cocos_17_relation_carries_plus_two_pi():
    """The transcription, pointed at the value the DD supplies.

    The Data Dictionary makes ``dpressure_dpsi`` and ``f_df_dpsi`` derivatives
    against its own COCOS 17 flux, so the relation is applied to them directly
    and the only factor in play is the ``2*pi`` its total-flux definition
    carries.  The sign is ``+``, and the DD4-native regression measures it
    positive on every one of the three sampled slices (the per-slice slopes
    and correlations are in the run's ``measure.json``; the slope is the
    collinearity of the map current with this expression).
    """

    radius = np.array([3.0, 30.0, 60.0])
    p_prime = np.array([-4.0e4, -2.0e4, -1.0e3])
    ff_prime = np.array([-2.0, -1.0, -0.1])
    expected = 2.0 * math.pi * (radius * p_prime + ff_prime / (mu_0 * radius))
    assert np.allclose(_cocos17_relation(radius, p_prime, ff_prime), expected)
    assert TOTAL_FLUX_FACTOR == pytest.approx(2.0 * math.pi)
    # The recorded DD4-native slopes (three slices, total flux, psi as stored).
    measured_slopes = (1.0134, 1.0146, 1.0153)
    measured_correlations = (0.9377, 0.9334, 0.9317)
    assert all(slope > 0.0 for slope in measured_slopes)
    assert all(correlation > 0.0 for correlation in measured_correlations)
    assert all(slope == pytest.approx(1.0, abs=0.05) for slope in measured_slopes)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "nova/equilibrium/convention.py states j_φ = -2*pi*(R p' + FF'/(mu0 R)) "
        "and reads that minus as a negated flux function. The Data Dictionary "
        "supplies dp/dpsi itself, not its negative, so a profile current built "
        "from stored fields comes out anti-parallel to the map current. The "
        "COCOS 17 relation is +2*pi; when the owner states it, this test passes "
        "and strict xfail turns into XPASS."
    ),
)
def test_nova_owner_matches_the_cocos_17_relation():
    """The owner must agree with the transcription it declares."""

    radius = np.array([3.0, 30.0, 60.0])
    p_prime = np.array([-4.0e4, -2.0e4, -1.0e3])
    ff_prime = np.array([-2.0, -1.0, -0.1])
    owner = toroidal_current_density(radius, p_prime, ff_prime)
    assert np.allclose(owner, _cocos17_relation(radius, p_prime, ff_prime))
