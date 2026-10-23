"""Contracts for the internal COCOS 17 sign sites.

Each test fixes a known COCOS 17 state and checks the site's sign-sensitive
output against the value the site produced before its bare negation was
replaced by a sign taken from the convention owner.
"""

from __future__ import annotations

from math import tau

import numpy as np

from nova.equilibrium.convention import (
    COCOS,
    CONVENTION,
    SIGMA_BP,
    TOTAL_FLUX_FACTOR,
)
from nova.imas.pulsedesign import psi_write_factor
from nova.io.sourcemap import (
    DD3_CONVENTION,
    DD4_CONVENTION,
    NOVA_CONVENTION,
    dd_equilibrium_convention,
)
from nova_cocos import convention


def test_the_internal_convention_is_stated_once():
    """nova's internal convention is COCOS 17, read from the one digits table."""
    assert CONVENTION.identifier == 17
    assert COCOS == 17
    assert CONVENTION.digits == convention(17).digits
    assert SIGMA_BP == -1


def test_the_total_flux_factor_follows_the_convention():
    """The 2*pi on the total flux is the convention's own e_Bp digit."""
    assert CONVENTION.e_bp == 1
    assert TOTAL_FLUX_FACTOR == tau ** CONVENTION.e_bp


def test_a_dd3_write_carries_the_flux_sign_flip():
    """A table writer's psi entries keep the sign they had before the refactor."""
    assert dd_equilibrium_convention("3.42.0") == DD3_CONVENTION
    assert dd_equilibrium_convention("3.38.1") == DD3_CONVENTION
    assert psi_write_factor(DD3_CONVENTION) == -1.0


def test_a_dd4_write_passes_the_internal_flux_through():
    """A DD4 write stores the internal COCOS 17 flux unchanged."""
    assert dd_equilibrium_convention("4.1.0") == DD4_CONVENTION
    assert psi_write_factor(DD4_CONVENTION) == 1.0


def test_the_written_psi_keeps_its_value():
    """A stored COCOS 17 flux is negated to DD3 exactly as the literal did."""
    internal = np.array([100.0, -50.0, 12.5])
    base = -internal
    assert np.array_equal(psi_write_factor(DD3_CONVENTION) * internal, base)


def test_the_flux_contour_levels_keep_their_sign():
    """The plotted psi levels carry the internal convention's sigma_Bp sign."""
    levels = np.linspace(-7.5, 107.8, 51)
    base = -levels[::-1]
    site = convention(NOVA_CONVENTION).sigma_bp * levels[::-1]
    assert np.array_equal(site, base)


def test_the_declared_conventions_name_the_dd_versions():
    """The owner's COCOS equals the convention the DD writers declare."""
    assert DD3_CONVENTION == 11
    assert DD4_CONVENTION == 17
    assert NOVA_CONVENTION == COCOS