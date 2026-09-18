"""Unit checks of the Shafranov combination discriminator's own arithmetic.

These tests never open the bank store: the identities, the boundary shape, the
attribution sentence and the unit-check recomputation are pure functions of
their arguments, and they are the parts whose failure would silently misreport
the five readings.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.shafranov_combination_discriminator import (
    READING_KEYS,
    attribute,
    boundary_shape,
    identity_combination,
    linear_combination,
    readings,
    recomputed_profile_moments,
)
from nova.biot.greens import MU0
from nova.equilibrium.diagnostics import shafranov_vertical_field


def test_identity_inversion_round_trips_the_forward_identity():
    """The inverted combination reproduces the field the identity predicts."""
    current, radius, minor = 801493.25, 0.86, 0.5847
    for combination in (-0.5, 0.0, 0.3126, 2.0):
        field = shafranov_vertical_field(current, radius, minor, combination)
        assert identity_combination(
            current, radius, minor, field
        ) == pytest.approx(combination, rel=1.0e-12)


def test_identity_inversion_refuses_geometry_the_forward_identity_refuses():
    """A zero current or a radius outside (0, 8R) is NaN, as forward is."""
    assert np.isnan(identity_combination(0.0, 0.86, 0.5, -0.01))
    assert np.isnan(identity_combination(1.0e6, 0.86, 8.0 * 0.86, -0.01))
    assert np.isnan(identity_combination(1.0e6, 0.86, 0.5, np.nan))


def test_boundary_shape_reads_half_extents_of_the_stored_ring():
    """A rectangular ring gives its half width as a and its aspect as kappa."""
    radial = np.linspace(0.4, 1.0, 41)
    height = np.linspace(-0.6, 0.6, 41)
    ring = np.vstack(
        (
            np.column_stack((radial, np.full_like(radial, -0.6))),
            np.column_stack((np.full_like(height, 1.0), height)),
            np.column_stack((radial, np.full_like(radial, 0.6))),
            np.column_stack((np.full_like(height, 0.4), height)),
        )
    )
    minor, elongation = boundary_shape(ring)
    assert minor == pytest.approx(0.3)
    assert elongation == pytest.approx(2.0)


def test_linear_combination_is_the_row_quantity():
    """The combination is beta_p + l_i/2, never either moment alone."""
    assert linear_combination(0.1985, 0.81042) == pytest.approx(0.60371, rel=1e-5)


def test_attribute_places_efit_and_names_the_closing_reading():
    """EFIT above the profiles closes on the nearest magnetics reading."""
    combination = {
        "efit_own": 0.60,
        "profiles": 0.45,
        "magnetics_circular": 0.31,
        "magnetics_elongated": 0.58,
        "magnetics_discrete": 0.31,
    }
    verdict = attribute(combination)
    assert verdict["closing_reading"] == "magnetics_elongated"
    assert "above" in verdict["sentence"]
    assert "0.5800" in verdict["sentence"]


def test_attribute_reports_below_when_efit_undershoots_the_profiles():
    """The side is stated from EFIT's own value and the profiles' value."""
    verdict = attribute(
        {
            "efit_own": 0.20,
            "profiles": 0.45,
            "magnetics_circular": 0.31,
            "magnetics_elongated": None,
            "magnetics_discrete": None,
        }
    )
    assert "below" in verdict["sentence"]
    assert verdict["closing_reading"] == "magnetics_circular"


def test_unit_check_recomputes_beta_and_inductance_from_the_integrals():
    """A misplaced mu0 or leading factor shows up as a relative difference."""
    current, radius = 8.0e5, 0.86
    pressure, field = 3.0e4, 1.0e-1
    beta = 4.0 * pressure / (MU0 * radius * current**2)
    inductance = 2.0 * field / (MU0 * MU0 * radius * current**2)
    observation = SimpleNamespace(
        plasma_current=current,
        major_radius=radius,
        pressure_integral=pressure,
        poloidal_field_integral=field,
        poloidal_beta=beta,
        internal_inductance=inductance,
    )
    check = recomputed_profile_moments(observation)
    assert check["poloidal_beta_relative_difference"] == pytest.approx(0.0, abs=1e-12)
    assert check["internal_inductance_relative_difference"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert check["mu0_h_m"] == MU0
    assert "mu0**2" in check["definition"]


def test_reading_keys_are_ordered_and_labelled():
    """Every reading the panel draws carries its label in one order."""
    combination = dict.fromkeys(READING_KEYS, 0.0)
    block = readings(combination)
    assert list(block) == list(READING_KEYS)
    assert all(entry["label"] for entry in block.values())