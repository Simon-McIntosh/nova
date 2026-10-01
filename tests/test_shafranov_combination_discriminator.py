"""Unit checks of the Shafranov combination discriminator's own arithmetic.

These tests never open the bank store: the identities, the boundary shape, the
attribution sentence and the unit-check recomputation are pure functions of
their arguments, and they are the parts whose failure would silently misreport
the five readings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import shafranov_combination_discriminator as discriminator
from benchmarks.shafranov_combination_discriminator import (
    MAGNETICS_PROVENANCE,
    READING_KEYS,
    READING_LABELS,
    attribute,
    boundary_shape,
    commensurability,
    constraint_context,
    convention_clause,
    identity_combination,
    implied_radius,
    linear_combination,
    readings,
    recomputed_profile_moments,
)
from nova.biot.greens import MU0
from nova.equilibrium.constraint import ConstraintContext
from nova.equilibrium.diagnostics import shafranov_vertical_field


def test_constraint_context_places_the_current_in_its_own_field():
    """Pin the field placement of the context a constraint's observed sees.

    ``ConstraintContext``'s middle two fields are both optional
    currents-or-classes, so a positional construction exchanges them: the plasma
    current lands in ``requested_class`` and the row is handed
    ``target_current=None``, which resolves its moments on the unnormalised path
    and reports a different combination.  The driver names every field at one
    constructor, and this test is what pins the placement.
    """
    assert ConstraintContext._fields == (
        "flux",
        "requested_class",
        "target_current",
        "shadow",
    )
    options = {"requested_class": "ITER"}
    context = constraint_context("flux-sentinel", 8.0e5, **options)
    assert context.flux == "flux-sentinel"
    assert context.requested_class == "ITER"
    assert context.target_current == 8.0e5
    assert context.shadow is None


def test_the_driver_never_builds_a_context_positionally():
    """Every context in the driver names its fields, so the exchange cannot recur."""
    source = Path(discriminator.__file__).read_text(encoding="utf-8")
    calls = re.findall(r"ConstraintContext\((.*?)\)", source, flags=re.S)
    assert calls
    for call in calls:
        assert "flux=" in call
        assert "target_current=" in call


def test_identity_inversion_round_trips_the_forward_identity():
    """The inverted combination reproduces the field the identity predicts."""
    current, radius, minor = 801493.25, 0.86, 0.5847
    for combination in (-0.5, 0.0, 0.3126, 2.0):
        field = shafranov_vertical_field(current, radius, minor, combination)
        assert identity_combination(current, radius, minor, field) == pytest.approx(
            combination, rel=1.0e-12
        )


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


def test_the_combination_label_never_claims_a_formula_free_reading():
    """Every magnetics label names its ln argument or its row, and the row's one
    says so, because the production row evaluates the same identity."""
    assert "discrete" not in READING_LABELS["magnetics_discrete"]
    assert "row" in READING_LABELS["magnetics_discrete"]
    assert "ln(8R/a)" in MAGNETICS_PROVENANCE["magnetics_discrete"]
    assert "identity" in MAGNETICS_PROVENANCE["magnetics_elongated"]


def test_the_formula_free_route_is_named_and_not_claimed():
    """The receipt says which route would be formula-free, and that it is absent."""
    text = MAGNETICS_PROVENANCE["magnetics_discrete"]
    assert "contour integral" in text
    assert "not a formula-free measurement" in text


def _stored_moments(current, radius, pressure, field):
    """Return the stored pair a reconstruction would report for one radius."""
    return {
        "betap": 4.0 * pressure / (MU0 * radius * current**2),
        "li": 2.0 * field / (MU0**2 * radius * current**2),
    }


def _unit_check(current, pressure, field):
    return {
        "definition": "nova's own definition",
        "plasma_current_a": current,
        "pressure_integral_j": pressure,
        "poloidal_field_integral_t2_m3": field,
    }


def test_implied_radius_recovers_the_radius_a_stored_scalar_used():
    """A scalar built with a known denominator implies that radius back."""
    current, radius, pressure, field = 8.0e5, 0.86, 3.0e4, 1.0e-1
    stored = _stored_moments(current, radius, pressure, field)
    assert implied_radius(
        stored["betap"], pressure, 4.0, MU0, current
    ) == pytest.approx(radius, rel=0.0)
    assert implied_radius(stored["li"], field, 2.0, MU0**2, current) == pytest.approx(
        radius, rel=0.0
    )
    assert implied_radius(0.0, pressure, 4.0, MU0, current) is None
    assert implied_radius(stored["betap"], pressure, 0.0, MU0, current) is None


def test_commensurability_reads_a_shared_convention_as_commensurate():
    """The check must be able to say the two conventions agree, not only differ."""
    current, major, pressure, field = 8.0e5, 0.86, 3.0e4, 1.0e-1
    block = commensurability(
        efit=_stored_moments(current, major, pressure, field),
        unit_check=_unit_check(current, pressure, field),
        major_radius=major,
        minor_radius=0.58,
    )
    assert block["implied_over_major_from_betap"] == pytest.approx(1.0, rel=1.0e-12)
    assert block["implied_over_major_from_inductance"] == pytest.approx(
        1.0, rel=1.0e-12
    )
    assert block["implied_over_minor_from_betap"] == pytest.approx(
        major / 0.58, rel=1.0e-12
    )
    assert block["implied_radius_inductance_over_betap"] == pytest.approx(
        1.0, rel=1.0e-12
    )
    assert "volume-weighted major radius" in block["nova_radius_convention"]


def test_commensurability_detects_a_minor_radius_convention():
    """A store normalised with the minor radius implies that radius, not nova's."""
    current, major, minor, pressure, field = 8.0e5, 0.86, 0.58, 3.0e4, 1.0e-1
    efit = _stored_moments(current, minor, pressure, field)
    block = commensurability(
        efit=efit,
        unit_check=_unit_check(current, pressure, field),
        major_radius=major,
        minor_radius=minor,
    )
    assert block["implied_over_major_from_betap"] == pytest.approx(
        minor / major, rel=1.0e-12
    )
    assert block["implied_over_minor_from_betap"] == pytest.approx(1.0, rel=1.0e-12)
    nova = linear_combination(
        4.0 * pressure / (MU0 * major * current**2),
        2.0 * field / (MU0**2 * major * current**2),
    )
    assert block["rescaled_efit_combination"] == pytest.approx(nova, rel=1.0e-12)
    clause = convention_clause(block, profile_combination=nova)
    assert clause.startswith("Commensurability")
    slash = f"{nova:.4f}"
    assert slash in clause
    assert "boundary's minor radius" in clause


def test_reading_keys_are_ordered_and_labelled():
    """Every reading the panel draws carries its label in one order."""
    combination = dict.fromkeys(READING_KEYS, 0.0)
    block = readings(combination)
    assert list(block) == list(READING_KEYS)
    assert all(entry["label"] for entry in block.values())


def test_the_committed_receipt_agrees_with_its_own_stored_numbers():
    """Pin the driver's output against the arithmetic the pure functions leave free.

    The pure checks above cannot see the driver: a regression that relabelled the
    profile column, or that handed ``commensurability`` a different radius, would
    leave every one of them green while the emitted receipt misstated its own
    mechanism.  The committed receipt is read here and required to satisfy the two
    claims it makes about itself, on every row.
    """
    receipt = json.loads(
        (discriminator.DEFAULT_DIRECTORY / "receipt.json").read_text(encoding="utf-8")
    )
    rows = receipt["rows_receipt"]
    assert rows
    shifted = 0
    for row in rows:
        normalisation = row["profile_normalisation"]
        assert normalisation["path"].startswith("normalised")
        # The column the receipt reports is the one the moment path used, so the
        # label and the stored numbers cannot disagree.
        assert (
            row["profile_implied_combination"]
            == normalisation["normalised_combination"]
        )
        if normalisation["combination_shift"] != 0.0:
            shifted += 1
        # The commensurability baseline is the volume-weighted radius the
        # convention string defines, not the current centroid.
        assert (
            row["commensurability"]["nova_major_radius_m"]
            == row["unit_check"]["major_radius_m"]
        )
        assert row["commensurability"]["nova_major_radius_m"] != row["major_radius_m"]
        assert (
            row["commensurability"]["boundary_minor_radius_m"] == row["minor_radius_m"]
        )
    # The label would be vacuous if the two columns were equal everywhere.
    assert shifted == len(rows)
