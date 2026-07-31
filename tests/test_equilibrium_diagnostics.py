"""Analytic contract for axisymmetric force-balance diagnostics."""

import math

import numpy as np
import pytest

from nova.equilibrium import (
    DECAY_INDEX_WINDOW,
    decay_index,
    shafranov_vertical_field,
    shafranov_vertical_field_elongated,
)


def test_circular_required_field_matches_signed_ring_balance():
    """A positive toroidal current needs an inward, negative vertical field."""
    field = shafranov_vertical_field(1.0e6, 1.7, 0.5, 1.25)
    expected = -1.0e-7 * 1.0e6 / 1.7 * (math.log(27.2) - 0.25)
    assert field == pytest.approx(expected, rel=1e-12)
    assert field == pytest.approx(-0.17960101, rel=1e-6)


def test_circular_required_field_reverses_with_current():
    positive = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    negative = shafranov_vertical_field(-6.0e5, 0.9, 0.55, 1.0)
    assert positive < 0.0
    assert negative == pytest.approx(-positive, rel=1e-12)


def test_circular_required_field_scales_with_pressure_and_inductance():
    lower = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    higher = shafranov_vertical_field(6.0e5, 0.9, 0.55, 2.0)
    expected_difference = -1.0e-7 * 6.0e5 / 0.9
    assert higher - lower == pytest.approx(expected_difference, rel=1e-12)


@pytest.mark.parametrize(
    ("major_radius", "minor_radius"),
    [(0.0, 0.5), (-0.9, 0.5), (0.9, 0.0), (0.9, -0.5), (0.9, 7.2)],
)
def test_circular_required_field_returns_nan_for_degenerate_geometry(
    major_radius,
    minor_radius,
):
    assert math.isnan(
        shafranov_vertical_field(
            6.0e5,
            major_radius,
            minor_radius,
            1.0,
        )
    )


def test_elongated_required_field_matches_area_equivalent_radius():
    field = shafranov_vertical_field_elongated(
        1.0e6,
        1.7,
        0.5,
        1.69,
        1.25,
    )
    expected = -1.0e-7 * 1.0e6 / 1.7 * (math.log(8.0 * 1.7 / (0.5 * 1.3)) - 0.25)
    assert field == pytest.approx(expected, rel=1e-12)


def test_elongated_required_field_reduces_to_circular_at_unit_elongation():
    circular = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elongated = shafranov_vertical_field_elongated(
        6.0e5,
        0.9,
        0.55,
        1.0,
        1.0,
    )
    assert elongated == pytest.approx(circular, rel=1e-14)


def test_elongation_softens_required_field_by_half_logarithm():
    circular = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elongated = shafranov_vertical_field_elongated(
        6.0e5,
        0.9,
        0.55,
        1.8,
        1.0,
    )
    assert abs(elongated) < abs(circular)
    expected_difference = -1.0e-7 * 6.0e5 / 0.9 * (0.5 * math.log(1.8))
    assert circular - elongated == pytest.approx(expected_difference, rel=1e-10)


@pytest.mark.parametrize("elongation", [0.0, -1.0, float("nan"), float("inf")])
def test_elongated_required_field_returns_nan_for_invalid_elongation(elongation):
    assert math.isnan(
        shafranov_vertical_field_elongated(
            6.0e5,
            0.9,
            0.5,
            elongation,
            1.0,
        )
    )


@pytest.mark.parametrize("exponent", [0.0, 1.0, 1.4, 3.0])
def test_decay_index_recovers_power_law_exponent(exponent):
    """A field proportional to radius^-exponent has that decay index."""
    radius = np.linspace(0.4, 1.4, 2001)
    vertical_field = -0.3 * radius**-exponent
    index = decay_index(radius, vertical_field)
    np.testing.assert_allclose(index[10:-10], exponent, atol=5e-3)


def test_decay_index_returns_nan_at_field_reversal():
    radius = np.linspace(0.4, 1.4, 101)
    vertical_field = radius - 0.9
    index = decay_index(radius, vertical_field)
    assert np.isnan(index[np.argmin(np.abs(vertical_field))])


def test_decay_index_propagates_nonfinite_field_samples():
    radius = np.linspace(0.4, 1.4, 7)
    vertical_field = np.full(radius.shape, np.nan)
    index = decay_index(radius, vertical_field)
    assert np.isnan(index).all()


def test_decay_index_preserves_gradient_errors_for_degenerate_samples():
    with pytest.raises(IndexError):
        decay_index(np.array([0.9]), np.array([0.2]))
    with pytest.raises(ValueError, match="distances must match"):
        decay_index(np.array([0.8, 1.0]), np.array([0.2]))


def test_decay_index_window_is_the_open_rigid_displacement_interval():
    assert DECAY_INDEX_WINDOW == (0.0, 1.5)
