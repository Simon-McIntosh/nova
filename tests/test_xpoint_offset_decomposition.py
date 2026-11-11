"""Independent fields distinguish whole-map motion from a saddle perturbation."""

import numpy as np
import pytest

from scripts.prototypes.diiid_xpoint_offset.decompose import (
    boundary_curve,
    classify_offsets,
    current_and_force,
    fit_translation,
    spline,
    stationary,
)


def diverted_field(radius, height):
    """An axis at (2,0) and a lower saddle at (2,-1), in metres."""
    return -((radius - 2) ** 2 + height**2 / 2 + height**3 / 3)


@pytest.fixture
def fields():
    radius, height = np.linspace(0.8, 3.2, 129), np.linspace(-1.5, 1.3, 145)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    reference = diverted_field(rr, zz)
    mask = (reference > -1 / 6) & (zz > -1)
    return radius, height, rr, zz, reference, mask


def test_known_translation_is_recovered_as_rigid(fields):
    radius, height, rr, zz, reference, mask = fields
    displacement = np.array([0.075, -0.06])
    solved = diverted_field(rr - displacement[0], zz - displacement[1]) + 0.43
    fit = fit_translation(radius, height, solved, reference, mask)
    np.testing.assert_allclose(
        fit["translation_m"],
        -displacement,
        atol=max(np.diff(radius).max(), np.diff(height).max()),
    )
    assert fit["rms_after_wb"] < 0.01 * fit["rms_before_wb"]
    assert classify_offsets(displacement, displacement, fit) == "rigid"


def test_saddle_only_perturbation_is_local(fields):
    radius, height, rr, zz, reference, mask = fields
    perturbation = 0.035 * (zz + 1) * np.exp(-((rr - 2) ** 2 + (zz + 1) ** 2) / 0.16**2)
    solved = reference + perturbation
    field = spline(radius, height, solved)
    axis, _ = stationary(field, [2, 0])
    saddle, _ = stationary(field, [2, -1], saddle=True)
    fit = fit_translation(radius, height, solved, reference, mask)
    assert np.linalg.norm(axis - [2, 0]) < 0.002
    assert np.linalg.norm(saddle - [2, -1]) > 0.015
    assert fit["rms_after_wb"] < fit["rms_before_wb"]
    assert classify_offsets(axis - [2, 0], saddle - [2, -1], fit) == "local"


def test_gauge_does_not_move_the_fit(fields):
    radius, height, rr, zz, reference, mask = fields
    solved = diverted_field(rr + 0.05, zz - 0.04)
    first = fit_translation(radius, height, solved, reference, mask)
    second = fit_translation(radius, height, solved + 30, reference - 12, mask)
    np.testing.assert_allclose(
        first["translation_m"], second["translation_m"], atol=1e-8
    )


def test_fit_refuses_empty_plasma_instead_of_reporting_zero(fields):
    radius, height, _, _, reference, mask = fields
    with pytest.raises(ValueError, match="twelve plasma nodes"):
        fit_translation(radius, height, reference, reference, np.zeros_like(mask))


def test_ampere_current_and_own_field_force_have_known_sign_and_magnitude():
    import jax
    from scipy.constants import mu_0
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    radius, height = np.linspace(1, 3, 25), np.linspace(-1, 1, 25)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    density, gradient = 2e5, 0.2
    flux = -(2 * np.pi * mu_0 * density / 3) * rr**3 + gradient * zz
    mask = (rr > 1.3) & (rr < 2.7) & (np.abs(zz) < 0.7)
    observed = current_and_force(radius, height, flux, mask)
    area = (radius[1] - radius[0]) * (height[1] - height[0])
    expected_current = float(mask.sum() * area * density)
    # The centred first derivative of R cubed has a known spacing correction.
    expected_density = density * (1 - (radius[1] - radius[0]) ** 2 / (3 * rr**2))
    expected_total = float(np.sum(expected_density[mask]) * area)
    assert expected_total > 0.99 * expected_current
    assert observed["total_current_a"] == pytest.approx(expected_total, rel=1e-12)
    assert observed["force_rz_n"][1] == pytest.approx(
        gradient * expected_total, rel=1e-12
    )
    derivative = -(2 * np.pi * mu_0 * density / 3) * (
        3 * rr**2 + (radius[1] - radius[0]) ** 2
    )
    radial_force = float(np.sum(expected_density[mask] * derivative[mask]) * area)
    assert observed["force_rz_n"][0] == pytest.approx(radial_force, rel=1e-12)
    assert observed["centroid_rz_m"][1] == pytest.approx(0, abs=1e-14)


def test_boundary_reader_closes_just_inside_the_saddle(fields):
    radius, height, _, _, reference, _ = fields
    curve = boundary_curve(radius, height, reference, np.array([2.0, 0.0]), -1 / 6)
    assert len(curve) > 100
    np.testing.assert_allclose(curve[0], curve[-1], atol=1e-10)
    assert curve[:, 1].min() > -1
    assert curve[:, 1].min() < -0.95
