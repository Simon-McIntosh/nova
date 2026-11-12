"""Known stationary-point displacements discriminate physical field sources."""

import numpy as np
import pytest

from scripts.prototypes.diiid_xpoint_attribution.attribute import (
    attribute_fields,
    fit_conductors,
    operator_decomposition,
    saddle,
)


@pytest.fixture
def maps():
    radius = np.linspace(1, 2, 61)
    height = np.linspace(-0.6, 0.4, 61)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    reference = (rr - 1.5) ** 2 - 2 * (zz + 0.1) ** 2
    return radius, height, rr, zz, reference


def test_known_coil_current_error_moves_saddle_in_coil_term(maps):
    radius, height, rr, zz, reference = maps
    # A locally uniform vertical field has flux proportional to R squared.
    response = rr**2
    current_error = 0.002
    coil = response * current_error
    result = attribute_fields(
        radius,
        height,
        reference,
        [1.5, -0.1],
        dict(coil=coil, profiles=np.zeros_like(coil)),
    )
    exact = np.array([1.5 / (1 + current_error) - 1.5, 0])
    actual, _ = saddle(radius, height, reference + coil, [1.5, -0.1])
    np.testing.assert_allclose(actual - [1.5, -0.1], exact, atol=1e-10)
    np.testing.assert_allclose(
        result["shifts_m"]["coil"], exact, rtol=0.003, atol=1e-10
    )
    np.testing.assert_allclose(result["shifts_m"]["profiles"], 0, atol=1e-12)
    assert result["dominant"] == "coil"


def test_profile_shape_error_moves_saddle_in_profile_term(maps):
    radius, height, rr, zz, reference = maps
    # This cubic perturbation has a nonzero GS source, unlike a vacuum field.
    profile = 0.01 * (zz + 0.5) ** 3
    result = attribute_fields(
        radius,
        height,
        reference,
        [1.5, -0.1],
        dict(coil=np.zeros_like(profile), profiles=profile),
    )
    actual, _ = saddle(radius, height, reference + profile, [1.5, -0.1])
    np.testing.assert_allclose(
        result["shifts_m"]["profiles"], actual - [1.5, -0.1], rtol=0.007, atol=1e-10
    )
    assert result["shifts_m"]["profiles"][1] > 0.001
    assert result["dominant"] == "profiles"


def test_constant_boundary_gauge_has_no_saddle_motion(maps):
    radius, height, rr, zz, reference = maps
    result = attribute_fields(
        radius, height, reference, [1.5, -0.1], dict(boundary=np.full_like(rr, 100))
    )
    np.testing.assert_allclose(result["total_shift_m"], 0, atol=1e-10)


def test_unqualified_extremum_is_refused(maps):
    radius, height, rr, zz, reference = maps
    with pytest.raises(ValueError, match="qualified saddle"):
        attribute_fields(
            radius,
            height,
            (rr - 1.5) ** 2 + (zz + 0.1) ** 2,
            [1.5, -0.1],
            dict(coil=reference),
        )


def test_conductor_fit_keeps_unexplained_spatial_residual(maps):
    radius, height, rr, zz, reference = maps
    response = np.c_[rr.ravel() ** 2, zz.ravel()]
    difference = response @ [0.03, -0.02] + 0.2 * (zz.ravel() + 0.1) ** 2 + 7
    result = fit_conductors(response, difference, np.ones(rr.size, bool))
    np.testing.assert_allclose(
        result["equivalent_current_a_turn"], [0.03, -0.02], atol=1e-12
    )
    assert np.linalg.norm(result["residual_flux_wb"]) > 0.1
    assert result["rank"] == 2


def test_held_field_is_separate_from_physical_current_error(maps):
    radius, height, rr, zz, reference = maps
    physical = reference + 0.003 * rr**2
    shadow = (abs(rr - 1.5) < 0.1) & (abs(zz + 0.1) < 0.1)
    held = np.where(shadow, 0.01 * (zz + 0.1), 0)
    parts = operator_decomposition(
        reference,
        physical + held,
        np.zeros_like(rr),
        reference,
        reference,
        physical,
        1,
        1,
        shadow,
    )
    np.testing.assert_allclose(parts["boundary_shadow"], held, atol=1e-15)
    np.testing.assert_allclose(
        sum(parts.values()), physical + held - reference, atol=1e-15
    )
    np.testing.assert_allclose(parts["plasma_response"], 0.003 * rr**2, atol=1e-15)


def test_unshadowed_operator_mismatch_cannot_be_called_boundary_input(maps):
    radius, height, rr, zz, reference = maps
    with pytest.raises(ValueError, match="does not close"):
        operator_decomposition(
            reference,
            reference + 0.01,
            np.zeros_like(rr),
            reference,
            reference,
            reference,
            1,
            1,
            np.zeros_like(rr, dtype=bool),
        )
