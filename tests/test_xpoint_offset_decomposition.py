"""Independent fields distinguish whole-map motion from a saddle perturbation."""

import numpy as np
import pytest

from scripts.prototypes.diiid_xpoint_offset.decompose import (
    boundary_curve,
    classify_offsets,
    current_and_force,
    fit_translation,
    analyze,
    HERE,
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
    assert classify_offsets(displacement, fit) == "rigid"


def test_saddle_only_perturbation_is_non_rigid(fields):
    radius, height, rr, zz, reference, mask = fields
    perturbation = 0.035 * (zz + 1) * np.exp(-((rr - 2) ** 2 + (zz + 1) ** 2) / 0.16**2)
    solved = reference + perturbation
    from scripts.diiid_map_fidelity.measure import polish

    axis_read = polish(solved, radius, height, [2, 0], False)
    saddle_read = polish(solved, radius, height, [2, -1], True)
    assert axis_read["valid"] and saddle_read["valid"]
    axis, saddle = np.asarray(axis_read["rz_m"]), np.asarray(saddle_read["rz_m"])
    fit = fit_translation(radius, height, solved, reference, mask)
    assert np.linalg.norm(axis - [2, 0]) < 0.002
    assert np.linalg.norm(saddle - [2, -1]) > 0.015
    assert fit["rms_after_wb"] < fit["rms_before_wb"]
    assert classify_offsets(saddle - [2, -1], fit) == "non-rigid"


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


@pytest.mark.parametrize(
    ("case", "maxima", "minima"),
    [
        ("frame-137-diverted", 4, 0),
        ("frame-146-diverted", 3, 1),
    ],
)
def test_committed_saved_bank_has_no_unique_magnetic_axis(case, maxima, minima):
    import json
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    entries = json.loads((HERE / "inputs.json").read_text())
    entry = next(e for e in entries if e["case"] == case and e["arm"] == "head")
    measured = analyze(entry)
    assert measured["solved_core"]["maxima_count"] == maxima
    assert measured["solved_core"]["minima_count"] == minima
    assert measured["reference_core"]["maxima_count"] == 1
    assert measured["reference_core"]["minima_count"] == 0
    expected = {
        "frame-137-diverted": (0.0065714485, 0.2531876274, 1.2269636052),
        "frame-146-diverted": (0.0248046570, 0.3339350728, 1.3026497079),
    }[case]
    assert measured["core_flatness_ratio"] == pytest.approx(expected[0], abs=1e-7)
    assert measured["solved_core_shape"][
        "central_to_shoulder_density"
    ] == pytest.approx(expected[1], abs=1e-7)
    assert measured["reference_core_shape"][
        "central_to_shoulder_density"
    ] == pytest.approx(expected[2], abs=1e-7)
    assert measured["axis_offset_cm"] is None
    assert measured["rigid_shift_residual_cm"] is None
    assert measured["axis_rz_m"] is None
    assert measured["verdict"] == "non-rigid"
    if case == "frame-137-diverted":
        assert measured["solved_core"]["maxima_flux_spread_mwb"] == pytest.approx(
            1.684887, abs=1e-4
        )
        np.testing.assert_allclose(
            measured["xpoint_offset_cm"], [-0.821247, 7.249452], atol=1e-4
        )
    else:
        minimum = measured["solved_core"]["in_plasma_minima"][0]
        assert min(minimum["hessian_eigenvalues_wb_per_m2"]) > 0
        np.testing.assert_allclose(
            measured["xpoint_offset_cm"], [-2.781548, 5.973147], atol=1e-4
        )
