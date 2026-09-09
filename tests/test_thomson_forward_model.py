"""Contracts for the non-inverting Thomson diagnostic prediction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nova.diagnostics import (
    THOMSON_FORWARD_INPUTS,
    ForwardEquilibrium,
    IndependentElectronProfiles,
    ThomsonChordGeometry,
    ThomsonInstrumentResponse,
    ThomsonScatteringPhysics,
    compare_thomson_prediction,
    predict_thomson,
)
from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _double_precision() -> None:
    configure_dtypes()


@pytest.fixture
def equilibrium() -> ForwardEquilibrium:
    radius = np.linspace(0.8, 1.4, 65)
    height = np.linspace(-0.4, 0.4, 65)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    psi_norm = ((radius_map - 1.1) / 0.45) ** 2 + (height_map / 0.5) ** 2
    return ForwardEquilibrium(
        radius_m=radius,
        height_m=height,
        flux_wb=2.0 * psi_norm,
        axis_flux_wb=0.0,
        boundary_flux_wb=2.0,
        identity="analytic solved equilibrium",
    )


@pytest.fixture
def geometry() -> ThomsonChordGeometry:
    positions = np.column_stack(
        (np.asarray((0.94, 1.02, 1.10, 1.18, 1.26)), np.zeros(5))
    )
    return ThomsonChordGeometry(
        name="midplane",
        channel_names=tuple(f"channel-{index}" for index in range(len(positions))),
        scattering_positions_m=positions,
        chord_direction_rz=(1.0, 0.0),
        scattering_volume_sigma_m=np.full(len(positions), 5.0e-4),
    )


@pytest.fixture
def scattering() -> ThomsonScatteringPhysics:
    return ThomsonScatteringPhysics(
        laser_wavelength_m=1064.0e-9,
        laser_energy_j=1.0,
        scattering_angle_rad=np.pi / 2.0,
    )


@pytest.fixture
def instrument() -> ThomsonInstrumentResponse:
    return ThomsonInstrumentResponse(
        spectral_band_edges_m=np.asarray(
            (
                (700.0e-9, 900.0e-9),
                (900.0e-9, 1040.0e-9),
                (1040.0e-9, 1088.0e-9),
                (1088.0e-9, 1300.0e-9),
            )
        ),
        throughput=np.asarray((0.18, 0.22, 0.25, 0.17)),
        solid_angle_sr=1.0e-4,
        scattering_length_m=5.0e-3,
        quantum_efficiency=0.7,
        background_photoelectrons=np.asarray((2.0, 2.0, 2.0, 2.0)),
    )


@pytest.fixture
def profiles() -> IndependentElectronProfiles:
    support = np.linspace(0.0, 1.2, 61)
    return IndependentElectronProfiles(
        psi_norm=support,
        electron_temperature_ev=900.0 - 650.0 * support,
        electron_density_m3=3.2e19 - 1.5e19 * support,
        provenance="declared independently of the solved equilibrium",
    )


def test_input_contract_contains_only_information_absent_from_gs() -> None:
    names = " ".join(item.name for item in THOMSON_FORWARD_INPUTS)

    assert len(THOMSON_FORWARD_INPUTS) == 5
    assert "flux" not in names
    assert "boundary" not in names
    assert "electron_temperature" in names
    assert "electron_density" in names
    assert all(item.absent_from_gs_forward for item in THOMSON_FORWARD_INPUTS)


def test_prediction_samples_the_solved_flux_and_forms_detector_counts(
    equilibrium,
    geometry,
    scattering,
    instrument,
    profiles,
) -> None:
    prediction = predict_thomson(
        equilibrium, geometry, scattering, instrument, profiles
    )
    coordinate = np.asarray(geometry.scattering_positions_m)
    expected_psi = ((coordinate[:, 0] - 1.1) / 0.45) ** 2

    assert prediction.equilibrium_identity == equilibrium.identity
    assert np.all(prediction.supported)
    assert np.asarray(prediction.spectral_photoelectrons).shape == (5, 4)
    assert np.all(np.asarray(prediction.spectral_photoelectrons) > 0.0)
    assert np.allclose(prediction.psi_norm, expected_psi, atol=1.5e-4)
    sampled_psi = np.asarray(prediction.psi_norm)
    expected_temperature = 900.0 - 650.0 * sampled_psi
    expected_density = 3.2e19 - 1.5e19 * sampled_psi
    assert np.allclose(
        prediction.electron_temperature_ev, expected_temperature, rtol=1.0e-12
    )
    assert np.allclose(prediction.electron_density_m3, expected_density, rtol=1.0e-12)


def test_scattered_signal_scales_with_density_and_broadens_with_temperature(
    equilibrium,
    geometry,
    scattering,
    instrument,
    profiles,
) -> None:
    baseline = predict_thomson(equilibrium, geometry, scattering, instrument, profiles)
    dense = predict_thomson(
        equilibrium,
        geometry,
        scattering,
        replace(instrument, background_photoelectrons=np.zeros(4)),
        replace(
            profiles,
            electron_density_m3=2.0 * np.asarray(profiles.electron_density_m3),
        ),
    )
    no_background = predict_thomson(
        equilibrium,
        geometry,
        scattering,
        replace(instrument, background_photoelectrons=np.zeros(4)),
        profiles,
    )
    hot = predict_thomson(
        equilibrium,
        geometry,
        scattering,
        instrument,
        replace(
            profiles,
            electron_temperature_ev=4.0 * np.asarray(profiles.electron_temperature_ev),
        ),
    )

    assert np.allclose(
        dense.spectral_photoelectrons,
        2.0 * np.asarray(no_background.spectral_photoelectrons),
    )
    assert np.all(
        np.asarray(hot.spectral_width_m) > np.asarray(baseline.spectral_width_m)
    )
    assert not np.allclose(
        np.asarray(hot.spectral_photoelectrons),
        np.asarray(baseline.spectral_photoelectrons),
    )


def test_comparison_reports_disagreement_without_mutating_prediction(
    equilibrium,
    geometry,
    scattering,
    instrument,
    profiles,
) -> None:
    prediction = predict_thomson(
        equilibrium, geometry, scattering, instrument, profiles
    )
    temperature_before = np.asarray(prediction.electron_temperature_ev).copy()
    density_before = np.asarray(prediction.electron_density_m3).copy()
    measured_temperature = temperature_before + np.asarray(
        (10.0, -20.0, 30.0, -40.0, 50.0)
    )
    measured_density = density_before + np.asarray((1.0, -2.0, 3.0, -4.0, 5.0)) * 1.0e18

    comparison = compare_thomson_prediction(
        prediction, measured_temperature, measured_density
    )

    assert comparison.compared_channels == 5
    assert comparison.temperature_rmse_ev == pytest.approx(np.sqrt(1100.0))
    assert comparison.temperature_median_absolute_error_ev == 30.0
    assert comparison.density_rmse_m3 == pytest.approx(np.sqrt(11.0) * 1.0e18)
    assert comparison.density_median_absolute_error_m3 == pytest.approx(3.0e18)
    assert np.array_equal(prediction.electron_temperature_ev, temperature_before)
    assert np.array_equal(prediction.electron_density_m3, density_before)


def test_prediction_refuses_non_nova_flux_convention(equilibrium) -> None:
    with pytest.raises(ValueError, match="COCOS 17"):
        replace(equilibrium, cocos=3)
