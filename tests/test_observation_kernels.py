"""Deterministic observation kernels and their numerical receipts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import (
    synthesize_thomson,
    virtual_flux_loops,
    virtual_poloidal_probes,
)
from nova.frame.coilset import CoilSet
from nova.jax.config import configure_dtypes

GRADIENT_RELATIVE_TOLERANCE = 2.0e-8


def _labelled_flux_map():
    radius = np.linspace(0.8, 1.4, 25)
    height = np.linspace(-0.3, 0.3, 25)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    psi_norm = ((radius_map - 1.1) / 0.45) ** 2 + (height_map / 0.4) ** 2
    return radius, height, 2.0 * psi_norm


def _profile_arrays():
    support = np.linspace(0.0, 1.0, 21)
    temperature = 100.0 + 40.0 * support + 8.0 * support**2
    density = 2.0e19 - 3.0e18 * support + 5.0e17 * support**2
    return support, temperature, density


def _chord_coordinates():
    return np.asarray(
        [
            [[0.91, -0.08], [1.02, -0.03], [1.13, 0.02]],
            [[1.00, 0.12], [1.16, 0.09], [1.27, 0.04]],
        ]
    )


def _coilset():
    coilset = CoilSet(dcoil=-1, field_attrs=["Br", "Bz", "Psi"])
    coilset.coil.insert(
        1.65,
        0.18,
        0.08,
        0.12,
        nturn=18.0,
        name="source",
        section="rectangle",
    )
    return coilset


def test_thomson_receipt_bounds_resolved_analytic_error():
    configure_dtypes()
    radius, height, flux = _labelled_flux_map()
    support, temperature, density = _profile_arrays()
    coordinates = _chord_coordinates()

    signals = synthesize_thomson(
        radius,
        height,
        flux,
        support,
        temperature,
        density,
        coordinates,
        axis_flux=0.0,
        boundary_flux=2.0,
    )
    expected_psi = ((coordinates[..., 0] - 1.1) / 0.45) ** 2 + (
        coordinates[..., 1] / 0.4
    ) ** 2
    expected_temperature = 100.0 + 40.0 * expected_psi + 8.0 * expected_psi**2
    expected_density = 2.0e19 - 3.0e18 * expected_psi + 5.0e17 * expected_psi**2
    receipt = signals.receipt

    assert np.all(receipt.interpolation_support.supported)
    assert receipt.units == ("eV", "m^-3")
    assert receipt.cocos == 17
    assert "bilinear" in receipt.interpolation_support.method
    maximum_error = np.asarray(receipt.numerical_error_bound)
    temperature_error = np.max(
        np.abs(np.asarray(signals.electron_temperature) - expected_temperature)
    )
    density_error = np.max(
        np.abs(np.asarray(signals.electron_density) - expected_density)
    )
    assert temperature_error <= maximum_error[0]
    assert density_error <= maximum_error[1]


def test_thomson_gradient_matches_central_difference():
    configure_dtypes()
    radius, height, flux = _labelled_flux_map()
    support, temperature, density = _profile_arrays()
    coordinates = _chord_coordinates()
    flux_direction = np.linspace(-0.2, 0.3, flux.size).reshape(flux.shape)
    temperature_direction = np.linspace(-0.7, 0.4, support.size)
    density_direction = np.linspace(0.3e17, -0.2e17, support.size)

    def response(displacement):
        signals = synthesize_thomson(
            radius,
            height,
            jnp.asarray(flux) + displacement * jnp.asarray(flux_direction),
            support,
            jnp.asarray(temperature)
            + displacement * jnp.asarray(temperature_direction),
            jnp.asarray(density) + displacement * jnp.asarray(density_direction),
            coordinates,
            axis_flux=0.0,
            boundary_flux=2.0,
        )
        return jnp.sum(signals.electron_temperature) + jnp.sum(
            signals.electron_density / 1.0e17
        )

    automatic = float(jax.grad(response)(0.0))
    step = 1.0e-5
    central = float((response(step) - response(-step)) / (2.0 * step))
    relative_error = abs(automatic - central) / max(abs(central), 1.0)
    assert relative_error < GRADIENT_RELATIVE_TOLERANCE


def test_virtual_flux_loops_compose_loop_and_point_factories():
    coordinates = np.asarray([[1.2, -0.1], [1.8, 0.0], [2.1, 0.3]])
    signals = virtual_flux_loops(_coilset(), coordinates)

    assert signals.values.shape == (3,)
    assert np.all(np.isfinite(signals.values))
    assert signals.receipt.units == ("Wb",)
    assert signals.receipt.cocos == 17
    assert np.all(signals.receipt.interpolation_support.supported)
    assert np.max(np.asarray(signals.receipt.numerical_error_bound)) <= 2.0e-15


def test_virtual_poloidal_probes_compose_probe_and_point_factories():
    coordinates = np.asarray([[1.2, -0.1], [1.8, 0.0], [2.1, 0.3]])
    inverse_root_two = 1.0 / np.sqrt(2.0)
    orientation = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [inverse_root_two, inverse_root_two]]
    )
    signals = virtual_poloidal_probes(_coilset(), coordinates, orientation)

    assert signals.values.shape == (3,)
    assert np.all(np.isfinite(signals.values))
    assert signals.receipt.units == ("T",)
    assert signals.receipt.cocos == 17
    assert np.all(signals.receipt.interpolation_support.supported)
    assert np.max(np.asarray(signals.receipt.numerical_error_bound)) <= 2.0e-15
