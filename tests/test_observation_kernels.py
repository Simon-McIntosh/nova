"""Deterministic observation kernels and their numerical receipts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from nova.equilibrium import (
    synthesize_thomson,
    virtual_flux_loops,
    virtual_poloidal_probes,
)
from nova.frame.coilset import CoilSet
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.observation import (
    ConstraintPinSet,
    ConstraintViolationError,
    IsofluxPin,
    MomentIntegralSupport,
    MomentPin,
    PinUncertainty,
)
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


def _public_observation_profile():
    radius, height, flux = _labelled_flux_map()
    lattice = FluxLattice(radius, height)
    profile = object.__new__(ForwardProfile)
    profile.lattice = lattice
    coordinate = lattice.coordinate
    core = np.ones(lattice.node_count, dtype=bool)
    core[::3] = False
    masks = SimpleNamespace(core=jnp.asarray(core))
    topology = SimpleNamespace(
        axis_flux=jnp.asarray(0.0), boundary_flux=jnp.asarray(2.0)
    )
    profile.operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=coordinate),
        read=lambda _state: (masks, topology),
        source=SimpleNamespace(closure_degrees=0),
    )

    def integral_state(state, requested_class=None, target_current=None):
        del requested_class, target_current
        current = 2.0 + 0.05 * jnp.asarray(state)[: lattice.node_count]
        return SimpleNamespace(cell_current=current), None, masks, topology, None

    profile._integral_state = integral_state
    return profile, np.asarray(flux).reshape(-1)


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


def test_public_thomson_map_jacobian_matches_central_difference():
    configure_dtypes()
    profile, flux = _public_observation_profile()
    support, temperature, density = _profile_arrays()
    coordinates = _chord_coordinates()
    direction = np.linspace(-0.2, 0.3, flux.size)

    automatic = (
        np.asarray(
            profile.thomson_observation_jacobian(
                flux, support, temperature, density, coordinates
            )
        )
        @ direction
    )
    step = 1.0e-5
    upper = profile.thomson_observation_map(
        flux + step * direction, support, temperature, density, coordinates
    )
    lower = profile.thomson_observation_map(
        flux - step * direction, support, temperature, density, coordinates
    )
    central = np.asarray((upper - lower) / (2.0 * step))
    relative_error = np.linalg.norm(automatic - central) / max(
        np.linalg.norm(central), 1.0
    )

    assert relative_error < GRADIENT_RELATIVE_TOLERANCE


def test_current_moment_map_declares_support_and_matches_central_difference():
    configure_dtypes()
    profile, flux = _public_observation_profile()
    direction = np.linspace(-0.3, 0.4, flux.size)
    support = MomentIntegralSupport.ALL_DOMAIN

    observation = profile.current_moment_observation(flux, support=support)
    automatic = (
        np.asarray(profile.current_moment_jacobian(flux, support=support)) @ direction
    )
    step = 1.0e-5
    upper = profile.current_moment_map(flux + step * direction, support=support)
    lower = profile.current_moment_map(flux - step * direction, support=support)
    central = np.asarray((upper - lower) / (2.0 * step))
    relative_error = np.linalg.norm(automatic - central) / max(
        np.linalg.norm(central), 1.0
    )

    assert observation.support is support
    assert relative_error < GRADIENT_RELATIVE_TOLERANCE
    confined = profile.current_moment_observation(
        flux, support=MomentIntegralSupport.CONFINED_CORE
    )
    assert confined.support is MomentIntegralSupport.CONFINED_CORE
    assert confined.plasma_current < observation.plasma_current


def test_typed_pins_qualify_the_public_solve_without_statistical_state():
    configure_dtypes()
    profile, flux = _public_observation_profile()
    coordinate = _chord_coordinates().reshape(-1, 2)
    sampled = profile.thomson_observation(
        flux,
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        coordinate[:2],
    )
    moment = profile.current_moment_observation(
        flux, support=MomentIntegralSupport.ALL_DOMAIN
    )
    pins = ConstraintPinSet(
        isoflux=(
            IsofluxPin(
                tuple(coordinate[0]),
                tuple(coordinate[1]),
                float(np.mean(np.asarray(sampled.psi_norm))),
                PinUncertainty(0.5, "1", "trusted chord-pair interval"),
            ),
        ),
        moments=(
            MomentPin(
                "plasma_current",
                float(moment.plasma_current),
                PinUncertainty(1.0, "A", "declared current interval"),
                MomentIntegralSupport.ALL_DOMAIN,
            ),
        ),
    )
    equilibrium = SimpleNamespace(flux=jnp.asarray(flux))
    profile._solve_accelerated = lambda *args, **options: equilibrium

    solved = profile.solve(flux, pins=pins)

    assert solved is equilibrium
    assert bool(profile.constraints_satisfied(flux, pins))
    assert profile.constraint_jacobian(flux, pins).shape == (3, flux.size)


def test_public_solve_refuses_a_root_outside_a_pin_interval():
    configure_dtypes()
    profile, flux = _public_observation_profile()
    moment = profile.current_moment_observation(
        flux, support=MomentIntegralSupport.CONFINED_CORE
    )
    pins = ConstraintPinSet(
        moments=(
            MomentPin(
                "centroid_z",
                float(moment.centroid_z) + 0.1,
                PinUncertainty(1.0e-3, "m", "trusted centroid interval"),
                MomentIntegralSupport.CONFINED_CORE,
            ),
        )
    )
    profile._solve_accelerated = lambda *args, **options: SimpleNamespace(
        flux=jnp.asarray(flux)
    )

    with pytest.raises(ConstraintViolationError, match="trusted constraint"):
        profile.solve(flux, pins=pins)


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
