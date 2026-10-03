"""Derivative contracts for selected topology boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.flux_surface_connectivity import fit_tensor_spline
from nova.equilibrium.topology import TopologyClass
from tests.test_equilibrium_forward_solve import converged, machine
from tests.test_hex_flood_sn_secondary import GEOMETRIES, _topology_and_flux


STEP = 1.0e-5
BOUNDARY_TOLERANCE = 1.0e-8
CONTROL_TOLERANCE = 1.0e-10


@pytest.fixture(scope="module")
def terminal_fixture():
    """Build a structured single-null field for both admitted boundary classes."""
    topology, flux, inside, _radius, _height = _topology_and_flux(GEOMETRIES[0], 33)
    return topology, flux, inside


def _directions(shape: tuple[int, ...]) -> list[jax.Array]:
    """Return independent, unit-max perturbations with reproducible seeds."""
    return [
        jnp.asarray(values / np.max(np.abs(values)))
        for values in (
            np.random.default_rng(seed).standard_normal(shape) for seed in range(8)
        )
    ]


@pytest.mark.parametrize(
    "topology_class", (TopologyClass.DIVERTED, TopologyClass.LIMITED)
)
def test_boundary_flux_jvp_holds_the_admitted_point_fixed(
    terminal_fixture, topology_class
):
    """The published boundary value differentiates at its selected point."""
    topology, flux, inside = terminal_fixture

    def boundary_value(field):
        return topology.read_qualification(
            field, -1.0, inside, topology_class
        ).state.boundary_flux

    state = topology.read_qualification(flux, -1.0, inside, topology_class).state
    point = jax.lax.stop_gradient(state.boundary)

    def fixed_point_value(field):
        grid_flux, _wall_flux = topology.split_flux_map(field)
        values = grid_flux.reshape(
            (topology.connectivity_radius.size, topology.connectivity_height.size)
        ).T
        surface = fit_tensor_spline(
            topology.connectivity_radius, topology.connectivity_height, values
        )
        return surface(point[0], point[1])

    def smooth_control(value):
        return value**3 - 0.4 * value + jnp.exp(0.3 * value)

    assert np.isfinite(float(boundary_value(flux)))
    for direction in _directions(flux.shape):
        _value, tangent = jax.jvp(boundary_value, (flux,), (direction,))
        central = (
            fixed_point_value(flux + STEP * direction)
            - fixed_point_value(flux - STEP * direction)
        ) / (2.0 * STEP)
        error = abs(float(tangent - central)) / max(abs(float(tangent)), 1.0)
        assert error < BOUNDARY_TOLERANCE

        control_value = jnp.asarray(0.37, dtype=flux.dtype)
        control_direction = jnp.asarray(1.0, dtype=flux.dtype)
        _control, control_tangent = jax.jvp(
            smooth_control, (control_value,), (control_direction,)
        )
        control_central = (
            smooth_control(control_value + STEP * control_direction)
            - smooth_control(control_value - STEP * control_direction)
        ) / (2.0 * STEP)
        assert abs(float(control_tangent - control_central)) < CONTROL_TOLERANCE


def test_forward_terminal_boundary_flux_matches_held_point_difference():
    """The full moment-map fixture keeps the terminal boundary derivative local."""
    source = machine.__wrapped__()
    profile, _seed, _vacuum = source
    flux = jnp.asarray(converged.__wrapped__(source).flux)

    def boundary_value(field):
        _current, _integrals, _masks, topology, _amplitude = profile._integral_state(
            field
        )
        return topology.boundary_flux

    _current, _integrals, _masks, terminal, _amplitude = profile._integral_state(flux)
    point = jax.lax.stop_gradient(terminal.boundary)
    topology = profile.operator._fixed_design_topology

    def fixed_point_value(field):
        grid_flux, _wall_flux = topology.split_flux_map(field)
        values = grid_flux.reshape(
            (topology.connectivity_radius.size, topology.connectivity_height.size)
        ).T
        surface = fit_tensor_spline(
            topology.connectivity_radius, topology.connectivity_height, values
        )
        return surface(point[0], point[1])

    values = np.random.default_rng(11).standard_normal(flux.shape)
    direction = jnp.asarray(values / np.max(np.abs(values)), dtype=flux.dtype)
    _value, tangent = jax.jvp(boundary_value, (flux,), (direction,))
    central = (
        fixed_point_value(flux + STEP * direction)
        - fixed_point_value(flux - STEP * direction)
    ) / (2.0 * STEP)
    error = abs(float(tangent - central)) / max(abs(float(tangent)), 1.0)
    assert error < BOUNDARY_TOLERANCE
