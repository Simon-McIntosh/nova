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


def test_diverted_boundary_tangent_is_finite_without_wall_contact(terminal_fixture):
    """A stationary saddle value needs no wall-position derivative."""
    topology, flux, inside = terminal_fixture
    state = topology.read_qualification(
        flux, -1.0, inside, TopologyClass.DIVERTED
    ).state
    assert bool(state.boundary_is_xpoint)

    def boundary_value(field):
        return topology.read_qualification(
            field, -1.0, inside, TopologyClass.DIVERTED
        ).state.boundary_flux

    value, tangent = jax.jvp(boundary_value, (flux,), (_directions(flux.shape)[0],))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(tangent))


def _selected_boundary_value(topology, flux, state, boundary_is_xpoint):
    """Follow the selected wall bracket, or hold a stationary saddle."""
    point = jax.lax.stop_gradient(state.boundary)

    def surface_for(field):
        grid_flux, _wall_flux = topology.split_flux_map(field)
        values = grid_flux.reshape(
            (topology.connectivity_radius.size, topology.connectivity_height.size)
        ).T
        return fit_tensor_spline(
            topology.connectivity_radius, topology.connectivity_height, values
        )

    def contacts(surface):
        wall_flux = surface(
            topology.wall.coordinate[:, 0], topology.wall.coordinate[:, 1]
        )
        return topology._wall_anchor_candidates(wall_flux)[0][:, :2]

    contact_index = None
    if not bool(boundary_is_xpoint):
        positions = contacts(surface_for(flux))
        contact_index = int(jnp.argmin(jnp.sum((positions - point) ** 2, axis=1)))
        np.testing.assert_allclose(positions[contact_index], point, rtol=0, atol=1e-12)

    def value(field):
        surface = surface_for(field)
        position = point if contact_index is None else contacts(surface)[contact_index]
        return surface(position[0], position[1])

    np.testing.assert_allclose(value(flux), state.boundary_flux, rtol=0, atol=1e-12)
    return value


@pytest.mark.parametrize(
    "topology_class", (TopologyClass.DIVERTED, TopologyClass.LIMITED)
)
def test_boundary_flux_jvp_matches_the_admitted_boundary_motion(
    terminal_fixture, topology_class
):
    """Wall contacts move with the field; stationary saddles need no position term."""
    topology, flux, inside = terminal_fixture

    def boundary_value(field):
        return topology.read_qualification(
            field, -1.0, inside, topology_class
        ).state.boundary_flux

    state = topology.read_qualification(flux, -1.0, inside, topology_class).state
    difference_value = _selected_boundary_value(
        topology, flux, state, state.boundary_is_xpoint
    )

    def smooth_control(value):
        return value**3 - 0.4 * value + jnp.exp(0.3 * value)

    assert np.isfinite(float(boundary_value(flux)))
    for direction in _directions(flux.shape):
        _value, tangent = jax.jvp(boundary_value, (flux,), (direction,))
        central = (
            difference_value(flux + STEP * direction)
            - difference_value(flux - STEP * direction)
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


def test_forward_terminal_boundary_flux_matches_boundary_motion_difference():
    """Check the full forward chain inside wall segments 0→1 and 60→0."""
    source = machine.__wrapped__()
    profile, _seed, _vacuum = source
    flux = jnp.asarray(converged.__wrapped__(source).flux)
    wall = np.asarray(profile.operator._fixed_design_topology.wall.coordinate)

    def boundary_read(field):
        _current, _integrals, _masks, topology, _amplitude = profile._integral_state(
            field
        )
        return jnp.r_[topology.boundary, topology.boundary_flux]

    def assert_inside_segment(point, segment):
        start, end = wall[list(segment)]
        edge = end - start
        fraction = np.dot(np.asarray(point) - start, edge) / np.dot(edge, edge)
        assert 0.0 < fraction < 1.0
        np.testing.assert_allclose(point, start + fraction * edge, rtol=0, atol=1e-12)

    # The terminal contact is at vertex 0, where adjacent segment slopes differ.
    # Keep each centre and both perturbed contacts inside the same open segment.
    np.testing.assert_allclose(boundary_read(flux)[:2], wall[0], rtol=0, atol=1e-10)
    values = np.random.default_rng(11).standard_normal(flux.shape)
    direction = jnp.asarray(values / np.max(np.abs(values)), dtype=flux.dtype)
    for offset, segment in ((-1e-4, (0, 1)), (1e-4, (60, 0))):
        centre = flux + offset * direction
        value, tangent = jax.jvp(boundary_read, (centre,), (direction,))
        assert_inside_segment(value[:2], segment)
        for step in (1e-6, 5e-7):
            plus = boundary_read(centre + step * direction)
            minus = boundary_read(centre - step * direction)
            assert_inside_segment(plus[:2], segment)
            assert_inside_segment(minus[:2], segment)
            central = (plus[2] - minus[2]) / (2.0 * step)
            error = abs(float(tangent[2] - central)) / max(abs(float(tangent[2])), 1.0)
            assert error < BOUNDARY_TOLERANCE
