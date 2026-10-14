"""Closed-boundary current prescription and conductor flux linkage."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nova.circuit.conductor import ConductorSet
from nova.imas.prescribed_plasma import (
    plasma_flux_at_points,
    plasma_linked_flux,
    plasma_linked_flux_history,
)
from nova.imas.profiles import prescribe_current_density
from scripts.iter_prescribed_plasma.measure import current_vector


@pytest.fixture
def current():
    radius = np.linspace(4, 6, 41)
    height = np.linspace(-1, 1, 41)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    psi = 1 - (rr - 5) ** 2 - zz**2
    theta = np.linspace(0, 2 * np.pi, 121)
    outline = np.c_[5 + 0.8 * np.cos(theta), 0.8 * np.sin(theta)]
    return prescribe_current_density(
        radius,
        height,
        psi,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        outline,
        (5, 0),
    )


def test_closed_boundary_prescription_matches_current_and_centroid(current):
    assert current.boundary_source == "separatrix"
    assert current.endpoint_gap < 1e-12
    assert np.sum(current.cell_current) == pytest.approx(1e6)
    assert current.centroid == pytest.approx((5.006, 0), abs=0.05)
    assert np.all(current.cell_current[(current.radius - 5) ** 2 > 0.8**2] == 0)


def test_open_outline_uses_last_closed_flux_surface(current):
    open_outline = current.boundary[:-1]
    radius = current.radius
    height = current.height
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    replacement = prescribe_current_density(
        radius,
        height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        open_outline,
        (5, 0),
    )
    assert replacement.boundary_source == "last_closed_flux"
    assert replacement.boundary_fraction is not None
    assert replacement.endpoint_gap > 0.01
    assert np.sum(replacement.cell_current) == pytest.approx(1e6)


def test_divertor_legs_are_cut_at_x_point_before_stored_boundary(current):
    theta = np.linspace(-np.pi / 2, 3 * np.pi / 2, 121)
    x_point = (5.0, -0.8)
    diverted = np.vstack(
        ([4.6, -1.4], np.c_[5 + 0.8 * np.cos(theta), 0.8 * np.sin(theta)], [5.4, -1.4])
    )
    rr, zz = np.meshgrid(current.radius, current.height, indexing="ij")
    replacement = prescribe_current_density(
        current.radius,
        current.height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        diverted,
        (5, 0),
        x_points=[x_point],
        boundary_outline=current.boundary,
    )
    assert replacement.boundary_source == "separatrix_x_point"
    assert replacement.boundary_fraction is None
    assert replacement.endpoint_gap > 0.5
    assert replacement.boundary[0] == pytest.approx(x_point)


def test_stored_boundary_precedes_flux_contour(current):
    rr, zz = np.meshgrid(current.radius, current.height, indexing="ij")
    replacement = prescribe_current_density(
        current.radius,
        current.height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        current.boundary[:-1],
        (5, 0),
        boundary_outline=current.boundary,
    )
    assert replacement.boundary_source == "boundary_outline"
    assert replacement.boundary_fraction is None
    np.testing.assert_array_equal(replacement.boundary, current.boundary)


def test_missing_edge_gradient_extrapolates_from_finite_samples(current):
    radius = current.radius
    height = current.height
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    pressure_gradient = np.ones(50)
    pressure_gradient[-1] = np.nan
    replacement = prescribe_current_density(
        radius,
        height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        pressure_gradient,
        np.zeros(50),
        1,
        0.36,
        1e6,
        current.boundary,
        (5, 0),
    )
    assert np.all(np.isfinite(replacement.cell_current))
    assert replacement.scale_factor == pytest.approx(current.scale_factor)


def test_finite_section_and_filament_linkage(current):
    conductors = ConductorSet(
        r=np.array([7.0, 7.0]),
        z=np.array([0.4, -0.4]),
        dr=np.array([0.0, 0.2]),
        dz=np.array([0.0, 0.2]),
        current_share=np.ones(2),
        circuit=np.array([0, 1]),
    )
    circuits, linked = plasma_linked_flux(current, conductors)
    assert circuits.tolist() == [0, 1]
    assert linked.shape == (2,)
    assert linked[0] > 0 and linked[1] > 0
    shared = ConductorSet(
        r=np.array([7.0, 7.0]),
        z=np.array([0.4, 0.4]),
        dr=np.zeros(2),
        dz=np.zeros(2),
        current_share=np.array([1.0, 2.0]),
        circuit=np.zeros(2, dtype=int),
    )
    doubled = replace(
        current,
        current_density=2 * current.current_density,
        cell_current=2 * current.cell_current,
        plasma_current=2 * current.plasma_current,
    )
    ids, history = plasma_linked_flux_history([current, doubled], shared)
    expected = 3 * plasma_flux_at_points(current, np.array([7.0]), np.array([0.4]))[0]
    assert ids.tolist() == [0]
    assert history[:, 0] == pytest.approx([expected, 2 * expected])


def test_vertical_shift_follows_flux_gradient(current):
    radius = np.array([7.0])
    height = np.array([0.4])
    step = 0.01
    probe = 0.001
    base = plasma_flux_at_points(current, radius, height)[0]
    moved = plasma_flux_at_points(current, radius, height, vertical_offset=step)[0]
    derivative = (
        plasma_flux_at_points(current, radius, height + probe)[0]
        - plasma_flux_at_points(current, radius, height - probe)[0]
    ) / (2 * probe)
    predicted_change = -step * derivative
    assert abs(predicted_change) > 1e-4
    assert moved - base == pytest.approx(predicted_change, rel=0.03)


def test_conductor_currents_follow_the_equilibrium_time():
    active = SimpleNamespace(
        time=np.array([0.0, 1.0, 2.0]),
        coil=[
            SimpleNamespace(current=SimpleNamespace(data=np.array([0.0, 10.0, 20.0])))
        ],
    )
    passive = SimpleNamespace(
        time=np.array([0.0, 2.0]),
        loop=[SimpleNamespace(current=SimpleNamespace(data=np.array([4.0, 8.0])))],
    )
    values = current_vector(active, passive, [("coil", 0), ("passive", 0)], 1.5)
    assert values == pytest.approx([15.0, 7.0])
