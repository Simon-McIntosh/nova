"""Closed-boundary current prescription and conductor flux linkage."""

import numpy as np
import pytest

from nova.circuit.conductor import ConductorSet
from nova.imas.prescribed_plasma import (
    plasma_flux_at_points,
    plasma_linked_flux,
    plasma_linked_flux_history,
)
from nova.imas.profiles import prescribe_current_density


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
    assert replacement.endpoint_gap > 0.01
    assert np.sum(replacement.cell_current) == pytest.approx(1e6)


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
    _, history = plasma_linked_flux_history([current, current], conductors)
    np.testing.assert_array_equal(history[0], history[1])


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
