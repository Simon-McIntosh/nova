"""Contracts for the bank step's shared boundary integration."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import clip_quadrature, current, support_read


def test_geometry_tangent_uses_shared_boundary_integral():
    assert (
        current.moving_polygon_current_moment_tangent
        is clip_quadrature.moving_polygon_current_moment_tangent
    )
    source = inspect.getsource(support_read._normal_geometry_booking_tangent)
    assert "clip_quadrature.moving_polygon_current_moment_tangents(" in source
    assert "_UNIT_NODE" not in source
    assert "_UNIT_WEIGHT" not in source


def test_batched_boundary_integral_matches_single_ring_rule():
    class ConstantDensity:
        def current_density(self, radius, flux):
            return jnp.ones_like(radius)

    ring = jnp.asarray(((1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)))
    vertices = jnp.stack((ring, ring + jnp.asarray((0.5, 0.0))))
    velocity = 0.1 * vertices
    count = jnp.asarray((4, 4))
    centre = jnp.asarray((1.5, 0.0))
    coefficient = jnp.zeros(6)
    profile = ConstantDensity()
    actual = clip_quadrature.moving_polygon_current_moment_tangents(
        vertices, velocity, count, centre, 1.0, coefficient, profile
    )
    np.testing.assert_allclose(actual.cell_current, (0.4, 0.4), atol=1e-13)
    np.testing.assert_allclose(actual.radial_moment, (0.3, 0.6), atol=1e-13)
    np.testing.assert_allclose(actual.vertical_moment, (0.0, 0.0), atol=1e-13)
