"""Closed-form support-boundary and bank Jacobian checks."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import clip_quadrature, topology


class ConstantDensity:
    def current_density(self, radius, psi_norm):
        return jnp.ones_like(radius)


def test_level_set_boundary_integral_matches_area_derivative():
    """A horizontal level moves an exact two-unit-wide clipped boundary."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    measured = clip_quadrature.level_set_current_moment_tangent(
        vertices,
        jnp.asarray(4),
        jnp.zeros(2),
        jnp.asarray(1.0),
        coefficient,
        tangent,
        selected,
        ConstantDensity(),
    )
    step = 1e-5
    plus = topology.quadratic_cell_fragments(vertices, 4, coefficient + step * tangent)
    minus = topology.quadratic_cell_fragments(vertices, 4, coefficient - step * tangent)
    difference = (jnp.sum(plus.area) - jnp.sum(minus.area)) / (2.0 * step)
    assert jax.config.jax_enable_x64 is True
    np.testing.assert_allclose(measured.cell_current, 2.0, rtol=1e-6)
    np.testing.assert_allclose(measured.cell_current, difference, rtol=1e-6)
    np.testing.assert_allclose(measured.radial_moment, 0.0, atol=1e-8)


def test_curved_level_boundary_integral_matches_area_derivative():
    """The conic tangent includes both sides of a closed curved contour."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    measured = clip_quadrature.level_set_current_moment_tangent(
        vertices,
        4,
        jnp.zeros(2),
        1.0,
        coefficient,
        tangent,
        selected,
        ConstantDensity(),
    )
    step = 1e-5
    plus = topology.quadratic_cell_fragments(vertices, 4, coefficient + step * tangent)
    minus = topology.quadratic_cell_fragments(vertices, 4, coefficient - step * tangent)
    difference = (jnp.sum(plus.area) - jnp.sum(minus.area)) / (2.0 * step)
    np.testing.assert_allclose(measured.cell_current, np.pi, rtol=1e-6)
    np.testing.assert_allclose(measured.cell_current, difference, rtol=1e-6)


def test_fixed_polygon_density_chain_rule():
    """The interior term integrates the supplied flux-density derivative."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    measured = clip_quadrature.fixed_polygon_current_moment_tangent(
        vertices,
        jnp.asarray(4),
        jnp.zeros(2),
        1.0,
        coefficient,
        tangent,
        lambda radius, flux: 2.0 * jnp.ones_like(radius),
    )
    np.testing.assert_allclose(measured.cell_current, 8.0, rtol=1e-12)
    np.testing.assert_allclose(measured.radial_moment, 0.0, atol=1e-12)
    np.testing.assert_allclose(measured.vertical_moment, 0.0, atol=1e-12)
