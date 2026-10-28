"""Closed-form support-boundary and bank Jacobian checks."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import clip_quadrature, current, topology


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


def test_sampled_conic_vertex_velocity_matches_central_difference():
    """The polygonal boundary velocity follows its moving conic roots."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    velocity = current._conic_ring_tangents(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, tangent, selected, 64
    )
    step = 1e-6
    plus = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient + step * tangent, selected, 64
    )[0]
    minus = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient - step * tangent, selected, 64
    )[0]
    # At a vertical tangency the two sampled sides share a zero-length edge;
    # its endpoint velocity is immaterial to the polygon integral.
    interior = np.delete(np.arange(velocity.shape[1]), [0, 64, 65, 129])
    np.testing.assert_allclose(
        velocity[:, interior], ((plus - minus) / (2 * step))[:, interior], atol=1e-6
    )


def test_sampled_conic_shape_integral_matches_discrete_area_derivative():
    """The moving ring integral follows the same chords as the booking."""
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray([0.49, 0.0, 0.0, -1.0, 0.0, -1.0])
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected = jnp.asarray([True, True])
    rings, counts, _ = current._conic_rings(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, selected, 64
    )
    velocity = current._conic_ring_tangents(
        vertices, 4, jnp.zeros(2), 1.0, coefficient, tangent, selected, 64
    )
    boundary = jax.vmap(
        lambda ring, speed, count: (
            clip_quadrature.moving_polygon_current_moment_tangent(
                ring, speed, count, jnp.zeros(2), 1.0, coefficient, ConstantDensity()
            )
        )
    )(rings, velocity, counts)

    def area(shift):
        moving, number, _ = current._conic_rings(
            vertices, 4, jnp.zeros(2), 1.0, coefficient + shift * tangent, selected, 64
        )
        points, weights = clip_quadrature._quadrature_from_arrays(
            moving, number, jnp.zeros((len(moving), 2)), number >= 3, signed=True
        )
        del points
        return jnp.sum(weights)

    step = 1e-4
    difference = (area(step) - area(-step)) / (2 * step)
    np.testing.assert_allclose(jnp.sum(boundary.cell_current), difference, rtol=1e-6)
