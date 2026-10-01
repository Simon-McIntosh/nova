"""Confined support motion contributes to the current-moment tangent."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.clip_quadrature import clipped_support_current_moments
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.equilibrium.source import _FluxSelectedProfile
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.jax.config import configure_dtypes


class _UniformDensity:
    def __init__(self, value=1.0):
        self.value = value

    def current_density(self, radius, psi_norm):
        return jnp.full_like(psi_norm, self.value)


def _moment_map(open_density=None):
    configure_dtypes()
    assert jax.config.jax_enable_x64
    polygon = np.asarray([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]])
    mesh = AtomicCellMesh.from_cells([polygon])
    support = mesh.traced_clip(jnp.ones(len(mesh.node_coordinates)))
    profile = _FluxSelectedProfile(_UniformDensity(), open_density)

    def moments(cutoff):
        field = FluxFieldPolynomial(
            jnp.asarray([[1.5 - cutoff, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            jnp.asarray([[2.5, 0.0]]),
            jnp.ones((1, 2)),
            jnp.ones(1, dtype=bool),
        )
        return jnp.stack(
            clipped_support_current_moments(
                support, jnp.ones(1, dtype=bool), field, profile, cut_cell_capacity=1
            )
        )[:, 0]

    return jax.jit(moments)


def test_confined_moments_carry_continuous_boundary_motion():
    moments = _moment_map()
    cutoff = jnp.asarray(0.47)
    value, tangent = jax.jvp(moments, (cutoff,), (jnp.ones_like(cutoff),))
    expected = np.asarray([0.47, 0.47 * (0.47 / 2 - 0.5), 0.0])
    np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(tangent, [1.0, -0.03, 0.0], rtol=1e-10, atol=1e-12)
    differences = [
        (moments(cutoff + h) - moments(cutoff - h)) / (2 * h) for h in (1e-6, 1e-8)
    ]
    for difference in differences:
        np.testing.assert_allclose(difference, tangent, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(differences[0], differences[1], rtol=2e-6, atol=2e-8)
    adjoint = jax.grad(lambda value: moments(value)[0])(cutoff)
    np.testing.assert_allclose(adjoint, 1.0, rtol=1e-10, atol=1e-12)


def test_wholly_outside_cell_has_zero_confined_moments():
    moments = _moment_map()
    np.testing.assert_array_equal(moments(jnp.asarray(-0.5)), 0.0)
    np.testing.assert_allclose(moments(jnp.asarray(1.5)), [1.0, 0.0, 0.0], atol=1e-12)


def test_open_closure_uses_the_complement_of_moving_support():
    moments = _moment_map(_UniformDensity(2.0))
    cutoff = jnp.asarray(0.47)
    value, tangent = jax.jvp(moments, (cutoff,), (jnp.ones_like(cutoff),))
    np.testing.assert_allclose(value, [1.53, 0.12455, 0.0], atol=1e-11)
    np.testing.assert_allclose(tangent, [-1.0, 0.03, 0.0], atol=1e-10)
