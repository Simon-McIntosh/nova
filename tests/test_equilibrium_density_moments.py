"""Exact clipped density moments used by the plasma coupling."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.equilibrium.stencil_mesh import fixed_profile_current_moments


@dataclass(frozen=True)
class AffineDensity:
    """A profile whose spatial image is affine for an affine flux field."""

    def current_density(self, radius, psi_norm):
        return 2.0 + 3.0 * radius - 4.0 * psi_norm


def _geometry():
    cells = (
        np.asarray([[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]]),
        np.asarray([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]]),
    )
    centres = np.asarray([[1.5, 0.0], [2.5, 0.0]])
    mesh = AtomicCellMesh.from_cells(cells, centroids=centres)
    scale = np.ones_like(centres)
    # psi_norm = 0.1 + 0.2 * (R - Rc) - 0.3 * (Z - Zc)
    flux_coefficient = np.asarray([[0.1, 0.2, -0.3, 0.0, 0.0, 0.0]] * len(cells))
    return mesh, centres, scale, flux_coefficient


def _moments(signed_flux):
    mesh, centres, scale, flux_coefficient = _geometry()
    support = mesh.traced_clip(signed_flux)
    return fixed_profile_current_moments(
        AffineDensity(),
        support.support_vertices,
        support.vertex_count,
        support.centroids,
        centres,
        scale,
        jnp.asarray(flux_coefficient),
    ), support


def test_direct_quadrature_carries_exact_affine_density_moments():
    """Fixed Duffy nodes reproduce affine density moments on clipped cells."""
    mesh, centres, _scale, _coefficient = _geometry()
    signed = jnp.asarray(mesh.node_coordinates[:, 0] - 1.35)
    (current, first), support = _moments(signed)

    density_at_centre = 2.0 + 3.0 * centres[:, 0] - 4.0 * 0.1
    density_gradient = np.asarray([3.0 - 4.0 * 0.2, -4.0 * -0.3])
    expected_current = density_at_centre * np.asarray(support.area) + np.einsum(
        "d,nd->n", density_gradient, np.asarray(support.first_area_moment)
    )
    expected_first = density_at_centre[:, None] * np.asarray(
        support.first_area_moment
    ) + np.einsum("nij,j->ni", np.asarray(support.second_area_moment), density_gradient)
    np.testing.assert_allclose(current, expected_current, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(first, expected_first, rtol=2.0e-14, atol=2.0e-14)


def test_direct_density_moments_zero_nonparticipating_supports():
    """Topology-excluded supports carry exact zeros through the quadrature."""
    mesh, centres, scale, flux_coefficient = _geometry()
    support = mesh.traced_clip(jnp.ones(len(mesh.node_coordinates)))
    participation = jnp.asarray([True, False])
    qualified = support.qualify(participation)
    current, first = fixed_profile_current_moments(
        AffineDensity(),
        qualified.support_vertices,
        qualified.vertex_count,
        qualified.centroids,
        centres,
        scale,
        jnp.asarray(flux_coefficient),
        participation=participation,
    )

    assert qualified.vertex_count[1] == 0
    assert qualified.area[1] == 0.0
    np.testing.assert_array_equal(qualified.first_area_moment[1], np.zeros(2))
    np.testing.assert_array_equal(qualified.second_area_moment[1], np.zeros((2, 2)))
    assert current[0] != 0.0
    assert np.any(np.asarray(first[0]) != 0.0)
    assert current[1] == 0.0
    np.testing.assert_array_equal(first[1], np.zeros(2))


def test_direct_density_moments_trace_once_and_compose_under_vmap():
    """A moving fixed-capacity clip changes values without changing its trace."""
    mesh, _centres, _scale, _coefficient = _geometry()
    trace_count = 0

    def contracted(offset):
        nonlocal trace_count
        trace_count += 1
        signed = jnp.asarray(mesh.node_coordinates[:, 0]) - offset
        (current, first), _support = _moments(signed)
        return jnp.r_[jnp.sum(current), jnp.sum(first, axis=0)]

    compiled = jax.jit(contracted)
    offsets = jnp.linspace(1.05, 1.95, 10)
    values = [compiled(offset) for offset in offsets]
    jax.block_until_ready(values[-1])
    assert trace_count == 1
    batched = jax.vmap(compiled)(offsets)
    jax.block_until_ready(batched)
    assert trace_count == 1
    np.testing.assert_allclose(batched, jnp.stack(values), rtol=3.0e-15, atol=3.0e-15)
