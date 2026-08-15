"""Flux-kernel moments over a fixed polygon section."""

import numpy as np

from nova.biot.greens import greens_psi
from nova.biot.polygonanalytic import (
    polygon_analytic_flux,
    polygon_analytic_flux_moments,
)


def _hexagon(r0=3.0, z0=0.0, radius=0.12):
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack((r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)))


def _centroid(vertices):
    local = vertices - vertices[0]
    following = np.roll(local, -1, axis=0)
    cross = local[:, 0] * following[:, 1] - following[:, 0] * local[:, 1]
    return vertices[0] + np.sum((local + following) * cross[:, None], axis=0) / (
        3.0 * cross.sum()
    )


def _midpoint_reference(target_r, target_z, vertices, expansion_point, levels=8):
    """Converged conservative midpoint subdivision, independent of the blocks."""
    centre = _centroid(vertices)
    following = np.roll(vertices, -1, axis=0)
    triangles = np.stack(
        [np.broadcast_to(centre, vertices.shape), vertices, following], axis=1
    )

    def evaluate(one_triangles):
        point = one_triangles.mean(axis=1)
        first = one_triangles[:, 1] - one_triangles[:, 0]
        second = one_triangles[:, 2] - one_triangles[:, 0]
        weight = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        kernel = greens_psi(
            np.asarray(target_r)[:, None],
            np.asarray(target_z)[:, None],
            point[None, :, 0],
            point[None, :, 1],
        )
        return (
            np.stack(
                [
                    kernel @ weight,
                    kernel @ (weight * (point[:, 0] - expansion_point[0])),
                    kernel @ (weight * (point[:, 1] - expansion_point[1])),
                ]
            )
            / weight.sum()
        )

    coarse = None
    for level in range(levels):
        a, b, c = np.moveaxis(triangles, 1, 0)
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        triangles = np.concatenate(
            [
                np.stack((a, ab, ca), axis=1),
                np.stack((ab, b, bc), axis=1),
                np.stack((ca, bc, c), axis=1),
                np.stack((ab, bc, ca), axis=1),
            ]
        )
        if level == levels - 2:
            coarse = evaluate(triangles)
    fine = evaluate(triangles)
    extrapolated = (4.0 * fine - coarse) / 3.0
    return tuple(extrapolated)


def test_flux_moment_blocks_match_midpoint_subdivision_near_and_far():
    vertices = np.array(
        [[2.82, -0.08], [3.10, -0.12], [3.17, 0.06], [2.93, 0.15], [2.78, 0.04]]
    )
    centre = _centroid(vertices)
    target_r = np.array([2.70, 3.24, 4.5, 7.0])
    target_z = np.array([0.02, -0.02, 0.6, -1.5])
    reference = _midpoint_reference(target_r, target_z, vertices, centre)
    computed = polygon_analytic_flux_moments(target_r, target_z, vertices)
    for got, want in zip(computed, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=3e-8, atol=2e-17)


def test_zero_moment_coefficients_reduce_to_uniform_flux_kernel():
    vertices = _hexagon()
    target_r = np.array([2.7, 3.4, 5.0])
    target_z = np.array([0.03, -0.2, 0.8])
    uniform, radial, vertical = polygon_analytic_flux_moments(
        target_r, target_z, vertices
    )
    combined = uniform + 0.0 * radial + 0.0 * vertical
    np.testing.assert_array_equal(
        combined, polygon_analytic_flux(target_r, target_z, vertices)
    )


def test_flux_moment_expansion_point_translation_identity():
    vertices = _hexagon()
    centre = _centroid(vertices)
    displacement = np.array([0.037, -0.021])
    target_r = np.array([2.75, 3.35, 5.0])
    target_z = np.array([0.04, -0.18, 0.9])
    uniform, radial, vertical = polygon_analytic_flux_moments(
        target_r, target_z, vertices, expansion_point=centre
    )
    shifted = polygon_analytic_flux_moments(
        target_r, target_z, vertices, expansion_point=centre + displacement
    )
    np.testing.assert_allclose(
        shifted[1], radial - displacement[0] * uniform, rtol=0.0, atol=3e-21
    )
    np.testing.assert_allclose(
        shifted[2], vertical - displacement[1] * uniform, rtol=0.0, atol=3e-21
    )


def test_hexagon_vertical_flux_moment_vanishes_by_symmetry():
    vertices = _hexagon(z0=0.0)
    target_r = np.array([2.6, 3.5, 5.0])
    target_z = np.zeros_like(target_r)
    uniform, _, vertical = polygon_analytic_flux_moments(target_r, target_z, vertices)
    assert np.max(np.abs(vertical)) <= 2e-14 * np.max(np.abs(uniform))
