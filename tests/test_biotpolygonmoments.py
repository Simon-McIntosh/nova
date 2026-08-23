"""Flux-kernel moments over a fixed polygon section."""

import numpy as np

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux,
    polygon_analytic_flux_moments,
    polygon_analytic_greens,
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


def _field_midpoint_reference(target_r, target_z, vertices, expansion_point, levels=8):
    """Richardson-extrapolated field moments from conservative subdivision."""
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
        bz, br = greens_bz_br(
            np.asarray(target_r)[:, None],
            np.asarray(target_z)[:, None],
            point[None, :, 0],
            point[None, :, 1],
        )
        weighted = (
            weight,
            weight * (point[:, 0] - expansion_point[0]),
            weight * (point[:, 1] - expansion_point[1]),
        )
        return tuple(
            tuple(kernel @ one_weight / weight.sum() for one_weight in weighted)
            for kernel in (br, bz)
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
    return tuple(
        tuple(
            (4.0 * fine_value - coarse_value) / 3.0
            for fine_value, coarse_value in zip(fine_row, coarse_row, strict=True)
        )
        for fine_row, coarse_row in zip(fine, coarse, strict=True)
    )


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
    # The paired moment contraction and scalar uniform contraction measured a
    # 4.019e-13 maximum relative separation over these near/far targets.
    np.testing.assert_allclose(
        combined,
        polygon_analytic_flux(target_r, target_z, vertices),
        rtol=5.0e-13,
        atol=0.0,
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


def test_field_moment_blocks_match_midpoint_subdivision_near_and_far():
    vertices = np.array(
        [[2.82, -0.08], [3.10, -0.12], [3.17, 0.06], [2.93, 0.15], [2.78, 0.04]]
    )
    centre = _centroid(vertices)
    target_r = np.array([2.70, 3.24, 4.5, 7.0])
    target_z = np.array([0.02, -0.02, 0.6, -1.5])
    reference = _field_midpoint_reference(target_r, target_z, vertices, centre)
    computed = polygon_analytic_field_moments(target_r, target_z, vertices)
    for got_row, want_row in zip(computed, reference, strict=True):
        for got, want in zip(got_row, want_row, strict=True):
            np.testing.assert_allclose(got, want, rtol=5e-8, atol=2e-17)


def test_zero_field_moment_coefficients_reduce_to_uniform_kernels():
    vertices = _hexagon()
    target_r = np.array([2.7, 3.4, 5.0])
    target_z = np.array([0.03, -0.2, 0.8])
    computed = polygon_analytic_field_moments(target_r, target_z, vertices)
    uniform = polygon_analytic_greens(target_r, target_z, vertices)[1:]
    for row, expected in zip(computed, uniform, strict=True):
        # The paired moment contraction and scalar uniform contractions measured
        # a 3.252e-12 maximum relative separation across both field components.
        np.testing.assert_allclose(
            row[0] + 0.0 * row[1] + 0.0 * row[2],
            expected,
            rtol=3.5e-12,
            atol=0.0,
        )


def test_field_moment_expansion_point_translation_identity():
    vertices = _hexagon()
    centre = _centroid(vertices)
    displacement = np.array([0.037, -0.021])
    target_r = np.array([2.75, 3.35, 5.0])
    target_z = np.array([0.04, -0.18, 0.9])
    central = polygon_analytic_field_moments(
        target_r, target_z, vertices, expansion_point=centre
    )
    shifted = polygon_analytic_field_moments(
        target_r, target_z, vertices, expansion_point=centre + displacement
    )
    for base, moved in zip(central, shifted, strict=True):
        np.testing.assert_allclose(
            moved[1], base[1] - displacement[0] * base[0], rtol=0.0, atol=3e-21
        )
        np.testing.assert_allclose(
            moved[2], base[2] - displacement[1] * base[0], rtol=0.0, atol=3e-21
        )


def test_hexagon_field_moment_parity_zeros():
    vertices = _hexagon(z0=0.0)
    target_r = np.array([2.6, 3.5, 5.0])
    target_z = np.zeros_like(target_r)
    radial_field, vertical_field = polygon_analytic_field_moments(
        target_r, target_z, vertices
    )
    scale = max(np.max(np.abs(radial_field[2])), np.max(np.abs(vertical_field[0])))
    assert np.max(np.abs(radial_field[0])) <= 2e-14 * scale
    assert np.max(np.abs(radial_field[1])) <= 2e-14 * scale
    assert np.max(np.abs(vertical_field[2])) <= 2e-14 * scale
