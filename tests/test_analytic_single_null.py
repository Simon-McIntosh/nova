"""Exact contracts for the Cerfon--Freidberg single-null reference."""

from __future__ import annotations

import numpy as np

from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null


def test_single_null_has_the_requested_geometry_and_stationary_points():
    exact = cerfon_freidberg_single_null()

    assert exact.major_radius == 1.70
    assert exact.inverse_aspect_ratio == 0.32
    assert exact.elongation == 1.7
    assert exact.triangularity == 0.33
    assert exact.x_point[1] < exact.magnetic_axis[1]
    np.testing.assert_allclose(
        exact.gradient(exact.magnetic_axis[None, :]), 0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        exact.gradient(exact.x_point[None, :]), 0.0, atol=2.0e-14
    )
    assert np.all(
        np.linalg.eigvalsh(exact.hessian(exact.magnetic_axis[None, :])[0]) < 0.0
    )
    assert np.linalg.det(exact.hessian(exact.x_point[None, :])[0]) < 0.0


def test_single_null_separatrix_contains_the_shaping_anchors():
    exact = cerfon_freidberg_single_null()
    boundary = exact.separatrix()
    anchors = np.stack(
        (
            exact.inner_equatorial_point,
            exact.outer_equatorial_point,
            exact.upper_point,
            exact.x_point,
        )
    )

    np.testing.assert_allclose(exact.flux(anchors), 0.0, atol=2.0e-14)
    assert float(np.min(boundary[:, 0])) >= 0.85
    clearance = float(np.min(np.linalg.norm(boundary - exact.magnetic_axis, axis=1)))
    assert clearance >= 0.8 * exact.minor_radius


def test_single_null_satisfies_the_grad_shafranov_equation():
    exact = cerfon_freidberg_single_null()
    radial = np.linspace(1.08, 2.30, 29)
    vertical = np.linspace(-1.10, 1.00, 31)
    rr, zz = np.meshgrid(radial, vertical)
    points = np.column_stack((rr.ravel(), zz.ravel()))
    residual = exact.grad_shafranov_residual(points)
    source = exact.grad_shafranov_source(points)

    relative = float(np.max(np.abs(residual)) / np.max(np.abs(source)))
    assert relative < 1.0e-12
