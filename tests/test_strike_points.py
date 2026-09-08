"""Exact separatrix intersections with the wall on the Solov'ev fixture."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from apps.playable.solovev import AXIS_RADIUS, SEED_SPAN, _solovev
    from nova.equilibrium.forward import _intersect_wall_level_curve
    from nova.jax.config import configure_dtypes


def _solovev_wall_crossings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a wall whose lower segments cross the analytic boundary twice."""
    wall = np.array(
        [
            [0.7, -0.1],
            [1.0, -0.1],
            [1.3, -0.1],
            [1.3, 0.1],
            [1.0, 0.1],
            [0.7, 0.1],
        ],
        dtype=np.float64,
    )
    axis = np.array([AXIS_RADIUS, 0.0], dtype=np.float64)
    x_point = np.array([AXIS_RADIUS, -0.2], dtype=np.float64)
    axis_flux = _solovev(axis[0], axis[1])
    psi_norm = (_solovev(wall[:, 0], wall[:, 1]) - axis_flux) / -SEED_SPAN
    return wall, psi_norm, axis, x_point


def test_solovev_strike_points_are_exact_wall_level_intersections() -> None:
    """Each strike is a jitted level-one crossing on its recorded segment."""
    configure_dtypes()
    wall, psi_norm, axis, x_point = _solovev_wall_crossings()
    points, segments, parameters = jax.jit(_intersect_wall_level_curve)(
        jnp.asarray(wall),
        jnp.asarray(psi_norm),
        jnp.asarray(axis),
        jnp.asarray(x_point),
    )
    points = np.asarray(points)
    segments = np.asarray(segments)
    parameters = np.asarray(parameters)

    assert np.array_equal(segments, np.array([0, 1], dtype=np.int32))
    for point, segment, parameter in zip(points, segments, parameters, strict=True):
        following = (int(segment) + 1) % wall.shape[0]
        expected = wall[segment] * (1.0 - parameter) + wall[following] * parameter
        psi_at_point = (
            psi_norm[segment] * (1.0 - parameter) + psi_norm[following] * parameter
        )
        assert abs(psi_at_point - 1.0) <= 1.0e-10
        np.testing.assert_allclose(point, expected, atol=1.0e-12, rtol=0.0)
        assert np.min(np.linalg.norm(wall - point, axis=1)) > 1.0e-12
