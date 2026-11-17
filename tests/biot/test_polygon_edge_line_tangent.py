"""Field tangents stay finite and accurate beside an extended polygon edge."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nova.biot import polygonanalytic as kernel  # noqa: E402
from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402

assert jax.config.jax_enable_x64 is True


def _geometry():
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    polygon = np.array([[-0.5, -0.6], [0.5, -0.6], [0.5, 0.6], [-0.5, 0.6]])
    polygon = polygon @ rotation.T + [6, 0]
    edge = polygon[2] - polygon[1]
    normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
    point = polygon[1] + 2 * edge + 1e-12 * normal
    distance = abs(
        np.linalg.det(np.stack((edge, point - polygon[1])))
    ) / np.linalg.norm(edge)
    assert 0 < distance <= 1.001e-12
    return point, BiotMomentCoupling.from_polygons((polygon,))


def _gradient(point, coupling, xp):
    rows = kernel.packed_analytic_moments(
        xp,
        xp.asarray([point[0]]),
        xp.asarray([point[1]]),
        xp.asarray(coupling.edge),
        xp.asarray(coupling.weight),
        xp.asarray(coupling.norm),
        xp.asarray(coupling.centre.T),
        xp.asarray(coupling.centre.T),
        xp.asarray(coupling.reflection_axis),
        xp.asarray(coupling.reflection_partner),
    )
    return (
        2
        * xp.pi
        * point[0]
        * xp.stack((xp.stack(rows[6:9]), -xp.stack(rows[3:6])), axis=-1)[:, 0]
    )


@pytest.fixture(scope="module")
def derivatives():
    point, coupling = _geometry()
    actual = np.asarray(
        jax.jacfwd(lambda p: _gradient(p, coupling, jnp))(jnp.asarray(point))
    )
    estimates = []
    for step in (0.001, 0.0005, 0.00025):
        columns = []
        for direction in np.eye(2):
            offset = step * direction
            columns.append(
                (
                    _gradient(point + offset, coupling, np)
                    - _gradient(point - offset, coupling, np)
                )
                / (2 * step)
            )
        estimates.append(np.stack(columns, axis=-1))
    fourth = (4 * np.array(estimates[1:]) - np.array(estimates[:-1])) / 3
    reference = (16 * fourth[1] - fourth[0]) / 15
    return point, actual, reference


@pytest.mark.parametrize("channel", (0, 1, 2), ids=("uniform", "radial", "vertical"))
def test_extended_line_field_tangent(derivatives, channel):
    _, actual, reference = derivatives
    print(
        f"EDGE_LINE channel={channel} actual={actual[channel].tolist()} "
        f"reference={reference[channel].tolist()}"
    )
    np.testing.assert_allclose(
        actual[channel], reference[channel], rtol=1e-8, atol=1e-14
    )


def test_rotated_rectangle_boundary_integral(derivatives):
    """Both coordinate derivatives agree with the independent ring boundary integral."""
    point, actual, _ = derivatives
    # The 50-digit elliptic boundary integral in bank_hessian_diagnosis.projected
    # evaluates d_R psi_Z and d_Z psi_Z at this exact geometry and offset.
    mixed, vertical = -0.327298802735168, 0.47141068319614293
    observed = actual[0] * 240000
    np.testing.assert_allclose(observed[1], [mixed, vertical], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(observed[0, 1], mixed, rtol=1e-8, atol=1e-12)
    assert np.isfinite(point).all()
