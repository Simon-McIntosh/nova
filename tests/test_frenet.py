from itertools import product
import pytest

import numpy as np

from nova.geometry.frenet import Frenet
from nova.geometry.polyline import Arc, PolyArc


def test_binormal():
    points = np.zeros((10, 3))
    points[:, 0] = np.linspace(0, 5, len(points))
    points[-5:, 1] = np.linspace(0, 8, 5)
    frenet = Frenet(points)
    assert np.allclose(frenet.binormal, np.ones((len(points), 1)) * np.array([0, 0, 1]))
    assert np.allclose(frenet.torsion, 0)
    assert np.allclose(frenet.binormal, np.cross(frenet.tangent, frenet.normal))


def test_single_arc():
    radius = 5.3
    arc = Arc(
        np.array([(radius, 0, 0), (0, radius, 0), (-radius, 0, 0)], float),
        quadrant_segments=50,
    )
    frenet = Frenet(arc.path)
    assert np.allclose(frenet.curvature[2:-2], 1 / radius, atol=1e-3)
    assert np.allclose(frenet.torsion, 0)
    assert np.allclose(frenet.binormal, np.ones((len(frenet), 1)) * np.array([0, 0, 1]))


def test_single_line():
    points = np.zeros((10, 3))
    points[:, 0] = np.linspace(0, 3, 10)
    binormal = np.array([0, 1, 0])
    frenet = Frenet(points, binormal)
    assert np.allclose(
        frenet.binormal, np.ones((len(frenet), 1)) * binormal[np.newaxis, :]
    )
    assert np.allclose(frenet.binormal, np.cross(frenet.tangent, frenet.normal))


def test_binormal_error():
    points = np.zeros((10, 3))
    points[:, 1] = np.linspace(0, 3, 10)
    binormal = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        Frenet(points, binormal)


def test_torsion():
    polyarc = PolyArc(
        np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (-2, 0, 1), (-3, 0, 0)], float),
        resolution=10,
    )
    frenet = Frenet(polyarc.path)
    assert not np.allclose(frenet.torsion, 0)


@pytest.mark.parametrize(
    "radius,height", product([0.1, 5, 11.3], [0, 3, 7.2, -2.3, -0.1])
)
def test_helix(radius, height):
    theta = np.linspace(0, 6 * np.pi, 500)
    xcoord = radius * np.cos(theta)
    ycoord = radius * np.sin(theta)
    zcoord = height * theta
    frenet = Frenet(np.c_[xcoord, ycoord, zcoord])
    assert np.allclose(
        frenet.curvature[2:-2], radius / (radius**2 + height**2), rtol=1e-3
    )
    assert np.allclose(
        frenet.torsion[3:-3], height / (radius**2 + height**2), rtol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
