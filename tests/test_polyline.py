
import numpy as np
import pytest

from nova.geometry.polyline import ThreePointArc, PolyLine


def test_2d_arc_radius():
    arc = ThreePointArc((0, 1, 0), (1, 0, 0), (0, -1, 0))
    assert np.isclose(arc.radius, 1)


def test_2d_arc_center():
    arc = ThreePointArc((0, 1, 0), (1, 0, 0), (0, -1, 0))
    assert np.allclose(arc.center, (0, 0, 0))


def test_3d_arc_center():
    arc = ThreePointArc((0, 1, -3.4), (1, 0, -3.4), (0, -1, -3.4))
    assert np.isclose(arc.center[2], -3.4)


def test_negative_acute():
    arc = ThreePointArc((0, 1, 0), (1, 0, 0), (-1, 0, 0))
    assert np.allclose(arc.center, (0, 0, 0))
    assert np.allclose(arc.sample(2)[-1], (-1, 0, 0))


def test_points_on_polyline():
    line = PolyLine(np.array([(0, 0, 0), (1, 0.5, 0.1), (1.5, 1.1, 0.2),
                              (2, 2, 0.5), (5, -0.25, 5.8)]), 50)
    for point in line.points:
        delta = np.linalg.norm(line.curve - point[np.newaxis, :], axis=1)
        assert np.min(delta) < 0.1

#line = PolyLine(points)

if __name__ == '__main__':

    pytest.main([__file__])
