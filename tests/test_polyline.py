
import numpy as np
import pytest

from nova.geometry.polyline import Arc, PolyArc, PolyLine, ThreePointArc


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


def test_points_on_polyarc():
    line = PolyArc(np.array([(0, 0, 0), (1, 0.5, 0.1), (1.5, 1.1, 0.2),
                             (2, 2, 0.5), (5, -0.25, 5.8)]), 50)
    for point in line.points:
        delta = np.linalg.norm(line.curve - point[np.newaxis, :], axis=1)
        assert np.min(delta) < 0.1


def test_arc_length():
    line = PolyArc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)]), 100)
    arc = Arc(line.curve)
    assert np.isclose(arc.length, np.pi, atol=1e-3)


def test_match_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)]))
    assert arc.match


def test_match_arc_3d():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0.5)]))
    assert arc.match


def test_match_four_point_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]))
    assert arc.match


def test_missmatch_four_point_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0.01)]))
    assert not arc.match


def test_match_single_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 20)
    arc = Arc(line.curve)
    assert arc.match


def test_missmatch_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 20)
    arc = Arc(line.curve)
    assert not arc.match


def test_match_single_polyarc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.curve[:100])
    assert arc.match


def test_mismatch_single_polyarc_plus_one():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.curve[:101])
    assert not arc.match


def test_decimate_single_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 100)
    polyline = PolyLine(line.curve)
    assert len(polyline.segments) == 1
    assert len(polyline.segments[0]) == 3


def test_decimate_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    polyline = PolyLine(line.curve)
    assert len(polyline.segments) == 2
    assert [len(segment) == 3 for segment in polyline.segments] == [True, True]


def test_decimate_polyline():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 100)
    curve = line.curve
    for i in range(2):
        points = rng.random((7, 3))
        line = PolyArc(points, 100)
        curve = np.append(curve, rng.random((2, 3)), axis=0)
        curve = np.append(curve, line.curve, axis=0)
    polyline = PolyLine(curve, eps=1e-8)
    assert np.sum([len(segment) == 3 for segment in polyline.segments]) == 7
    assert np.sum([len(segment) == 2 for segment in polyline.segments]) == 6


if __name__ == '__main__':

    pytest.main([__file__])
