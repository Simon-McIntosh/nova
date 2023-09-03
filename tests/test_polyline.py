import numpy as np
import pytest
import vedo

from nova.assembly.fiducialdata import FiducialData
from nova.geometry.polyline import Arc, Line, PolyArc, PolyLine


def test_2d_arc_radius():
    arc = Arc(np.array([(0, 1, 0), (1, 0, 0), (0, -1, 0)]))
    assert np.isclose(arc.radius, 1)


def test_2d_arc_center():
    arc = Arc(np.array([(0, 1, 0), (1, 0, 0), (0, -1, 0)]))
    assert np.allclose(arc.center, (0, 0, 0))


def test_3d_arc_center():
    arc = Arc(np.array([(0, 1, -3.4), (1, 0, -3.4), (0, -1, -3.4)]))
    assert np.isclose(arc.center[2], -3.4)


def test_negative_acute():
    arc = Arc(np.array([(0, 1, 0), (1, 0, 0), (-1, 0, 0)]))
    assert np.allclose(arc.center, (0, 0, 0))
    assert np.allclose(arc.sample(2)[-1], (-1, 0, 0))


def test_points_on_polyarc():
    line = PolyArc(
        np.array(
            [(0, 0, 0), (1, 0.5, 0.1), (1.5, 1.1, 0.2), (2, 2, 0.5), (5, -0.25, 5.8)]
        ),
        50,
    )
    for point in line.points:
        delta = np.linalg.norm(line.curve - point[np.newaxis, :], axis=1)
        assert np.min(delta) < 0.1


def test_arc_length():
    line = PolyArc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)]), 100)
    arc = Arc(line.curve)
    assert np.isclose(arc.length, np.pi, atol=1e-3)


def test_match_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)]))
    assert arc.test


def test_match_arc_3d():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0.5)]))
    assert arc.test


def test_match_four_point_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]))
    assert arc.test


def test_missmatch_four_point_arc():
    arc = Arc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0.01)]))
    assert not arc.test


def test_match_single_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 20)
    arc = Arc(line.curve)
    assert arc.test


def test_missmatch_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 20)
    arc = Arc(line.curve)
    assert not arc.test


def test_match_single_polyarc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.curve[:100])
    assert arc.test


def test_mismatch_single_polyarc_plus_one():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.curve[:101])
    assert not arc.test


def test_three_point_line():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    polyline = PolyLine(points, minimum_arc_nodes=4)
    assert len(polyline.segments) == 2
    assert np.sum([isinstance(segment, Line) for segment in polyline.segments]) == 2


def test_four_point_line():
    rng = np.random.default_rng(2025)
    points = rng.random((4, 3))
    polyline = PolyLine(points, minimum_arc_nodes=4)
    assert len(polyline.segments) == 3
    assert np.sum([isinstance(segment, Line) for segment in polyline.segments]) == 3


def test_line_line_arc_line_line():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    points = np.append(
        points[:-1], PolyArc(np.r_[points[-1:], rng.random((2, 3))], 5).curve, axis=0
    )
    points = np.append(points[:-1], rng.random((2, 3)), axis=0)
    polyline = PolyLine(points, arc_eps=1e-3, line_eps=1e-3, minimum_arc_nodes=4)
    assert len(polyline.segments) == 5
    assert np.sum([isinstance(segment, Arc) for segment in polyline.segments]) == 1
    assert np.sum([isinstance(segment, Line) for segment in polyline.segments]) == 4


def test_decimate_single_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 100)
    polyline = PolyLine(line.curve)
    assert len(polyline.segments) == 1
    assert isinstance(polyline.segments[0], Arc)


def test_decimate_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    polyline = PolyLine(line.curve, arc_eps=1e-4)
    assert len(polyline.segments) == 2
    assert [isinstance(polyline.segments[0], Arc) for segment in polyline.segments] == [
        True,
        True,
    ]


def test_single_arc_hd():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    polyline = PolyLine(PolyArc(points, 405).curve, arc_eps=1e-3, line_eps=5e-3)
    assert len(polyline.segments) == 1
    assert np.sum([isinstance(segment, Arc) for segment in polyline.segments]) == 1


def test_decimate_polyline():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 100)
    curve = line.curve
    for i in range(2):
        points = rng.random((3, 3))
        line = PolyArc(points, 25)
        curve = np.append(curve, rng.random((2, 3)), axis=0)
        curve = np.append(curve, line.curve, axis=0)
    polyline = PolyLine(curve, arc_eps=1e-4, line_eps=5e-4, minimum_arc_nodes=4)
    assert np.sum([isinstance(segment, Arc) for segment in polyline.segments]) == 3
    assert np.sum([isinstance(segment, Line) for segment in polyline.segments]) == 6


def test_arc_plotfit():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    points[:, 0] = 0
    arc = Arc(points)
    with arc.test_plot():
        arc.plot_fit()


def test_arc_plot():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    arc = Arc(points)
    assert np.allclose(arc.start_point, points[0])
    assert np.allclose(arc.end_point, points[2])
    with arc.test_plot():
        arc.plot()


def test_polyarc_plot():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    polyarc = PolyArc(points)
    with polyarc.test_plot():
        polyarc.plot()


def test_circle():
    theta = np.linspace(0, 2 * np.pi)
    points = 3.2 * np.c_[np.zeros_like(theta), np.cos(theta), np.sin(theta)]
    curve = PolyLine(points)
    assert len(curve.segments) == 1
    with curve.test_plot():
        curve.plot()


def test_ellipse():
    theta = np.linspace(0, 2 * np.pi, 250)
    points = np.c_[np.zeros_like(theta), 1.2 * np.cos(theta), 1.0 * np.sin(theta)]
    curve = PolyLine(points)
    assert len(curve.segments) > 1


def test_line_3_point_error():
    with pytest.raises(AssertionError):
        Line(np.zeros((3, 3)))


def test_line_points():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert np.allclose(points[0], line.start_point)
    assert np.allclose(points[-1], line.end_point)
    assert not np.allclose(line.mid_point, (0, 0, 0))
    assert np.allclose(line.center, line.mid_point)
    assert np.allclose(line.nodes, line.points)

    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert np.isclose(np.linalg.norm(line.axis), 1)
    assert np.isclose(line.axis @ (line.end_point - line.start_point), line.length)


def test_line_plot():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    with line.test_plot():
        line.plot3d()


def test_line_path():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert np.allclose(line.path, line.points)


def test_line_name():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert line.name == "line"


def test_line_geometry():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    geom = line.geometry
    assert np.allclose(line.center, [geom[attr] for attr in ["x", "y", "z"]])
    assert np.allclose(line.axis, [geom[attr] for attr in ["dx", "dy", "dz"]])
    assert np.allclose(line.start_point, [geom[attr] for attr in ["x0", "y0", "z0"]])
    assert np.allclose(line.mid_point, [geom[attr] for attr in ["x1", "y1", "z1"]])
    assert np.allclose(line.end_point, [geom[attr] for attr in ["x2", "y2", "z2"]])


def test_arc_path():
    arc = Arc(np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]]))
    assert isinstance(arc.path, np.ndarray)


def test_arc_points():
    arc = Arc(np.array([[1, 0.5, 0], [0, 0.5, 1], [0, 0.5, -1]]))
    assert arc.test
    assert np.allclose(arc.center, (0, 0.5, 0))
    assert np.allclose(arc.axis, (0, 1, 0))
    assert np.allclose(arc.start_point, [1, 0.5, 0])
    assert np.allclose(arc.end_point, [0, 0.5, -1])
    assert np.isclose(arc.length, 2 * np.pi * 3 / 4)
    assert np.allclose(arc.mid_point, arc.sample(3)[1])
    assert np.isclose(arc.central_angle, 3 / 2 * np.pi)


def test_arc_points_fail():
    arc = Arc(np.array([[1, 0.5, 0], [0, -0.5, 1], [0, 0.5, 1], [0, 0.5, -1]]))
    assert not arc.test


def test_arc_name():
    arc = Arc(np.array([[1, 0.5, 0], [0, 0.5, 1], [0, 0.5, -1]]))
    assert arc.name == "arc"


@pytest.fixture
def fiducial_polyline():
    """Return TF coil centerline."""
    fiducial = FiducialData("RE", fill=True)
    return PolyLine(
        fiducial.data.centerline.data,
        arc_eps=1e-3,
        line_eps=2e-3,
        rdp_eps=1e-4,
        minimum_arc_nodes=3,
    )


def test_fiducial_polyline_geometry_segments(fiducial_polyline):
    segment_number = len(fiducial_polyline.segments)
    assert np.all(
        [
            len(fiducial_polyline.geometry[attr]) == segment_number
            for attr in fiducial_polyline.path_attrs
        ]
    )
    assert fiducial_polyline.geometry["segment"] == [
        "arc",
        "arc",
        "arc",
        "line",
        "arc",
        "arc",
        "arc",
    ]


def test_fiducial_polyline_frame(fiducial_polyline):
    assert len(fiducial_polyline.to_frame()) == len(fiducial_polyline.segments)


if __name__ == "__main__":
    pytest.main([__file__])
