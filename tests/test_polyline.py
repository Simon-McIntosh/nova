from itertools import product
import numpy as np
import pytest

from nova.geometry.polyline import Arc, Line, PolyArc, PolyLine
from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric
from nova.imas.test_utilities import mark


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
        delta = np.linalg.norm(line.path - point[np.newaxis, :], axis=1)
        assert np.min(delta) < 0.1


def test_single_polyarc_path():
    line = PolyArc(np.array([(1, 0.5, 0.1), (1.5, 1.1, 0.2), (2, 2, 0.5)]), 15)
    assert len(line.path) == 15


def test_multi_polyarc_path():
    line = PolyArc(
        np.array(
            [(0, 0, 0), (1, 0.5, 0.1), (1.5, 1.1, 0.2), (2, 2, 0.5), (5, -0.25, 5.8)]
        ),
        28,
    )
    assert len(line.path) == 2 * 28 - 1


def test_arc_length():
    line = PolyArc(np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)]), 100)
    arc = Arc(line.path)
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
    arc = Arc(line.path)
    assert arc.test


def test_missmatch_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 20)
    arc = Arc(line.path)
    assert not arc.test


def test_match_single_polyarc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.path[:100])
    assert arc.test


def test_mismatch_single_polyarc_plus_one():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    arc = Arc(line.path[:101])
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
        points[:-1], PolyArc(np.r_[points[-1:], rng.random((2, 3))], 5).path, axis=0
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
    polyline = PolyLine(line.path)
    assert len(polyline.segments) == 1
    assert isinstance(polyline.segments[0], Arc)


def test_decimate_dual_arc():
    rng = np.random.default_rng(2025)
    points = rng.random((5, 3))
    line = PolyArc(points, 100)
    polyline = PolyLine(line.path, arc_eps=1e-4)
    assert len(polyline.segments) == 2
    assert [isinstance(polyline.segments[0], Arc) for segment in polyline.segments] == [
        True,
        True,
    ]


def test_single_arc_hd():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    polyline = PolyLine(PolyArc(points, 405).path, arc_eps=1e-3, line_eps=5e-3)
    assert len(polyline.segments) == 1
    assert np.sum([isinstance(segment, Arc) for segment in polyline.segments]) == 1


def test_decimate_polyline():  # [arc, line, line, line, arc, line, line, line, arc]
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    arc = PolyArc(points, 100)
    curve = arc.path
    for i in range(2):
        points = rng.random((3, 3))
        arc = PolyArc(points, 25)
        curve = np.append(curve, rng.random((2, 3)), axis=0)
        curve = np.append(curve, arc.path, axis=0)
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
    polyline = PolyLine(points)
    assert len(polyline.segments) == 1
    with polyline.test_plot():
        polyline.plot()


def test_ellipse():
    theta = np.linspace(0, 2 * np.pi, 250)
    points = np.c_[np.zeros_like(theta), 1.2 * np.cos(theta), 1.0 * np.sin(theta)]
    polyline = PolyLine(points)
    assert len(polyline.segments) > 1


def test_line_3_point_error():
    with pytest.raises(AssertionError):
        Line(np.zeros((3, 3)))


def test_line_points():
    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert np.allclose(points[0], line.start_point)
    assert np.allclose(points[-1], line.end_point)
    assert np.allclose(line.center, np.mean(points, axis=0))
    assert np.allclose(line.nodes, line.points)

    rng = np.random.default_rng(2025)
    points = rng.random((2, 3))
    line = Line(points)
    assert np.isclose(np.linalg.norm(line.normal), 1)
    assert np.isclose(np.linalg.norm(line.axis), 1)
    assert np.isclose(np.linalg.norm(line.end_point - line.start_point), line.length)


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
    assert np.allclose(line.center, [geom[attr] for attr in ["x0", "y0", "z0"]])
    assert np.allclose(line.axis, [geom[attr] for attr in ["ax", "ay", "az"]])
    assert np.allclose(line.start_point, [geom[attr] for attr in ["x1", "y1", "z1"]])
    assert np.allclose(line.end_point, [geom[attr] for attr in ["x2", "y2", "z2"]])


def test_arc_path():
    arc = Arc(np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]]))
    assert isinstance(arc.path, np.ndarray)


def test_arc_points():
    arc = Arc(np.array([[1, 0.5, 0], [0, 0.5, 1], [0, 0.5, -1]]))
    assert arc.test
    assert np.allclose(arc.center, (0, 0.5, 0))
    assert np.allclose(arc.axis, (0, -1, 0))
    assert np.allclose(arc.start_point, [1, 0.5, 0])
    assert np.allclose(arc.end_point, [0, 0.5, -1])
    assert np.isclose(arc.length, 2 * np.pi * 3 / 4)
    assert np.allclose(arc.intermediate_point, arc.sample(3)[1])
    assert np.isclose(arc.central_angle, 3 / 2 * np.pi)


@pytest.mark.parametrize(
    "start_theta, central_angle, radius",
    product(
        [0.5, 2 * np.pi / 3, -3 / 2 * np.pi, np.pi, -np.pi],
        [0.5, 2 * np.pi / 3, -3 / 2 * np.pi, np.pi, -np.pi],
        [0.3, 1.7],
    ),
)
def test_shifted_arc_length(start_theta, central_angle, radius):
    theta = np.linspace(start_theta, start_theta + central_angle, 4)
    points = np.c_[
        radius * np.cos(theta), radius * np.sin(theta), -3.3 * np.ones_like(theta)
    ]
    arc = Arc(points)
    assert np.isclose(theta[-1] - theta[0], central_angle)
    assert np.isclose(arc.central_angle, abs(central_angle))
    assert np.isclose(arc.length, radius * abs(central_angle))


def test_arc_points_fail():
    arc = Arc(np.array([[1, 0.5, 0], [0, -0.5, 1], [0, 0.5, 1], [0, 0.5, -1]]))
    assert not arc.test


def test_arc_name():
    arc = Arc(np.array([[1, 0.5, 0], [0, 0.5, 1], [0, 0.5, -1]]))
    assert arc.name == "arc"


@pytest.fixture
def cc_polyline():
    """Return TF coil centerline."""
    coil = CoilsNonAxisymmetric(111003, 1)
    number = coil.data.points_length[0].data
    points = coil.data.points[0, :number].data
    return PolyLine(
        points,
        arc_eps=1e-3,
        line_eps=5e-2,
        rdp_eps=1e-3,
        minimum_arc_nodes=4,
    )


@mark["coils_non_axisymmetric"]
def test_fiducial_polyline_geometry_segments(cc_polyline):
    segment_number = len(cc_polyline.segments)
    assert np.all(
        [
            len(cc_polyline.path_geometry[attr]) == segment_number
            for attr in cc_polyline.path_attrs
        ]
    )
    assert cc_polyline.path_geometry["segment"] == [
        "arc",
        "arc",
        "line",
        "arc",
        "arc",
        "arc",
        "line",
        "arc",
        "arc",
    ]


@mark["coils_non_axisymmetric"]
def test_fiducial_polyline_frame(cc_polyline):
    assert len(cc_polyline.to_frame()) == len(cc_polyline.segments)


def test_straight_line_normal():
    path = np.array(
        [
            [-1, 0, -1],
            [-0.5, 0, -1],
            [0, 0, -1],
            [np.cos(np.pi / 3), 0, -np.sin(np.pi / 3)],
            [np.cos(np.pi / 4), 0, -np.sin(np.pi / 4)],
            [1, 0, 0],
            [1, 0, 0.5],
            [1, 0, 1],
        ]
    )
    polyline = PolyLine(path, minimum_arc_nodes=4)
    normal = np.c_[
        polyline._to_list("nx"), polyline._to_list("ny"), polyline._to_list("nz")
    ]
    reference = np.zeros((3, 3))
    reference[:, 2] = 1
    reference[2] = [-1, 0, 0]
    assert len(polyline) == 3
    assert np.allclose(np.linalg.norm(normal, axis=1), np.ones(3))
    assert np.allclose(normal, reference)


def test_polyline_nodes():
    polyline = PolyLine(np.array([[-1, 0, -1], [-0.5, 1, -1], [0, 0, -1]]))
    assert polyline.nodes.shape == (3, 3)


def test_polyline_path():
    polyline = PolyLine(
        np.array([[-1, 0, -1], [-0.5, 0.5, -1], [-0.5, -0.5, -1]]),
        quadrant_segments=21,
        minimum_arc_nodes=3,
    )
    assert polyline.path.shape == (
        int(
            polyline.quadrant_segments
            * polyline.segments[0].central_angle
            / (np.pi / 2)
        ),
        3,
    )


def test_single_arc_positive_axis():
    arc = Arc(np.array([(0, 5, 0), (0, 0, 5), (0, -5, 0)]))
    assert np.allclose(arc.axis, (1, 0, 0))


def test_single_arc_negative_axis():
    arc = Arc(np.array([(-1, 0, 0), (0, 1, 0), (1, 0, 0)]))
    assert np.allclose(arc.axis, (0, 0, -1))


@pytest.mark.parametrize(
    "points",
    (
        [(-1, 0, 0), (0, 1, 0), (1, 0, 0)],
        [(1.2, 0, 0), (0, 1.2, 0), (-1.2, 0, 0)],
        [(-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    ),
)
def test_single_arc_unit_orthogonal_axes(points):
    axes = Arc(np.array(points)).arc_axes
    for i in range(3):
        assert np.isclose(np.linalg.norm(axes[i]), 1)
    assert np.allclose(np.cross(axes[0], axes[1]), axes[2])
    assert np.allclose(np.cross(axes[2], axes[0]), axes[1])
    assert np.allclose(np.cross(axes[1], axes[2]), axes[0])


if __name__ == "__main__":
    pytest.main([__file__])
