import pytest

import numpy as np

from nova.biot.biotframe import Source
from nova.geometry.polyline import PolyArc, PolyLine


@pytest.fixture
def polyline():
    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    points = np.append(
        points[:-1], PolyArc(np.r_[points[-1:], rng.random((2, 3))], 5).path, axis=0
    )
    points = np.append(points[:-1], rng.random((2, 3)), axis=0)
    return PolyLine(points, arc_eps=1e-3, line_eps=1e-3, minimum_arc_nodes=4)


def test_segments_not_implemented(polyline):
    path_geometry = polyline.path_geometry
    path_geometry["segment"][-1:] = ["circle"]
    with pytest.raises(NotImplementedError):
        Source(path_geometry).space.start_axes


def test_line_and_arc(polyline):
    source = Source(polyline.path_geometry)
    assert len(source) == source.space._arc_number + source.space._line_number


def test_line_start_axes():
    polyline = PolyLine(
        np.array([(0, 0, 0), (1, 0, 0), (1, 2, 0)], float), minimum_arc_nodes=4
    )
    source = Source(polyline.path_geometry)
    axes = source.space.start_axes
    assert np.allclose(axes[0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(axes[1], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


def test_arc_start_axes():
    polyline = PolyLine(
        np.array([(0, 0, 0), (1, 1, 0), (0, 2, 0)], float), minimum_arc_nodes=3
    )
    source = Source(polyline.path_geometry)
    axes = source.space.start_axes
    assert np.allclose(axes[0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


if __name__ == "__main__":
    pytest.main([__file__])
