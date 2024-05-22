import pytest

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.matrix import Matrix
from nova.geometry.polyline import PolyLine


@pytest.fixture
def matrix():
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [3, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=4)
    source = Source(polyline.path_geometry)
    target = Target({"x": np.linspace(5, 7.5, 10), "z": 0.5})
    return Matrix(source, target)


def test_coordinate_axes_shape(matrix):
    assert matrix.coordinate_axes.shape == (10, 4, 3, 3)


def test_stack_shape(matrix):
    points = matrix.target.stack(*list("xyz"))
    assert points.shape == (10, 4, 3)


def test_coordinate_axes_einsum_shape(matrix):
    points = matrix.target.stack(*list("xyz"))
    _points = np.einsum("ijk,ijkm->ijm", points, matrix.coordinate_axes)
    assert points.shape == _points.shape


def test_coord_loc(matrix):
    assert len(matrix.loc.data["source"]) == 0
    assert len(matrix.loc.data["target"]) == 0
    assert matrix.loc["source", "x"].shape == matrix.shape
    assert list(matrix.loc.data["source"].keys()) == ["x", "y", "z"]


def test_source_coordinates_roundtrip(matrix):
    points = matrix.source.stack("x1", "y1", "z1")
    local_points = matrix.loc.to_local(points)
    global_points = matrix.loc.to_global(local_points)
    assert np.allclose(local_points[..., :2], 0)
    assert np.allclose(points, global_points)


@pytest.mark.parametrize("frame", ["source", "target"])
def test_local_frame_roundtrip(matrix, frame):
    points = getattr(matrix, frame).stack(*list("xyz"))
    local_points = np.stack([matrix.loc[frame, attr] for attr in "xyz"], axis=-1)
    global_points = matrix.loc.to_global(local_points)
    assert np.allclose(points, global_points)


if __name__ == "__main__":
    pytest.main([__file__])
