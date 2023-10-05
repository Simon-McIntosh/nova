from itertools import product
import pytest

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.groupset import GroupSet
from nova.geometry.polyline import PolyLine


@pytest.fixture
def multiline():
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=4)
    source = Source(polyline.path_geometry)
    source.set_target(1)
    return source


@pytest.fixture
def multiarc():
    points = np.array(
        [[-1, 2, 0], [-1, 1, -2], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float
    )
    polyline = PolyLine(points, minimum_arc_nodes=3)
    source = Source(polyline.path_geometry)
    source.set_target(1)
    return source


@pytest.fixture
def groupset():
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [3, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=4)
    source = Source(polyline.path_geometry)
    target = Target({"x": np.linspace(5, 7.5, 10), "z": 0.5})
    return GroupSet(source, target)


def test_coordinate_axes_shape(groupset):
    assert groupset.coordinate_axes.shape == (10, 4, 3, 3)


def test_stack_shape(groupset):
    points = groupset.target.stack(*list("xyz"))
    assert points.shape == (10, 4, 3)


def test_coordinate_axes_einsum_shape(groupset):
    points = groupset.target.stack(*list("xyz"))
    _points = np.einsum("ijk,ijkm->ijm", points, groupset.coordinate_axes)
    assert points.shape == _points.shape


def test_coord_loc(groupset):
    assert len(groupset.local.data["source"]) == 0
    assert len(groupset.local.data["target"]) == 0
    assert groupset.local["source", "x"].shape == groupset.shape
    assert list(groupset.local.data["source"].keys()) == ["x", "y", "z"]


def test_source_coordinates_roundtrip(groupset):
    points = groupset.source.stack("x1", "y1", "z1")
    local_points = groupset.local.to_local(points)
    global_points = groupset.local.to_global(local_points)
    assert np.allclose(local_points[..., :2], 0)
    assert np.allclose(points, global_points)


@pytest.mark.parametrize("frame", ["source", "target"])
def test_local_frame_roundtrip(groupset, frame):
    points = getattr(groupset, frame).stack(*list("xyz"))
    local_points = np.stack([groupset.local[frame, attr] for attr in "xyz"], axis=-1)
    global_points = groupset.local.to_global(local_points)
    assert np.allclose(points, global_points)


@pytest.mark.parametrize("source, axis", product(["multiline", "multiarc"], ["n", "a"]))
def test_source_axes(source, axis, request):
    source = request.getfixturevalue(source)
    vector = {"n": [1, 0, 0], "a": [0, 0, 1]}[axis]
    points = source.stack(*[f"{axis}{coord}" for coord in "xyz"])
    local_points = source.space._rotate_to_local(points[0])
    global_points = source.space._rotate_to_global(local_points)
    assert np.allclose(local_points, np.tile(vector, (1, len(points))))
    assert np.allclose(points, global_points)


if __name__ == "__main__":
    pytest.main([__file__])
