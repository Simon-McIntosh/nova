import pytest

import numpy as np

from nova.biot.biotframe import Source, Target
from nova.biot.groupset import GroupSet
from nova.geometry.polyline import PolyLine


@pytest.fixture
def groupset():
    rng = np.random.default_rng(2025)
    points = rng.random((4, 3))
    polyline = PolyLine(points)
    source = Source(polyline.path_geometry)
    target = Target({"x": np.linspace(5, 7.5, 10), "z": 0.5})
    return GroupSet(source, target)


def test_transform_shape(groupset):
    assert groupset.transform.shape == (10, 2, 3, 3)


def test_stack_shape(groupset):
    points = groupset.target.stack(*list("xyz"))
    assert points.shape == (10, 2, 3)


def test_transform_einsum_shape(groupset):
    points = groupset.target.stack(*list("xyz"))
    _points = np.einsum("ijk,ijkm->ijm", points, groupset.transform)
    assert points.shape == _points.shape


if __name__ == "__main__":
    pytest.main([__file__])
