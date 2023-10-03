import pytest

import numpy as np

from nova.biot.arc import Arc
from nova.biot.biotframe import Source, Target
from nova.frame.coilset import CoilSet


@pytest.fixture
def source():
    coilset = CoilSet()
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]])
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 0, -1.8], [-5, 0, 3.2]])
    )
    return Source(coilset.subframe)


@pytest.fixture
def arc(source):
    return Arc(source, Target({"x": np.linspace(2, 5, 7), "z": -0.3}))


def test_transform_shape(source):
    assert np.shape(source.space.transform) == (2, 3, 3)


def test_end_points(source):
    assert np.allclose(source.start_point, [5, 0, 3.2])
    assert np.allclose(source.end_point, [-5, 0, 3.2])


def test_transform_roundtrip(source):
    assert np.allclose(
        source.start_point,
        source.space.to_global(source.space.to_local(source.start_point)),
    )


def test_transform_local_plane(source):
    assert np.isclose(
        source.space.to_local(source.start_point)[0, 2],
        source.space.to_local(source.end_point)[0, 2],
    )


def test_transform_local_axis(source):
    assert np.allclose(source.space.to_local(source.axis)[1], [0, 0, 1])


if __name__ == "__main__":
    pytest.main([__file__])
