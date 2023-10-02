import pytest

import numpy as np

from nova.biot.arc import Arc
from nova.frame.coilset import CoilSet


@pytest.fixture
def arc():
    coilset = CoilSet()
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]])
    )
    coilset.winding.insert(
        {"c": (0, 0, 0.5)}, np.array([[5, 0, 3.2], [0, 0, -1.8], [-5, 0, 3.2]])
    )
    return Arc(coilset.subframe)


def test_transform_shape(arc):
    assert np.shape(arc.source.space.transform) == (2, 3, 3)


def test_end_points(arc):
    assert np.allclose(arc.source.start_point, [5, 0, 3.2])
    assert np.allclose(arc.source.end_point, [-5, 0, 3.2])


def test_transform_roundtrip(arc):
    assert np.allclose(
        arc.source.start_point, arc._to_global(arc._to_local(arc.source.start_point))
    )


def test_transform_local_plane(arc):
    assert np.isclose(
        arc._to_local(arc.source.start_point)[0, 2],
        arc._to_local(arc.source.end_point)[0, 2],
    )


def test_transform_local_axis(arc):
    assert np.allclose(arc._to_local(arc.source.axis)[1], [0, 0, 1])


if __name__ == "__main__":
    pytest.main([__file__])
