from itertools import product
import numpy as np
import pytest

from nova.geometry.rotate import to_vector


def test_to_vector():
    rotate = to_vector([0, 0, 1], [0, 1, 0])
    assert np.allclose(rotate.as_rotvec(), [-np.pi / 2, 0, 0])


def test_to_vector_pi():
    rotate = to_vector([0, 0, 1], [0, 0, -1])
    assert np.allclose(rotate.as_rotvec(), [np.pi, 0, 0])


@pytest.mark.parametrize(
    "axis,vector",
    product(
        [[0, 0, 1], [0.4, 5, -3.2], [0, -1, 1]],
        [[1, 0, 3.2], [0, 0, 1], [7.6, -3, -2.7], [0, 0, -1]],
    ),
)
def test_dot(axis, vector):
    rotate = to_vector(axis, vector)
    axis = rotate.apply(axis)
    assert np.isclose(
        np.dot(axis, vector), np.linalg.norm(axis) * np.linalg.norm(vector)
    )


if __name__ == "__main__":
    pytest.main([__file__])
