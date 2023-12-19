import pytest

import numpy as np

from nova.frame.coilset import CoilSet


@pytest.mark.parametrize("shape", [(2,), (3,), (2, 2), (2, 3), (1, 43, 2), (4, 4, 3)])
def test_shape(shape):
    coilset = CoilSet()
    coilset.coil.insert(3, 5.4, 0.1, 0.1)
    coilset.point.solve(np.ones(shape))


@pytest.mark.parametrize("shape", [(1,), (3, 1), (5, 4, 4)])
def test_shape_error(shape):
    coilset = CoilSet()
    coilset.coil.insert(3, 5.4, 0.1, 0.1)
    with pytest.raises(ValueError):
        coilset.point.solve(np.ones(shape))


@pytest.mark.parametrize("shape", [(2,), (3, 1, 7, 3), (1, 1, 2)])
def test_data(shape):
    coilset = CoilSet()
    coilset.coil.insert(3, 5.4, 0.1, 0.1)
    points = np.ones(shape)
    coilset.point.solve(points)
    point_number = np.prod(points.reshape(-1, points.shape[-1]).shape[:-1])
    assert coilset.point.data.sizes["target"] == point_number
    assert all([attr in coilset.point.data.coords for attr in "xyz"])


if __name__ == "__main__":
    pytest.main([__file__])
