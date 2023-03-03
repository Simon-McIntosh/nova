import pytest

import numpy as np
import xarray

from nova.biot.biotarray import BiotArray


@pytest.fixture
def data():
    return xarray.Dataset(dict(x=range(3), z=range(4)))


def test_keyerror(data):
    biotarray = BiotArray(data)
    with pytest.raises(KeyError):
        biotarray['x']


def test_getitem(data):
    biotarray = BiotArray(data, array_attrs=['x'])
    biotarray.load_arrays()
    assert np.allclose(biotarray['x'], range(3))


def test_keys(data):
    biotarray = BiotArray(data, array_attrs=['x', 'z'])
    biotarray.load_arrays()
    assert list(biotarray.array.keys()) == ['x', 'z']


def test_skip_attr(data):
    biotarray = BiotArray(data, array_attrs=['x', 'y', 'z'])
    biotarray.load_arrays()
    assert list(biotarray.array.keys()) == ['x', 'z']
    with pytest.raises(KeyError):
        biotarray['y']


if __name__ == '__main__':
    pytest.main([__file__])
