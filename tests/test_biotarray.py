import pytest

import numpy as np
import xarray

from nova.biot.array import Array


@pytest.fixture
def data():
    return xarray.Dataset(dict(x=range(3), z=range(4)))


def test_keyerror(data):
    array = Array(data)
    with pytest.raises(KeyError):
        array['x']


def test_getitem(data):
    array = Array(data, array_attrs=['x'])
    array.load_arrays()
    assert np.allclose(array['x'], range(3))


def test_keys(data):
    array = Array(data, array_attrs=['x', 'z'])
    array.load_arrays()
    assert list(array.array.keys()) == ['x', 'z']


def test_skip_attr(data):
    array = Array(data, array_attrs=['x', 'y', 'z'])
    array.load_arrays()
    assert list(array.array.keys()) == ['x', 'z']
    with pytest.raises(KeyError):
        array['y']


if __name__ == '__main__':
    pytest.main([__file__])
