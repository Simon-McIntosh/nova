
import pytest
import numpy as np

from nova.electromagnetic.coilframe import CoilFrame


def test_getattr_():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x']})
    assert isinstance(frame.x, np.ndarray)


def test_setattr_ndarray():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x', 'z']})
    assert isinstance(frame.x, np.ndarray)


def test_setattr_update_array_false():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x', 'z']})
    _ = frame.x
    assert not frame.metaarray.update_frame['x']


def test_setattr_update_array_true():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x', 'z']})
    _ = frame.x
    assert not frame.metaarray.update_frame['z']


if __name__ == '__main__':

    pytest.main([__file__])
