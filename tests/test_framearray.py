
import pytest
import numpy as np

from nova.electromagnetic.coilframe import CoilFrame


def test_getattr():
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
    assert not frame.metaarray.update_array['x']


def test_getattr_update():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x', 'z']})
    _ = frame.x
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and not update_frame['x']


def test_setattr_update():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x', 'z']})
    frame.x = [1, 2]
    update_array = frame.metaarray.update_array
    update_frame = frame.metaarray.update_frame
    assert not update_array['x'] and update_frame['x']

if __name__ == '__main__':

    pytest.main([__file__])
