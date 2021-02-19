
import pytest
import numpy as np

from nova.electromagnetic.coilframe import CoilFrame


def test_getattr_():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x']})
    assert isinstance(frame.x, np.ndarray)

def test_setattr_update():
    frame = CoilFrame({'x': [3, 2], 'z': 0},
                      metadata={'Required': ['x', 'z'],
                                'Array': ['x']})
    #frame.x = [5, 6]
    print(frame)
    print(frame.x)
    #assert isinstance(frame.x, np.ndarray)


if __name__ == '__main__':

    #pytest.main([__file__])
    test_setattr_update()
