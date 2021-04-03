import pytest
import numpy as np
import pickle

from nova.electromagnetic.frameset import FrameSet



'''


def test_pickle():
    'test coilset sterilization'
    frameset = FrameSet()
    frameset.add_coil(1, [2, 3, 4], 0.4, 0.5, label='Coil', delim='_')
    _cs = FrameSet()
    cs_p = pickle.dumps(frameset.coilset)
    __cs = pickle.loads(cs_p)
    _cs.coilset = __cs
    assert frameset.coil.equals(_cs.coil)


if __name__ == '__main__':
    pytest.main([__file__])
'''
