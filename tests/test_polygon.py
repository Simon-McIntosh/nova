import pytest

import numpy as np

from nova.electromagnetic.framespace import FrameSpace


def test_fix_aspect():
    framespace = FrameSpace(required=['x', 'z', 'dl', 'dt'])
    framespace.insert(4, 6, 0.1, 0.5, section='sq')
    assert framespace.dx[0] == framespace.dz[0]


def test_free_aspect():
    framespace = FrameSpace(required=['x', 'z', 'dl', 'dt'])
    framespace.insert(4, 6, 0.1, 0.5, section='r')
    assert framespace.dx[0] != framespace.dz[0]


def test_circular_cross_section():
    'check framespace area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 2.5, section='o')
    assert np.isclose(np.pi*2.5**2/4, framespace.area[0], rtol=1e-8)


def test_rectangular_cross_section():
    'check subcoil area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle')
    assert np.isclose(2.5*1.5, framespace.area[0], rtol=1e-8)


def test_invalid_cross_section():
    'check framespace area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    with pytest.raises(KeyError):
        framespace.insert(1.75, 0.5, 2.5, 2.5, section='P')


if __name__ == '__main__':

    pytest.main([__file__])
