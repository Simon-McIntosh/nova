import pytest

import numpy as np

from nova.electromagnetic.frame import Frame


def test_fix_aspect():
    frame = Frame()
    frame.insert(4, 6, 0.1, 0.5, section='sq')
    assert frame.dx[0] == frame.dz[0]


def test_free_aspect():
    frame = Frame()
    frame.insert(4, 6, 0.1, 0.5, section='r')
    assert frame.dx[0] != frame.dz[0]


def test_circular_cross_section():
    'check frame area sum equals circle area'
    frame = Frame(Required=['x', 'z', 'dl', 'dt'])
    frame.insert(1.75, 0.5, 2.5, 2.5, section='o')
    assert np.isclose(np.pi*2.5**2/4, frame.dA[0], rtol=1e-8)


def test_rectangular_cross_section():
    'check subcoil area sum equals circle area'
    frame = Frame(Required=['x', 'z', 'dl', 'dt'])
    frame.insert(1.75, 0.5, 2.5, 1.5, section='rectangle')
    assert np.isclose(2.5*1.5, frame.dA[0], rtol=1e-8)


def test_invalid_cross_section():
    'check frame area sum equals circle area'
    frame = Frame(Required=['x', 'z', 'dl', 'dt'])
    with pytest.raises(KeyError):
        frame.insert(1.75, 0.5, 2.5, 2.5, section='P')


if __name__ == '__main__':

    pytest.main([__file__])
