import pytest

import numpy as np

from nova.frame.framespace import FrameSpace


def test_drop_subspace():
    framespace = FrameSpace(Required=['x', 'z'], label='PF',
                            Additional=['Ic'], Subspace=['Ic'])
    framespace.insert(2, range(2))
    framespace.insert(1, range(3), link=True)
    framespace.insert(3, 7)
    framespace.drop('PF4')
    framespace.drop(['PF0', 'PF1'])
    assert framespace.subspace.index.to_list() == ['PF2', 'PF5']


def test_multipoint_factor():
    framespace = FrameSpace(required=['Ic'], Subspace=['Ic'], Array=['Ic'])
    framespace.insert(5*np.ones(2), nturn=[1, 3])
    framespace.insert(2*np.ones(2), nturn=[2, 1], link=True, factor=-0.5)
    framespace.insert(7.75*np.ones(2))
    framespace.insert(6*np.ones(3), link=True, factor=-5.0, nturn=[1, 2, 3])
    assert framespace.loc[:, 'It'].to_list() == [5, 15, 4, -1, 7.75, 7.75,
                                                 6, -60, -90]


def test_multipoint_factor_Ic_It():
    framespace = FrameSpace(required=['Ic'],
                            Subspace=['Ic', 'It'], Array=['Ic'])
    framespace.insert(5*np.ones(2), nturn=[1, 3])  # note, turn info discarded
    framespace.insert(2*np.ones(2), nturn=[2, 1], link=True, factor=-0.5)
    framespace.insert(7.75*np.ones(2))
    framespace.insert(6*np.ones(3), link=True, factor=-5.0, nturn=[1, 2, 3])
    assert framespace.loc[:, 'It'].to_list() == [5, 15, 4, 4, 7.75, 7.75,
                                                 6, 6, 6]


def test_fix_aspect():
    framespace = FrameSpace(required=['x', 'z', 'dl', 'dt'],
                            available=['section', 'poly'])
    framespace.insert(4, 6, 0.1, 0.5, section='sq')
    assert framespace.dx[0] == framespace.dz[0]


def test_rectangular_cross_section():
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle')
    assert np.isclose(2.5*1.5, framespace.area[0], rtol=1e-8)


def test_skin_section_area():
    framespace = FrameSpace(Required=['x', 'z'])
    framespace.insert(1.75, 0.2, dl=0.5, dt=0.1, section='skin')
    assert np.isclose(framespace.area[0], framespace.poly[0].area, rtol=1e-1)


def test_loop_length():
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle')
    assert np.isclose(framespace.dy[0], 2*np.pi*1.75)


def test_loop_length_factor():
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle', dy=-0.5)
    assert np.isclose(framespace.dy[0], np.pi*1.75)


def test_loop_length_abs():
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle', dy=22.3)
    assert np.isclose(framespace.dy[0], 22.3)


if __name__ == '__main__':

    pytest.main([__file__])
