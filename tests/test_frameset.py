import pytest

from nova.electromagnetic.frameset import FrameSet


def test_dpol():
    frameset = FrameSet(dpol=0.15)
    assert frameset.dpol == 0.15


def test_dplasma():
    frameset = FrameSet(dplasma=0.333)
    assert frameset.dplasma == 0.333


def test_dshell():
    frameset = FrameSet(dshell=0.7)
    assert frameset.dshell == 0.7


def test_dfield():
    frameset = FrameSet(dfield=-1)
    assert frameset.dfield == -1


def test_dpol_default():
    frameset = FrameSet(dpol=3)
    frameset.poloidal(range(3), 1, 0.1, 0.1, mesh=False, dpol=-1, label='PF')
    assert frameset.frame.loc['PF0', 'delta'] == -1
    assert frameset.dpol == 3


if __name__ == '__main__':
    pytest.main([__file__])
