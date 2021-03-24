import pytest

from nova.electromagnetic.coilframe import CoilFrame


def test_dpol():
    coilframe = CoilFrame(dpol=0.15)
    assert coilframe.dpol == 0.15


def test_dplasma():
    coilframe = CoilFrame(dplasma=0.333)
    assert coilframe.dplasma == 0.333


def test_dshell():
    coilframe = CoilFrame(dshell=0.7)
    assert coilframe.dshell == 0.7


def test_dfield():
    coilframe = CoilFrame(dfield=-1)
    assert coilframe.dfield == -1


def test_dpol_default():
    coilframe = CoilFrame(dpol=3)
    coilframe.add_poloidal(range(3), 1, 0.1, 0.1,
                           mesh=False, dpol=-1, label='PF')
    assert coilframe.frame.loc['PF0', 'delta'] == -1
    assert coilframe.dpol == 3


if __name__ == '__main__':
    pytest.main([__file__])
