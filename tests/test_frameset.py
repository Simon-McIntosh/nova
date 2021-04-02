import pytest

import numpy as np

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
    frameset.pfcoil.insert(range(3), 1, 0.1, 0.1, mesh=False, delta=-1,
                           label='PF')
    assert frameset.frame.loc['PF0', 'delta'] == -1
    assert frameset.dpol == 3


def test_pfcoil():
    frameset = FrameSet(required=['x', 'z', 'dl'])
    frameset.pfcoil.insert([4, 6, 7, 9, 9.5, 6], [1, 1, 2, 1.5, -1, -1.5],
                           0.1, delta=0.05, label='PF')
    assert len(frameset.frame) == 6
    assert len(frameset.subframe) == 24


def test_filament_number():
    frameset = FrameSet(turn_fraction=0.5, section='circle')
    frameset.pfcoil.insert(1.75, 0.5, 2.5, 2.5, turn_fraction=1, delta=-20)
    assert frameset.frame['Nf'][0] == 20


def test_circular_cross_section():
    frameset = FrameSet(turn_fraction=0.5, section='circle')
    frameset.pfcoil.insert(1.75, 0.5, 2.5, 2.5, turn_fraction=1, delta=-20)
    assert np.isclose(np.pi*2.5**2/4, frameset.subframe.dA.sum(), rtol=1e-3)


def test_rectangular_cross_section():
    frameset = FrameSet(turn_fraction=0.5, section='rectangle')
    frameset.pfcoil.insert(1.75, 0.5, 2.5, 1.5, turn_fraction=1, delta=-20)
    assert np.isclose(2.5*1.5, frameset.subframe.dA.sum(), rtol=1e-3)


def test_flag_current_update():
    frameset = FrameSet(dpol=0)
    frameset.pfcoil.insert(1, 2, 0.4, 0.5, Ic=5, active=True)
    frameset.pfcoil.insert(1, 3, 0.4, 0.5, Ic=6.7, active=False)
    frameset.pfcoil.insert(1, 4, 0.4, 0.5, Ic=4, active=True)
    frameset.frame.subspace.loc[frameset.frame.active, 'Ic'] = [3.2, 5.8]
    assert np.allclose(frameset.frame.Ic, [3.2, 6.7, 5.8])


def test_multipoint_link():
    frameset = FrameSet(dpol=0, subspace=['Ic'])
    frameset.pfcoil.insert(7, -3, 1.5, 1.5, name='PF6', part='PF',
                           Nt=4, It=1e6, turn='circle', turn_fraction=0.7,
                           delta=0.12)
    frameset.pfcoil.insert(7, -0.5, 1.5, 1.5, name='PF8', part='PF', Nt=5,
                           Ic=2e3, section='circle', turn='square', delta=0.12)
    # Ic[PF8] = -0.5*Ic[PF6]
    frameset.frame.multipoint.link(['PF6', 'PF8'], -0.5)
    assert frameset.frame.loc[:, 'It'].to_list() == [1e6, -0.5*1e6/4*5]


def test_shell_cross_section():
    frameset = FrameSet()
    frameset.pfshell.insert([1, 1, 3], [3, 4, 4], dt=0.1)
    assert np.isclose(frameset.frame.dA.sum(), 3*0.1, atol=5e-3)


def test_shell_additional():
    frameset = FrameSet()
    frameset.pfshell.insert([1, 1, 3], [3, 4, 4], dt=0.1, plasma=True)
    assert frameset.frame.plasma.to_numpy().all()


def test_shell_subshell():
    frameset = FrameSet()
    frameset.pfshell.insert([1, 2], [5, 5], dt=0.1, delta=0.1)
    assert len(frameset.subframe) == 10


def test_shell_turns():
    frameset = FrameSet()
    frameset.pfshell.insert([1, 2], [5, 5], dt=0.1, delta=0.1)
    assert np.isclose(frameset.subframe.dA, frameset.subframe.Nt).all()


if __name__ == '__main__':

    pytest.main([__file__])
