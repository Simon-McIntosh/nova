import pytest

import numpy as np

from nova.electromagnetic.coilset import CoilSet


def test_dpol():
    coilset = CoilSet(dcoil=0.15)
    assert coilset.dcoil == 0.15


def test_dplasma():
    coilset = CoilSet(dplasma=0.333)
    assert coilset.dplasma == 0.333


def test_dshell():
    coilset = CoilSet(dshell=0.7)
    assert coilset.dshell == 0.7


def test_dfield():
    coilset = CoilSet(dfield=-1)
    assert coilset.dfield == -1


def test_dpol_default():
    coilset = CoilSet(dcoil=3)
    coilset.coil.insert(range(3), 1, 0.1, 0.1, mesh=False, delta=-1,
                        label='PF')
    assert coilset.frame.loc['PF0', 'delta'] == -1
    assert coilset.dcoil == 3


def test_pfcoil():
    coilset = CoilSet(required=['x', 'z', 'dl'])
    coilset.coil.insert([4, 6, 7, 9, 9.5, 6], [1, 1, 2, 1.5, -1, -1.5],
                        0.1, delta=0.05, label='PF')
    assert len(coilset.frame) == 6
    assert len(coilset.subframe) == 24


def test_filament_number():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, scale=1, delta=-20)
    assert coilset.frame['Nf'][0] == 20


def test_circular_cross_section():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, section='circle',
                        scale=1, delta=-20)
    assert np.isclose(np.pi*2.5**2/4, coilset.subframe.dA.sum(), rtol=1e-3)


def test_rectangular_cross_section():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 1.5, section='rectangle',
                        scale=1, delta=-20)
    assert np.isclose(2.5*1.5, coilset.subframe.dA.sum(), rtol=1e-3)


def test_flag_current_update():
    coilset = CoilSet(dcoil=0)
    coilset.coil.insert(1, 2, 0.4, 0.5, Ic=5, active=True)
    coilset.coil.insert(1, 3, 0.4, 0.5, Ic=6.7, active=False)
    coilset.coil.insert(1, 4, 0.4, 0.5, Ic=4, active=True)
    coilset.frame.subspace.loc[coilset.frame.active, 'Ic'] = [3.2, 5.8]
    assert np.allclose(coilset.frame.Ic, [3.2, 6.7, 5.8])


def test_multipoint_link():
    coilset = CoilSet(dcoil=0, subspace=['Ic'])
    coilset.coil.insert(7, -3, 1.5, 1.5, name='PF6', part='PF',
                        Nt=4, It=1e6, turn='circle', scale=0.7,
                        delta=0.12)
    coilset.coil.insert(7, -0.5, 1.5, 1.5, name='PF8', part='PF', Nt=5,
                        Ic=2e3, section='circle', turn='square', delta=0.12)
    # Ic[PF8] = -0.5*Ic[PF6]
    coilset.frame.multipoint.link(['PF6', 'PF8'], -0.5)
    assert coilset.frame.loc[:, 'It'].to_list() == [1e6, -0.5*1e6/4*5]


def test_shell_cross_section():
    coilset = CoilSet()
    coilset.shell.insert([1, 1, 3], [3, 4, 4], dt=0.1)
    assert np.isclose(coilset.frame.dA.sum(), 3*0.1, atol=5e-3)


def test_shell_additional():
    coilset = CoilSet()
    coilset.shell.insert([1, 1, 3], [3, 4, 4], dt=0.1, plasma=True)
    assert coilset.frame.plasma.to_numpy().all()


def test_shell_subshell():
    coilset = CoilSet()
    coilset.shell.insert([1, 2], [5, 5], dt=0.1, delta=0.1)
    assert len(coilset.subframe) == 10


def test_shell_turns():
    coilset = CoilSet()
    coilset.shell.insert([1, 2], [5, 5], dt=0.1, delta=0.1)
    assert np.isclose(coilset.subframe.dA, coilset.subframe.Nt).all()


def test_shell():
    coilset = CoilSet()
    coilset.shell.insert([4, 6, 7, 9, 9.5, 6],
                         [1, 1, 2, 1.5, -1, -1.5], 0.1,
                         dshell=1, delta=-1, label='vvin')
    assert len(coilset.frame) == 11


if __name__ == '__main__':

    pytest.main([__file__])
