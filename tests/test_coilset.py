
import tempfile

import numpy as np
import pytest

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
    coilset.coil.insert(range(3), 1, 0.1, 0.1, delta=-1, label='PF')
    assert coilset.frame.loc['PF0', 'delta'] == -1
    assert coilset.dcoil == 3


def test_pfcoil():
    coilset = CoilSet()
    coilset.coil.required = ['x', 'z', 'dl']
    coilset.coil.insert([4, 6, 7], [1, 1, 2], 0.1, delta=0.05, label='PF')
    assert len(coilset.frame) == 3
    assert len(coilset.subframe) == 12


def test_subframe_turn_number():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, scale=1, delta=-4,
                        nturn=22.2)
    assert coilset.subframe.nturn.sum() == 22.2


def test_filament_number():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, scale=1, delta=-6)
    assert len(coilset.subframe) == 6


def test_square_filament_number():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, scale=1, delta=-6, turn='sq')
    assert len(coilset.subframe) == 4


def test_circular_cross_section():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 2.5, section='disc', turn='r',
                        delta=-4)
    assert np.isclose(np.pi*2.5**2/4, coilset.subframe.area.sum(), rtol=1e-3)


def test_rectangular_cross_section():
    coilset = CoilSet()
    coilset.coil.insert(1.75, 0.5, 2.5, 1.5, section='rectangle',
                        scale=1, delta=-6)
    assert np.isclose(2.5*1.5, coilset.subframe.area.sum(), rtol=1e-3)


def test_flag_current_update():
    coilset = CoilSet(dcoil=0)
    coilset.coil.insert(1, 2, 0.4, 0.5, Ic=5, active=True)
    coilset.coil.insert(1, 3, 0.4, 0.5, Ic=6.7, active=False)
    coilset.coil.insert(1, 4, 0.4, 0.5, Ic=4, active=True)
    coilset.subframe.subspace.loc[coilset.subframe.active, 'Ic'] = [3.2, 5.8]
    assert np.allclose(coilset.subframe.Ic, [3.2, 6.7, 5.8])


def test_multipoint_link():
    coilset = CoilSet(dcoil=0, subspace=['Ic'])
    coilset.coil.insert(7, -3, 1.5, 1.5, name='PF6', part='PF',
                        nturn=4, It=1e6, turn='disc', scale=0.7,
                        delta=0.12)
    coilset.coil.insert(7, -0.5, 1.5, 1.5, name='PF8', part='PF', nturn=5,
                        Ic=2e3, section='disc', turn='square', delta=0.12)
    # Ic[PF8] = -0.5*Ic[PF6]
    coilset.link(['PF6', 'PF8'], -0.5)
    assert coilset.subframe.It[coilset.subframe.frame == 'PF8'].sum() == \
        -0.5*1e6/4*5


def test_shell_cross_section():
    coilset = CoilSet()
    coilset.shell.insert([1, 1, 3], [3, 4, 4], 0, 0.1)
    assert np.isclose(coilset.frame.area.sum(), 3*0.1, atol=5e-3)


def test_shell_additional():
    coilset = CoilSet(array=['plasma'])
    coilset.shell.insert([1, 1, 3], [3, 4, 4], 0, 0.1, delta=0, plasma=True)
    assert coilset.subframe.plasma.all()


def test_shell_subshell():
    coilset = CoilSet()
    coilset.shell.insert([1, 2], [5, 5], 0, 0.01, delta=0.1)
    assert len(coilset.subframe) == 10


def test_shell_turns():
    coilset = CoilSet()
    coilset.shell.insert([1, 2], [5, 5], 0, 0.1, delta=0.1)
    assert np.isclose(coilset.subframe.area, coilset.subframe.nturn).all()


def test_shell():
    coilset = CoilSet()
    coilset.shell.insert([4, 6, 7, 9, 9.5, 6],
                         [1, 1, 2, 1.5, -1, -1.5], 1, 0.1,
                         delta=-1, label='vvin')
    assert len(coilset.frame) == 11


def test_aspect_horizontal():
    coilset = CoilSet()
    coilset.coil.insert(1, 1, 0.75, 0.5, delta=-9,
                        section='r', turn='s', fill=True)
    assert np.isclose(coilset.subframe.area.sum(), 0.75*0.5)


def test_aspect_vertical():
    coilset = CoilSet()
    coilset.coil.insert(1, 1, 0.5, 0.75, delta=-9,
                        section='r', turn='s', fill=True)
    assert np.isclose(coilset.subframe.area.sum(), 0.75*0.5)


def test_tile_hex():
    coilset = CoilSet()
    coilset.coil.insert(1, 1, 0.5, 0.75, delta=-4,
                        section='dsk', nturn=9, turn='hex', tile=True)
    assert np.isclose(coilset.subframe.area.sum(), np.pi*0.5**2/4, 1e-3)


def test_plasma_single():
    coilset = CoilSet(dplasma=0)
    coilset.plasma.insert([[1, 2, 2, 1.5, 1, 1], [1, 1, 2, 2.5, 1.5, 1]],
                          turn='r', tile=False)
    assert len(coilset.subframe) == 1


def test_plasma_hex():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert([[1, 2, 2, 1.5, 1, 1], [1, 1, 2, 2.5, 1.5, 1]])
    assert sum([section == 'hexagon'
                for section in coilset.subframe.section]) == 2


def test_coil_multipoint_link():
    coilset = CoilSet(dcoil=-3)
    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True)
    coilset.coil.insert(1.8, [0.5, 1], 0.25, 0.45)
    coilset.coil.insert(2.8, [0.5, 1, 1.5], 0.25, 0.45, link=True)
    assert len(coilset.sLoc) == 3
    assert len(coilset.sloc) == 3


def test_relink():
    coilset = CoilSet(dcoil=-3)
    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True)
    coilset.coil.insert(1.8, [0.5, 1], 0.25, 0.45, nturn=2)
    coilset.link(['Coil2', 'Coil3'])
    coilset.sloc['Coil0', 'Ic'] = 11.1
    assert all(coilset.subframe.Ic == 11.1)


def test_array_format():
    coilset = CoilSet(dcoil=-3)
    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True)
    coilset.sloc['Ic'] = 11
    assert isinstance(coilset.sloc[0, 'Ic'], float)


def test_store_load_poly():
    coilset = CoilSet(dcoil=-35, dplasma=-40)
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.store(tmp.name)
        new_coilset = CoilSet().load(tmp.name)
    assert np.isclose(coilset.frame.poly[0].area,
                      new_coilset.frame.poly[0].area)


if __name__ == '__main__':

    pytest.main([__file__])
