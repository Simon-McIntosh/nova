
import tempfile

import unittest.mock

import numpy as np
import pytest

from nova.frame.coilset import CoilSet
from nova.geometry.polygon import Polygon


def test_dpol():
    coilset = CoilSet(dcoil=0.15)
    assert coilset.dcoil == 0.15


def test_dplasma():
    coilset = CoilSet(nplasma=-0.333)
    assert coilset.nplasma == -0.333


def test_dshell():
    coilset = CoilSet(dshell=0.7)
    assert coilset.dshell == 0.7


def test_nfield():
    coilset = CoilSet(nfield=1)
    assert coilset.nfield == 1


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
    assert np.isclose(np.pi*2.5**2/4, coilset.subframe.area.sum(), rtol=5e-3)


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
    coilset.linkframe(['PF6', 'PF8'], -0.5)
    index = coilset.loc['frame'] == 'PF8'
    assert np.isclose(coilset.loc[index, 'It'].sum(), -0.5*1e6/4*5)


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
    assert np.isclose(coilset.subframe.area.sum(), np.pi*0.5**2/4, 5e-3)


def test_plasma_single():
    coilset = CoilSet(nplasma=-0)
    coilset.firstwall.insert([[1, 2, 2, 1.5, 1, 1], [1, 1, 2, 2.5, 1.5, 1]],
                             turn='r', tile=False)
    assert len(coilset.subframe) == 1


def test_plasma_hex():
    coilset = CoilSet(nplasma=-0.5)
    coilset.firstwall.insert([[1, 2, 2, 1.5, 1, 1], [1, 1, 2, 2.5, 1.5, 1]],
                             turn='hex')
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
    coilset.linkframe(['Coil2', 'Coil3'])
    coilset.sloc['Coil0', 'Ic'] = 11.1
    assert all(coilset.subframe.Ic == 11.1)


def test_array_format():
    coilset = CoilSet(dcoil=-3)
    coilset.coil.insert(0.8, [0.5, 1, 1.5], 0.25, 0.45, link=True)
    coilset.sloc['Ic'] = 11
    assert isinstance(coilset.sloc[0, 'Ic'], float)


def test_store_load_poly():
    coilset = CoilSet(dcoil=-3, nplasma=8)
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.filepath = tmp.name
        coilset.store()
        new_coilset = CoilSet()
        new_coilset.filepath = tmp.name
        new_coilset.load()
        coilset._clear()
    assert np.isclose(coilset.frame.poly[0].area,
                      new_coilset.frame.poly[0].area)


def test_store_load_multipoly():
    coilset = CoilSet(dcoil=-3, nplasma=8)
    coilset.coil.insert(Polygon(dict(rect=(1, 2, 0.3, 0.6), disc=[4, 3, 0.5])))
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.filepath = tmp.name
        coilset.store()
        new_coilset = CoilSet()
        new_coilset.filepath = tmp.name
        new_coilset.load()
        coilset._clear()
    assert np.isclose(coilset.frame.poly[0].area,
                      new_coilset.frame.poly[0].area)


def test_store_load_version():
    coilset = CoilSet(dcoil=-3, nplasma=8)
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.filepath = tmp.name
        coilset.store()
        new_coilset = CoilSet()
        new_coilset.filepath = tmp.name
        new_coilset.load()
        coilset._clear()
    assert coilset.frame.version['index'] != new_coilset.frame.version['index']


def test_plasma_array_attributes():
    coilset = CoilSet(nplasma=5)
    coilset.firstwall.insert({'ellip': [2.5, 1.7, 1.6, 2.2]}, turn='hex')
    _ = coilset.plasma
    assert all([attr in coilset.subframe.metaframe.array for attr in
                ['Ic', 'nturn', 'area']])


def test_array_views_insert():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, -0.5, 0.95, 0.95)
    nturn = coilset.loc['nturn']
    coilset.coil.insert(1, -0.5, 0.95, 0.95)
    assert id(nturn) != id(coilset.loc['nturn'])
    assert all(np.isnan(nturn))


def test_array_views_insert_subspace():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, -0.5, 0.95, 0.95)
    Ic = coilset.sloc['Ic']
    coilset.coil.insert(1, -0.5, 0.95, 0.95)
    assert id(Ic) != id(coilset.sloc['Ic'])
    assert all(np.isnan(Ic))


def test_array_views_solve():
    coilset = CoilSet(dcoil=-5)
    coilset.firstwall.insert(3, -0.5, 0.95, 0.95, delta=-5)
    coilset.coil.insert(3, -0.5, 0.95, 0.95)
    Ic = coilset.sloc['Ic']
    nturn = coilset.loc['nturn']
    coilset.sloc['Ic'] = 7.7
    coilset.sloc['plasma', 'Ic'] = 6.6
    coilset.grid.solve(10, 0.05)
    Ic[:1] = 5.5
    assert id(Ic) == id(coilset.sloc['Ic'])
    assert id(nturn) == id(coilset.loc['nturn'])
    assert np.allclose(Ic, coilset.sloc['Ic'])


def test_biot_solve_index_version():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, -0.5, 0.95, 0.95)
    coilset.grid.solve(10, 0.05)
    index_hash = coilset.subframe.loc_hash('index')
    assert coilset.subframe.version['index'] == index_hash


def test_biot_solve_no_plasma():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, -0.5, 0.95, 0.95)
    coilset.plasma.separatrix = Polygon(dict(o=(3, 0, 1)))
    coilset.grid.solve(10, 0.05)
    coilset.sloc['Ic'] = 5
    assert (coilset.grid.psi != 0).all()


def test_biotdata_numpy():
    with unittest.mock.patch.dict('os.environ', dict(XPU='numpy')):
        coilset = CoilSet(nplasma=3)
        coilset.firstwall.insert(3, -0.5, 0.95, 0.95)
        coilset.grid.solve(10, 0.05)
        assert isinstance(coilset.grid.operator['Psi'].matrix, np.ndarray)


def test_biot_link_dataarray_dataset():
    coilset = CoilSet(nplasma=20)
    coilset.firstwall.insert(3, -0.5, 0.95, 0.95)
    coilset.grid.solve(10, 0.05)
    Psi = coilset.grid.operator['Psi'].matrix.copy()
    coilset.plasma.separatrix = ((2.5, -1), (3.5, -1), (3, 0))
    coilset.grid.update_turns('Psi')
    assert (coilset.grid.operator['Psi'].matrix ==
            coilset.grid.data['Psi']).all()
    assert (Psi != coilset.grid.data['Psi']).any()


def test_biot_multiframe_plasma():
    coilset = CoilSet(nplasma=20, dcoil=-1)
    coilset.coil.insert(3, 0, 0.25, 0.25, Ic=1)
    coilset.firstwall.insert(3, -0.5, 0.5, 0.5, Ic=1)
    coilset.firstwall.insert(3, 0.5, 0.5, 0.5, name='second_plasma_rejoin')
    coilset.grid.solve(50, 0.05)
    assert coilset.grid.data.attrs['plasma_index'] == 1


def test_add_coilset():
    active = CoilSet(nplasma=5, dcoil=-5)
    active.coil.insert(4, 5, 0.1, 0.1, name='PF1')
    active.firstwall.insert(3, 4, 0.5, 0.8, name='pl')
    passive = CoilSet(dshell=-10)
    passive.shell.insert({'e': [3, 4, 1.6, 2.2]}, -2, 0.1, name='shell')
    coilset = active + passive
    assert coilset.frame.index.to_list() == ['PF1', 'pl', 'shell0', 'shell1']


def test_iadd_coilset():
    coilset = CoilSet(nplasma=5, dcoil=-5)
    coilset.coil.insert(4, 5, 0.1, 0.1, name='PF1')
    coilset.firstwall.insert(3, 4, 0.5, 0.8, name='pl')
    passive = CoilSet(dshell=-10)
    passive.shell.insert({'e': [3, 4, 1.6, 2.2]}, -2, 0.1, name='shell')
    coilset += passive
    assert coilset.frame.index.to_list() == ['PF1', 'pl', 'shell0', 'shell1']


def test_add_clear_biot():
    active = CoilSet(nplasma=5, dcoil=-5)
    active.coil.insert(4, 5, 0.1, 0.1, name='PF1')
    active.firstwall.insert(3, 4, 0.5, 0.8, name='pl')
    active.probe.solve([(1, 2), (3, 4), (1, 1)])
    passive = CoilSet(dshell=-10)
    passive.shell.insert({'e': [3, 4, 1.6, 2.2]}, -2, 0.1, name='shell')
    coilset = active + passive
    assert len(coilset.probe.data) == 0
    assert len(active.probe.data) != 0


def test_iadd_clear_biot():
    coilset = CoilSet(nplasma=5, dcoil=-5)
    coilset.coil.insert(4, 5, 0.1, 0.1, name='PF1')
    coilset.firstwall.insert(3, 4, 0.5, 0.8, name='pl')
    coilset.probe.solve([(1, 2), (3, 4), (1, 1)])
    passive = CoilSet(dshell=-10)
    passive.shell.insert({'e': [3, 4, 1.6, 2.2]}, -2, 0.1, name='shell')
    coilset += passive
    assert len(coilset.probe.data) == 0


if __name__ == '__main__':

    pytest.main([__file__])
