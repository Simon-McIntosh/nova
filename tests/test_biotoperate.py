import pytest

import numpy as np

from nova.frame.coilset import CoilSet


def test_grid_shape():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, 0, 0.1, 0.1)
    coilset.grid.solve(10)
    assert coilset.grid.shape == (coilset.grid.data.dims['x'],
                                  coilset.grid.data.dims['z'])


def test_grid_shaped_array():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(3, 0, 0.1, 0.5)
    coilset.grid.solve(9)
    assert coilset.grid.shape == coilset.grid.psi_.shape


def test_grid_shaped_array_address():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(5, -2, 0.7, 0.5)
    coilset.grid.solve(5)
    psi_ = coilset.grid.psi_
    coilset.sloc['Ic'] = 10
    assert psi_.ctypes.data == coilset.grid.psi_.ctypes.data


def test_point_shaped_array():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(5, -2, 0.7, 0.5, Ic=10)
    coilset.point.solve(((1, 2), (4, 5), (7, 3)))
    assert len(coilset.point.shape) == 1


def test_point_shaped_array_address():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(5, -2, 0.7, 0.5, Ic=-10)
    coilset.point.solve(((1, 12), (4, 5), (7, -3)))
    assert coilset.point.psi.ctypes.data == coilset.point.psi_.ctypes.data


def test_nturn_hash_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    nturn_hash = coilset.subframe.version['nturn']
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    assert coilset.subframe.version['nturn'] != nturn_hash


def test_nturn_Psi_update():
    coilset = CoilSet(dcoil=-5, dplasma=-15)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    Psi = coilset.plasmagrid.data['Psi'].values.copy()
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    coilset.plasmagrid.update_turns('Psi')
    assert np.not_equal(coilset.plasmagrid.data['Psi'].values, Psi).all()


def test_nturn_skip_Psi_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc['Ic'] = 1
    psi_hash = coilset.aloc_hash['nturn']
    psi = coilset.plasmagrid.psi
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    coilset.plasmagrid.version['Psi'] = psi_hash  # skip update
    assert np.allclose(coilset.plasmagrid.psi, psi)


def test_nturn_current_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc['Ic'] = 1e6
    psi = coilset.plasmagrid.psi.copy()
    coilset.sloc['Ic'] = 2e6
    assert np.not_equal(coilset.plasmagrid.psi, psi).all()


def test_nturn_skip_current_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc['Ic'] = 1
    current_hash = coilset.aloc_hash['Ic']
    psi = coilset.plasmagrid.psi
    coilset.sloc['Ic'] = 2
    coilset.plasmagrid.version['psi'] = current_hash  # skip updated
    assert np.allclose(coilset.plasmagrid.psi, psi)


if __name__ == '__main__':

    pytest.main([__file__])
