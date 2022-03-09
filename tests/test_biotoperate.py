import pytest

import numpy as np

from nova.electromagnetic.coilset import CoilSet


def test_nturn_hash_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    nturn_hash = coilset.subframe.version['nturn']
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    assert coilset.subframe.version['nturn'] != nturn_hash


def test_nturn_Psi_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
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
