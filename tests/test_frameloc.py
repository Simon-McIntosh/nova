import pytest

import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.dataframe import SubSpaceColumnError, ColumnError


def test_get_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    with pytest.raises(SubSpaceColumnError):
        _ = coilset.sloc['Ic']


def test_set_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    with pytest.raises(SubSpaceColumnError):
        coilset.sloc['Ic'] = [8.8, 8.8]


def test_get_key_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    with pytest.raises(KeyError):
        _ = coilset.loc['turn']


def test_set_column_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    with pytest.raises(ColumnError):
        coilset.loc['turn'] = [8.8, 8.8]


def test_get_current_frame():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    assert np.isclose(coilset.loc['Ic'], [7.7, 7.7]).all()


def test_get_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert([1, 3, 7], Ic=[7.7, 8.3, 6.6])
    assert np.isclose(coilset.sloc['Ic'], [7.7, 8.3, 6.6]).all()


def test_get_current_subspace_array():
    coilset = CoilSet(dcoil=-1, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert([1, 3], Ic=[7.7, 6.6])
    assert np.isclose(coilset.sloc['Ic'], [7.7, 6.6]).all()


def test_set_current_frame():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'],
                      additional=['Ic'])
    coilset.coil.insert(1.5)
    coilset.loc['active', 'Ic'] = [8.8, 7.7]
    assert np.isclose(coilset.loc['Ic'], [8.8, 7.7]).all()


def test_set_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert(1.5)
    coilset.sloc['Ic'] = [8.8]
    assert np.isclose(coilset.loc['Ic'], [8.8]).all()


def test_set_current_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert(1.5)
    with pytest.raises(ValueError):
        coilset.sloc['Ic'] = [8.8, 8.8]


def test_set_current_subspace_array():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(1.5)
    coilset.sloc['Ic'] = [8.8]
    assert np.isclose(coilset.sloc['Ic'], [8.8]).all()


def test_get_current_subset():
    coilset = CoilSet(dcoil=-1, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(3, plasma=False)
    coilset.coil.insert(6.6, plasma=True)
    coilset.coil.insert([1.2, 2.2], active=False)
    coilset.link(['Coil0', 'Coil3'])
    coilset.sloc['active', 'Ic'] = [8.8, 7.7]
    assert np.isclose(coilset.loc['Ic'], [8.8, 7.7, 0, 8.8]).all()


def test_get_current_insert_default():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(3)
    #coilset.plasma.insert({'s': [3.25, 0, 0.25]}, delta=-2)
    coilset.shell.insert([2.2, 3.2], [-0.1, 0.3], -2, 0.05, delta=-3)
    coilset.sloc['Ic'] = 4.4
    #coilset.sloc['plasma', 'Ic'] = [9.9]
    coilset.sloc['active', 'Ic'] = [3.3]
    coilset.sloc['coil', 'Ic'] = [5.5]
    #print(coilset.subframe)
    assert np.isclose(coilset.sloc['Ic'], [5.5, 3.3, 4.4, 4.4]).all()


if __name__ == '__main__':

    pytest.main([__file__])

