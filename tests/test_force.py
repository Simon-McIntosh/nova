import pytest

import numpy as np
from nova.frame.coilset import CoilSet


def build_testset():
    coilset = CoilSet(dforce=-3, dcoil=-2, nplasma=3)
    coilset.coil.insert(5, 1, 0.1, 0.1)
    coilset.shell.insert({'e': [5, 1, 1.75, 1.0]}, 13, 0.05, delta=-9)
    coilset.coil.insert(5, 2, 0.1, 0.2)
    coilset.coil.insert(5.2, 2, 0.1, 0.2)
    coilset.firstwall.insert(5.4, 1, 0.3, 0.6, section='e')
    coilset.linkframe(['Coil2', 'Coil0'])
    coilset.force.solve()
    return coilset


testset = build_testset()


def test_turn_number():
    coilset = CoilSet(dforce=-5, dcoil=-2)
    coilset.coil.insert(5, range(3), 0.1, 0.3, nturn=[1, 2, 3])
    coilset.force.solve()
    assert np.isclose(coilset.force.target.nturn.sum(), 6)


def test_negative_delta():
    coilset = CoilSet(dforce=-9, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_positive_delta():
    coilset = CoilSet(dforce=0.1, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_zero_delta():
    coilset = CoilSet(dforce=0, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 1


def test_matrix_length():
    assert len(testset.Loc['coil', :]) == len(testset.force.Br)


if __name__ == '__main__':

    pytest.main([__file__])
