import os
from pathlib import Path
import pytest
import tempfile

import numpy as np
from nova.frame.coilset import CoilSet


@pytest.fixture
def linked():
    coilset = CoilSet(nforce=10, dcoil=-1, nplasma=3)
    coilset.coil.insert(5, 1, 0.1, 0.1, nturn=1)
    coilset.shell.insert({'e': [5, 1, 1.75, 1.0]}, 13, 0.05, delta=-9)
    coilset.shell.insert({'e': [5, 1, 1.95, 1.2]}, 13, 0.05, delta=-9)
    coilset.coil.insert(5, 2, 0.1, 0.2, nturn=1.3)
    coilset.coil.insert(5.2, 2, 0.1, 0.2, nturn=1.25)
    coilset.firstwall.insert(5.4, 1, 0.3, 0.6, section='e', Ic=-15e6)
    coilset.linkframe(['Coil2', 'Coil0'])
    coilset.sloc['coil', 'Ic'] = -15e6
    coilset.force.solve()
    return coilset


def test_turn_number():
    coilset = CoilSet(nforce=5, dcoil=-2)
    coilset.coil.insert(5, range(3), 0.1, 0.3, nturn=[1, 2, 3])
    coilset.force.solve()
    assert np.isclose(coilset.force.target.nturn.sum(), 6)


def test_negative_delta():
    coilset = CoilSet(nforce=9, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_positive_delta():
    coilset = CoilSet(nforce=-0.1, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_unit_delta():
    coilset = CoilSet(nforce=1, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 1


def test_matrix_attrs(linked):
    for attr in ['Fr', 'Fz', 'Fc']:
        assert attr in linked.force.data


def test_matrix_length(linked):
    assert len(linked.Loc['coil', :]) == len(linked.force.Fr)


def test_store_load(linked):
    fr = linked.force.fr
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        linked.filepath = tmp.name
        linked.store()
        del linked
        path = Path(tmp.name)
        coilset = CoilSet(filename=path.name, dirname=path.parent).load()
        coilset._clear()
    os.unlink(tmp.name)
    assert np.allclose(fr, coilset.force.fr)


def test_resolution():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(5, [5, 6], 0.9, 0.1, Ic=45e3, nturn=500)
    coilset.force.solve(100)
    fr_lowres = coilset.force.fr
    coilset.force.solve(200)
    fr_highres = coilset.force.fr
    assert np.allclose(fr_lowres, fr_highres, rtol=1e-3)


if __name__ == '__main__':

    pytest.main([__file__])
