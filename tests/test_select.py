import pytest

import numpy as np

from nova.electromagnetic.frame import Frame


def test_plasma_subspace():
    frame = Frame(Required=['x'], Available=['active', 'link'],
                  Array=['plasma'])
    frame.insert(range(2), active=True, link=True)
    frame.insert(range(2), active=False, link=True)
    frame.insert(3, plasma=True)
    assert list(frame.subspace.plasma) == [False, False, True]


def test_coil():
    frame = Frame(Required=['x'], Available=['active'], Array=['coil'])
    frame.insert(range(2), plasma=[True, False])
    assert list(frame.coil) == [False, True]


def test_select_subspace():
    frame = Frame(Required=['x'], Available=['plasma'])
    assert np.array([attr in frame.metaframe.subspace for attr in
                     frame.select.avalible]).all()


def test_generate_false():
    frame = Frame(Required=['x'])
    assert frame.select.generate is not True


def test_generate_false_attrs():
    frame = Frame(Required=['x'], Additional=[])
    assert frame.columns.to_list() == ['x']


if __name__ == '__main__':

    pytest.main([__file__])
