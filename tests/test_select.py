import pytest

import numpy as np

from nova.frame.framespace import FrameSpace


def test_plasma_subspace():
    framespace = FrameSpace(Required=['x'], Available=['active', 'link'],
                            Array=['plasma'])
    framespace.insert(range(2), active=True, link=True)
    framespace.insert(range(2), active=False, link=True)
    framespace.insert(3, plasma=True)
    assert list(framespace.subspace.plasma) == [False, False, True]


def test_coil():
    framespace = FrameSpace(Required=['x'], Available=['active'],
                            Array=['coil'])
    framespace.insert(range(3), plasma=[True, False, True])
    assert list(framespace.coil) == [False, True, False]


def test_fix():
    framespace = FrameSpace(Required=['x'])
    framespace.insert(range(3), fix=[True, False, True])
    assert list(framespace.free) == [False, True, False]


def test_ferritic():
    framespace = FrameSpace(Required=['x'])
    framespace.insert(range(3), ferritic=[True, False, True])
    assert list(framespace.ferritic) == [True, False, True]


def test_select_subspace():
    framespace = FrameSpace(Required=['x'], Available=['plasma'])
    assert np.array([attr in framespace.metaframe.subspace for attr in
                     framespace.select.avalible
                     if attr not in framespace.select.superspace]).all()


def test_generate_false():
    framespace = FrameSpace(Required=['x'])
    assert not framespace.hasattrs('select')


def test_generate_false_attrs():
    framespace = FrameSpace(Required=['x'], Additional=[])
    assert framespace.columns.to_list() == ['x']


if __name__ == '__main__':

    pytest.main([__file__])
