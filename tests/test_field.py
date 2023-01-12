import pytest

import numpy as np
from nova.biot.field import Sample
from nova.frame.coilset import CoilSet
from nova.frame.framespace import FrameSpace


frame = FrameSpace(required=['x', 'z', 'dl', 'dt'], available=['poly'])
frame.insert(5, 0, 1, 3)


def test_sample_corners():
    sample = Sample(frame.poly[0].boundary, delta=0)
    assert len(sample['radius']) == len(sample['height']) == 4


@pytest.mark.parametrize('delta', [-1, -2, -7])
def test_sample_negative_delta_node_number(delta):
    sample = Sample(frame.poly[0].boundary, delta)
    assert np.allclose(sample['node_number'], -delta*np.ones(4))


@pytest.mark.parametrize('delta', [-1, -11])
def test_sample_negative_delta_boundary_length(delta):
    sample = Sample(frame.poly[0].boundary, delta)
    assert len(sample['radius']) == -delta*4


def test_negative_float_error():
    with pytest.raises(TypeError):
        Sample(frame.poly[0].boundary, -1.1)


def test_boundary_length_error():
    with pytest.raises(IndexError):
        Sample(np.zeros((1, 2)))


def test_boundary_ring_error():
    with pytest.raises(ValueError):
        Sample(np.array([[1, 2], [2, 2], [1, 2.1]]))


def test_boundary_positive_delta_a():
    sample = Sample(frame.poly[0].boundary, 1.0)
    assert len(sample) == 2*1 + 2*3


def test_boundary_positive_delta_b():
    sample = Sample(frame.poly[0].boundary, 1.5)
    assert len(sample) == (2*1 + 2*2)


def test_multipolygon():
    coilset = CoilSet(dfield=-2)
    coilset.coil.insert({'ring': [4.2, -0.4, 1.25, 0.5]})
    coilset.field.solve()
    assert len(coilset.field) == 0


def test_plasma():
    coilset = CoilSet(dfield=0, nplasma=50)
    coilset.firstwall.insert({'ellip': [1, 0, 0.1, 0.3]})
    coilset.coil.insert(1.1, 0, 0.05, 0.1)
    coilset.field.solve()
    assert len(coilset.field) == 4


def test_radial_field_matrix_length():
    coilset = CoilSet(dfield=0.01)
    coilset.coil.insert([1.1, 1.2, 1.3], 0, 0.075, 0.15, Ic=15e3)
    coilset.field.solve()
    assert len(coilset.field) == len(coilset.field.data.Br)
    assert len(coilset.field.bn) == len(coilset.frame)


def test_radial_field_single_coil():
    coilset = CoilSet(dfield=-3)
    coilset.coil.insert(1.3, 0, 0.075, 0.15, Ic=15e3)
    coilset.field.solve()
    coilset.point.solve(coilset.field.points)
    assert np.isclose(coilset.field.bn[0], coilset.point.bn.max())


if __name__ == '__main__':

    pytest.main([__file__])
