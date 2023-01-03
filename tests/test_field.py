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
    assert len(sample['radius']) == 2*1 + 2*3


def test_boundary_positive_delta_b():
    sample = Sample(frame.poly[0].boundary, 1.5)
    assert len(sample['radius']) == (2*1 + 2*2)


if __name__ == '__main__':

    pytest.main([__file__])
