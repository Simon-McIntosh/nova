import itertools

import numpy as np
import pytest

from nova.frame.coilset import CoilSet


@pytest.mark.parametrize(
    ("ngrid", "noverlap"), itertools.product([10, None], [10, None])
)
def test_number_getter(ngrid, noverlap):
    coilset = CoilSet(ngrid=ngrid, noverlap=noverlap)
    if ngrid is None or noverlap is None:
        assert coilset.overlap.number is None
        return
    assert coilset.overlap.number == (ngrid, noverlap)
    assert CoilSet(ngrid=ngrid).overlap.number is None
    assert CoilSet(noverlap=noverlap).overlap.number is None


def test_number_setter():
    ngrid, noverlap = 22, 34
    coilset = CoilSet(noverlap=noverlap)
    coilset.overlap.number = ngrid
    assert coilset.overlap.number == (ngrid, noverlap)
    coilset.overlap.number = (ngrid, 55)
    assert coilset.overlap.number == (ngrid, 55)


@pytest.mark.parametrize("segment", ["circle", "cylinder"])
def test_symetric(segment):
    coilset = CoilSet(ngrid=10, noverlap=12)
    coilset.coil.insert(4, 1, 0.5, 0.5, ifttt=False, segment=segment, Ic=1e3)
    coilset.overlap.solve()
    for attr in np.unique([attr.split("_")[0] for attr in coilset.overlap.attrs]):
        assert np.allclose(getattr(coilset.overlap, f"{attr.lower()}_abs_")[:, 1:], 0)


if __name__ == "__main__":
    pytest.main([__file__])
