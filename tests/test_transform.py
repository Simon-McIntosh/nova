
import numpy as np
import pytest
import xarray

from nova.assembly.transform import Rotate


def test_clock_unwind():
    assert np.allclose(Rotate().anticlock(Rotate().clock([1, 2, 3])),
                       [1, 2, 3])


def test_anticlock_unwind():
    assert np.allclose(Rotate().clock(Rotate().anticlock([1, 2, 3])),
                       [1, 2, 3])


def test_double_clock():
    rotate_9 = Rotate(ncoil=9)
    rotate_18 = Rotate(ncoil=18)
    assert np.allclose(
        rotate_9.anticlock(rotate_18.clock(
            rotate_18.clock([1, 2, 3]))), [1, 2, 3])


def test_carteasian_cylindrical_carteasian():
    rng = np.random.default_rng(2025)
    dataarray = xarray.DataArray(rng.random((12, 3)),
                                 dims=('point', 'space'),
                                 coords=dict(space=list('xyz')))
    rotate = Rotate(ncoil=18)
    cylindrical = rotate.to_cylindrical(dataarray)
    carteasian = rotate.to_cartesian(cylindrical)
    assert np.allclose(carteasian.values, dataarray.values)
    assert not np.allclose(cylindrical.values, carteasian.values)


if __name__ == '__main__':

    pytest.main([__file__])
