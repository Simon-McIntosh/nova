
import numpy as np
import pytest

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


if __name__ == '__main__':

    pytest.main([__file__])
