import numpy as np
import pytest

from nova.frame.coilset import CoilSet


def test_rect_volume():
    coilset = CoilSet(delta=-100, available=["vtk"])
    theta = np.linspace(0, 2 * np.pi, 100)
    coilset.winding.insert(
        {"rect": [0, 0, 0.3, 0.7]},
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
    )
    volume = 0.3 * 0.7 * 2 * np.pi * 5
    assert np.isclose(coilset.frame.volume[0], volume, 1e-3)
    assert np.isclose(coilset.subframe.volume.sum(), volume, 1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
