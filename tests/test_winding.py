import tempfile

import numpy as np
import pytest

from nova.frame.coilset import CoilSet
from nova.graphics.plot import Plot


def test_rect_volume_poloidal_plane():
    coilset = CoilSet(delta=0)
    theta = np.linspace(0, 2 * np.pi, 50)
    coilset.winding.insert(
        {"rect": [0, 0, 0.3, 0.7]},
        5 * np.c_[np.cos(theta), np.zeros_like(theta), np.sin(theta)],
    )
    volume = 0.3 * 0.7 * 2 * np.pi * 5
    assert np.isclose(coilset.frame.volume.iloc[0], volume, 1e-2)
    assert np.isclose(coilset.subframe.volume.sum(), volume, 1e-2)


def test_rect_volume_toroidal_plane():
    coilset = CoilSet(dwinding=0)
    theta = np.linspace(0, 2 * np.pi, 50)
    coilset.winding.insert(
        {"rect": [0, 0, 0.3, 0.7]},
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
    )
    volume = 0.3 * 0.7 * 2 * np.pi * 5
    assert np.isclose(coilset.frame.volume.iloc[0], volume, 1e-2)
    assert np.isclose(coilset.subframe.volume.sum(), volume, 1e-2)


def test_polyplot_subframe():
    coilset = CoilSet(delta=0)
    theta = np.linspace(0, 2 * np.pi, 10)
    coilset.winding.insert(
        {"rect": [0, 0, 0.3, 0.7]},
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
    )
    with Plot().test_plot():
        coilset.subframe.polyplot()


def test_store_load():
    coilset = CoilSet(dwinding=0)
    theta = np.linspace(0, 2 * np.pi, 20)
    coilset.winding.insert(
        {"rect": [0, 0, 0.3, 0.7]},
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
    )
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.filepath = tmp.name
        coilset.store()
        new_coilset = CoilSet()
        new_coilset.filepath = tmp.name
        new_coilset.load()
        coilset._clear()


if __name__ == "__main__":
    pytest.main([__file__])
