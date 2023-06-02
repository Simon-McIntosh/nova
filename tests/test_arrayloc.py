import numpy as np
import pytest

from nova.frame.coilset import CoilSet


def test_aloc():
    coilset = CoilSet()
    coilset.coil.insert(3, range(4), 0.1, 0.1)
    coilset.aloc["nturn"] = 5
    assert np.allclose(coilset.aloc["nturn"], 5 * np.ones(4))
    assert isinstance(coilset.aloc["nturn"], np.ndarray)


def test_aloc_index():
    coilset = CoilSet(dplasma=-9, tplasma="rectangle")
    coilset.coil.insert(3, range(4), 0.1, 0.1)
    coilset.firstwall.insert(4, 1, 0.5, 0.5, tile=False)
    coilset.aloc["coil", "nturn"] = 1
    coilset.aloc["plasma", "nturn"] = 1 / 9
    assert coilset.aloc["coil", "nturn"].sum() == 4
    assert coilset.aloc["plasma", "nturn"].sum() == 1


def test_saloc():
    coilset = CoilSet()
    coilset.coil.insert(3, range(4), 0.1, 0.1)
    coilset.coil.insert(4, range(4), 0.1, 0.1, link=True)
    coilset.saloc["Ic"] = [5, 5, 5, 5, 4]
    assert np.allclose(coilset.aloc["Ic"], [5, 5, 5, 5, 4, 4, 4, 4])
    assert np.allclose(coilset.saloc["Ic"], [5, 5, 5, 5, 4])
    assert isinstance(coilset.saloc["Ic"], np.ndarray)


def test_asloc_index():
    coilset = CoilSet(dplasma=5)
    coilset.coil.insert(3, range(4), 0.1, 0.1, link=True)
    coilset.firstwall.insert(dict(ellipse=[1, 0, 0.5, 1.5]))
    coilset.saloc["coil", "Ic"] = 5
    coilset.saloc["plasma", "Ic"] = -4
    assert np.allclose(coilset.saloc["Ic"], [5, -4])


if __name__ == "__main__":
    pytest.main([__file__])
