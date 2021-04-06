import pytest

import numpy as np

from nova.electromagnetic.coilgrid import CoilGrid


def test_rectangle():
    coilgrid = CoilGrid({'r': [6, 3.3, 12, 0.5]})
    assert coilgrid.polycoil.poly.area == 6


def test_square():
    coilgrid = CoilGrid({'sq': [6, 3.3, 3]})
    assert coilgrid.polycoil.poly.area == 9


def test_limit():
    coilgrid = CoilGrid([0, 5, 2, 3])
    assert coilgrid.polycoil.poly.area == 5


def test_loop():
    coilgrid = CoilGrid([[0, 2, 1, 0], [0, 0, 1, 0]])
    assert coilgrid.polycoil.poly.area == 1


def test_loop_transpose():
    coilgrid = CoilGrid(np.array([[0, 2, 1, 0], [0, 0, 1, 0]]).T)
    assert coilgrid.polycoil.poly.area == 1


def test_multipolygon():
    coilgrid = CoilGrid({'c1': [6, 3, 12, 0.2], 'c2': [6.1999, 3, 0.3, 0.2]})
    assert np.isclose(coilgrid.polycoil.poly.area, 2 * np.pi * 0.2**2 / 4,
                      1e-3)


if __name__ == '__main__':

    pytest.main([__file__])
