import pytest

import numpy as np

from nova.geometry.polygon import Polygon
from nova.electromagnetic.polygrid import PolyGrid


def test_rectangle():
    polygrid = PolyGrid({'r': [6, 3.3, 12, 0.5]})
    assert polygrid.area == 6


def test_square():
    polygrid = PolyGrid({'sq': [6, 3.3, 3]})
    assert polygrid.area == 9


def test_limit():
    polygrid = PolyGrid([0, 5, 2, 3])
    assert polygrid.area == 5


def test_loop():
    polygrid = PolyGrid([[0, 2, 1, 0], [0, 0, 1, 0]])
    assert polygrid.area == 1


def test_loop_transpose():
    polygrid = PolyGrid(np.array([[0, 2, 1, 0], [0, 0, 1, 0]]).T)
    assert polygrid.area == 1


def test_multipolygon():
    polygrid = PolyGrid({'o1': [6, 3, 12, 0.2], 'o2': [6.1999, 3, 0.3, 0.2]})
    assert np.isclose(polygrid.area, 2 * np.pi * 0.2**2 / 4, 1e-3)


def test_square_in_rectangle():
    polygrid = PolyGrid({'r': [3, 5, 0.2, 0.1]}, delta=0, turn='sq')
    assert np.isclose(polygrid.polyarea, 0.1**2)


def test_disc_in_rectangle():
    polygrid = PolyGrid({'r': [3, 5, 0.1, 0.2]}, delta=0, turn='o')
    assert np.isclose(polygrid.polyarea, np.pi * 0.1**2 / 4, 1e-3)


def test_rectangle_in_rectangle():
    polygrid = PolyGrid({'r': [3, 5, 0.1, 0.15]}, delta=0, turn='r')
    assert np.isclose(polygrid.polyarea, 0.1*0.15)


def test_hexagon_in_rectangle_horizontal_constraint():
    polygrid = PolyGrid({'r': [3, 5, 0.1, 0.15]}, delta=0, turn='hex')
    edge_length = 0.1/2
    assert np.isclose(polygrid.polyarea, 3**1.5 / 2 * edge_length**2)


def test_hexagon_in_rectangle_vertical_constraint():
    polygrid = PolyGrid({'r': [3, 5, 0.1, 0.05]}, delta=0, turn='hex')
    edge_length = 0.05/np.sqrt(3)
    assert np.isclose(polygrid.polyarea, 3**1.5 / 2 * edge_length**2)


def test_hexagon_in_disc():
    polygrid = PolyGrid({'o': [3, 5, 0.1]}, delta=0, turn='hex')
    edge_length = 0.1/2
    assert np.isclose(polygrid.polyarea, 3**1.5 / 2 * edge_length**2)


def test_rectangle_grid_horizontal():
    polygrid = PolyGrid({'r': [6, 3, 0.4, 0.2]}, delta=-5, turn='o')
    assert len(polygrid) == 6


def test_rectangle_grid_vertical():
    polygrid = PolyGrid({'r': [6, 3, 0.1, 0.2]}, delta=-7, turn='o')
    assert len(polygrid) == 8


def test_disc_disc_grid_odd_under():
    polygrid = PolyGrid({'o': [6, 3, 0.4, 0.2]}, delta=-3, turn='o')
    assert len(polygrid) == 4


def test_disc_disc_grid_odd_over():
    polygrid = PolyGrid({'o': [6, 3, 0.4, 0.2]}, delta=-5, turn='o')
    assert len(polygrid) == 9


def test_disc_square_grid_odd_over():
    polygrid = PolyGrid({'o': [6, 3, 0.4, 0.2]}, delta=-5, turn='sq')
    assert len(polygrid) == 9


def test_disc_rectangle_grid_odd_over():
    polygrid = PolyGrid({'o': [6, 3, 0.4, 0.2]}, delta=-5, turn='r')
    assert len(polygrid) == 6


def test_poly_buffer():
    polygrid = PolyGrid({'o': [6, 3, 0.4, 0.2]}, delta=-5,
                        turn='sk', scale=1)
    assert np.sum([section == 'skin'
                   for section in polygrid.frame.section]) == 5


def test_square_effective_nfilament():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='sq', tile=False)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments)


def test_square_effective_nfilament_tile():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='sq', tile=True)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments)


def test_rectangle_effective_nfilament():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='r', tile=False)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments)


def test_rectangle_effective_nfilament_tile():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='r', tile=True)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments)


def test_disc_effective_nfilament():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='o', tile=False)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments, 1e-3)


def test_disc_effective_nfilament_tile():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-49,
                        turn='o', tile=True)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments, 1e-2)


def test_hexagon_effective_nfilament():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-25,
                        turn='hx', tile=False, fill=True)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments, 1e-3)


def test_hexagon_effective_nfilament_tile():
    polygrid = PolyGrid({'r': [6, 3, 2.5, 2.5]}, delta=-49,
                        turn='hex', tile=True)
    assert np.isclose(-polygrid.delta, polygrid.polyfilaments, 1e-2)


if __name__ == '__main__':

    pytest.main([__file__])
