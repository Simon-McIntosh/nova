import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.geometry.polygon import Polygon


def test_polygon_separatrix_loop():
    coilset = CoilSet(dplasma=-35)
    coilset.firstwall.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.separatrix = Polygon(dict(c=(3, 3, 4))).points[:, ::2]
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)


def test_polygon_separatrix_polygon():
    coilset = CoilSet(dplasma=-35)
    coilset.firstwall.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.separatrix = dict(c=(3, 3, 4))
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)


def test_array_separatrix():
    coilset = CoilSet(dplasma=0.1)
    coilset.firstwall.insert([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    coilset.plasma.separatrix = np.array([[1, 2, 1.5, 1], [0, 0, 2, 0]]).T
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), 0.5**2, 0.1)


def test_separatrix_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.firstwall.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.separatrix = shapely.geometry.Point(3, 3).buffer(2)
    assert np.isclose(coilset.loc['plasma', 'nturn'].sum(), 1)


def test_polarity():
    coilset = CoilSet(dplasma=-10, dcoil=-10)
    coilset.coil.insert(4.65, [-0.3, 0.3], 0.1, 0.5)
    coilset.firstwall.insert({'ellip': [5, 0, 0.5, 0.75]}, It=-15e6)
    coilset.plasma.separatrix = {'disc': [5, 0, 0.3]}
    assert coilset.plasma.polarity == -1


def test_coil_xpoint():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(6.5, [-1, 0, 1], 0.4, 0.4, Ic=-15e6)
    coilset.grid.solve(100, [6, 7.0, -0.8, 0.8])
    assert coilset.grid.x_point_number == 2
    assert coilset.grid.o_point_number == 1


def test_coil_cylinder_xpoint():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.5, [-1, 0, 1], 0.4, 0.4, Ic=-15e6,
                        segment='cylinder')
    coilset.grid.solve(60, [6, 7.0, -0.8, 0.8])
    assert coilset.grid.x_point_number == 2
    assert coilset.grid.o_point_number == 1


def test_grid_xpoint_coil():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(6.5, [-1, 1], 0.4, 0.1, Ic=-15e6)
    coilset.coil.insert(dict(e=[6.5, 0, 0.4, 0.6]), Ic=-15e6,
                        turn='h', delta=-10)
    coilset.grid.solve(60, [6, 6.8, -0.95, 0.95])
    assert coilset.grid.x_point_number == 2
    assert coilset.grid.o_point_number == 1


def test_grid_xpoint():
    coilset = CoilSet(dcoil=-5, dplasma=-75)
    coilset.coil.insert(6.5, [-1, 1], 0.4, 0.1, Ic=-15e6)
    coilset.firstwall.insert({'e': [6.5, 0, 1.2, 1.6]}, Ic=-15e6)
    coilset.plasma.separatrix = {'e': [6.5, 0, 0.4, 0.6]}
    coilset.grid.solve(60, [6, 6.8, -0.95, 0.95])
    assert coilset.grid.x_point_number == 2
    assert coilset.grid.o_point_number == 1


def test_plasmagrid_xpoint():
    coilset = CoilSet(dcoil=-5, dplasma=-80)
    coilset.coil.insert(6.5, [-1, 1], 0.4, 0.2, Ic=-15e6)
    coilset.firstwall.insert({'e': [6.5, 0, 1.2, 1.6]}, Ic=-15e6)
    coilset.plasma.separatrix = {'e': [6.5, 0, 0.4, 0.6]}
    coilset.plasmagrid.solve()
    assert coilset.plasmagrid.x_point_number == 2
    assert coilset.plasmagrid.o_point_number == 1


if __name__ == '__main__':
    pytest.main([__file__])
