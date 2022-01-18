import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.error import GridError
from nova.geometry.polygon import Polygon


def test_centroid_x():
    coilset = CoilSet()
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 3, 3, 1]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.x == coilset.frame.x[0] == 1.5


def test_centroid_z():
    coilset = CoilSet()
    coilset.plasma.insert([[1, 2, 1.5, 1], [0, 0, 3, 0]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.y == 1 == coilset.frame.z[0]


def test_circle():
    coilset = CoilSet()
    polygon = shapely.geometry.Point(1, 1).buffer(0.5)
    coilset.plasma.insert(polygon)
    assert np.isclose(coilset.subframe.area.sum(), np.pi*0.5**2, 5e-3)


def test_plasma_turns():
    coilset = CoilSet(dplasma=0.25)
    coilset.plasma.insert({'ellipse': [1.7, 1, 0.5, 0.85]})
    assert coilset.frame.nturn[0] == 1
    assert np.isclose(coilset.subframe.nturn.sum(), 1)


def test_plasma_part():
    coilset = CoilSet(dplasma=0)
    coilset.plasma.insert({'o': [1.7, 1, 0.5]})
    assert coilset.frame.part[0] == 'plasma'


def test_polygon_separatrix_loop():
    coilset = CoilSet(dplasma=-35)
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    loop = Polygon(dict(c=(3, 3, 4))).points[:, ::2]
    coilset.plasma.update_separatrix(loop)
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)


def test_polygon_separatrix_polygon():
    coilset = CoilSet(dplasma=-35)
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.update_separatrix(dict(c=(3, 3, 4)))
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)


def test_array_separatrix():
    coilset = CoilSet(dplasma=0.1)
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    coilset.plasma.update_separatrix(np.array([[1, 2, 1.5, 1],
                                               [0, 0, 2, 0]]).T)
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), 0.5**2, 0.1)


def test_separatrix_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.update_separatrix(shapely.geometry.Point(3, 3).buffer(2))
    assert np.isclose(coilset.loc['plasma', 'nturn'].sum(), 1)


def test_polarity():
    coilset = CoilSet(dplasma=-10, dcoil=-10)
    coilset.coil.insert(4.65, [-0.3, 0.3], 0.1, 0.5)
    coilset.plasma.insert({'ellip': [5, 0, 0.5, 0.75]}, It=-15e6)
    coilset.plasma.update_separatrix({'disc': [5, 0, 0.3]})
    assert coilset.plasma.polarity == -1


def test_multiframe_area():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.plasma.insert(dict(o=(5, 0, 1.6)), name='PLcore')
    assert np.isclose(coilset.loc['plasma', 'area'].sum(), 1/4*np.pi*2**2,
                      atol=1e-3)


def test_multiframe_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.plasma.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.coil.insert(6.5, 0, 0.2, 0.8)
    coilset.plasma.insert(dict(o=(5, 0, 1.6)), name='PLcore')
    assert np.isclose(coilset.loc['plasma', 'nturn'].sum(), 1)


def test_grid_no_plasma():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(6.5, 0, 0.2, 0.8)
    with pytest.raises(GridError):
        coilset.grid.solve(100,  index='plasma')


def test_plasmagrid_no_plasma():
    coilset = CoilSet(dcoil=-5)
    coilset.coil.insert(6.5, 0, 0.2, 0.8)
    with pytest.raises(GridError):
        coilset.plasmagrid.solve()


test_plasmagrid_no_plasma()

if __name__ == '__main__':

    pytest.main([__file__])
