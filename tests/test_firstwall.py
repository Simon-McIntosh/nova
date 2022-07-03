import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.error import GridError, PlasmaGridError


def test_centroid_x():
    coilset = CoilSet()
    coilset.firstwall.insert([[1, 2, 2, 1, 1], [1, 1, 3, 3, 1]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.x == coilset.frame.x[0] == 1.5


def test_centroid_z():
    coilset = CoilSet()
    coilset.firstwall.insert([[1, 2, 1.5, 1], [0, 0, 3, 0]])
    poly = coilset.frame.poly[0]
    assert poly.centroid.y == 1 == coilset.frame.z[0]


def test_circle():
    coilset = CoilSet()
    polygon = shapely.geometry.Point(1, 1).buffer(0.5)
    coilset.firstwall.insert(polygon)
    assert np.isclose(coilset.subframe.area.sum(), np.pi*0.5**2, 5e-3)


def test_plasma_turns():
    coilset = CoilSet(dplasma=0.25)
    coilset.firstwall.insert({'ellipse': [1.7, 1, 0.5, 0.85]})
    assert coilset.frame.nturn[0] == 1
    assert np.isclose(coilset.subframe.nturn.sum(), 1)


def test_plasma_part():
    coilset = CoilSet(dplasma=0)
    coilset.firstwall.insert({'o': [1.7, 1, 0.5]})
    assert coilset.frame.part[0] == 'plasma'


def test_multiframe_area():
    coilset = CoilSet(dplasma=0.5)
    coilset.firstwall.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.firstwall.insert(dict(o=(5, 0, 1.6)), name='PLcore')
    assert np.isclose(coilset.loc['plasma', 'area'].sum(), 1/4*np.pi*2**2,
                      atol=1e-2)


def test_multiframe_nturn():
    coilset = CoilSet(dplasma=0.5)
    coilset.firstwall.insert(dict(sk=(5, 0, 2, 0.2)), name='PLedge', delta=0.2)
    coilset.coil.insert(6.5, 0, 0.2, 0.8)
    coilset.firstwall.insert(dict(o=(5, 0, 1.6)), name='PLcore')
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


def test_nohex_plasma_grid_error():
    coilset = CoilSet(dcoil=-5, dplasma=-10, tplasma='rect')
    coilset.firstwall.insert({'e': [6.5, 0, 1.2, 1.6]}, Ic=-15e6)
    with pytest.raises(PlasmaGridError):
        coilset.plasmagrid.solve()


if __name__ == '__main__':
    pytest.main([__file__])
