import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet


def test_centroid_x():
    coilset = CoilSet()
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 3, 3, 1]])
    coilset.plot
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


def test_polygon_separatrix():
    coilset = CoilSet(dplasma=0.5, additional=['ionize'])
    coilset.plasma.insert([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    coilset.plasma.separatrix = shapely.geometry.Point(3, 3).buffer(2)

    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), np.pi*2**2, 0.05)


def test_array_separatrix():
    coilset = CoilSet(dplasma=0.1, additional=['ionize'])
    coilset.plasma.insert([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    coilset.plasma.separatrix = np.array([[1, 2, 1.5, 1], [0, 0, 2, 0]]).T
    assert np.isclose(
        coilset.subframe.area[coilset.subframe.ionize].sum(), 0.5**2, 0.1)


if __name__ == '__main__':

    pytest.main([__file__])
