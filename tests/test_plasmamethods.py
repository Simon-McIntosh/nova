import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet


def test_centroid_x(plot=False):
    cs = CoilSet()
    cs.add_plasma([[1, 2, 2, 1, 1], [1, 1, 3, 3, 1]])
    if plot:
        cs.plot()
    plasma_polygon = cs.coil.polygon[0]
    assert plasma_polygon.centroid.x == cs.coil.x[0] == 1.5


def test_centroid_z(plot=False):
    cs = CoilSet()
    cs.add_plasma([[1, 2, 1.5, 1], [0, 0, 3, 0]])
    if plot:
        cs.plot()
    plasma_polygon = cs.coil.polygon[0]
    assert plasma_polygon.centroid.y == 1 == cs.coil.z[0]


def test_circle(plot=False):
    cs = CoilSet()
    polygon = shapely.geometry.Point(1, 1).buffer(0.5)
    cs.add_plasma(polygon)
    if plot:
        cs.plot(True)
    assert np.isclose(cs.subcoil.dA[cs.subcoil.plasma].sum(),
                      np.pi*0.5**2, 5e-3)


def test_polygon_separatrix(plot=False):
    cs = CoilSet(dPlasma=0.25)
    cs.add_plasma([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    cs.separatrix = shapely.geometry.Point(3, 3).buffer(2)
    if plot:
        cs.plot(True)
    assert np.isclose(cs.subcoil.dA[cs.subcoil.plasma][cs.ionize_index].sum(),
                      np.pi*2**2, 0.05)


def test_array_separatrix(plot=False):
    cs = CoilSet(dPlasma=0.05)
    cs.add_plasma([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    cs.separatrix = np.array([[1, 2, 1.5, 1], [0, 0, 2, 0]]).T
    if plot:
        cs.plot()
        cs.plot(True, feedback=True)
    assert np.isclose(cs.subcoil.dA[cs.subcoil.plasma][cs.ionize_index].sum(),
                      0.5**2, 1e-3)


if __name__ == '__main__':
    #pytest.main([__file__])
    test_array_separatrix(True)
