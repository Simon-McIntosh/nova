import pytest
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


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
    assert plasma_polygon.centroid.y == 1 == cs.coil.z


def test_circle(plot=False):
    cs = CoilSet()
    polygon = shapely.geometry.Point(1, 1).buffer(0.5)
    cs.add_plasma(polygon)
    if plot:
        cs.plot(True)
    assert np.isclose(cs.subcoil.dA.sum(), np.pi*0.5**2, 5e-3)


def test_polygon_separatrix(plot=False):
    cs = CoilSet(dPlasma=0.25)
    cs.add_plasma([[1, 5, 5, 1, 1], [1, 1, 5, 5, 1]])
    cs.separatrix = shapely.geometry.Point(3, 3).buffer(2)
    if plot:
        cs.plot(True)
    assert np.isclose(cs.subcoil.dA[cs.ionize_index].sum(),
                      np.pi*2**2, 0.05)


def test_array_separatrix(plot=False):
    cs = CoilSet(dPlasma=0.05)
    cs.add_plasma([[1, 2, 2, 1, 1], [1, 1, 2, 2, 1]])
    cs.separatrix = np.array([[1, 2, 1.5, 1], [0, 0, 2, 0]]).T
    if plot:
        cs.plot()
        cs.plot(True)
    assert np.isclose(cs.subcoil.dA[cs.ionize_index].sum(),
                      0.5**2, 1e-3)


def Opoint_curvature(sign, plot):
    cs = CoilSet()
    polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)
    cs.plasmagrid.generate_grid(expand=0.2, n=2e2)  # generate plasma grid
    cs.Ic = sign * 15e6 * np.ones(2)
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        plt.plot(*cs.plasmagrid.Opoint, 'ko')
    assert cs.plasmagrid.field_null(cs.plasmagrid.Opoint) == 'O'


def test_Opoint_curvature_Ip_positive(plot=False):
    Opoint_curvature(1, plot)


def test_Opoint_curvature_Ip_negative(plot=False):
    Opoint_curvature(-1, plot)


def test_Xpoint_curvature_Ip_positive(plot=False):
    cs = CoilSet()
    polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)
    cs.plasmagrid.generate_grid(expand=1, n=2e4)  # generate plasma grid
    cs.Ic = [15e6, 15e6]

    plt.figure()
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        plt.plot(*cs.plasmagrid.Opoint, 'ko')
        plt.plot(*cs.plasmagrid.Xpoint, 'kX')
        plt.plot(*cs.plasmagrid._Xpoint.T, 'kX')
    assert cs.plasmagrid.field_null(cs.plasmagrid.Xpoint) == 'X'


if __name__ == '__main__':
    #pytest.main([__file__])
    test_Xpoint_curvature_Ip_positive(True)
