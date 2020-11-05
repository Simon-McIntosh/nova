import pytest
import numpy as np
import shapely.geometry

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


def test_spline_update(plot=False):
    update = np.zeros(5, dtype=bool)

    cs = CoilSet()
    cs.add_plasma([5, 6, 0, 1], dPlasma=0.25)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.5)
    update[0] = cs.plasmagrid._update_B_spline  # True
    cs.solve_biot()
    update[1] = cs.plasmagrid._update_B_spline  # False
    cs.Ip = 15e6
    update[2] = cs.plasmagrid._update_B_spline  # True
    cs.plasmagrid.interpolate('B')
    update[3] = cs.plasmagrid._update_B_spline  # False
    update[4] = cs.plasmagrid._update_Psi_spline  # True
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot()
    assert np.equal(update, [True, False, True, False, True]).all()


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
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint) == 'O'


def test_Opoint_curvature_Ip_positive(plot=False):
    Opoint_curvature(1, plot)


def test_Opoint_curvature_Ip_negative(plot=False):
    Opoint_curvature(-1, plot)


def test_Xpoint_curvature_Ip_positive(plot=False):
    cs = CoilSet()
    polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)
    cs.plasmagrid.generate_grid(expand=2.9, n=2e2)  # generate plasma grid
    cs.Ic = [15e6, 15e6]

    plt.figure()
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        plt.plot(*cs.plasmagrid.Opoint, 'ko')
        plt.plot(*cs.plasmagrid.Xpoint, 'kX')
        plt.plot(*cs.plasmagrid._Xpoint.T, 'kX')
    assert cs.plasmagrid.null_type(cs.plasmagrid.Xpoint) == 'X'

if __name__ == '__main__':
    #pytest.main([__file__])
    test_Xpoint_curvature_Ip_positive(True)
    #test_spline_update(True)