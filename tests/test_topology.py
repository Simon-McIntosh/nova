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
    cs.Ic = sign * 15e6
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        plt.plot(*cs.plasmagrid.Opoint.T, 'ko')
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint[0]) == 'O'


def test_Opoint_curvature_Ip_positive(plot=False):
    Opoint_curvature(1, plot)


def test_Opoint_curvature_Ip_negative(plot=False):
    Opoint_curvature(-1, plot)


def global_null(sign, plot):
    cs = CoilSet()
    polygon = shapely.geometry.Point(5, 0).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, [-2, 2], 0.75, 0.75, dCoil=0.2)
    cs.add_coil(7.8, 0, 0.75, 0.75, label='Xcoil', dCoil=0.2)
    cs.plasmagrid.generate_grid(expand=5, plasma_n=1e3)  # generate plasma grid
    cs.Ic = sign*15e6
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        cs.plasmagrid._global_null(True)
    assert cs.plasmagrid.nX == 3 and cs.plasmagrid.nO == 4


def test_global_null_positive(plot=False):
    global_null(1, plot)

def test_global_null_negative(plot=False):
    global_null(-1, plot)

if __name__ == '__main__':
    pytest.main([__file__])

    test_global_null_positive(True)
    #test_global_null_negative(True)
    #test_spline_update(True)