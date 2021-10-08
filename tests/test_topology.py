import pytest
import numpy as np
import shapely.geometry

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


def test_spline_update(plot=False):
    update = np.zeros(5, dtype=bool)

    cs = CoilSet(required=['x', 'z', 'dx', 'dz'], dplasma=0.25, dcoil=0.5)
    cs.plasma.insert(5.5, 0.5, 1, 1, Ic=15e6)
    cs.coil.insert(5, -0.5, 0.75, 0.75)
    print(cs.sloc['Plasma', 'Ic'])
    cs.plot()
    '''
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
    '''
    assert np.equal(update, [True, False, True, False, True]).all()
#test_spline_update()


def null_curvature(sign, plot):
    cs = CoilSet()
    polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.5)
    cs.plasmagrid.generate_grid(expand=0.25, n=1e2)  # generate plasma grid
    cs.Ic = sign * 15e6
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux()
        cs.plasmagrid.global_null(True)
    return cs


def test_Opoint_curvature_Ip_positive(plot=False):
    cs = null_curvature(1, plot)
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint[0]) == 'O'


def test_Opoint_curvature_Ip_negative(plot=False):
    cs = null_curvature(-1, plot)
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint[0]) == 'O'


def global_null(sign, plot=False):
    cs = CoilSet()
    cs.add_coil(5, [-2, 2], 0.75, 0.75, dCoil=0.5)
    cs.add_coil(7.8, 0, 0.75, 0.75, label='Xcoil', dCoil=0.5)
    polygon = shapely.geometry.Point(4, 0).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.3, expand=5, n=3e2,
                  boundary='limit', limit=[3.2, 8.5, -2.5, 2.5])
    cs.plasmagrid.optimizer = 'newton'
    cs.plasmagrid.filter_sigma = 0  # disable interpolant filter
    cs.plasmagrid.cluster = True
    cs.Ic = sign*15e6
    cs.plasmagrid.global_null(plot)
    if plot:
        cs.plot(True)
        cs.plasmagrid.plot_flux(levels=51)
        plt.contour(cs.plasmagrid.x2d, cs.plasmagrid.z2d,
                    cs.plasmagrid.interpolate('Psi').ev(cs.plasmagrid.x2d,
                                                        cs.plasmagrid.z2d),
                    levels=cs.plasmagrid.levels, colors='C3')
    return cs


def test_global_null_Ip_positive(plot=False):
    cs = global_null(1, plot)
    assert cs.plasmagrid.nX == 3 and cs.plasmagrid.nO == 4


def test_global_null_Ip_negative(plot=False):
    cs = global_null(-1, plot)
    assert cs.plasmagrid.nX == 3 and cs.plasmagrid.nO == 4


def test_ftol_rel_attribute():
    cs = CoilSet()
    cs.add_plasma([4, 5, -1, 1], dPlasma=1)
    cs.plasmagrid.ftol_rel = 1e-4
    assert np.isclose(cs.plasmagrid._get_opt('field').get_ftol_rel(), 1e-4)


def test_xtol_rel_attribute():
    cs = CoilSet()
    cs.add_plasma([4, 5, -1, 1], dPlasma=1)
    cs.plasmagrid.xtol_rel = 1e-2
    assert np.isclose(cs.plasmagrid._get_opt('field').get_xtol_rel(), 1e-2)


if __name__ == '__main__':
    #pytest.main([__file__])
    test_spline_update()

    # test_xtol_rel_attribute()
    #cs = global_null(1, plot=True)
