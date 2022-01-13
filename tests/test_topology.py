import pytest
import numpy as np
import shapely.geometry

from nova.electromagnetic.coilset import CoilSet
from nova.geometry.polygon import Polygon


def test_plasma_update(plot=False):
    update = np.full(4, False)
    coilset = CoilSet(required=['x', 'z', 'dx', 'dz'],
                      dplasma=-5, dcoil=0.5)
    coilset.plasma.insert(dict(disc=[4.5, 0.5, 1]), Ic=15e6)
    coilset.coil.insert(5, -0.5, 0.75, 0.75, Ic=15e6)
    coilset.grid.solve(20, 0.2, 'plasma')

    grid = coilset.grid.version
    subframe = coilset.subframe.version
    update[0] = grid['Psi'] != subframe['plasma']  # False
    separatrix = Polygon(dict(c=[4.5, 0.5, 2.5])).boundary
    coilset.plasma.update_separatrix(separatrix)
    update[1] = grid['Psi'] != subframe['plasma']  # True
    _ = coilset.grid.Psi
    update[2] = grid['Psi'] != subframe['plasma']  # False
    update[3] = grid['Br'] != subframe['plasma']  # True
    if plot:
        coilset.plot()
        coilset.grid.plot()
    assert np.equal(update, [False, True, False, True]).all()


def null_curvature(sign, plot):
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.plasma.insert(dict(o=[5, 1, 0.5]), Ic=15e6)
    coilset.coil.insert(5, 0, 0.75, 0.75, Ic=15e6)

    coilset.grid.solve(1e2, 0.25)  # generate plasma grid
    coilset.sloc.Ic *= sign
    if plot:
        coilset.plot()
        coilset.grid.plot()
        #cs.plasmagrid.global_null(True)
    assert False
    return coilset

#null_curvature(1, True)

def test_Opoint_curvature_Ip_positive(plot=False):
    cs = null_curvature(1, plot)
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint[0]) == 'O'


def test_Opoint_curvature_Ip_negative(plot=False):
    cs = null_curvature(-1, plot)
    assert cs.plasmagrid.null_type(cs.plasmagrid.Opoint[0]) == 'O'


def global_null(sign, plot=False):
    coilset = CoilSet(dcoil=0.5, dplasma=0.3)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)))
    coilset.grid.solve(3e2, 0.05) #[3.2, 8.5, -2.5, 2.5])
    coilset.sloc['Ic'] = sign*15e6

    if plot:
        coilset.plot()
        coilset.grid.plot(levels=51)

    return coilset


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
