import pytest
import numpy as np

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
    coilset.coil.insert(5, 0, 0.75, 0.75, Ic=-15e6)
    coilset.grid.solve(1e2, 0.25)  # generate plasma grid
    coilset.sloc['plasma', 'Ic'] *= sign
    if plot:
        coilset.plot()
        coilset.grid.plot()
    return coilset


def test_Opoint_curvature_Ip_positive(plot=False):
    coilset = null_curvature(1, plot)
    assert coilset.grid.x_point_number == 0
    assert coilset.grid.o_point_number == 2


def test_Opoint_curvature_Ip_negative(plot=False):
    coilset = null_curvature(-1, plot)
    assert coilset.grid.x_point_number == 1
    assert coilset.grid.o_point_number == 2


def test_multi_xpoint(plot=False):
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.coil.insert(5, [-1, 1], 0.75, 0.75, Ic=[1, 1])
    coilset.plasma.insert(dict(o=[5.25, 0, 0.5]), Ic=0.1)
    coilset.grid.solve(3e2, 1.25, 'plasma')  # generate plasma grid
    if plot:
        coilset.plot()
        coilset.grid.plot()
    assert coilset.grid.x_point_number == 3
    assert coilset.grid.o_point_number == 2


def test_empty():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.coil.insert(5, [-1, 1], 0.75, 0.75, Ic=[-1, 1])
    coilset.plasma.insert(dict(o=[5.25, 0, 0.5]), Ic=0)
    coilset.grid.solve(1e2, 0.25, 'plasma')  # generate plasma grid
    assert coilset.grid.x_point_number == 0
    assert coilset.grid.o_point_number == 0


def global_null(sign, plot=False):
    coilset = CoilSet(dcoil=0.5, dplasma=0.3)
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.plasma.insert(dict(o=(4, 0, 0.5)))
    coilset.grid.solve(500, 0.05)
    coilset.sloc['Ic'] = sign*15e6
    if plot:
        coilset.plot()
        coilset.grid.plot(levels=51)
    return coilset


def test_global_null_Ip_positive(plot=False):
    coilset = global_null(1, plot)
    assert coilset.grid.x_point_number == 3
    assert coilset.grid.o_point_number == 4


def test_global_null_Ip_negative(plot=False):
    coilset = global_null(-1, plot)
    assert coilset.grid.x_point_number == 3
    assert coilset.grid.o_point_number == 4


if __name__ == '__main__':
    pytest.main([__file__])

    # test_xtol_rel_attribute()
    #cs = global_null(1, plot=True)
