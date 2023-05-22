import pytest
import numpy as np

from itertools import product

from nova.frame.coilset import CoilSet
from nova.geometry import select
from nova.geometry.polygon import Polygon


def meshgrid():
    x, z = np.meshgrid(np.arange(1, 4, 1, dtype=float),
                       np.arange(1, 4, 1, dtype=float), indexing='ij')
    x, z, = x.flatten(), z.flatten()
    return x, z


def coefficient_matrix(x, z):
    return np.c_[x**2, z**2, x, z, x*z, np.ones_like(x)]


def quadratic_surface(x, z, null_type: int, xo=2, zo=2):
    if null_type == 0:  # saddle
        return (x-xo)**2 + -(z-zo)**2
    if null_type == -1:  # minimum
        return (x-xo)**2 + (z-zo)**2
    if null_type == 1:  # maximum
        return -(x-xo)**2 + -(z-zo)**2
    if null_type == 2:  # plane
        return x + z + 1


@pytest.mark.parametrize('null_type', [-1, 0, 1, 2])
def test_quadratic_coefficents(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    coef = select.quadratic_surface(x, z, psi)
    assert np.allclose(psi, coefficient_matrix(x, z) @ coef)


@pytest.mark.parametrize('null_type', [-1, 0, 1])
def test_quadratic_null_type(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    coef = select.quadratic_surface(x, z, psi)
    assert select.null_type(coef) == null_type


@pytest.mark.parametrize('null_type,coordinate',
                         product([-1, 0, 1],
                                 [(0.8, 2.7), (2.2, 2.2), (-1, 5.2), (2, 2)]))
def test_quadratic_coordinate(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    coef = select.quadratic_surface(x, z, psi)
    assert np.allclose(select.null_coordinate(coef), coordinate)


@pytest.mark.parametrize('null_type,coordinate',
                         product([-1, 0, 1], [(1, 2.7), (2.2, 2.2), (2, 3)]))
def test_quadratic_coordinate_cluster(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    coef = select.quadratic_surface(x, z, psi)
    select.null_coordinate(coef, (x, z))


@pytest.mark.parametrize('null_type,coordinate',
                         product([-1, 0, 1],
                                 [(-4, 2.7), (8, 2),
                                  (2, -4), (0.8, 8)]))
def test_quadratic_coordinate_xcluster(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    coef = select.quadratic_surface(x, z, psi)
    with pytest.raises(AssertionError):
        select.null_coordinate(coef, (x, z))


def test_quadratic_plane_surface():
    x, z = meshgrid()
    psi = quadratic_surface(x, z, 2)
    coef = select.quadratic_surface(x, z, psi)
    with pytest.raises(ValueError):
        select.null_type(coef)


@pytest.mark.parametrize('null,coordinate',
                         product([-1, 0, 1], [(1, 2.7), (2.2, 2.2), (2, 3)]))
def test_subnull(null, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null, *coordinate)
    null_coords, null_psi, null_type = select.subnull(x, z, psi)
    assert np.allclose(null_coords, coordinate)
    assert np.isclose(null_psi, 0)
    assert null_type == null


def test_grid_2d():
    coilset = CoilSet()
    coilset.coil.insert(5, [-1, 1], 0.75, 0.5, Ic=1e3)
    coilset.grid.solve(50, 0.2)
    assert coilset.grid.x_point_number == 1
    assert np.isclose(coilset.grid.x_points[0][1], 0, atol=1e-2)


def test_grid_1d():
    coilset = CoilSet(dplasma=-50, tplasma='hex')
    coilset.firstwall.insert(dict(ellip=[0.5, 0, 0.075, 0.15]), Ic=15e3)
    coilset.coil.insert(0.5, [-0.08, 0.08], 0.01, 0.01, Ic=5e3)
    coilset.plasmagrid.solve()
    assert coilset.plasmagrid.o_points[0][0] != \
        coilset.plasmagrid.x_points[0][0]
    assert coilset.plasmagrid.o_point_number == 1
    assert coilset.plasmagrid.x_point_number == 2


def test_plasma_update(plot=False):
    update = np.full(4, False)
    coilset = CoilSet(required=['x', 'z', 'dx', 'dz'],
                      dplasma=-5, dcoil=0.5, tplasma='hex')
    coilset.firstwall.insert(dict(disc=[4.5, 0.5, 1]), Ic=15e6)
    coilset.coil.insert(5, -0.5, 0.75, 0.75, Ic=15e6)
    coilset.grid.solve(20, 0.2, 'plasma')
    grid = coilset.grid.version
    subframe = coilset.subframe.version
    update[0] = grid['Psi'] != subframe['nturn']  # False
    coilset.plasma.separatrix = Polygon(dict(c=[4.5, 0.5, 0.5])).boundary
    update[1] = grid['Psi'] != subframe['nturn']  # True
    _ = coilset.grid.psi
    update[2] = grid['Psi'] != subframe['nturn']  # False
    update[3] = grid['Br'] != subframe['nturn']  # True
    if plot:
        coilset.plot()
        coilset.grid.plot()
    assert np.equal(update, [False, True, False, True]).all()


def null_curvature(sign, plot):
    coilset = CoilSet(dcoil=-5, dplasma=-5, tplasma='hex')
    coilset.firstwall.insert(dict(o=[5, 1, 0.5]), Ic=15e6)
    coilset.coil.insert(5, -0.25, 0.75, 0.75, Ic=-15e6)
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
    coilset = CoilSet(dcoil=-5, dplasma=-5, tplasma='hex')
    coilset.coil.insert(5, [-1.1, 1.1], 0.75, 0.75, Ic=[1, 1])
    coilset.firstwall.insert(dict(o=[5.25, 0, 0.5]), Ic=0.5)
    coilset.grid.solve(250, 1.5, 'plasma')  # generate plasma grid
    if plot:
        coilset.plot()
        coilset.grid.plot()
    assert coilset.grid.x_point_number == 2
    assert coilset.grid.o_point_number == 1


def test_empty():
    coilset = CoilSet(dcoil=-5, dplasma=-5, tplasma='hex')
    coilset.coil.insert(5, [-1, 1], 0.75, 0.75, Ic=[-1, 1])
    coilset.firstwall.insert(dict(o=[5.25, 0, 0.5]), Ic=0)
    coilset.grid.solve(1e2, 0.25, 'plasma')  # generate plasma grid
    assert coilset.grid.x_point_number == 0
    assert coilset.grid.o_point_number == 0


def global_null(sign, plot=False):
    coilset = CoilSet(dcoil=0.5, dplasma=0.4, tplasma='hex')
    coilset.coil.insert(5, [-2, 2], 0.75, 0.75)
    coilset.coil.insert(7.8, 0, 0.75, 0.75, label='Xcoil')
    coilset.firstwall.insert(dict(o=(4, 0, 0.5)))
    coilset.grid.solve(500, 0.05)
    coilset.sloc['Ic'] = sign*15e6
    if plot:
        coilset.plot()
        coilset.grid.plot(levels=51)
    return coilset


def test_unique_null():
    coilset = CoilSet()
    coilset.coil.insert([1.1, 1.2, 1.3], 0, 0.075, 0.15, Ic=15e3)
    coilset.grid.solve(100)
    assert coilset.grid.x_point_number == 1
    assert coilset.grid.o_point_number == 2


def test_global_null_Ip_positive(plot=False):
    coilset = global_null(1, plot)
    assert coilset.grid.x_point_number == 3
    assert coilset.grid.o_point_number == 4


def test_global_null_Ip_negative(plot=False):
    coilset = global_null(-1, plot)
    assert coilset.grid.x_point_number == 3
    assert coilset.grid.o_point_number == 4


def test_plasma_coil_parity(plot=False):
    coilset = CoilSet(dcoil=-4, dplasma=-4, tplasma='hex')
    coilset.coil.insert(5, 0.75, 0.75, 0.75, turn='r')
    coilset.firstwall.insert({'r': [5, -0.75, 0.75, 0.75]}, turn='r')
    coilset.grid.solve(150, 0.05)
    coilset.sloc[:, 'Ic'] = 15e6
    coilset.grid.update_turns('Psi', svd=False)
    if plot:
        coilset.plot()
        coilset.grid.plot()
    assert np.isclose(coilset.grid.x_points[0][1], 0, atol=1e-3)


def test_plasma_unique_psi_axis():
    coilset = CoilSet(dplasma=-20, tplasma='hex')
    coilset.firstwall.insert({'e': [0.5, 0, 0.2, 0.1]})
    coilset.coil.insert(0.5, [-0.05, 0.05], 0.01, 0.01, Ic=1e3)
    coilset.plasma.solve()
    coilset.sloc['plasma', 'Ic'] = 1e3
    with pytest.raises(IndexError):
        coilset.plasma.psi_axis


def test_plasma_x_point():
    coilset = CoilSet(dplasma=-100, tplasma='hex')
    coilset.firstwall.insert({'e': [0.5, 0, 0.1, 0.2]})
    coilset.coil.insert(0.485, [-0.12, 0.12], 0.03, 0.03, Ic=5e3)
    coilset.plasma.solve()
    coilset.sloc['plasma', 'Ic'] = 7e3
    coilset.saloc['Ic'][1] = 4.5e3
    assert coilset.plasmagrid['x_point'][1] > 0
    coilset.saloc['Ic'][1] = 5.5e3
    coilset.plasma.grid.check_null()
    assert coilset.plasmagrid['x_point'][1] < 0


if __name__ == '__main__':
    pytest.main([__file__])
