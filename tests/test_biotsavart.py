import pytest
from numpy import allclose
import numpy as np


from nova.biot.biotmatrix import BiotMatrix
from nova.biot.biotframe import BiotFrame
from nova.biot.biotgrid import BiotGrid
from nova.biot.biotpoint import BiotPoint
from nova.biot.biotring import BiotRing
from nova.biot.biotsolve import BiotSolve
from nova.frame.coilset import CoilSet

segments = ['ring', 'cylinder']


def test_biotreduce():
    biotframe = BiotFrame()
    biotframe.insert(range(3), 0)
    biotframe.insert(range(3), 1, link=True)
    biotframe.insert(range(3), 2, link=False)
    biotframe.insert(range(3), 3, link=True)
    biotframe.multipoint.link(['Coil0', 'Coil11', 'Coil2', 'Coil8'])
    assert biotframe.biotreduce.indices == [0, 1, 2, 3, 6, 7, 8, 9, 11]
    assert list(biotframe.biotreduce.link) == [2, 6, 8]
    assert biotframe.biotreduce.index.to_list() == \
        [f'Coil{i}' for i in [0, 1, 3, 6, 7, 9]]


def test_subframe_lock():
    biotframe = BiotFrame(subspace=['Ic'])
    biotframe.insert([1, 3], 0, dl=0.95, dt=0.95, section='hex')
    assert biotframe.lock('subspace') is False


def test_link_negative_factor():
    biotframe = BiotFrame(label='C')
    biotframe.insert(1, 0)
    biotframe.insert(1, 0)
    biotframe.multipoint.link(['C0', 'C1'], -1)
    biot = BiotRing(biotframe, biotframe, reduce=[True, True])
    assert np.isclose(biot.compute('Psi')[0][0, 0], 0)


def test_random_segment_error():
    biotframe = BiotFrame(label='C')
    biotframe.insert(1, 0, segment='circle')
    biotframe.insert(1, 0, segment='random')
    with pytest.raises(NotImplementedError):
        BiotSolve(biotframe, biotframe)


@pytest.mark.parametrize('segment', segments)
def test_ITER_subinductance_matrix(segment):
    """
    Test inductance calculation against DDD values for 2 CS and 1 PF coil.

    Baseline (old) CS geometory used.
    """
    coilset = CoilSet(dcoil=0.25)
    coilset.coil.insert(3.9431, 7.5641, 0.9590, 0.9841, nturn=248.64,
                        name='PF1', part='PF', segment=segment)
    coilset.coil.insert(1.722, 5.313, 0.719, 2.075, nturn=554,
                        name='CS3U', part='CS', segment=segment)
    coilset.coil.insert(1.722, 3.188, 0.719, 2.075, nturn=554,
                        name='CS2U', part='CS', segment=segment)
    biot = BiotRing(coilset.subframe, coilset.subframe,
                    turns=[True, True], reduce=[True, True])
    Mc_ddd = [[7.076E-01, 1.348E-01, 6.021E-02],  # referance
              [1.348E-01, 7.954E-01, 2.471E-01],
              [6.021E-02, 2.471E-01, 7.954E-01]]
    assert allclose(Mc_ddd, biot.compute('Psi')[0], atol=5e-3)


def test_solenoid_grid():
    """verify solenoid vertical field using grid biot instance."""
    nturn, height, current = 500, 30, 1e3
    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(1.5, 0, 0.01, height, nturn=nturn, section='rect')
    coilset.sloc['Ic'] = current
    biotgrid = BiotGrid(*coilset.frames)
    biotgrid.solve(4, [1e-9, 1.5, 0, 1])
    Bz_theory = BiotMatrix.mu_0 * nturn * current / height
    Bz_grid = np.dot(biotgrid.data.Bz, coilset.sloc['Ic'])
    assert allclose(Bz_grid[0], Bz_theory, atol=5e-3)


@pytest.mark.parametrize('segment', segments)
def test_solenoid_probe(segment):
    """Verify solenoid vertical field using probe biot instance."""
    nturn, height, current = 500, 30, 1e3
    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(1.5, 0, 0.01, height, nturn=nturn,
                        section='rectangle', segment=segment)
    coilset.sloc['Ic'] = current
    biotpoint = BiotPoint(*coilset.frames)
    biotpoint.solve((1e-9, 0))
    Bz_theory = BiotMatrix.mu_0 * nturn * current / height
    Bz_point = np.dot(biotpoint.data.Bz, coilset.sloc['Ic'])
    assert allclose(Bz_point, Bz_theory, atol=5e-3)


def test_ring_ring_coil_pair():
    coilset = CoilSet(dcoil=-10)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment='ring')
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment='ring')
    coilset.point.solve([[8, 0]])
    assert np.isclose(coilset.point.psi, 0)


def test_cyliner_cylinder_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment='cylinder')
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment='cylinder',
                        delta=-10)
    coilset.point.solve([[8, 0]])
    assert np.isclose(coilset.point.psi, 0)


def test_cylinder_ring_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=-15e6, segment='cylinder')
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=15e6, segment='ring',
                        delta=-10)
    coilset.point.solve([[7, 0]])
    coilset.plot()
    coilset.grid.solve(1000)
    coilset.grid.plot()
    print(coilset.point.psi)
    assert np.isclose(coilset.point.psi, 0, atol=1e-3)

test_cylinder_ring_coil_pair()
assert False

@pytest.mark.parametrize('segment', segments)
def test_hemholtz_flux(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    point_radius = 0.1
    coilset.point.solve([[point_radius, 0]])
    Bz = (4/5)**(3/2) * BiotMatrix.mu_0
    psi = Bz * np.pi*point_radius**2
    assert np.isclose(coilset.point.psi[0], psi)


@pytest.mark.parametrize('segment', segments)
def test_hemholtz_field(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    coilset.point.solve([[1e-3, 0]])
    bz = (4/5)**(3/2) * BiotMatrix.mu_0
    assert np.isclose(coilset.point.bz[0], bz)


def test_coil_cylinder_isfinite_farfield():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.5, [-1, 0, 1], 0.4, 0.4, Ic=-15e6,
                        segment='cylinder')
    coilset.grid.solve(60, [6, 7.0, -0.8, 0.8])
    assert np.isfinite(coilset.grid.psi).all()


def test_coil_cylinder_isfinite_coil():
    coilset = CoilSet(dcoil=-2**3)
    coilset.coil.insert(0.3, 0, 0.15, 0.15, segment='cylinder', Ic=5e3)
    coilset.grid.solve(10**2, 0)
    assert np.isfinite(coilset.grid.psi).all()


if __name__ == '__main__':
    pytest.main([__file__])
