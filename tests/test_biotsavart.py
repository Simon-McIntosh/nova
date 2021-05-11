import pytest
from numpy import allclose
import numpy as np

from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.coilset import CoilSet


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
    biot = Biot(biotframe, biotframe, reduce=[True, True])
    assert np.isclose(biot.static.Psi[0], 0)


def test_random_segment_error():
    biotframe = BiotFrame(label='C')
    biotframe.insert(1, 0, segment='circle')
    biotframe.insert(1, 0, segment='random')
    with pytest.raises(NotImplementedError):
        Biot(biotframe, biotframe)


def test_ITER_subinductance_matrix():
    """
    Test inductance calculation against DDD values for 2 CS and 1 PF coil.

    Baseline (old) CS geometory used.
    """
    coilset = CoilSet(dcoil=0.25)
    coilset.coil.insert(3.9431, 7.5641, 0.9590, 0.9841, nturn=248.64,
                        name='PF1', part='PF')
    coilset.coil.insert(1.722, 5.313, 0.719, 2.075, nturn=554,
                        name='CS3U', part='CS')
    coilset.coil.insert(1.722, 3.188, 0.719, 2.075, nturn=554,
                        name='CS2U', part='CS')
    biot = Biot(coilset.subframe, coilset.subframe,
                turns=[True, True], reduce=[True, True])
    Mc_ddd = [[7.076E-01, 1.348E-01, 6.021E-02],  # referance
              [1.348E-01, 7.954E-01, 2.471E-01],
              [6.021E-02, 2.471E-01, 7.954E-01]]
    assert allclose(Mc_ddd, biot.static.Psi, atol=5e-3)


def test_solenoid_grid(plot=False):
    """verify solenoid vertical field using grid biot instance."""
    N, L, Ic = 500, 30, 1e3
    cs = CoilSet()
    cs.add_coil(1.5, 0, 0.01, L, Nt=N, turn_section='rectangle', dCoil=0.5)
    cs.Ic = Ic
    cs.biot_instances = {'grid': 'grid'}
    if plot:
        cs.grid.generate_grid(limit=[1e-5, 3, -0.6*L, 0.6*L], n=1e4)
        cs.grid.plot_flux()
        cs.plot()
    cs.grid.generate_grid(limit=[1e-9, 1.5, 0, 1], n=4)
    Bz_theory = mu_o * N * Ic / L
    Bz = cs.grid.Bz[0, 0]
    assert allclose(Bz, Bz_theory, atol=5e-3)
    return cs, Bz, Bz_theory


def test_solenoid_probe():
    """Verify solenoid vertical field using probe biot instance."""
    N, L, Ic = 500, 30, 1e3
    cs = CoilSet()
    cs.add_coil(1.5, 0, 0.01, L, Nt=N, turn_section='rectangle', dCoil=0.5)
    cs.Ic = Ic
    cs.biot_instances = {'probe': 'probe'}
    cs.probe.add_target(1e-9, 0)
    Bz_theory = mu_o * N * Ic / L
    Bz = cs.probe.Bz[0]
    assert allclose(Bz, Bz_theory, atol=5e-3)
    return cs, Bz, Bz_theory


if __name__ == '__main__':
    pytest.main([__file__])

    # test_ITER_subinductance_matrix()

    # cs = test_ITER_subinductance_matrix(plot=True)
    # cs, Bz, Bz_theory = test_solenoid_grid(plot=True)
    # cs, Bz, Bz_theory = test_solenoid_target()
