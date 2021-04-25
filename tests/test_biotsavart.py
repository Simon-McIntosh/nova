import pytest
from numpy import allclose

from nova.electromagnetic.biotelements import mu_o


def test_ITER_subinductance_matrix(plot=False):
    """
    Test inductance calculation against DDD values for 2 CS and 1 PF coil.

    Baseline (old) CS geometory used.
    """
    cs = CoilSet(dCoil=0.25, turn_fraction=0.665,
                 biot_instances={'mutual': 'mutual'})
    cs.add_coil(3.9431, 7.5641, 0.9590, 0.9841, Nt=248.64,
                name='PF1', part='PF')
    cs.add_coil(1.722, 5.313, 0.719, 2.075, Nt=554, name='CS3U', part='CS')
    cs.add_coil(1.722, 3.188, 0.719, 2.075, Nt=554, name='CS2U', part='CS')
    cs.mutual.solve_interaction()  # nova
    Mc_ddd = [[7.076E-01, 1.348E-01, 6.021E-02],  # referance
              [1.348E-01, 7.954E-01, 2.471E-01],
              [6.021E-02, 2.471E-01, 7.954E-01]]
    if plot:
        cs.Ic = [5e6, -4e6, 2e6]
        cs.plot(label=True)
        cs.grid.generate_grid()
        cs.grid.plot_flux()
    assert allclose(Mc_ddd, cs.mutual._psi, atol=5e-3)
    return cs


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
