from nova.electromagnetic.coilset import CoilSet
import numpy as np


def test_solenoid_grid(plot=False):
    
    N, L, Ic = 500, 30, 1e3
    
    cs = CoilSet()
    cs.add_coil(1.5, 0, 0.01, L, Nt=N, turn_section='rectangle', dCoil=0.5)
    cs.Ic = Ic
    
    if plot:
        cs.grid.generate_grid(limit=[1e-5, 3, -0.6*L, 0.6*L], n=1e4)
        cs.grid.plot_flux()
        cs.plot()

    cs.grid.generate_grid(limit=[1e-9, 1.5, 0, 1], n=4)
        
    Bz_theory = cs.grid.mu_o * N * Ic / L
    Bz = cs.grid.Bz[0, 0]
    
    assert np.allclose(Bz, Bz_theory, atol=5e-4)
    return cs, Bz, Bz_theory

def test_solenoid_target():
    
    N, L, Ic = 500, 300, 1e3
    
    cs = CoilSet()
    cs.add_coil(1.5, 0, 0.01, L, Nt=N, turn_section='rectangle', dCoil=0.5)
    cs.Ic = Ic


    cs.target.add_targets([1e-9, 0])
    cs.target.solve_interaction()
    
    Bz_theory = cs.grid.mu_o * N * Ic / L
    Bz = cs.target.Bz[0]
    
    assert np.allclose(Bz, Bz_theory, atol=5e-4)
    return cs, Bz, Bz_theory

if __name__ == '__main__':
    # cs, Bz, Bz_theory = test_solenoid_grid(plot=True)
    cs, Bz, Bz_theory = test_solenoid_target()
    print(Bz, Bz_theory)

