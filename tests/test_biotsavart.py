from numpy import allclose

from nova.electromagnetic.coilset import CoilSet


def test_inductance(plot=False):
    '''
    test inductance calculation against DDD values for 2 CS and 1 PF coil
    baseline (old) CS geometory used
    '''
    
    cs = CoilSet(dCoil=-1, turn_fraction=0.665)
    cs.add_coil(3.9431, 7.5641, 0.9590, 0.9841, Nt=248.64,
                name='PF1', part='PF')
    cs.add_coil(1.722, 5.313, 0.719, 2.075, Nt=554, name='CS3U', part='CS')
    cs.add_coil(1.722, 3.188, 0.719, 2.075, Nt=554, name='CS2U', part='CS')
    # calculate
    cs.mutual.solve_interaction()        
    # referance
    Mc_ddd = [[7.076E-01, 1.348E-01, 6.021E-02],
              [1.348E-01, 7.954E-01, 2.471E-01],
              [6.021E-02, 2.471E-01, 7.954E-01]]
    if plot:
        cs.Ic = [5e6, -4e6, 2e6]
        cs.plot(label=True)
        cs.grid.generate_grid()
        cs.grid.plot_flux()
        
    assert allclose(Mc_ddd, cs.mutual.psi , atol=5e-3)
    return cs


if __name__ == '__main__':
    cs = test_inductance(plot=False)
