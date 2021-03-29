import pytest
import numpy as np
import pickle

from nova.electromagnetic.frameset import FrameSet


def test_poloidal():
    frameset = FrameSet()  # Required=['x', 'z', 'dl']
    frameset.poloidal([4, 6, 7, 9, 9.5, 6], [1, 1, 2, 1.5, -1, -1.5], 0.1,
                      dpol=0.05, label='PF')
    assert len(frameset.frame) == 11
#test_poloidal()

'''
def test_shell():
    frameset = FrameSet()
    frameset.add_shell([4, 6, 7, 9, 9.5, 6], [1, 1, 2, 1.5, -1, -1.5], 0.1,
                 dShell=1, dCoil=-1, name='vvin')
    assert frameset._nC == 11


def test_circular_cross_section():
    'check subcoil area sum equals circle area'
    frameset = FrameSet(turn_fraction=0.5, cross_section='circle')
    frameset.add_coil(1.75, 0.5, 2.5, 2.5, turn_fraction=1, dCoil=-20)
    assert np.isclose(np.pi*2.5**2/4, frameset.subcoil.dA.sum(), rtol=1e-3)


def test_rectangular_cross_section():
    'check subcoil area sum equals circle area'
    frameset = FrameSet(turn_fraction=0.5, cross_section='rectangle')
    frameset.add_coil(1.75, 0.5, 2.5, 1.5, turn_fraction=1, dCoil=-20)
    assert np.isclose(2.5*1.5, frameset.subcoil.dA.sum(), rtol=1e-3)


def test_current_update():
    'test current update for active coils (active=True)'
    frameset = FrameSet(current_update='active', dCoil=0)
    frameset.add_coil(1, 2, 0.4, 0.5, Ic=5, active=True)
    frameset.add_coil(1, 3, 0.4, 0.5, Ic=6.7, active=False)
    frameset.add_coil(1, 4, 0.4, 0.5, Ic=4, active=True)
    frameset.Ic = [3.2, 5.8]
    frameset.current_update = 'full'
    assert np.allclose(frameset.Ic, [3.2, 6.7, 5.8])


def test_mpc():
    'query Ic from dataframe for pair of coils with mpc constraint'
    frameset = FrameSet(current_update='active', dCoil=0)
    frameset.add_coil(7, -3, 1.5, 1.5, name='PF6', part='PF', Nt=4, It=1e6,
                turn_section='circle', turn_fraction=0.7, dCoil=0.12)
    frameset.add_coil(7, -0.5, 1.5, 1.5, name='PF8', part='PF', Nt=5, Ic=2e3,
                cross_section='circle', turn_section='square', dCoil=0.12)
    frameset.add_mpc(['PF6', 'PF8'], -0.5)  # Ic[PF8] = -0.5*Ic[PF6]
    assert np.allclose(frameset.coil['It'].values, [1e6, -0.5*1e6/4*5])


def test_pickle():
    'test coilset sterilization'
    frameset = FrameSet()
    frameset.add_coil(1, [2, 3, 4], 0.4, 0.5, label='Coil', delim='_')
    _cs = FrameSet()
    cs_p = pickle.dumps(frameset.coilset)
    __cs = pickle.loads(cs_p)
    _cs.coilset = __cs
    assert frameset.coil.equals(_cs.coil)


if __name__ == '__main__':
    pytest.main([__file__])
'''
