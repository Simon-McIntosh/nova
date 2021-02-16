import pytest
import numpy as np
import pickle

from nova.electromagnetic.coilset import CoilSet


def test_dCoil():
    'test setting of kwarg dCoil'
    cs = CoilSet(dCoil=0.15)
    assert cs.dCoil == 0.15


def test_dCoil_default():
    'test instance default attribute setting'
    cs = CoilSet()
    cs.coilset = {'default_attributes': {'dCoil': 3}}
    assert cs.dCoil == 3


def test_dCoil_memory():
    'test persistance of default attributes (set once)'
    cs = CoilSet(dCoil=0.15)
    cs.coilset = {'default_attributes': {'dCoil': 3}}
    assert cs.dCoil == 0.15


def test_dPlasma():
    'test dPlasma and coilset shorthand'
    cs = CoilSet()
    cs.coilset = {'dPlasma': 0.333}
    assert cs.dPlasma == 0.333


def test_add_shell():
    cs = CoilSet()
    cs.add_shell([4, 6, 7, 9, 9.5, 6], [1, 1, 2, 1.5, -1, -1.5], 0.1,
                 dShell=1, dCoil=-1, name='vvin')
    assert cs._nC == 11


def test_circular_cross_section():
    'check subcoil area sum equals circle area'
    cs = CoilSet(turn_fraction=0.5, cross_section='circle')
    cs.add_coil(1.75, 0.5, 2.5, 2.5, turn_fraction=1, dCoil=-20)
    assert np.isclose(np.pi*2.5**2/4, cs.subcoil.dA.sum(), rtol=1e-3)


def test_rectangular_cross_section():
    'check subcoil area sum equals circle area'
    cs = CoilSet(turn_fraction=0.5, cross_section='rectangle')
    cs.add_coil(1.75, 0.5, 2.5, 1.5, turn_fraction=1, dCoil=-20)
    assert np.isclose(2.5*1.5, cs.subcoil.dA.sum(), rtol=1e-3)


def test_current_update():
    'test current update for active coils (active=True)'
    cs = CoilSet(current_update='active', dCoil=0)
    cs.add_coil(1, 2, 0.4, 0.5, Ic=5, active=True)
    cs.add_coil(1, 3, 0.4, 0.5, Ic=6.7, active=False)
    cs.add_coil(1, 4, 0.4, 0.5, Ic=4, active=True)
    cs.Ic = [3.2, 5.8]
    cs.current_update = 'full'
    assert np.allclose(cs.Ic, [3.2, 6.7, 5.8])


def test_mpc():
    'query Ic from dataframe for pair of coils with mpc constraint'
    cs = CoilSet(current_update='active', dCoil=0)
    cs.add_coil(7, -3, 1.5, 1.5, name='PF6', part='PF', Nt=4, It=1e6,
                turn_section='circle', turn_fraction=0.7, dCoil=0.12)
    cs.add_coil(7, -0.5, 1.5, 1.5, name='PF8', part='PF', Nt=5, Ic=2e3,
                cross_section='circle', turn_section='square', dCoil=0.12)
    cs.add_mpc(['PF6', 'PF8'], -0.5)  # Ic[PF8] = -0.5*Ic[PF6]
    assert np.allclose(cs.coil['It'].values, [1e6, -0.5*1e6/4*5])


def test_pickle():
    'test coilset sterilization'
    cs = CoilSet()
    cs.add_coil(1, [2, 3, 4], 0.4, 0.5, label='Coil', delim='_')
    _cs = CoilSet()
    cs_p = pickle.dumps(cs.coilset)
    __cs = pickle.loads(cs_p)
    _cs.coilset = __cs
    assert cs.coil.equals(_cs.coil)


if __name__ == '__main__':
    pytest.main([__file__])
