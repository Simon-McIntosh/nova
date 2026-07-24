"""Fixed-point contract for the forward free-boundary equilibrium solve."""

import numpy as np
import pytest

from nova.equilibrium.forward import ForwardProfile
from nova.frame.coilset import CoilSet


@pytest.fixture(scope="module")
def plasma():
    """Return a solved diverted plasma component with a trial separatrix."""
    coilset = CoilSet(dcoil=-5, dplasma=-80, tplasma="hex", nwall=3, nlevelset=1500)
    coilset.coil.insert(6.5, [-1.1, 1.1], 0.4, 0.2, Ic=-15e6)
    coilset.firstwall.insert({"e": [6.5, 0, 1.2, 1.6]}, Ic=-15e6, turn="hex")
    coilset.plasma.separatrix = {"e": [6.5, 0, 0.5, 0.7]}
    coilset.plasma.solve()
    return coilset.plasma


def test_residual_vanishes_at_the_solution(plasma):
    """A converged flux map reproduces itself through the free-boundary map."""
    forward = ForwardProfile(plasma)
    trial = np.array(plasma.psi, float)
    assert np.linalg.norm(forward.residual(trial)) > 1
    psi = forward.solve(f_rtol=1e-4)
    assert np.linalg.norm(forward.residual(psi)) < 1e-3 * np.linalg.norm(psi)


def test_solve_writes_the_solution_back(plasma):
    """The solve leaves the plasma component holding the converged flux map."""
    forward = ForwardProfile(plasma)
    psi = forward.solve(f_rtol=1e-4)
    assert np.allclose(plasma.psi, psi, rtol=1e-5)
    assert psi.size == plasma.grid.number + plasma.wall.number


def test_ionization_follows_the_flux_map(plasma):
    """Writing a flux map re-ionizes the filaments and renormalizes the turns."""
    forward = ForwardProfile(plasma)
    forward.solve(f_rtol=1e-4)
    assert plasma.aloc["plasma", "ionize"].sum() > 0
    assert np.isclose(np.asarray(plasma.aloc["ionize", "nturn"]).sum(), 1)
