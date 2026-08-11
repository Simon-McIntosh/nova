import os
from pathlib import Path
import pytest
import tempfile

import numpy as np
from nova.frame.coilset import CoilSet

# a conductor pair whose vertical force is set by the other coil and whose radial
# force is dominated by its own section, so both limits of the target rule show
PAIR = [
    ("PFa", 8.0, 6.5, 0.65, 0.45, 248.0, 45e3),
    ("PFb", 3.9, 7.6, 0.80, 0.60, 553.0, -30e3),
]


def conductor_pair(nforce=16, dcoil=-2):
    """Return the two-conductor coilset the target-rule measurements share."""
    coilset = CoilSet(nforce=nforce, dcoil=dcoil)
    for name, x, z, dx, dz, nturn, current in PAIR:
        coilset.coil.insert(x, z, dx, dz, nturn=nturn, Ic=current, name=name)
    return coilset


@pytest.fixture
def linked():
    coilset = CoilSet(nforce=10, dcoil=-2, dplasma=-3, tplasma="hex")
    coilset.coil.insert(5, 1, 0.1, 0.1, nturn=1)
    coilset.shell.insert({"e": [5, 1, 1.75, 1.0]}, 13, 0.05, delta=-9)
    coilset.shell.insert({"e": [5, 1, 1.95, 1.2]}, 13, 0.05, delta=-9)
    coilset.coil.insert(5, 2, 0.1, 0.2, nturn=1.3)
    coilset.coil.insert(5.2, 2, 0.1, 0.2, nturn=1.25)
    coilset.firstwall.insert(5.4, 1, 0.3, 0.6, section="e", Ic=-15e6)
    coilset.linkframe(["Coil2", "Coil0"])
    coilset.sloc["coil", "Ic"] = -15e6
    coilset.force.solve()
    return coilset


def test_turn_number():
    coilset = CoilSet(nforce=5, dcoil=-2)
    coilset.coil.insert(5, range(3), 0.1, 0.3, nturn=[1, 2, 3])
    coilset.force.solve()
    assert np.isclose(coilset.force.target.nturn.sum(), 6)


def test_negative_delta_frame():
    coilset = CoilSet(nforce=9, dcoil=-1)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_negative_delta_subframe():
    coilset = CoilSet(nforce=12, dcoil=-16)
    coilset.coil.insert(5, 6, 0.3, 0.3)
    coilset.force.solve()
    assert len(coilset.force) == 12


def test_positive_delta():
    coilset = CoilSet(nforce=-0.1, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 9


def test_unit_delta():
    coilset = CoilSet(nforce=1, dcoil=-2)
    coilset.coil.insert(5, 6, 0.9, 0.1)
    coilset.force.solve()
    assert len(coilset.force) == 1


def test_matrix_attrs(linked):
    for attr in ["Fr", "Fz", "Fc"]:
        assert attr in linked.force.data


def test_matrix_length(linked):
    assert len(linked.Loc["coil", :]) == len(linked.force.Fr)


def test_store_load(linked):
    fr = linked.force.fr
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        linked.filepath = tmp.name
        linked.store()
        del linked
        path = Path(tmp.name)
        coilset = CoilSet(filename=path.name, dirname=path.parent).load()
        coilset._clear()
    os.unlink(tmp.name)
    assert np.allclose(fr, coilset.force.fr)


def test_resolution():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(5, [5, 6], 0.9, 0.1, Ic=45e3, nturn=500)
    coilset.force.solve(100)
    fr_lowres = coilset.force.fr
    coilset.force.solve(200)
    fr_highres = coilset.force.fr
    assert np.allclose(fr_lowres, fr_highres, rtol=1e-3)


def test_totals_do_not_depend_on_the_source_mesh():
    """Integrating each source section leaves the force free of its subdivision."""
    reference = conductor_pair(dcoil=-2)
    reference.force.solve()
    for dcoil in (-5, -20):
        coilset = conductor_pair(dcoil=dcoil)
        coilset.force.solve()
        np.testing.assert_allclose(coilset.force.fr, reference.force.fr, rtol=1e-12)
        np.testing.assert_allclose(coilset.force.fz, reference.force.fz, rtol=1e-12)
        np.testing.assert_allclose(coilset.force.fc, reference.force.fc, rtol=1e-12)


def test_moment_arm_turns_about_the_conductor():
    """An arm divided by the cell would grow without bound as the tiling refines."""
    for nforce in (1, 4, 16, 64, 256):
        coilset = conductor_pair(nforce=nforce)
        coilset.force.solve()
        assert np.max(np.abs(coilset.force.target.delta_z)) <= 0.5 + 1e-12
        assert np.max(np.abs(coilset.force.target.delta_r)) <= 0.5 + 1e-12


def test_crushing_moment_converges_as_the_tiling_refines():
    """The first moment has a limit, so successive refinements approach it."""
    moment = []
    for nforce in (8, 32, 128, 512):
        coilset = conductor_pair(nforce=nforce)
        coilset.force.solve()
        moment.append(coilset.force.fc)
    step = np.abs(np.diff(np.asarray(moment), axis=0))
    assert np.all(step[1] < step[0])
    assert np.all(step[2] < step[1])


if __name__ == "__main__":
    pytest.main([__file__])
