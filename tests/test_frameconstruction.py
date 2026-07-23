"""Regression coverage for the coilset construction tier over the columnar store.

These guard the frame -> subframe meshing path that assembles a subframe from a
frame row and a generated poly/shell grid. The failure they lock in was a
multi-column ``frame.loc[index, :]`` sub-frame selection reaching the columnar
store as if the trailing slice were a column name.
"""

import numpy as np

from nova.frame.coilset import CoilSet


def test_bare_coil_insert_builds_subframe():
    """A single coil insert populates one frame row and a meshed subframe."""
    coilset = CoilSet()
    index = coilset.coil.insert(5, 0.5, 0.1, 0.1, Ic=1e3)
    assert list(index) == ["Coil0"]
    assert len(coilset.frame) == 1
    assert len(coilset.subframe) >= 1
    # the frame row carries the requested coil geometry
    assert np.isclose(coilset.Loc["Coil0", "x"], 5)
    assert np.isclose(coilset.Loc["Coil0", "z"], 0.5)


def test_coil_insert_propagates_current_to_subframe():
    """The inserted coil current reads back through loc and the array accessor."""
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=["Ic"])
    coilset.coil.insert([1, 3], required=["x"], Ic=[7.7, 6.6])
    assert np.isclose(coilset.sloc["Ic"], [7.7, 6.6]).all()
    assert np.isclose(
        coilset.aloc["Ic"], coilset.sloc["Ic"][coilset.subframe.subref]
    ).all()


def test_coil_subframe_turns_normalise():
    """Meshed subframe turns sum to the requested frame nturn per coil."""
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(5, 0.5, 0.3, 0.3, nturn=1)
    assert np.isclose(coilset.subframe.nturn.sum(), 1.0)
