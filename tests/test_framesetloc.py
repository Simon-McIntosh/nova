import pytest

import numpy as np

from nova.frame.coilset import CoilSet
from nova.frame.error import SubSpaceKeyError, ColumnError


def test_get_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[])
    coilset.coil.insert(1, required=["x"], Ic=[7.7])
    with pytest.raises(SubSpaceKeyError):
        _ = coilset.sloc["Ic"]


def test_set_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[])
    coilset.coil.insert(1, required=["x"], Ic=[7.7])
    with pytest.raises(SubSpaceKeyError):
        coilset.sloc["Ic"] = [8.8, 8.8]


def test_get_key_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[])
    coilset.coil.insert(1, required=["x"], Ic=[7.7])
    with pytest.raises(KeyError):
        _ = coilset.loc["turn"]


def test_set_column_error():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[])
    coilset.coil.insert(1, required=["x"], Ic=[7.7])
    with pytest.raises(ColumnError):
        coilset.loc["turn"] = [8.8, 8.8]


def test_get_current_subframe():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[])
    coilset.coil.insert(1, required=["x"], Ic=[7.7])
    assert np.isclose(coilset.loc["Ic"], [7.7, 7.7]).all()


def test_get_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert([1, 3, 7], required=["x"], Ic=[7.7, 8.3, 6.6])
    assert np.isclose(coilset.sloc["Ic"], [7.7, 8.3, 6.6]).all()


def test_get_current_subspace_array():
    coilset = CoilSet(dcoil=-1, subspace=["Ic"], array=["Ic"])
    coilset.coil.insert([1, 3], required=["x"], Ic=[7.7, 6.6])
    assert np.isclose(coilset.sloc["Ic"], [7.7, 6.6]).all()


def test_set_current_subframe():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], additional=["Ic"])
    coilset.coil.insert(1.5, required=["x"])
    coilset.loc["active", "Ic"] = [8.8, 7.7]
    assert np.isclose(coilset.loc["Ic"], [8.8, 7.7]).all()


def test_set_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    coilset.sloc["Ic"] = [8.8]
    assert np.isclose(coilset.loc["Ic"], [8.8]).all()


def test_set_current_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    with pytest.raises(ValueError):
        coilset.sloc["Ic"] = [8.8, 8.8]


def test_set_current_subspace_array():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=["Ic"])
    coilset.coil.insert(1.5, required=["x"])
    coilset.sloc["Ic"] = [8.8]
    assert np.isclose(coilset.sloc["Ic"], [8.8]).all()


def test_get_current_subset():
    coilset = CoilSet(dcoil=-1, subspace=["Ic"], array=["Ic"])
    coilset.coil.insert(3, required=["x"], plasma=False)
    coilset.coil.insert(6.6, required=["x"], plasma=True)
    coilset.coil.insert([1.2, 2.2], required=["x"], active=False)
    coilset.linkframe(["Coil0", "Coil3"])
    coilset.sloc["active", "Ic"] = [8.8, 7.7]
    assert np.isclose(coilset.loc["Ic"], [8.8, 7.7, 0, 8.8]).all()


def test_get_current_insert_default():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=["Ic"])
    coilset.coil.insert(range(2), required=["x"], link=True)
    coilset.firstwall.insert({"s": [3.25, 0, 0.25]}, delta=-2)
    coilset.shell.insert([2.2, 3.2], [-0.1, 0.3], -2, 0.05, delta=-3, link=True)
    coilset.sloc["Ic"] = 4.4
    coilset.sloc["plasma", "Ic"] = [9.9]
    coilset.sloc["active", "Ic"] = [3.3]
    coilset.sloc["coil", "Ic"] = [5.5]
    assert np.isclose(coilset.sloc["Ic"], [5.5, 3.3, 4.4, 4.4]).all()


def test_set_current_frame_error():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    with pytest.raises(ColumnError):
        coilset.Loc["Ic"] = [8.8, 8.8]


def test_set_current_frame_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    with pytest.raises(SubSpaceKeyError):
        coilset.sLoc["It"] = [8.8, 8.8]


def test_get_current_frame_keyerror():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    with pytest.raises(KeyError):
        _ = coilset.Loc["Ic"]


def test_get_current_subspace_keyerror():
    coilset = CoilSet(dcoil=-2, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"])
    with pytest.raises(SubSpaceKeyError):
        _ = coilset.sLoc["It"]


def test_set_frame_It_subspace_Ic():
    coilset = CoilSet(dcoil=-1, subspace=["Ic"], array=[])
    coilset.coil.insert(1.5, required=["x"], nturn=3)
    coilset.loc["It"] = 9.9
    assert np.isclose(coilset.sloc["Ic"][0], 3.3)


def test_subframe_plasma_index():
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(3, 0, 0.1, 0.1)
    coilset.firstwall.insert({"sq": [3.25, 0, 0.25]}, delta=-4, turn="sq", tile=False)
    assert list(coilset.loc["plasma", "plasma"]) == [True, True, True, True]


if __name__ == "__main__":
    pytest.main([__file__])
