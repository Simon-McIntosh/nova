import os
import pathlib

import pytest
import tempfile

import numpy as np

from nova.frame.frameset import FrameSet
from nova.frame.error import SpaceKeyError


def test_space_setattr_error():
    frameset = FrameSet(required=["rms"], additional=["Ic"])
    frameset.subframe.insert([2, 4], It=6, link=True)
    with pytest.raises(SpaceKeyError):
        frameset.subframe.Ic = 7


def test_store_load():
    frameset = FrameSet(required=["rms"], additional=["Ic"])
    frameset.subframe.insert([2, 4], It=6, link=True)
    subframe = frameset.subframe
    with tempfile.NamedTemporaryFile() as tmp:
        frameset.filepath = tmp.name
        frameset.store()
        del frameset
        frameset = FrameSet()
        frameset.filepath = tmp.name
        frameset.load()
        frameset._clear()
    assert (frameset.subframe.link == subframe.link).all()
    assert np.isclose(frameset.sloc["Ic"], [6]).all()


def test_frameset_persists_through_zarr(tmp_path):
    """The frameset cache is a grouped zarr store, not a netCDF file."""
    frameset = FrameSet(required=["rms"], additional=["Ic"])
    frameset.subframe.insert([2, 4], It=6, link=True)
    frameset.filepath = os.path.join(tmp_path, "frames")
    assert frameset.filepath.suffix == ".zarr"
    frameset.store()
    assert frameset.filepath.is_dir()  # zarr store is a directory
    # frame and subframe live as named groups within the one store
    import zarr

    root = zarr.open_group(store=str(frameset.filepath), mode="r")
    assert {"frame", "subframe"} <= set(root.group_keys())


def _coilset_with_operator(tmp_path):
    """Return a two-coil CoilSet with a solved operator, cache path set."""
    from nova.frame.coilset import CoilSet

    coilset = CoilSet()
    coilset.coil.insert({"r": [4.0, 0.5, 0.2, 0.3]}, name="C1", part="pf", Ic=1e3)
    coilset.coil.insert({"r": [5.0, -0.5, 0.2, 0.3]}, name="C2", part="pf", Ic=1e3)
    coilset.inductance.solve()
    coilset.filepath = os.path.join(tmp_path, "coilset")
    return coilset


def test_coilset_frames_and_operators_round_trip(tmp_path):
    """A cold build stores frames and a solved operator; a reload matches."""
    import xarray

    coilset = _coilset_with_operator(tmp_path)
    frame_link = np.asarray(coilset.frame.link)
    operator = coilset.inductance.data.copy(deep=True)
    coilset.store()

    from nova.frame.coilset import CoilSet

    reloaded = CoilSet()
    reloaded.filepath = os.path.join(tmp_path, "coilset")
    reloaded.load()
    assert len(reloaded.frame) == 2
    assert (np.asarray(reloaded.frame.link) == frame_link).all()
    xarray.testing.assert_identical(reloaded.inductance.data, operator)


def _group_signature(store_path, group):
    """Return a stable signature of a zarr group's on-disk chunk bytes."""
    root = pathlib.Path(store_path) / group
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_operator_rebuild_keeps_frame_group_intact(tmp_path):
    """Re-storing a solved operator evicts only its group, not the frame."""
    coilset = _coilset_with_operator(tmp_path)
    coilset.store()
    before = _group_signature(coilset.filepath, "frame")

    # solve and re-store the operator; the frame group must not be rewritten
    coilset.inductance.solve()
    coilset.store()
    after = _group_signature(coilset.filepath, "frame")
    assert before == after


def test_subspace_dataframe_access():
    frameset = FrameSet(required=["x", "z"], additional=["Ic"], subspace=["Ic"])
    frameset.subframe.insert(2, range(2), Ic=0)
    frameset.sloc["Ic"] = 10
    assert frameset.sloc[:, ["Ic"]].squeeze().to_list() == [10, 10]


if __name__ == "__main__":
    pytest.main([__file__])
