import os
import tempfile

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("ferritic"):
    import vedo

from nova.frame.coilset import CoilSet


def test_insert_box_subframe():
    coilset = CoilSet()
    box = vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3)
    coilset.ferritic.insert(box)
    assert np.isclose(coilset.subframe.volume[0], 6)


def test_insert_doublebox_subframe():
    coilset = CoilSet()
    box = [
        vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3),
        vedo.shapes.Box(pos=(7, 0, 1), length=1, width=2, height=3),
    ]
    coilset.ferritic.insert(box)
    assert np.isclose(coilset.subframe.volume.sum(), 12)


def test_insert_doublebox_frame_volume():
    coilset = CoilSet()
    box = [
        vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3),
        vedo.shapes.Box(pos=(7, 0, 1), length=1, width=2, height=3),
    ]
    coilset.ferritic.insert(box)
    assert np.isclose(coilset.frame.volume.iloc[0], 12)


def test_insert_doublebox_frame_centroid():
    coilset = CoilSet()
    box = [
        vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3),
        vedo.shapes.Box(pos=(7, 0, 1), length=1, width=2, height=3),
    ]
    coilset.ferritic.insert(box, label="Fi", offset=1)
    assert coilset.frame.loc["Fi1", ["x", "y", "z"]].values.tolist() == [6, 0, 0.5]


def test_insert_frame():
    coilset = CoilSet()
    box = [
        vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3),
        vedo.shapes.Box(pos=(7, 0, 1), length=1, width=2, height=3),
    ]
    coilset.ferritic.insert(box, part="Fi1", name="fi5", offset=0)
    coilset.ferritic.insert(box, part="Fi2", label="fi", offset=0)
    subframe = coilset.subframe.copy()
    coilset = CoilSet()
    coilset.ferritic.insert_frame(subframe)
    assert coilset.frame.index.to_list() == ["fi5", "fi6"]
    assert coilset.subframe.index.to_list() == ["fi5", "fi5_1", "fi6", "fi6_1"]


def test_insert_frame_fromfile():
    coilset = CoilSet()
    box = [
        vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3),
        vedo.shapes.Box(pos=(7, 0, 1), length=1, width=2, height=3),
    ]
    coilset.ferritic.insert(box, part="Fi1", label="fi", offset=1)
    coilset.ferritic.insert(box, part="Fi2", label="Fi", offset=0)
    subframe = coilset.subframe.copy()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        subframe.store(tmp.name)
        coilset = CoilSet()
        coilset.ferritic.insert(tmp.name)
    os.unlink(tmp.name)
    assert coilset.frame.index.to_list() == ["fi1", "Fi0"]


def test_select():
    coilset = CoilSet()
    box = vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3)
    coilset.ferritic.insert(box)
    assert coilset.subframe["ferritic"][0]
    assert not coilset.subframe["active"][0]
    assert coilset.subframe["passive"][0]
    assert coilset.subframe["fix"][0]
    assert not coilset.subframe["free"][0]
    assert not coilset.subframe["coil"][0]


if __name__ == "__main__":
    pytest.main([__file__])
