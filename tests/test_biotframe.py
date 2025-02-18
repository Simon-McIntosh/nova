import pytest

import numpy as np

from nova.biot.biotframe import BiotFrame, Source
from nova.biot.crosssection import CrossSection
from nova.frame.framelink import FrameLink
from nova.frame.dataframe import DataFrame


def test_turnturn_square():
    biotframe = BiotFrame()
    biotframe.insert(range(2), 0.1, section="sq")
    assert np.isclose(
        biotframe.turnturn, biotframe.biotsection.section_factor["square"]
    ).all()


def test_turnturn_disc():
    framelink = FrameLink(required=["x"])
    framelink.insert(range(3), delta=-2, section="o")
    biotframe = BiotFrame(framelink)
    assert np.isclose(
        biotframe.turnturn, biotframe.biotsection.section_factor["disc"]
    ).all()


def test_turnturn_skin():
    framelink = FrameLink(required=["x"])
    framelink.insert(range(3), delta=-2, section="sk")
    biotframe = BiotFrame(framelink)
    assert np.isclose(
        biotframe.turnturn, biotframe.biotsection.section_factor["skin"]
    ).all()


def test_framelink_section():
    framelink = FrameLink(required=["x"])
    framelink.insert(range(4), delta=-2, section=["o", "sq", "hex", "skin"])
    biotframe = BiotFrame(framelink)
    assert biotframe.section.to_list() == ["disc", "square", "hexagon", "skin"]


def test_framelink_insert_keyerror():
    framelink = FrameLink(required=["x"])
    framelink.insert(range(4), delta=-2, section=["o", "sq", "hex", "skin"])
    biotframe = BiotFrame()
    with pytest.raises(KeyError):
        biotframe.insert(framelink)


def test_section_keyerror():
    frame = DataFrame({"section": ["hexagon", "random"]})
    section = CrossSection(frame)
    with pytest.raises(KeyError):
        section.initialize()


def test_target_shape():
    biotframe = BiotFrame(range(3), range(3))
    biotframe.biotshape.set_target(12)
    assert biotframe.biotshape.source == 3
    assert biotframe.biotshape.target == 12
    assert biotframe.biotshape.region == "source"


def test_source_shape():
    biotframe = BiotFrame(range(2), range(2))
    biotframe.biotshape.set_source(6)
    assert biotframe.biotshape.source == 6
    assert biotframe.biotshape.target == 2
    assert biotframe.biotshape.region == "target"


def test_region_not_set_errot():
    biotframe = BiotFrame(range(2), range(2))
    with pytest.raises(IndexError):
        biotframe("x")


def test_matrix_shape():
    biotframe = BiotFrame({"x": range(2), "z": range(2)})
    biotframe.biotshape.set_target(6)
    assert biotframe("x").shape == (6, 2)


def test_biotreduce_indices_link():
    source = Source(
        dict(x=range(8), z=1),
        label="Coil",
        link=["", "Coil3", "", "", "", "Coil4", "", ""],
    )
    assert source.biotreduce.indices == [0, 1, 2, 3, 4, 6, 7]
    assert source.biotreduce.link == {3: [1, 1.0]}


def test_stack():
    biotframe = BiotFrame({"x": range(5), "z": range(5)})
    biotframe.biotshape.set_target(3)
    assert biotframe.stack("x", "z").shape == (3, 5, 2)


if __name__ == "__main__":
    pytest.main([__file__])
