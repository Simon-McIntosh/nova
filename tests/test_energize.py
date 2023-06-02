import pytest

from nova.frame.framelink import FrameLink


def test_in_energize():
    framelink = FrameLink(Required=["x", "z"], Additional=["It"])
    assert framelink.hascol("energize", "It")


def test_set_loc_nturn():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=1)
    framelink.loc[:, "It"] = 6.6
    framelink.loc[:, "nturn"] = 2.2
    assert framelink.loc[:, "It"].to_list() == [14.52, 14.52]


def test_set_item_Ic():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=0.5)
    framelink["Ic"] = 6.6
    assert framelink["It"].to_list() == [3.3, 3.3]


def test_set_item_It():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=0.25)
    framelink["It"] = 6.6
    assert framelink["Ic"].to_list() == [26.4, 26.4]


def test_set_item_nturn():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=1)
    framelink["It"] = 6.6
    framelink["nturn"] = 2.2
    assert framelink["It"].to_list() == [14.52, 14.52]


def test_set_attr_Ic():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=0.5)
    framelink.Ic = 6.6
    assert framelink.It.to_list() == [3.3, 3.3]


def test_set_attr_It():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=0.25)
    framelink.It = 6.6
    assert framelink.Ic.to_list() == [26.4, 26.4]


def test_set_attr_nturn():
    framelink = FrameLink(Required=["x", "z"], Additional=["Ic"])
    framelink.insert(0.5, [6, 8.3], nturn=1)
    framelink.It = 6.6
    framelink.nturn = 2.2
    assert framelink.It.to_list() == [14.52, 14.52]


if __name__ == "__main__":
    pytest.main([__file__])
