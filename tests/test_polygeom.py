import pytest

import numpy as np

from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon


def test_bbox_length_thickness():
    polygon = Polygon([0, 5, -1, 1], "polyname")
    geom = PolyGeom(polygon)
    assert np.isclose(geom.length, 5)
    assert np.isclose(geom.thickness, 2)


def test_dict_length_thickness():
    polygon = Polygon(dict(skin=(3, 0, 0.5, 0.2)))
    geom = PolyGeom(polygon)
    assert np.isclose(geom.length, 0.5)
    assert np.isclose(geom.thickness, 0.2)


def test_loop_length_thickness():
    polygon = Polygon([[0, 0], [1, 0], [1, 1]])
    geom = PolyGeom(polygon)
    assert geom.length is None
    assert geom.thickness is None


def test_section_name():
    box = PolyGeom(Polygon([0, 5, -1, 1], name="box"))
    skin = PolyGeom(Polygon(dict(sk=(3, 0, 0.5, 0.2)), name="hoop"))
    loop = PolyGeom(Polygon([[0, 0], [1, 0], [1, 1]]))
    assert box.section == "rectangle"
    assert skin.section == "skin"
    assert skin.name == "hoop"
    assert loop.section == "polyloop"
    assert loop.name == "polyloop"


def test_rectangular_cross_section():
    polygon = Polygon(dict(r=(1.75, 0.5, 2.5, 1.5)))
    geom = PolyGeom(polygon)
    assert np.isclose(2.5 * 1.5, geom.area, rtol=1e-8)


def test_free_aspect():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))))
    assert geom.segment_delta.x != geom.segment_delta.z


def test_circular_cross_section():
    "check framespace area sum equals circle area"
    geom = PolyGeom(Polygon(dict(o=(1.75, 0.5, 2.5, 2.5))))
    assert np.isclose(np.pi * 2.5**2 / 4, geom.area, rtol=1e-8)


def test_ring_centroid():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), "ring")
    assert np.allclose(geom.centroid, [4, 0, 6])


def test_ring_delta():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), "ring")
    assert np.allclose(geom.segment_delta, [0.1, 2 * np.pi * 4, 0.5])


def test_other_delta():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), "other")
    assert np.allclose(geom.segment_delta, [0.1, 0, 0.5])


if __name__ == "__main__":
    pytest.main([__file__])
