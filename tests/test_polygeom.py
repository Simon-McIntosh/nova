import pytest

import numpy as np

from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon


def test_bbox_length_thickness():
    polygon = Polygon([0, 5, -1, 1])
    geom = PolyGeom(polygon)
    assert np.isclose(geom.length, 5)
    assert np.isclose(geom.thickness, 2)


def test_polyname():
    polygon = Polygon([0, 5, -1, 1], "polygon_name")
    assert polygon.name == "polygon_name"


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


def test_skin_cross_section():
    "check framespace area sum equals circle area"
    inner_width = 0.04
    outer_width = 0.05
    factor = 1 - inner_width / outer_width
    geom = PolyGeom(Polygon({"sk": (1.75, 0.5, outer_width, factor)}))
    assert np.isclose(
        np.pi * (outer_width**2 - inner_width**2) / 4, geom.area, rtol=1e-8
    )


def test_box_cross_section():
    "check framespace area sum equals circle area"
    inner_width = 0.04
    outer_width = 0.05
    factor = 1 - inner_width / outer_width
    geom = PolyGeom(Polygon({"box": (1.75, 0.5, outer_width, factor)}))
    assert np.isclose(outer_width**2 - inner_width**2, geom.area, rtol=1e-8)


def test_circle_centroid():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), segment="circle")
    assert np.allclose(geom.centroid, [4, 0, 6])


def test_circle_delta():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), segment="circle")
    assert np.allclose(geom.segment_delta, [0.1, 2 * np.pi * 4, 0.5])


def test_other_delta():
    geom = PolyGeom(Polygon(dict(rec=(4, 6, 0.1, 0.5))), segment="other")
    assert np.allclose(geom.segment_delta, [0.1, 0, 0.5])


if __name__ == "__main__":
    pytest.main([__file__])
