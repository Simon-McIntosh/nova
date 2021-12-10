import pytest

import numpy as np
import shapely.geometry

from nova.geometry.polygon import Polygon


def test_polygon_shapely():
    poly = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon = Polygon(poly)
    assert polygon.poly == poly


def test_polygon_dict():
    polygon = Polygon(dict(rect=(1, 2, 0.3, 0.6)))
    assert np.isclose(polygon.poly.area, 0.3*0.6)


def test_polygon_square():
    polygon = Polygon(dict(sq=(1, 2, 0.3, 0.6)))
    assert np.isclose(polygon.poly.area, 0.3**2)


def test_polygon_square_max_min():
    polygon = Polygon(dict(sq=(1, 2, 0.6, 0.2)))
    assert np.isclose(polygon.poly.area, 0.2**2)


def test_polygon_disc():
    polygon = Polygon(dict(o=(1, 2, 0.2)))
    assert np.isclose(polygon.poly.area, np.pi*0.1**2, rtol=1e-3)


def test_polygon_disc_max_min():
    polygon = Polygon(dict(o=(1, 2, 0.6, 0.2)))
    assert np.isclose(polygon.poly.area, np.pi*0.1**2, rtol=1e-3)


def test_polygon_multi_dict():
    polygon = Polygon(dict(rect=(1, 2, 0.3, 0.6), disc=[4, 3, 0.5]))
    assert np.isclose(polygon.poly.area, 0.3*0.6 + np.pi*0.25**2, rtol=1e-3)


def test_polygon_bbox():
    polygon = Polygon([0, 5, -1, 1])
    assert np.isclose(polygon.poly.area, 10)


def test_polygon_eq():
    poly = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon_a = Polygon(poly)
    polygon_b = Polygon(poly)
    assert polygon_a.poly == polygon_b.poly


def test_polygon_eq_dual_source():
    poly_a = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    poly_b = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon_a = Polygon(poly_a)
    polygon_b = Polygon(poly_b)
    assert polygon_a.poly == polygon_b.poly


def test_polygon_ne():
    poly_a = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    poly_b = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 2]])
    polygon_a = Polygon(poly_a)
    polygon_b = Polygon(poly_b)
    assert polygon_a.poly != polygon_b.poly


def test_polygon_loop():
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    radius = 5
    loop = np.array([radius * np.cos(theta), radius * np.sin(theta)]).T
    polygon = Polygon(loop)
    assert np.isclose(polygon.poly.area, np.pi*5**2, rtol=1e-2)


def test_polygon_names():
    polybox = Polygon([0, 5, -1, 1])
    polyname = Polygon([0, 5, -1, 1], 'bbox')
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    loop = np.array([np.cos(theta), np.sin(theta)]).T
    polyloop = Polygon(loop)
    assert polybox.name == 'rectangle'
    assert polyname.name == 'bbox'
    assert polyname.metadata['section'] == 'rectangle'
    assert polyloop.name == 'polyloop'


def test_invalid_cross_section():
    with pytest.raises(KeyError):
        Polygon(dict(P=(1, 2, 0.3, 0.6)))


def test_skin_thickness_error():
    with pytest.raises(ValueError):
        Polygon(dict(skin=(1.75, 0.5, 2.5, 1.5)))


def test_rectangular_cross_section():
    polygon = Polygon(dict(r=(1.75, 0.5, 2.5, 1.5)))
    assert np.isclose(2.5*1.5, polygon.poly.area, rtol=1e-8)


def test_free_aspect():
    polygon = Polygon(dict(rec=(4, 6, 0.1, 0.5)))
    assert polygon.width != polygon.height


if __name__ == '__main__':
    pytest.main([__file__])
