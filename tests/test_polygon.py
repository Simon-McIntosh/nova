import pytest

import numpy as np
import pygeos
import shapely.geometry

from nova.electromagnetic.framespace import FrameSpace
from nova.geometry.polygon import Polygon


def test_polygon_pygeos():
    poly = pygeos.polygons([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon = Polygon(poly)
    assert polygon.poly == poly


def test_polygon_shapely():
    poly = shapely.geometry.Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon = Polygon(poly)
    assert polygon.poly == poly


def test_polygon_dict():
    polygon = Polygon(dict(rect=(1, 2, 0.3, 0.6)))
    assert np.isclose(polygon.poly.area, 0.3*0.6)


def test_polygon_multi_dict():
    polygon = Polygon(dict(rect=(1, 2, 0.3, 0.6), disc=[4, 3, 0.5]))
    assert np.isclose(polygon.poly.area, 0.3*0.6 + np.pi*0.25**2, rtol=1e-3)


def test_polygon_bbox():
    polygon = Polygon([0, 5, -1, 1])
    assert np.isclose(polygon.poly.area, 10)


def test_polygon_hash_eq():
    poly = pygeos.polygons([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon_a = Polygon(poly)
    polygon_b = Polygon(poly)
    assert hash(polygon_a) == hash(polygon_b)


def test_polygon_hash_neq():
    poly_a = pygeos.polygons([[0, 0], [0, 10], [10, 10], [10, 0]])
    poly_b = pygeos.polygons([[0, 0], [0, 10], [10, 10], [10, 0]])
    polygon_a = Polygon(poly_a)
    polygon_b = Polygon(poly_b)
    assert hash(polygon_a) != hash(polygon_b)


def test_polygon_loop():
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    radius = 5
    loop = np.array([radius * np.cos(theta), radius * np.sin(theta)]).T
    polygon = Polygon(loop)
    assert np.isclose(polygon.poly.area, np.pi*5**2, rtol=1e-2)


def test_fix_aspect():
    framespace = FrameSpace(required=['x', 'z', 'dl', 'dt'],
                            available=['section', 'poly'])
    framespace.insert(4, 6, 0.1, 0.5, section='sq')
    assert framespace.dx[0] == framespace.dz[0]


if __name__ == '__main__':
    #pytest.main([__file__])
    test_fix_aspect()
'''



def test_free_aspect():
    framespace = FrameSpace(required=['x', 'z', 'dl', 'dt'])
    framespace.insert(4, 6, 0.1, 0.5, section='r')
    assert framespace.dx[0] != framespace.dz[0]


def test_circular_cross_section():
    'check framespace area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 2.5, section='o')
    assert np.isclose(np.pi*2.5**2/4, framespace.area[0], rtol=1e-8)


def test_rectangular_cross_section():
    'check subcoil area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    framespace.insert(1.75, 0.5, 2.5, 1.5, section='rectangle')
    assert np.isclose(2.5*1.5, framespace.area[0], rtol=1e-8)


def test_skin_thickness_error():
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    with pytest.raises(ValueError):
        framespace.insert(1.75, 0.5, 2.5, 1.5, section='skin')


def test_skin_section_area():
    framespace = FrameSpace(Required=['x', 'z'])
    framespace.insert(1.75, 0.2, dl=0.5, dt=0.1, section='skin')
    assert np.isclose(framespace.area[0], framespace.poly[0].area, rtol=1e-1)


def test_invalid_cross_section():
    'check framespace area sum equals circle area'
    framespace = FrameSpace(Required=['x', 'z', 'dl', 'dt'])
    with pytest.raises(KeyError):
        framespace.insert(1.75, 0.5, 2.5, 2.5, section='P')


if __name__ == '__main__':

    pytest.main([__file__])
'''
