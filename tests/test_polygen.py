import pytest

import numpy as np

from nova.geometry.polygen import PolyGen


def test_disc_area():
    diameter = 5.2
    polygon = PolyGen("disc")(3, 7, diameter)
    assert np.isclose(polygon.area, np.pi * diameter**2 / 4, rtol=5e-3)


def test_bound_square_area():
    side = 3.2
    polygon = PolyGen("square")(-3, 2, side, 2 * side)
    assert np.isclose(polygon.area, side**2, rtol=1e-3)


def test_x_centroid():
    polygon = PolyGen("square")(-3, 2, 0.1)
    assert np.isclose(polygon.centroid.x, -3)


def test_y_centroid():
    polygon = PolyGen("disc")(-3, 2, 12)
    assert np.isclose(polygon.centroid.y, 2)


def test_ellipse_area():
    dx, dz = 5.2, 2.1
    polygon = PolyGen("ellipse")(3, 7, dx, dz)
    assert np.isclose(polygon.area, np.pi * dx * dz / 4, rtol=5e-3)


def test_rectangle_area():
    dx, dz = 5.2, 2.1
    polygon = PolyGen("rectangle")(-3, -7, dx, dz)
    assert np.isclose(polygon.area, dx * dz, rtol=1e-3)


def test_skin_area():
    d, dt = 5.2, 0.9
    polygon = PolyGen("skin")(0, 5, d, dt)
    area = np.pi * d**2 / 4 * (1 - (1 - dt) ** 2)
    assert np.isclose(polygon.area, area, rtol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__])
