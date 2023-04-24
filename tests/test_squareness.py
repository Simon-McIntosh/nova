
from itertools import product
import numpy as np
import pytest

from nova.geometry.separatrix import Quadrant


@pytest.mark.parametrize('minor_point,major_point',
                         product([(3, 0), (0.3, 0)],
                                 [(1.5, 1.2), (1.5, -3)]))
def test_quadrant(minor_point, major_point):
    quadrant = Quadrant(minor_point, major_point)
    if minor_point[0] > major_point[0]:
        if minor_point[1] < major_point[1]:
            assert quadrant.quadrant == 0
            return
        assert quadrant.quadrant == 3
        return
    if minor_point[1] < major_point[1]:
        assert quadrant.quadrant == 1
        return
    assert quadrant.quadrant == 2


@pytest.mark.parametrize('minor_point,major_point',
                         product([(3, 0), (0.4, -0.3)],
                                 [(0, 5), (2.4, -5)]))
def test_zero_squareness(minor_point, major_point):
    quadrant = Quadrant(minor_point, major_point)
    assert np.isclose(quadrant.squareness(quadrant.ellipse_point), 0)


@pytest.mark.parametrize('minor_point,major_point',
                         product([(3, -0.2), (0.4, 0.3)],
                                 [(0, 7), (-2.4, -5)]))
def test_unit_squareness(minor_point, major_point):
    quadrant = Quadrant(minor_point, major_point)
    square_point = quadrant.axis + np.array([quadrant.minor_radius,
                                             quadrant.major_radius])
    assert np.isclose(quadrant.squareness(square_point), 1)


if __name__ == '__main__':

    pytest.main([__file__])
