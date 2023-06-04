from itertools import product
import numpy as np
import pytest

from nova.geometry.plasmapoints import PlasmaPoints
from nova.geometry.plasmaprofile import PlasmaProfile
from nova.geometry.quadrant import Quadrant
from nova.geometry.separatrix import LCFS


@pytest.mark.parametrize("radius,height", product([0, 2.5, 5], [-1.3, 0, 7.2]))
def test_profile_axis(radius, height):
    separatrix = PlasmaProfile().limiter(radius, height, 1, 1, 0)
    assert np.allclose(np.mean(separatrix.points, 0), (radius, height), atol=1e-2)


@pytest.mark.parametrize(
    "minor_radius,elongation,triangularity",
    product([1, 5.2], [0.8, 1, 1.5], [-0.2, 0.2, 0.5]),
)
def test_limiter_profile(minor_radius, elongation, triangularity):
    profile = PlasmaProfile().limiter(5.2, 0, minor_radius, elongation, triangularity)
    shape = LCFS(profile.points)
    attrs = ["minor_radius", "elongation", "triangularity"]
    assert np.allclose(
        np.array([minor_radius, elongation, triangularity]), shape(attrs), atol=1e-2
    )


def test_theta_upper():
    assert PlasmaProfile().theta_upper[-1] < np.pi


@pytest.mark.parametrize(
    "minor_radius,elongation,triangularity",
    product([1, 5.2], [1.5, 2.1, 2.5], [-0.2, 0.2, 0.5]),
)
def test_sn_profile(minor_radius, elongation, triangularity):
    profile = PlasmaProfile().single_null(
        5.2, 0, minor_radius, elongation, triangularity
    )
    shape = LCFS(profile.points)
    attrs = ["minor_radius", "elongation", "triangularity"]
    assert np.allclose(
        np.array([minor_radius, elongation, triangularity]), shape(attrs), atol=1e-2
    )


def test_sn_x_point():
    profile = PlasmaProfile().single_null(5.2, 3, 2, 1.5, 0, x_point=(4.2, 0))
    assert np.allclose(profile.x_point, (4.2, 0))
    assert np.isclose(profile.geometric_radius, 5.2)


def test_elongation():
    plasma = PlasmaPoints(coef=dict(elongation=2.3))
    assert plasma.elongation == 2.3


def test_upper_elongation_lower():
    plasma = PlasmaPoints(coef=dict(elongation_upper=3, elongation_lower=2))
    assert plasma.elongation == 2.5


def test_elongation_upper():
    plasma = PlasmaPoints(coef=dict(elongation_upper=3, elongation_lower=2))
    assert plasma.elongation_upper == 3


def test_elongation_upper_from_lower():
    plasma = PlasmaPoints(coef=dict(elongation_lower=2.5))
    assert plasma.elongation_upper == 2.5


def test_elongation_lower():
    plasma = PlasmaPoints(coef=dict(elongation_upper=3, elongation_lower=1.4))
    assert plasma.elongation_lower == 1.4


def test_elongation_lower_from_upper():
    plasma = PlasmaPoints(coef=dict(elongation_upper=2.4))
    assert plasma.elongation_lower == 2.4


def test_elongation_over_constraint_error():
    with pytest.raises(AssertionError):
        PlasmaPoints(
            coef=dict(elongation=2.4, elongation_upper=3, elongation_lower=1.4)
        )


def test_triangularty_over_constraint():
    plasma = PlasmaPoints(
        coef=dict(triangularity=2.5, triangularity_upper=3, triangularity_lower=2)
    )
    assert plasma.triangularity == 2.5
    assert plasma.triangularity_upper == 3
    assert plasma.triangularity_lower == 2


def test_triangularity_lower():
    plasma = PlasmaPoints(coef=dict(triangularity_upper=3))
    assert plasma.triangularity_lower == 3


@pytest.mark.parametrize(
    "minor_point,major_point", product([(3, 0), (0.3, 0)], [(1.5, 1.2), (1.5, -3)])
)
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


@pytest.mark.parametrize(
    "minor_point,major_point", product([(3, 0), (0.4, -0.3)], [(0, 5), (2.4, -5)])
)
def test_zero_squareness(minor_point, major_point):
    quadrant = Quadrant(minor_point, major_point)
    assert np.isclose(quadrant.squareness(quadrant.ellipse_point), 0)


@pytest.mark.parametrize(
    "minor_point,major_point", product([(3, -0.2), (0.4, 0.3)], [(0, 7), (-2.4, -5)])
)
def test_unit_squareness(minor_point, major_point):
    quadrant = Quadrant(minor_point, major_point)
    square_point = quadrant.axis + np.array(
        [quadrant.minor_radius, quadrant.major_radius]
    )
    assert np.isclose(quadrant.squareness(square_point), 1)


@pytest.mark.parametrize("quadrant", range(4))
def test_quadrant_angles(quadrant):
    geometric_axis = (5.2, 0.2)
    minor_radius, elongation, triangularity = 0.5, 1.4, 0.3
    profile = PlasmaProfile(point_number=51).limiter(
        *geometric_axis, minor_radius, elongation, triangularity
    )
    shape = LCFS(profile.points)
    point = shape.quadrant_point(quadrant) - shape.quadrant_axis(quadrant)
    angle = np.arctan2(point[1], point[0])
    if angle < 0:
        angle += 2 * np.pi
    assert np.isclose(
        angle, shape.quadrant(quadrant).theta + quadrant * np.pi / 2, rtol=1e-3
    )


def test_square_circle():
    geometric_axis = (5.2, 0.2)
    minor_radius, elongation, triangularity = 0.5, 1, 0
    profile = PlasmaProfile(point_number=51).limiter(
        *geometric_axis, minor_radius, elongation, triangularity
    )
    shape = LCFS(profile.points)
    for i in range(4):
        assert np.isclose(shape.squareness(i), 0, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
