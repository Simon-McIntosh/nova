from itertools import product
import pytest

import numpy as np

from nova.biot.separatrix import LCFS, PlasmaShape, Separatrix


@pytest.mark.parametrize('radius,height', product([0, 2.5, 5], [-1.3, 0, 7.2]))
def test_profile_axis(radius, height):
    separatrix = Separatrix(radius, height).limiter(1, 1, 0)
    assert np.allclose(np.mean(separatrix.points, 0), (radius, height),
                       atol=1e-2)


@pytest.mark.parametrize('minor_radius,elongation,triangularity',
                         product([1, 5.2], [0.8, 1, 1.5], [-0.2, 0.2, 0.5]))
def test_limiter_profile(minor_radius, elongation, triangularity):
    profile = Separatrix(5.2, 0).limiter(minor_radius, elongation,
                                         triangularity)
    lcfs = LCFS(profile.points)
    assert np.allclose(np.array([minor_radius, elongation, triangularity]),
                       lcfs(profile.attrs), atol=1e-2)


def test_theta_upper():
    assert Separatrix(0, 0).theta_upper[-1] < np.pi


@pytest.mark.parametrize('minor_radius,elongation,triangularity',
                         product([1, 5.2], [1.5, 2.1, 2.5], [-0.2, 0.2, 0.5]))
def test_sn_profile(minor_radius, elongation, triangularity):
    profile = Separatrix(5.2, 0).single_null(minor_radius, elongation,
                                             triangularity)
    lcfs = LCFS(profile.points)
    assert np.allclose(np.array([minor_radius, elongation, triangularity]),
                       lcfs(profile.attrs), atol=1e-2)


def test_sn_x_point():
    profile = Separatrix(5.2, 3).single_null(2, 1.5, 0, x_point=(3.2, 0))
    assert np.allclose(profile.x_point, (3.2, 0))
    assert profile.radius == 3.2


def test_elongation():
    plasma = PlasmaShape(dict(elongation=2.3))
    assert plasma.elongation == 2.3


def test_upper_lower_elongation():
    plasma = PlasmaShape(dict(upper_elongation=3, lower_elongation=2))
    assert plasma.elongation == 2.5


def test_upper_elongation():
    plasma = PlasmaShape(dict(upper_elongation=3, lower_elongation=2))
    assert plasma.upper_elongation == 3


def test_upper_elongation_from_lower():
    plasma = PlasmaShape(dict(lower_elongation=2.5))
    assert plasma.upper_elongation == 2.5


def test_lower_elongation():
    plasma = PlasmaShape(dict(upper_elongation=3, lower_elongation=1.4))
    assert plasma.lower_elongation == 1.4


def test_lower_elongation_from_upper():
    plasma = PlasmaShape(dict(upper_elongation=2.4))
    assert plasma.lower_elongation == 2.4


def test_elongation_over_constraint_error():
    with pytest.raises(AssertionError):
        PlasmaShape(dict(elongation=2.4,
                         upper_elongation=3, lower_elongation=1.4))


def test_triangularty_over_constraint():
    plasma = PlasmaShape(dict(triangularity=2.5,
                         upper_triangularity=3, lower_triangularity=2))
    assert plasma.triangularity == 2.5
    assert plasma.upper_triangularity == 3
    assert plasma.lower_triangularity == 2


def test_lower_triangularity():
    plasma = PlasmaShape(dict(upper_triangularity=3))
    assert plasma.lower_triangularity == 3


if __name__ == '__main__':

    pytest.main([__file__])
