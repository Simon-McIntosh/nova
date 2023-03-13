from itertools import product
import pytest

import numpy as np

from nova.biot.separatrix import LCFS, Miller


@pytest.mark.parametrize('radius,height', product([0, 2.5, 5], [-1.3, 0, 7.2]))
def test_profile_axis(radius, height):
    separatrix = Miller(radius, height).limiter(1, 1, 0)
    assert np.allclose(np.mean(separatrix.points, 0), (radius, height),
                       atol=1e-2)


@pytest.mark.parametrize('minor_radius,elongation,triangularity',
                         product([1, 5.2], [0.8, 1, 1.5], [-0.2, 0.2, 0.5]))
def test_limiter_profile(minor_radius, elongation, triangularity):
    profile = Miller(5.2, 0).limiter(minor_radius, elongation, triangularity)
    lcfs = LCFS(profile.points)
    assert np.allclose(np.array([minor_radius, elongation, triangularity]),
                       lcfs(profile.attrs), atol=1e-2)


def test_theta_upper():
    assert Miller(0, 0).theta_upper[-1] < np.pi


if __name__ == '__main__':

    pytest.main([__file__])
