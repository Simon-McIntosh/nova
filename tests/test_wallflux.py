from itertools import product

import numpy as np
import pytest
from shapely.geometry import Point

from nova.biot.limiter import Limiter
from nova.geometry import select
from nova.graphics.plot import Plot


def meshwall():
    xy = Point(0, 0).buffer(5, 2).boundary.xy
    return np.array(xy[0]), np.array(xy[1])


def coefficient_matrix(w):
    return np.c_[w**2, w, np.ones_like(w)]


def quadratic_wall(w, null_type: int, wo=2):
    if null_type == -1:  # minimum
        return (w - wo) ** 2
    if null_type == 1:  # maximum
        return -((w - wo) ** 2)
    if null_type == 2:  # plane
        return w + 1


def quadratic_surface(x, z, null_type: int, xo=2, zo=2):
    if null_type == -1:  # minimum
        return (x - xo) ** 2 + (z - zo) ** 2
    if null_type == 1:  # maximum
        return -((x - xo) ** 2) + -((z - zo) ** 2)


def test_length_2d():
    x = np.linspace(0, np.cos(np.pi / 4), 10, dtype=float)
    z = np.linspace(0, np.sin(np.pi / 4), 10, dtype=float)
    w = select.length_2d(x, z)
    assert len(w) == 10
    assert np.isclose(w[-1], 1)


@pytest.mark.parametrize("null_type", [-1, 1, 2])
def test_quadratic_coefficents(null_type: int):
    w = select.length_2d(*meshwall())
    psi = quadratic_wall(w, null_type)
    coef = select.host_quadratic_wall(w, psi)
    assert np.allclose(psi, coefficient_matrix(w) @ coef)


@pytest.mark.parametrize(
    "null_type,null_position", list(product([-1, 1], [0.1, 3.3, 11, 26.7]))
)
def test_wall_length(null_type, null_position):
    w_coordinate = select.length_2d(*meshwall())
    psi = quadratic_wall(w_coordinate, null_type, null_position)
    coef = select.host_quadratic_wall(w_coordinate, psi)
    assert np.allclose(select.wall_length(coef), null_position)


@pytest.mark.parametrize(
    "null_type,null_position", list(product([-1, 1], [0, 0.1, 3.3, 11, 26.7]))
)
def test_wall_coordinate(null_type, null_position, plot=False):
    x_cluster, z_cluster = meshwall()
    w_cluster = select.length_2d(x_cluster, z_cluster)
    psi = quadratic_wall(w_cluster, null_type, null_position)
    _null_coordinate = (
        np.interp(null_position, w_cluster, x_cluster),
        np.interp(null_position, w_cluster, z_cluster),
    )
    coef = select.host_quadratic_wall(w_cluster, psi)
    w_coordinate = select.wall_length(coef)
    null_coordinate = select.wall_coordinate(
        w_coordinate, x_cluster, z_cluster, w_cluster
    )
    if plot:
        plt = Plot().plt
        plt.plot(x_cluster, z_cluster, "-o")
        plt.plot(*_null_coordinate, "d")
        plt.plot(*null_coordinate, "X")
        plt.axis("equal")
    assert np.allclose(null_coordinate, _null_coordinate)


@pytest.mark.parametrize(
    "null_type,null_position,null_flux",
    list(product([-1, 1], [0, 5.5, 30.6], [0, 3.76, -12.3])),
)
def test_wall_flux(null_type, null_position, null_flux):
    x_cluster, z_cluster = meshwall()
    w_cluster = select.length_2d(x_cluster, z_cluster)
    null_coordinate = (
        np.interp(null_position, w_cluster, x_cluster),
        np.interp(null_position, w_cluster, z_cluster),
    )
    psi_cluster = quadratic_surface(x_cluster, z_cluster, null_type, *null_coordinate)
    psi_cluster += null_flux
    result = select.host_wall_flux(x_cluster, z_cluster, psi_cluster, null_type)
    assert result.shape == (4,)
    assert result[3] == null_type
    assert np.isclose(result[2], null_flux, atol=0.01 * np.max(abs(psi_cluster)))


def test_wall_flux_zero_polarity_returns_four_nan_values():
    """A missing wall-limit point has the same fixed-shape sentinel as traced code."""
    x_wall, z_wall = meshwall()
    result = select.host_wall_flux(x_wall, z_wall, np.zeros_like(x_wall), 0)

    assert result.shape == (4,)
    assert np.all(np.isnan(result))


def test_limiter_zero_current_publishes_nan_wall_state():
    """Zero plasma current publishes no wall point or wall flux."""
    x_wall, z_wall = meshwall()
    limiter = Limiter()
    limiter.array.update(x=x_wall, z=z_wall)

    limiter.update_wall(np.zeros_like(x_wall), 0)

    assert np.isnan(limiter.w_psi)
    assert np.all(np.isnan(limiter.w_point))


if __name__ == "__main__":
    pytest.main([__file__])
