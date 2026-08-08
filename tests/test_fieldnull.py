"""Field-null categorisation on the jax select/null stack.

The serial numba field-null finder is superseded by the vmap-safe jax stack
(``nova.jax.select`` and ``nova.jax.null``); these tests exercise the jax
implementations of quadratic-surface fitting, null typing, sub-grid null
interpolation on both the 2D grid stencil and the 1D wall loop.
"""

from itertools import product

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import
from nova.jax.config import enable_x64

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.jax import select

    enable_x64()


def meshgrid():
    """Return a flattened 3x3 unit-spaced grid."""
    x, z = np.meshgrid(
        np.arange(1, 4, 1, dtype=float), np.arange(1, 4, 1, dtype=float), indexing="ij"
    )
    return x.flatten(), z.flatten()


def coefficient_matrix(x, z):
    """Return the quadratic design matrix used to recover psi."""
    return np.c_[x**2, z**2, x, z, x * z, np.ones_like(x)]


def quadratic_surface(x, z, null_type: int, xo=2, zo=2):
    """Return an analytic quadratic surface of the requested null type."""
    if null_type == 0:  # saddle
        return (x - xo) ** 2 + -((z - zo) ** 2)
    if null_type == -1:  # minimum
        return (x - xo) ** 2 + (z - zo) ** 2
    if null_type == 1:  # maximum
        return -((x - xo) ** 2) + -((z - zo) ** 2)
    if null_type == 2:  # plane
        return x + z + 1


@pytest.mark.parametrize("null_type", [-1, 0, 1, 2])
def test_quadratic_coefficents(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    coef = np.asarray(
        select.quadratic_surface(jnp.array(x), jnp.array(z), jnp.array(psi))
    )
    assert np.allclose(psi, coefficient_matrix(x, z) @ coef)


@pytest.mark.parametrize("null_type", [-1, 0, 1])
def test_quadratic_null_type(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    coef = select.quadratic_surface(jnp.array(x), jnp.array(z), jnp.array(psi))
    assert int(select.null_type(coef)) == null_type


def test_quadratic_plane_surface():
    """A plane surface has a degenerate second-order form (null_type is nan)."""
    x, z = meshgrid()
    psi = quadratic_surface(x, z, 2)
    coef = select.quadratic_surface(jnp.array(x), jnp.array(z), jnp.array(psi))
    assert np.isnan(float(select.null_type(coef)))


@pytest.mark.parametrize(
    "null_type,coordinate",
    product([-1, 0, 1], [(0.8, 2.7), (2.2, 2.2), (-1, 5.2), (2, 2)]),
)
def test_quadratic_coordinate(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    coef = select.quadratic_surface(jnp.array(x), jnp.array(z), jnp.array(psi))
    null_coordinate = np.asarray(select.null_coordinate(coef))
    assert np.allclose(null_coordinate, coordinate)


@pytest.mark.parametrize(
    "null_type,coordinate", product([-1, 0, 1], [(1, 2.7), (2.2, 2.2), (2, 3)])
)
def test_subnull(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    cluster = jnp.array(np.c_[x, z, psi].T)  # (3, N): [x; z; psi]
    null_coords_psi_type = np.asarray(select.subnull(cluster))
    assert np.allclose(null_coords_psi_type[:2], coordinate)
    assert np.isclose(null_coords_psi_type[2], 0)
    assert int(null_coords_psi_type[3]) == null_type


@pytest.mark.parametrize("polarity", [1, -1])
def test_wall_flux(polarity):
    """The 1D wall-loop finder locates the extreme-flux panel on the loop."""
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    x = 1.0 + 0.4 * np.cos(theta)
    z = 0.4 * np.sin(theta)
    # flux extremum on the loop sits at the outboard vertex (theta = 0)
    psi = polarity * -((x - 1.4) ** 2 + z**2)
    out = np.asarray(
        select.wall_flux(jnp.array(x), jnp.array(z), jnp.array(psi), polarity)
    )
    assert np.allclose(out[:2], [1.4, 0.0], atol=1e-6)
    assert np.isfinite(out[2])


def test_wall_flux_zero_polarity_is_nan():
    """A zero polarity marks the wall null as undefined (all nan)."""
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x = 1.0 + 0.4 * np.cos(theta)
    z = 0.4 * np.sin(theta)
    psi = -((x - 1.4) ** 2 + z**2)
    out = np.asarray(select.wall_flux(jnp.array(x), jnp.array(z), jnp.array(psi), 0))
    assert np.all(np.isnan(out))


if __name__ == "__main__":
    pytest.main([__file__])
