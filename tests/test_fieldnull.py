"""Agreement contract for host and traced field-null categorisation."""

from itertools import product

import numpy as np
import pytest

from nova.biot.fieldnull import DataNull
from nova.geometry import select as host_select
from nova.utilities.importmanager import skip_import
from nova.jax.config import enable_x64

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.jax import select as traced_select

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
    host_coef = host_select.quadratic_surface(x, z, psi)
    traced_coef = np.asarray(
        traced_select.quadratic_surface(jnp.array(x), jnp.array(z), jnp.array(psi))
    )
    assert np.allclose(psi, coefficient_matrix(x, z) @ host_coef)
    assert np.allclose(psi, coefficient_matrix(x, z) @ traced_coef)
    assert np.allclose(host_coef, traced_coef)


@pytest.mark.parametrize("null_type", [-1, 0, 1])
def test_quadratic_null_type(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    host_coef = host_select.quadratic_surface(x, z, psi)
    traced_coef = traced_select.quadratic_surface(
        jnp.array(x), jnp.array(z), jnp.array(psi)
    )
    assert int(host_select.null_type(host_coef)) == null_type
    assert int(traced_select.null_type(traced_coef)) == null_type


def test_quadratic_plane_surface():
    """A plane has a degenerate quadratic form reported as NaN by both routes."""
    x, z = meshgrid()
    psi = quadratic_surface(x, z, 2)
    host_coef = host_select.quadratic_surface(x, z, psi)
    traced_coef = traced_select.quadratic_surface(
        jnp.array(x), jnp.array(z), jnp.array(psi)
    )
    assert np.isnan(host_select.null_type(host_coef))
    assert np.isnan(float(traced_select.null_type(traced_coef)))


@pytest.mark.parametrize(
    "cross_term",
    [pytest.param(0.0, id="zero"), pytest.param(np.sqrt(5e-31), id="tiny-negative")],
)
def test_null_coordinate_determinant_floor_is_finite_and_sign_preserving(
    cross_term,
):
    """The shared determinant floor remains nonzero for degenerate quadratics."""
    coefficients = np.array([0.0, 0.0, 1.0, 1.0, cross_term, 1.0])
    determinant = -(cross_term**2)
    host_coordinate = np.asarray(host_select.null_coordinate(coefficients))
    traced_coordinate = np.asarray(
        traced_select.null_coordinate(jnp.asarray(coefficients))
    )

    assert abs(determinant) < 1e-30
    assert np.all(np.isfinite(host_coordinate))
    assert np.all(np.isfinite(traced_coordinate))
    assert np.allclose(host_coordinate, traced_coordinate)
    if determinant < 0:
        assert np.all(host_coordinate < 0)


@pytest.mark.parametrize(
    "null_type,coordinate",
    list(product([-1, 0, 1], [(0.8, 2.7), (2.2, 2.2), (-1, 5.2), (2, 2)])),
)
def test_quadratic_coordinate(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    host_coef = host_select.quadratic_surface(x, z, psi)
    traced_coef = traced_select.quadratic_surface(
        jnp.array(x), jnp.array(z), jnp.array(psi)
    )
    host_coordinate = np.asarray(host_select.null_coordinate(host_coef, (x, z)))
    traced_coordinate = np.asarray(traced_select.null_coordinate(traced_coef))
    assert np.allclose(host_coordinate, coordinate)
    assert np.allclose(traced_coordinate, coordinate)
    assert np.allclose(host_coordinate, traced_coordinate)


@pytest.mark.parametrize(
    "null_type,coordinate",
    list(product([-1, 0, 1], [(1, 2.7), (2.2, 2.2), (2, 3)])),
)
def test_subnull(null_type, coordinate):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type, *coordinate)
    host_result = host_select.subnull(x, z, psi)
    traced_result = np.asarray(
        traced_select.subnull(jnp.array(x), jnp.array(z), jnp.array(psi))
    )
    assert host_result.shape == (4,)
    assert traced_result.shape == (4,)
    assert np.allclose(host_result, traced_result)
    assert np.allclose(traced_result[:2], coordinate)
    assert np.isclose(traced_result[2], 0)
    assert int(traced_result[3]) == null_type


def test_host_null_store_consumes_flat_subnull_results():
    """Adaptive host filtering retains point, flux, and type from flat results."""
    stored = DataNull._unique([np.array([1.5, -0.2, 3.4, 0.0])])

    assert np.allclose(stored["points"], [[1.5, -0.2]])
    assert np.allclose(stored["psi"], [3.4])
    assert np.allclose(stored["null_type"], [0.0])


@pytest.mark.parametrize("polarity", [1, -1])
def test_wall_flux(polarity):
    """Both wall-loop routes locate and classify the extreme-flux panel."""
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    x = 1.0 + 0.4 * np.cos(theta)
    z = 0.4 * np.sin(theta)
    # flux extremum on the loop sits at the outboard vertex (theta = 0)
    psi = polarity * -((x - 1.4) ** 2 + z**2)
    host_result = host_select.wall_flux(x, z, psi, polarity)
    traced_result = np.asarray(
        traced_select.wall_flux(jnp.array(x), jnp.array(z), jnp.array(psi), polarity)
    )
    assert host_result.shape == (4,)
    assert traced_result.shape == (4,)
    assert np.allclose(host_result, traced_result, atol=1e-6)
    assert np.allclose(traced_result[:2], [1.4, 0.0], atol=1e-6)
    assert np.isfinite(traced_result[2])


def test_wall_flux_zero_polarity_is_nan():
    """A zero polarity has the same fixed-shape NaN sentinel on both routes."""
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x = 1.0 + 0.4 * np.cos(theta)
    z = 0.4 * np.sin(theta)
    psi = -((x - 1.4) ** 2 + z**2)
    host_result = host_select.wall_flux(x, z, psi, 0)
    traced_result = np.asarray(
        traced_select.wall_flux(jnp.array(x), jnp.array(z), jnp.array(psi), 0)
    )
    assert host_result.shape == (4,)
    assert traced_result.shape == (4,)
    assert np.all(np.isnan(host_result))
    assert np.all(np.isnan(traced_result))


if __name__ == "__main__":
    pytest.main([__file__])
