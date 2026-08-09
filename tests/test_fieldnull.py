"""Agreement contract for host and traced field-null categorisation."""

from itertools import product
import subprocess
import sys

import numpy as np
import pytest

from nova.biot.fieldnull import DataNull
from nova.geometry import select
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _explicit_dtype_policy():
    """Install explicit dtype support after test-module collection."""
    configure_dtypes()


def _double(value):
    """Return an explicit traced float64 array for physical coordinates."""
    return jnp.asarray(value, dtype=jnp.float64)


def test_host_fieldnull_consumers_import_without_jax():
    """Eager field-null, grid, and IMAS flux imports do not require JAX."""
    script = """
import importlib
import sys

sys.modules['jax'] = None

for name in ('nova.biot.fieldnull', 'nova.biot.grid', 'nova.imas.flux'):
    importlib.import_module(name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
    host_coef = select.host_quadratic_surface(x, z, psi)
    traced_coef = np.asarray(
        select.traced_quadratic_surface(_double(x), _double(z), _double(psi))
    )
    assert np.allclose(psi, coefficient_matrix(x, z) @ host_coef)
    assert np.allclose(psi, coefficient_matrix(x, z) @ traced_coef)
    assert np.allclose(host_coef, traced_coef)


@pytest.mark.parametrize("null_type", [-1, 0, 1])
def test_quadratic_null_type(null_type: int):
    x, z = meshgrid()
    psi = quadratic_surface(x, z, null_type)
    host_coef = select.host_quadratic_surface(x, z, psi)
    traced_coef = select.traced_quadratic_surface(_double(x), _double(z), _double(psi))
    assert int(select.null_type(host_coef)) == null_type
    assert int(select.null_type(traced_coef, array_namespace=jnp)) == null_type


def test_quadratic_plane_surface():
    """A plane has a degenerate quadratic form reported as NaN by both routes."""
    x, z = meshgrid()
    psi = quadratic_surface(x, z, 2)
    host_coef = select.host_quadratic_surface(x, z, psi)
    traced_coef = select.traced_quadratic_surface(_double(x), _double(z), _double(psi))
    assert np.isnan(select.null_type(host_coef))
    assert np.isnan(float(select.null_type(traced_coef, array_namespace=jnp)))


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
    host_coordinate = np.asarray(select.null_coordinate(coefficients))
    traced_coordinate = np.asarray(
        select.null_coordinate(_double(coefficients), array_namespace=jnp)
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
    host_coef = select.host_quadratic_surface(x, z, psi)
    traced_coef = select.traced_quadratic_surface(_double(x), _double(z), _double(psi))
    host_coordinate = np.asarray(select.null_coordinate(host_coef, (x, z)))
    traced_coordinate = np.asarray(
        select.null_coordinate(traced_coef, array_namespace=jnp)
    )
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
    host_result = select.host_subnull(x, z, psi)
    traced_result = np.asarray(
        select.traced_subnull(_double(x), _double(z), _double(psi))
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
    host_result = select.host_wall_flux(x, z, psi, polarity)
    traced_result = np.asarray(
        select.traced_wall_flux(_double(x), _double(z), _double(psi), polarity)
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
    host_result = select.host_wall_flux(x, z, psi, 0)
    traced_result = np.asarray(
        select.traced_wall_flux(_double(x), _double(z), _double(psi), 0)
    )
    assert host_result.shape == (4,)
    assert traced_result.shape == (4,)
    assert np.all(np.isnan(host_result))
    assert np.all(np.isnan(traced_result))


if __name__ == "__main__":
    pytest.main([__file__])
