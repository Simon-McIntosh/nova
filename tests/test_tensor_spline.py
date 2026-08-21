"""Checks for the traced global tensor cubic spline."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from nova.io import geqdsk
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline


configure_dtypes()

_TORAX_GEOMETRY_DIRECTORY = (
    Path(__import__("torax").__file__).parent / "data" / "third_party" / "geo"
)
_REFERENCE_TOLERANCE = 2.0e-10
_GRADIENT_TOLERANCE = 2.0e-6


def _analytic_map(radial, vertical):
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    return (
        np.sin(1.3 * radial_grid) * np.cos(0.7 * vertical_grid)
        + 0.08 * radial_grid**3 * vertical_grid
        - 0.13 * radial_grid * vertical_grid**2
    )


def _reference_errors(radial, vertical, values):
    spline = fit_tensor_spline(
        jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(values)
    )
    query_radial = np.linspace(radial[0] + 0.017, radial[-1] - 0.023, 11)
    query_vertical = np.linspace(vertical[0] + 0.019, vertical[-1] - 0.029, 11)
    reference = RectBivariateSpline(radial, vertical, values.T, kx=3, ky=3, s=0)
    actual = spline.evaluate(jnp.asarray(query_radial), jnp.asarray(query_vertical))
    derivative_orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    errors = []
    for result, (radial_order, vertical_order) in zip(
        actual, derivative_orders, strict=True
    ):
        expected = reference.ev(
            query_radial,
            query_vertical,
            dx=radial_order,
            dy=vertical_order,
        )
        errors.append(float(np.max(np.abs(np.asarray(result) - expected))))
    return errors


def _eqdsk_map(filename):
    data = geqdsk.read(str(_TORAX_GEOMETRY_DIRECTORY / filename))
    return np.asarray(data["x"]), np.asarray(data["z"]), np.asarray(data["psi"]).T


@pytest.mark.parametrize(
    "case",
    ("analytic", "iterhybrid_cocos17.eqdsk", "STEP_SPP_001_ECHD_ftop.eqdsk"),
)
def test_values_gradient_and_hessian_match_scipy(case):
    """Values and both derivative orders match the global SciPy referee."""
    if case == "analytic":
        radial = np.linspace(1.1, 3.8, 15) ** 1.03
        vertical = np.linspace(-1.4, 1.7, 13) ** 3 / 4.0
        vertical = np.sort(vertical)
        values = _analytic_map(radial, vertical)
    else:
        radial, vertical, values = _eqdsk_map(case)
    errors = _reference_errors(radial, vertical, values)
    print(f"{case} max_absolute_errors={errors}")
    assert max(errors) < _REFERENCE_TOLERANCE


def test_cell_coefficients_are_c2_across_every_interior_boundary():
    """Cell Bernstein blocks have continuous values and two derivatives."""
    radial = jnp.asarray(np.linspace(1.0, 3.6, 12) ** 1.04)
    vertical = jnp.asarray(np.sort(np.linspace(-1.2, 1.5, 10) ** 3 / 3.0))
    values = jnp.asarray(_analytic_map(np.asarray(radial), np.asarray(vertical)))
    spline = fit_tensor_spline(radial, vertical, values)
    coefficient = spline.cell_coefficients
    radial_spacing = jnp.diff(radial)
    vertical_spacing = jnp.diff(vertical)

    radial_value_jump = coefficient[:, :-1, :, 3] - coefficient[:, 1:, :, 0]
    radial_first_jump = (
        3.0
        * (coefficient[:, :-1, :, 3] - coefficient[:, :-1, :, 2])
        / radial_spacing[:-1][None, :, None]
        - 3.0
        * (coefficient[:, 1:, :, 1] - coefficient[:, 1:, :, 0])
        / radial_spacing[1:][None, :, None]
    )
    radial_second_jump = (
        6.0
        * (
            coefficient[:, :-1, :, 3]
            - 2.0 * coefficient[:, :-1, :, 2]
            + coefficient[:, :-1, :, 1]
        )
        / radial_spacing[:-1][None, :, None] ** 2
        - 6.0
        * (
            coefficient[:, 1:, :, 2]
            - 2.0 * coefficient[:, 1:, :, 1]
            + coefficient[:, 1:, :, 0]
        )
        / radial_spacing[1:][None, :, None] ** 2
    )

    vertical_value_jump = coefficient[:-1, :, 3, :] - coefficient[1:, :, 0, :]
    vertical_first_jump = (
        3.0
        * (coefficient[:-1, :, 3, :] - coefficient[:-1, :, 2, :])
        / vertical_spacing[:-1, None, None]
        - 3.0
        * (coefficient[1:, :, 1, :] - coefficient[1:, :, 0, :])
        / vertical_spacing[1:, None, None]
    )
    vertical_second_jump = (
        6.0
        * (
            coefficient[:-1, :, 3, :]
            - 2.0 * coefficient[:-1, :, 2, :]
            + coefficient[:-1, :, 1, :]
        )
        / vertical_spacing[:-1, None, None] ** 2
        - 6.0
        * (
            coefficient[1:, :, 2, :]
            - 2.0 * coefficient[1:, :, 1, :]
            + coefficient[1:, :, 0, :]
        )
        / vertical_spacing[1:, None, None] ** 2
    )
    jumps = (
        radial_value_jump,
        radial_first_jump,
        radial_second_jump,
        vertical_value_jump,
        vertical_first_jump,
        vertical_second_jump,
    )
    maxima = [float(jnp.max(jnp.abs(jump))) for jump in jumps]
    print(f"C2_boundary_max_absolute_jumps={maxima}")
    assert max(maxima) < _REFERENCE_TOLERANCE


def test_point_gradient_with_respect_to_map_matches_finite_difference():
    """Reverse-mode map sensitivities match central finite differences."""
    radial = jnp.linspace(1.2, 3.7, 9)
    vertical = jnp.linspace(-1.1, 1.4, 8)
    values = jnp.asarray(_analytic_map(np.asarray(radial), np.asarray(vertical)))
    point = (jnp.asarray(2.17), jnp.asarray(0.23))

    def point_value(sampled_values):
        return fit_tensor_spline(radial, vertical, sampled_values)(*point)

    gradient = jax.grad(point_value)(values)
    direction = jnp.reshape(jnp.sin(jnp.arange(values.size) + 0.4), values.shape)
    direction = direction / jnp.linalg.norm(direction)
    step = 2.0e-5
    finite_difference = (
        point_value(values + step * direction) - point_value(values - step * direction)
    ) / (2.0 * step)
    automatic = jnp.vdot(gradient, direction)
    relative_error = float(
        jnp.abs(automatic - finite_difference)
        / jnp.maximum(jnp.abs(finite_difference), 1.0e-14)
    )
    print(f"map_gradient_directional_relative_error={relative_error:.9e}")
    assert relative_error < _GRADIENT_TOLERANCE


def test_jit_eager_and_vmap_keep_fixed_shapes():
    """Compilation and a two-map vmap preserve results and array shapes."""
    radial = jnp.linspace(1.0, 3.0, 8)
    vertical = jnp.linspace(-1.0, 1.0, 7)
    values = jnp.asarray(_analytic_map(np.asarray(radial), np.asarray(vertical)))
    query_radial = jnp.asarray([1.19, 1.83, 2.61])
    query_vertical = jnp.asarray([-0.71, 0.14, 0.68])

    def fit_and_evaluate(sampled_values):
        spline = fit_tensor_spline(radial, vertical, sampled_values)
        return spline.cell_coefficients, spline.evaluate(
            query_radial, query_vertical
        ).value

    eager_coefficients, eager_values = fit_and_evaluate(values)
    compiled_coefficients, compiled_values = jax.jit(fit_and_evaluate)(values)
    batch = jnp.stack((values, 1.7 * values - 0.2))
    batch_coefficients, batch_values = jax.jit(jax.vmap(fit_and_evaluate))(batch)

    expected_shape = (vertical.size - 1, radial.size - 1, 4, 4)
    assert eager_coefficients.shape == expected_shape
    assert compiled_coefficients.shape == expected_shape
    assert batch_coefficients.shape == (2,) + expected_shape
    assert eager_values.shape == query_radial.shape
    assert batch_values.shape == (2,) + query_radial.shape
    np.testing.assert_allclose(
        compiled_coefficients, eager_coefficients, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(compiled_values, eager_values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(batch_values[0], eager_values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        batch_values[1], 1.7 * eager_values - 0.2, rtol=0, atol=1e-12
    )
