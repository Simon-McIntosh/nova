"""Checks for the conditioned curved-interface split spline."""

import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.hex_cell_field_feasibility import (
    AXIS,
    LOBE_OFFSET,
    SADDLE,
    _base_flux,
    hex_lattice,
    solovev_flux,
)
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
import nova.linalg.split_spline as split_spline_module
from nova.linalg.split_spline import fit_split_spline


configure_dtypes()

_BASELINE = Path(__file__).parents[1] / "docs/figures/hex-cell-single-grid/metrics.json"
_CROSS_BACKEND_SOLVE_ATOL = 1.0e-9


def _half_offset_lattice(vertical_size=19, radial_size=21):
    radial = jnp.linspace(-1.1, 1.1, radial_size)
    vertical = jnp.linspace(-0.9, 0.9, vertical_size)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    offset = 0.5 * (radial[1] - radial[0]) * (jnp.arange(vertical_size) % 2)
    return radial_grid + offset[:, None], vertical_grid


def _level_set(radial, vertical):
    return (radial / 0.78) ** 2 + (vertical / 0.61) ** 2 - 1.0


def _manufactured_field(radial, vertical):
    level = _level_set(radial, vertical)
    interior = (
        0.4
        + 0.3 * radial
        - 0.2 * vertical
        + 0.17 * radial * vertical
        + 0.08 * radial**2
    )
    correction = 0.7 + 0.11 * radial - 0.09 * vertical
    return interior + correction * jnp.maximum(level, 0.0) ** 2


def _hessian(evaluation):
    return jnp.stack(
        (
            jnp.stack(
                (evaluation.radial_second_derivative, evaluation.mixed_derivative),
                axis=-1,
            ),
            jnp.stack(
                (evaluation.mixed_derivative, evaluation.vertical_second_derivative),
                axis=-1,
            ),
        ),
        axis=-2,
    )


def test_regions_are_c2_with_c1_interface_and_positive_curvature_jump():
    """Regional Hessians stay smooth while the curved interface is exactly C1."""
    radial, vertical = _half_offset_lattice()
    level = _level_set(radial, vertical)
    values = _manufactured_field(radial, vertical)
    spline = fit_split_spline(radial, vertical, values, level)

    angle = jnp.linspace(0.18, 2.0 * jnp.pi - 0.18, 31)
    interface = jnp.stack((0.78 * jnp.cos(angle), 0.61 * jnp.sin(angle)), axis=-1)
    level_gradient = jnp.stack(
        (2.0 * interface[:, 0] / 0.78**2, 2.0 * interface[:, 1] / 0.61**2),
        axis=-1,
    )
    normal = level_gradient / jnp.linalg.norm(level_gradient, axis=-1, keepdims=True)
    displacement = 2.0e-6
    interior_point = interface - displacement * normal
    exterior_point = interface + displacement * normal
    interior_eval = spline.evaluate(interior_point[:, 0], interior_point[:, 1])
    exterior_eval = spline.evaluate(exterior_point[:, 0], exterior_point[:, 1])
    interior_gradient = jnp.stack(
        (interior_eval.radial_derivative, interior_eval.vertical_derivative), axis=-1
    )
    exterior_gradient = jnp.stack(
        (exterior_eval.radial_derivative, exterior_eval.vertical_derivative), axis=-1
    )
    interior_hessian = _hessian(interior_eval)
    exterior_hessian = _hessian(exterior_eval)
    interior_interface_value = interior_eval.value + displacement * jnp.sum(
        interior_gradient * normal, axis=-1
    )
    exterior_interface_value = exterior_eval.value - displacement * jnp.sum(
        exterior_gradient * normal, axis=-1
    )
    interior_interface_gradient = interior_gradient + displacement * jnp.einsum(
        "...ij,...j->...i", interior_hessian, normal
    )
    exterior_interface_gradient = exterior_gradient - displacement * jnp.einsum(
        "...ij,...j->...i", exterior_hessian, normal
    )
    value_gap = jnp.max(jnp.abs(exterior_interface_value - interior_interface_value))
    gradient_gap = jnp.max(
        jnp.abs(exterior_interface_gradient - interior_interface_gradient)
    )
    interior_normal_curvature = jnp.einsum(
        "...i,...ij,...j->...", normal, interior_hessian, normal
    )
    exterior_normal_curvature = jnp.einsum(
        "...i,...ij,...j->...", normal, exterior_hessian, normal
    )
    curvature_jump = exterior_normal_curvature - interior_normal_curvature

    probe = jnp.asarray(((-0.52, 0.08), (0.92, 0.18)))
    direction = jnp.asarray((0.37, -0.61))
    nearby = probe + 1.0e-6 * direction
    hessian_change = jnp.max(
        jnp.abs(
            _hessian(spline.evaluate(nearby[:, 0], nearby[:, 1]))
            - _hessian(spline.evaluate(probe[:, 0], probe[:, 1]))
        )
    )

    print(
        "interface_max_gaps="
        f"value:{float(value_gap):.6e},gradient:{float(gradient_gap):.6e} "
        f"minimum_normal_curvature_jump={float(jnp.min(curvature_jump)):.6e} "
        f"regional_hessian_change={float(hessian_change):.6e}"
    )
    assert value_gap < 1.0e-8
    assert gradient_gap < 1.0e-8
    assert jnp.min(curvature_jump) > 1.0
    assert hessian_change < 2.0e-5


def test_fixed_shape_fit_and_evaluation_have_jit_vmap_parity_and_receipts():
    """Compiled solves agree across backends while padding stays exactly zero.

    CPU and CUDA linear-solve reductions need not be bitwise identical. The
    cross-backend bound covers their sub-nanolevel reduction difference while
    remaining far below the field-accuracy thresholds; shapes, execution masks,
    and inactive values remain exact invariants.
    """
    radial, vertical = _half_offset_lattice(13, 15)
    level = _level_set(radial, vertical)
    values = _manufactured_field(radial, vertical)
    query_radial = jnp.asarray((-0.41, 0.12, 0.87, 0.0))
    query_vertical = jnp.asarray((0.11, -0.22, 0.19, 0.0))
    query_valid = jnp.asarray((True, True, True, False))

    def fit_and_evaluate(sampled_values, execute):
        spline = fit_split_spline(
            radial, vertical, sampled_values, level, execute=execute
        )
        evaluated = spline.evaluate_fixed(
            query_radial, query_vertical, query_valid & execute
        )
        return (
            spline.coefficients,
            spline.fit_executed,
            spline.sample_count,
            evaluated.value,
            evaluated.executed,
        )

    eager = fit_and_evaluate(values, jnp.asarray(True))
    compiled = jax.jit(fit_and_evaluate)
    compiled_active = compiled(values, jnp.asarray(True))
    compiled(values, jnp.asarray(True))
    batch_values = jnp.stack((values, 1.3 * values - 0.2))
    batched = jax.jit(jax.vmap(fit_and_evaluate))(
        batch_values, jnp.asarray((True, False))
    )

    assert compiled._cache_size() == 1
    np.testing.assert_allclose(
        compiled_active[0], eager[0], rtol=0, atol=_CROSS_BACKEND_SOLVE_ATOL
    )
    np.testing.assert_allclose(
        compiled_active[3], eager[3], rtol=0, atol=_CROSS_BACKEND_SOLVE_ATOL
    )
    assert batched[0].shape == (2, 2, 5, 5)
    assert batched[1].tolist() == [True, False]
    assert batched[2].tolist() == [radial.size, 0]
    assert not np.any(np.asarray(batched[0][1]))
    assert not np.any(np.asarray(batched[3][1]))
    assert batched[4][0].tolist() == query_valid.tolist()
    assert not np.any(np.asarray(batched[4][1]))


def test_implicit_coefficient_derivative_matches_finite_difference():
    """The normal-equation VJP matches a directional finite difference."""
    radial, vertical = _half_offset_lattice(12, 14)
    level = _level_set(radial, vertical)
    values = _manufactured_field(radial, vertical)
    point = (jnp.asarray(0.37), jnp.asarray(-0.16))

    def point_value(sampled_values):
        return fit_split_spline(radial, vertical, sampled_values, level)(*point)

    gradient = jax.grad(point_value)(values)
    direction = jnp.sin(jnp.arange(values.size).reshape(values.shape) + 0.37)
    direction = direction / jnp.linalg.norm(direction)
    step = 2.0e-5
    finite_difference = (
        point_value(values + step * direction) - point_value(values - step * direction)
    ) / (2.0 * step)
    automatic = jnp.vdot(gradient, direction)
    relative_error = jnp.abs(automatic - finite_difference) / jnp.maximum(
        jnp.abs(finite_difference), 1.0e-14
    )
    implementation = inspect.getsource(split_spline_module)

    print(
        f"implicit_directional_relative_error={float(relative_error):.9e} "
        f"automatic={float(automatic):.9e} "
        f"finite_difference={float(finite_difference):.9e}"
    )
    assert relative_error < 3.0e-6
    assert "@jax.custom_vjp" in implementation
    assert "def _normal_solve_reverse" in implementation
    assert "lstsq" not in implementation


def test_prototype_solovev_advantage_and_stationary_polish_are_retained():
    """The production split reproduces the frozen curved-interface field."""
    centres, _, _ = hex_lattice()
    radial = jnp.asarray(centres[..., 0])
    vertical = jnp.asarray(centres[..., 1])
    points = jnp.asarray(centres.reshape(-1, 2))
    values = solovev_flux(points).reshape(radial.shape)
    level = (_base_flux(points) - LOBE_OFFSET**4).reshape(radial.shape)
    spline = fit_split_spline(radial, vertical, values, level)

    query_radial = jnp.linspace(0.49, 1.51, 42)
    query_vertical = jnp.linspace(-0.70, 0.70, 46)
    radial_grid, vertical_grid = jnp.meshgrid(query_radial, query_vertical)
    query = jnp.stack((radial_grid, vertical_grid), axis=-1).reshape(-1, 2)
    exact_hessian = jax.vmap(jax.hessian(solovev_flux))(query)
    evaluation = spline.evaluate(query[:, 0], query[:, 1])
    fitted_hessian = _hessian(evaluation)
    level_value = _base_flux(query) - LOBE_OFFSET**4
    level_gradient = jax.vmap(jax.grad(_base_flux))(query)
    signed_distance = level_value / jnp.linalg.norm(level_gradient, axis=-1)
    boundary_band = jnp.abs(signed_distance) <= 0.05
    split_error = jnp.sqrt(
        jnp.mean((fitted_hessian[boundary_band] - exact_hessian[boundary_band]) ** 2)
    )

    baseline = json.loads(_BASELINE.read_text())
    global_error = baseline["representation"]["boundary_band"]["global"][
        "second_derivative_rms"
    ]
    advantage = global_error / float(split_error)
    seeds = jnp.asarray((AXIS + (0.005, 0.0), SADDLE + (0.0, 0.005)))
    polished = polish_stationary_points(
        spline, seeds, jnp.asarray((True, True)), stationary_steps=8
    )
    position_error = jnp.linalg.norm(
        polished["position_rz"] - jnp.asarray((AXIS, SADDLE)), axis=-1
    )

    print(
        f"condition_number={float(spline.condition_number):.6e} "
        f"split_boundary_band_hessian_rms={float(split_error):.6e} "
        f"frozen_global_hessian_rms={global_error:.6e} "
        f"split_advantage={advantage:.6e} "
        f"polish_errors_m={np.asarray(position_error).tolist()}"
    )
    assert spline.condition_number < 1.0e7
    assert split_error < 1.0e-6
    assert advantage > 1.0e5
    assert jnp.max(position_error) < 2.0e-8
    assert polished["converged"].tolist() == [True, True]
