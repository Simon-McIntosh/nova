"""Conditioned boundary-split field fitting on structured point lattices.

The fitted field is a smooth interior tensor patch plus an exterior correction
of the form ``positive_part(level_set)**2 * correction``.  The squared level
set makes the value and gradient identical on both sides of the curved zero
level while allowing the normal curvature to jump.  Both patches and the level
set use normalized tensor Bernstein coordinates, keeping their coefficient
systems small and independent of the number of fixed-shape samples.

The coefficient solve carries authored implicit forward and reverse rules at
the normal equations. Automatic differentiation therefore differentiates the
solved system rather than the linear algebra used to obtain its coefficients.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from nova.jax.tree_util import Pytree
from nova.linalg.tensor_spline import (
    FixedTensorSplineEvaluation,
    TensorSplineEvaluation,
    _bernstein_matrix,
    evaluate_tensor_bernstein,
    mask_tensor_spline_evaluation,
)


@jax.custom_vjp
def _implicit_normal_solve(
    design: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    regularization: jax.Array,
) -> jax.Array:
    """Solve regularized normal equations with an implicit reverse rule."""
    weighted_design = weights[:, None] * design
    normal = design.T @ weighted_design
    normal = normal + regularization * jnp.eye(normal.shape[0], dtype=normal.dtype)
    right_hand_side = design.T @ (weights * values)
    return jnp.linalg.solve(normal, right_hand_side)


def _normal_solve_forward(design, values, weights, regularization):
    coefficient = _implicit_normal_solve(design, values, weights, regularization)
    return coefficient, (design, values, weights, regularization, coefficient)


def _normal_solve_reverse(saved, coefficient_cotangent):
    design, values, weights, regularization, coefficient = saved
    weighted_design = weights[:, None] * design
    normal = design.T @ weighted_design
    normal = normal + regularization * jnp.eye(normal.shape[0], dtype=normal.dtype)
    adjoint = jnp.linalg.solve(normal.T, coefficient_cotangent)
    residual = design @ coefficient - values
    projected_adjoint = design @ adjoint
    design_cotangent = -(
        (weights * residual)[:, None] * adjoint[None, :]
        + (weights * projected_adjoint)[:, None] * coefficient[None, :]
    )
    values_cotangent = weights * projected_adjoint
    weights_cotangent = -residual * projected_adjoint
    regularization_cotangent = -jnp.vdot(adjoint, coefficient)
    return (
        design_cotangent,
        values_cotangent,
        weights_cotangent,
        regularization_cotangent,
    )


_implicit_normal_solve.defvjp(_normal_solve_forward, _normal_solve_reverse)


@jax.custom_jvp
def _differentiable_normal_solve(design, values, weights, regularization):
    """Expose the normal-equation solve to both differentiation modes."""
    return _implicit_normal_solve(design, values, weights, regularization)


@_differentiable_normal_solve.defjvp
def _differentiable_normal_solve_jvp(primals, tangents):
    design, values, weights, regularization = primals
    design_dot, values_dot, weights_dot, regularization_dot = tangents
    coefficient = _differentiable_normal_solve(*primals)

    weighted_design = weights[:, None] * design
    weighted_design_dot = weights_dot[:, None] * design + weights[:, None] * design_dot
    normal = design.T @ weighted_design
    identity = jnp.eye(normal.shape[0], dtype=normal.dtype)
    normal = normal + regularization * identity
    normal_dot = design_dot.T @ weighted_design + design.T @ weighted_design_dot
    normal_dot = normal_dot + regularization_dot * identity

    weighted_values = weights * values
    weighted_values_dot = weights_dot * values + weights * values_dot
    right_hand_side_dot = (
        design_dot.T @ weighted_values + design.T @ weighted_values_dot
    )
    coefficient_dot = jnp.linalg.solve(
        normal, right_hand_side_dot - normal_dot @ coefficient
    )
    return coefficient, coefficient_dot


def _coordinate_grids(
    radial: jax.Array, vertical: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Broadcast axis or half-offset coordinates to the sample array."""
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    if radial.ndim == 1 and vertical.ndim == 1:
        if values.shape != (vertical.size, radial.size):
            raise ValueError("values must have shape (vertical.size, radial.size)")
        radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
        return radial_grid, vertical_grid
    if radial.shape == values.shape and vertical.ndim == 1:
        if vertical.size != values.shape[0]:
            raise ValueError("the vertical axis must match the first sample dimension")
        return radial, jnp.broadcast_to(vertical[:, None], values.shape)
    if vertical.shape == values.shape and radial.ndim == 1:
        if radial.size != values.shape[1]:
            raise ValueError("the radial axis must match the second sample dimension")
        return jnp.broadcast_to(radial[None, :], values.shape), vertical
    if radial.shape == values.shape and vertical.shape == values.shape:
        return radial, vertical
    raise ValueError("coordinates must be axes or arrays with the values shape")


def _patch_design(
    radial: jax.Array,
    vertical: jax.Array,
    radial_bounds: jax.Array,
    vertical_bounds: jax.Array,
    order: int,
) -> jax.Array:
    radial_local = (radial - radial_bounds[0]) / (radial_bounds[1] - radial_bounds[0])
    vertical_local = (vertical - vertical_bounds[0]) / (
        vertical_bounds[1] - vertical_bounds[0]
    )
    radial_basis = _bernstein_matrix(radial_local, order)
    vertical_basis = _bernstein_matrix(vertical_local, order)
    return jnp.einsum("...i,...j->...ij", vertical_basis, radial_basis).reshape(
        radial.size, (order + 1) ** 2
    )


def _conditioned_fit(
    design: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    regularization: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Column-scale a fixed design before solving its normal equations."""
    squared_norm = jnp.sum(weights[:, None] * design**2, axis=0)
    scale = jnp.sqrt(squared_norm)
    scale = jnp.where(scale > jnp.finfo(design.dtype).tiny, scale, 1.0)
    scaled_design = design / scale
    normal = scaled_design.T @ (weights[:, None] * scaled_design)
    normal = normal + regularization * jnp.eye(normal.shape[0], dtype=normal.dtype)
    condition_number = jnp.linalg.cond(normal)
    scaled_coefficient = _differentiable_normal_solve(
        scaled_design, values, weights, regularization
    )
    normal_right_hand_side = scaled_design.T @ (weights * values)
    normal_residual = normal @ scaled_coefficient - normal_right_hand_side
    residual_scale = jnp.maximum(
        jnp.linalg.norm(normal_right_hand_side), jnp.finfo(design.dtype).tiny
    )
    return (
        scaled_coefficient / scale,
        condition_number,
        jnp.linalg.norm(normal_residual) / residual_scale,
        scale,
    )


@dataclass
@jax.tree_util.register_pytree_node_class
class SplitTensorBSpline(Pytree):
    """A curved-interface field with the tensor-spline evaluation contract."""

    radial: jax.Array
    vertical: jax.Array
    coefficients: jax.Array
    level_set_coefficients: jax.Array
    condition_number: jax.Array
    fit_executed: jax.Array
    sample_count: jax.Array
    solve_iterations: jax.Array
    solve_residual: jax.Array
    solve_converged: jax.Array
    sample_rms_residual: jax.Array
    field_column_scale: jax.Array
    field_scale: jax.Array

    @property
    def interior_coefficients(self) -> jax.Array:
        """Return the smooth patch used throughout the interior."""
        return self.coefficients[0]

    @property
    def exterior_correction_coefficients(self) -> jax.Array:
        """Return the patch multiplied by the squared exterior level set."""
        return self.coefficients[1]

    def __call__(self, radial: jax.Array, vertical: jax.Array) -> jax.Array:
        """Evaluate the split field value at arbitrary paired coordinates."""
        return self.evaluate(radial, vertical).value

    def _patch_evaluation(
        self, coefficient: jax.Array, radial: jax.Array, vertical: jax.Array
    ) -> TensorSplineEvaluation:
        radial_spacing = self.radial[1] - self.radial[0]
        vertical_spacing = self.vertical[1] - self.vertical[0]
        radial_local = (radial - self.radial[0]) / radial_spacing
        vertical_local = (vertical - self.vertical[0]) / vertical_spacing
        return evaluate_tensor_bernstein(
            coefficient,
            radial_local,
            vertical_local,
            radial_spacing,
            vertical_spacing,
        )

    def evaluate(
        self, radial: jax.Array, vertical: jax.Array
    ) -> TensorSplineEvaluation:
        """Evaluate values, first derivatives, and Hessian entries."""
        radial, vertical = jnp.broadcast_arrays(
            jnp.asarray(radial, dtype=self.coefficients.dtype),
            jnp.asarray(vertical, dtype=self.coefficients.dtype),
        )
        interior = self._patch_evaluation(self.interior_coefficients, radial, vertical)
        correction = self._patch_evaluation(
            self.exterior_correction_coefficients, radial, vertical
        )
        level = self._patch_evaluation(self.level_set_coefficients, radial, vertical)
        exterior_level = jnp.where(level.value > 0.0, level.value, 0.0)
        weight = exterior_level**2
        weight_radial = jnp.where(
            level.value > 0.0,
            2.0 * level.value * level.radial_derivative,
            0.0,
        )
        weight_vertical = jnp.where(
            level.value > 0.0,
            2.0 * level.value * level.vertical_derivative,
            0.0,
        )
        weight_radial_second = jnp.where(
            level.value > 0.0,
            2.0
            * (
                level.radial_derivative**2
                + level.value * level.radial_second_derivative
            ),
            0.0,
        )
        weight_mixed = jnp.where(
            level.value > 0.0,
            2.0
            * (
                level.radial_derivative * level.vertical_derivative
                + level.value * level.mixed_derivative
            ),
            0.0,
        )
        weight_vertical_second = jnp.where(
            level.value > 0.0,
            2.0
            * (
                level.vertical_derivative**2
                + level.value * level.vertical_second_derivative
            ),
            0.0,
        )
        return TensorSplineEvaluation(
            value=interior.value + weight * correction.value,
            radial_derivative=(
                interior.radial_derivative
                + weight_radial * correction.value
                + weight * correction.radial_derivative
            ),
            vertical_derivative=(
                interior.vertical_derivative
                + weight_vertical * correction.value
                + weight * correction.vertical_derivative
            ),
            radial_second_derivative=(
                interior.radial_second_derivative
                + weight_radial_second * correction.value
                + 2.0 * weight_radial * correction.radial_derivative
                + weight * correction.radial_second_derivative
            ),
            mixed_derivative=(
                interior.mixed_derivative
                + weight_mixed * correction.value
                + weight_radial * correction.vertical_derivative
                + weight_vertical * correction.radial_derivative
                + weight * correction.mixed_derivative
            ),
            vertical_second_derivative=(
                interior.vertical_second_derivative
                + weight_vertical_second * correction.value
                + 2.0 * weight_vertical * correction.vertical_derivative
                + weight * correction.vertical_second_derivative
            ),
        )

    def evaluate_fixed(
        self, radial: jax.Array, vertical: jax.Array, valid: jax.Array
    ) -> FixedTensorSplineEvaluation:
        """Evaluate fixed slots with exact-zero inactive padding."""
        return mask_tensor_spline_evaluation(self.evaluate(radial, vertical), valid)

    def scaled_basis_receipt(
        self, radial: jax.Array, vertical: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return active count plus value and gradient operator norms.

        The field solve uses ``A D^-1`` and reports the condition number in its
        scaled coefficient coordinates.  These operators therefore evaluate
        ``B D^-1`` and ``B' D^-1``.  The exterior level weight is divided by the
        square of the field range before fitting and in these basis operators,
        so interior and correction columns both map flux-valued scaled
        coefficients to a field value (or gradient) before their Euclidean
        norms combine.  The stored correction coefficient is converted back to
        the physical level-squared parameterization used by field evaluation.
        """
        radial, vertical = jnp.broadcast_arrays(
            jnp.asarray(radial, dtype=self.coefficients.dtype),
            jnp.asarray(vertical, dtype=self.coefficients.dtype),
        )
        coefficient_shape = self.interior_coefficients.shape
        coefficient_count = self.interior_coefficients.size
        identity = jnp.eye(coefficient_count, dtype=self.coefficients.dtype).reshape(
            (coefficient_count,) + coefficient_shape
        )
        basis = jax.vmap(
            lambda coefficient: self._patch_evaluation(coefficient, radial, vertical)
        )(identity)
        level = self._patch_evaluation(self.level_set_coefficients, radial, vertical)
        exterior_level = jnp.maximum(level.value, 0.0)
        inverse_field_scale_squared = self.field_scale**-2
        weight = exterior_level**2 * inverse_field_scale_squared
        weight_radial = jnp.where(
            level.value > 0.0,
            2.0 * level.value * level.radial_derivative * inverse_field_scale_squared,
            0.0,
        )
        weight_vertical = jnp.where(
            level.value > 0.0,
            2.0 * level.value * level.vertical_derivative * inverse_field_scale_squared,
            0.0,
        )
        interior_gradient_basis = jnp.stack(
            (basis.radial_derivative, basis.vertical_derivative), axis=-1
        )
        correction_gradient_basis = jnp.stack(
            (
                weight_radial * basis.value + weight * basis.radial_derivative,
                weight_vertical * basis.value + weight * basis.vertical_derivative,
            ),
            axis=-1,
        )
        interior_scale = self.field_column_scale[0].reshape(
            (coefficient_count,) + (1,) * radial.ndim
        )
        correction_scale = self.field_column_scale[1].reshape(
            (coefficient_count,) + (1,) * radial.ndim
        )
        interior_value_basis = basis.value / interior_scale
        correction_value_basis = weight * basis.value / correction_scale
        interior_gradient_basis = interior_gradient_basis / interior_scale[..., None]
        correction_gradient_basis = (
            correction_gradient_basis / correction_scale[..., None]
        )
        active_correction = level.value > 0.0
        squared_norm = jnp.sum(interior_gradient_basis**2, axis=(0, -1))
        squared_norm += jnp.where(
            active_correction,
            jnp.sum(correction_gradient_basis**2, axis=(0, -1)),
            0.0,
        )
        active_count = coefficient_count * (1 + active_correction.astype(jnp.int32))
        value_squared_norm = jnp.sum(interior_value_basis**2, axis=0)
        value_squared_norm += jnp.where(
            active_correction,
            jnp.sum(correction_value_basis**2, axis=0),
            0.0,
        )
        return active_count, jnp.sqrt(value_squared_norm), jnp.sqrt(squared_norm)

    def tree_flatten(self):
        """Return all fixed-shape arrays for JAX transformations."""
        return (
            self.radial,
            self.vertical,
            self.coefficients,
            self.level_set_coefficients,
            self.condition_number,
            self.fit_executed,
            self.sample_count,
            self.solve_iterations,
            self.solve_residual,
            self.solve_converged,
            self.sample_rms_residual,
            self.field_column_scale,
            self.field_scale,
        ), {}


def fit_split_spline(
    radial: jax.Array,
    vertical: jax.Array,
    values: jax.Array,
    level_set: jax.Array,
    *,
    valid: jax.Array | None = None,
    execute: jax.Array = True,
    order: int = 4,
    regularization: float | None = None,
) -> SplitTensorBSpline:
    """Fit a C1 interface split with C2-or-better regional patches.

    ``level_set`` is negative in the interior and positive in the exterior.
    Coordinates may be one-dimensional rectangular axes or arrays matching the
    values, including alternate-row half offsets. ``valid`` masks padded sample
    slots while ``execute`` masks an entire fixed-capacity batch member.  An
    inactive member returns exact-zero coefficients and a false
    ``fit_executed`` receipt without changing any array shape.
    """
    values = jnp.asarray(values)
    level_set = jnp.asarray(level_set, dtype=values.dtype)
    if values.ndim != 2 or level_set.shape != values.shape:
        raise ValueError(
            "values and level_set must be equal-shape two-dimensional arrays"
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise TypeError("split spline values must have a floating-point dtype")
    if order < 2:
        raise ValueError("split spline order must be at least two")
    if valid is None:
        valid = jnp.ones(values.shape, dtype=bool)
    else:
        valid = jnp.asarray(valid, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("valid must have the values shape")
    execute = jnp.asarray(execute, dtype=bool)
    radial_grid, vertical_grid = _coordinate_grids(radial, vertical, values)
    sample_count = jnp.sum(valid, dtype=jnp.int32)
    radial_min = jnp.min(jnp.where(valid, radial_grid, jnp.inf))
    radial_max = jnp.max(jnp.where(valid, radial_grid, -jnp.inf))
    vertical_min = jnp.min(jnp.where(valid, vertical_grid, jnp.inf))
    vertical_max = jnp.max(jnp.where(valid, vertical_grid, -jnp.inf))
    coordinate_support = (
        jnp.isfinite(radial_min)
        & jnp.isfinite(radial_max)
        & jnp.isfinite(vertical_min)
        & jnp.isfinite(vertical_max)
        & (radial_max > radial_min)
        & (vertical_max > vertical_min)
    )
    coefficient_count = 2 * (order + 1) ** 2
    fit_executed = execute & coordinate_support & (sample_count >= coefficient_count)
    radial_bounds = jnp.where(
        coordinate_support,
        jnp.stack((radial_min, radial_max)),
        jnp.asarray((0.0, 1.0), dtype=values.dtype),
    )
    vertical_bounds = jnp.where(
        coordinate_support,
        jnp.stack((vertical_min, vertical_max)),
        jnp.asarray((0.0, 1.0), dtype=values.dtype),
    )
    safe_values = jnp.where(valid, values, 0.0)
    safe_level_set = jnp.where(valid, level_set, 0.0)
    weights = (valid & fit_executed).reshape(-1).astype(values.dtype)
    if regularization is None:
        regularization = 1.0e-12
    regularization_array = jnp.asarray(regularization, dtype=values.dtype)

    base_design = _patch_design(
        radial_grid,
        vertical_grid,
        radial_bounds,
        vertical_bounds,
        order,
    )
    level_coefficient, _level_condition, level_residual, _level_scale = (
        _conditioned_fit(
            base_design,
            safe_level_set.reshape(-1),
            weights,
            regularization_array,
        )
    )
    patch_shape = (order + 1, order + 1)
    level_patch = level_coefficient.reshape(patch_shape)
    level_at_samples = base_design @ level_coefficient
    field_min = jnp.min(jnp.where(valid, values, jnp.inf))
    field_max = jnp.max(jnp.where(valid, values, -jnp.inf))
    field_scale = jnp.maximum(field_max - field_min, jnp.finfo(values.dtype).tiny)
    field_scale = jnp.where(coordinate_support, field_scale, 1.0)
    exterior_weight = (jnp.maximum(level_at_samples, 0.0) / field_scale) ** 2
    split_design = jnp.concatenate(
        (base_design, exterior_weight[:, None] * base_design), axis=1
    )
    field_coefficient, field_condition, field_residual, field_column_scale = (
        _conditioned_fit(
            split_design,
            safe_values.reshape(-1),
            weights,
            regularization_array,
        )
    )
    field_patch = field_coefficient.reshape((2,) + patch_shape)
    field_patch = field_patch.at[1].divide(field_scale**2)
    sample_residual = weights * (
        split_design @ field_coefficient - safe_values.reshape(-1)
    )
    sample_rms_residual = jnp.sqrt(
        jnp.sum(sample_residual**2)
        / jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    )
    field_patch = jnp.where(fit_executed, field_patch, 0.0)
    level_patch = jnp.where(fit_executed, level_patch, 0.0)
    condition_number = jnp.where(fit_executed, field_condition, 1.0)
    solve_residual = jnp.where(
        fit_executed, jnp.maximum(level_residual, field_residual), 0.0
    )
    solve_tolerance = 64.0 * jnp.sqrt(jnp.finfo(values.dtype).eps)
    solve_converged = (
        fit_executed
        & jnp.isfinite(condition_number)
        & jnp.isfinite(solve_residual)
        & (solve_residual <= solve_tolerance)
    )
    return SplitTensorBSpline(
        radial_bounds,
        vertical_bounds,
        field_patch,
        level_patch,
        condition_number,
        fit_executed,
        sample_count,
        jnp.where(fit_executed, 1, 0).astype(jnp.int32),
        solve_residual,
        solve_converged,
        jnp.where(fit_executed, sample_rms_residual, 0.0),
        field_column_scale.reshape((2,) + patch_shape),
        field_scale,
    )


__all__ = ["SplitTensorBSpline", "fit_split_spline"]
