"""Traced tensor-product cubic spline interpolation on structured lattices.

The fit uses not-a-knot end conditions, matching the default boundary condition
of SciPy's cubic ``RectBivariateSpline``.  Each coordinate direction is solved
with one batched tridiagonal interpolation solve.  The resulting per-cell
Bernstein blocks are a linear function of the sampled values and are therefore
differentiable with respect to the complete input map.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.jax.tree_util import Pytree
from nova.linalg.interpolant import Bernstein


class TensorSplineEvaluation(NamedTuple):
    """Value, gradient, and Hessian entries at paired coordinates."""

    value: jax.Array
    radial_derivative: jax.Array
    vertical_derivative: jax.Array
    radial_second_derivative: jax.Array
    mixed_derivative: jax.Array
    vertical_second_derivative: jax.Array


def _solve_tridiagonal(
    lower: jax.Array,
    diagonal: jax.Array,
    upper: jax.Array,
    right_hand_side: jax.Array,
) -> jax.Array:
    """Solve a tridiagonal system for any fixed batch of right-hand sides."""
    first_upper = upper[0] / diagonal[0]
    first_right = right_hand_side[..., 0] / diagonal[0]

    def eliminate(carry, inputs):
        prior_upper, prior_right = carry
        lower_value, diagonal_value, upper_value, right_value = inputs
        pivot = diagonal_value - lower_value * prior_upper
        next_upper = upper_value / pivot
        next_right = (right_value - lower_value * prior_right) / pivot
        return (next_upper, next_right), (next_upper, next_right)

    _, (upper_tail, right_tail) = jax.lax.scan(
        eliminate,
        (first_upper, first_right),
        (
            lower[1:],
            diagonal[1:],
            upper[1:],
            jnp.moveaxis(right_hand_side[..., 1:], -1, 0),
        ),
    )
    reduced_upper = jnp.concatenate((first_upper[None], upper_tail), axis=0)
    reduced_right = jnp.concatenate((first_right[None, ...], right_tail), axis=0)

    def substitute(next_value, index):
        value = reduced_right[index] - reduced_upper[index] * next_value
        return value, value

    last_value = reduced_right[-1]
    _, reversed_values = jax.lax.scan(
        substitute,
        last_value,
        jnp.arange(diagonal.size - 2, -1, -1),
    )
    solution = jnp.concatenate(
        (jnp.flip(reversed_values, axis=0), last_value[None, ...]), axis=0
    )
    return jnp.moveaxis(solution, 0, -1)


def _not_a_knot_second_derivatives(
    coordinate: jax.Array, values: jax.Array
) -> jax.Array:
    """Return knot second derivatives from a banded not-a-knot solve."""
    spacing = jnp.diff(coordinate)
    interior_count = coordinate.size - 2
    slope = jnp.diff(values, axis=-1) / spacing
    right_hand_side = 6.0 * jnp.diff(slope, axis=-1)

    lower = jnp.zeros(interior_count, dtype=coordinate.dtype)
    lower = lower.at[1:].set(spacing[1:interior_count])
    diagonal = 2.0 * (spacing[:interior_count] + spacing[1 : interior_count + 1])
    upper = jnp.zeros(interior_count, dtype=coordinate.dtype)
    upper = upper.at[:-1].set(spacing[1:interior_count])

    first_spacing = spacing[0]
    second_spacing = spacing[1]
    diagonal = diagonal.at[0].add(
        first_spacing * (first_spacing + second_spacing) / second_spacing
    )
    upper = upper.at[0].set(second_spacing - first_spacing**2 / second_spacing)

    penultimate_spacing = spacing[-2]
    final_spacing = spacing[-1]
    lower = lower.at[-1].set(
        penultimate_spacing - final_spacing**2 / penultimate_spacing
    )
    diagonal = diagonal.at[-1].add(
        final_spacing * (penultimate_spacing + final_spacing) / penultimate_spacing
    )

    interior = _solve_tridiagonal(lower, diagonal, upper, right_hand_side)
    first = (
        (first_spacing + second_spacing) * interior[..., 0]
        - first_spacing * interior[..., 1]
    ) / second_spacing
    final = (
        (penultimate_spacing + final_spacing) * interior[..., -1]
        - final_spacing * interior[..., -2]
    ) / penultimate_spacing
    return jnp.concatenate((first[..., None], interior, final[..., None]), axis=-1)


def _cubic_bernstein_control(coordinate: jax.Array, values: jax.Array) -> jax.Array:
    """Fit cubic intervals and return their four Bernstein control values."""
    spacing = jnp.diff(coordinate)
    second_derivative = _not_a_knot_second_derivatives(coordinate, values)
    slope = jnp.diff(values, axis=-1) / spacing
    left_derivative = (
        slope
        - spacing
        * (2.0 * second_derivative[..., :-1] + second_derivative[..., 1:])
        / 6.0
    )
    right_derivative = (
        slope
        + spacing
        * (second_derivative[..., :-1] + 2.0 * second_derivative[..., 1:])
        / 6.0
    )
    return jnp.stack(
        (
            values[..., :-1],
            values[..., :-1] + spacing * left_derivative / 3.0,
            values[..., 1:] - spacing * right_derivative / 3.0,
            values[..., 1:],
        ),
        axis=-1,
    )


def _bernstein_matrix(coordinate: jax.Array, order: int) -> jax.Array:
    """Evaluate Nova's Bernstein pytree while preserving leading dimensions."""
    coordinate = jnp.asarray(coordinate)
    return (
        Bernstein(order=order)
        .coefficent_matrix(coordinate.reshape(-1))
        .reshape(coordinate.shape + (order + 1,))
    )


def _tensor_bernstein(
    coefficient: jax.Array,
    radial: jax.Array,
    vertical: jax.Array,
    radial_order: int,
    vertical_order: int,
) -> jax.Array:
    """Evaluate tensor Bernstein blocks at paired local coordinates."""
    radial_basis = _bernstein_matrix(radial, radial_order)
    vertical_basis = _bernstein_matrix(vertical, vertical_order)
    return jnp.einsum("...i,...ij,...j->...", vertical_basis, coefficient, radial_basis)


@dataclass
@jax.tree_util.register_pytree_node_class
class TensorBSpline(Pytree):
    """Global not-a-knot cubic tensor spline represented by cell blocks."""

    radial: jax.Array
    vertical: jax.Array
    coefficients: jax.Array

    @property
    def cell_coefficients(self) -> jax.Array:
        """Return the fixed ``(vertical, radial, 4, 4)`` Bernstein blocks."""
        return self.coefficients

    def __call__(self, radial: jax.Array, vertical: jax.Array) -> jax.Array:
        """Evaluate the spline value at arbitrary paired coordinates."""
        return self.evaluate(radial, vertical).value

    def evaluate(
        self, radial: jax.Array, vertical: jax.Array
    ) -> TensorSplineEvaluation:
        """Evaluate values, both first derivatives, and all Hessian entries."""
        radial, vertical = jnp.broadcast_arrays(
            jnp.asarray(radial, dtype=self.coefficients.dtype),
            jnp.asarray(vertical, dtype=self.coefficients.dtype),
        )
        radial_cell = jnp.clip(
            jnp.searchsorted(self.radial, radial, side="right") - 1,
            0,
            self.radial.size - 2,
        )
        vertical_cell = jnp.clip(
            jnp.searchsorted(self.vertical, vertical, side="right") - 1,
            0,
            self.vertical.size - 2,
        )
        radial_spacing = self.radial[radial_cell + 1] - self.radial[radial_cell]
        vertical_spacing = (
            self.vertical[vertical_cell + 1] - self.vertical[vertical_cell]
        )
        radial_local = (radial - self.radial[radial_cell]) / radial_spacing
        vertical_local = (vertical - self.vertical[vertical_cell]) / vertical_spacing
        coefficient = self.coefficients[vertical_cell, radial_cell]

        radial_coefficient = 3.0 * jnp.diff(coefficient, axis=-1)
        vertical_coefficient = 3.0 * jnp.diff(coefficient, axis=-2)
        radial_second = 2.0 * jnp.diff(radial_coefficient, axis=-1)
        vertical_second = 2.0 * jnp.diff(vertical_coefficient, axis=-2)
        mixed = 3.0 * jnp.diff(vertical_coefficient, axis=-1)

        return TensorSplineEvaluation(
            value=_tensor_bernstein(coefficient, radial_local, vertical_local, 3, 3),
            radial_derivative=_tensor_bernstein(
                radial_coefficient, radial_local, vertical_local, 2, 3
            )
            / radial_spacing,
            vertical_derivative=_tensor_bernstein(
                vertical_coefficient, radial_local, vertical_local, 3, 2
            )
            / vertical_spacing,
            radial_second_derivative=_tensor_bernstein(
                radial_second, radial_local, vertical_local, 1, 3
            )
            / radial_spacing**2,
            mixed_derivative=_tensor_bernstein(
                mixed, radial_local, vertical_local, 2, 2
            )
            / (radial_spacing * vertical_spacing),
            vertical_second_derivative=_tensor_bernstein(
                vertical_second, radial_local, vertical_local, 3, 1
            )
            / vertical_spacing**2,
        )

    def tree_flatten(self):
        """Return traced arrays and static metadata for JAX transformations."""
        return (self.radial, self.vertical, self.coefficients), {}


def fit_tensor_spline(
    radial: jax.Array, vertical: jax.Array, values: jax.Array
) -> TensorBSpline:
    """Fit a global cubic tensor spline to ``values[vertical, radial]``."""
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    if radial.ndim != 1 or vertical.ndim != 1 or values.ndim != 2:
        raise ValueError(
            "coordinates must be one-dimensional and values two-dimensional"
        )
    if radial.size < 4 or vertical.size < 4:
        raise ValueError("cubic interpolation requires at least four points per axis")
    if values.shape != (vertical.size, radial.size):
        raise ValueError("values must have shape (vertical.size, radial.size)")

    radial_control = _cubic_bernstein_control(radial, values)
    vertical_input = jnp.moveaxis(radial_control, 0, -1)
    vertical_control = _cubic_bernstein_control(vertical, vertical_input)
    coefficients = jnp.transpose(vertical_control, (2, 0, 3, 1))
    return TensorBSpline(radial, vertical, coefficients)


__all__ = ["TensorBSpline", "TensorSplineEvaluation", "fit_tensor_spline"]
