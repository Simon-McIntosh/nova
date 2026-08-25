"""Fixed-shape coefficient carriers for exact-output fixed-point maps.

The carrier compresses an iterate; it is not a physics field.  A caller may
subtract a known external field so coefficients represent only the plasma
flux, then restore that field before evaluating the ordinary exact-output map.
Residuals and admissibility always use total-field values, never projected
coefficients.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.linalg.tensor_spline import fit_tensor_spline
from nova.jax.config import configure_dtypes

__all__ = [
    "CoefficientCarrier",
    "DenseNewtonResult",
    "IterateRoute",
    "coefficient_fixed_point_map",
    "dense_newton",
    "exact_fixed_point_map",
    "relative_exact_residual",
    "select_fixed_point_map",
]


class IterateRoute(str, Enum):
    """Explicit call-site selection of the fixed-point iterate representation."""

    EXACT_VALUES = "exact_values"
    COEFFICIENT_CARRIER = "coefficient_carrier"


class DenseNewtonResult(NamedTuple):
    """Host-controlled dense Newton solve with exact-output qualifications."""

    coefficients: jax.Array
    exact_state: jax.Array
    exact_output: jax.Array
    exact_residual: jax.Array
    trace: jax.Array
    admitted_advances: int
    newton_step_equivalents: float
    jacobian_seconds: float
    solve_seconds: float


@dataclass(frozen=True)
class CoefficientCarrier:
    """Linear expansion and least-squares projection for spline knot values."""

    radial: jax.Array
    vertical: jax.Array
    coordinate: jax.Array
    expansion: jax.Array
    projection: jax.Array

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        """Return the vertical-by-radial knot-value shape."""
        return (int(self.vertical.size), int(self.radial.size))

    @property
    def coefficient_count(self) -> int:
        """Return the number of scalar knot values carried by one iterate."""
        return int(self.expansion.shape[1])

    @property
    def exact_size(self) -> int:
        """Return the number of exact-output values represented."""
        return int(self.expansion.shape[0])

    @classmethod
    def from_coordinates(
        cls,
        coordinate,
        *,
        radial_knots: int,
        vertical_knots: int,
        rcond: float | None = None,
    ) -> CoefficientCarrier:
        """Build the fixed spline expansion and its least-squares projection."""
        configure_dtypes()
        points = np.asarray(coordinate, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinate must have shape (points, 2)")
        if radial_knots < 4 or vertical_knots < 4:
            raise ValueError("a cubic carrier requires at least four knots per axis")
        if not np.all(np.isfinite(points)):
            raise ValueError("carrier coordinates must be finite")

        radial = np.linspace(points[:, 0].min(), points[:, 0].max(), radial_knots)
        vertical = np.linspace(points[:, 1].min(), points[:, 1].max(), vertical_knots)
        radial_device = jnp.asarray(radial)
        vertical_device = jnp.asarray(vertical)
        coordinate_device = jnp.asarray(points)
        coefficient_count = radial_knots * vertical_knots

        def expand(flat_values):
            values = flat_values.reshape(vertical_knots, radial_knots)
            spline = fit_tensor_spline(radial_device, vertical_device, values)
            return spline(coordinate_device[:, 0], coordinate_device[:, 1])

        expansion = np.asarray(
            jax.jacfwd(expand)(jnp.zeros(coefficient_count, dtype=jnp.float64)),
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(expansion) != coefficient_count:
            raise ValueError("carrier expansion is rank deficient on these coordinates")
        projection = (
            np.linalg.pinv(expansion)
            if rcond is None
            else np.linalg.pinv(expansion, rcond=rcond)
        )
        return cls(
            radial=radial_device,
            vertical=vertical_device,
            coordinate=coordinate_device,
            expansion=jnp.asarray(expansion),
            projection=jnp.asarray(projection),
        )

    def expand(self, coefficients) -> jax.Array:
        """Expand one flat coefficient vector at every exact-output point."""
        values = jnp.asarray(coefficients)
        if values.shape != (self.coefficient_count,):
            raise ValueError(
                f"coefficients must have shape ({self.coefficient_count},)"
            )
        return self.expansion @ values

    def project(self, exact_values) -> jax.Array:
        """Project one exact-output vector onto the fixed coefficient space."""
        values = jnp.asarray(exact_values)
        if values.shape != (self.exact_size,):
            raise ValueError(f"exact_values must have shape ({self.exact_size},)")
        return self.projection @ values

    def projection_floor(self, exact_values) -> jax.Array:
        """Return the relative sup error of the best represented field."""
        values = jnp.asarray(exact_values)
        represented = self.expand(self.project(values))
        return jnp.max(jnp.abs(represented - values)) / jnp.maximum(
            jnp.max(jnp.abs(values)), jnp.asarray(1.0e-30, dtype=values.dtype)
        )


def relative_exact_residual(exact_output, exact_state) -> jax.Array:
    """Return the relative sup residual on exact-output values."""
    output = jnp.asarray(exact_output)
    state = jnp.asarray(exact_state)
    return jnp.max(jnp.abs(output - state)) / jnp.maximum(
        jnp.max(jnp.abs(output)), jnp.asarray(1.0e-30, dtype=output.dtype)
    )


def exact_fixed_point_map(exact_map: Callable[[jax.Array], jax.Array]):
    """Return the ordinary exact-value map without changing its state."""

    def mapped(exact_state):
        return exact_map(exact_state)

    return mapped


def _known_external(external, carrier: CoefficientCarrier) -> jax.Array:
    """Return a validated known field, or zeros for an unshifted carrier."""
    if external is None:
        return jnp.zeros(carrier.exact_size, dtype=carrier.expansion.dtype)
    values = jnp.asarray(external)
    if values.shape != (carrier.exact_size,):
        raise ValueError(f"external must have shape ({carrier.exact_size},)")
    return values


def coefficient_fixed_point_map(
    exact_map: Callable[[jax.Array], jax.Array],
    carrier: CoefficientCarrier,
    *,
    external=None,
):
    """Return a projected plasma-only map and its exact total-field output."""
    known_external = _known_external(external, carrier)

    def exact_state(coefficients):
        return known_external + carrier.expand(coefficients)

    def mapped(coefficients):
        return carrier.project(exact_map(exact_state(coefficients)) - known_external)

    def exact_output(coefficients):
        return exact_map(exact_state(coefficients))

    return mapped, exact_output


def select_fixed_point_map(
    route: IterateRoute | str,
    exact_map: Callable[[jax.Array], jax.Array],
    *,
    carrier: CoefficientCarrier | None = None,
    external=None,
):
    """Select the exact-value or coefficient state explicitly at the call site."""
    selected = IterateRoute(route)
    if selected is IterateRoute.EXACT_VALUES:
        return exact_fixed_point_map(exact_map)
    if carrier is None:
        raise ValueError("the coefficient route requires a carrier")
    return coefficient_fixed_point_map(exact_map, carrier, external=external)[0]


def dense_newton(
    exact_map: Callable[[jax.Array], jax.Array],
    carrier: CoefficientCarrier,
    initial_coefficients,
    *,
    steps: int,
    admissible: Callable[[jax.Array], jax.Array | bool] | None = None,
    factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    external=None,
) -> DenseNewtonResult:
    """Drive the exact residual in the carrier subspace with dense steps."""
    from time import perf_counter

    configure_dtypes()
    if steps < 1:
        raise ValueError("steps must be positive")
    if not factors or any(factor <= 0.0 or factor > 1.0 for factor in factors):
        raise ValueError("admission factors must lie in (0, 1]")

    coefficients = jnp.asarray(initial_coefficients, dtype=jnp.float64)
    if coefficients.shape != (carrier.coefficient_count,):
        raise ValueError(
            f"initial_coefficients must have shape ({carrier.coefficient_count},)"
        )
    known_external = _known_external(external, carrier)
    qualifies = (
        admissible
        if admissible is not None
        else lambda value: jnp.all(jnp.isfinite(value))
    )

    def exact_residual(value):
        state = known_external + carrier.expand(value)
        return exact_map(state) - state

    def evaluated(value):
        state = known_external + carrier.expand(value)
        output = exact_map(state)
        return state, output, relative_exact_residual(output, state)

    state, output, residual = evaluated(coefficients)
    trace = [float(residual)]
    admitted = 0
    equivalents = 0.0
    jacobian_seconds = 0.0
    solve_seconds = 0.0

    for _ in range(steps):
        jacobian_started = perf_counter()
        residual_vector = exact_residual(coefficients)
        jacobian = jax.jacfwd(exact_residual)(coefficients)
        jacobian.block_until_ready()
        jacobian_seconds += perf_counter() - jacobian_started

        solve_started = perf_counter()
        step = jnp.linalg.lstsq(jacobian, -residual_vector, rcond=None)[0]
        step.block_until_ready()
        solve_seconds += perf_counter() - solve_started

        chosen = None
        for factor in factors:
            candidate = coefficients + factor * step
            candidate_state, candidate_output, candidate_residual = evaluated(candidate)
            candidate_residual.block_until_ready()
            if bool(qualifies(candidate_output)) and float(candidate_residual) < float(
                residual
            ):
                chosen = (
                    candidate,
                    candidate_state,
                    candidate_output,
                    candidate_residual,
                    factor,
                )
                break
        if chosen is None:
            break
        coefficients, state, output, residual, factor = chosen
        admitted += 1
        equivalents += factor
        trace.append(float(residual))

    return DenseNewtonResult(
        coefficients=coefficients,
        exact_state=state,
        exact_output=output,
        exact_residual=residual,
        trace=jnp.asarray(trace),
        admitted_advances=admitted,
        newton_step_equivalents=equivalents,
        jacobian_seconds=jacobian_seconds,
        solve_seconds=solve_seconds,
    )
