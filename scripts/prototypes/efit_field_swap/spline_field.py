"""Present an imas-efit trace grid through nova's tensor spline evaluator.

The record stores samples of imas-efit's not-a-knot bicubic psi spline. Nova's
``fit_tensor_spline`` owns the equivalent interpolant and its derivatives; the
adapter supplies only the field interface expected by the topology reader.
No Biot-Savart current or operator is booked in this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nova.equilibrium.topology import FieldJet
from nova.linalg.tensor_spline import TensorBSpline, fit_tensor_spline

if TYPE_CHECKING:
    from scripts.prototypes.efit_field_swap.trace_record import TraceRecord


@dataclass(frozen=True)
class TraceSplineField:
    """A trace record's psi as a differentiable point field."""

    spline: TensorBSpline

    @classmethod
    def from_record(cls, record: TraceRecord) -> TraceSplineField:
        return cls(
            fit_tensor_spline(
                jnp.asarray(record.rgrid),
                jnp.asarray(record.zgrid),
                jnp.asarray(record.psi).T,
            )
        )

    def value(self, point):
        point = jnp.asarray(point)
        return self.spline(point[0], point[1])

    def evaluate(self, point):
        point = jnp.asarray(point)
        result = self.spline.evaluate(point[0], point[1])
        gradient = jnp.stack((result.radial_derivative, result.vertical_derivative))
        hessian = jnp.asarray(
            (
                (result.radial_second_derivative, result.mixed_derivative),
                (result.mixed_derivative, result.vertical_second_derivative),
            )
        )
        return FieldJet(result.value, gradient, hessian)

    def sampled_coefficient(self, geometry):
        """Fit the same local quadratic per cell that the operator field supplies."""
        points = geometry.sample_points.reshape(-1, 2)
        values = jax.vmap(self.value)(points).reshape(geometry.sample_points.shape[:-1])
        return jnp.einsum("nks,ns->nk", geometry.fit_inverse, values)


jax.tree_util.register_dataclass(
    TraceSplineField,
    data_fields=["spline"],
    meta_fields=[],
)
