"""Present a trace record's psi grid to nova's read as a differentiable field.

imas-efit carries its flux as a bidimensional bicubic spline (``type(EZspline2)
``, initialised with not-a-knot boundary conditions ``bcs=(0,0)`` at
``src/EFIT/null_detection.f90``). The record stores the rectilinear grid and the
psi values that the spline is built from, so the adapter rebuilds the same
tensor-product not-a-knot cubic spline here. This spline, not a Biot-Savart
operator, is the field: nova's read consumes it exactly as it consumes its own
operator field, and no plasma current is booked.

Evaluation avoids a hand-written derivative recurrence. Each derivative of a
B-spline is itself a B-spline of one lower degree whose coefficients are a
divided difference of the parent's, so value, gradient and Hessian are four
tensor contractions against the same three cheap basis kernels. The result is
pure JAX, so the kernel's ``jax.jacfwd`` on the Hessian (the third- and
fourth-derivative saddle curvature terms) also sees this field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.topology import FieldJet

if TYPE_CHECKING:
    from scripts.prototypes.efit_field_swap.trace_record import TraceRecord


def _difference(coefficients, knots, degree, axis):
    """Coefficients of the derivative spline along one axis."""
    if axis == 0:
        left = np.take(coefficients, np.arange(1, coefficients.shape[0]), axis=0)
        right = np.take(coefficients, np.arange(0, coefficients.shape[0] - 1), axis=0)
        denominator = knots[degree + 1 : -1] - knots[1:-degree - 1]
    else:
        left = np.take(coefficients, np.arange(1, coefficients.shape[1]), axis=1)
        right = np.take(coefficients, np.arange(0, coefficients.shape[1] - 1), axis=1)
        denominator = knots[degree + 1 : -1] - knots[1:-degree - 1]
    shape = [1, 1]
    shape[axis] = denominator.shape[0]
    return degree * (left - right) / denominator.reshape(shape)


def _basis(knots, point, degree):
    """The ``degree + 1`` nonzero B-spline basis values and their span.

    Cox-de Boor over the local knots only; the triangular table is unrolled
    because the spline degree is a static integer.
    """
    span = jnp.clip(
        jnp.searchsorted(knots, point, side="right") - 1,
        degree,
        knots.shape[0] - degree - 2,
    )
    left = [None]
    right = [None]
    for order in range(1, degree + 1):
        left.append(point - jnp.take(knots, span + 1 - order))
        right.append(jnp.take(knots, span + order) - point)
    table = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    table[0][0] = 1.0
    for order in range(1, degree + 1):
        saved = 0.0
        for row in range(order):
            denominator = right[row + 1] + left[order - row]
            denominator = jnp.where(denominator == 0.0, 1.0, denominator)
            temp = table[row][order - 1] / denominator
            table[row][order] = saved + right[row + 1] * temp
            saved = left[order - row] * temp
        table[order][order] = saved
    values = jnp.stack(tuple(table[row][degree] for row in range(degree + 1)))
    return span, values


def _block(coefficients, span_r, span_z, degree_r, degree_z):
    rows = span_r - degree_r + jnp.arange(degree_r + 1)
    cols = span_z - degree_z + jnp.arange(degree_z + 1)
    return coefficients[rows[:, None], cols[None, :]]


@dataclass(frozen=True)
class TraceSplineField:
    """A record's bicubic psi field as a JAX-differentiable point evaluator."""

    value_coefficients: jax.Array
    dr_coefficients: jax.Array
    dz_coefficients: jax.Array
    drr_coefficients: jax.Array
    drz_coefficients: jax.Array
    dzz_coefficients: jax.Array
    knots_r: jax.Array
    knots_z: jax.Array
    knots_r_first: jax.Array
    knots_z_first: jax.Array
    knots_r_second: jax.Array
    knots_z_second: jax.Array

    @classmethod
    def from_record(cls, record: TraceRecord) -> TraceSplineField:
        knots_r, knots_z, flat = RectBivariateSpline(
            np.asarray(record.rgrid),
            np.asarray(record.zgrid),
            np.asarray(record.psi),
            kx=3,
            ky=3,
            s=0,
        ).tck
        grid = (knots_r.size - 4, knots_z.size - 4)
        coefficients = np.asarray(flat).reshape(grid)
        dr = _difference(coefficients, knots_r, 3, axis=0)
        dz = _difference(coefficients, knots_z, 3, axis=1)
        drr = _difference(dr, knots_r[1:-1], 2, axis=0)
        dzz = _difference(dz, knots_z[1:-1], 2, axis=1)
        drz = _difference(dr, knots_z, 3, axis=1)
        return cls(
            jnp.asarray(coefficients),
            jnp.asarray(dr),
            jnp.asarray(dz),
            jnp.asarray(drr),
            jnp.asarray(drz),
            jnp.asarray(dzz),
            jnp.asarray(knots_r),
            jnp.asarray(knots_z),
            jnp.asarray(knots_r[1:-1]),
            jnp.asarray(knots_z[1:-1]),
            jnp.asarray(knots_r[2:-2]),
            jnp.asarray(knots_z[2:-2]),
        )

    def value(self, point):
        """The scalar field at one RZ point."""
        point = jnp.asarray(point)
        span_r, basis_r = _basis(self.knots_r, point[0], 3)
        span_z, basis_z = _basis(self.knots_z, point[1], 3)
        block = _block(self.value_coefficients, span_r, span_z, 3, 3)
        return jnp.einsum("a,ab,b->", basis_r, block, basis_z)

    def evaluate(self, point):
        """Value, gradient and Hessian of the field at one RZ point."""
        point = jnp.asarray(point)
        span_r, basis_r = _basis(self.knots_r, point[0], 3)
        span_z, basis_z = _basis(self.knots_z, point[1], 3)
        flat = jnp.einsum(
            "a,ab,b->",
            basis_r,
            _block(self.value_coefficients, span_r, span_z, 3, 3),
            basis_z,
        )

        first_r, basis_r1 = _basis(self.knots_r_first, point[0], 2)
        first_z, basis_z1 = _basis(self.knots_z_first, point[1], 2)
        grad_r = jnp.einsum(
            "a,ab,b->",
            basis_r1,
            _block(self.dr_coefficients, first_r, span_z, 2, 3),
            basis_z,
        )
        grad_z = jnp.einsum(
            "a,ab,b->",
            basis_r,
            _block(self.dz_coefficients, span_r, first_z, 3, 2),
            basis_z1,
        )

        second_r, basis_r2 = _basis(self.knots_r_second, point[0], 1)
        second_z, basis_z2 = _basis(self.knots_z_second, point[1], 1)
        hess_rr = jnp.einsum(
            "a,ab,b->",
            basis_r2,
            _block(self.drr_coefficients, second_r, span_z, 1, 3),
            basis_z,
        )
        hess_zz = jnp.einsum(
            "a,ab,b->",
            basis_r,
            _block(self.dzz_coefficients, span_r, second_z, 3, 1),
            basis_z2,
        )
        hess_rz = jnp.einsum(
            "a,ab,b->",
            basis_r1,
            _block(self.drz_coefficients, first_r, first_z, 2, 2),
            basis_z1,
        )
        gradient = jnp.stack((grad_r, grad_z))
        hessian = jnp.asarray(((hess_rr, hess_rz), (hess_rz, hess_zz)))
        return FieldJet(flat, gradient, hessian)

    def sampled_coefficient(self, geometry):
        """Fit the same local quadratic per cell that the operator field supplies.

        The kernel expects a six-term Bell polynomial in each cell's own scaled
        frame. Here the fit samples this record's spline at the cell's fixed
        stencil, contracted with the geometry's own pseudo-inverse.
        """
        points = geometry.sample_points.reshape(-1, 2)
        values = jax.vmap(self.value)(points).reshape(
            geometry.sample_points.shape[:-1]
        )
        return jnp.einsum("nks,ns->nk", geometry.fit_inverse, values)

jax.tree_util.register_dataclass(
    TraceSplineField,
    data_fields=[
        "value_coefficients",
        "dr_coefficients",
        "dz_coefficients",
        "drr_coefficients",
        "drz_coefficients",
        "dzz_coefficients",
        "knots_r",
        "knots_z",
        "knots_r_first",
        "knots_z_first",
        "knots_r_second",
        "knots_z_second",
    ],
    meta_fields=[],
)
