"""Exact closed-form moments of a local polynomial density over a polygon.

A clipped cell's support is a polygon whose boundary is a straight chain joined
to the traced level arc. When the density over that region is a polynomial in
the cell-local coordinates, its area, first and second moments are boundary
integrals by Green's theorem, with the antiderivative ``A_pq`` of the monomial
``(p, q)``:

    A_pq(x, q) = x**(p+1) y**q / (p+1)
    integral over P of x**p y**q dA  =  closed contour of A_pq dy

Every edge is a straight segment, so its contour piece is ``dy`` times the exact
integral of ``A_pq`` along it. Parametrising the edge from ``(x0, y0)`` to
``(x1, y1)`` as ``x(t) = x0 + t dx`` and ``y(t) = y0 + t dy`` for ``t`` in
``[0, 1]``, the monomial ``(p, q)`` receives

    dy * sum over i <= p+1, j <= q of
        binom(p+1, i) binom(q, j) x0**(p+1-i) y0**(q-j) dx**i dy**j
        / ((p+1) (i + j + 1))

from that edge, which is exact for every polynomial integrand: no quadrature
node, rule order or convergence argument enters the reduction.

The monomial table covers the whole triangle ``p + q <= 4``. The local density
is quadratic, carried by the six coefficients ``(1, x, y, x**2, xy, y**2)``, so
its second moments reach total degree four and each of its moments is one
contraction of those coefficients against this table.

Every table here is a compile-time constant, an unused vertex slot carries an
exactly zero edge weight, and the traversal sense is read from the polygon's own
signed area, so neither a shape nor a branch depends on data. The result is the
polygon's own region; how closely that region tracks the true arc is set by the
arc's vertex spacing alone, never by a rule order.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "LOCAL_MONOMIAL_POWERS",
    "QUADRATIC_POWERS",
    "PolygonDensityMoments",
    "polygon_density_moments",
    "polygon_monomial_moments",
]

#: Exponent pairs of the monomial triangle through total degree four.
LOCAL_MONOMIAL_POWERS: tuple[tuple[int, int], ...] = tuple(
    (radial, total - radial) for total in range(5) for radial in range(total + 1)
)
#: Exponent pairs of the quadratic local density, in the order its six-sample
#: unisolvent fit publishes its coefficients.
QUADRATIC_POWERS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2),
)
_SLOT = {power: index for index, power in enumerate(LOCAL_MONOMIAL_POWERS)}
#: Widest ``x`` exponent an edge integral reaches: the antiderivative of a
#: degree-four monomial is degree five, so the binomial table runs to five.
_ANTIDERIVATIVE_DEGREE = 1 + max(radial for radial, _ in LOCAL_MONOMIAL_POWERS)
_MAX_VERTICAL = max(vertical for _, vertical in LOCAL_MONOMIAL_POWERS)
_BINOMIAL = np.zeros(
    (_ANTIDERIVATIVE_DEGREE + 1, _ANTIDERIVATIVE_DEGREE + 1), dtype=np.float64
)
for _row in range(_ANTIDERIVATIVE_DEGREE + 1):
    for _column in range(_row + 1):
        _BINOMIAL[_row, _column] = float(math.comb(_row, _column))


def _edge_weights() -> np.ndarray:
    """Return the fixed weight table of each monomial's edge power product.

    Entry ``(m, i, j)`` is the coefficient of ``x0**(a-i) y0**(b-j) dx**i dy**j``
    in monomial ``m``'s edge integral, where ``(a, b) = (p + 1, q)``. Slots
    outside the monomial's exponent box stay zero, so the contraction carries
    them as exact-zero padding instead of branching on them.
    """
    table = np.zeros(
        (len(LOCAL_MONOMIAL_POWERS), _ANTIDERIVATIVE_DEGREE + 1, _MAX_VERTICAL + 1),
        dtype=np.float64,
    )
    for slot, (radial, vertical) in enumerate(LOCAL_MONOMIAL_POWERS):
        span = radial + 1
        for i in range(span + 1):
            for j in range(vertical + 1):
                table[slot, i, j] = (
                    _BINOMIAL[span, i] * _BINOMIAL[vertical, j] / (span * (i + j + 1))
                )
    return table


def _residual_exponents(radial: bool) -> np.ndarray:
    """Return each term's residual exponent of ``x0`` or of ``y0``.

    Entry ``(m, k)`` is the power of that vertex coordinate multiplying term
    ``k`` of monomial ``m``. Slots outside the monomial's exponent box hold
    zero, which its zero weight makes irrelevant and keeps the table static.
    """
    width = _ANTIDERIVATIVE_DEGREE + 1 if radial else _MAX_VERTICAL + 1
    table = np.zeros((len(LOCAL_MONOMIAL_POWERS), width), dtype=np.float64)
    for slot, (radial_power, vertical_power) in enumerate(LOCAL_MONOMIAL_POWERS):
        span = radial_power + 1 if radial else vertical_power
        for index in range(span + 1):
            residual = radial_power + 1 - index if radial else vertical_power - index
            table[slot, index] = max(residual, 0)
    return table


_WEIGHT = _edge_weights()
_X_RESIDUAL = _residual_exponents(True)
_Y_RESIDUAL = _residual_exponents(False)
_X_EXPONENT = np.arange(_ANTIDERIVATIVE_DEGREE + 1, dtype=np.float64)
_Y_EXPONENT = np.arange(_MAX_VERTICAL + 1, dtype=np.float64)


class PolygonDensityMoments(NamedTuple):
    """Exponent-weighted integrals of a local density over a polygon.

    Each component is the exact integral of the density times the named
    monomial, in the coordinates the density's coefficients are written in.
    """

    area: jax.Array
    radial: jax.Array
    vertical: jax.Array
    radial_squared: jax.Array
    radial_vertical: jax.Array
    vertical_squared: jax.Array


def polygon_monomial_moments(vertices, count) -> jax.Array:
    """Integrate the monomials through total degree four over each polygon.

    ``vertices`` is a fixed-capacity ``(cells, capacity, 2)`` buffer of local
    coordinates and ``count`` gives each cell's live vertex count; slots at or
    past the count are exact-zero padding and contribute nothing. The result is
    ``(cells, len(LOCAL_MONOMIAL_POWERS))``, entry ``(cell, m)`` being the exact
    area moment ``integral of x**p y**q dA`` over that cell's polygon, ordered
    as ``LOCAL_MONOMIAL_POWERS`` and signed by the polygon's traversal sense.
    """
    point = jnp.asarray(vertices)
    live = jnp.asarray(count)
    if point.ndim != 3 or point.shape[-1] != 2:
        raise ValueError("vertices must have shape (cells, capacity, 2)")
    if live.shape != (point.shape[0],):
        raise ValueError("count must carry one vertex count per cell")
    capacity = point.shape[1]
    slot = jnp.arange(capacity)
    valid = slot[None, :] < live[:, None]
    following_slot = jnp.where(slot[None, :] + 1 < live[:, None], slot[None, :] + 1, 0)
    following = jnp.take_along_axis(point, following_slot[..., None], axis=1)
    x0 = point[..., 0]
    y0 = point[..., 1]
    dx = following[..., 0] - x0
    dy = following[..., 1] - y0
    radial_power = (
        x0[..., None, None] ** _X_RESIDUAL[None, None, :, :]
        * dx[..., None, None] ** _X_EXPONENT[None, None, None, :]
    )
    vertical_power = (
        y0[..., None, None] ** _Y_RESIDUAL[None, None, :, :]
        * dy[..., None, None] ** _Y_EXPONENT[None, None, None, :]
    )
    edge = jnp.einsum(
        "mij,ncmi,ncmj->ncm",
        jnp.asarray(_WEIGHT, dtype=point.dtype),
        radial_power,
        vertical_power,
    )
    cross = x0 * following[..., 1] - following[..., 0] * y0
    cross = jnp.where(valid, cross, 0.0)
    orientation = jnp.where(jnp.sum(cross, axis=1) < 0.0, -1.0, 1.0)
    contribution = jnp.where(valid[..., None], edge * dy[..., None], 0.0)
    return orientation[:, None] * jnp.sum(contribution, axis=1)


def polygon_density_moments(vertices, count, coefficients) -> PolygonDensityMoments:
    """Integrate a quadratic local density over each polygon exactly.

    ``coefficients`` carries the six ``QUADRATIC_POWERS`` coefficients of the
    density in the same local coordinates as ``vertices``. Each returned moment
    is exact for that quadratic, so the polygon's vertex spacing, not any rule
    order, is what sets how closely the result tracks the region the clip books.
    """
    table = polygon_monomial_moments(vertices, count)
    coefficient = jnp.asarray(coefficients)
    if coefficient.shape != (len(vertices), len(QUADRATIC_POWERS)):
        raise ValueError("coefficients must have shape (cells, 6)")

    def contract(radial_shift: int, vertical_shift: int) -> jax.Array:
        slots = jnp.asarray(
            tuple(
                _SLOT[(radial + radial_shift, vertical + vertical_shift)]
                for radial, vertical in QUADRATIC_POWERS
            )
        )
        return jnp.einsum("nm,nm->n", coefficient, table[:, slots])

    return PolygonDensityMoments(
        area=contract(0, 0),
        radial=contract(1, 0),
        vertical=contract(0, 1),
        radial_squared=contract(2, 0),
        radial_vertical=contract(1, 1),
        vertical_squared=contract(0, 2),
    )
