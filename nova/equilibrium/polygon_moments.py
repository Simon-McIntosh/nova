"""Exact closed-form moments of a local polynomial density over a polygon.

A clipped cell's support is a polygon whose boundary is a straight chain joined
to the traced level arc. When the density over that region is a polynomial in
the cell-local coordinates, its area, first and second moments are one
contraction of the density's coefficients against the exact monomial integrals
of the polygon.

The monomial integrals themselves come from the shared owner
``clip_quadrature._straight_edge_monomial_moments``, which serves every polygon
moment in the package. The monomial table it returns covers the whole triangle
``p + q <= 4``, ordered by total degree then increasing radial exponent, and its
unused vertex slots carry an exactly zero contribution and read their traversal
sense from the polygon's own signed area, so neither a shape nor a branch depends
on data. The local density is quadratic, carried by the six coefficients
``(1, x, y, x**2, xy, y**2)``, so its second moments reach total degree four and
each of its moments is one contraction of those coefficients against that table.
How closely the polygon tracks the true arc is set by the arc's vertex spacing
alone, never by a rule order.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "LOCAL_MONOMIAL_POWERS",
    "QUADRATIC_POWERS",
    "PolygonDensityMoments",
    "polygon_density_moments",
]

#: Exponent pairs of the monomial triangle through total degree four, in the
#: column order the shared owner reduction publishes them.
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


def polygon_density_moments(vertices, count, coefficients) -> PolygonDensityMoments:
    """Integrate a quadratic local density over each polygon exactly.

    ``vertices`` is a fixed-capacity ``(cells, capacity, 2)`` buffer of local
    coordinates and ``count`` gives each cell's live vertex count; slots at or
    past the count are exact-zero padding and contribute nothing.
    ``coefficients`` carries the six ``QUADRATIC_POWERS`` coefficients of the
    density in the same local coordinates as ``vertices``. Each returned moment
    is exact for that quadratic, so the polygon's vertex spacing, not any rule
    order, is what sets how closely the result tracks the region the clip books.

    The monomial integrals come straight from the shared owner
    ``clip_quadrature._straight_edge_monomial_moments``, whose columns are the
    ``(radial, total - radial)`` enumeration ``LOCAL_MONOMIAL_POWERS`` declares.
    """
    from nova.equilibrium.clip_quadrature import _straight_edge_monomial_moments

    degree = max(radial + vertical for radial, vertical in LOCAL_MONOMIAL_POWERS)
    table = _straight_edge_monomial_moments(vertices, count, max_degree=degree)
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
