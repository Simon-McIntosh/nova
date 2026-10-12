"""The polygon-moment reductions consolidate onto one owner.

An exact-rational census of 600 polygons (convex, concave, sliver, near
collinear, hex and hole-fragment rings) selected
``clip_quadrature._straight_edge_monomial_moments`` as the most accurate general
reduction. This module holds that selection: the owner reproduces monomial
integrals to machine precision against ``fractions.Fraction``, and the parallel
reductions that were deleted do not reappear.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp  # noqa: E402

from nova.equilibrium.clip_quadrature import _straight_edge_monomial_moments  # noqa: E402
from nova.equilibrium.separatrix_clip import _centred_degree_two_moments  # noqa: E402

# total degree -> the (p, q) exponent pairs in the owner's column order.
_POWERS = [(total, radial) for total in range(5) for radial in range(total + 1)]


def _exact_monomials(ring: np.ndarray) -> dict[tuple[int, int], float]:
    """Exact signed-fan monomial integrals over a ring of double coordinates."""
    pts = [(Fraction(float(x)), Fraction(float(y))) for x, y in ring]
    area_twice = Fraction(0)
    out = {(p, total - p): Fraction(0) for total, p in _POWERS}
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True):
        cross = x0 * y1 - x1 * y0
        area_twice += cross
        for total, radial in _POWERS:
            vertical_power = total - radial
            edge = Fraction(0)
            for i in range(radial + 1):
                rfac = math.comb(radial, i) * x0**i * x1 ** (radial - i)
                for j in range(vertical_power + 1):
                    d = i + j
                    simplex = Fraction(
                        math.factorial(d) * math.factorial(total - d),
                        math.factorial(total + 2),
                    )
                    vfac = (
                        math.comb(vertical_power, j)
                        * y0**j
                        * y1 ** (vertical_power - j)
                    )
                    edge += simplex * rfac * vfac
            out[(radial, vertical_power)] += cross * edge
    sign = -1 if area_twice < 0 else 1
    return {power: float(sign * value) for power, value in out.items()}


def _rings():
    return [
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.2, -0.4], [1.3, -0.4], [1.3, 0.8], [0.2, 0.8]]),
        np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.866],
                [-0.5, 0.866],
                [-1.0, 0.0],
                [-0.5, -0.866],
                [0.5, -0.866],
            ]
        ),
        np.asarray([[0.0, 0.0], [0.3, 1e-10], [0.6, 0.0]]),  # sliver
        np.asarray([[-0.7, 0.3], [0.4, -0.2], [0.9, 0.5], [0.1, 1.1], [-0.5, 0.9]]),
    ]


def test_owner_integrates_monomials_to_machine_precision():
    rings = _rings()
    capacity = max(len(r) for r in rings)
    vertices = np.zeros((len(rings), capacity, 2))
    count = np.zeros(len(rings), dtype=int)
    for index, ring in enumerate(rings):
        vertices[index, : len(ring)] = ring
        count[index] = len(ring)
    table = np.asarray(
        _straight_edge_monomial_moments(
            jnp.asarray(vertices), jnp.asarray(count), max_degree=4
        )
    )
    worst_rel = 0.0
    worst_abs = 0.0
    for index, ring in enumerate(rings):
        reference = _exact_monomials(ring)
        for column, (total, radial) in enumerate(_POWERS):
            power = (radial, total - radial)
            expected = reference[power]
            got = float(table[index, column])
            worst_abs = max(worst_abs, abs(got - expected))
            if abs(expected) > 1e-6:
                worst_rel = max(worst_rel, abs(got - expected) / abs(expected))
    assert worst_abs < 1e-13
    assert worst_rel < 1e-11


def test_degree_two_view_reproduces_area_first_and_second_moments():
    ring = np.asarray([[0.2, -0.4], [1.3, -0.4], [1.3, 0.8], [0.2, 0.8]])
    vertices = ring[None, :, :]
    count = np.asarray([4])
    centroids = np.asarray([[0.75, 0.2]])
    area, first, second = _centred_degree_two_moments(
        jnp.asarray(vertices), jnp.asarray(count), jnp.asarray(centroids)
    )
    area = np.asarray(area)[0]
    first = np.asarray(first)[0]
    second = np.asarray(second)[0]

    local = ring - centroids[0]
    raw = _exact_monomials(local)
    assert abs(area - raw[(0, 0)]) < 1e-13
    np.testing.assert_allclose(first, [raw[(1, 0)], raw[(0, 1)]], atol=1e-13)
    np.testing.assert_allclose(
        second,
        [[raw[(2, 0)], raw[(1, 1)]], [raw[(1, 1)], raw[(0, 2)]]],
        atol=1e-13,
    )
