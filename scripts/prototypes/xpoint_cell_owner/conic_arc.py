"""Closed-form moments of the square root of the slice discriminant.

This is the exact primitive the closed-form conic route needs, and it is the
piece the numerical integrator replaces. The closed-form route is: on each slice
abscissa interval the region is bounded above and below by either a polygon edge
(affine in the slice abscissa u) or a branch of the level-set hyperbola, whose
transverse coordinate is

    v(u) = (-B(u) + sign * sqrt(D(u))) / (2 A),

with D the slice discriminant, quadratic in u. The density antiderivative is a
polynomial of degree four in v with coefficients polynomial in u, so every term
expands into a polynomial part and a polynomial part times sqrt(D). The
polynomial part integrates term by term; the square-root part reduces to the
moments J_m(u) = integral of u**m sqrt(D(u)) du below, which are elementary.

Nothing here samples the region, so the primitive is exact to roundoff rather
than to a quadrature rule.
"""

from __future__ import annotations

from math import comb

import numpy as np

# Degree bound on the square-root polynomial part for one cell term: the density
# antiderivative is quartic in v, the edge and branch expansions contribute one
# power of u per power of v, so order eight covers every term with margin.
MOMENT_ORDER = 8


def moments_of_root(discriminant):
    """Return [J_0, ..., J_7] with J_m(u) = integral of u**m sqrt(D(u)) du.

    D is the slice discriminant as an ascending coefficient array of length
    three. The Euler reduction expresses every moment through J_0, whose closed
    reduction depends on the sign of the leading coefficient because sqrt(D) is
    real over a different set in each case.
    """
    coefficients = list(np.asarray(discriminant, dtype=float)) + [0.0, 0.0, 0.0]
    constant, linear, quadratic = coefficients[:3]
    invariant = 4.0 * quadratic * constant - linear * linear

    def root(u):
        value = quadratic * u * u + linear * u + constant
        return np.sqrt(np.maximum(value, 0.0))

    def primitive_zero(u):
        u = np.asarray(u, dtype=float)
        value = root(u)
        part = (2.0 * quadratic * u + linear) * value / (4.0 * quadratic)
        if quadratic > 0.0:
            argument = 2.0 * np.sqrt(quadratic) * value + 2.0 * quadratic * u + linear
            return part + (invariant / (8.0 * quadratic**1.5)) * np.log(
                np.abs(argument) + 1e-300
            )
        return part + (invariant / (8.0 * (-quadratic) ** 1.5)) * np.arcsin(
            np.clip((2.0 * quadratic * u + linear) / np.sqrt(-invariant), -1.0, 1.0)
        )

    def linear_moment(u, index):
        u = np.asarray(u, dtype=float)
        if abs(linear) < 1e-30:
            return np.full_like(u, np.nan)
        value = root(u)
        total = np.zeros_like(u)
        for i in range(index + 1):
            total = total + (
                (-constant) ** (index - i)
                * comb(index, i)
                / (i + 1.5)
                * value ** (2 * i + 3)
            )
        return total / linear ** (index + 1)

    moments: list = []

    def make(index):
        if abs(quadratic) < 1e-14:
            return lambda u: linear_moment(u, index)
        if index == 0:
            return primitive_zero

        def moment(u):
            if index == 1:
                return (root(u) ** 3 - 0.5 * linear * 3.0 * moments[0](u)) / (
                    3.0 * quadratic
                )
            return (
                np.asarray(u, dtype=float) ** (index - 1) * root(u) ** 3
                - linear * (index + 0.5) * moments[index - 1](u)
                - constant * (index - 1) * moments[index - 2](u)
            ) / (quadratic * (index + 2.0))

        return moment

    for index in range(MOMENT_ORDER):
        moments.append(make(index))
    return moments, root


def integrate_root_polynomial(discriminant, coefficients, lower, upper):
    """Exact definite integral of poly(u) * sqrt(D(u)) over [lower, upper].

    ``coefficients`` are ascending in u. Each monomial uses its closed-form
    moment, so the result carries no quadrature error.
    """
    moments, _ = moments_of_root(discriminant)
    total = 0.0
    for index, coefficient in enumerate(np.asarray(coefficients, dtype=float)):
        if abs(coefficient) < 1e-300:
            continue
        total += coefficient * (moments[index](upper) - moments[index](lower))
    return float(total)
