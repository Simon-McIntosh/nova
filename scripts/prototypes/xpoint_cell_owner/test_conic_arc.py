"""Verify the closed-form square-root moments against dense quadrature."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from scripts.prototypes.xpoint_cell_owner.conic_arc import (
    MOMENT_ORDER,
    integrate_root_polynomial,
    moments_of_root,
)

# Ascending coefficients [d0, d1, d2]. D = 4 - u**2: negative leading
# coefficient, sqrt(D) real between the roots, the branch case the X-point cell
# presents.
DOWNWARD = (4.0, 0.0, -1.0)
# D = 1 + u**2: positive leading coefficient, sqrt(D) real everywhere.
UPWARD = (1.0, 0.0, 1.0)
# D = 1 + 2 u: a linear discriminant, exercising the substitution branch.
LINEAR = (1.0, 2.0, 0.0)


def _dense(discriminant, coefficients, lower, upper):
    nodes, weights = leggauss(400)
    locations = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    scale = 0.5 * (upper - lower)
    _, root = moments_of_root(discriminant)
    polynomial = np.polyval(np.asarray(coefficients)[::-1], locations)
    return float(np.sum(weights * polynomial * root(locations)) * scale)


def _integrand_accuracy(discriminant, lower, upper):
    """Largest scaled gap between the primitive's difference quotient and its integrand.

    The gap is scaled by each moment's own size, not taken relative to the local
    value: at the zeros of the odd moments the integrand vanishes while its
    difference quotient carries the curvature term, so a pointwise relative
    measure is unbounded there for any finite step.
    """
    moments, root = moments_of_root(discriminant)
    step = 1e-6
    worst = 0.0
    for index in range(MOMENT_ORDER):
        locations = np.linspace(lower, upper, 7)
        exact_values = locations**index * root(locations)
        scale = 1.0 + float(np.max(np.abs(exact_values)))
        for u in locations:
            finite = (moments[index](u + step) - moments[index](u - step)) / (2 * step)
            exact = u**index * root(u)
            worst = max(worst, abs(finite - exact) / scale)
    return worst


def test_moment_derivatives_reproduce_the_integrand():
    # Kept clear of the square-root zeros at u = +/-2, where the difference
    # quotient of the primitive is dominated by the branch-point curvature.
    assert _integrand_accuracy(DOWNWARD, -1.5, 1.5) < 1e-7
    assert _integrand_accuracy(UPWARD, 2.5, 6.0) < 1e-7
    assert _integrand_accuracy(LINEAR, 0.5, 5.0) < 1e-7


def test_closed_form_matches_dense_quadrature_to_machine_precision():
    coefficients = (1.0, -0.5, 2.0, 0.75)
    for discriminant, lower, upper in (
        (DOWNWARD, -1.9, 1.9),
        (UPWARD, 2.5, 6.0),
        (LINEAR, 0.5, 5.0),
    ):
        closed = integrate_root_polynomial(discriminant, coefficients, lower, upper)
        dense = _dense(discriminant, coefficients, lower, upper)
        assert abs(closed - dense) / abs(dense) < 1e-12


def test_downward_discriminant_matches_a_hand_integral():
    # D = 4 - u**2, so the m = 0 moment over [-2, 2] is the area of a
    # semicircle of radius 2.
    moments, _ = moments_of_root(DOWNWARD)
    area = moments[0](2.0) - moments[0](-2.0)
    assert abs(area - 0.5 * np.pi * 4.0) < 1e-12


def test_perturbed_discriminant_breaks_the_dense_agreement():
    # Negative control: shift one term of the discriminant used by the CLOSED
    # FORM only, leaving the quadrature on the unperturbed one. The agreed
    # integral must separate, so the agreement above is a property of this
    # primitive rather than of the quadrature it is compared against.
    coefficients = (1.0, -0.5, 2.0, 0.75)
    lower, upper = -1.9, 1.9
    dense = _dense(DOWNWARD, coefficients, lower, upper)
    perturbed = (DOWNWARD[0] + 1e-3, DOWNWARD[1], DOWNWARD[2])
    shifted = integrate_root_polynomial(perturbed, coefficients, lower, upper)
    assert abs(shifted - dense) / abs(dense) > 1e-4
