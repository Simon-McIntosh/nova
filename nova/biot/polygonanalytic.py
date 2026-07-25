"""Closed-form full-turn flux of a polygon-section ring: the toroidal integral done.

The kernel in :mod:`nova.biot.polygon` reduces the cross-section surface
integral to a contour sum over the polygon's edges, does the edge-parameter
integral in closed form, and then integrates the remaining angle ``phi``
numerically.  Urankar does that last integral analytically as well (Part V, IEEE
Trans. Magn. 26(3), 1171-1180, 1990), leaving per edge only two smooth
``arsinh`` integrals that "evade an analytical treatment as yet".  This module
carries out that reduction for the FULL TURN.

The reduction.  Write ``phi = pi - 2 a`` (the paper's system-invariant angle
transformation) and ``x = sin^2 a``.  The full-turn integrand is even about
``phi = pi``, and even again in ``a``, so one period collapses onto four times
the quarter range ``a`` in ``[0, pi/2]`` -- the paper's case C, in which every
elliptic integral is COMPLETE.  Per edge and per limit ``u = z' - z``:

    W(u) = 4 integral_0^(pi/2) da (-cos 2a) g(u, a)

with ``g`` the edge antiderivative of :mod:`nova.biot.polygon`, and the edge's
contribution to the flux is ``-(W(u2) - W(u1))``.  In ``x`` every quantity ``g``
is built from is a POLYNOMIAL:

    cos phi = 2x - 1                    sin^2 phi = 4x(1 - x)
    X = r' - r cos phi                  Y  = r1 - r cos phi
    Gamma = u + b1 X                    G^2 = u^2 + 4 r^2 x(1 - x)
    B^2 = Y^2 + a0^2 r^2 sin^2 phi      D   = a sqrt(1 - k^2 x)

with ``a^2 = u^2 + (r + r')^2`` and ``k^2 = 4 r r'/a^2`` the usual ring modulus.
The four terms of ``g`` then reduce as follows.

* ``Gamma D/(2 a0^2)`` is a polynomial times ``D``: complete integrals of the
  first and second kind, nothing more.
* ``u r cos phi arsinh beta1`` is integrated by parts.  Its weight ``cos^2 2a``
  splits into its mean and an oscillatory part whose antiderivative
  ``sin 4a/8`` vanishes at BOTH ends, so no boundary term survives; the mean
  leaves the paper's residual ``integral arsinh beta1 da``, and the rest becomes
  rational, over ``G^2 D``.
* the ``arsinh beta2`` term likewise, its weight expanded in ``cos 2n a`` so the
  antiderivative of the oscillatory part again vanishes at both ends; the
  residual is ``integral arsinh beta2 da`` and the rest is rational over
  ``B^2 D``.  ``B^2 + Gamma^2 = a0^2 D^2`` is what makes the derivative rational.
* ``-(r^2/2) sin 2phi arctan beta3`` has weight ``-2 r^2 sin 2a cos^2 2a``, whose
  antiderivative ``cos^3 2a`` does NOT vanish at the ends, so this one leaves an
  explicit boundary term.  There ``sin phi -> 0`` drives ``beta3`` to infinity and
  the arctangent to ``+/- pi/2`` with sign ``sign(u (r1 +/- r))`` -- the same
  dead-band the rectangle kernel carries.  The interior part is rational over
  ``G^2 B^2 D``, and it separates onto the two denominators EXACTLY: with ``N``
  the arctangent's numerator and ``W = (r sin phi D)^2``, so that
  ``G^2 B^2 = N^2 + W``, the derivative's numerator ``2 W N' - N W'`` equals
  ``2 G^2 B^2 N' - N (G^2 B^2)'``, whose quotient is
  ``2 N' - N (G^2)'/G^2 - N (B^2)'/B^2`` -- a polynomial plus one term over each
  denominator on its own.  Splitting it that way rather than by partial fractions
  over the combined pole set is what the accuracy at a slender section rests on:
  the two denominators each carry a root just past an end of the range, those
  roots approach one another as the section thins, and a joint expansion divides
  by their difference.

So the whole evaluation is: two ``arsinh`` quadratures, one arctangent boundary
term, and rational functions of ``x`` over the product of ``G^2``, ``B^2`` and
``D`` -- and each of those denominators factors into linear pieces whose moments
are the complete integrals of :mod:`nova.biot.elliptic`.

Conditioning, which is the whole difficulty.  ``G^2`` has roots at
``(r +/- c)/(2 r)`` with ``c^2 = u^2 + r^2``, and ``B^2`` has roots of opposite
sign.  A target level with a polygon vertex sends ``u -> 0`` and one ``G^2`` root
to each end of the range; a target on an edge's extended line sends ``r1 -> r``
and a ``B^2`` root to ``x = 1``.  Contracting a numerator against moments that
are individually dominated by such a near-singularity subtracts large numbers to
get a small one, so those numerators are DEFLATED instead -- the residue
evaluated at the pole, where the numerator's own vanishing is explicit.

What remains is a loss that grows with the section's aspect ratio, measured
against it in ``tests/test_biotpolygonanalytic.py``.  Each pole family's largest
moment is multiplied by the numerator's value at the END of the integration
range, an ``(section/r0)^2`` quantity that the moment contraction forms as an
alternating sum of terms of order one; two families lose two decades each.
Expanding the numerators in the variable that vanishes where the pole sits --
``cos^2 a`` for the ``a = pi/2`` end, ``sin^2 a`` for the other -- would carry
that value as a coefficient rather than a sum and is the remaining work.  Until
then a slender section at large major radius is the hard case here and the easy
case for the boundary quadrature, which is the opposite of what the cost
comparison alone would suggest.

Sign and unit conventions, and the ``psi [Wb/A]`` normalisation, are those of
:func:`nova.biot.polygon.polygon_greens`.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.elliptic import pole_moments, stable_sn_moments
from nova.biot.polygon import pack_section

__all__ = ["polygon_analytic_flux", "polygon_analytic_greens"]

# Highest power of x = sin^2 a the reduced numerators reach, plus one.  The
# arctangent term is the deepest: weight cos^3 2a against a numerator of degree
# four over G^2 B^2.
_ORDERS = 9

# Nodes for the two arsinh integrals the reduction leaves numerical.  Their
# integrands are analytic on the open range and vary on the scale of the
# target's standoff from the edge, which Gauss-Legendre endpoint clustering
# resolves for any standoff a physical section presents.
_NODES = 48

# A pole nearer the origin than this sits close enough to the integration range
# -- or to a neighbourhood of its endpoints -- that its moments are dominated by
# the near-singularity while the answer is not, and the numerator is deflated
# against it rather than contracted with its moments.
_REACH = 2.0


def _multiply(left: list, right: list) -> list:
    """Return the product of two polynomials in x, coefficients ascending."""
    out: list = [0.0] * (len(left) + len(right) - 1)
    for i, one in enumerate(left):
        for j, other in enumerate(right):
            out[i + j] = out[i + j] + one * other
    return out


def _add(*polynomials: list) -> list:
    """Return the sum of polynomials in x, coefficients ascending."""
    out: list = [0.0] * max(len(term) for term in polynomials)
    for term in polynomials:
        for index, coefficient in enumerate(term):
            out[index] = out[index] + coefficient
    return out


def _scale(polynomial: list, factor) -> list:
    """Return the polynomial multiplied through by a scalar."""
    return [coefficient * factor for coefficient in polynomial]


def _evaluate(polynomial: list, x):
    """Return the polynomial at x, by Horner."""
    out = 0.0
    for coefficient in reversed(polynomial):
        out = out * x + coefficient
    return out


def _to_cosine(polynomial: list, degree: int) -> list:
    """Rewrite a polynomial in x as one in t = cos 2a = 1 - 2x."""
    out: list = [0.0] * (degree + 1)
    for order, coefficient in enumerate(polynomial):
        for term in range(order + 1):
            out[term] = out[term] + coefficient * comb(order, term) * (-1.0) ** term / (
                2.0**order
            )
    return out


def _from_cosine(polynomial: list, degree: int) -> list:
    """Rewrite a polynomial in t = cos 2a as one in x."""
    out: list = [0.0] * (degree + 1)
    for order, coefficient in enumerate(polynomial):
        for term in range(order + 1):
            out[term] = out[term] + coefficient * comb(order, term) * (-2.0) ** term
    return out


def _derivative(polynomial: list) -> list:
    """Return d/dx of a polynomial in x, coefficients ascending."""
    if len(polynomial) < 2:
        return [0.0 * polynomial[0]]
    return [
        (order + 1.0) * coefficient for order, coefficient in enumerate(polynomial[1:])
    ]


def _pole_basis(poles, scale, parameter, moments, count):
    """Return ``integral x^m / (scale prod_i (1 - n_i x) Delta) da`` for each m.

    Partial fractions over the pole set.  Every pair of poles the reduction
    produces is separated in sign -- ``G^2`` has one root either side of the
    range and ``B^2``, having a negative leading coefficient and a positive value
    at the origin, likewise -- so the residue weights are bounded and the split
    costs nothing.  A characteristic passed as zero is not a pole at all: its own
    weight vanishes and it drops out of the others' denominators by itself, which
    is how a vertical edge's runaway root and a deflated pole are both expressed
    without a separate branch.
    """
    stacks = [
        pole_moments(
            characteristic, parameter, count, complement=complement, moments=moments
        )
        for characteristic, complement in poles
    ]
    order = len(poles) - 1
    out: list = [0.0] * count
    absent = np.ones(np.shape(parameter), dtype=bool)
    for index, (characteristic, _) in enumerate(poles):
        present = characteristic != 0.0
        absent = absent & ~present
        weight = np.where(present, characteristic, 0.0) ** order
        for other, (neighbour, _) in enumerate(poles):
            if other != index:
                weight = weight / np.where(present, characteristic - neighbour, 1.0)
        for moment in range(count):
            out[moment] = out[moment] + weight * stacks[index][moment]
    # with no pole left the product is unity, which the weights above cannot say
    return [np.where(absent, moments[m], out[m]) / scale for m in range(count)]


def _deflate(numerator: list, characteristic):
    """Split ``N(x) = (x - 1/n) M(x) + N(1/n)``; return ``(M, N(1/n))``."""
    location = 1.0 / characteristic
    quotient: list = [None] * (len(numerator) - 1)
    carry = numerator[-1]
    for index in range(len(numerator) - 2, -1, -1):
        quotient[index] = carry
        carry = numerator[index] + location * carry
    return quotient, carry


def _reduce(numerator, poles, scale, parameter, moments, count):
    """Return ``integral numerator(x) / (scale prod_i (1 - n_i x) Delta) da``.

    A pole within :data:`_REACH` of the origin is removed by deflating the
    numerator against it: the residue is evaluated AT the pole, where the
    numerator's vanishing there is carried by the evaluation rather than
    reconstructed from cancelling moments, and the deflated quotient no longer
    sees the pole.  The residue keeps its own pole and every pole not yet
    deflated, which is the pole set the expression still carries at that point.
    """
    working = list(numerator)
    live = [
        np.asarray(characteristic, dtype=np.float64) + 0.0 * parameter
        for characteristic, _ in poles
    ]
    total = 0.0
    for index, (characteristic, _) in enumerate(poles):
        with np.errstate(divide="ignore", invalid="ignore"):
            near = np.abs(1.0 / characteristic) <= _REACH
        if not np.any(near):
            continue
        held = np.where(near, characteristic, 1.0)
        quotient, residue = _deflate(working, held)
        working = [
            np.where(near, -quotient[order] / held, working[order])
            if order < len(quotient)
            else np.where(near, 0.0, working[order])
            for order in range(len(working))
        ]
        carried = [(live[i], poles[i][1]) for i in range(len(poles))]
        total = (
            total
            + np.where(near, residue, 0.0)
            * _pole_basis(carried, scale, parameter, moments, 1)[0]
        )
        live[index] = np.where(near, 0.0, live[index])
    basis = _pole_basis(
        [(live[i], poles[i][1]) for i in range(len(poles))],
        scale,
        parameter,
        moments,
        count,
    )
    for order, coefficient in enumerate(working):
        total = total + coefficient * basis[order]
    return total


def _edge_terms(r, z, edge, which, nodes):
    """Return ``(W_psi, W_r, W_z)`` for one edge at one of its two limits.

    Each is the full-turn angle integral of one of the paper's edge integrands,
    ``4 integral_0^(pi/2) da`` of it, the quarter range being all the full turn
    needs.  The flux comes from eq 10b and the two field components from eq 11b,
    which gives the field its OWN integrand rather than a derivative of the
    potential -- so all three are built from the same four transcendentals with
    different polynomial weights, and share every reduction below.  Taking the
    field this way avoids differentiating the reduced flux, which would need the
    derivatives of ``K``, ``E`` and ``Pi`` with respect to both the modulus and
    the characteristic.

    Eq 11b's azimuthal component is its radial bracket weighted by ``sin phi``
    instead of ``cos phi``.  The bracket is even about ``phi = pi`` and ``sin phi``
    is odd, so it integrates to nothing over the full turn: the ring carries no
    toroidal field, reproduced by the algebra rather than assumed, and it is not
    formed here.

    ``r`` must be positive.  ``psi`` and ``B_Z`` are even in ``r`` and ``B_R`` is
    odd, all three following from ``g(-r, phi) = g(r, pi - phi)``; the reduction's
    modulus and characteristics are defined for a positive radius only.
    """
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    a0 = np.sqrt(a02)
    r1 = ra - b1 * (za - z)
    u = (zb - z) if which else (za - z)
    # u = 0 -- a target exactly level with this end of the edge -- puts one G^2
    # root at each end of the integration range, where the factorisation the
    # reduction runs on does not exist.  The flux is continuous through u = 0, so
    # a floor keeps the evaluation finite; it does NOT make it accurate, because
    # small u is the confluent limit either way.  The floor is a fraction of the
    # edge's own height so it carries no absolute length scale.
    u = np.where(u == 0.0, 1e-9 * abs(zb - za), u)
    rp = r1 + b1 * u

    a2 = u * u + (r + rp) ** 2
    a = np.sqrt(a2)
    parameter = 4.0 * r * rp / a2
    c = np.sqrt(u * u + r * r)
    one = np.ones_like(parameter)

    edge_radius = [rp + r, -2.0 * r * one]  # X = r' - r cos phi
    plane_radius = [r1 + r, -2.0 * r * one]  # Y = r1 - r cos phi
    cosine = [-one, 2.0 * one]  # cos phi
    sine_squared = [0.0 * one, 4.0 * one, -4.0 * one]  # sin^2 phi
    g_squared = [u * u, 4.0 * r * r * one, -4.0 * r * r * one]
    b_squared = _add(
        _multiply(plane_radius, plane_radius), _scale(sine_squared, a02 * r * r)
    )
    gamma = _add([u * one], _scale(edge_radius, b1))
    arctan_numerator = _add(_scale(edge_radius, u), _scale(g_squared, -b1))

    moments = stable_sn_moments(parameter, _ORDERS + 60)
    delta = [moments[m] - parameter * moments[m + 1] for m in range(_ORDERS)]

    # G^2 = u^2 (1 - n x)(1 - n' x), roots (r +/- c)/(2 r).  The complement of
    # the first is u^2/(c + r)^2, which is how it keeps its digits as u -> 0.
    ring_poles = (
        (2.0 * r / (c + r), u * u / (c + r) ** 2),
        (-2.0 * r * (c + r) / (u * u), 1.0 + 2.0 * r * (c + r) / (u * u)),
    )
    # B^2 = c0 + c1 x + c2 x^2, leading coefficient -4 b1^2 r^2 and so of
    # opposite sign to the value at the origin: the roots straddle it.  Taking
    # the pivot rather than the roots keeps the vanishing-slope limit finite,
    # where one root runs to infinity and its characteristic to zero.
    c0, c1, c2 = b_squared
    pivot = -0.5 * (
        c1 + np.where(c1 < 0.0, -1.0, 1.0) * np.sqrt(c1 * c1 - 4.0 * c2 * c0)
    )
    # a vanishing pivot means both x coefficients vanish, so B^2 is constant and
    # carries no pole at all -- the on-axis target, and a vertical edge whose
    # plane radius passes through the axis
    constant = pivot == 0.0
    held = np.where(constant, 1.0, pivot)
    near_edge = np.where(constant, 0.0, held / c0)
    far_edge = np.where(constant, 0.0, c2 / held)
    edge_poles = ((near_edge, 1.0 - near_edge), (far_edge, 1.0 - far_edge))

    def contract(numerator, basis):
        total = 0.0
        for order, coefficient in enumerate(numerator):
            total = total + coefficient * basis[order]
        return total

    def over(numerator, poles, scale):
        return _reduce(numerator, poles, scale, parameter, moments, _ORDERS)

    # the two arsinh integrals the toroidal reduction leaves numerical
    node, weight = leggauss(nodes)
    angle = 0.25 * np.pi * (node + 1.0)
    quadrature_weight = 0.25 * np.pi * weight
    x = np.sin(angle) ** 2
    column = [
        [coefficient[:, None] for coefficient in polynomial]
        for polynomial in (edge_radius, g_squared, gamma, b_squared)
    ]
    first = (
        np.arcsinh(_evaluate(column[0], x) / np.sqrt(_evaluate(column[1], x)))
        @ quadrature_weight
    )
    second = (
        np.arcsinh(_evaluate(column[2], x) / np.sqrt(_evaluate(column[3], x)))
        @ quadrature_weight
    )

    cos2a = _scale(cosine, -1.0)
    edge_slope = _derivative(b_squared)
    # The arctangent's three pieces below each carry an explicit factor of the
    # target radius -- N' and (B^2)' one, (G^2)' two -- against the single factor
    # the reduction divides by.  Cancelling them here rather than numerically is
    # what keeps the term finite on the axis, where all three vanish together.
    ring_slope_over_radius_squared = [4.0 * one, -8.0 * one]
    edge_slope_over_radius = [4.0 * (a02 * r - (r + r1)), -8.0 * b1 * b1 * r]
    arctan_slope_over_radius = [-2.0 * (u + 2.0 * b1 * r), 8.0 * b1 * r]

    def against_first_arsinh():
        """Return ``integral cos^2 2a arsinh beta1 da`` over the quarter range.

        By parts.  ``cos^2 2a`` splits into its mean and an oscillatory part whose
        antiderivative ``sin 4a/8`` vanishes at BOTH ends, so no boundary term
        survives; the mean leaves the paper's residual arsinh quadrature and the
        rest is rational over ``G^2 D``.  Both the flux's ``u r cos phi`` weight
        and the field's ``r cos^2 phi`` weight land on this same integral, ``cos
        phi = -cos 2a`` making them the same shape.
        """
        return 0.5 * first + (0.5 * r / a) * over(
            _multiply(
                _multiply(sine_squared, _scale(cosine, -1.0)),
                _add(g_squared, _scale(_multiply(edge_radius, cosine), -r)),
            ),
            ring_poles,
            u * u,
        )

    def against_second_arsinh(weighting):
        """Return ``integral weighting(x) arsinh beta2 da`` over the quarter range.

        By parts, with the weight expanded in ``cos 2n a`` so that the
        antiderivative of its oscillatory part -- a sum of ``sin 2n a / 2n`` --
        vanishes at both ends and only the mean's residual quadrature survives as
        a boundary-free term.  ``sin 2n a`` factors as ``sin 2a`` times a
        polynomial in ``cos 2a``, so what multiplies the derivative is ``sin^2 phi``
        times a polynomial in ``x``, and the part of that derivative carrying
        ``B^2`` cancels the denominator outright.  Weights up to degree three
        appear: three in the flux, two and one in the field.
        """
        mean, harmonic, second_harmonic, third_harmonic = _to_cosine(weighting, 3)
        mean = mean + 0.5 * second_harmonic
        oscillation = _from_cosine(
            [
                0.5 * harmonic + 0.375 * third_harmonic - third_harmonic / 24.0,
                0.25 * second_harmonic,
                third_harmonic / 6.0,
            ],
            2,
        )
        core = _multiply(sine_squared, oscillation)
        return (
            mean * second
            + (2.0 * b1 * r / (a0 * a)) * contract(core, moments)
            + (0.5 / (a0 * a))
            * over(_multiply(core, _multiply(gamma, edge_slope)), edge_poles, c0)
        )

    # sin phi -> 0 at both ends drives beta3 to infinity, so the arctangent lands
    # on +/- pi/2 with these signs -- the same dead-band the rectangle kernel has
    at_zero = 0.5 * np.pi * np.sign(u * (r1 + r))
    at_half = 0.5 * np.pi * np.sign(u * (r1 - r))

    def against_arctan(weighting):
        """Return ``integral sin 2a weighting(cos 2a) arctan beta3 da``.

        By parts again, but here the antiderivative of the weight does NOT vanish
        at the ends -- ``sin 2a`` integrates to a cosine -- so unlike the two
        arsinh terms this one leaves an explicit boundary contribution, evaluated
        from the arctangent's endpoint limits above.  The weight is given as a
        polynomial in ``cos 2a`` because that is what makes its antiderivative a
        polynomial too: degree two for the flux, degree zero for the field.

        The interior part is rational over ``G^2 B^2 D``, and it separates onto the
        two denominators EXACTLY.  Writing ``N`` for the arctangent's numerator and
        ``W = (r sin phi D)^2``, so that ``G^2 B^2 = N^2 + W``, the derivative's
        numerator ``2 W N' - N W'`` equals ``2 G^2 B^2 N' - N (G^2 B^2)'`` -- the
        ``N^2`` parts cancel identically -- so dividing through leaves

            2 N' - N (G^2)'/G^2 - N (B^2)'/B^2

        a polynomial plus one term over each denominator on its own.  Reducing it
        this way rather than by partial fractions over the combined pole set is what
        holds the term at a slender section: ``G^2`` and ``B^2`` each carry a root
        just past the end of the integration range, those two roots approach one
        another as the section thins against its major radius, and a joint expansion
        divides by their difference.
        """
        primitive = [0.0 * one] + [
            coefficient / (order + 1.0) for order, coefficient in enumerate(weighting)
        ]
        upper = sum(primitive)
        lower = sum(
            coefficient * (-1.0) ** order for order, coefficient in enumerate(primitive)
        )
        boundary = -0.5 * (lower * at_half - upper * at_zero)
        weight = _from_cosine(primitive, len(primitive) - 1)
        return boundary + (0.25 / a) * (
            2.0 * contract(_multiply(weight, arctan_slope_over_radius), moments)
            - r
            * over(
                _multiply(
                    weight, _multiply(arctan_numerator, ring_slope_over_radius_squared)
                ),
                ring_poles,
                u * u,
            )
            - over(
                _multiply(weight, _multiply(arctan_numerator, edge_slope_over_radius)),
                edge_poles,
                c0,
            )
        )

    # the flux, eq 10b weighted by cos phi
    flux = (
        (2.0 * a / a02) * contract(_multiply(cosine, gamma), delta)
        + 4.0 * u * r * against_first_arsinh()
        + 4.0
        * against_second_arsinh(
            _scale(
                _multiply(
                    cosine,
                    _add(
                        b_squared,
                        _scale(_multiply(cosine, plane_radius), 2.0 * a02 * r),
                    ),
                ),
                0.5 / (a02 * a0),
            )
        )
        - 4.0 * r * r * against_arctan([0.0 * one, 0.0 * one, one])
    )

    # the radial field, eq 11b's first component
    radial = (
        (4.0 * a / a02) * contract(cosine, delta)
        + 4.0 * r * against_first_arsinh()
        + 4.0
        * against_second_arsinh(
            _scale(
                _multiply(cosine, _add([r1 * one], _scale(cosine, b1 * b1 * r))),
                -b1 / (a02 * a0),
            )
        )
    )

    # the vertical field, eq 11b's third component.  Its arsinh beta1 weight is
    # constant, so that term is the residual quadrature itself with no reduction.
    vertical = (
        4.0 * u * first
        + 4.0
        * against_second_arsinh(
            _scale(
                _add(
                    [b1 * b1 * r1 * one],
                    _scale(cosine, -(2.0 * a02 - 1.0) * r),
                ),
                1.0 / (a02 * a0),
            )
        )
        - 4.0 * r * against_arctan([one])
        - (4.0 * b1 * b1 / a02) * a * delta[0]
    )
    return flux, radial, vertical


def _edge_flux(r, z, edge, which, nodes):
    """Return the flux integrand's full-turn angle integral for one edge limit."""
    return _edge_terms(r, z, edge, which, nodes)[0]


def _edge_field(r, z, edge, which, nodes):
    """Return the two field integrands' full-turn angle integrals for one limit."""
    return _edge_terms(r, z, edge, which, nodes)[1:]


def polygon_analytic_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    nodes: int = _NODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_R, B_Z)`` per ampere at targets, all in closed form.

    Signature, sign conventions and units match
    :func:`nova.biot.polygon.polygon_greens`: psi in Wb/A, field in T/A, per
    ampere of total conductor current at uniform azimuthal current density.
    ``vertices`` is the ``(n, 2)`` array of section corners in either
    orientation; horizontal edges contribute nothing and are skipped.  Accuracy
    is governed by the section aspect ratio, as the module docstring describes.

    Unlike the shipped kernel this does not form the field by dividing a flux
    gradient by ``2 pi r``, so it stays finite on the axis, where ``B_R`` is zero
    by the parity of the reduction rather than by cancellation.
    """
    edges, weights, norm = pack_section(vertices)
    signed = np.asarray(target_r, dtype=np.float64)
    height = np.asarray(target_z, dtype=np.float64)
    shape = signed.shape
    r = np.abs(signed).ravel()
    z = np.broadcast_to(height, shape).ravel()
    flux = np.zeros_like(r)
    radial = np.zeros_like(r)
    vertical = np.zeros_like(r)
    for index in range(len(edges)):
        if weights[index] == 0.0:
            continue
        upper = _edge_terms(r, z, edges[index], 1, nodes)
        lower = _edge_terms(r, z, edges[index], 0, nodes)
        flux = flux - (upper[0] - lower[0])
        radial = radial - (upper[1] - lower[1])
        vertical = vertical - (upper[2] - lower[2])
    # the packed norm folds in the [0, pi] doubling the quarter range already has,
    # so psi keeps the 2 pi R of the total flux and the field does not
    return (
        (0.5 * norm * r * flux).reshape(shape),
        # B_R is ODD in r, which is what makes it exactly zero on the axis
        (norm / (4.0 * np.pi) * np.sign(signed).ravel() * radial).reshape(shape),
        (norm / (4.0 * np.pi) * vertical).reshape(shape),
    )


def polygon_analytic_flux(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    nodes: int = _NODES,
) -> np.ndarray:
    """Return psi [Wb/A] alone, for a caller that does not want the field."""
    return polygon_analytic_greens(target_r, target_z, vertices, nodes=nodes)[0]
