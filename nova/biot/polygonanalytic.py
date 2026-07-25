"""Closed-form full-turn flux and field of a polygon-section ring.

The kernel in :mod:`nova.biot.polygon` reduces the cross-section surface integral
to a contour sum over the polygon's edges, does the edge-parameter integral in
closed form, and then integrates the remaining angle ``phi`` numerically.
Urankar does that last integral analytically as well (Part V, IEEE Trans. Magn.
26(3), 1171-1180, 1990), leaving per edge only two smooth ``arsinh`` integrals
that "evade an analytical treatment as yet".  This module carries out that
reduction for the FULL TURN, for the flux and for both field components.

The reduction.  Write ``phi = pi - 2 a`` (the paper's system-invariant angle
transformation) and carry BOTH range variables,

    x = sin^2 a        vanishing at a = 0    (phi = pi)
    y = cos^2 a        vanishing at a = pi/2 (phi = 0),      x + y = 1

The full-turn integrand is even about ``phi = pi``, and even again in ``a``, so
one period collapses onto four times the quarter range -- the paper's case C, in
which every elliptic integral is COMPLETE.  Per edge and per limit ``u = z' - z``
the flux integrand gives

    W(u) = 4 integral_0^(pi/2) da (-cos 2a) g(u, a)

with ``g`` the edge antiderivative of :mod:`nova.biot.polygon`, and the edge's
contribution is ``-(W(u2) - W(u1))``.  The field comes from the paper's OWN field
integrand (eq 11b) rather than from differentiating the reduced flux, which would
need the derivatives of ``K``, ``E`` and ``Pi`` with respect to both the modulus
and the characteristic; all three components are built from the same four
transcendentals with different polynomial weights and share every reduction here.
Eq 11b's azimuthal component is the radial bracket weighted by ``sin phi`` instead
of ``cos phi``: the bracket is even about ``phi = pi`` and ``sin phi`` is odd, so
the ring carries no toroidal field.  That is reproduced by the algebra rather than
assumed, and the component is not formed.

In either range variable every quantity ``g`` is built from is a POLYNOMIAL:

    cos phi = -s (2v - 1)               sin^2 phi = 4 x y
    X = r' - r cos phi                  Y  = r1 - r cos phi
    Gamma = u + b1 X                    G^2 = u^2 + 4 r^2 x y
    B^2 = Y^2 + a0^2 r^2 sin^2 phi      D   = a sqrt(1 - k^2 x)

with ``v`` the basis variable and ``s = -1`` for ``x``, ``+1`` for ``y``, and with
``a^2 = u^2 + (r + r')^2`` and ``k^2 = 4 r r'/a^2`` the usual ring modulus.  The
four terms of ``g`` then reduce as follows.

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
  denominator on its own.  This is why the paper's eq 14c carries only the ``G^2``
  family and eq 14d only the ``B^2`` family, and it matters: a joint expansion
  over the combined pole set divides by the difference of the two near-unit
  characteristics, and that difference falls as the square of the section's
  aspect ratio.

So the whole evaluation is: two ``arsinh`` quadratures, one arctangent boundary
term, and rational functions over ``G^2``, ``B^2`` and ``D``.

Conditioning, which is the whole difficulty and the reason for two bases.  Each
denominator is a quadratic in ``y``, strictly positive over the range, with one
root just past each end:

    G^2 = 4 r^2 (y + d)(x + d),   d = u^2/(2 r (r + c)),   c^2 = u^2 + r^2
    B^2 = 4 b1^2 r^2 (y + p)(x + q)

and BOTH ``d`` and ``p`` fall as the square of the section's aspect ratio -- ``d``
because the edge's height does, ``p`` because the target's offset from the edge's
extended line does.  A pole that close to the ``y = 0`` end makes every plain
moment ``integral x^m/((y + p) Delta) da`` diverge together, since ``x^m = 1``
there for every ``m``, so contracting a numerator against them rebuilds its value
AT that end as an alternating sum of coefficients of order one -- and that value
is itself of order the squared aspect ratio.  Two decades per pole family, two
families, and the loss grows as the fourth power of the aspect ratio.

The cure is to take each pole factor in the basis that vanishes at ITS OWN end,
so the numerator's value there is a LEADING COEFFICIENT rather than a sum:

    (y + shift)  ->  complement basis, moments of cos^2 a
    (x + shift)  ->  plain basis, moments of sin^2 a

and to split between the two factors rather than between the two denominators'
roots -- the factors sum to ``1 + shift_y + shift_x``, of order one however close
either root comes to the range.  Both bases are therefore built separately from
the geometry, because the complement basis's leading coefficients are the small
offsets ``r' - r``, ``r1 - r`` and their squares, exactly what a conversion from
the plain basis would reconstruct as differences of terms of order the major
radius.

Sign and unit conventions, and the ``psi [Wb/A]`` normalisation, are those of
:func:`nova.biot.polygon.polygon_greens`.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.elliptic import (
    cn_pole_moments,
    sn_pole_moments,
    stable_cn_moments,
    stable_sn_moments,
)
from nova.biot.polygon import pack_section

__all__ = ["polygon_analytic_flux", "polygon_analytic_greens"]

# Highest power of a range variable the reduced numerators reach, plus one.  The
# arctangent term is the deepest: weight cos^3 2a against a numerator of degree
# four over G^2 B^2.
_ORDERS = 9

# Orders the pole families need past the last one wanted, so that a recursion run
# downward from an arbitrary seed has converged; matches the elliptic module's own
# headroom, which sets what the moment stacks handed to them carry.
_HEADROOM = 60

# Nodes for the two arsinh integrals the reduction leaves numerical, split evenly
# between the two graded panels.  Their integrands are analytic on the open range
# and carry a boundary layer at each end whose width the panel grading absorbs, so
# what is left to resolve is the O(1) variation between: the whole-section
# deviation saturates at 64 and does not move again through 128.
_NODES = 64


def _multiply(left: list, right: list) -> list:
    """Return the product of two polynomials, coefficients ascending."""
    out: list = [0.0] * (len(left) + len(right) - 1)
    for i, one in enumerate(left):
        for j, other in enumerate(right):
            out[i + j] = out[i + j] + one * other
    return out


def _add(*polynomials: list) -> list:
    """Return the sum of polynomials, coefficients ascending."""
    out: list = [0.0] * max(len(term) for term in polynomials)
    for term in polynomials:
        for index, coefficient in enumerate(term):
            out[index] = out[index] + coefficient
    return out


def _scale(polynomial: list, factor) -> list:
    """Return the polynomial multiplied through by a scalar."""
    return [coefficient * factor for coefficient in polynomial]


def _derivative(polynomial: list) -> list:
    """Return the derivative with respect to the basis variable."""
    if len(polynomial) < 2:
        return [0.0 * polynomial[0]]
    return [
        (order + 1.0) * coefficient for order, coefficient in enumerate(polynomial[1:])
    ]


def _contract(numerator: list, basis: list):
    """Return the numerator contracted against a moment stack."""
    total = 0.0
    for order, coefficient in enumerate(numerator):
        total = total + coefficient * basis[order]
    return total


def _to_cosine(polynomial: list, degree: int, sign: float) -> list:
    """Rewrite a polynomial in the basis variable as one in ``t = cos 2a``.

    ``v = (1 + sign t)/2``, the inverse of the relation :func:`_from_cosine`
    carries.
    """
    out: list = [0.0] * (degree + 1)
    for order, coefficient in enumerate(polynomial):
        for term in range(order + 1):
            out[term] = out[term] + coefficient * comb(order, term) * sign**term / (
                2.0**order
            )
    return out


def _from_cosine(polynomial: list, degree: int, sign: float) -> list:
    """Rewrite a polynomial in ``t = cos 2a`` as one in the basis variable.

    ``t = -sign (1 - 2 v)``: the plain basis takes ``v = sin^2 a`` with
    ``sign = -1`` and the complement basis ``v = cos^2 a`` with ``sign = +1``.
    """
    out: list = [0.0] * (degree + 1)
    for order, coefficient in enumerate(polynomial):
        scaled = coefficient * (-sign) ** order
        for term in range(order + 1):
            out[term] = out[term] + scaled * comb(order, term) * (-2.0) ** term
    return out


def _split_denominator(polynomial: list, plain_constant):
    """Return the pole factors of a denominator, one per end of the range.

    ``polynomial`` is the denominator as a quadratic in ``y = cos^2 a``, and
    ``plain_constant`` its value at ``y = 1``, passed separately because that is
    the plain basis's leading coefficient and the geometry gives it exactly.

    Both of the reduction's denominators are strictly positive over ``[0, 1]``, so
    each root lies past one end -- below zero, or above one.  A root past the
    ``y = 0`` end is carried as ``(y + shift)`` and taken in the complement basis;
    one past the other end is ``(x + shift)`` in the plain basis.  The two factors
    sum to ``1 + shift_y + shift_x``, so splitting between them divides by a
    quantity of order one however close either root comes to the range.

    Roots are taken by the pivot, never as a difference: the far shift comes from
    the product of the plain-basis roots rather than from subtracting the range
    end, and the near one from the product of the complement-basis roots.  A
    vanishing leading coefficient -- a vertical edge, whose ``B^2`` is linear in
    ``y`` -- leaves one factor, on whichever side its single root falls, and a
    vanishing linear coefficient too -- a target on the axis -- leaves a constant
    denominator with no factor at all.

    Returns ``(weight_y, weight_x, weight_plain, shift_y, shift_x)``.  A dead
    factor carries a shift of one, so its moments stay finite, and a weight of
    zero; ``weight_plain`` is non-zero only when both are dead.
    """
    constant, linear, quadratic = polynomial
    curved = quadratic != 0.0
    sloped = (~curved) & (linear != 0.0)

    discriminant = linear * linear - 4.0 * constant * quadratic
    pivot = -0.5 * (
        linear + np.where(linear < 0.0, -1.0, 1.0) * np.sqrt(np.abs(discriminant))
    )
    held_pivot = np.where(pivot != 0.0, pivot, 1.0)
    held_quadratic = np.where(curved, quadratic, 1.0)
    held_linear = np.where(sloped, linear, 1.0)
    # of the two roots the pivot gives the larger in magnitude and the constant
    # over it the smaller; they straddle zero, so the lower one is the near shift
    near = np.where(
        curved,
        -np.minimum(held_pivot / held_quadratic, constant / held_pivot),
        0.0,
    )

    below = sloped & (linear > 0.0)
    above = sloped & (linear < 0.0)
    live_y = curved | below
    live_x = curved | above
    shift_y = np.where(curved, near, np.where(below, constant / held_linear, 1.0))
    shift_x = np.where(
        curved,
        plain_constant / (-held_quadratic * (1.0 + near)),
        np.where(above, -plain_constant / held_linear, 1.0),
    )

    leading = np.where(curved, -quadratic, np.where(sloped, np.abs(linear), constant))
    held_leading = np.where(leading != 0.0, leading, 1.0)
    divisor = held_leading * np.where(live_y & live_x, 1.0 + shift_y + shift_x, 1.0)
    return (
        np.where(live_y, 1.0 / divisor, 0.0),
        np.where(live_x, 1.0 / divisor, 0.0),
        np.where(live_y | live_x, 0.0, 1.0 / held_leading),
        shift_y,
        shift_x,
    )


def _edge_terms(r, z, edge, which, nodes):
    """Return ``(W_psi, W_r, W_z)`` for one edge at one of its two limits.

    Each is the full-turn angle integral of one of the paper's edge integrands,
    ``4 integral_0^(pi/2) da`` of it, the quarter range being all the full turn
    needs.  The flux comes from eq 10b and the two field components from eq 11b.

    ``r`` must be positive.  ``psi`` and ``B_Z`` are even in ``r`` and ``B_R`` is
    odd, all three following from ``g(-r, phi) = g(r, pi - phi)``; the reduction's
    modulus and characteristics are defined for a positive radius only.
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    a0 = np.sqrt(a02)
    # r1 - r and r' - r are the complement basis's leading coefficients, so both
    # are formed from the geometry rather than from r1 and r': r' collapses to the
    # edge endpoint radius at each limit, exactly
    height = za - z
    r1 = ra - b1 * height
    plane_offset = (ra - r) - b1 * height
    u = (zb - z) if which else height
    edge_offset = (rb - r) if which else (ra - r)
    # u = 0 -- a target exactly level with this end of the edge -- puts one G^2
    # root at each end of the integration range, where the factorisation the
    # reduction runs on does not exist.  The flux is continuous through u = 0, so
    # a floor keeps the evaluation finite; it does NOT make it accurate, because
    # small u is the confluent limit either way.  The floor is a fraction of the
    # edge's own height so it carries no absolute length scale.
    u = np.where(u == 0.0, 1e-9 * abs(zb - za), u)
    rp = edge_offset + r

    a2 = u * u + (r + rp) ** 2
    a = np.sqrt(a2)
    parameter = 4.0 * r * rp / a2
    # 1 - k^2 = (u^2 + (r - r')^2)/a^2, which the float parameter cannot express:
    # the third-kind integrals are as sensitive to this complement as to their own
    parameter_complement = (u * u + edge_offset**2) / a2
    one = np.ones_like(parameter)

    plain_moments = stable_sn_moments(parameter, _ORDERS + _HEADROOM)
    complement_moments = stable_cn_moments(
        parameter, _ORDERS + _HEADROOM, complement=parameter_complement
    )
    delta = [
        plain_moments[m] - parameter * plain_moments[m + 1] for m in range(_ORDERS)
    ]

    def blocks(sign):
        """Return the integrand's building blocks in one basis.

        ``sign = -1`` gives them in ``x = sin^2 a`` and ``+1`` in ``y = cos^2 a``;
        ``cos phi = -sign (2 v - 1)`` is the only difference and every block below
        follows from it.  ``d/dx`` of a block is ``-sign`` times its derivative in
        the basis variable, which is what the arctangent's reduction needs -- the
        paper's integration by parts runs in ``x``.
        """
        offset = edge_offset if sign > 0.0 else rp + r
        level = plane_offset if sign > 0.0 else r1 + r
        to_x = -sign
        plane_squared = [
            level * level,
            4.0 * sign * r * (r1 + sign * b1 * b1 * r),
            -4.0 * b1 * b1 * r * r * one,
        ]
        return {
            "sign": sign,
            "cosine": [sign * one, -2.0 * sign * one],
            "sine_squared": [0.0 * one, 4.0 * one, -4.0 * one],
            "edge_radius": [offset, 2.0 * sign * r],
            "plane_radius": [level, 2.0 * sign * r],
            "ring_squared": [u * u, 4.0 * r * r * one, -4.0 * r * r * one],
            "plane_squared": plane_squared,
            "gamma": [u + b1 * offset, 2.0 * sign * b1 * r],
            # N = u X - b1 G^2.  Its value at either end collapses onto u times the
            # plane radius there, because r' - b1 u is r1 exactly -- which is what
            # makes the near-end coefficient u (r1 - r), a leading coefficient of
            # order the squared aspect ratio instead of an alternating sum.
            "arctan_numerator": [
                u * level,
                2.0 * sign * u * r - 4.0 * b1 * r * r,
                4.0 * b1 * r * r * one,
            ],
            # each of the arctangent's three pieces carries an explicit factor of
            # the target radius against the single factor the reduction divides
            # by; cancelling them here rather than numerically is what keeps the
            # term finite on the axis, where all three vanish together
            "arctan_slope_over_radius": _scale(
                [2.0 * sign * u - 4.0 * b1 * r, 8.0 * b1 * r * one], to_x
            ),
            "ring_slope_over_radius_squared": [to_x * 4.0 * one, -to_x * 8.0 * one],
            "edge_slope": _scale(_derivative(plane_squared), to_x),
            "edge_slope_over_radius": _scale(
                [
                    4.0 * sign * (r1 + sign * b1 * b1 * r),
                    -8.0 * b1 * b1 * r * one,
                ],
                to_x,
            ),
        }

    plain = blocks(-1.0)
    complement = blocks(1.0)

    def denominator(polynomial_key, plain_constant):
        """Return one denominator's split weights with its two moment stacks."""
        weight_y, weight_x, weight_plain, shift_y, shift_x = _split_denominator(
            complement[polynomial_key], plain_constant
        )
        return (
            weight_y,
            weight_x,
            weight_plain,
            cn_pole_moments(
                shift_y,
                parameter,
                _ORDERS,
                moments=complement_moments,
                parameter_complement=parameter_complement,
            ),
            sn_pole_moments(
                shift_x,
                parameter,
                _ORDERS,
                moments=plain_moments,
                parameter_complement=parameter_complement,
            ),
        )

    ring = denominator("ring_squared", u * u)
    edge_denominator = denominator("plane_squared", (r1 + r) ** 2)

    def across(build, split):
        """Return ``integral build/(denominator Delta) da``, each end in its basis."""
        weight_y, weight_x, weight_plain, y_stack, x_stack = split
        plain_numerator = build(plain)
        return (
            weight_y * _contract(build(complement), y_stack)
            + weight_x * _contract(plain_numerator, x_stack)
            + weight_plain * _contract(plain_numerator, plain_moments)
        )

    # the two arsinh integrals the toroidal reduction leaves numerical.  Their
    # arguments are formed from x y and the complement basis's small offsets, both
    # exact, rather than by evaluating either basis's polynomial near its far end
    node, weight = leggauss(nodes // 2)
    radius = r[:, None]
    level = u[:, None]
    held_radius = np.where(r > 0.0, r, 1.0)

    def _width(scale):
        """Return the angle a boundary layer of this half-width turns over on.

        Clipped above at one, where the layer is as wide as the range and the map
        below is already near the identity, and below off zero, where the layer has
        collapsed onto the range end and the integrand is genuinely log-singular
        -- integrable, and left to the vanishing weight the map gives it there.
        """
        return np.clip(scale, 1e-13, 1.0)[:, None]

    def residual(widths, argument):
        """Return ``integral_0^(pi/2) arsinh(argument) da`` over graded panels.

        Both residual integrands carry a boundary layer at each end of the range:
        ``G^2 = u^2 + r^2 sin^2 2a`` bottoms out at ``u^2`` there and ``B^2`` at
        ``(r1 -/+ r)^2``, so each turns over on an angle that falls with the
        target's offset from the edge end's level, or from the edge's extended
        line.  A plain rule cannot see a layer narrower than its endpoint node
        spacing, and a target grid lined up with a section vertex produces offsets
        of 1e-4 section radii, where 48 nodes lose seven digits.

        The quarter range is therefore halved and each half stretched by
        ``a = width sinh(s)`` from its own end.  That is EXACT for the layer's
        leading quadratic -- ``u^2 + 4 r^2 width^2 sinh^2 s = u^2 cosh^2 s`` --
        so it moves the integrand's nearest singularity from a distance ``width``
        off the real axis to a distance ``pi/2``, and the same node count reaches
        round-off.  ``x`` and ``y`` are formed from the offset within each half
        rather than from the angle, so the one that vanishes keeps its digits.
        """
        total = 0.0
        for half, width in enumerate(widths):
            span = np.arcsinh(0.25 * np.pi / width)
            stretch = 0.5 * span * (node + 1.0)[None, :]
            offset = width * np.sinh(stretch)
            near = np.sin(offset) ** 2
            far = np.cos(offset) ** 2
            x, y = (far, near) if half else (near, far)
            jacobian = 0.5 * span * width * np.cosh(stretch)
            total = total + (jacobian * argument(x, y)) @ weight
        return total

    def ring_argument(x, y):
        return np.arcsinh(
            (edge_offset[:, None] + 2.0 * radius * y)
            / np.sqrt(level**2 + 4.0 * radius**2 * x * y)
        )

    def plane_argument(x, y):
        return np.arcsinh(
            (level + b1 * edge_offset[:, None] + 2.0 * b1 * radius * y)
            / np.sqrt(
                (plane_offset[:, None] + 2.0 * radius * y) ** 2
                + 4.0 * a02 * radius**2 * x * y
            )
        )

    ring_layer = _width(np.abs(u) / (2.0 * held_radius))
    first = residual((ring_layer, ring_layer), ring_argument)
    second = residual(
        (
            _width(np.abs(r1 + r) / (2.0 * a0 * held_radius)),
            _width(np.abs(plane_offset) / (2.0 * a0 * held_radius)),
        ),
        plane_argument,
    )

    def first_arsinh():
        """Return ``integral cos^2 2a arsinh beta1 da`` over the quarter range.

        By parts.  ``cos^2 2a`` splits into its mean and an oscillatory part whose
        antiderivative ``sin 4a/8`` vanishes at BOTH ends, so no boundary term
        survives; the mean leaves the paper's residual arsinh quadrature and the
        rest is rational over ``G^2 D``.  Both the flux's ``u r cos phi`` weight
        and the field's ``r cos^2 phi`` weight land on this same integral, ``cos
        phi = -cos 2a`` making them the same shape.
        """
        return 0.5 * first + (0.5 * r / a) * across(
            lambda block: _multiply(
                _multiply(block["sine_squared"], _scale(block["cosine"], -1.0)),
                _add(
                    block["ring_squared"],
                    _scale(_multiply(block["edge_radius"], block["cosine"]), -r),
                ),
            ),
            ring,
        )

    against_first_arsinh = first_arsinh()

    def against_second_arsinh(build):
        """Return ``integral build(v) arsinh beta2 da`` over the quarter range.

        By parts, with the weight expanded in ``cos 2n a`` so that the
        antiderivative of its oscillatory part -- a sum of ``sin 2n a / 2n`` --
        vanishes at both ends and only the mean's residual quadrature survives as
        a boundary-free term.  ``sin 2n a`` factors as ``sin 2a`` times a
        polynomial in ``cos 2a``, so what multiplies the derivative is ``sin^2 phi``
        times a polynomial, and the part of that derivative carrying ``B^2``
        cancels the denominator outright.  Weights up to degree three appear:
        three in the flux, two and one in the field.

        The harmonic coefficients are the same numbers in either basis, so they are
        taken once from the plain weighting and cast back into both.
        """
        mean, harmonic, second_harmonic, third_harmonic = _to_cosine(
            build(plain), 3, -1.0
        )
        mean = mean + 0.5 * second_harmonic
        oscillation = [
            0.5 * harmonic + 0.375 * third_harmonic - third_harmonic / 24.0,
            0.25 * second_harmonic,
            third_harmonic / 6.0,
        ]

        def core(block):
            return _multiply(
                block["sine_squared"], _from_cosine(oscillation, 2, block["sign"])
            )

        return (
            mean * second
            + (2.0 * b1 * r / (a0 * a)) * _contract(core(plain), plain_moments)
            + (0.5 / (a0 * a))
            * across(
                lambda block: _multiply(
                    core(block), _multiply(block["gamma"], block["edge_slope"])
                ),
                edge_denominator,
            )
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

        The interior part is rational over ``G^2 B^2 D`` and separates onto the two
        denominators exactly, as the module docstring derives.  This is the term
        the two bases exist for: its numerator's value at the far end of the range
        is ``u (r1 - r)`` times the weight, of order the squared aspect ratio, and
        it multiplies the largest moment either pole family has.
        """
        primitive = [0.0 * one] + [
            coefficient / (order + 1.0) for order, coefficient in enumerate(weighting)
        ]
        upper = sum(primitive)
        lower = sum(
            coefficient * (-1.0) ** order for order, coefficient in enumerate(primitive)
        )
        boundary = -0.5 * (lower * at_half - upper * at_zero)

        def weight_in(block):
            return _from_cosine(primitive, len(primitive) - 1, block["sign"])

        def over(key):
            def build(block):
                return _multiply(
                    weight_in(block),
                    _multiply(block["arctan_numerator"], block[key]),
                )

            return build

        return boundary + (0.25 / a) * (
            2.0
            * _contract(
                _multiply(weight_in(plain), plain["arctan_slope_over_radius"]),
                plain_moments,
            )
            - r * across(over("ring_slope_over_radius_squared"), ring)
            - across(over("edge_slope_over_radius"), edge_denominator)
        )

    # the flux, eq 10b weighted by cos phi
    flux = (
        (2.0 * a / a02) * _contract(_multiply(plain["cosine"], plain["gamma"]), delta)
        + 4.0 * u * r * against_first_arsinh
        + 4.0
        * against_second_arsinh(
            lambda block: _scale(
                _multiply(
                    block["cosine"],
                    _add(
                        block["plane_squared"],
                        _scale(
                            _multiply(block["cosine"], block["plane_radius"]),
                            2.0 * a02 * r,
                        ),
                    ),
                ),
                0.5 / (a02 * a0),
            )
        )
        - 4.0 * r * r * against_arctan([0.0 * one, 0.0 * one, one])
    )

    # the radial field, eq 11b's first component
    radial = (
        (4.0 * a / a02) * _contract(plain["cosine"], delta)
        + 4.0 * r * against_first_arsinh
        + 4.0
        * against_second_arsinh(
            lambda block: _scale(
                _multiply(
                    block["cosine"],
                    _add([r1 * one], _scale(block["cosine"], b1 * b1 * r)),
                ),
                -b1 / (a02 * a0),
            )
        )
    )

    # the vertical field, eq 11b's third component.  Its arsinh beta1 weight is
    # constant, so that term is the residual quadrature itself with no reduction.
    #
    # The D term is LINEAR in the edge slope.  Eq 11b prints it quadratic, which
    # agrees at slope zero and slope one and nowhere else, so a section of
    # rectangles and 45-degree edges hides it entirely.  Reducing the flux
    # antiderivative by eq 4b's own prescription -- cos phi dg/dr - (sin phi/r)
    # dg/dphi -- leaves a rational part whose B^2 denominator cancels through
    # r^2 sin^2 phi + Y^2/a0^2 = B^2/a0^2, and whose remainder collapses on
    # Gamma = a0^2 u + b1 Y and Gamma^2 + B^2 = a0^2 D^2 to exactly -b1 D/a0^2.
    vertical = (
        4.0 * u * first
        + 4.0
        * against_second_arsinh(
            lambda block: _scale(
                _add(
                    [b1 * b1 * r1 * one],
                    _scale(block["cosine"], -(2.0 * a02 - 1.0) * r),
                ),
                1.0 / (a02 * a0),
            )
        )
        - 4.0 * r * against_arctan([one])
        - (4.0 * b1 / a02) * a * delta[0]
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
    orientation; horizontal edges contribute nothing and are skipped.

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
