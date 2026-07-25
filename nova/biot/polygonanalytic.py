"""Closed-form full-turn flux and field of a polygon-section ring.

The kernel in :mod:`nova.biot.polygon` reduces the cross-section surface integral
to a contour sum over the polygon's edges, does the edge-parameter integral in
closed form, and then integrates the remaining angle ``phi`` numerically.
Urankar does that last integral analytically as well (Part V, IEEE Trans. Magn.
26(3), 1171-1180, 1990), leaving per edge only two smooth ``arsinh`` integrals
that "evade an analytical treatment as yet".  This module carries out that
reduction for the FULL TURN, for the flux and for both field components.

The reduction.  Write ``phi = pi - 2 a`` (the paper's system-invariant angle
transformation) and carry the two ends of the quarter range as separate
variables,

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

Every quantity ``g`` is built from is a POLYNOMIAL in ``t = cos 2a = y - x``:

    cos phi = -t                        sin^2 phi = 1 - t^2 = 4 x y
    X = r' - r cos phi                  Y  = r1 - r cos phi
    Gamma = u + b1 X                    G^2 = u^2 + r^2 sin^2 phi
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
  denominator on its own.  This is why the paper's eq 14c carries only the ``G^2``
  family and eq 14d only the ``B^2`` family, and it matters: a joint expansion
  over the combined pole set divides by the difference of the two near-unit
  characteristics, and that difference falls as the square of the section's
  aspect ratio.

So the whole evaluation is: two ``arsinh`` quadratures, one arctangent boundary
term, and rational functions over ``G^2``, ``B^2`` and ``D``.

Conditioning, which is the whole difficulty.  Two effects, and each needs its own
representation of the same polynomial.

The FIRST is ordinary basis conditioning.  The reduced numerators reach degree
six and are bounded, over the range, by roughly the squared major radius; written
in powers of a range variable their coefficients reach ten thousand times that,
because the monomial basis on a unit interval is that badly conditioned by degree
six.  Contracting such a numerator against a family of same-signed moments then
forms the answer out of terms that exceed it by as much, and the loss is real:
measured over the four acceptance sections it put each edge value at some
hundreds of ulp rather than a few, and the section sum differences those values
against each other by up to five decades.  Every numerator here is therefore
carried in the HARMONIC basis ``cos 2n a``, whose coefficients are bounded by the
function's own size, and contracted against :func:`nova.biot.elliptic.harmonic_moments`.

The SECOND is the pole structure, and it pulls the other way.  Each denominator
is a quadratic with one root just past each end of the range,

    G^2 = 4 r^2 (y + d)(x + d),   d = u^2/(2 r (r + c)),   c^2 = u^2 + r^2
    B^2 = 4 b1^2 r^2 (y + p)(x + q)

and BOTH ``d`` and ``p`` fall as the square of the section's aspect ratio -- ``d``
because the edge's height does, ``p`` because the target's offset from the edge's
extended line does.  A root that close makes the pole's own moment large, so the
weight it carries -- the numerator's value AT that end -- must be exact in the
RELATIVE sense, and that value is itself of order the squared aspect ratio.  No
harmonic series delivers it: it is an alternating sum of coefficients of order
one.  So each range function is carried as

    N = N(phi = 0) x  +  N(phi = pi) y  +  x y T

with both end values formed directly from the geometry, exactly, and only the
BULK ``T`` as a harmonic series.  Products multiply end values and sums add them,
so exactness survives the algebra; and because ``x y/(y + p)`` is bounded by one,
the rounding left in ``T`` reaches the answer unamplified however close the root
comes.  Splitting a denominator is then immediate rather than a root-finding
problem, its two shifts following from its own two end values and its ``x y``
coefficient.

A target ON a section VERTEX drives every one of those small quantities to zero at
once, and it is a working configuration rather than a contrived one: these sections
are evaluated ACROSS themselves inside a plasma bundle, so a grid row lands on a
corner by alignment.  There ``u = 0`` and ``r' = r1 = r`` together, so both of
``G^2``'s roots and ``B^2``'s near root sit ON the range end and the modulus reaches
one, where ``K`` diverges logarithmically.  Two things carry it:

* ``k'^2 = (u^2 + (r' - r)^2)/a^2`` -- the target's squared distance to the edge's
  OWN END over the squared ring span -- is formed from the geometry and handed to
  every complete integral, because a float parameter cannot carry its own complement
  and ``K`` is then wrong by ``eps/k'^2``.  That error, not the confluence, is what
  used to make a target within a micron of a corner useless.
* AT the corner the divergence itself cancels.  ``u = 0`` collapses the
  arctangent's numerator onto ``N = -b1 G^2`` exactly, so its interior part becomes
  ``-b1[(G^2)' - G^2 (B^2)'/B^2]``, and at ``phi = 0`` with ``r' = r`` the ratio
  ``G^2/B^2`` goes to ``1/a0^2`` while ``(B^2)'`` goes to ``a0^2 (G^2)'`` -- the
  bracket vanishes.  Every other term's weight on ``K`` carries a factor of
  ``sin^2 phi``, ``u`` or ``r1 -/+ r`` and vanishes on its own.  So the reduction's
  total weight on ``K`` is zero and :func:`nova.biot.elliptic._complete_kind`
  evaluates the moments' finite parts instead.

Sign and unit conventions, and the ``psi [Wb/A]`` normalisation, are those of
:func:`nova.biot.polygon.polygon_greens`.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.elliptic import (
    POLE_HEADROOM,
    cn_pole_moment,
    harmonic_moments,
    harmonic_pole_moments,
    harmonic_root_moments,
    sn_pole_moment,
)
from nova.biot.polygon import pack_section

__all__ = ["polygon_analytic_flux", "polygon_analytic_greens"]

# Harmonics the reduced numerators reach, plus one.  The arctangent term is the
# deepest: an antiderivative weight of degree three against a numerator of degree
# two over a denominator whose derivative is degree one.  One more is carried so
# the root moments, which fold each harmonic onto its neighbours, reach the same
# order as the plain ones.
_HARMONICS = 8

# Where a denominator's root is far enough past the range for the pole family's own
# moments to be contracted directly, and near enough below it for the weight on the
# pole to have to be taken out exactly.  The two routes overlap comfortably: at this
# shift the direct family decays by 0.38 per order and the deflation the other route
# runs grows by three, so each holds a couple of orders inside round-off.
_POLE_SWITCH = 0.25

# Beyond this the factor varies across the range by less than round-off, so its
# root's exact distance stops mattering and holding it here keeps the pole family's
# recursion inside the exponent range.
_POLE_CEILING = 1e12

# Narrowest boundary layer the graded panels below chase before giving up on it.
# It is now a guard rather than a trade-off: the configurations that used to collapse
# a layer onto the range end -- a target level with an edge end, or on an edge's
# extended line, or on a vertex -- are the ones whose logarithm is removed
# analytically, and their width comes out of the OTHER end quantity instead of out of
# this floor.  What is left for it to catch is a layer that is merely narrow.
_LAYER_FLOOR = 1e-8

# Nodes for the two arsinh integrals the reduction leaves numerical, split evenly
# between the two graded panels.  What they resolve is the O(1) variation between the
# two ends, the layers themselves being carried by the grading and their logarithms
# removed in closed form.  Measured over the four acceptance sections: the whole-
# section deviation is 2.3e-10 at 64 and saturates at 6e-12 from 128 on, the gap
# being the most slender section's near-contour targets.
_NODES = 128

# The two range variables as harmonic series, and their product.  ``t = cos 2a``
# is the first harmonic, so ``x = (1 - t)/2``, ``y = (1 + t)/2`` and
# ``x y = (1 - cos 4a)/8``.
_PLAIN = [0.5, -0.5]
_COMPLEMENT = [0.5, 0.5]
_BOTH = [0.125, 0.0, -0.125]


def _harmonic_multiply(left: list, right: list) -> list:
    """Return the product of two harmonic series.

    ``cos 2m a cos 2n a = (cos 2(m + n) a + cos 2|m - n| a)/2`` -- a POSITIVE
    combination, which is why a product of bounded factors keeps bounded
    coefficients here where a monomial product does not.
    """
    if not left or not right:
        return []
    out: list = [0.0] * (len(left) + len(right) - 1)
    for index, one in enumerate(left):
        for other_index, other in enumerate(right):
            term = 0.5 * one * other
            out[index + other_index] = out[index + other_index] + term
            out[abs(index - other_index)] = out[abs(index - other_index)] + term
    return out


def _harmonic_add(*series: list) -> list:
    """Return the sum of harmonic series."""
    length = max((len(term) for term in series), default=0)
    out: list = [0.0] * length
    for term in series:
        for index, coefficient in enumerate(term):
            out[index] = out[index] + coefficient
    return out


def _harmonic_scale(series: list, factor) -> list:
    """Return the harmonic series multiplied through by a scalar."""
    return [coefficient * factor for coefficient in series]


def _range(bulk: list, near, far) -> tuple:
    """Return the range function ``near x + far y + x y bulk``.

    ``near`` is its value at ``phi = 0`` (``a = pi/2``, the source point closest
    to the target in angle) and ``far`` its value at ``phi = pi``.  Both are held
    apart from the series so a pole sitting on either end multiplies an exact
    quantity; see the module docstring.
    """
    return (bulk, near, far)


def _product(left: tuple, right: tuple) -> tuple:
    """Return the product of two range functions, end values exact.

    ``x^2 = x - x y`` and ``y^2 = y - x y`` fold the squares back, leaving the
    cross term ``-(near1 - far1)(near2 - far2)`` in the bulk -- so the product's
    end values are the products of the factors' own, formed without touching the
    series.
    """
    bulk, near, far = left
    other_bulk, other_near, other_far = right
    return (
        _harmonic_add(
            _harmonic_multiply(_BOTH, _harmonic_multiply(bulk, other_bulk)),
            # each factor's own end values ride on the OTHER factor's bulk, and the
            # pair collapses onto the single two-term series they span
            _harmonic_multiply([0.5 * (near + far), 0.5 * (far - near)], other_bulk),
            _harmonic_multiply(
                [0.5 * (other_near + other_far), 0.5 * (other_far - other_near)], bulk
            ),
            [-(near - far) * (other_near - other_far)],
        ),
        near * other_near,
        far * other_far,
    )


def _sum(*terms: tuple) -> tuple:
    """Return the sum of range functions."""
    return (
        _harmonic_add(*[term[0] for term in terms]),
        sum(term[1] for term in terms),
        sum(term[2] for term in terms),
    )


def _times(term: tuple, factor) -> tuple:
    """Return the range function multiplied through by a scalar."""
    return (_harmonic_scale(term[0], factor), term[1] * factor, term[2] * factor)


def _across_the_range(term: tuple) -> list:
    """Return the range function as one harmonic series."""
    bulk, near, far = term
    return _harmonic_add(
        [0.5 * (near + far), 0.5 * (far - near)],
        _harmonic_multiply(_BOTH, bulk),
    )


def _sine_squared_times(series: list) -> tuple:
    """Return ``sin^2 phi`` times a harmonic series, as a range function.

    ``sin^2 phi = 4 x y`` vanishes at both ends, so the product's end values are
    exactly zero whatever the series is -- and a numerator carrying this factor
    puts no weight at all on either pole.  Which is why only the arctangent term,
    the one term without it, needs its end values from the geometry.
    """
    return (_harmonic_scale(series, 4.0), 0.0 * series[0], 0.0 * series[0])


def _contract(numerator: list, moments: list):
    """Return the harmonic series contracted against a moment family."""
    total = 0.0
    for order, coefficient in enumerate(numerator):
        total = total + coefficient * moments[order]
    return total


def _deflate(series: list, root):
    """Return ``(quotient, value)`` with ``series = (t - root) quotient + value``.

    Clenshaw's recursion, which is the harmonic basis's synthetic division.  Run
    downward it follows the branch that grows away from the range, so it is the
    stable direction for a root outside it -- and both denominators' roots are
    outside it by construction.
    """
    degree = len(series) - 1
    if degree < 1:
        return [], (series[0] if series else 0.0)
    quotient: list = [0.0] * degree
    upper = 0.0
    current = 0.0
    for order in range(degree, 1, -1):
        current, upper = 2.0 * series[order] + 2.0 * root * current - upper, current
        quotient[order - 1] = current
    quotient[0] = series[1] + root * current - 0.5 * upper
    return quotient, series[0] + root * quotient[0] - 0.5 * current


class _Edge:
    """One polygon edge at one of its two limits, reduced against one target set.

    Holds the moment families and the two denominator splits, so the flux and both
    field components share every elliptic evaluation and differ only in the
    polynomial weights they contract.
    """

    def __init__(self, r, z, edge, which, nodes):
        r = np.asarray(r, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        ra, za, rb, zb = edge
        self.slope = b1 = (rb - ra) / (zb - za)
        self.squared_slope = a02 = 1.0 + b1 * b1
        self.axial_slope = np.sqrt(a02)
        # r1 - r and r' - r are the end values a pole weight is formed from, so both
        # come from the geometry rather than from r1 and r': r' collapses to the
        # edge endpoint radius at each limit, exactly.
        lower_level = za - z
        upper_level = zb - z
        lower_offset = ra - r
        upper_offset = rb - r
        self.level = u = upper_level if which else lower_level
        self.offset = edge_offset = upper_offset if which else lower_offset
        # r1 - r is the target's offset from the edge's EXTENDED LINE, taken as the
        # cross product of its offsets to the two endpoints over the edge's height
        # rather than as r1 less r.  That vanishes EXACTLY at either endpoint, where
        # both of that endpoint's own offsets are exactly zero, and both endpoints
        # matter: a target on one is the vertex degeneracy for this edge, and a target
        # on the other still lies on the line, where the plane denominator's near root
        # sits on the range end.  Either subtraction form is exact at only one of the
        # two.
        self.plane_offset = plane_offset = (
            lower_offset * upper_level - upper_offset * lower_level
        ) / (zb - za)
        self.plane_radius_value = r1 = r + plane_offset
        self.radius = r
        # u = 0 -- a target exactly level with this end of the edge -- drives BOTH of
        # G^2's roots onto the ends of the integration range, and the split below
        # carries that exactly: each shift is zero, and the numerator's value at the
        # end it sits on is zero with it, because every one carries either sin^2 phi
        # or u (r1 -/+ r).  The pole seeds return zero for a divergent moment on the
        # same reasoning, so no floor is needed and the configuration a grid whose
        # rows line up with the section's corners produces is evaluated rather than
        # approached.
        #
        # A target ON the edge's endpoint -- a section VERTEX -- adds r' = r and
        # r1 = r to that, so the modulus reaches one as well and the plane
        # denominator's near root joins the ring's on the range end.  What diverges
        # there is the complete integral of the first kind; the reduction's total
        # weight on it is zero, because the section's flux and field are bounded at
        # its own corner, and the elliptic module evaluates that finite part directly.
        rp = edge_offset + r

        a2 = u * u + (r + rp) ** 2
        self.span = np.sqrt(a2)
        parameter = 4.0 * r * rp / a2
        # 1 - k^2 = (u^2 + (r - r')^2)/a^2 -- the target's squared distance to the edge
        # END over the squared ring span -- which the float parameter cannot express.
        # EVERY complete integral is as sensitive to this complement as the third-kind
        # ones are to their own, K by eps/k'^2, so all of them are given it.
        self.parameter = parameter
        self.parameter_complement = complement = (u * u + edge_offset**2) / a2
        one = np.ones_like(parameter)
        self.one = one

        # one order past the harmonics for the root moments, and the pole family's
        # headroom past that, since its system is closed there
        self.moments = harmonic_moments(
            parameter, _HARMONICS + POLE_HEADROOM + 2, complement=complement
        )
        self.root_moments = harmonic_root_moments(self.moments, parameter)

        # the range functions the four terms are built from, each with its two end
        # values formed from the geometry
        self.cosine = _range([], one, -one)
        self.edge_radius = _range([], edge_offset, rp + r)
        self.plane_radius = _range([], plane_offset, r1 + r)
        self.ring_squared = _range([4.0 * r * r], u * u, u * u)
        self.plane_squared = _range(
            [4.0 * b1 * b1 * r * r], plane_offset**2, (r1 + r) ** 2
        )
        self.gamma = _range([], u + b1 * edge_offset, u + b1 * (rp + r))
        # N = u X - b1 G^2.  Its value at either end collapses onto u times the plane
        # radius there, because r' - b1 u is r1 exactly -- which is what makes the
        # near-end weight u (r1 - r), of order the squared aspect ratio, a product of
        # exact quantities instead of an alternating sum.
        self.arctan_numerator = _range(
            [-4.0 * b1 * r * r], u * plane_offset, u * (r1 + r)
        )
        # each of the arctangent's three pieces carries an explicit factor of the
        # target radius against the single factor the reduction divides by;
        # cancelling them here rather than numerically is what keeps the term finite
        # on the axis, where all three vanish together.  A range function's
        # derivative in x follows from its end values and its bulk directly:
        # d/dx (n x + f y + x y T) = (n - f) + (y - x) T for constant T.
        self.arctan_slope_over_radius = _range(
            [], -2.0 * u + 4.0 * b1 * r, -2.0 * u - 4.0 * b1 * r
        )
        self.ring_slope_over_radius_squared = _range([], -4.0 * one, 4.0 * one)
        self.edge_slope = _range(
            [],
            -4.0 * r1 * r - 4.0 * b1 * b1 * r * r,
            -4.0 * r1 * r + 4.0 * b1 * b1 * r * r,
        )
        self.edge_slope_over_radius = _range(
            [], -4.0 * r1 - 4.0 * b1 * b1 * r, -4.0 * r1 + 4.0 * b1 * b1 * r
        )

        self.ring = self._split(self.ring_squared)
        self.plane = self._split(self.plane_squared)
        self._residual_quadrature(nodes)

    def _split(self, denominator: tuple) -> tuple:
        """Return one denominator's two pole shifts, weights and seeds.

        A denominator ``L x y + n x + f y`` factors as ``L (y + p)(x + q)`` with
        ``n = L p (1 + q)`` and ``f = L q (1 + p)``, so the shifts follow from the
        two end values and the ``x y`` coefficient without a root-finding step and
        without a subtraction: ``p`` comes from the positive root of
        ``p^2 + p(1 + f/L - n/L) - n/L``, taken by its pivot, and ``q`` from ``f``
        over ``L (1 + p)``.  Both denominators are strictly positive over the range,
        so both shifts are non-negative and each root lies past its own end.

        Partial fractions between the two FACTORS rather than between the two roots
        divides by ``1 + p + q``, of order one however close either root comes to
        the range.  A vanishing ``x y`` coefficient -- a vertical edge, whose ``B^2``
        is linear -- leaves one factor, on whichever side its single root falls, and
        a vanishing end difference too -- a target on the axis -- leaves a constant
        denominator with no factor at all.
        """
        bulk, near, far = denominator
        leading = bulk[0] if bulk else 0.0 * near
        curved = leading != 0.0
        held_leading = np.where(curved, leading, 1.0)
        offset = near / held_leading
        pivot = 1.0 + far / held_leading - offset
        shift_y = np.where(
            curved,
            2.0 * offset / (pivot + np.sqrt(pivot * pivot + 4.0 * offset)),
            0.0,
        )
        shift_x = np.where(curved, far / (held_leading * (1.0 + shift_y)), 0.0)

        rising = (~curved) & (far > near)
        falling = (~curved) & (far < near)
        gap = np.where(curved, 1.0, np.where(rising, far - near, near - far))
        held_gap = np.where(gap != 0.0, gap, 1.0)
        shift_y = np.where(rising, near / held_gap, shift_y)
        shift_x = np.where(falling, far / held_gap, shift_x)
        live_y = curved | rising
        live_x = curved | falling

        divisor = np.where(curved, held_leading * (1.0 + shift_y + shift_x), held_gap)
        shift_y = np.where(live_y, shift_y, 1.0)
        shift_x = np.where(live_x, shift_x, 1.0)
        # a root so far past the range that the factor is constant across it to
        # round-off is held here, where the family's own decay still separates its
        # orders; past that the factor IS constant and the family is P/shift
        capped_y = np.minimum(shift_y, _POLE_CEILING)
        capped_x = np.minimum(shift_x, _POLE_CEILING)
        seed_y = cn_pole_moment(
            capped_y, self.parameter, parameter_complement=self.parameter_complement
        )
        seed_x = sn_pole_moment(
            capped_x, self.parameter, parameter_complement=self.parameter_complement
        )
        return (
            np.where(live_y, 1.0 / divisor, 0.0),
            np.where(live_x, 1.0 / divisor, 0.0),
            np.where(live_y | live_x, 0.0, 1.0 / np.where(near != 0.0, near, 1.0)),
            shift_y,
            shift_x,
            seed_y,
            seed_x,
            self._family(capped_y, seed_y, False),
            self._family(capped_x, seed_x, True),
        )

    def _family(self, shift, seed, mirrored: bool):
        """Return the pole family, or nothing where no root is far enough out.

        The family is the route for a FAR root only, and a far root is the
        exception: both of the ring denominator's shifts are the edge's own height
        over the ring span, squared, and the plane's near one is the target's
        offset from the edge's line.  Skipping it when no column needs it takes a
        tridiagonal solve of forty orders out of the common build.
        """
        if not np.any(shift > _POLE_SWITCH):
            return None
        return harmonic_pole_moments(
            shift, seed, self.moments, _HARMONICS + 1, mirrored=mirrored
        )

    def plain(self, term: tuple):
        """Return ``integral term/Delta da`` over the quarter range."""
        return _contract(_across_the_range(term), self.moments)

    def against_root(self, term: tuple):
        """Return ``integral term Delta da`` over the quarter range."""
        return _contract(_across_the_range(term), self.root_moments)

    def _pole(self, numerator: tuple, shift, seed, family, mirrored: bool):
        """Return ``integral numerator/((v + shift) Delta) da`` past one end.

        Two routes, and which one holds depends on how far past the end the root
        sits.  A NEAR root is the hard case and the reason for the end values: the
        pole's own moment grows without bound as the root reaches the range, so the
        weight on it -- the numerator's value AT that end -- must be exact in the
        relative sense, which no series of harmonics delivers.  Taking it out
        analytically leaves the rest of the numerator reaching the pole only through
        ``x y/(v + shift)``, bounded by one, and a deflation of the bulk whose own
        rounding the shift then multiplies away.

        A FAR root is the easy case and the first route is the wrong one for it:
        the terms it separates grow with the shift while their sum falls, so at a
        shift of a hundred it has thrown away four decades.  There the pole is no
        pole at all -- the factor varies by a fraction of itself across the range --
        and the family's own moments, which need no exactness anywhere, are
        contracted directly.
        """
        bulk, near, far = numerator
        end, other = (far, near) if mirrored else (near, far)
        root = (1.0 if mirrored else -1.0) * (1.0 + 2.0 * shift)
        quotient, value = _deflate(bulk, root) if bulk else ([], 0.0)
        held = (
            (end * (1.0 + shift) - other * shift) * seed
            + (other - end) * self.moments[0]
            + _contract(
                _harmonic_multiply([0.5 + shift, 0.5 if mirrored else -0.5], bulk),
                self.moments,
            )
            - shift
            * (1.0 + shift)
            * (
                value * seed
                + (-2.0 if mirrored else 2.0) * _contract(quotient, self.moments)
            )
        )
        if family is None:
            return held
        return np.where(
            shift <= _POLE_SWITCH,
            held,
            _contract(_across_the_range(numerator), family),
        )

    def across(self, numerator: tuple, split: tuple):
        """Return ``integral numerator/(denominator Delta) da``.

        Split between the denominator's two factors, each taken with the range
        variable that vanishes where its own root sits.  The two factors sum to
        ``1 + p + q``, so partial fractions between them divides by a quantity of
        order one however close either root comes to the range -- where a partial
        fraction between the two ROOTS would divide by their difference, which
        falls as the square of the section's aspect ratio.
        """
        (
            weight_y,
            weight_x,
            weight_plain,
            shift_y,
            shift_x,
            seed_y,
            seed_x,
            family_y,
            family_x,
        ) = split
        return (
            weight_plain * self.plain(numerator)
            + weight_y * self._pole(numerator, shift_y, seed_y, family_y, False)
            + weight_x * self._pole(numerator, shift_x, seed_x, family_x, True)
        )

    def _residual_quadrature(self, nodes: int):
        """Evaluate the two ``arsinh`` integrals the toroidal reduction leaves.

        Both are log-singular where their denominator vanishes, which is at a range
        end whenever the target is level with an edge end or sits on the edge's
        extended line -- a grid across a section hits both by alignment.  The
        logarithm is removed ANALYTICALLY rather than resolved: ``arsinh`` splits as

            arsinh(N/W) = log(N + sqrt(N^2 + W^2)) - log W

        with the first term bounded, so subtracting a model of ``log W`` that
        matches its end behaviour leaves a bounded integrand, and the model's own
        integral is elementary.  Near either end both denominators go as
        ``sqrt(w^2 + h^2 b^2)`` in the offset ``b`` from that end -- ``w`` the
        target's offset from the edge end's level, or from the edge's line, and
        ``h`` the ring span -- so that is the model, and the SAME two quantities set
        the panel grading.  Arguments are formed from ``x y`` and the small end
        offsets, both exact, rather than by evaluating a polynomial near its far
        end.
        """
        r = self.radius
        u = self.level
        b1 = self.slope
        node, weight = leggauss(nodes // 2)
        radius = r[:, None]
        level = u[:, None]
        held_radius = np.where(r > 0.0, r, 1.0)
        limit = 0.25 * np.pi

        def model_integral(offset, scale):
            """Return ``integral_0^(pi/4) log sqrt(offset^2 + scale^2 b^2) db``.

            Elementary, and finite in both degenerate directions: on the axis the
            model is constant and this is the width times its log; at a target level
            with the edge end the model collapses onto ``scale b`` and the
            arctangent term vanishes with the offset.
            """
            held_scale = np.where(scale > 0.0, scale, 1.0)
            held_offset = np.where(offset > 0.0, offset, 1.0)
            return 0.5 * (
                limit * np.log(offset**2 + (scale * limit) ** 2)
                - 2.0 * limit
                + np.where(
                    scale > 0.0,
                    2.0
                    * offset
                    / held_scale
                    * np.arctan(held_scale * limit / held_offset),
                    2.0 * limit,
                )
            )

        def regularised(numerator, denominator, model, sign):
            """Return ``arsinh(N/W) + sign log(model)``, bounded at the range end.

            The branch follows the sign of the numerator's own value AT that end,
            which is what the logarithm's coefficient is: positive there the
            ``N + sqrt(N^2 + W^2)`` form is the stable one, negative there its
            mirror ``-log(sqrt(N^2 + W^2) - N)``, and exactly zero there means the
            numerator vanishes with the denominator and no logarithm survives at
            all -- the configuration a target ON a section vertex produces.

            Whichever branch the END picks, the numerator's sign can turn over
            INSIDE the half -- the azimuthal weight ``b1 X`` sweeps a whole ring
            span -- and there that branch's own sum cancels.  Its value is
            recovered through ``W^2`` from the other one instead, the two being
            reciprocal about it, and the other is a sum of positives exactly where
            the first is a difference.

            All three cases are then ONE logarithm: the no-logarithm branch is the
            positive one with the model replaced by unity, since ``arsinh`` is
            itself ``log(N + sqrt(N^2 + W^2)) - log W``.
            """
            # the plain root rather than the guarded one: both arguments are of
            # order the ring span here, so nothing overflows and the guard costs
            # several times the square root it protects
            root = np.sqrt(numerator * numerator + denominator * denominator)
            direct = numerator + root
            mirror = root - numerator
            positive = sign >= 0.0
            pick = np.where(positive, direct, mirror)
            other = np.where(positive, mirror, direct)
            base = np.where(
                positive == (numerator >= 0.0),
                pick,
                denominator * denominator / np.where(other > 0.0, other, 1.0),
            )
            return np.where(sign < 0.0, -1.0, 1.0) * np.log(
                base * np.where(sign != 0.0, model, 1.0) / denominator
            )

        def residual(halves, pieces):
            """Return ``integral_0^(pi/2) arsinh(N/W) da``, log removed per half.

            The quarter range is halved and each half stretched by
            ``b = width sinh(s)`` from its own end.  That map is EXACT for the
            model's quadratic -- ``w^2 + h^2 width^2 sinh^2 s = w^2 cosh^2 s`` --
            so it carries what is left of the boundary layer after the logarithm
            has gone, and the layer's own width is set by the pair the model is
            built from: the denominator's end value and the numerator's, since it
            is their ratio the ``arsinh`` turns over on.
            """
            total = 0.0
            for half, (offset, end, scale) in enumerate(halves):
                # The denominator turns over at its own end value over the ring
                # span, and the arsinh saturates where the numerator overtakes it --
                # never nearer the end than that, so grading on the denominator's
                # scale reaches both.  What is new here is what happens when that
                # scale VANISHES: with the logarithm gone the model is then exact to
                # the order that matters, nothing is left at the denominator's own
                # scale, and the remaining feature is the numerator's.  A target on a
                # section vertex sends both to zero and needs no grading at all --
                # which is the whole gain, because a floor set low enough for that
                # case is what used to thin the nodes everywhere else.
                reach = np.where(offset > 0.0, offset, np.abs(end))
                width = np.where(
                    reach > 0.0, np.clip(reach / scale, _LAYER_FLOOR, 1.0), 1.0
                )
                span = np.arcsinh(limit / width)[:, None]
                stretch = 0.5 * span * (node + 1.0)[None, :]
                held = width[:, None]
                stretched = np.sinh(stretch)
                panel = held * stretched
                # the panel never reaches a quarter turn, so the complement is a
                # subtraction rather than a second transcendental, and the map's
                # own jacobian follows from the sinh it has already taken
                near = np.sin(panel) ** 2
                x, y = (1.0 - near, near) if half else (near, 1.0 - near)
                numerator, denominator = pieces(x, y)
                sign = np.sign(end)[:, None]
                scaled = scale[:, None] * panel
                model = np.sqrt(offset[:, None] ** 2 + scaled * scaled)
                jacobian = 0.5 * span * held * np.sqrt(1.0 + stretched * stretched)
                total = (
                    total
                    + (jacobian * regularised(numerator, denominator, model, sign))
                    @ weight
                    - np.sign(end) * model_integral(offset, scale)
                )
            return total

        def ring_pieces(x, y):
            return (
                self.offset[:, None] + 2.0 * radius * y,
                np.sqrt(level**2 + 4.0 * radius**2 * x * y),
            )

        def plane_pieces(x, y):
            return (
                level + b1 * self.offset[:, None] + 2.0 * b1 * radius * y,
                np.sqrt(
                    (self.plane_offset[:, None] + 2.0 * radius * y) ** 2
                    + 4.0 * self.squared_slope * radius**2 * x * y
                ),
            )

        rp = self.offset + r
        r1 = self.plane_radius_value

        def curvature(coefficient):
            """Return the model's scale from the denominator's own expansion.

            ``W^2 = w^2 + h^2 b^2 + O(b^4)`` at each end, and taking ``h`` from that
            expansion rather than from the ring span alone is what leaves NOTHING at
            the end's own scale for the quadrature to resolve.  It matters where the
            offset is small: the term it adds is of relative size ``w`` over the ring
            span, which is exactly the size of the feature it removes.  A
            non-positive curvature means the denominator does not turn over at that
            end at all, and then its offset is of order the ring span and there is no
            layer to model.
            """
            return np.where(
                coefficient > 0.0,
                2.0 * np.sqrt(np.abs(held_radius * coefficient)),
                2.0 * self.axial_slope * held_radius,
            )

        span = 2.0 * held_radius
        self.ring_residual = residual(
            (
                (np.abs(u), rp + r, span),
                (np.abs(u), self.offset, span),
            ),
            ring_pieces,
        )
        self.plane_residual = residual(
            (
                (
                    np.abs(r1 + r),
                    u + b1 * (rp + r),
                    curvature(b1 * b1 * held_radius - r1),
                ),
                (
                    np.abs(self.plane_offset),
                    u + b1 * self.offset,
                    curvature(r1 + b1 * b1 * held_radius),
                ),
            ),
            plane_pieces,
        )

    def terms(self):
        """Return ``(W_psi, W_r, W_z)``, the three edge integrands' angle integrals.

        Each is ``4 integral_0^(pi/2) da`` of one of the paper's edge integrands,
        the quarter range being all the full turn needs.  The flux comes from
        eq 10b and the two field components from eq 11b.
        """
        one = self.one
        r = self.radius
        u = self.level
        b1 = self.slope
        a = self.span
        a0 = self.axial_slope
        a02 = self.squared_slope
        r1 = self.plane_radius_value

        def first_arsinh():
            """Return ``integral cos^2 2a arsinh beta1 da`` over the quarter range.

            By parts.  ``cos^2 2a`` splits into its mean and an oscillatory part
            whose antiderivative ``sin 4a/8`` vanishes at BOTH ends, so no boundary
            term survives; the mean leaves the paper's residual arsinh quadrature
            and the rest is rational over ``G^2 D``.  Both the flux's
            ``u r cos phi`` weight and the field's ``r cos^2 phi`` weight land on
            this same integral, ``cos phi = -cos 2a`` making them the same shape.
            """
            core = _times(
                _product(
                    self.cosine,
                    _sum(
                        self.ring_squared,
                        _times(_product(self.edge_radius, self.cosine), -r),
                    ),
                ),
                -1.0,
            )
            return 0.5 * self.ring_residual + (0.5 * r / a) * self.across(
                _sine_squared_times(_across_the_range(core)), self.ring
            )

        against_first_arsinh = first_arsinh()
        # the three components differ only in the weights they put on the same
        # reductions, so every product that does not carry a weight is formed once
        plane_derivative = _product(self.gamma, self.edge_slope)
        over_ring = _product(self.arctan_numerator, self.ring_slope_over_radius_squared)
        over_plane = _product(self.arctan_numerator, self.edge_slope_over_radius)

        def against_second_arsinh(build):
            """Return ``integral build arsinh beta2 da`` over the quarter range.

            By parts, with the weight taken in ``cos 2n a`` so that the
            antiderivative of its oscillatory part -- a sum of ``sin 2n a/2n`` --
            vanishes at both ends and only the mean's residual quadrature survives
            as a boundary-free term.  ``sin 2n a`` factors as ``sin 2a`` times a
            polynomial in ``cos 2a``, so what multiplies the derivative is
            ``sin^2 phi`` times a polynomial, and the part of that derivative
            carrying ``B^2`` cancels the denominator outright.  Weights up to degree
            three appear: three in the flux, two and one in the field.
            """
            weight = _across_the_range(build()) + [0.0 * one] * 4
            # the antiderivative of the oscillatory part is
            # sum_n w_n sin 2n a/(2n) = sin 2a times a polynomial in cos 2a,
            # because sin 2n a factors that way; the mean is the harmonic
            # coefficient of order zero and leaves the residual quadrature
            mean = weight[0]
            core = _sine_squared_times(
                [
                    0.5 * weight[1] + weight[3] / 6.0,
                    0.5 * weight[2],
                    weight[3] / 3.0,
                ]
            )
            return (
                mean * self.plane_residual
                + (2.0 * b1 * r / (a0 * a)) * self.plain(core)
                + (0.5 / (a0 * a))
                * self.across(_product(core, plane_derivative), self.plane)
            )

        # sin phi -> 0 at both ends drives beta3 to infinity, so the arctangent lands
        # on +/- pi/2 with these signs -- the same dead-band the rectangle kernel has.
        #
        # Not at the edge's own endpoint, though.  beta3's numerator collapses onto
        # u (r1 -/+ r) there, so when that vanishes the endpoint limit is set by the
        # next order in sin^2 phi instead: at phi = 0 the numerator goes as -b1 G^2
        # and the denominator as r sin phi D, and with the target ON the endpoint both
        # go as sin^2 phi with the ratio -b1 exactly -- the arctangent lands on
        # -arctan b1 rather than on zero or on the dead-band.  phi = pi keeps its zero,
        # its own numerator carrying one more power of sin phi than its denominator.
        at_zero = 0.5 * np.pi * np.sign(u * (r1 + r))
        at_half = np.where(
            self.parameter_complement > 0.0,
            0.5 * np.pi * np.sign(u * self.plane_offset),
            -np.arctan(b1) * one,
        )
        variable = _range([], -one, one)

        def against_arctan(weighting):
            """Return ``integral sin 2a weighting(cos 2a) arctan beta3 da``.

            By parts again, but here the antiderivative of the weight does NOT
            vanish at the ends -- ``sin 2a`` integrates to a cosine -- so unlike the
            two arsinh terms this one leaves an explicit boundary contribution,
            evaluated from the arctangent's endpoint limits above.  The weight is
            given as a polynomial in ``cos 2a`` because that is what makes its
            antiderivative a polynomial too: degree two for the flux, degree zero
            for the field.

            The interior part is rational over ``G^2 B^2 D`` and separates onto the
            two denominators exactly, as the module docstring derives.  This is the
            term whose end values the geometry has to supply: its numerator's value
            at the near end is ``u (r1 - r)`` times the weight, of order the squared
            aspect ratio, and it multiplies the pole moment that grows as the root
            approaches.
            """
            primitive = [0.0] + [
                coefficient / (order + 1.0)
                for order, coefficient in enumerate(weighting)
            ]
            upper = sum(primitive)
            lower = sum(
                coefficient * (-1.0) ** order
                for order, coefficient in enumerate(primitive)
            )
            boundary = -0.5 * (lower * at_half - upper * at_zero)
            # the weight is the same fixed polynomial for every target, so it is
            # built once out of scalars rather than once per column
            weight = _range([], 0.0, 0.0)
            for coefficient in reversed(primitive):
                weight = _sum(
                    _product(weight, variable), _range([], coefficient, coefficient)
                )

            return boundary + (0.25 / a) * (
                2.0 * self.plain(_product(weight, self.arctan_slope_over_radius))
                - r * self.across(_product(weight, over_ring), self.ring)
                - self.across(_product(weight, over_plane), self.plane)
            )

        # the flux, eq 10b weighted by cos phi
        flux = (
            (2.0 * a / a02) * self.against_root(_product(self.cosine, self.gamma))
            + 4.0 * u * r * against_first_arsinh
            + 4.0
            * against_second_arsinh(
                lambda: _times(
                    _product(
                        self.cosine,
                        _sum(
                            self.plane_squared,
                            _times(
                                _product(self.cosine, self.plane_radius),
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
            (4.0 * a / a02) * self.against_root(self.cosine)
            + 4.0 * r * against_first_arsinh
            + 4.0
            * against_second_arsinh(
                lambda: _times(
                    _product(
                        self.cosine,
                        _sum(
                            _range([], r1 * one, r1 * one),
                            _times(self.cosine, b1 * b1 * r),
                        ),
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
            4.0 * u * self.ring_residual
            + 4.0
            * against_second_arsinh(
                lambda: _times(
                    _sum(
                        _range([], b1 * b1 * r1 * one, b1 * b1 * r1 * one),
                        _times(self.cosine, -(2.0 * a02 - 1.0) * r),
                    ),
                    1.0 / (a02 * a0),
                )
            )
            - 4.0 * r * against_arctan([one])
            - (4.0 * b1 / a02) * a * self.root_moments[0]
        )
        return flux, radial, vertical


def _edge_terms(r, z, edge, which, nodes):
    """Return ``(W_psi, W_r, W_z)`` for one edge at one of its two limits.

    ``r`` must be positive.  ``psi`` and ``B_Z`` are even in ``r`` and ``B_R`` is
    odd, all three following from ``g(-r, phi) = g(r, pi - phi)``; the reduction's
    modulus and characteristics are defined for a positive radius only.
    """
    return _Edge(r, z, edge, which, nodes).terms()


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
