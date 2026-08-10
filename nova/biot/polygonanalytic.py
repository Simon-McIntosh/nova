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

Organisation, which is where the cost is.  A limit of an edge is one of the
polygon's CORNERS, and most of the reduction sees only that corner: ``u`` and the
corner's own radius fix the ring modulus and its complement, and with them the
whole harmonic moment stack, the ``G^2`` split with both of its pole seeds, and
the first residual ``arsinh`` integral.  The edge's SLOPE reaches only ``B^2``,
the arctangent's numerator and the derivative blocks.  Every corner of a closed
section is the end of two edges, so the corner part is formed once and read from
both -- :class:`_Vertex` holds it and :class:`_Edge` holds the rest.

That split says something stronger about one term.  The ``arsinh beta1``
contribution is a function of the corner alone in all three components, and a
section sums each edge's antiderivative as its lower limit less its upper, so
around a closed chain of edges it CANCELS: every corner carries it twice, once
with either sign.  It survives only where a horizontal edge is dropped (paper
eq 7a -- its integrand vanishes identically), which breaks the chain at that
edge's two ends.  So it is accumulated per corner against the signed number of
live edges meeting there, and for a section with no horizontal edge it is never
formed at all.

Conditioning, which is the whole difficulty.  Every polynomial below is carried
as a :mod:`nova.biot.rangefunction` object -- two exact end values plus a
harmonic bulk -- and that module sets out why neither half of the representation
can be dropped.  What is specific to this reduction is the pole structure it
answers to.  Each denominator is a quadratic with one root just past each end of
the range,

    G^2 = 4 r^2 (y + d)(x + d),   d = u^2/(2 r (r + c)),   c^2 = u^2 + r^2
    B^2 = 4 b1^2 r^2 (y + p)(x + q)

and BOTH ``d`` and ``p`` fall as the square of the section's aspect ratio -- ``d``
because the edge's height does, ``p`` because the target's offset from the edge's
extended line does.  A root that close makes the pole's own moment large, so the
weight it carries -- the numerator's value AT that end -- has to be exact in the
relative sense, which is what the end values supply.  Splitting a denominator is
then immediate rather than a root-finding problem, its two shifts following from
its own two end values and its ``x y`` coefficient.  The loss the harmonic bulk
avoids is measured: written in powers of a range variable instead, each edge
value came out at some hundreds of ulp rather than a few, and the section sum
differences those values against each other by up to five decades.

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

from nova.biot.elliptic import (
    POLE_HEADROOM,
    cn_pole_moment,
    harmonic_moments,
    sn_pole_moment,
)
from nova.biot.gradedresidual import QUARTER, graded_residual
from nova.biot.momentchannel import Channel
from nova.biot.polygon import _held_edge, _packed_topology, pack_section
from nova.biot.rangefunction import (
    across_the_range,
    product,
    range_function,
    scaled,
    sine_squared_times,
    total,
)

__all__ = [
    "packed_analytic_greens",
    "polygon_analytic_flux",
    "polygon_analytic_greens",
]

# Harmonics the reduced numerators reach, plus one.  The arctangent term is the
# deepest: an antiderivative weight of degree three against a numerator of degree
# two over a denominator whose derivative is degree one.  One more is carried so
# the root moments, which fold each harmonic onto its neighbours, reach the same
# order as the plain ones.
_HARMONICS = 8

# Nodes for the two arsinh integrals the reduction leaves numerical, split evenly
# between the two graded panels.  What they resolve is the O(1) variation between the
# two ends, the layers themselves being carried by the grading and their logarithms
# removed in closed form.  Measured over the four acceptance sections: the whole-
# section deviation is 2.3e-10 at 64 and saturates at 6e-12 from 128 on, the gap
# being the most slender section's near-contour targets.
_NODES = 128

# Both panels span one end of the quarter range to the other: over a full turn every
# amplitude is the same right angle, so neither panel stops short of its own end.
_PANEL = (0.0, QUARTER)

# ``d/dx`` of ``G^2`` over the target radius squared, and the range variable
# itself: both are the same fixed pair of end values for every target, so they are
# built once out of scalars rather than once per column.
_RING_SLOPE_OVER_RADIUS_SQUARED = range_function([], -4.0, 4.0)
_VARIABLE = range_function([], -1.0, 1.0)


class _Vertex:
    """The reduction against one target set at one polygon CORNER.

    Everything held here is a function of the target and that corner alone.  ``u``
    and the corner's own radius fix the ring modulus and its complement, and with
    them the harmonic moment stack, its radical-weighted fold, the ``G^2`` split
    with both of its pole seeds, and the first of the two residual ``arsinh``
    integrals; none of it sees the slope of the edge the corner belongs to.  Every
    corner of a closed section is the end of two edges, so this is formed once and
    read from both.

    The moment machinery lives here too -- the contractions and the pole route --
    because it is the moment stack that decides them, and :class:`_Edge` reaches
    through to it with the numerators the slope contributes.

    ``residual`` says whether the first residual integral is wanted.  The one term
    that uses it, :meth:`arsinh_terms`, is itself a function of the corner alone
    and so cancels around an unbroken chain of edges, and there it is not formed.
    """

    def __init__(self, r, z, corner_r, corner_z, nodes, *, residual: bool, xp=np):
        self.xp = xp
        r = xp.asarray(r)
        z = xp.asarray(z)
        # r' - r is an end value a pole weight is formed from, so it comes from the
        # geometry rather than from r' less r: r' collapses to this corner's radius
        # at this limit, exactly.
        self.level = u = corner_z - z
        self.offset = offset = corner_r - r
        self.radius = r
        rp = offset + r
        # u = 0 -- a target exactly level with this corner -- drives BOTH of G^2's
        # roots onto the ends of the integration range, and the split below carries
        # that exactly: each shift is zero, and the numerator's value at the end it
        # sits on is zero with it, because every one carries either sin^2 phi or
        # u (r1 -/+ r).  The pole seeds return zero for a divergent moment on the
        # same reasoning, so no floor is needed and the configuration a grid whose
        # rows line up with the section's corners produces is evaluated rather than
        # approached.
        #
        # A target ON the corner adds r' = r and r1 = r to that, so the modulus
        # reaches one as well and the plane denominator's near root joins the ring's
        # on the range end.  What diverges there is the complete integral of the
        # first kind; the reduction's total weight on it is zero, because the
        # section's flux and field are bounded at its own corner, and the elliptic
        # module evaluates that finite part directly.
        self.radius_sum = radius_sum = r + rp
        a2 = u * u + radius_sum**2
        self.span = xp.sqrt(a2)
        parameter = 4.0 * r * rp / a2
        # 1 - k^2 = (u^2 + (r - r')^2)/a^2 -- the target's squared distance to this
        # corner over the squared ring span -- which the float parameter cannot
        # express.  EVERY complete integral is as sensitive to this complement as
        # the third-kind ones are to their own, K by eps/k'^2, so all of them are
        # given it.
        self.parameter = parameter
        self.parameter_complement = complement = (u * u + offset**2) / a2
        one = xp.ones_like(parameter)
        self.one = one

        # one order past the harmonics for the root moments, and the pole family's
        # headroom past that, since its system is closed there
        self.channel = Channel(
            harmonic_moments(
                parameter, _HARMONICS + POLE_HEADROOM + 2, complement=complement, xp=xp
            ),
            parameter,
            harmonics=_HARMONICS,
            cn_seed=lambda shift: cn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=xp
            ),
            sn_seed=lambda shift: sn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=xp
            ),
            xp=xp,
        )
        self.moments = self.channel.moments
        self.root_moments = self.channel.root_moments

        # the corner's own range functions, each with its two end values formed
        # from the geometry
        self.cosine = range_function([], one, -one)
        self.edge_radius = range_function([], offset, radius_sum)
        self.ring_squared = range_function([4.0 * r * r], u * u, u * u)
        self.ring = self.split(self.ring_squared)
        # Both residual quadratures run to the same limit -- the corner's amplitude
        # -- so the panels belong to the corner rather than to either integral, and
        # :class:`_Edge` reads them for its own.  Over a full turn every amplitude is
        # the same right angle and both panels reach their own end of the range.
        self.panels = (_PANEL, _PANEL)
        self.ring_residual = self._first_residual(nodes) if residual else None

    # The moment machinery is :class:`nova.biot.momentchannel.Channel`, which both
    # reductions share; these read through to this corner's own family so that
    # :class:`_Edge` asks the corner for a contraction rather than reaching past it.
    def split(self, denominator: tuple) -> tuple:
        """Return one denominator's two pole shifts, weights, seeds and families.

        The ring denominator is the corner's own; the plane denominator is the
        edge's, and :class:`_Edge` splits it against this corner's modulus.
        """
        return self.channel.split(denominator)

    def plain(self, term: tuple):
        """Return ``integral term/Delta da`` over the quarter range."""
        return self.channel.plain(term)

    def against_root(self, term: tuple):
        """Return ``integral term Delta da`` over the quarter range."""
        return self.channel.against_root(term)

    def across(self, numerator: tuple, split: tuple):
        """Return ``integral numerator/(denominator Delta) da``."""
        return self.channel.across(numerator, split)

    def _first_residual(self, nodes: int):
        """Evaluate the ``arsinh beta1`` integral, over the ring denominator.

        Its boundary layers are set by ``u`` -- the target's offset from this
        corner's own level -- at both ends, and its curvature by the ring span;
        :func:`nova.biot.gradedresidual.graded_residual` carries the rest.
        """
        xp = self.xp
        radius = self.radius[:, None]
        level = self.level[:, None]
        offset = self.offset[:, None]
        span = 2.0 * xp.where(self.radius > 0.0, self.radius, 1.0)

        def pieces(x, y):
            return (
                offset + 2.0 * radius * y,
                xp.sqrt(level**2 + 4.0 * radius**2 * x * y),
            )

        level_offset = xp.abs(self.level)
        return graded_residual(
            (
                (level_offset, self.radius_sum, span, *self.panels[0]),
                (level_offset, self.offset, span, *self.panels[1]),
            ),
            pieces,
            nodes,
            xp,
        )

    def arsinh_terms(self):
        """Return the ``arsinh beta1`` contribution to the three integrands.

        A function of the target and this corner alone in all three components,
        which is why the section accumulates it per corner: around an unbroken
        chain of edges every corner carries it twice with opposite signs and it
        cancels exactly.

        By parts.  ``cos^2 2a`` splits into its mean and an oscillatory part whose
        antiderivative ``sin 4a/8`` vanishes at BOTH ends, so no boundary term
        survives; the mean leaves the paper's residual arsinh quadrature and the
        rest is rational over ``G^2 D``.  Both the flux's ``u r cos phi`` weight and
        the radial field's ``r cos^2 phi`` weight land on that same integral,
        ``cos phi = -cos 2a`` making them the same shape; the vertical field's
        weight is constant, so its term is the residual quadrature itself with no
        reduction.
        """
        r = self.radius
        u = self.level
        core = scaled(
            product(
                self.cosine,
                total(
                    self.ring_squared,
                    scaled(product(self.edge_radius, self.cosine), -r),
                ),
            ),
            -1.0,
        )
        first = 0.5 * self.ring_residual + (0.5 * r / self.span) * self.across(
            sine_squared_times(across_the_range(core)), self.ring
        )
        return 4.0 * u * r * first, 4.0 * r * first, 4.0 * u * self.ring_residual


class _Edge:
    """One polygon edge's slope-dependent reduction against one target set.

    Holds what the edge's SLOPE reaches and nothing more: the plane denominator
    ``B^2``, the arctangent's numerator and the derivative blocks.  Both are
    independent of which of the edge's two limits is being evaluated, so they are
    formed once for the pair.  Everything the slope does not reach belongs to the
    corner and is held by :class:`_Vertex`, which :meth:`terms` takes.
    """

    def __init__(self, r, z, edge, nodes, *, xp=np):
        self.xp = xp
        r = xp.asarray(r)
        z = xp.asarray(z)
        ra, za, rb, zb = edge
        self.nodes = nodes
        self.radius = r
        self.slope = b1 = (rb - ra) / (zb - za)
        self.squared_slope = a02 = 1.0 + b1 * b1
        self.axial_slope = xp.sqrt(a02)
        # r1 - r is the target's offset from the edge's EXTENDED LINE, taken as the
        # cross product of its offsets to the two endpoints over the edge's height
        # rather than as r1 less r.  That vanishes EXACTLY at either endpoint, where
        # both of that endpoint's own offsets are exactly zero, and both endpoints
        # matter: a target on one is the vertex degeneracy for this edge, and a
        # target on the other still lies on the line, where the plane denominator's
        # near root sits on the range end.  Either subtraction form is exact at only
        # one of the two.
        self.plane_offset = plane_offset = (
            (ra - r) * (zb - z) - (rb - r) * (za - z)
        ) / (zb - za)
        self.plane_radius_value = r1 = r + plane_offset
        self.plane_radius = range_function([], plane_offset, r1 + r)
        self.plane_squared = range_function(
            [4.0 * b1 * b1 * r * r], plane_offset**2, (r1 + r) ** 2
        )
        self.edge_slope = range_function(
            [],
            -4.0 * r1 * r - 4.0 * b1 * b1 * r * r,
            -4.0 * r1 * r + 4.0 * b1 * b1 * r * r,
        )
        self.edge_slope_over_radius = range_function(
            [], -4.0 * r1 - 4.0 * b1 * b1 * r, -4.0 * r1 + 4.0 * b1 * b1 * r
        )

    def _second_residual(self, vertex: _Vertex):
        """Evaluate the ``arsinh beta2`` integral, over the plane denominator.

        Its boundary layers are set by the target's offset from the edge's extended
        line at one end and by the sum radius at the other, and its curvature comes
        from the denominator's OWN expansion rather than from the ring span alone:
        ``W^2 = w^2 + h^2 b^2 + O(b^4)`` at each end, and taking ``h`` from that is
        what leaves NOTHING at the end's own scale for the quadrature to resolve.
        It matters where the offset is small -- the term it adds is of relative size
        ``w`` over the ring span, exactly the size of the feature it removes.  A
        non-positive curvature means the denominator does not turn over at that end
        at all, and then its offset is of order the ring span and there is no layer
        to model.
        """
        xp = self.xp
        r = self.radius
        u = vertex.level
        b1 = self.slope
        radius = r[:, None]
        level = u[:, None]
        offset = vertex.offset
        plane_offset = self.plane_offset
        r1 = self.plane_radius_value
        held_radius = xp.where(r > 0.0, r, 1.0)
        # the slope belongs to the edge, so it is per PAIR when each pair carries its
        # own section and a scalar when they all share one; a trailing axis for the
        # quadrature's nodes covers both, being a length of one in the second case
        slope = xp.asarray(b1)[..., None]
        squared_slope = xp.asarray(self.squared_slope)[..., None]

        def pieces(x, y):
            return (
                level + slope * offset[:, None] + 2.0 * slope * radius * y,
                xp.sqrt(
                    (plane_offset[:, None] + 2.0 * radius * y) ** 2
                    + 4.0 * squared_slope * radius**2 * x * y
                ),
            )

        def curvature(coefficient):
            return xp.where(
                coefficient > 0.0,
                2.0 * xp.sqrt(xp.abs(held_radius * coefficient)),
                2.0 * self.axial_slope * held_radius,
            )

        return graded_residual(
            (
                (
                    xp.abs(r1 + r),
                    u + b1 * vertex.radius_sum,
                    curvature(b1 * b1 * held_radius - r1),
                    *vertex.panels[0],
                ),
                (
                    xp.abs(plane_offset),
                    u + b1 * offset,
                    curvature(r1 + b1 * b1 * held_radius),
                    *vertex.panels[1],
                ),
            ),
            pieces,
            self.nodes,
            xp,
        )

    def terms(self, vertex: _Vertex):
        """Return ``(W_psi, W_r, W_z)``, this edge's integrands at one limit.

        Each is ``4 integral_0^(pi/2) da`` of one of the paper's edge integrands,
        the quarter range being all the full turn needs, LESS the ``arsinh beta1``
        contribution -- which is the corner's own, and is accumulated there.  The
        flux comes from eq 10b and the two field components from eq 11b.
        """
        xp = self.xp
        one = vertex.one
        r = self.radius
        u = vertex.level
        b1 = self.slope
        a = vertex.span
        a0 = self.axial_slope
        a02 = self.squared_slope
        r1 = self.plane_radius_value
        # the plane denominator is the edge's, its split this corner's modulus
        plane = vertex.split(self.plane_squared)
        plane_residual = self._second_residual(vertex)
        gamma = range_function([], u + b1 * vertex.offset, u + b1 * vertex.radius_sum)
        # N = u X - b1 G^2.  Its value at either end collapses onto u times the
        # plane radius there, because r' - b1 u is r1 exactly -- which is what makes
        # the near-end weight u (r1 - r), of order the squared aspect ratio, a
        # product of exact quantities instead of an alternating sum.
        arctan_numerator = range_function(
            [-4.0 * b1 * r * r], u * self.plane_offset, u * (r1 + r)
        )
        # each of the arctangent's three pieces carries an explicit factor of the
        # target radius against the single factor the reduction divides by;
        # cancelling them here rather than numerically is what keeps the term finite
        # on the axis, where all three vanish together.  A range function's
        # derivative in x follows from its end values and its bulk directly:
        # d/dx (n x + f y + x y T) = (n - f) + (y - x) T for constant T.
        arctan_slope_over_radius = range_function(
            [], -2.0 * u + 4.0 * b1 * r, -2.0 * u - 4.0 * b1 * r
        )

        # the three components differ only in the weights they put on the same
        # reductions, so every product that does not carry a weight is formed once
        plane_derivative = product(gamma, self.edge_slope)
        over_ring = product(arctan_numerator, _RING_SLOPE_OVER_RADIUS_SQUARED)
        over_plane = product(arctan_numerator, self.edge_slope_over_radius)

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
            weight = across_the_range(build()) + [0.0 * one] * 4
            # the antiderivative of the oscillatory part is
            # sum_n w_n sin 2n a/(2n) = sin 2a times a polynomial in cos 2a,
            # because sin 2n a factors that way; the mean is the harmonic
            # coefficient of order zero and leaves the residual quadrature
            mean = weight[0]
            core = sine_squared_times(
                [
                    0.5 * weight[1] + weight[3] / 6.0,
                    0.5 * weight[2],
                    weight[3] / 3.0,
                ]
            )
            return (
                mean * plane_residual
                + (2.0 * b1 * r / (a0 * a)) * vertex.plain(core)
                + (0.5 / (a0 * a))
                * vertex.across(product(core, plane_derivative), plane)
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
        at_zero = 0.5 * np.pi * xp.sign(u * (r1 + r))
        at_half = xp.where(
            vertex.parameter_complement > 0.0,
            0.5 * np.pi * xp.sign(u * self.plane_offset),
            -xp.arctan(b1) * one,
        )

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
            weight = range_function([], 0.0, 0.0)
            for coefficient in reversed(primitive):
                weight = total(
                    product(weight, _VARIABLE),
                    range_function([], coefficient, coefficient),
                )

            return boundary + (0.25 / a) * (
                2.0 * vertex.plain(product(weight, arctan_slope_over_radius))
                - r * vertex.across(product(weight, over_ring), vertex.ring)
                - vertex.across(product(weight, over_plane), plane)
            )

        # the flux, eq 10b weighted by cos phi
        flux = (
            (2.0 * a / a02) * vertex.against_root(product(vertex.cosine, gamma))
            + 4.0
            * against_second_arsinh(
                lambda: scaled(
                    product(
                        vertex.cosine,
                        total(
                            self.plane_squared,
                            scaled(
                                product(vertex.cosine, self.plane_radius),
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
        radial = (4.0 * a / a02) * vertex.against_root(vertex.cosine) + 4.0 * (
            against_second_arsinh(
                lambda: scaled(
                    product(
                        vertex.cosine,
                        total(
                            range_function([], r1 * one, r1 * one),
                            scaled(vertex.cosine, b1 * b1 * r),
                        ),
                    ),
                    -b1 / (a02 * a0),
                )
            )
        )

        # the vertical field, eq 11b's third component.
        #
        # The D term is LINEAR in the edge slope.  Eq 11b prints it quadratic, which
        # agrees at slope zero and slope one and nowhere else, so a section of
        # rectangles and 45-degree edges hides it entirely.  Reducing the flux
        # antiderivative by eq 4b's own prescription -- cos phi dg/dr - (sin phi/r)
        # dg/dphi -- leaves a rational part whose B^2 denominator cancels through
        # r^2 sin^2 phi + Y^2/a0^2 = B^2/a0^2, and whose remainder collapses on
        # Gamma = a0^2 u + b1 Y and Gamma^2 + B^2 = a0^2 D^2 to exactly -b1 D/a0^2.
        vertical = (
            4.0
            * against_second_arsinh(
                lambda: scaled(
                    total(
                        range_function([], b1 * b1 * r1 * one, b1 * b1 * r1 * one),
                        scaled(vertex.cosine, -(2.0 * a02 - 1.0) * r),
                    ),
                    1.0 / (a02 * a0),
                )
            )
            - 4.0 * r * against_arctan([one])
            - (4.0 * b1 / a02) * a * vertex.root_moments[0]
        )
        return flux, radial, vertical


def _edge_terms(r, z, edge, which, nodes):
    """Return ``(W_psi, W_r, W_z)`` for one edge at one of its two limits.

    ``r`` must be positive.  ``psi`` and ``B_Z`` are even in ``r`` and ``B_R`` is
    odd, all three following from ``g(-r, phi) = g(r, pi - phi)``; the reduction's
    modulus and characteristics are defined for a positive radius only.

    The whole per-limit value, corner term included -- which is what the reduction
    is checked against edge by edge.  A section instead accumulates the corner term
    once per corner, where most of it cancels.
    """
    ra, za, rb, zb = edge
    corner_r, corner_z = (rb, zb) if which else (ra, za)
    vertex = _Vertex(r, z, corner_r, corner_z, nodes, residual=True)
    return tuple(
        term + corner
        for term, corner in zip(
            _Edge(r, z, edge, nodes).terms(vertex), vertex.arsinh_terms()
        )
    )


def _edge_flux(r, z, edge, which, nodes):
    """Return the flux integrand's full-turn angle integral for one edge limit."""
    return _edge_terms(r, z, edge, which, nodes)[0]


def _edge_field(r, z, edge, which, nodes):
    """Return the two field integrands' full-turn angle integrals for one limit."""
    return _edge_terms(r, z, edge, which, nodes)[1:]


def _axis_vertical_field(
    target_z: np.ndarray, vertices: np.ndarray, norm: float
) -> np.ndarray:
    """Return the finite symmetry-axis ``B_Z`` from a polygon boundary integral.

    A circular filament at ``(r', z')`` contributes
    ``mu0 r'^2 / (2 (r'^2 + (z-z')^2)^(3/2))`` on the axis.  Its section-area
    integral is the Green-theorem contour integral of
    ``(z-z') / hypot(r', z-z')`` with respect to ``r'``.  Each straight edge has
    the closed antiderivative below.  Exact collinearity with the axis makes its
    logarithmic coefficient zero, so the logarithm's inputs are held before
    evaluation in that case.
    """
    z = np.asarray(target_z, dtype=np.float64).ravel()
    section = np.asarray(vertices, dtype=np.float64)
    following = np.roll(section, -1, axis=0)
    contributions = []
    for (ra, za), (rb, zb) in zip(section, following, strict=True):
        dr = rb - ra
        if dr == 0.0:
            continue
        dz = zb - za
        length = np.hypot(dr, dz)
        length2 = length * length
        ua = z - za
        ub = z - zb
        distance_a = np.hypot(ra, ua)
        distance_b = np.hypot(rb, ub)
        along_a = (ra * dr - ua * dz) / length
        along_b = (rb * dr - ub * dz) / length
        perpendicular = ua + dz * along_a / length
        gap = np.abs(ra * (-dz) - ua * dr) / length
        has_logarithm = gap > 0.0
        held_gap = np.where(has_logarithm, gap, 1.0)
        logarithm = np.arcsinh(along_b / held_gap) - np.arcsinh(along_a / held_gap)
        logarithm = np.where(has_logarithm, logarithm, 0.0)
        contributions.append(
            dr
            * (
                -dz / length2 * (distance_b - distance_a)
                + perpendicular / length * logarithm
            )
        )
    if not contributions:
        return np.zeros_like(z)
    # ``norm = -mu0 / signed_area`` for the packed contour orientation.
    return -0.5 * norm * np.sum(np.stack(contributions), axis=0)


def _packed_axis_vertical_field(xp, target_z, edge, present, norm):
    """Return the exact finite axis field for one packed section per pair lane."""
    target_z = xp.asarray(target_z)
    contributions = []
    for index in range(edge.shape[0]):
        raw_ra, raw_za, raw_rb, raw_zb = edge[index]
        raw_length = xp.hypot(raw_rb - raw_ra, raw_zb - raw_za)
        active = present[index] & (raw_length > 0.0)
        # Pads and zero-length rows contribute zero.  Give those lanes a benign
        # horizontal edge before every division and logarithm is evaluated;
        # vertical material edges stay live so their geometry tangents survive.
        ra = xp.where(active, raw_ra, xp.ones_like(target_z))
        za = xp.where(active, raw_za, target_z + 1.0)
        rb = xp.where(active, raw_rb, xp.full_like(target_z, 2.0))
        zb = xp.where(active, raw_zb, target_z + 1.0)
        dr = rb - ra
        dz = zb - za
        length = xp.hypot(dr, dz)
        length2 = length * length
        ua = target_z - za
        ub = target_z - zb
        distance_a = xp.hypot(ra, ua)
        distance_b = xp.hypot(rb, ub)
        along_a = (ra * dr - ua * dz) / length
        along_b = (rb * dr - ub * dz) / length
        perpendicular = ua + dz * along_a / length
        gap = xp.abs(-ra * dz - ua * dr) / length
        has_logarithm = active & (gap > 0.0)
        held_gap = xp.where(has_logarithm, gap, 1.0)
        logarithm = xp.arcsinh(along_b / held_gap) - xp.arcsinh(along_a / held_gap)
        logarithm = xp.where(has_logarithm, logarithm, 0.0)
        contribution = dr * (
            -dz / length2 * (distance_b - distance_a)
            + perpendicular / length * logarithm
        )
        contributions.append(xp.where(active, contribution, 0.0))
    return -0.5 * norm * xp.sum(xp.stack(contributions, axis=0), axis=0)


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

    Edge ``i`` runs from corner ``i`` to corner ``i + 1``, so each corner is a
    limit of two edges and its part of the reduction -- the moment stack, the ring
    split, the first residual -- is formed once for both.  A corner is released as
    soon as the later of its two edges has been evaluated, so the build holds three
    moment stacks rather than one per corner.
    """
    edges, weights, norm = pack_section(vertices)
    signed = np.asarray(target_r, dtype=np.float64)
    height = np.asarray(target_z, dtype=np.float64)
    shape = signed.shape
    r = np.abs(signed).ravel()
    z = np.broadcast_to(height, shape).ravel()
    axis = r == 0.0
    if np.any(axis):
        rows = [np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)]
        rows[2][axis] = _axis_vertical_field(z[axis], vertices, norm)
        off_axis = ~axis
        if np.any(off_axis):
            computed = polygon_analytic_greens(
                signed.ravel()[off_axis], z[off_axis], vertices, nodes=nodes
            )
            for row, values in zip(rows, computed, strict=True):
                row[off_axis] = values
        return tuple(row.reshape(shape) for row in rows)
    flux = np.zeros_like(r)
    radial = np.zeros_like(r)
    vertical = np.zeros_like(r)
    sides = len(edges)
    live = weights != 0.0
    # The signed number of live edges leaving each corner: +1 for the edge that
    # starts there, -1 for the one that ends there, since a section sums each edge
    # as its lower limit less its upper.  Around an unbroken chain the two cancel
    # and the corner's own term is not formed at all; a dropped horizontal edge
    # leaves +/-1 at each of its two ends, which are the only corners where the
    # arsinh beta1 contribution survives.
    chain = live.astype(np.int64) - np.roll(live, 1).astype(np.int64)
    # the later of the two live edges that read each corner, which is what releases
    # its moment stack; corner 0 is read by the last edge and so held throughout
    last_read: dict[int, int] = {}
    for index in np.flatnonzero(live):
        last_read[int(index)] = int(index)
        last_read[int(index + 1) % sides] = int(index)
    corners: dict[int, _Vertex] = {}

    def corner_part(index: int, corner_r: float, corner_z: float) -> _Vertex:
        if index not in corners:
            corners[index] = _Vertex(
                r, z, corner_r, corner_z, nodes, residual=bool(chain[index])
            )
        return corners[index]

    for index in range(sides):
        if not live[index]:
            continue
        ra, za, rb, zb = edges[index]
        lower = corner_part(index, ra, za)
        upper = corner_part((index + 1) % sides, rb, zb)
        edge = _Edge(r, z, edges[index], nodes)
        # the antiderivative is of order the squared major radius where the flux is
        # not, so an edge's two limits are differenced against each other before
        # anything else is added to them
        high = edge.terms(upper)
        low = edge.terms(lower)
        flux = flux - (high[0] - low[0])
        radial = radial - (high[1] - low[1])
        vertical = vertical - (high[2] - low[2])
        # this edge's own two corners, once each -- a one-sided section would name
        # the same one twice -- and each released if this was its later edge
        for corner in dict.fromkeys((index, (index + 1) % sides)):
            if last_read[corner] != index:
                continue
            if chain[corner]:
                one_flux, one_radial, one_vertical = corners[corner].arsinh_terms()
                flux = flux + chain[corner] * one_flux
                radial = radial + chain[corner] * one_radial
                vertical = vertical + chain[corner] * one_vertical
            # nothing here holds the corner part by name, so dropping it from the
            # dictionary is what frees its moment stack
            del corners[corner]
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


def packed_analytic_greens(
    xp,
    target_r,
    target_z,
    edge,
    weight,
    norm,
    *,
    nodes: int = _NODES,
):
    """Return ``(psi, B_R, B_Z)`` per pair from the PACKED section form, unbranched.

    The same reduction as :func:`polygon_analytic_greens`, driven over the packed
    arrays :func:`nova.biot.polygon.pack_section` and
    :func:`~nova.biot.polygon.pad_batch` produce -- ``edge`` ``(E, 4, ...)``,
    ``weight`` ``(E, ...)`` and ``norm`` ``(...)``.  A dropped horizontal edge
    carries positive zero and a batch pad negative zero.  Both are arithmetic
    zeros, while the sign bit closes each heterogeneous contour at its own last
    row.  Every trailing axis broadcasts, so ONE call evaluates a whole tile of
    pairs, each pair against its own section.

    Written this way because nothing in it inspects a value.  Three things the host
    driver does with Python control flow are arithmetic here instead:

    * a dead edge is held at finite target-relative geometry BEFORE its slope or
      corner reductions are formed, then multiplied by its zero weight, so the loop
      over edges runs to a STATIC bound without evaluating a masked singularity;
    * real horizontal edges retain their endpoints, while pad corners are held at
      the same benign geometry.  The sign-bit topology makes the last real row wrap
      to row zero independently for every pair;
    * the ``arsinh beta1`` term is still accumulated ONCE per corner against the
      signed number of live edges meeting there, so it still cancels EXACTLY around
      an unbroken chain rather than to round-off.

    What that costs, and it is the whole difference between the two paths: every
    corner's pole families and first residual are formed whether the geometry needs
    them or not, because deciding needs a value.  The arithmetic is otherwise
    identical and the two agree to round-off.
    """
    signed = xp.asarray(target_r)
    radius = xp.abs(signed)
    height = xp.asarray(target_z) + xp.zeros_like(signed)
    sides = edge.shape[0]
    live, present, chain, _ = _packed_topology(xp, weight)
    axis = radius == 0.0
    evaluation_radius = xp.where(axis, xp.ones_like(radius), radius)

    def corner(index: int):
        """Return one topology-correct corner with unused pad lanes held finite."""
        present_here = present[index]
        wrap = (~present_here) & live[index - 1]
        corner_r = xp.where(present_here, edge[index][0], edge[0][0])
        corner_z = xp.where(present_here, edge[index][1], edge[0][1])
        active = present_here | wrap
        return (
            xp.where(active, corner_r, evaluation_radius + 1.0),
            xp.where(active, corner_z, height + 1.0),
        )

    lower_r, lower_z = (
        xp.stack([corner(index)[axis] for index in range(sides)], axis=0)
        for axis in range(2)
    )
    corner_r = xp.stack((lower_r, xp.roll(lower_r, -1, axis=0)), axis=0)
    corner_z = xp.stack((lower_z, xp.roll(lower_z, -1, axis=0)), axis=0)
    held = [
        _held_edge(xp, edge[index], live[index], evaluation_radius, height)
        for index in range(sides)
    ]
    held_edge = tuple(
        xp.stack([one_edge[coordinate] for one_edge in held], axis=0)
        for coordinate in range(4)
    )
    endpoint_shape = (2, sides) + evaluation_radius.shape

    def endpoint_lanes(values):
        repeated = xp.stack((values, values), axis=0)
        return xp.broadcast_to(repeated, endpoint_shape).reshape(-1)

    lane_radius = xp.broadcast_to(evaluation_radius, endpoint_shape).reshape(-1)
    lane_height = xp.broadcast_to(height, endpoint_shape).reshape(-1)
    vertex = _Vertex(
        lane_radius,
        lane_height,
        corner_r.reshape(-1),
        corner_z.reshape(-1),
        nodes,
        residual=True,
        xp=xp,
    )
    part = _Edge(
        lane_radius,
        lane_height,
        tuple(endpoint_lanes(coordinate) for coordinate in held_edge),
        nodes,
        xp=xp,
    )
    terms = tuple(value.reshape(endpoint_shape) for value in part.terms(vertex))
    residual = tuple(value.reshape(endpoint_shape) for value in vertex.arsinh_terms())

    def ordered_edge_sum(values):
        """Match the host contour order without duplicating the moment graph."""
        total = xp.zeros_like(values[0])
        for index in range(sides):
            total = total + values[index]
        return total

    rows = []
    for edge_term, corner_term in zip(terms, residual, strict=True):
        # the antiderivative is of order the squared major radius where the flux is
        # not, so each edge's two limits are differenced before the edge axis is
        # reduced.  Only the lower copy owns the corner residual; the two contour
        # passes retain the scalar host's accumulation order.
        rows.append(
            ordered_edge_sum(-weight * (edge_term[1] - edge_term[0]))
            + ordered_edge_sum(chain * corner_term[0])
        )
    flux, radial, vertical = rows
    axis_vertical = _packed_axis_vertical_field(xp, height, edge, present, norm)
    return (
        xp.where(axis, 0.0, 0.5 * norm * radius * flux),
        # B_R is ODD in r, which is what makes it exactly zero on the axis
        xp.where(
            axis,
            0.0,
            norm / (4.0 * np.pi) * xp.sign(signed) * radial,
        ),
        xp.where(axis, axis_vertical, norm / (4.0 * np.pi) * vertical),
    )
