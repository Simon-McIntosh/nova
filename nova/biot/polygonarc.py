"""Closed-form finite-arc potential and field of a polygon-section conductor.

:mod:`nova.biot.polygonanalytic` carries Urankar's Part V reduction over a FULL
TURN, where the paper's angle transformation collapses one period onto four
times a quarter range and every elliptic integral is complete.  This module is
the same reduction with the sweep stopped short.  What changes is not one
parameter but five things, and each is answered by a piece already landed:

* the integrals stop being complete -- :mod:`nova.biot.incompleteelliptic`;
* every plain moment gains a boundary term -- :mod:`nova.biot.incompletemoments`;
* the amplitude becomes a per-target field with three printed cases, which are
  one formula in the half-turn count -- :mod:`nova.biot.arcamplitude`;
* two rows appear that a full turn's parity annihilates, and they contract
  against a moment family the ring has no counterpart for --
  :func:`nova.biot.incompletemoments.sine_moments`;
* every integration by parts the reduction performs regains a boundary term at
  the interior end, which is what is derived below.

**The five rows.**  Per edge, per limit ``u = z' - z``, per row ``l``, write

    X_l(alpha) = integral_0^alpha da w_l g_l

with ``cos phi = -cos 2a`` and ``sin phi = sin 2a``.  The five weights are

    A_r     w = -sin phi   integrand g (eq 9)              fold EVEN
    A_phi   w =  cos phi   integrand g                     fold ODD
    B_r     w =  cos phi   integrand eq 11b's bracket      fold ODD
    B_phi   w =  sin phi   integrand eq 11b's bracket      fold EVEN
    B_z     w =  1         integrand eq 11b's z row        fold ODD

and :func:`nova.biot.arcamplitude.fold` turns the pair of amplitudes into the
arc's value.  An ODD row's antiderivative is odd in the amplitude and half-turn
periodic only up to ``2 X_l(pi/2)`` per half turn, so it supplies that quarter
value; an EVEN row is half-turn periodic outright and does not.  The three rows
that supply one are exactly the three the full turn forms, so the quarter comes
from :mod:`nova.biot.polygonanalytic` itself rather than being re-derived -- and
a full sweep then reproduces the ring by the fold's own arithmetic.

**Which family each term lands in.**  Every transcendental's derivative carries
exactly one explicit factor of ``sin 2a`` and the by-parts weight supplies the
other, so every term of a ``cos phi`` row contracts against the plain harmonic
moments and every term of a ``sin phi`` row against the ``sin 2a``-weighted ones.
No term mixes them, and :class:`nova.biot.momentchannel.Channel` is one family
with the contractions and the pole route that both share.

**The by-parts constants, which are not free.**  For a ``cos phi`` row the weight
is an even series and splits into its mean and an oscillatory part whose
antiderivative is a sum of ``sin 2na/(2n)``; that vanishes at ``a = 0`` on its own
and at a quarter turn as well, so only the interior UPPER end is new, and it is
that sum at the amplitude times the transcendental there.  For a ``sin phi`` row
the weight is a pure sine series and the constant has to be chosen, and the two
residual terms want OPPOSITE choices:

* ``arsinh beta1`` takes the antiderivative vanishing at the LOWER limit,
  ``sin^2 2a/2`` for a ``sin 4a`` weight.  It diverges at ``a = 0`` for a target
  level with the corner, and only that zero makes the product finite -- taking
  ``-cos 4a/4`` instead puts a divergence into a term that has none.  It happens
  to vanish at a quarter turn too, so this row needs nothing at the other end.
* ``arsinh beta2`` takes the one vanishing at the UPPER limit, ``W - W(alpha)``.
  Its antiderivative in the range variable does NOT vanish at the far end of the
  range, and the transcendental it would multiply there runs to a logarithm
  whenever the target sits on the edge's extended line -- which at a quarter-turn
  amplitude is the configuration a full sweep IS.  Subtracting the value at the
  amplitude puts an exact zero on that weight, and the near-pole seed, which
  diverges with it, is then multiplied by a weight that vanishes as fast.  What
  is left at the other end is a logarithm at ``r1 = -r``, which needs the edge's
  extended line to cross the axis at the target's own height.

**A pure sine weight has no mean**, so the ``sin phi`` rows need no ``arsinh``
quadrature at all: both residual integrals stay on the ``cos phi`` rows.

**The arctangent.**  The ``cos phi`` rows keep the full turn's antiderivative
``-(1/2) P(t)``, so the ``a = 0`` dead band survives as it is -- ``+/- pi/2`` with
the sign of ``u (r1 + r)`` -- and the upper boundary becomes the arctangent
itself, taken as ``arctan2`` of its numerator against ``r sin 2a D``.  That
denominator is non-negative over the range, so the two-argument form is the
principal value and collapses onto the dead band of its own accord at either end.
The ``sin phi`` row's arctangent weight is ``sin^2 2a cos 2a``, whose
antiderivative ``-(r^2/6) sin^3 2a`` vanishes at both ends, so only the upper
boundary survives and no dead band is formed.

Sign and unit conventions are eq 3a/4a's: every row is returned per ampere of
total conductor current at uniform azimuthal current density, in the TARGET's own
cylindrical basis, and ``2 pi r A_phi`` is the flux the full turn returns as
``psi``.
"""

from __future__ import annotations

import numpy as np

from nova.biot.arcamplitude import arc_limits, fold
from nova.biot.elliptic import POLE_HEADROOM
from nova.biot.gradedresidual import graded_residual
from nova.biot.incompletemoments import (
    cn_pole_moment,
    harmonic_cosines,
    harmonic_moments,
    sine_cn_pole_moment,
    sine_moments,
    sine_sn_pole_moment,
    sn_pole_moment,
)
from nova.biot.momentchannel import Channel, factorise
from nova.biot.polygon import pack_section
from nova.biot.polygonanalytic import _NODES, _Edge as _RingEdge, _Vertex as _RingVertex
from nova.biot.rangefunction import (
    across_the_range,
    product,
    range_function,
    rising_integral,
    scaled,
    sine_squared_times,
    total,
)

__all__ = ["packed_arc_greens", "polygon_arc_greens"]

# Harmonics the reduced numerators reach, plus one -- the full turn's count, and
# for the same reason: the arctangent term is the deepest, an antiderivative
# weight of degree three against a numerator of degree two over a denominator
# whose derivative is degree one, and the root moments fold each harmonic onto
# its neighbours.  The odd rows' antiderivative raises its weight's degree by one
# where the even rows' lowers it, so neither reaches further than this.
_HARMONICS = 8

# The rows, in the order every tuple below carries them.  Which family each
# contracts against follows from its weight: the plain harmonic moments for a
# cos-phi row, the sin 2a-weighted ones for a sin-phi row.
_ROWS = ("A_r", "A_phi", "B_r", "B_phi", "B_z")

# One end of the reduction's angle range to the other.
_HALF_TURN = 0.5 * np.pi

# ``d/dx`` of ``G^2`` over the target radius squared, and the range variable
# itself: both are the same fixed pair of end values for every target.
_RING_SLOPE_OVER_RADIUS_SQUARED = range_function([], -4.0, 4.0)
_VARIABLE = range_function([], -1.0, 1.0)


def _horner(coefficients, variable):
    """Return the polynomial in ``t`` evaluated at the amplitude's own ``cos 2a``."""
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * variable + coefficient
    return value


class _Vertex:
    """The arc's reduction against one target set, at one CORNER and one amplitude.

    The full turn's :class:`nova.biot.polygonanalytic._Vertex` holds the same
    corner geometry; what is added here is the amplitude, which the arc's two ends
    make a per-target field.  It reaches into everything: both moment families
    stop there, both residual quadratures stop there, and every by-parts boundary
    term is evaluated there -- so a corner is held once per LIMIT rather than once,
    and the two are independent evaluations sharing only the geometry.

    Two families rather than one.  The plain harmonic moments serve the three rows
    weighted by ``cos phi`` and the ``sin 2a``-weighted ones the two weighted by
    ``sin phi``; a denominator is factorised ONCE and seeded twice, since the
    shifts belong to the denominator and the seeds to the family.
    """

    def __init__(self, r, z, corner, angle, nodes, *, residual: bool, xp=np):
        self.xp = xp
        r = xp.asarray(r)
        z = xp.asarray(z)
        corner_r, corner_z = corner
        # r' - r is an end value a pole weight is formed from, so it comes from the
        # geometry rather than from r' less r: r' collapses to this corner's radius
        # at this limit, exactly.
        self.level = u = corner_z - z
        self.offset = offset = corner_r - r
        self.radius = r
        rp = offset + r
        self.radius_sum = radius_sum = r + rp
        a2 = u * u + radius_sum**2
        self.span = xp.sqrt(a2)
        parameter = 4.0 * r * rp / a2
        self.parameter = parameter
        self.parameter_complement = complement = (u * u + offset**2) / a2
        one = xp.ones_like(parameter)
        self.one = one

        # The amplitude as its own (sine, cosine) pair rather than as an angle, for
        # the reason :mod:`nova.biot.incompleteelliptic` sets out: a target in the
        # plane of an arc's end has a cosine of exactly zero there and 6e-17 if the
        # angle is assembled first, and that end is what the full turn IS.  Every
        # range variable below inherits the exactness, so a quarter-turn amplitude
        # puts x, y and cos 2a on exactly (1, 0, -1).
        amplitude, sine, cosine = angle
        self.amplitude = amplitude = xp.asarray(amplitude) + xp.zeros_like(r)
        self.amplitude_sine = sine = xp.asarray(sine) + xp.zeros_like(r)
        self.amplitude_cosine = cosine = xp.asarray(cosine) + xp.zeros_like(r)
        self.sine_squared = sine * sine
        self.cosine_squared = cosine * cosine
        self.variable = (cosine - sine) * (cosine + sine)
        self.sine_double = 2.0 * sine * cosine
        # the radical AT the amplitude, from the two exact quantities rather than
        # from 1 - k^2 sin^2 a, which is a cancellation at a target on the ring
        self.radical = xp.sqrt(complement + parameter * cosine * cosine)
        self.harmonics = harmonic_cosines(sine, cosine, _HARMONICS + 6, xp=xp)

        # Both residual quadratures still have their boundary layers at a = 0 and
        # a = pi/2, because that is where the denominators vanish, but the range
        # stops at the amplitude and the upper layer may now be outside it.  The
        # range is halved and each panel graded from its own end in the offset from
        # it; at a quarter turn the two are the full turn's own.
        self.panels = (
            (0.0 * amplitude, 0.5 * amplitude),
            (_HALF_TURN - amplitude, _HALF_TURN - 0.5 * amplitude),
        )

        count = _HARMONICS + POLE_HEADROOM + 2
        common = dict(complement=complement, sine=sine, cosine=cosine, xp=xp)
        self.channels = (
            Channel(
                harmonic_moments(amplitude, parameter, count, **common),
                parameter,
                harmonics=_HARMONICS,
                cn_seed=lambda shift: cn_pole_moment(
                    shift, parameter, sine, cosine, complement=complement, xp=xp
                ),
                sn_seed=lambda shift: sn_pole_moment(
                    shift, parameter, sine, cosine, complement=complement, xp=xp
                ),
                xp=xp,
            ),
            Channel(
                sine_moments(amplitude, parameter, count, **common),
                parameter,
                harmonics=_HARMONICS,
                cn_seed=lambda shift: sine_cn_pole_moment(
                    shift, parameter, sine, cosine, complement=complement, xp=xp
                ),
                sn_seed=lambda shift: sine_sn_pole_moment(
                    shift, parameter, sine, cosine, complement=complement, xp=xp
                ),
                xp=xp,
            ),
        )

        # the corner's own range functions, each with its two end values formed
        # from the geometry
        self.cosine = range_function([], one, -one)
        self.edge_radius = range_function([], offset, radius_sum)
        self.ring_squared = range_function([4.0 * r * r], u * u, u * u)
        self.ring = self.split(self.ring_squared)
        self.ring_residual = self._first_residual(nodes) if residual else None

    def split(self, denominator: tuple) -> tuple:
        """Return the denominator's factorisation seeded against BOTH families.

        The shifts and the partial-fraction weights belong to the denominator and
        the seeds to the family, so a reduction carrying two families factorises
        once and seeds twice.  Indexed by :data:`_CHANNEL`.
        """
        factors = factorise(denominator, self.xp)
        return tuple((factors, channel.poles(factors)) for channel in self.channels)

    def at_amplitude(self, series: list):
        """Return a harmonic series evaluated at the amplitude."""
        return sum(
            coefficient * harmonic
            for coefficient, harmonic in zip(series, self.harmonics)
        )

    def value(self, term: tuple):
        """Return a range function evaluated at the amplitude."""
        bulk, near, far = term
        x, y = self.sine_squared, self.cosine_squared
        return near * x + far * y + x * y * self.at_amplitude(bulk)

    def _first_residual(self, nodes: int):
        """Evaluate the ``arsinh beta1`` integral to the amplitude, over ``G^2``."""
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

    def _first_arsinh(self):
        """Return ``arsinh beta1`` AT the amplitude, the new upper boundary value.

        ``G`` vanishes only where the target is level with this corner and the
        amplitude has reached one of the range's own ends -- and there every weight
        that multiplies this carries ``sin 2a`` or its square, both exactly zero,
        so the branch returns the value the product needs rather than an infinity
        it would have to cancel.
        """
        xp = self.xp
        x, y = self.sine_squared, self.cosine_squared
        numerator = self.offset * x + self.radius_sum * y
        denominator = xp.sqrt(self.level**2 + 4.0 * self.radius**2 * x * y)
        held = xp.where(denominator > 0.0, denominator, 1.0)
        return xp.where(denominator > 0.0, xp.arcsinh(numerator / held), 0.0)

    def arsinh_terms(self):
        """Return the ``arsinh beta1`` contribution to all five rows.

        A function of the target and this corner alone in every row, exactly as it
        is over a full turn -- the amplitude is shared by every corner -- so the
        section still accumulates it per corner and it still cancels around an
        unbroken chain of edges.

        By parts, the even rows' weight ``cos^2 2a`` splits into its mean, which
        leaves the residual quadrature, and ``sin 4a/8``, which is the new upper
        boundary.  The odd rows' weight is ``sin 4a/2`` whose antiderivative
        ``sin^2 2a/2`` vanishes at BOTH ends of the range and so needs no residual
        quadrature and no lower boundary -- and must be taken as that one rather
        than as ``-cos 4a/4``, since ``arsinh beta1`` diverges at ``a = 0`` for a
        target level with the corner and only the zero makes the product finite.
        """
        r = self.radius
        u = self.level
        even, odd = self.channels
        bare = total(
            self.ring_squared, scaled(product(self.edge_radius, self.cosine), -r)
        )
        core = scaled(product(self.cosine, bare), -1.0)
        arsinh = self._first_arsinh()
        rational = 0.5 * r / self.span
        first = (
            0.5 * self.ring_residual
            + rational
            * even.across(sine_squared_times(across_the_range(core)), self.ring[0])
            + 0.25 * self.sine_double * self.variable * arsinh
        )
        odd_first = (
            rational
            * odd.across(sine_squared_times(across_the_range(bare)), self.ring[1])
            + 0.25 * self.sine_double**2 * arsinh
        )
        return (
            4.0 * u * r * odd_first,
            4.0 * u * r * first,
            4.0 * r * first,
            -4.0 * r * odd_first,
            4.0 * u * self.ring_residual,
        )


class _Edge(_RingEdge):
    """One polygon edge's slope-dependent reduction, at an interior amplitude.

    The slope geometry, the plane denominator and the second residual quadrature
    are the full turn's own -- unchanged, because none of them knows where the
    range ends except through the panels the vertex carries.  What is overridden
    is :meth:`terms`, which forms five rows where the ring forms three and adds a
    boundary term to every integration by parts.
    """

    def _second_arsinh(self, vertex: _Vertex, x, y):
        """Return ``arsinh beta2`` at a point of the range given by ``(x, y)``.

        ``B`` vanishes at the near end where the target sits on the edge's extended
        line, and at the far end where that line crosses the axis at the target's
        own height.  The first is reached at a quarter-turn amplitude and is what
        the odd rows' by-parts constant is chosen to put an exact zero on; the
        second is the one corner the choice leaves open, and both return zero here
        rather than an infinity, so a weight that is not zero there is wrong by a
        logarithm rather than by a NaN.
        """
        xp = self.xp
        r = self.radius
        b1 = self.slope
        gamma = vertex.level + b1 * (vertex.offset * x + vertex.radius_sum * y)
        plane = xp.sqrt(
            self.plane_offset**2 * x
            + (self.plane_radius_value + r) ** 2 * y
            + 4.0 * b1 * b1 * r * r * x * y
        )
        held = xp.where(plane > 0.0, plane, 1.0)
        return xp.where(plane > 0.0, xp.arcsinh(gamma / held), 0.0)

    def terms(self, vertex: _Vertex):
        """Return ``4 X_l(alpha)`` for the five rows, at this edge and one limit.

        Four times, so the value is the full turn's own per-limit quantity where
        the amplitude reaches a quarter turn -- which is the cheapest check the
        reduction has and the one that localises everything after it.
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
        even, odd = vertex.channels
        # the plane denominator is the edge's, its split this corner's modulus and
        # both of this corner's families
        plane = vertex.split(self.plane_squared)
        plane_residual = self._second_residual(vertex)
        gamma = range_function([], u + b1 * vertex.offset, u + b1 * vertex.radius_sum)
        # N = u X - b1 G^2, whose value at either end collapses onto u times the
        # plane radius there, because r' - b1 u is r1 exactly
        arctan_numerator = range_function(
            [-4.0 * b1 * r * r], u * self.plane_offset, u * (r1 + r)
        )
        arctan_slope_over_radius = range_function(
            [], -2.0 * u + 4.0 * b1 * r, -2.0 * u - 4.0 * b1 * r
        )
        plane_derivative = product(gamma, self.edge_slope)
        over_ring = product(arctan_numerator, _RING_SLOPE_OVER_RADIUS_SQUARED)
        over_plane = product(arctan_numerator, self.edge_slope_over_radius)
        reached = self._second_arsinh(
            vertex, vertex.sine_squared, vertex.cosine_squared
        )
        at_start = self._second_arsinh(vertex, 0.0 * one, one)

        def even_second_arsinh(build):
            """Return ``integral build arsinh beta2 da``, the weight even in ``a``.

            The oscillatory antiderivative ``sum w_n sin 2na/(2n)`` vanishes at
            ``a = 0`` on its own, so only the upper end is new; ``sin 2n a`` factors
            as ``sin 2a`` times a polynomial in ``cos 2a``, which is both what makes
            the interior part contract against the plain family and what the
            boundary value is evaluated from.
            """
            weight = across_the_range(build()) + [0.0 * one] * 4
            oscillatory = [
                0.5 * weight[1] + weight[3] / 6.0,
                0.5 * weight[2],
                weight[3] / 3.0,
            ]
            core = sine_squared_times(oscillatory)
            return (
                weight[0] * plane_residual
                + vertex.sine_double * vertex.at_amplitude(oscillatory) * reached
                + (2.0 * b1 * r / (a0 * a)) * even.plain(core)
                + (0.5 / (a0 * a))
                * even.across(product(core, plane_derivative), plane[0])
            )

        def odd_second_arsinh(build):
            """Return ``integral sin 2a build arsinh beta2 da``, ``build`` even.

            The antiderivative is ``build`` integrated in the range variable, and
            the one taken is the one vanishing at the AMPLITUDE: the near end is
            where ``B`` collapses for a target on the edge's extended line, and
            subtracting the value there puts an exact zero on the weight the
            near-pole seed's own divergence multiplies.  What is left is the lower
            boundary, where ``B`` is the sum radius and finite.
            """
            rising = rising_integral(across_the_range(build()))
            at_amplitude = vertex.value(rising)
            shifted = (rising[0], rising[1] - at_amplitude, rising[2] - at_amplitude)
            return (
                at_amplitude * at_start
                + (2.0 * b1 * r / (a0 * a)) * odd.plain(shifted)
                + (0.5 / (a0 * a))
                * odd.across(product(shifted, plane_derivative), plane[1])
            )

        # sin phi -> 0 at a = 0 drives beta3 to infinity, so the arctangent lands on
        # +/- pi/2 with this sign -- the same dead-band the rectangle kernel has, and
        # the one end an interior amplitude leaves where the full turn had it.
        at_zero = 0.5 * np.pi * xp.sign(u * (r1 + r))
        at_quarter = xp.where(
            vertex.parameter_complement > 0.0,
            0.5 * np.pi * xp.sign(u * self.plane_offset),
            -xp.arctan(b1) * one,
        )
        # The upper boundary is the arctangent itself.  Its denominator is
        # non-negative over the range, so arctan2 IS the principal value and runs
        # onto either dead band of its own accord; it vanishes only where the
        # amplitude has reached a range end, and there the dead band is taken
        # directly -- including the vertex limit -arctan b1, where the numerator
        # vanishes with it.
        upper_denominator = r * vertex.sine_double * a * vertex.radical
        at_amplitude = xp.where(
            upper_denominator > 0.0,
            xp.arctan2(vertex.value(arctan_numerator), upper_denominator),
            xp.where(vertex.variable > 0.0, at_zero, at_quarter),
        )

        def interior(weight, index):
            """Return the arctangent's rational part, split onto both denominators.

            Exactly as over a full turn: the derivative's quotient is
            ``2 N' - N (G^2)'/G^2 - N (B^2)'/B^2``, a polynomial plus one term over
            each denominator on its own, and each explicit factor of the target
            radius is cancelled here rather than numerically so the term stays
            finite on the axis.
            """
            channel = vertex.channels[index]
            return (
                2.0 * channel.plain(product(weight, arctan_slope_over_radius))
                - r * channel.across(product(weight, over_ring), vertex.ring[index])
                - channel.across(product(weight, over_plane), plane[index])
            )

        def even_arctan(weighting):
            """Return ``integral sin 2a weighting(cos 2a) arctan beta3 da``."""
            primitive = [0.0 * one] + [
                coefficient / (order + 1.0)
                for order, coefficient in enumerate(weighting)
            ]
            boundary = -0.5 * (
                _horner(primitive, vertex.variable) * at_amplitude
                - sum(primitive) * at_zero
            )
            # the weight is the same fixed polynomial for every target, so it is
            # built once out of scalars rather than once per column
            weight = range_function([], 0.0, 0.0)
            for coefficient in reversed(primitive):
                weight = total(
                    product(weight, _VARIABLE),
                    range_function([], coefficient, coefficient),
                )
            return boundary + (0.25 / a) * interior(weight, 0)

        def odd_arctan():
            """Return ``integral -r^2 sin^2 2a cos 2a arctan beta3 da``.

            The one arctangent term an odd row carries, and its weight's
            antiderivative ``-(r^2/6) sin^3 2a`` vanishes at both ends of the range
            -- so there is no dead band here and no lower boundary, only the value
            at the amplitude.
            """
            weight = sine_squared_times([-(r * r / 6.0) * one])
            return -(r * r / 6.0) * vertex.sine_double**3 * at_amplitude - (
                0.5 / a
            ) * interior(weight, 1)

        # the coefficient eq 9's arsinh beta2 carries, shared by the two potential
        # rows -- they differ only in the weight in front of it
        def potential_coefficient():
            return total(
                self.plane_squared,
                scaled(product(vertex.cosine, self.plane_radius), 2.0 * a02 * r),
            )

        # and eq 11b's, shared by the radial and azimuthal field rows
        def bracket_coefficient():
            return scaled(
                total(
                    range_function([], r1 * one, r1 * one),
                    scaled(vertex.cosine, b1 * b1 * r),
                ),
                -b1 / (a02 * a0),
            )

        # the potential's radial row, eq 10b weighted by -sin phi.  A full turn
        # annihilates it: cn K = 0 is the same statement as this fold being even.
        potential_radial = (
            -(2.0 * a / a02) * odd.against_root(gamma)
            + 4.0
            * odd_second_arsinh(
                lambda: scaled(potential_coefficient(), -0.5 / (a02 * a0))
            )
            + 4.0 * odd_arctan()
        )

        # the potential's azimuthal row, eq 10b weighted by cos phi -- the flux
        potential_azimuthal = (
            (2.0 * a / a02) * even.against_root(product(vertex.cosine, gamma))
            + 4.0
            * even_second_arsinh(
                lambda: scaled(
                    product(vertex.cosine, potential_coefficient()), 0.5 / (a02 * a0)
                )
            )
            - 4.0 * r * r * even_arctan([0.0 * one, 0.0 * one, one])
        )

        # the radial field, eq 11b's first component
        radial = (4.0 * a / a02) * even.against_root(vertex.cosine) + 4.0 * (
            even_second_arsinh(lambda: product(vertex.cosine, bracket_coefficient()))
        )

        # the azimuthal field, eq 11b's second component: the same bracket weighted
        # by sin phi instead of cos phi.  Over a full turn the bracket is even about
        # phi = pi and the weight odd, which is why the ring carries no toroidal
        # field; an arc breaks the parity and the row has to be formed.
        azimuthal = (4.0 * a / a02) * odd.root_moments[0] + 4.0 * odd_second_arsinh(
            bracket_coefficient
        )

        # the vertical field, eq 11b's third component.  The D term is LINEAR in the
        # edge slope; eq 11b prints it quadratic, which agrees at slope zero and
        # slope one and nowhere else.
        vertical = (
            4.0
            * even_second_arsinh(
                lambda: scaled(
                    total(
                        range_function([], b1 * b1 * r1 * one, b1 * b1 * r1 * one),
                        scaled(vertex.cosine, -(2.0 * a02 - 1.0) * r),
                    ),
                    1.0 / (a02 * a0),
                )
            )
            - 4.0 * r * even_arctan([one])
            - (4.0 * b1 / a02) * a * even.root_moments[0]
        )
        return potential_radial, potential_azimuthal, radial, azimuthal, vertical


def _folded(limits, rows, quarter):
    """Return the arc's per-limit value, in the full turn's own convention.

    ``rows`` is one ``(row, limit)`` table of ``4 X_l`` and ``quarter`` the same at
    a quarter-turn amplitude, present for the three ODD rows and ``None`` for the
    two even ones.  :func:`nova.biot.arcamplitude.fold` gives
    ``sum_i fold_i = -2 X_l(pi/2)`` at a full sweep, so scaling by ``-1/2`` returns
    exactly what :mod:`nova.biot.polygonanalytic` calls ``W_l`` and the assembly
    below is the ring's own.
    """
    return tuple(
        -0.5
        * sum(
            fold(limit, row[index], quarter[index]) for limit, row in zip(limits, rows)
        )
        for index in range(len(_ROWS))
    )


def packed_arc_greens(
    xp,
    target_r,
    target_z,
    target_phi,
    edge,
    weight,
    norm,
    start,
    end,
    *,
    nodes: int = _NODES,
):
    """Return the finite-arc rows for one fixed-shape list of target/source pairs.

    ``edge`` is ``(E, 4, N)`` and ``weight`` ``(E, N)`` after selecting ``N``
    pairs from :func:`nova.biot.polygon.pad_batch`; all other inputs are length
    ``N``.  The arithmetic is deliberately independent of which edges are live:
    every padded edge and every possible broken-chain residual is evaluated, then
    multiplied by its numeric weight.  That makes the same driver executable by
    NumPy and traceable by JAX without changing the finite-arc reduction.

    The host driver below skips zero-weight edges, shares adjacent corners, and
    omits residual quadratures on a closed edge chain.  Those are valuable scalar
    shortcuts but depend on geometry values.  This packed form exchanges them for
    one static graph that can be mapped over a GPU tile.
    """
    radius = xp.abs(xp.asarray(target_r))
    z = xp.asarray(target_z) + xp.zeros_like(radius)
    phi = xp.asarray(target_phi) + xp.zeros_like(radius)
    limits = arc_limits(phi, start, end, xp=xp)
    rows = [xp.zeros_like(radius) for _ in _ROWS]
    live = xp.asarray(weight != 0.0, dtype=radius.dtype)
    chain = live - xp.roll(live, 1, axis=0)
    sides = edge.shape[0]

    def quarter_rows(values):
        return (None, values[0], values[1], None, values[2])

    def corner_parts(corner_r, corner_z):
        arc = tuple(
            _Vertex(
                radius,
                z,
                (corner_r, corner_z),
                (limit.amplitude, limit.sine, limit.cosine),
                nodes,
                residual=True,
                xp=xp,
            )
            for limit in limits
        )
        ring = _RingVertex(
            radius,
            z,
            corner_r,
            corner_z,
            nodes,
            residual=True,
            xp=xp,
        )
        return arc, ring

    def residual(parts):
        arc, ring = parts
        return _folded(
            limits,
            [vertex.arsinh_terms() for vertex in arc],
            quarter_rows(ring.arsinh_terms()),
        )

    for index in range(sides):
        ra, za, rb, zb = edge[index]
        lower = corner_parts(ra, za)
        upper = corner_parts(rb, zb)
        part = _Edge(radius, z, edge[index], nodes, xp=xp)
        high = _folded(
            limits,
            [part.terms(vertex) for vertex in upper[0]],
            quarter_rows(_RingEdge.terms(part, upper[1])),
        )
        low = _folded(
            limits,
            [part.terms(vertex) for vertex in lower[0]],
            quarter_rows(_RingEdge.terms(part, lower[1])),
        )
        edge_weight = weight[index]
        lower_residual = residual(lower)
        upper_residual = residual(upper)
        lower_chain = live[index] * chain[index]
        upper_chain = live[index] * chain[(index + 1) % sides]
        for row_index in range(len(rows)):
            rows[row_index] = (
                rows[row_index]
                + edge_weight * (low[row_index] - high[row_index])
                + lower_chain * lower_residual[row_index]
                + upper_chain * upper_residual[row_index]
            )
    factor = norm / (4.0 * np.pi)
    return tuple(factor * row for row in rows)


def polygon_arc_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    vertices: np.ndarray,
    start: float,
    end: float,
    *,
    nodes: int = _NODES,
) -> tuple[np.ndarray, ...]:
    """Return ``(A_r, A_phi, B_r, B_phi, B_z)`` per ampere, all in closed form.

    The finite-arc counterpart of
    :func:`nova.biot.polygonanalytic.polygon_analytic_greens`: a polygon
    cross-section, given as the ``(n, 2)`` array of its corners in the ``r-z``
    plane, swept from azimuth ``start`` to ``end`` about the ``z`` axis.  Every row
    is in the TARGET's own cylindrical basis and per ampere of total conductor
    current at uniform azimuthal current density, so the flux the full turn
    returns is ``2 pi r A_phi``.

    ``target_r`` is a cylindrical radius and must be non-negative: a negative one
    is the same point at azimuth plus a half turn, which the azimuth already
    reaches, so no parity is applied to it here.

    Edge ``i`` runs from corner ``i`` to corner ``i + 1``, and each corner is
    evaluated at BOTH of the arc's amplitudes plus, for the three rows whose fold
    is odd, at a quarter turn -- and that last one is the full turn's own
    evaluation rather than a re-derivation, which is what makes a full sweep
    reproduce the ring by the fold's arithmetic alone.
    """
    edges, weights, norm = pack_section(vertices)
    radius = np.abs(np.asarray(target_r, dtype=np.float64))
    shape = radius.shape
    r = radius.ravel()
    z = np.broadcast_to(np.asarray(target_z, dtype=np.float64), shape).ravel()
    phi = np.broadcast_to(np.asarray(target_phi, dtype=np.float64), shape).ravel()
    limits = arc_limits(phi, start, end)
    rows = [np.zeros_like(r) for _ in _ROWS]
    sides = len(edges)
    live = weights != 0.0
    # the signed number of live edges leaving each corner, as over a full turn: the
    # arsinh beta1 term is the corner's own at every amplitude, so it still cancels
    # around an unbroken chain and is still formed only where one is broken
    chain = live.astype(np.int64) - np.roll(live, 1).astype(np.int64)
    last_read: dict[int, int] = {}
    for index in np.flatnonzero(live):
        last_read[int(index)] = int(index)
        last_read[int(index + 1) % sides] = int(index)
    corners: dict[int, tuple] = {}

    def corner_part(index: int, corner_r: float, corner_z: float) -> tuple:
        """Return this corner at both amplitudes, and at a quarter turn."""
        if index not in corners:
            residual = bool(chain[index])
            corners[index] = (
                tuple(
                    _Vertex(
                        r,
                        z,
                        (corner_r, corner_z),
                        (limit.amplitude, limit.sine, limit.cosine),
                        nodes,
                        residual=residual,
                    )
                    for limit in limits
                ),
                _RingVertex(r, z, corner_r, corner_z, nodes, residual=residual),
            )
        return corners[index]

    def quarter_rows(values: tuple) -> tuple:
        """Spread the full turn's three rows onto the five, ``None`` for the even."""
        return (None, values[0], values[1], None, values[2])

    for index in range(sides):
        if not live[index]:
            continue
        ra, za, rb, zb = edges[index]
        lower = corner_part(index, ra, za)
        upper = corner_part((index + 1) % sides, rb, zb)
        part = _Edge(r, z, edges[index], nodes)
        # the antiderivative is of order the squared major radius where the flux is
        # not, so an edge's two limits are differenced against each other first
        high = _folded(
            limits,
            [part.terms(vertex) for vertex in upper[0]],
            quarter_rows(_RingEdge.terms(part, upper[1])),
        )
        low = _folded(
            limits,
            [part.terms(vertex) for vertex in lower[0]],
            quarter_rows(_RingEdge.terms(part, lower[1])),
        )
        for row, one_high, one_low in zip(rows, high, low):
            row -= one_high - one_low
        for index_corner in dict.fromkeys((index, (index + 1) % sides)):
            if last_read[index_corner] != index:
                continue
            if chain[index_corner]:
                arc_corner, ring_corner = corners[index_corner]
                folded = _folded(
                    limits,
                    [vertex.arsinh_terms() for vertex in arc_corner],
                    quarter_rows(ring_corner.arsinh_terms()),
                )
                for row, one in zip(rows, folded):
                    row += chain[index_corner] * one
            del corners[index_corner]
    factor = norm / (4.0 * np.pi)
    return tuple((factor * row).reshape(shape) for row in rows)
