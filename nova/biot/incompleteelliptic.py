"""Incomplete elliptic integrals at an interior amplitude, from the complement.

The full-turn polygon-section reduction (:mod:`nova.biot.polygonanalytic`)
integrates the transformed angle over the whole quarter range ``a in [0, pi/2]``,
where every elliptic integral is COMPLETE and :mod:`nova.biot.completeelliptic`
is the only special function it needs.  A finite ARC stops the range short: the
paper's amplitude ``alpha_i = (pi + phi - phi_i)/2`` is set by where the target
sits relative to the arc's own ends, so the upper limit is interior and every
integral regains its argument.  This module is the complete module's counterpart
at that interior limit.

Two things carry over unchanged, because the geometry has not changed -- only the
range.

* **The argument is the modulus COMPLEMENT.**  ``k'^2`` is the target's squared
  distance to an edge's end over the squared ring span, and a float parameter
  cannot carry it: ``1 - k^2`` is known only to ``eps`` however ``k^2`` was
  formed, and the first kind, which grows like ``-log k'``, is then wrong by
  ``eps/k'^2``.  Every entry point here takes the complement.
* **The trip count is FIXED.**  Nothing below iterates to a convergence test or
  branches on a value, so one implementation serves numpy on the host and a
  compiled kernel on a device, and it differentiates.  ``xp`` is the array
  namespace: numpy by default, ``jax.numpy`` inside a trace.

What changes is the descent.  Bulirsch's ``cel`` carries the whole quarter range
in one Bartky iteration and has no amplitude to carry; at an interior limit the
amplitude must descend with the modulus, which is the arithmetic-geometric mean
in its original Gauss form rather than Bartky's rearrangement of it.  Each trip
replaces the running pair by its two means and DOUBLES the amplitude,

    a <- (a + b)/2,   b <- sqrt(a b),   phi <- phi + arctan2((b/a) sin phi, cos phi)

and the whole difficulty of the step is that last line.  Written the way the
identity usually is -- ``arctan((b/a) tan phi)`` -- it collapses at the one
configuration this module has to reproduce exactly: a quarter-turn amplitude has
a tangent of 1.6e16 rather than infinity, so a small ratio returns a step of
nothing where the true step is a quarter turn, the amplitude never doubles, and
the first kind saturates near 38 whatever the modulus does.  Taking the two parts
separately removes the saturation and takes the continuous branch with it; the
whole turns the arctangent cannot see are restored arithmetically, by requiring
the step to land near ``2 phi``.  :func:`_stepped_amplitude` carries both.

The two means converge quadratically once they are of the same size, and the
trips before that halve the exponent gap, so the same fourteen trips that carry
``cel`` over three hundred decades of complement carry this one.  The amplitude
grows by a factor of two per trip and so reaches some ten thousand radians; that
costs nothing, because the answer is ``phi_N/(2^N a_N)`` and what survives into it
is the RELATIVE accuracy of ``phi_N``.

One corner is not covered and is measured rather than hidden.  Where the
amplitude is within about 1e-7 of a quarter turn AND the complement is below
about 1e-25, the first step's own arctangent underflows before the descent has
begun and the routine returns the confluence's elementary value instead of the
one a hair away from it, for a relative error that reaches 1e-4 at the extreme.
Geometrically that is a target on a section CORNER and within a rounding error of
the arc's END PLANE at once -- within a femtometre of the arc's end edge -- and
the accompanying test asserts the bound in both directions.

The confluence ``k'^2 = 0`` is a target ON the source ring.  Unlike the complete
case it is not generally a divergence here: the descent stalls -- a geometric mean
of zero stays zero -- and the integrals collapse onto the elementary
``F = log((1 + sin phi)/cos phi)`` and ``E = sin phi``, both finite for any
amplitude short of a quarter turn.  Only the corner where the amplitude reaches a
quarter turn AS WELL is the full turn's own divergence, and there the complete
module's finite-part convention is returned so the two agree in the limit the arc
closes.

The THIRD kind, :func:`incomplete_pole`, does not come off that descent at all,
and that is worth stating because it looks as though it should.  What makes the
complete third kind ONE fixed-trip evaluation over eighteen decades of pole is
Bartky's REARRANGEMENT of the mean, and there is no amplitude anywhere in it; the
incomplete counterpart is a different algorithm rather than the same one with an
argument added.  It is built on Carlson's symmetric forms
(:mod:`nova.biot.symmetricelliptic`) instead, and on the reflection that makes
them hold at a near pole -- see :func:`incomplete_pole`.
"""

from __future__ import annotations

import numpy as np

from nova.biot.symmetricelliptic import symmetric_kinds

__all__ = ["TRIPS", "incomplete_kind", "incomplete_pole"]

# Trips of the arithmetic-geometric mean.  Each takes the geometric mean of the
# running pair, so the correct digits DOUBLE per trip once the two are of the same
# size, and the trips before that halve the exponent gap -- which is what makes one
# constant cover the whole double range of complement rather than a decade or two.
# The count matches :mod:`nova.biot.completeelliptic` because the mean is the same
# one; the accompanying test asserts it in both directions.
TRIPS = 14


def _stepped_amplitude(amplitude, sine, cosine, ratio, xp):
    """Return the amplitude of the next arithmetic-geometric mean step.

    Gauss's theorem adds to the amplitude the angle whose tangent is
    ``ratio tan phi``, and the whole difficulty is taking it on the right branch.
    Two things are needed and neither is a branch in the control-flow sense.

    First, the angle comes from the amplitude's SINE AND COSINE separately rather
    than from their ratio.  ``arctan(ratio tan phi)`` is what the identity is
    usually written as, and it is wrong wherever ``tan`` saturates: a quarter-turn
    amplitude has a tangent of 1.6e16 rather than infinity, so a small ``ratio``
    returns a step of nothing where the true step is a quarter turn.  That is not
    a rounding error but a collapse -- the amplitude then never doubles and the
    first kind saturates near 38 however far the modulus goes -- and it happens at
    exactly the configuration the full turn IS.  ``arctan2`` of the two parts is
    the same angle with the saturation removed, and it takes the continuous branch
    over a whole half-turn for free.

    Second, the amplitude runs to some ten thousand radians over the descent while
    ``arctan2`` returns a value in one turn, so the whole turns are restored by
    requiring the step to land near ``2 phi``: ``round((phi - principal)/2 pi)``.
    Taking the count from the principal value rather than from the amplitude alone
    is what keeps it right in BOTH regimes -- a ratio near one, where the step is
    nearly the amplitude itself, and a ratio near zero, where it is nearly nothing.
    """
    principal = xp.arctan2(ratio * sine, cosine)
    return (
        amplitude
        + principal
        + 2.0 * np.pi * xp.round((amplitude - principal) / (2.0 * np.pi))
    )


def incomplete_kind(
    amplitude,
    complement,
    *,
    sine=None,
    cosine=None,
    parameter=None,
    xp=np,
    trips: int = TRIPS,
):
    """Return ``(F, E)`` at ``amplitude``, from the modulus complement ``k'^2``.

        F(phi, k) = integral_0^phi da/sqrt(1 - k^2 sin^2 a)
        E(phi, k) = integral_0^phi da sqrt(1 - k^2 sin^2 a)

    Both come off ONE descent.  The first kind is the doubled amplitude over the
    mean, ``phi_N/(2^N a_N)``; the second is the first scaled by the descent's own
    squared differences together with a sum over the sines of the amplitudes it
    passed through,

        E = F [1 - (1/2) sum_n 2^n c_n^2] + sum_(n>=1) c_n sin phi_n

    with ``c_0 = k`` and ``c_n`` half the difference of the pair at step ``n``.
    Every ``c_n`` past the first is a cancelling difference of two means, but it is
    also quadratically small, so what it costs the sum is bounded by round-off on
    the LARGEST term rather than accumulating -- and every term scales with the
    amplitude, so a small amplitude keeps its relative accuracy rather than meeting
    an absolute floor.

    ``parameter`` supplies ``k^2`` where the caller has it exactly.  It enters only
    ``c_0``, whose value at a DISTANT target -- the small-parameter limit -- would
    otherwise be formed as ``1 - k'^2`` against a complement of nearly one.  The
    polygon reduction knows both from the geometry, so neither is a subtraction.

    ``sine`` and ``cosine`` supply the amplitude's own pair where the caller has
    them exactly, and near a quarter turn that is not a refinement but the
    difference between an answer and a collapse.  The arc's amplitude is
    ``(pi + psi)/2`` for an azimuthal separation ``psi`` from one of its ends, so
    its cosine is ``-sin(psi/2)`` -- exact, and exactly zero where the separation
    vanishes.  Taken instead as ``cos`` of the assembled amplitude it is 6e-17
    there rather than zero, which is the same relative error as having no digits
    at all, and :func:`_stepped_amplitude` explains what that costs.

    At ``k'^2 = 0`` the descent stalls on a geometric mean of zero and the
    elementary values are returned instead: ``F = log((1 + sin phi)/cos phi)`` and
    ``E = sin phi``.  Where the amplitude reaches a quarter turn as well, that
    first kind is the full turn's logarithmic divergence and
    :func:`nova.biot.completeelliptic.complete_kind`'s finite part -- zero -- is
    returned in its place, so the arc agrees with the ring in the limit it closes.
    The second kind needs no such convention: ``sin(pi/2) = 1`` is already
    ``E(1)``.
    """
    amplitude = xp.asarray(amplitude)
    complement = xp.asarray(complement) + xp.zeros_like(amplitude)
    if parameter is None:
        parameter = 1.0 - complement
    parameter = xp.asarray(parameter) + xp.zeros_like(amplitude)
    sine = xp.sin(amplitude) if sine is None else xp.asarray(sine)
    cosine = xp.cos(amplitude) if cosine is None else xp.asarray(cosine)

    # the complement is held at one where it is not positive, so the descent runs
    # on ordinary numbers and its derivative stays finite; the confluence's value
    # comes from the elementary branch below
    reachable = complement > 0.0
    held = xp.where(reachable, complement, 1.0)

    mean = xp.ones_like(held)
    geometric = xp.sqrt(held)
    phase = amplitude
    # the n = 0 term of the second kind's weight is k^2/2, and it is the one place
    # the parameter enters: every later difference is between means already of the
    # same size, so none of them carries a cancellation that matters
    weight = 0.5 * xp.where(reachable, parameter, 1.0)
    sines = xp.zeros_like(held)
    scale = 1.0
    # only the FIRST step sees the caller's own pair; every later amplitude is one
    # this routine formed itself, and by then the running ratio is of order one, so
    # the tangent it goes through has nothing left to saturate
    step_sine, step_cosine = sine, cosine
    for _ in range(trips):
        phase = _stepped_amplitude(phase, step_sine, step_cosine, geometric / mean, xp)
        step_sine, step_cosine = xp.sin(phase), xp.cos(phase)
        difference = 0.5 * (mean - geometric)
        mean, geometric = 0.5 * (mean + geometric), xp.sqrt(mean * geometric)
        scale = 2.0 * scale
        weight = weight + 0.5 * scale * difference * difference
        sines = sines + difference * step_sine

    first = phase / (scale * mean)
    second = first * (1.0 - weight) + sines

    # a quarter-turn amplitude at the confluence IS the full turn's divergence, and
    # the finite part the complete module assigns it is zero; anywhere short of it
    # the elementary first kind is finite and is the whole value
    live = cosine > 0.0
    elementary = xp.where(live, xp.log((1.0 + sine) / xp.where(live, cosine, 1.0)), 0.0)
    return (
        xp.where(reachable, first, elementary),
        xp.where(reachable, second, sine),
    )


def incomplete_pole(pole, complement, sine, cosine, *, xp=np, trips: int = TRIPS):
    """Return ``integral_0^phi da/((cos^2 a + p sin^2 a) sqrt(1 - k^2 sin^2 a))``.

    The interior-amplitude counterpart of
    :func:`nova.biot.completeelliptic.complete_pole`, in the same arrangement and
    with the same two arguments: ``pole`` is the denominator's value at the FAR
    end of the range, which is ``1 - n`` for the usual characteristic, and
    ``complement`` is ``k'^2``.  Both are quantities a caller forms from its own
    geometry, neither by subtraction; see the complete routine for why that is the
    whole accuracy question near a range end.

    The amplitude is taken as its ``sine`` and ``cosine`` and NOT as an angle,
    because the angle never enters -- ``cos^2 phi``, ``sin phi`` and the two
    denominators at the amplitude are all the evaluation asks for.  The arc's own
    amplitude is ``(pi + psi)/2`` for an azimuthal separation ``psi`` from one of
    its ends, so the pair is ``(cos(psi/2), -sin(psi/2))``, exact and exactly
    ``(1, 0)`` where the separation vanishes; the cosine of the assembled angle is
    6e-17 there instead, which at a quarter turn is the difference between the
    ring's value and no value at all.

    **The reflection is the substance of this routine.**  Written out directly the
    integral is ``F + (n/3) sin^3 phi R_J`` with ``n = 1 - p``, and for a root just
    past the NEAR end of the range -- which is where the range STARTS, so it is
    reached at every corner rather than occasionally -- ``n`` is hugely negative
    and those two terms are of opposite sign and nearly equal.  Their sum falls as
    ``1/sqrt(p)`` while each stays of order ``F``, so the arrangement throws away
    half the decades of the pole: measured against the extended-precision integral,
    6.3e-11 at ``p = 1e12``, which is a section a millionth as thick as it is wide.
    The weight this seed carries is the numerator's value at that same end and is
    small by the same amount, so the loss reaches the answer undiminished.

    Reflecting the pole onto its partner ``k'^2/p`` removes it.  The two are
    related exactly, by an identity whose elementary part carries the whole growth,

        Pi(phi, p) = [k'^2 (p - 1)/(p (p - k'^2))] Pi(phi, k'^2/p)
                   + [k^2/(p - k'^2)] F(phi, k)
                   + [mu/(p - k'^2)] arctan(mu sin phi cos phi/Delta),
        mu^2 = (p - 1)(p - k'^2)/p

    and for ``p > 1`` every one of those three terms is POSITIVE, so there is no
    cancellation left anywhere: the arctangent is the ``arctan(sqrt(p) tan phi)/
    sqrt(p)`` the range would have if the modulus were one, the first kind's term
    carries the factor ``k^2/(p - 1)`` that is small exactly where the direct route
    is worst, and the partner pole is BELOW one -- the orientation in which the
    symmetric forms are a sum of positives.  Measured over the same sweep the
    reflected route holds 1.8e-15 at every pole from 1e-12 to 1e12 and every
    complement down to 1e-300.

    So one evaluation covers both orientations: the pole handed to
    :func:`nova.biot.symmetricelliptic.symmetric_kinds` is the smaller of ``p`` and
    ``k'^2/p``, and the three coefficients above collapse to ``(1, 0)`` where no
    reflection is wanted.  The arctangent's weight vanishes with ``mu`` there, so
    the same expression serves both and nothing selects on the result.

    The symmetric form's own accuracy needs one thing from here beyond the
    arguments.  Its degenerate term turns on the three gaps between the pole
    argument and the other three, and with ``x = cos^2 phi``,
    ``y = cos^2 phi + k'^2 sin^2 phi``, ``z = 1`` and the pole argument
    ``cos^2 phi + partner sin^2 phi`` every one of them is a PRODUCT,

        pole - x = partner sin^2 phi
        pole - y = (partner - k'^2) sin^2 phi
        pole - z = (partner - 1) sin^2 phi

    the last because ``cos^2 phi - 1`` is ``-sin^2 phi``.  A small partner pole
    puts the pole argument within a rounding error of both ``x`` and ``y``, so
    taking the gaps by subtraction there leaves a handful of digits; taken as
    products they are exact, and better than exact arithmetic on the assembled
    pole would be -- a denormal partner rounds the pole argument onto ``x``
    outright, where the product still carries digits.

    A pole of zero puts the root ON the far end of the range.  At an interior
    amplitude that is short of the range and the integral is finite, but zero is
    returned, as the complete routine returns it, so the arc and the ring agree in
    the limit the arc closes; the callers that reach it carry a weight on such a
    pole that is itself exactly zero.
    """
    pole = xp.asarray(pole)
    complement = xp.asarray(complement) + xp.zeros_like(pole)
    sine = xp.asarray(sine) + xp.zeros_like(pole)
    cosine = xp.asarray(cosine) + xp.zeros_like(pole)

    live = pole > 0.0
    held = xp.where(live, pole, 1.0)
    # the reflection, taken on the ARGUMENTS so that one evaluation serves both
    # orientations; below one the partner would be the worse of the two
    reflected = pole > 1.0
    partner = xp.where(reflected, complement / held, held)

    squared_cosine, squared_sine = cosine * cosine, sine * sine
    radical = squared_cosine + complement * squared_sine
    weight = squared_cosine + partner * squared_sine

    gap = xp.where(reflected, pole - complement, 1.0)
    growth = xp.sqrt(
        xp.where(reflected, (pole - 1.0) * (pole - complement) / held, 0.0)
    )
    partner_weight = xp.where(reflected, complement * (pole - 1.0) / (held * gap), 1.0)
    first_weight = partner_weight + xp.where(reflected, (1.0 - complement) / gap, 0.0)

    # the three gaps between the pole argument and the other three, as PRODUCTS of
    # quantities already held: the symmetric form's degenerate term hangs on their
    # difference, and the pole argument is within a rounding error of both
    # squared_cosine and radical wherever the partner pole is small
    gaps = (
        partner * squared_sine,
        (partner - complement) * squared_sine,
        (partner - 1.0) * squared_sine,
    )

    # the third kind's own weight rides INTO the accumulation: at a near pole and a
    # target on the source ring the symmetric form passes 1e308 while the answer it
    # belongs to is 1e-06, and this factor is small by the same amount
    first, third = symmetric_kinds(
        squared_cosine,
        radical,
        1.0,
        weight,
        partner_weight * (1.0 - partner) * sine * squared_sine / 3.0,
        gaps=gaps,
        xp=xp,
        trips=trips,
    )
    value = (
        first_weight * sine * first
        + third
        + growth * xp.arctan2(growth * sine * cosine, xp.sqrt(radical)) / gap
    )
    return xp.where(live & (weight > 0.0), value, 0.0)
