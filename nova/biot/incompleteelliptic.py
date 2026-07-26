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
"""

from __future__ import annotations

import numpy as np

__all__ = ["TRIPS", "incomplete_kind"]

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
