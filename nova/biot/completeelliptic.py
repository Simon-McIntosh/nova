"""Complete elliptic integrals of all three kinds, from the modulus COMPLEMENT.

One integral covers all three kinds -- Bulirsch's ``cel``, whose natural arguments
are the complementary modulus and the complement of the characteristic:

    cel(k', p, a, b) = integral_0^(pi/2) da (a cos^2 a + b sin^2 a)
                       / ((cos^2 a + p sin^2 a) sqrt(cos^2 a + k'^2 sin^2 a))

    K = cel(k', 1, 1, 1)    E = cel(k', 1, 1, k'^2)    Pi(n | m) = cel(k', 1 - n, 1, 1)

Why the complement and not the parameter.  A float parameter cannot carry its own
complement: ``1 - k'^2`` is known only to ``eps`` however ``k'^2`` was formed, and
``K``, which grows like ``-log k'``, is then wrong by ``eps/k'^2`` -- measured
against the extended-precision mean, 2.4e-10 at ``k'^2 = 1e-8``, 2.6e-03 at 1e-16,
and infinite below, where the parameter rounds to one outright.  For a ring the
complement is the target's squared distance to the source point over the squared
ring span, so a micron from a metre-scale ring is already 1e-12 and the loss is a
working configuration rather than a corner case.  Every argument here is therefore
the complement, formed by the caller from the geometry and never by subtraction.
The third kind's pole argument is a complement for the same reason: ``Pi`` grows
like ``(1 - n)^(-1/2)``, so forming ``1 - n`` here would cap the accuracy at
``eps/(1 - n)``.

Why THIS routine and not Carlson's.  The host already has Carlson's symmetric
forms through scipy, and they take the complement, so for the first and second
kinds the two routes agree to a couple of ulp and the choice is a cost one.  The
third kind is where they part.  ``R_J`` is the expensive one -- five times ``R_F``
and twenty times a Cephes ``K`` per element -- and the polygon reduction needs
several complete ``Pi`` per corner, so the third kind decides the cost.  It also
decides the accuracy: written as ``R_F + (n/3) R_J`` the two terms are of opposite
sign and nearly equal once the pole is far past the range, which is exactly the
common configuration (a root a squared aspect ratio past the end), and the
arrangement that avoids it needs the pole reflected onto the other end of the
range as a separate case.  ``cel`` has neither problem: ``p`` enters as itself, the
descent is a sum of positives, and one routine spans eighteen decades of pole.

And why it can be traced.  The descent is arithmetic and square roots with a FIXED
trip count -- no convergence test, no data-dependent branch, no iteration whose
length depends on a value -- so one implementation serves numpy on the host and a
compiled kernel on a device, and it differentiates.  ``xp`` is the array namespace:
numpy by default, ``jax.numpy`` inside a trace.  Nothing below inspects a value.

The confluence ``k'^2 = 0`` -- a target ON the source ring -- is where the first
kind diverges logarithmically, and it is returned as the FINITE PART: the
divergence enters ``cel`` with coefficient ``b/p``, and subtracting it leaves the
elementary

    cel(0, p, a, b)_finite = (a - b/p) integral_0^(pi/2) cos a da/(cos^2 a + p sin^2 a)

whose integral is ``arctan(sqrt(p - 1))/sqrt(p - 1)`` above one and
``artanh(sqrt(1 - p))/sqrt(1 - p)`` below it, and one at ``p = 1``.  That single
expression reproduces every convention the callers need without a special case:
zero for ``K``, one for ``E``, and for the pole form the arctangent the reduction's
own limit leaves.  The convention is sound because the reduction that consumes
these puts a total weight of ZERO on the divergence -- the flux and field of a
section are bounded at its own corner -- so the answer is linear in whatever is
assigned here with a slope of zero, and the finite part is the assignment that
evaluates it directly rather than as a large cancellation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TRIPS", "complete_kind", "complete_pole"]

# Trips of the descent.  Each one takes the geometric mean of the running modulus
# pair, so the number of correct digits DOUBLES per trip once the two are of the
# same size, and the trips before that halve the exponent gap -- which is why one
# constant covers three hundred decades of complement rather than a decade or two.
# Measured (``tests/test_biotcompleteelliptic.py``): twelve trips bring the whole
# double range, denormals included, onto the algorithm's own round-off floor, and
# ten leave the smallest complements wrong in the sixth decimal.  Two spare.
TRIPS = 14

_HALF_PI = 0.5 * np.pi


def _descent(complement, xp, trips: int = TRIPS):
    """Return the descent's radicals in order, and its final arithmetic sum.

    Bulirsch's iteration is the arithmetic-geometric mean of ``1`` and ``k'``
    carrying a factor of two per trip: with ``arithmetic`` the running sum and
    ``radical`` the quantity under the next square root,

        arithmetic <- arithmetic + modulus,   modulus <- 2 sqrt(radical),
        radical <- modulus arithmetic

    from ``arithmetic = 1`` and ``radical = modulus = k'``.  Nothing in it depends
    on the pole, so a caller with several poles at one modulus -- which is what a
    polygon corner is -- pays for the descent once and accumulates against these.

    The complement is held at one where it is not positive: the confluence's value
    comes from :func:`_finite_part` instead, and holding the ARGUMENT rather than
    only masking the result is what keeps a derivative finite there as well.
    """
    complement = xp.asarray(complement)
    held = xp.where(complement > 0.0, complement, 1.0)
    modulus = xp.sqrt(held)
    radical = modulus
    arithmetic = xp.ones_like(modulus)
    radicals = []
    for _ in range(trips):
        radicals.append(radical)
        arithmetic = arithmetic + modulus
        modulus = 2.0 * xp.sqrt(radical)
        radical = modulus * arithmetic
    return radicals, arithmetic


def _accumulate(radicals, arithmetic, pole, cosine_weight, sine_weight, xp):
    """Return ``cel`` from a descent, for one pole and one pair of weights.

    The two numerator weights ride the descent as a pair -- each trip folds the
    ``sin^2`` weight into the ``cos^2`` one and doubles what is left -- alongside
    the pole's own root, which accumulates the same radicals the modulus does.
    Only this last part sees the pole, so the descent above is shared.
    """
    pole_root = xp.sqrt(pole)
    cosine_part = cosine_weight + xp.zeros_like(arithmetic)
    sine_part = sine_weight / pole_root + xp.zeros_like(arithmetic)
    for radical in radicals:
        previous = cosine_part
        cosine_part = cosine_part + sine_part / pole_root
        gain = radical / pole_root
        sine_part = 2.0 * (sine_part + previous * gain)
        pole_root = pole_root + gain
    return (
        _HALF_PI
        * (sine_part + cosine_part * arithmetic)
        / (arithmetic * (arithmetic + pole_root))
    )


def _finite_part(pole, cosine_weight, sine_weight, xp):
    """Return ``cel`` less its divergence, for a modulus complement of zero.

    ``(a - b/p)`` times the elementary integral of ``cos a/(cos^2 a + p sin^2 a)``;
    see the module docstring for where the two come from.  The integral is an
    arctangent past one and an area hyperbolic tangent below it, and the latter is
    taken as ``log((1 + t)/sqrt(p))/t`` rather than ``artanh(t)/t``: ``t`` reaches
    one to round-off for a small pole, where ``artanh`` overflows, while ``1 - t``
    is ``p/(1 + t)`` exactly and the logarithm of the ratio is not.
    """
    rising = pole - 1.0
    root = xp.sqrt(xp.abs(rising))
    held_root = xp.where(root > 0.0, root, 1.0)
    held_pole = xp.where(rising < 0.0, pole, 1.0)
    over = xp.where(
        rising > 0.0,
        xp.arctan(root) / held_root,
        xp.log((1.0 + root) / xp.sqrt(held_pole)) / held_root,
    )
    return (cosine_weight - sine_weight / pole) * xp.where(rising == 0.0, 1.0, over)


def complete_kind(complement, *, xp=np, trips: int = TRIPS):
    """Return ``(K, E)`` from the modulus complement ``k'^2``.

    Both kinds come off ONE descent -- they share the pole ``p = 1`` and differ only
    in the weight on ``sin^2 a``, which is one for ``K`` and the complement itself
    for ``E`` -- so the second kind costs a handful of multiplies rather than a
    second iteration.

    At ``k'^2 = 0`` the finite parts are exactly ``(0, 1)``, and they are the general
    expression rather than a stipulation: ``a - b/p`` is ``1 - 1 = 0`` for ``K`` and
    ``1 - 0 = 1`` for ``E``, both against an integral of one at ``p = 1``.  ``E(1) = 1``
    is the true value; the zero for ``K`` is the finite-part convention.
    """
    complement = xp.asarray(complement)
    radicals, arithmetic = _descent(complement, xp, trips)
    reachable = complement > 0.0
    held = xp.where(reachable, complement, 1.0)
    return (
        xp.where(reachable, _accumulate(radicals, arithmetic, 1.0, 1.0, 1.0, xp), 0.0),
        xp.where(reachable, _accumulate(radicals, arithmetic, 1.0, 1.0, held, xp), 1.0),
    )


def complete_pole(pole, complement, *, xp=np, trips: int = TRIPS):
    """Return ``integral_0^(pi/2) da/((cos^2 a + p sin^2 a) sqrt(1 - k^2 sin^2 a))``.

    The complete integral of the third kind in the arrangement a caller that knows
    its geometry can supply exactly: ``pole`` is the denominator's value at the far
    end of the range, which is ``1 - n`` for the usual characteristic, and
    ``complement`` is ``k'^2``.  A pole below one puts the root past the NEAR end of
    the range and above one past the far end; the polygon reduction reaches both,
    over eighteen decades either side, and this is one expression for all of it.

    A pole of zero puts the root ON the range end, where the integral diverges and
    no finite part exists -- the divergence is a square root there rather than a
    logarithm, so it does not separate.  Zero is returned, which is the convention
    of the callers that can reach it: their numerator's weight on such a pole is
    itself exactly zero, the two vanishing together with the same geometric
    quantity.
    """
    pole = xp.asarray(pole)
    complement = xp.asarray(complement)
    live = pole > 0.0
    held_pole = xp.where(live, pole, 1.0)
    radicals, arithmetic = _descent(complement, xp, trips)
    value = xp.where(
        complement > 0.0,
        _accumulate(radicals, arithmetic, held_pole, 1.0, 1.0, xp),
        _finite_part(held_pole, 1.0, 1.0, xp),
    )
    return xp.where(live, value, 0.0)
