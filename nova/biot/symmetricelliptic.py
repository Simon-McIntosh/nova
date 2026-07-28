"""Carlson's symmetric elliptic integrals, at a fixed trip count.

:mod:`nova.biot.completeelliptic` evaluates all three complete kinds through one
Bartky descent, and :mod:`nova.biot.incompleteelliptic` carries the first two to
an interior amplitude by putting the amplitude into the arithmetic-geometric mean
the descent already runs.  The THIRD kind does not follow that second step, and
the reason is structural rather than a matter of effort: Bartky's rearrangement
is what turns the third kind into one fixed-trip descent over eighteen decades of
pole, and it has no amplitude in it at all.  Its incomplete counterpart is a
different algorithm, not the same one with an argument added.

Carlson's symmetric forms are that different algorithm.  Two of them cover what
the arc needs,

    R_F(x, y, z) = (1/2) integral_0^inf dt/sqrt((t + x)(t + y)(t + z))
    R_J(x, y, z, w) = (3/2) integral_0^inf dt/((t + w) sqrt((t + x)(t + y)(t + z)))

together with the degenerate ``R_C(x, y) = R_F(x, y, y)``, which is elementary.
Both run the SAME duplication on ``(x, y, z)`` -- each trip replaces the three by
their own means and quarters them -- so one pass returns both and the first kind
is free beside the third.  Nothing in the duplication is a convergence test or a
branch on a value, so one implementation serves numpy on the host and a compiled
kernel on a device, and it differentiates.  ``xp`` is the array namespace.

Why a FIXED trip count reaches.  The step ``lambda = sqrt(xy) + sqrt(yz) +
sqrt(zx)`` is a geometric mean where the arguments are far apart, so the trips
before they are of one size HALVE their exponent gap, exactly as the
arithmetic-geometric mean does; after that the deviations the closing series is
written in fall as the fourth power per trip.  The gap is at its widest when the
target sits on the source ring -- ``y`` is then the modulus complement and ``z``
is one -- and the same fourteen trips that carry the two descents carry this:
measured, twelve reach the round-off floor across the whole double range of
complement and ten leave it wrong in the ninth decimal.

Three arrangements in here are not cosmetic, and each was measured on the way in.

* **The degenerate form's two roots are never subtracted from one another.**
  Their difference factorises exactly into the three gaps ``pole - x``,
  ``pole - y``, ``pole - z`` -- see :func:`_elementary` -- and those gaps QUARTER
  under the duplication step, so a caller that can form them without a
  subtraction gets them carried through the whole descent at full precision.
  Subtracting the two roots instead keeps only the digits by which they differ,
  and they COINCIDE wherever the pole argument equals two of the other three,
  which the incomplete third kind reaches to within a rounding error.  Measured
  at a pole of 1e-12 against the extended-precision integral: the subtracted form
  holds 1.3e-15 under numpy and 1.2e-09 once a compiled kernel contracts its
  multiply-adds, where the carried gaps hold the numpy figure on both.
* **The degenerate form takes its two arguments as their own square roots.**
  ``R_C`` is wanted at ``(sigma^2, tau^2)`` where the caller has ``sigma`` and
  ``tau`` themselves, and squaring them first underflows: at a pole a squared
  aspect ratio past a range end and a target a rounding error from the source
  ring, ``tau^2`` is 1e-612 where ``tau`` is 1e-306.  Taking the difference of the
  squares as ``sqrt|tau - sigma| sqrt(tau + sigma)`` keeps every intermediate
  inside the exponent range.
* **The logarithm is taken of the small quantity, not of a ratio near one.**
  ``sigma`` and ``tau`` COINCIDE whenever the pole argument equals one of the
  other three -- which the arc reaches at a vanishing amplitude exactly, and
  approaches wherever an edge end is level with the target -- and written as
  ``log((sigma + root)/tau)`` the ratio is then one to within its own rounding,
  for a relative error of 4.3e-09 in the assembled third kind.  Measured, and the
  accompanying test holds it: ``log1p`` of ``root + |tau - sigma|`` has the same
  value and no cancellation.

``R_J`` carries a SCALE through its accumulation rather than returning a value
for the caller to multiply.  Its own value is unbounded where the pole argument
reaches zero, and at the extreme the arc reaches -- a target within a rounding
error of the source ring, at a section a millionth as thick as it is wide -- it
passes 1e308 while the answer it belongs to is 1e-06.  The weight it is destined
for is small by the same amount, so folding it in term by term is what keeps the
whole evaluation inside the exponent range; it costs one multiply per trip.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TRIPS", "symmetric_kinds"]

# Trips of the duplication.  Each takes a geometric mean of the running triple, so
# the trips before the three are of one size halve their exponent gap and the ones
# after take the closing series' deviations down by a factor of four apiece.  The
# count matches the two descents because the mechanism is theirs; the accompanying
# test asserts it in both directions.
TRIPS = 14


def _elementary(first, second, weight, xp, *, gaps=None):
    """Return ``weight R_C(first^2, second^2)`` from the two roots themselves.

    The one closed form of the family, and the whole of ``R_J``'s per-trip cost:

        R_C(s^2, t^2) = arctan(sqrt(t^2 - s^2)/s)/sqrt(t^2 - s^2),   t > s
                      = artanh(sqrt(s^2 - t^2)/s)/sqrt(s^2 - t^2),   t < s

    one analytic function taken on whichever side it is real, and ``1/s`` where
    they meet.  The arguments arrive as ``s`` and ``t`` because the caller forms
    them that way and their squares do not always exist -- see the module
    docstring -- and the difference of the squares is therefore taken as the
    product of the two halves' roots.

    The hyperbolic branch is written through ``log1p`` of ``root + |t - s|``.
    ``artanh(u)/u`` would be the direct transcription and it saturates as ``u``
    reaches one, which is a pole argument at the range end; ``log((s + root)/t)``
    cures that and fails at the OTHER end, where ``s`` and ``t`` coincide and the
    ratio is one to within its own rounding.  The form here has neither: the
    argument of ``log1p`` is a sum of non-negative terms, small where the two
    roots meet and large where one of them vanishes.

    ``weight`` divides the root BEFORE it multiplies the transcendental, which is
    where the third kind's scale has to enter: the degenerate form itself passes
    the exponent range at a near pole on the source ring -- ``1/tau`` with ``tau``
    at 1e-309 -- and dividing the small weight by the small root is the same
    quotient with both ends inside it.

    ``gaps`` supplies ``(pole - x, pole - y, pole - z)`` where the descent carries
    them, and it is what keeps ``t - s`` from being a cancelling subtraction.  The
    two roots are, in the descent's own quantities,

        s = pole (sqrt x + sqrt y + sqrt z) + sqrt(x y z)
        t = sqrt(pole) (pole + sqrt(x y) + sqrt(y z) + sqrt(z x))

    so ``t - s`` factorises EXACTLY, as
    ``(sqrt(pole) - sqrt x)(sqrt(pole) - sqrt y)(sqrt(pole) - sqrt z)``, and
    multiplying that by its own conjugate -- which is ``t + s``, a sum of
    positives -- gives ``t^2 - s^2 = (pole - x)(pole - y)(pole - z)``.  Hence

        root = sqrt|(pole - x)(pole - y)(pole - z)|,
        t - s = sign(gaps) root^2/(t + s)

    with nothing subtracted anywhere.  The two coincide wherever the pole argument
    equals one of the other three, and the arc reaches that configuration to
    within a rounding error, so the direct subtraction keeps only the digits by
    which they differ and fused arithmetic on a device keeps fewer still.

    ``root`` is formed as the product of the three gaps' OWN square roots rather
    than as the root of their product, for the reason the module docstring gives
    for the arguments arriving as roots: the product itself leaves the exponent
    range where each factor is near the small end of it.  ``t - s`` is likewise
    scaled down by ``t + s`` between its two factors of ``root``, and since
    ``|t - s| <= root`` always, neither multiply can leave the range that ``root``
    is already in.

    The branch is taken on the sign of the gap product and on ``root`` being
    non-zero, so a ``root`` beneath the exponent range takes the confluent value
    the limit has rather than the vanishing one a saturated quotient would give.
    """
    if gaps is None:
        difference = second - first
        root = xp.sqrt(xp.abs(difference)) * xp.sqrt(second + first)
        sign = difference
    else:
        first_root, second_root, third_root = (xp.sqrt(xp.abs(gap)) for gap in gaps)
        root = first_root * second_root * third_root
        sign = xp.sign(gaps[0]) * xp.sign(gaps[1]) * xp.sign(gaps[2])
        difference = sign * root * (root / (second + first))
    live = root != 0.0
    held = xp.where(live, root, 1.0)
    scaled = weight / held
    held_first = xp.where(first > 0.0, first, 1.0)
    held_second = xp.where(second > 0.0, second, 1.0)
    return xp.where(
        live & (sign > 0.0),
        scaled * xp.arctan2(root, first),
        xp.where(
            live & (sign < 0.0),
            scaled * xp.log1p((root - difference) / held_second),
            weight / held_first,
        ),
    )


def symmetric_kinds(x, y, z, pole, scale=1.0, *, gaps=None, xp=np, trips: int = TRIPS):
    """Return ``(R_F(x, y, z), scale R_J(x, y, z, pole))`` off ONE duplication.

    The two share their whole iteration -- ``R_J`` differs only in carrying the
    pole through the same quartering and accumulating one degenerate form per
    trip -- so a caller that wants both, which is what an incomplete third kind
    is, pays for one descent.

    ``scale`` multiplies the third kind INSIDE the accumulation rather than
    after it.  See the module docstring: ``R_J`` alone leaves the exponent range
    at the configurations the arc reaches, and its weight is small by the same
    factor, so the product is the quantity that exists.

    ``gaps`` supplies ``(pole - x, pole - y, pole - z)`` where the caller can form
    them WITHOUT a subtraction, and what it buys is the degenerate form's whole
    accuracy at a pole argument that nearly equals one of the other three -- see
    :func:`_elementary`, where the two roots then differ in their last digits and
    the term hangs on the difference.  Supplying them costs three multiplies a
    trip and is worth six decades at the configurations the arc reaches; omitting
    them falls back to the subtraction, so a caller with no exact gaps keeps a
    usable routine.

    The gaps need no recomputation at any trip, because the step replaces every
    one of the four by ``(term + step)/4`` with the SAME ``step``: the difference
    of two of them is therefore exactly quartered, and quartering carries the full
    precision where re-subtracting the quartered terms carries only what is left
    of it.  A caller's gap may be finer than the assembled ``pole`` can represent
    -- the reflected pole argument can be denormal, and ``pole`` then rounds onto
    ``x`` while the gap still has digits -- and that inconsistency is beneath a
    quantity the descent cannot resolve either way, so it costs nothing.

    All four arguments must be non-negative and ``pole`` positive; the caller
    holds them so, since a value inspected here would be a branch.
    """
    x, y, z, pole = (xp.asarray(term) for term in (x, y, z, pole))
    shape = xp.zeros_like(x + y + z + pole)
    if gaps is not None:
        gaps = tuple(xp.asarray(gap) for gap in gaps)
        shape = xp.zeros_like(shape + gaps[0] + gaps[1] + gaps[2])
        gaps = tuple(gap + shape for gap in gaps)
    x, y, z, pole = (term + shape for term in (x, y, z, pole))
    total = shape
    factor = scale
    for _ in range(trips):
        root_x, root_y, root_z = xp.sqrt(x), xp.sqrt(y), xp.sqrt(z)
        step = root_x * root_y + root_y * root_z + root_z * root_x
        total = total + _elementary(
            pole * (root_x + root_y + root_z) + root_x * root_y * root_z,
            xp.sqrt(pole) * (pole + step),
            factor,
            xp,
            gaps=gaps,
        )
        factor = 0.25 * factor
        x, y, z, pole = (0.25 * (term + step) for term in (x, y, z, pole))
        if gaps is not None:
            gaps = tuple(0.25 * gap for gap in gaps)

    # the first kind's closing series, in the deviations of the three from their
    # own mean -- which sum to zero, so the third is not formed independently
    mean = (x + y + z) / 3.0
    first_x, first_y = (mean - x) / mean, (mean - y) / mean
    first_z = -first_x - first_y
    pair = first_x * first_y - first_z * first_z
    triple = first_x * first_y * first_z
    first = (
        1.0
        - pair / 10.0
        + triple / 14.0
        + pair * pair / 24.0
        - 3.0 * pair * triple / 44.0
    ) / xp.sqrt(mean)

    # the third kind's, in the deviations from the weighted mean the pole enters
    # twice, and its own accumulated sum
    mean = (x + y + z + 2.0 * pole) / 5.0
    third_x, third_y = (mean - x) / mean, (mean - y) / mean
    third_z, third_pole = (mean - z) / mean, (mean - pole) / mean
    pair = third_x * (third_y + third_z) + third_y * third_z
    triple = third_x * third_y * third_z
    squared = third_pole * third_pole
    reduced = pair - 3.0 * squared
    mixed = triple + 2.0 * third_pole * (pair - squared)
    series = (
        1.0
        + reduced * (-3.0 / 14.0 + reduced * 9.0 / 88.0 - mixed * 9.0 / 52.0)
        + triple * (1.0 / 6.0 + third_pole * (-3.0 / 11.0 + third_pole * 3.0 / 26.0))
        + third_pole * pair * (1.0 / 3.0 - third_pole * 3.0 / 22.0)
        - third_pole * squared / 3.0
    )
    return first, 3.0 * total + factor * series / (mean * xp.sqrt(mean))
