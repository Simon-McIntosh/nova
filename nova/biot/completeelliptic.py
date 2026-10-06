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

import jax
import numpy as np

from nova.biot.rangefunction import _array_program

from nova.biot.pairedfloat import add as paired_add
from nova.biot.pairedfloat import divide as paired_divide
from nova.biot.pairedfloat import multiply as paired_multiply
from nova.biot.pairedfloat import scale as paired_scale
from nova.biot.pairedfloat import square_root as paired_square_root
from nova.biot.pairedfloat import value as paired_value
from nova.biot.pairedfloat import where as paired_where
from nova.biot.pairedfloat import wrap as paired_wrap

__all__ = [
    "TRIPS",
    "complete_kind",
    "complete_kind_paired",
    "complete_pole",
    "complete_pole_paired",
]

# Trips of the descent.  Each one takes the geometric mean of the running modulus
# pair, so the number of correct digits DOUBLES per trip once the two are of the
# same size, and the trips before that halve the exponent gap -- which is why one
# constant covers three hundred decades of complement rather than a decade or two.
# Measured (``tests/test_biotcompleteelliptic.py``): twelve trips bring the whole
# double range, denormals included, onto the algorithm's own round-off floor, and
# ten leave the smallest complements wrong in the sixth decimal.  Two spare.
TRIPS = 14

_HALF_PI = 0.5 * np.pi


@_array_program
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


@_array_program
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


@_array_program
def _descent_paired(complement, xp, trips: int = TRIPS):
    reachable = paired_value(complement) > 0.0
    held = paired_where(reachable, complement, paired_wrap(1.0), xp)
    modulus = paired_square_root(held, xp)
    radical = modulus
    arithmetic = paired_wrap(xp.ones_like(modulus[0]))
    radicals = []
    for _ in range(trips):
        radicals.append(radical)
        arithmetic = paired_add(arithmetic, modulus)
        modulus = paired_scale(paired_square_root(radical, xp), 2.0)
        radical = paired_multiply(modulus, arithmetic)
    return radicals, arithmetic


@_array_program
def _accumulate_paired(radicals, arithmetic, pole, cosine_weight, sine_weight, xp):
    pole_root = paired_square_root(pole, xp)
    cosine_part = paired_wrap(cosine_weight + xp.zeros_like(arithmetic[0]))
    sine_pair = (
        sine_weight
        if isinstance(sine_weight, tuple)
        else paired_wrap(sine_weight + xp.zeros_like(arithmetic[0]))
    )
    sine_part = paired_divide(sine_pair, pole_root)
    for radical in radicals:
        previous = cosine_part
        cosine_part = paired_add(cosine_part, paired_divide(sine_part, pole_root))
        gain = paired_divide(radical, pole_root)
        sine_part = paired_scale(
            paired_add(sine_part, paired_multiply(previous, gain)), 2.0
        )
        pole_root = paired_add(pole_root, gain)
    numerator = paired_scale(
        paired_add(sine_part, paired_multiply(cosine_part, arithmetic)),
        _HALF_PI,
    )
    denominator = paired_multiply(arithmetic, paired_add(arithmetic, pole_root))
    return paired_divide(numerator, denominator)


@_array_program
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
    separated = rising != 0.0
    # The elementary factor tends to one where the pole equals one.  Hold the
    # root argument before evaluation at that confluence: taking sqrt(0) and only
    # masking its quotient afterward leaves an unbounded tangent in both JAX
    # differentiation modes.
    root = xp.sqrt(xp.where(separated, xp.abs(rising), 1.0))
    held_pole = xp.where(rising < 0.0, pole, 1.0)
    over = xp.where(
        rising > 0.0,
        xp.arctan(root) / root,
        (xp.log1p(root) - 0.5 * xp.log(held_pole)) / root,
    )
    # Close to one, both elementary branches are the same analytic series in
    # ``rising``.  Besides avoiding the logarithm's near-unit ratio, this keeps
    # the next representable poles on either side of one correctly rounded.
    close = xp.abs(rising) < 1e-4
    series_rising = xp.where(close, rising, 0.0)
    series = xp.ones_like(rising)
    power = xp.ones_like(rising)
    for order in range(1, 8):
        power = -power * series_rising
        series = series + power / (2 * order + 1)
    elementary = xp.where(close, series, over)

    # ``a - b/p`` loses the whole small difference when ``a == b`` and the pole
    # is immediately below one.  This equivalent arrangement exposes ``p - 1``
    # directly; Sterbenz subtraction makes that difference exact around one.
    coefficient = (cosine_weight - sine_weight) + sine_weight * rising / pole
    return coefficient * elementary


@_array_program
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


@_array_program
def complete_kind_paired(complement, *, xp=np, trips: int = TRIPS):
    """Return paired first- and second-kind values from a paired complement."""
    reachable = paired_value(complement) > 0.0
    held = paired_where(reachable, complement, paired_wrap(1.0), xp)
    radicals, arithmetic = _descent_paired(held, xp, trips)
    one = paired_wrap(xp.ones_like(arithmetic[0]))
    first = _accumulate_paired(radicals, arithmetic, one, 1.0, 1.0, xp)
    second = _accumulate_paired(radicals, arithmetic, one, 1.0, held, xp)
    return (
        paired_where(reachable, first, paired_wrap(0.0 * arithmetic[0]), xp),
        paired_where(reachable, second, paired_wrap(xp.ones_like(arithmetic[0])), xp),
    )


@_array_program
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


@_array_program
def complete_pole_paired(pole, complement, *, xp=np, trips: int = TRIPS):
    """Return the complete pole integral with paired descent arithmetic."""
    live = paired_value(pole) > 0.0
    held_pole = paired_where(live, pole, paired_wrap(1.0), xp)
    reachable = paired_value(complement) > 0.0
    held_complement = paired_where(reachable, complement, paired_wrap(1.0), xp)
    radicals, arithmetic = _descent_paired(held_complement, xp, trips)
    evaluated = _accumulate_paired(radicals, arithmetic, held_pole, 1.0, 1.0, xp)
    finite = paired_wrap(_finite_part(paired_value(held_pole), 1.0, 1.0, xp))
    value = paired_where(reachable, evaluated, finite, xp)
    return paired_where(live, value, paired_wrap(0.0 * value[0]), xp)


# Tangents.  Each function below returns ``(primal, tangent)`` in the structure
# ``jax.jvp`` gives for the function it is named after, and carries the tangent
# through the SAME fixed-trip descent and accumulation as its own recurrence:
# every trip's update is differentiated once, by hand, beside the primal update,
# so a trace holds one tangent step per primal step rather than the derivative
# of the whole unrolled loop.  The primal half is computed by the primal helpers
# themselves, so it is the primal's arithmetic exactly.
#
# Each elementary rule is written in the arrangement ``jax.jvp`` itself uses --
# ``g/y - g_y x/y^2`` for a quotient, ``g (0.5/sqrt x)`` for a root -- because a
# tangent at a small complement is a cancellation of terms of order ``1/k'^2``,
# and two arrangements that differ by an ulp per term would then differ in the
# leading digits of the result.  ``None`` is a tangent known to be zero, and its
# term is left out rather than added as zero, as ``jax.jvp`` leaves it out.


def _tangent_sum(*tangents):
    """Return the sum of the tangents that are present, or ``None``."""
    present = [tangent for tangent in tangents if tangent is not None]
    if not present:
        return None
    total = present[0]
    for tangent in present[1:]:
        total = total + tangent
    return total


def _product_tangent(left, d_left, right, d_right):
    """Return the tangent of ``left * right`` by the product rule."""
    return _tangent_sum(
        None if d_left is None else d_left * right,
        None if d_right is None else left * d_right,
    )


def _quotient_tangent(numerator, d_numerator, denominator, d_denominator):
    """Return the tangent of ``numerator/denominator``."""
    return _tangent_sum(
        None if d_numerator is None else d_numerator / denominator,
        None
        if d_denominator is None
        else (-d_denominator * numerator) * (1.0 / (denominator * denominator)),
    )


def _root_tangent(root, d_radicand):
    """Return the tangent of ``root = sqrt(radicand)``."""
    return None if d_radicand is None else d_radicand * (0.5 / root)


def _scale_tangent(factor, tangent):
    """Return the tangent of ``factor * value`` for a constant factor."""
    return None if tangent is None else factor * tangent


def _held_tangent(condition, tangent, xp):
    """Return the tangent of ``where(condition, value, constant)``."""
    return None if tangent is None else xp.where(condition, tangent, 0.0)


def _descent_tangent_step(radical, d_radical, running, d_running, xp):
    """Return one trip's next modulus and radical with their tangents.

    The modulus is twice the root of the radical, and the next radical is that
    modulus times the trip's updated sum, differentiated by the product rule.
    """
    root = xp.sqrt(radical)
    modulus = 2.0 * root
    d_modulus = _scale_tangent(2.0, _root_tangent(root, d_radical))
    return (
        modulus,
        d_modulus,
        modulus * running,
        _product_tangent(modulus, d_modulus, running, d_running),
    )


def _descent_tangent(complement, d_complement, xp, trips: int = TRIPS):
    """Return ``((radicals, sum), (d_radicals, d_sum))``, the descent and tangent."""
    complement = xp.asarray(complement)
    radicals, arithmetic = _descent(complement, xp, trips)
    reachable = complement > 0.0
    held = xp.where(reachable, complement, 1.0)
    modulus = xp.sqrt(held)
    d_modulus = _root_tangent(modulus, _held_tangent(reachable, d_complement, xp))
    radical, d_radical = modulus, d_modulus
    running, d_running = xp.ones_like(modulus), None
    d_radicals = []
    for _ in range(trips):
        d_radicals.append(d_radical)
        running = running + modulus
        d_running = _tangent_sum(d_running, d_modulus)
        modulus, d_modulus, radical, d_radical = _descent_tangent_step(
            radical, d_radical, running, d_running, xp
        )
    return (radicals, arithmetic), (d_radicals, d_running)


def _accumulate_tangent(
    radicals,
    d_radicals,
    arithmetic,
    d_arithmetic,
    pole,
    d_pole,
    cosine_weight,
    d_cosine_weight,
    sine_weight,
    d_sine_weight,
    xp,
):
    """Return :func:`_accumulate` and its tangent in every floating input."""
    value = _accumulate(radicals, arithmetic, pole, cosine_weight, sine_weight, xp)
    pole_root = xp.sqrt(pole)
    d_pole_root = _root_tangent(pole_root, d_pole)
    cosine_part = cosine_weight + xp.zeros_like(arithmetic)
    d_cosine_part = d_cosine_weight
    sine_part = sine_weight / pole_root
    d_sine_part = _quotient_tangent(sine_weight, d_sine_weight, pole_root, d_pole_root)
    sine_part = sine_part + xp.zeros_like(arithmetic)
    for radical, d_radical in zip(radicals, d_radicals):
        previous, d_previous = cosine_part, d_cosine_part
        d_cosine_part = _tangent_sum(
            d_cosine_part,
            _quotient_tangent(sine_part, d_sine_part, pole_root, d_pole_root),
        )
        cosine_part = cosine_part + sine_part / pole_root
        gain = radical / pole_root
        d_gain = _quotient_tangent(radical, d_radical, pole_root, d_pole_root)
        d_sine_part = _scale_tangent(
            2.0,
            _tangent_sum(
                d_sine_part, _product_tangent(previous, d_previous, gain, d_gain)
            ),
        )
        sine_part = 2.0 * (sine_part + previous * gain)
        pole_root = pole_root + gain
        d_pole_root = _tangent_sum(d_pole_root, d_gain)
    numerator = _HALF_PI * (sine_part + cosine_part * arithmetic)
    d_numerator = _scale_tangent(
        _HALF_PI,
        _tangent_sum(
            d_sine_part,
            _product_tangent(cosine_part, d_cosine_part, arithmetic, d_arithmetic),
        ),
    )
    span = arithmetic + pole_root
    denominator = arithmetic * span
    d_denominator = _product_tangent(
        arithmetic, d_arithmetic, span, _tangent_sum(d_arithmetic, d_pole_root)
    )
    return value, _quotient_tangent(numerator, d_numerator, denominator, d_denominator)


def _finite_part_series_tangent(series_rising, d_series_rising, xp):
    """Return the tangent of the finite part's series in ``p - 1``."""
    power = xp.ones_like(series_rising)
    d_power = None
    d_series = None
    for order in range(1, 8):
        d_power = _scale_tangent(
            -1.0, _product_tangent(power, d_power, series_rising, d_series_rising)
        )
        power = -power * series_rising
        d_series = _tangent_sum(
            d_series, None if d_power is None else d_power / (2 * order + 1)
        )
    return d_series


def _finite_part_tangent(
    pole, d_pole, cosine_weight, d_cosine_weight, sine_weight, d_sine_weight, xp
):
    """Return :func:`_finite_part` and its tangent in every floating input."""
    value = _finite_part(pole, cosine_weight, sine_weight, xp)
    rising = pole - 1.0
    separated = rising != 0.0
    magnitude = xp.abs(rising)
    d_magnitude = xp.where(rising >= 0.0, d_pole, -d_pole)
    root = xp.sqrt(xp.where(separated, magnitude, 1.0))
    d_root = _root_tangent(root, xp.where(separated, d_magnitude, 0.0))
    below = rising < 0.0
    held_pole = xp.where(below, pole, 1.0)
    d_held_pole = xp.where(below, d_pole, 0.0)
    arctangent = xp.arctan(root)
    d_circular = _quotient_tangent(
        arctangent, d_root / (1.0 + root * root), root, d_root
    )
    logarithm = xp.log1p(root) - 0.5 * xp.log(held_pole)
    d_logarithm = d_root / (root + 1.0) - 0.5 * (d_held_pole / held_pole)
    d_hyperbolic = _quotient_tangent(logarithm, d_logarithm, root, d_root)
    d_over = xp.where(rising > 0.0, d_circular, d_hyperbolic)
    close = xp.abs(rising) < 1e-4
    series_rising = xp.where(close, rising, 0.0)
    d_series = _finite_part_series_tangent(
        series_rising, xp.where(close, d_pole, 0.0), xp
    )
    over = xp.where(rising > 0.0, arctangent / root, logarithm / root)
    series = xp.ones_like(rising)
    power = xp.ones_like(rising)
    for order in range(1, 8):
        power = -power * series_rising
        series = series + power / (2 * order + 1)
    elementary = xp.where(close, series, over)
    d_elementary = xp.where(close, d_series, d_over)
    scaled = sine_weight * rising
    coefficient = (cosine_weight - sine_weight) + scaled / pole
    d_coefficient = _tangent_sum(
        d_cosine_weight,
        _scale_tangent(-1.0, d_sine_weight),
        _quotient_tangent(
            scaled,
            _product_tangent(sine_weight, d_sine_weight, rising, d_pole),
            pole,
            d_pole,
        ),
    )
    return value, _product_tangent(coefficient, d_coefficient, elementary, d_elementary)


def _descent_tangent_scanned(complement, d_complement, trips: int = TRIPS):
    """Return the stacked radicals and final sum of the descent, with tangents.

    The same trips as :func:`_descent_tangent`, carried as one scanned step so
    the compiled program holds one trip rather than ``trips`` of them.
    """
    jnp = jax.numpy
    reachable = complement > 0.0
    held = jnp.where(reachable, complement, 1.0)
    modulus = jnp.sqrt(held)
    d_modulus = _root_tangent(modulus, jnp.where(reachable, d_complement, 0.0))
    running = jnp.ones_like(modulus)

    def trip(carry, _):
        modulus, d_modulus, radical, d_radical, running, d_running = carry
        running = running + modulus
        d_running = d_running + d_modulus
        modulus, d_modulus, next_radical, d_next_radical = _descent_tangent_step(
            radical, d_radical, running, d_running, jnp
        )
        carry = (modulus, d_modulus, next_radical, d_next_radical, running, d_running)
        return carry, (radical, d_radical)

    initial = (modulus, d_modulus, modulus, d_modulus, running, jnp.zeros_like(running))
    final, (radicals, d_radicals) = jax.lax.scan(trip, initial, None, length=trips)
    return (radicals, final[4]), (d_radicals, final[5])


def _accumulate_tangent_scanned(
    radicals, d_radicals, arithmetic, d_arithmetic, sine_weight, d_sine_weight
):
    """Return the tangent of :func:`_accumulate` at a pole of one, scanned.

    The pole of the first and second kinds is one with no tangent, so only the
    sine weight and the descent carry one; the trips are one scanned step.
    """
    jnp = jax.numpy
    zero = jnp.zeros_like(arithmetic)
    pole_root = jnp.ones_like(arithmetic)
    sine_part = sine_weight / pole_root + zero
    d_sine_part = zero if d_sine_weight is None else d_sine_weight / pole_root + zero

    def trip(carry, radical_pair):
        cosine_part, sine_part, pole_root, d_cosine, d_sine, d_root = carry
        radical, d_radical = radical_pair
        d_next_cosine = d_cosine + _quotient_tangent(
            sine_part, d_sine, pole_root, d_root
        )
        next_cosine = cosine_part + sine_part / pole_root
        gain = radical / pole_root
        d_gain = _quotient_tangent(radical, d_radical, pole_root, d_root)
        d_sine = 2.0 * (d_sine + _product_tangent(cosine_part, d_cosine, gain, d_gain))
        sine_part = 2.0 * (sine_part + cosine_part * gain)
        return (
            next_cosine, sine_part, pole_root + gain,
            d_next_cosine, d_sine, d_root + d_gain,
        ), None  # fmt: skip

    initial = (zero + 1.0, sine_part, pole_root, zero, d_sine_part, zero)
    final, _ = jax.lax.scan(trip, initial, (radicals, d_radicals))
    cosine_part, sine_part, pole_root, d_cosine, d_sine, d_root = final
    numerator = _HALF_PI * (sine_part + cosine_part * arithmetic)
    d_numerator = _HALF_PI * (
        d_sine + _product_tangent(cosine_part, d_cosine, arithmetic, d_arithmetic)
    )
    span = arithmetic + pole_root
    denominator = arithmetic * span
    d_denominator = _product_tangent(
        arithmetic, d_arithmetic, span, d_arithmetic + d_root
    )
    return _quotient_tangent(numerator, d_numerator, denominator, d_denominator)


def _complete_kind_tangent(complement, d_complement, *, xp=np, trips: int = TRIPS):
    """Return ``((K, E), (dK, dE))``, :func:`complete_kind` and its tangent.

    The primal is :func:`complete_kind` itself; the tangent is the descent and
    both accumulations carried as scanned trips, so its compiled program holds
    one trip of each rather than ``trips`` of them.
    """
    complement = xp.asarray(complement)
    d_complement = xp.asarray(d_complement) + xp.zeros_like(complement)
    values = complete_kind(complement, xp=xp, trips=trips)
    (radicals, arithmetic), (d_radicals, d_arithmetic) = _descent_tangent_scanned(
        complement, d_complement, trips
    )
    reachable = complement > 0.0
    held = xp.where(reachable, complement, 1.0)
    d_held = xp.where(reachable, d_complement, 0.0)
    d_first = _accumulate_tangent_scanned(
        radicals, d_radicals, arithmetic, d_arithmetic, 1.0, None
    )
    d_second = _accumulate_tangent_scanned(
        radicals, d_radicals, arithmetic, d_arithmetic, held, d_held
    )
    return values, (
        xp.where(reachable, d_first, 0.0),
        xp.where(reachable, d_second, 0.0),
    )


def _complete_pole_tangent(
    pole, d_pole, complement, d_complement, *, xp=np, trips: int = TRIPS
):
    """Return :func:`complete_pole` and its tangent in the pole and complement."""
    pole = xp.asarray(pole)
    complement = xp.asarray(complement)
    d_pole = xp.asarray(d_pole) + xp.zeros_like(pole)
    d_complement = xp.asarray(d_complement) + xp.zeros_like(complement)
    live = pole > 0.0
    held_pole = xp.where(live, pole, 1.0)
    d_held_pole = xp.where(live, d_pole, 0.0)
    (radicals, arithmetic), (d_radicals, d_arithmetic) = _descent_tangent(
        complement, d_complement, xp, trips
    )
    evaluated, d_evaluated = _accumulate_tangent(
        radicals, d_radicals, arithmetic, d_arithmetic,
        held_pole, d_held_pole, 1.0, None, 1.0, None, xp,
    )  # fmt: skip
    finite, d_finite = _finite_part_tangent(
        held_pole, d_held_pole, 1.0, None, 1.0, None, xp
    )
    reachable = complement > 0.0
    value = xp.where(reachable, evaluated, finite)
    d_value = xp.where(reachable, d_evaluated, d_finite)
    return xp.where(live, value, 0.0), xp.where(live, d_value, 0.0)
