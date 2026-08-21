"""A function on the reduction's angle range, held so its two ends stay exact.

The polygon-section reductions -- :mod:`nova.biot.polygonanalytic` for the full
turn and :mod:`nova.biot.polygonarc` for a finite arc -- both contract
polynomials in the transformed angle against families of elliptic moments.  Two
different things have to be true of the representation those polynomials are
carried in, and one basis cannot do both, so the object here carries two.

Write ``phi = pi - 2 a`` and take the two ends of the quarter range as separate
variables,

    x = sin^2 a        vanishing at a = 0    (phi = pi)
    y = cos^2 a        vanishing at a = pi/2 (phi = 0),      x + y = 1

so that ``t = cos 2a = y - x``, ``cos phi = -t`` and ``sin^2 phi = 4 x y``.

The FIRST requirement is ordinary basis conditioning.  The reduced numerators
reach degree six and are bounded, over the range, by roughly the squared major
radius; written in powers of ``t`` their coefficients reach ten thousand times
that, because the monomial basis on a unit interval is that badly conditioned by
degree six.  Contracting such a numerator against a family of same-signed
moments then forms the answer out of terms that exceed it by as much.  So the
BULK of every numerator is carried in the harmonic basis ``cos 2n a``, whose
coefficients are bounded by the function's own size -- a plain Python list of
coefficients, index ``n`` -- and :func:`harmonic_multiply` keeps them bounded
through a product because ``cos 2m a cos 2n a`` splits into a POSITIVE
combination of two harmonics.

The SECOND requirement pulls the other way.  Each denominator the reductions
divide by has a root just past one end of the range, and both of the shifts
involved fall as the square of the section's aspect ratio.  A root that close
makes the pole's own moment large, so the weight it carries -- the numerator's
value AT that end -- must be exact in the RELATIVE sense, and that value is
itself of order the squared aspect ratio.  No harmonic series delivers it: it is
an alternating sum of coefficients of order one.  So a range function is

    N = N(phi = 0) x  +  N(phi = pi) y  +  x y T

with both end values formed directly from the geometry, exactly, and only the
bulk ``T`` as a harmonic series.  :func:`product` and :func:`total` multiply and
add end values on their own, so exactness survives the algebra; and because
``x y/(y + p)`` is bounded by one, the rounding left in ``T`` reaches the answer
unamplified however close the root comes.

The representation is the plain tuple ``(bulk, near, far)`` rather than a class:
these objects are built and combined a few hundred times per corner inside the
reductions' inner assembly, they are traced through ``jax`` as often as they are
evaluated on the host, and every operation on them is a free function here.
"""

from __future__ import annotations

from nova.biot.pairedfloat import add as paired_add
from nova.biot.pairedfloat import multiply as paired_multiply
from nova.biot.pairedfloat import scale as paired_scale
from nova.biot.pairedfloat import subtract as paired_subtract
from nova.biot.pairedfloat import wrap as paired_wrap

__all__ = [
    "across_the_range",
    "as_range_function",
    "contract",
    "deflate",
    "harmonic_add",
    "harmonic_multiply",
    "harmonic_scale",
    "product",
    "paired_across_the_range",
    "paired_deflate",
    "paired_harmonic_multiply",
    "paired_product",
    "paired_range_function",
    "paired_scaled",
    "paired_sine_squared_times",
    "paired_total",
    "range_function",
    "rising_integral",
    "scaled",
    "sine_squared_times",
    "total",
]

# ``x y = (1 - cos 4a)/8`` -- the factor a range function's bulk rides on, as a
# harmonic series.  ``x`` and ``y`` themselves are ``(1 -/+ cos 2a)/2``, which is
# what :func:`across_the_range` folds the two end values onto.
_BOTH_ENDS = [0.125, 0.0, -0.125]


def harmonic_multiply(left: list, right: list) -> list:
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


def paired_harmonic_multiply(left: list, right: list) -> list:
    if not left or not right:
        return []
    zero = paired_wrap(0.0 * left[0][0] * right[0][0])
    out = [zero] * (len(left) + len(right) - 1)
    for index, one in enumerate(left):
        for other_index, other in enumerate(right):
            term = paired_scale(paired_multiply(one, other), 0.5)
            out[index + other_index] = paired_add(out[index + other_index], term)
            out[abs(index - other_index)] = paired_add(
                out[abs(index - other_index)], term
            )
    return out


def _paired_harmonic_add(*series: list) -> list:
    length = max((len(term) for term in series), default=0)
    if length == 0:
        return []
    exemplar = next(term[0] for term in series if term)
    out = [paired_wrap(0.0 * exemplar[0])] * length
    for term in series:
        for index, coefficient in enumerate(term):
            out[index] = paired_add(out[index], coefficient)
    return out


def _paired_harmonic_scale(series: list, factor) -> list:
    return [paired_multiply(coefficient, factor) for coefficient in series]


def harmonic_add(*series: list) -> list:
    """Return the sum of harmonic series."""
    length = max((len(term) for term in series), default=0)
    out: list = [0.0] * length
    for term in series:
        for index, coefficient in enumerate(term):
            out[index] = out[index] + coefficient
    return out


def harmonic_scale(series: list, factor) -> list:
    """Return the harmonic series multiplied through by a scalar."""
    return [coefficient * factor for coefficient in series]


def range_function(bulk: list, near, far) -> tuple:
    """Return the range function ``near x + far y + x y bulk``.

    ``near`` is its value at ``phi = 0`` (``a = pi/2``, the source point closest
    to the target in angle) and ``far`` its value at ``phi = pi``.  Both are held
    apart from the series so a pole sitting on either end multiplies an exact
    quantity; see the module docstring.
    """
    return (bulk, near, far)


def paired_range_function(bulk: list, near, far) -> tuple:
    """Return a range function whose coefficients retain paired-fp64 residues."""
    return bulk, near, far


def product(left: tuple, right: tuple) -> tuple:
    """Return the product of two range functions, end values exact.

    ``x^2 = x - x y`` and ``y^2 = y - x y`` fold the squares back, leaving the
    cross term ``-(near1 - far1)(near2 - far2)`` in the bulk -- so the product's
    end values are the products of the factors' own, formed without touching the
    series.
    """
    bulk, near, far = left
    other_bulk, other_near, other_far = right
    return (
        harmonic_add(
            harmonic_multiply(_BOTH_ENDS, harmonic_multiply(bulk, other_bulk)),
            # each factor's own end values ride on the OTHER factor's bulk, and the
            # pair collapses onto the single two-term series they span
            harmonic_multiply([0.5 * (near + far), 0.5 * (far - near)], other_bulk),
            harmonic_multiply(
                [0.5 * (other_near + other_far), 0.5 * (other_far - other_near)], bulk
            ),
            [-(near - far) * (other_near - other_far)],
        ),
        near * other_near,
        far * other_far,
    )


def paired_product(left: tuple, right: tuple) -> tuple:
    """Multiply paired range functions without rounding their coefficients."""
    bulk, near, far = left
    other_bulk, other_near, other_far = right
    both_ends = [paired_wrap(value) for value in _BOTH_ENDS]
    mean = paired_scale(paired_add(near, far), 0.5)
    slope = paired_scale(paired_subtract(far, near), 0.5)
    other_mean = paired_scale(paired_add(other_near, other_far), 0.5)
    other_slope = paired_scale(paired_subtract(other_far, other_near), 0.5)
    return (
        _paired_harmonic_add(
            paired_harmonic_multiply(
                both_ends, paired_harmonic_multiply(bulk, other_bulk)
            ),
            paired_harmonic_multiply([mean, slope], other_bulk),
            paired_harmonic_multiply([other_mean, other_slope], bulk),
            [
                paired_scale(
                    paired_multiply(
                        paired_subtract(near, far),
                        paired_subtract(other_near, other_far),
                    ),
                    -1.0,
                )
            ],
        ),
        paired_multiply(near, other_near),
        paired_multiply(far, other_far),
    )


def total(*terms: tuple) -> tuple:
    """Return the sum of range functions."""
    return (
        harmonic_add(*[term[0] for term in terms]),
        sum(term[1] for term in terms),
        sum(term[2] for term in terms),
    )


def paired_total(*terms: tuple) -> tuple:
    """Add paired range functions coefficient by coefficient."""
    return (
        _paired_harmonic_add(*[term[0] for term in terms]),
        _paired_sum(term[1] for term in terms),
        _paired_sum(term[2] for term in terms),
    )


def _paired_sum(values) -> tuple:
    values = iter(values)
    total_value = next(values)
    for value in values:
        total_value = paired_add(total_value, value)
    return total_value


def scaled(term: tuple, factor) -> tuple:
    """Return the range function multiplied through by a scalar."""
    return (harmonic_scale(term[0], factor), term[1] * factor, term[2] * factor)


def paired_scaled(term: tuple, factor) -> tuple:
    """Multiply a paired range function by a paired scalar."""
    return (
        _paired_harmonic_scale(term[0], factor),
        paired_multiply(term[1], factor),
        paired_multiply(term[2], factor),
    )


def across_the_range(term: tuple) -> list:
    """Return the range function as one harmonic series."""
    bulk, near, far = term
    return harmonic_add(
        [0.5 * (near + far), 0.5 * (far - near)],
        harmonic_multiply(_BOTH_ENDS, bulk),
    )


def paired_across_the_range(term: tuple) -> list:
    """Return a paired range function as paired harmonic coefficients."""
    bulk, near, far = term
    ends = [
        paired_scale(paired_add(near, far), 0.5),
        paired_scale(paired_subtract(far, near), 0.5),
    ]
    return _paired_harmonic_add(
        ends,
        paired_harmonic_multiply([paired_wrap(value) for value in _BOTH_ENDS], bulk),
    )


def _split_at_both_ends(series: list) -> tuple:
    """Return ``(bulk, half, far)`` of a harmonic series, by double deflation.

    ``series = (t^2 - 1) q + (t - 1) half + far`` with ``t^2 - 1 = -4 x y`` and
    ``t - 1 = -2 x``, so the two remainders ARE the range function's two ends:
    ``far`` is the value at ``t = 1`` and ``far - 2 half`` the value at
    ``t = -1``.  :func:`deflate` performs each step; running it twice is what
    turns a series back into the representation the reductions carry.
    """
    quotient, far = deflate(series, 1.0)
    bulk, half = deflate(quotient, -1.0)
    return harmonic_scale(bulk, -4.0), half, far


def as_range_function(series: list) -> tuple:
    """Return the harmonic series as a range function -- the inverse of
    :func:`across_the_range`.

    Both end values come out of the series here rather than from the geometry, so
    this is the route for a series whose ends are not separately known.  Where
    one of them is known EXACTLY -- and for the antiderivative
    :func:`rising_integral` builds, one of them is exactly zero -- take the route
    that imposes it instead: a value recovered from a series is only as good as
    the cancellation in it.
    """
    bulk, half, far = _split_at_both_ends(series)
    return (bulk, far - 2.0 * half, far)


def _chebyshev_integral(series: list) -> list:
    """Return the ``t``-antiderivative of a harmonic series, constant discarded.

    ``2 integral T_n dt = T_(n+1)/(n+1) - T_(n-1)/(n-1)`` for ``n >= 2``, with
    ``integral T_0 dt = T_1`` and ``integral T_1 dt = (T_2 + T_0)/4``.  The
    ``T_0`` term is left out: every caller fixes the constant by an end value
    instead, which is the whole point of doing this in the range representation.
    """
    if not series:
        return []
    out: list = [0.0 * series[0]] * (len(series) + 1)
    for order, coefficient in enumerate(series):
        if order == 0:
            out[1] = out[1] + coefficient
        elif order == 1:
            out[2] = out[2] + 0.25 * coefficient
        else:
            out[order + 1] = out[order + 1] + 0.5 * coefficient / (order + 1)
            out[order - 1] = out[order - 1] - 0.5 * coefficient / (order - 1)
    return out


def rising_integral(series: list) -> tuple:
    """Return ``integral_0^a sin 2s C(s) ds`` as a range function, ``C`` the series.

    The antiderivative an odd row's weight leaves.  A row weighted by ``sin phi``
    carries ``sin 2a`` against an even coefficient, and ``dx = sin 2a da`` -- so
    the antiderivative is just ``C`` integrated in the range variable ``x``, and
    the one that vanishes at the LOWER limit is the one whose constant is fixed at
    ``x = 0``.  That end is ``phi = pi``, the range function's FAR value, and it
    comes out exactly zero here because the constant is imposed rather than
    summed: the deflation's own remainder is what is discarded.

    Exactness there is not cosmetic.  The transcendental this multiplies diverges
    logarithmically at whichever end its denominator vanishes on, and the pole
    family's seed diverges with it; the two cancel, and they cancel to round-off
    only if the weight the divergence carries is the exact zero rather than a
    residue of the series it was summed from.
    """
    if not series:
        return ([], 0.0, 0.0)
    bulk, half, _ = _split_at_both_ends(
        harmonic_scale(_chebyshev_integral(series), -0.5)
    )
    return (bulk, -2.0 * half, 0.0 * half)


def sine_squared_times(series: list) -> tuple:
    """Return ``sin^2 phi`` times a harmonic series, as a range function.

    ``sin^2 phi = 4 x y`` vanishes at both ends, so the product's end values are
    exactly zero whatever the series is -- and a numerator carrying this factor
    puts no weight at all on either pole.  Which is why only the arctangent term,
    the one term without it, needs its end values from the geometry.
    """
    return (harmonic_scale(series, 4.0), 0.0 * series[0], 0.0 * series[0])


def paired_sine_squared_times(series: list) -> tuple:
    """Multiply a paired harmonic series by ``sin^2 phi`` exactly at both ends."""
    zero = paired_wrap(0.0 * series[0][0])
    return _paired_harmonic_scale(series, paired_wrap(4.0)), zero, zero


def paired_deflate(series: list, root):
    """Deflate a paired harmonic series at a paired root."""
    degree = len(series) - 1
    if degree < 1:
        return [], (series[0] if series else paired_wrap(0.0))
    zero = paired_wrap(0.0 * series[0][0])
    quotient = [zero] * degree
    upper = zero
    current = zero
    for order in range(degree, 1, -1):
        current, upper = (
            paired_subtract(
                paired_add(
                    paired_scale(series[order], 2.0),
                    paired_scale(paired_multiply(root, current), 2.0),
                ),
                upper,
            ),
            current,
        )
        quotient[order - 1] = current
    quotient[0] = paired_subtract(
        paired_add(series[1], paired_multiply(root, current)),
        paired_scale(upper, 0.5),
    )
    return quotient, paired_subtract(
        paired_add(series[0], paired_multiply(root, quotient[0])),
        paired_scale(current, 0.5),
    )


def contract(numerator: list, moments: list):
    """Return the harmonic series contracted against a moment family."""
    total_value = 0.0
    for order, coefficient in enumerate(numerator):
        total_value = total_value + coefficient * moments[order]
    return total_value


def deflate(series: list, root):
    """Return ``(quotient, value)`` with ``series = (t - root) quotient + value``.

    Clenshaw's recursion, which is the harmonic basis's synthetic division.  Run
    downward it follows the branch that grows away from the range, so it is the
    stable direction for a root outside it -- and every denominator's roots are
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
