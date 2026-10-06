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

from functools import wraps
from inspect import signature

try:
    import jax
except ModuleNotFoundError as error:
    if error.name != "jax":
        raise
    jax = None

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


def _is_staged(value):
    """Distinguish staged operands from eager differentiation and batching."""
    if jax is None:
        return False
    while isinstance(value, jax.core.Tracer):
        if hasattr(value, "primal"):
            value = value.primal
        elif hasattr(value, "batch_dim"):
            value = value.val
        else:
            return value.to_concrete_value() is None
    return False


def _array_program(function):
    """Reuse each static-shape helper graph across algebraic call sites."""
    if jax is None:
        return function
    static = tuple(
        name
        for name in ("xp", "count", "mirrored", "trips", "coincident")
        if name in signature(function).parameters
    )
    compiled = jax.jit(function, static_argnames=static)

    @wraps(function)
    def evaluate(*args, **kwargs):
        if any(_is_staged(value) for value in jax.tree.leaves((args, kwargs))):
            return compiled(*args, **kwargs)
        return function(*args, **kwargs)

    return evaluate


@_array_program
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
            rising = index + other_index
            falling = abs(index - other_index)
            first, second = _harmonic_pair_step(
                out[rising],
                out[falling],
                one,
                other,
                coincident=rising == falling,
            )
            out[rising] = first
            out[falling] = second
    return out


@_array_program
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


@_array_program
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


@_array_program
def _paired_harmonic_scale(series: list, factor) -> list:
    return [paired_multiply(coefficient, factor) for coefficient in series]


@_array_program
def harmonic_add(*series: list) -> list:
    """Return the sum of harmonic series."""
    length = max((len(term) for term in series), default=0)
    out: list = [0.0] * length
    for term in series:
        for index, coefficient in enumerate(term):
            out[index] = out[index] + coefficient
    return out


@_array_program
def harmonic_scale(series: list, factor) -> list:
    """Return the harmonic series multiplied through by a scalar."""
    return [coefficient * factor for coefficient in series]


@_array_program
def range_function(bulk: list, near, far) -> tuple:
    """Return the range function ``near x + far y + x y bulk``.

    ``near`` is its value at ``phi = 0`` (``a = pi/2``, the source point closest
    to the target in angle) and ``far`` its value at ``phi = pi``.  Both are held
    apart from the series so a pole sitting on either end multiplies an exact
    quantity; see the module docstring.
    """
    return (bulk, near, far)


@_array_program
def paired_range_function(bulk: list, near, far) -> tuple:
    """Return a range function whose coefficients retain paired-fp64 residues."""
    return bulk, near, far


@_array_program
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


@_array_program
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


@_array_program
def total(*terms: tuple) -> tuple:
    """Return the sum of range functions."""
    return (
        harmonic_add(*[term[0] for term in terms]),
        sum(term[1] for term in terms),
        sum(term[2] for term in terms),
    )


@_array_program
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


@_array_program
def scaled(term: tuple, factor) -> tuple:
    """Return the range function multiplied through by a scalar."""
    return (harmonic_scale(term[0], factor), term[1] * factor, term[2] * factor)


@_array_program
def paired_scaled(term: tuple, factor) -> tuple:
    """Multiply a paired range function by a paired scalar."""
    return (
        _paired_harmonic_scale(term[0], factor),
        paired_multiply(term[1], factor),
        paired_multiply(term[2], factor),
    )


@_array_program
def across_the_range(term: tuple) -> list:
    """Return the range function as one harmonic series."""
    bulk, near, far = term
    return harmonic_add(
        [0.5 * (near + far), 0.5 * (far - near)],
        harmonic_multiply(_BOTH_ENDS, bulk),
    )


@_array_program
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


@_array_program
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


@_array_program
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


@_array_program
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


@_array_program
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


@_array_program
def sine_squared_times(series: list) -> tuple:
    """Return ``sin^2 phi`` times a harmonic series, as a range function.

    ``sin^2 phi = 4 x y`` vanishes at both ends, so the product's end values are
    exactly zero whatever the series is -- and a numerator carrying this factor
    puts no weight at all on either pole.  Which is why only the arctangent term,
    the one term without it, needs its end values from the geometry.
    """
    return (harmonic_scale(series, 4.0), 0.0 * series[0], 0.0 * series[0])


@_array_program
def paired_sine_squared_times(series: list) -> tuple:
    """Multiply a paired harmonic series by ``sin^2 phi`` exactly at both ends."""
    zero = paired_wrap(0.0 * series[0][0])
    return _paired_harmonic_scale(series, paired_wrap(4.0)), zero, zero


@_array_program
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


@_array_program
def contract(numerator: list, moments: list):
    """Return the harmonic series contracted against a moment family."""
    total_value = 0.0
    for order, coefficient in enumerate(numerator):
        total_value = _product_sum_step(total_value, coefficient, moments[order])
    return total_value


@_array_program
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


@_array_program
def _harmonic_pair_step(rising, falling, one, other, *, coincident):
    """Apply both product-to-sum contributions in their serial update order."""
    term = 0.5 * one * other
    rising = rising + term
    falling = (rising if coincident else falling) + term
    return rising, falling


@_array_program
def _product_sum_step(total_value, coefficient, moment):
    """Retain the multiply then accumulate expression at each term."""
    return total_value + coefficient * moment


# Tangents of the operations above, each ``(primal, tangent)`` in the structure
# ``jax.jvp`` gives for the function it is named after.  Every operation here is
# polynomial in its coefficients, so a tangent is the same recurrence carried
# once more with the product rule applied term by term -- written in the order
# and arrangement ``jax.jvp`` uses, because a tangent coefficient is often a
# cancellation of terms far larger than itself and two arrangements that differ
# by an ulp per term would differ in the leading digits of the result.
# ``None`` is a tangent known to be zero (a constant such as the harmonic basis
# of the two ends), and its term is left out rather than added as zero, as
# ``jax.jvp`` leaves it out.  A series' tangent is a list of the same length.
# Unrolled forms are used throughout: the loops are over the short static
# lengths the reductions carry, and each step is its own staged program so its
# compiled body is shared across every call site.


def _tangent_sum(*tangents):
    """Return the sum of the tangents that are present, or ``None``."""
    present = [tangent for tangent in tangents if tangent is not None]
    if not present:
        return None
    total_value = present[0]
    for tangent in present[1:]:
        total_value = total_value + tangent
    return total_value


def _tangent_difference(d_left, d_right):
    """Return the tangent of ``left - right``."""
    if d_right is None:
        return d_left
    if d_left is None:
        return -d_right
    return d_left - d_right


def _product_tangent(left, d_left, right, d_right):
    """Return the tangent of ``left * right`` by the product rule."""
    return _tangent_sum(
        None if d_left is None else d_left * right,
        None if d_right is None else left * d_right,
    )


def _quotient_tangent(numerator, d_numerator, denominator, d_denominator):
    """Return the tangent of ``numerator / denominator``.

    Ordered as the quotient and the reciprocal square are formed, so a value
    whose two halves cancel rounds as the primal program's own tangent does.
    """
    return _tangent_sum(
        None if d_numerator is None else d_numerator / denominator,
        None
        if d_denominator is None
        else (-d_denominator * numerator) * (1.0 / (denominator * denominator)),
    )


def _scale_tangent(factor, tangent):
    """Return the tangent of ``factor * value`` for a constant factor."""
    return None if tangent is None else factor * tangent


def _series_tangent(series: list, tangent) -> list:
    """Return a series' tangent as a list of its length."""
    return [None] * len(series) if tangent is None else list(tangent)


@_array_program
def _harmonic_pair_step_tangent(
    d_rising, d_falling, one, d_one, other, d_other, *, coincident
):
    """Return the tangent of :func:`_harmonic_pair_step`'s two accumulators."""
    d_term = _product_tangent(0.5 * one, _scale_tangent(0.5, d_one), other, d_other)
    d_rising = _tangent_sum(d_rising, d_term)
    d_falling = _tangent_sum(d_rising if coincident else d_falling, d_term)
    return d_rising, d_falling


@_array_program
def _harmonic_multiply_tangent(left: list, d_left, right: list, d_right):
    """Return :func:`harmonic_multiply` and its tangent in both factors."""
    value = harmonic_multiply(left, right)
    if not left or not right:
        return value, []
    d_left = _series_tangent(left, d_left)
    d_right = _series_tangent(right, d_right)
    d_out: list = [None] * (len(left) + len(right) - 1)
    for index, one in enumerate(left):
        for other_index, other in enumerate(right):
            rising = index + other_index
            falling = abs(index - other_index)
            d_out[rising], d_out[falling] = _harmonic_pair_step_tangent(
                d_out[rising],
                d_out[falling],
                one,
                d_left[index],
                other,
                d_right[other_index],
                coincident=rising == falling,
            )
    return value, d_out


@_array_program
def _harmonic_sum_tangent(series: tuple, d_series: tuple):
    """Return :func:`harmonic_add` of the series and its tangent."""
    value = harmonic_add(*series)
    d_out: list = [None] * len(value)
    for term, d_term in zip(series, d_series, strict=True):
        d_term = _series_tangent(term, d_term)
        for index in range(len(term)):
            d_out[index] = _tangent_sum(d_out[index], d_term[index])
    return value, d_out


@_array_program
def _harmonic_scale_tangent(series: list, d_series, factor, d_factor):
    """Return :func:`harmonic_scale` and its tangent in series and factor."""
    d_series = _series_tangent(series, d_series)
    return harmonic_scale(series, factor), [
        _product_tangent(coefficient, d_coefficient, factor, d_factor)
        for coefficient, d_coefficient in zip(series, d_series, strict=True)
    ]


@_array_program
def _range_function_tangent(bulk: list, d_bulk, near, d_near, far, d_far):
    """Return :func:`range_function` and its tangent, which is the identity."""
    return (bulk, near, far), (d_bulk, d_near, d_far)


@_array_program
def _paired_range_function_tangent(bulk: list, d_bulk, near, d_near, far, d_far):
    """Return :func:`paired_range_function` and its tangent, the identity."""
    return (bulk, near, far), (d_bulk, d_near, d_far)


def _ends_series_tangent(near, d_near, far, d_far):
    """Return the two-term series ``[(near + far)/2, (far - near)/2]`` and tangent."""
    return (
        [0.5 * (near + far), 0.5 * (far - near)],
        [
            _scale_tangent(0.5, _tangent_sum(d_near, d_far)),
            _scale_tangent(0.5, _tangent_difference(d_far, d_near)),
        ],
    )


@_array_program
def _product_range_tangent(left: tuple, d_left: tuple, right: tuple, d_right: tuple):
    """Return :func:`product` and its tangent in both range functions."""
    bulk, near, far = left
    d_bulk, d_near, d_far = d_left
    other_bulk, other_near, other_far = right
    d_other_bulk, d_other_near, d_other_far = d_right
    cross, d_cross = _harmonic_multiply_tangent(bulk, d_bulk, other_bulk, d_other_bulk)
    both, d_both = _harmonic_multiply_tangent(_BOTH_ENDS, None, cross, d_cross)
    ends, d_ends = _ends_series_tangent(near, d_near, far, d_far)
    other_ends, d_other_ends = _ends_series_tangent(
        other_near, d_other_near, other_far, d_other_far
    )
    onto_other, d_onto_other = _harmonic_multiply_tangent(
        ends, d_ends, other_bulk, d_other_bulk
    )
    onto_bulk, d_onto_bulk = _harmonic_multiply_tangent(
        other_ends, d_other_ends, bulk, d_bulk
    )
    gap = near - far
    other_gap = other_near - other_far
    d_gap = _tangent_difference(d_near, d_far)
    d_other_gap = _tangent_difference(d_other_near, d_other_far)
    constant = -gap * other_gap
    d_constant = _product_tangent(
        -gap, None if d_gap is None else -d_gap, other_gap, d_other_gap
    )
    series, d_series = _harmonic_sum_tangent(
        (both, onto_other, onto_bulk, [constant]),
        (d_both, d_onto_other, d_onto_bulk, [d_constant]),
    )
    return (series, near * other_near, far * other_far), (
        d_series,
        _product_tangent(near, d_near, other_near, d_other_near),
        _product_tangent(far, d_far, other_far, d_other_far),
    )


@_array_program
def _total_tangent(terms: tuple, d_terms: tuple):
    """Return :func:`total` of the range functions and its tangent."""
    series, d_series = _harmonic_sum_tangent(
        tuple(term[0] for term in terms), tuple(term[0] for term in d_terms)
    )
    return (
        (series, sum(term[1] for term in terms), sum(term[2] for term in terms)),
        (
            d_series,
            _tangent_sum(*[term[1] for term in d_terms]),
            _tangent_sum(*[term[2] for term in d_terms]),
        ),
    )


@_array_program
def _scaled_tangent(term: tuple, d_term: tuple, factor, d_factor):
    """Return :func:`scaled` and its tangent in the range function and factor."""
    series, d_series = _harmonic_scale_tangent(term[0], d_term[0], factor, d_factor)
    return (series, term[1] * factor, term[2] * factor), (
        d_series,
        _product_tangent(term[1], d_term[1], factor, d_factor),
        _product_tangent(term[2], d_term[2], factor, d_factor),
    )


@_array_program
def _across_the_range_tangent(term: tuple, d_term: tuple):
    """Return :func:`across_the_range` and its tangent."""
    bulk, near, far = term
    d_bulk, d_near, d_far = d_term
    ends, d_ends = _ends_series_tangent(near, d_near, far, d_far)
    spread, d_spread = _harmonic_multiply_tangent(_BOTH_ENDS, None, bulk, d_bulk)
    return _harmonic_sum_tangent((ends, spread), (d_ends, d_spread))


@_array_program
def _sine_squared_times_tangent(series: list, d_series):
    """Return :func:`sine_squared_times` and its tangent."""
    bulk, d_bulk = _harmonic_scale_tangent(series, d_series, 4.0, None)
    d_first = _series_tangent(series, d_series)[0]
    return (bulk, 0.0 * series[0], 0.0 * series[0]), (
        d_bulk,
        _scale_tangent(0.0, d_first),
        _scale_tangent(0.0, d_first),
    )


@_array_program
def _deflate_step_tangent(
    coefficient, d_coefficient, root, d_root, current, d_current, upper, d_upper
):
    """Return one downward step of :func:`deflate` and its tangent."""
    pull = 2.0 * root
    d_pull = _scale_tangent(2.0, d_root)
    value = 2.0 * coefficient + pull * current - upper
    d_value = _tangent_difference(
        _tangent_sum(
            _scale_tangent(2.0, d_coefficient),
            _product_tangent(pull, d_pull, current, d_current),
        ),
        d_upper,
    )
    return value, d_value


@_array_program
def _deflate_tangent(series: list, d_series, root, d_root):
    """Return :func:`deflate` and its tangent in the series and the root.

    The downward recursion is differentiated step by step, from the same start
    and over the same orders, so the tangent is the same synthetic division
    applied to the tangent series with the root's own tangent entering each step.
    """
    degree = len(series) - 1
    d_series = _series_tangent(series, d_series)
    if degree < 1:
        return ([], series[0] if series else 0.0), ([], d_series[0] if series else None)
    quotient: list = [0.0] * degree
    d_quotient: list = [None] * degree
    upper, d_upper = 0.0, None
    current, d_current = 0.0, None
    for order in range(degree, 1, -1):
        (current, d_current), (upper, d_upper) = (
            _deflate_step_tangent(
                series[order],
                d_series[order],
                root,
                d_root,
                current,
                d_current,
                upper,
                d_upper,
            ),
            (current, d_current),
        )
        quotient[order - 1] = current
        d_quotient[order - 1] = d_current
    quotient[0] = series[1] + root * current - 0.5 * upper
    d_quotient[0] = _tangent_difference(
        _tangent_sum(d_series[1], _product_tangent(root, d_root, current, d_current)),
        _scale_tangent(0.5, d_upper),
    )
    remainder = series[0] + root * quotient[0] - 0.5 * current
    d_remainder = _tangent_difference(
        _tangent_sum(
            d_series[0], _product_tangent(root, d_root, quotient[0], d_quotient[0])
        ),
        _scale_tangent(0.5, d_current),
    )
    return (quotient, remainder), (d_quotient, d_remainder)


@_array_program
def _product_sum_step_tangent(
    d_total_value, coefficient, d_coefficient, moment, d_moment
):
    """Return the tangent of :func:`_product_sum_step`'s accumulator."""
    return _tangent_sum(
        d_total_value, _product_tangent(coefficient, d_coefficient, moment, d_moment)
    )


@_array_program
def _contract_tangent(numerator: list, d_numerator, moments: list, d_moments):
    """Return :func:`contract` and its tangent in the numerator and moments."""
    d_numerator = _series_tangent(numerator, d_numerator)
    d_moments = _series_tangent(moments, d_moments)
    d_total_value = None
    for order, coefficient in enumerate(numerator):
        d_total_value = _product_sum_step_tangent(
            d_total_value,
            coefficient,
            d_numerator[order],
            moments[order],
            d_moments[order],
        )
    return contract(numerator, moments), d_total_value


@_array_program
def _chebyshev_integral_tangent(series: list, d_series):
    """Return :func:`_chebyshev_integral` and its tangent."""
    d_series = _series_tangent(series, d_series)
    value = _chebyshev_integral(series)
    if not series:
        return value, []
    d_out: list = [_scale_tangent(0.0, d_series[0])] * (len(series) + 1)
    for order, d_coefficient in enumerate(d_series):
        if order == 0:
            d_out[1] = _tangent_sum(d_out[1], d_coefficient)
        elif order == 1:
            d_out[2] = _tangent_sum(d_out[2], _scale_tangent(0.25, d_coefficient))
        else:
            d_out[order + 1] = _tangent_sum(
                d_out[order + 1],
                None if d_coefficient is None else (0.5 * d_coefficient) / (order + 1),
            )
            d_out[order - 1] = _tangent_difference(
                d_out[order - 1],
                None if d_coefficient is None else (0.5 * d_coefficient) / (order - 1),
            )
    return value, d_out


@_array_program
def _split_at_both_ends_tangent(series: list, d_series):
    """Return :func:`_split_at_both_ends` and its tangent."""
    (quotient, far), (d_quotient, d_far) = _deflate_tangent(series, d_series, 1.0, None)
    (bulk, half), (d_bulk, d_half) = _deflate_tangent(quotient, d_quotient, -1.0, None)
    scaled_bulk, d_scaled = _harmonic_scale_tangent(bulk, d_bulk, -4.0, None)
    return (scaled_bulk, half, far), (d_scaled, d_half, d_far)


@_array_program
def _rising_integral_tangent(series: list, d_series):
    """Return :func:`rising_integral` and its tangent."""
    integral, d_integral = _chebyshev_integral_tangent(series, d_series)
    halved, d_halved = _harmonic_scale_tangent(integral, d_integral, -0.5, None)
    (bulk, half, _), (d_bulk, d_half, _) = _split_at_both_ends_tangent(halved, d_halved)
    return (bulk, -2.0 * half, 0.0 * half), (
        d_bulk,
        _scale_tangent(-2.0, d_half),
        _scale_tangent(0.0, d_half),
    )
