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

from functools import partial, wraps
from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np

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


def _namespace(*values):
    """Select the array namespace without inspecting traced values."""
    return (
        jnp
        if any(
            isinstance(value, (jax.Array, jax.core.Tracer))
            for value in jax.tree.leaves(values)
        )
        else np
    )


def _array_program(function):
    """Reuse each static-shape helper graph across algebraic call sites."""
    static = tuple(
        name
        for name in ("xp", "count", "mirrored")
        if name in signature(function).parameters
    )
    compiled = jax.jit(function, static_argnames=static)

    @wraps(function)
    def evaluate(*args, **kwargs):
        if _namespace(args, kwargs) is jnp:
            return compiled(*args, **kwargs)
        return function(*args, **kwargs)

    return evaluate


def _pack(values, xp):
    return xp.stack(xp.broadcast_arrays(*values))


def _unpack(values):
    return list(values)


def _pack_pairs(values, xp):
    return tuple(_pack([value[index] for value in values], xp) for index in (0, 1))


def _unpack_pairs(values):
    return list(zip(*values, strict=True))


def _scan(function, initial, values, xp, *, reverse=False):
    """One bounded recurrence on device, with the same ordered host arithmetic."""
    if xp is jnp:
        return jax.lax.scan(function, initial, values, reverse=reverse)
    length = len(jax.tree.leaves(values)[0])
    outputs = []
    state = initial
    for index in range(length - 1, -1, -1) if reverse else range(length):
        state, output = function(
            state, jax.tree.map(lambda value: value[index], values)
        )
        outputs.append(output)
    if reverse:
        outputs.reverse()
    return state, jax.tree.map(lambda *items: np.stack(items), *outputs)


@partial(jax.jit, static_argnames=("xp",))
def _multiply_packed(left, right, *, xp):
    # Each lane retains the nested left-then-right accumulation order.
    rows, columns = np.indices((left.shape[0], right.shape[0]))
    shape = np.broadcast_shapes(left.shape[1:], right.shape[1:])
    left = xp.broadcast_to(
        left.reshape(
            (left.shape[0],) + (1,) * (len(shape) - left.ndim + 1) + left.shape[1:]
        ),
        (left.shape[0],) + shape,
    )
    right = xp.broadcast_to(
        right.reshape(
            (right.shape[0],) + (1,) * (len(shape) - right.ndim + 1) + right.shape[1:]
        ),
        (right.shape[0],) + shape,
    )
    width = left.shape[0] + right.shape[0] - 1
    orders = xp.arange(width).reshape((width,) + (1,) * len(shape))
    initial = xp.zeros((width,) + shape, dtype=xp.result_type(left, right))

    def accumulate(state, indices):
        index, other_index = indices
        term = 0.5 * left[index] * right[other_index]
        state = xp.where(orders == index + other_index, state + term, state)
        state = xp.where(orders == xp.abs(index - other_index), state + term, state)
        return state, None

    result, _ = _scan(
        accumulate, initial, (xp.asarray(rows.ravel()), xp.asarray(columns.ravel())), xp
    )
    return result


def _multiply(left, right, xp):
    if xp is jnp:
        return _multiply_packed(left, right, xp=xp)
    return _multiply_packed.__wrapped__(left, right, xp=xp)


@_array_program
def harmonic_multiply(left: list, right: list) -> list:
    """Return the product of two harmonic series.

    ``cos 2m a cos 2n a = (cos 2(m + n) a + cos 2|m - n| a)/2`` -- a POSITIVE
    combination, which is why a product of bounded factors keeps bounded
    coefficients here where a monomial product does not.
    """
    if not left or not right:
        return []
    xp = _namespace(left, right)
    return _unpack(_multiply(_pack(left, xp), _pack(right, xp), xp))


@_array_program
def paired_harmonic_multiply(left: list, right: list) -> list:
    if not left or not right:
        return []
    xp = _namespace(left, right)
    high = xp.broadcast_arrays(*(value[0] for value in left + right))
    low = xp.broadcast_arrays(*(value[1] for value in left + right))
    left_high, left_low = xp.stack(high[: len(left)]), xp.stack(low[: len(left)])
    right_high, right_low = xp.stack(high[len(left) :]), xp.stack(low[len(left) :])
    rows, columns = np.indices((len(left), len(right)))
    terms = paired_scale(
        paired_multiply(
            (left_high[:, None], left_low[:, None]),
            (right_high[None, :], right_low[None, :]),
        ),
        0.5,
    )
    shape = (len(left) + len(right) - 1,) + terms[0].shape[2:]
    zero = paired_wrap(xp.zeros(shape, dtype=terms[0].dtype))
    orders = xp.arange(shape[0]).reshape((-1,) + (1,) * (len(shape) - 1))

    def accumulate(state, item):
        term_high, term_low, rising, falling = item
        term = term_high, term_low
        for index in (rising, falling):
            added = paired_add(state, term)
            state = tuple(
                xp.where(orders == index, value, prior)
                for value, prior in zip(added, state, strict=True)
            )
        return state, None

    result, _ = _scan(
        accumulate,
        zero,
        (
            terms[0].reshape((-1,) + shape[1:]),
            terms[1].reshape((-1,) + shape[1:]),
            xp.asarray((rows + columns).ravel()),
            xp.asarray(abs(rows - columns).ravel()),
        ),
        xp,
    )
    return _unpack_pairs(result)


@_array_program
def _paired_harmonic_add(*series: list) -> list:
    length = max((len(term) for term in series), default=0)
    if length == 0:
        return []
    xp = _namespace(series)
    exemplar = next(term[0] for term in series if term)
    zero = paired_wrap(0.0 * exemplar[0])
    high, low = _pack_pairs(
        [value for term in series for value in term + [zero] * (length - len(term))], xp
    )
    high = high.reshape((len(series), length) + high.shape[1:])
    low = low.reshape((len(series), length) + low.shape[1:])

    def add(state, term):
        return paired_add(state, term), None

    initial = paired_wrap(xp.zeros_like(high[0]))
    result, _ = _scan(add, initial, (high, low), xp)
    return _unpack_pairs(result)


@_array_program
def _paired_harmonic_scale(series: list, factor) -> list:
    if not series:
        return []
    xp = _namespace(series, factor)
    # Broadcast coefficients before adding the leading term axis.
    high = xp.broadcast_arrays(*(value[0] for value in series), factor[0])
    low = xp.broadcast_arrays(*(value[1] for value in series), factor[1])
    return _unpack_pairs(
        paired_multiply((xp.stack(high[:-1]), xp.stack(low[:-1])), factor)
    )


@_array_program
def harmonic_add(*series: list) -> list:
    """Return the sum of harmonic series."""
    length = max((len(term) for term in series), default=0)
    if length == 0:
        return []
    xp = _namespace(series)
    packed = _pack(
        [value for term in series for value in term + [0.0] * (length - len(term))], xp
    )
    packed = packed.reshape((len(series), length) + packed.shape[1:])

    def add(state, term):
        return state + term, None

    result, _ = _scan(add, xp.zeros_like(packed[0]), packed, xp)
    return _unpack(result)


@_array_program
def harmonic_scale(series: list, factor) -> list:
    """Return the harmonic series multiplied through by a scalar."""
    if not series:
        return []
    xp = _namespace(series, factor)
    return _unpack(_pack(xp.broadcast_arrays(*series, factor)[:-1], xp) * factor)


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
    values = list(values)
    xp = _namespace(values)

    def add(state, value):
        return paired_add(state, value), None

    result, _ = (
        _scan(add, values[0], _pack_pairs(values[1:], xp), xp)
        if len(values) > 1
        else (values[0], None)
    )
    return result


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
    xp = _namespace(series)
    packed = _pack(series, xp)
    width = len(series) + 1
    orders = xp.arange(width).reshape((width,) + (1,) * (packed.ndim - 1))
    initial = xp.zeros((width,) + packed.shape[1:], dtype=packed.dtype)

    def integrate(state, item):
        order, coefficient = item
        rising = xp.where(
            order == 0,
            coefficient,
            xp.where(order == 1, 0.25 * coefficient, 0.5 * coefficient / (order + 1)),
        )
        state = xp.where(orders == order + 1, state + rising, state)
        falling = 0.5 * coefficient / xp.maximum(order - 1, 1)
        state = xp.where((order >= 2) & (orders == order - 1), state - falling, state)
        return state, None

    result, _ = _scan(integrate, initial, (xp.arange(len(series)), packed), xp)
    return _unpack(result)


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
    xp = _namespace(series, root)
    high = xp.stack(xp.broadcast_arrays(*(value[0] for value in series), root[0])[:-1])
    low = xp.stack(xp.broadcast_arrays(*(value[1] for value in series), root[1])[:-1])
    zero = paired_wrap(xp.zeros_like(high[0] + root[0]))

    def descend(state, coefficient):
        current, upper = state
        value = paired_subtract(
            paired_add(
                paired_scale(coefficient, 2.0),
                paired_scale(paired_multiply(root, current), 2.0),
            ),
            upper,
        )
        return (value, current), value

    (current, upper), tail = (
        _scan(descend, (zero, zero), (high[2:], low[2:]), xp, reverse=True)
        if degree > 1
        else ((zero, zero), (high[:0], low[:0]))
    )
    leading = paired_subtract(
        paired_add(series[1], paired_multiply(root, current)), paired_scale(upper, 0.5)
    )
    quotient = [leading] + _unpack_pairs(tail)
    value = paired_subtract(
        paired_add(series[0], paired_multiply(root, leading)),
        paired_scale(current, 0.5),
    )
    return quotient, value


@_array_program
def contract(numerator: list, moments: list):
    """Return the harmonic series contracted against a moment family."""
    if not numerator:
        return 0.0
    xp = _namespace(numerator, moments)
    coefficients = _pack(numerator, xp)
    values = _pack(moments[: len(numerator)], xp)
    shape = np.broadcast_shapes(coefficients.shape[1:], values.shape[1:])

    def accumulate(total_value, term):
        coefficient, moment = term
        return total_value + coefficient * moment, None

    result, _ = _scan(
        accumulate,
        xp.zeros(shape, dtype=xp.result_type(coefficients, values)),
        (coefficients, values),
        xp,
    )
    return result


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
    xp = _namespace(series, root)
    packed = _pack(xp.broadcast_arrays(*series, root)[:-1], xp)
    zero = xp.zeros_like(packed[0] + root)

    def descend(state, coefficient):
        current, upper = state
        value = 2.0 * coefficient + 2.0 * root * current - upper
        return (value, current), value

    (current, upper), tail = (
        _scan(descend, (zero, zero), packed[2:], xp, reverse=True)
        if degree > 1
        else ((zero, zero), packed[:0])
    )
    leading = series[1] + root * current - 0.5 * upper
    return [leading] + _unpack(tail), series[0] + root * leading - 0.5 * current
