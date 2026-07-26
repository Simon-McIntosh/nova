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

__all__ = [
    "across_the_range",
    "contract",
    "deflate",
    "harmonic_add",
    "harmonic_multiply",
    "harmonic_scale",
    "product",
    "range_function",
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


def total(*terms: tuple) -> tuple:
    """Return the sum of range functions."""
    return (
        harmonic_add(*[term[0] for term in terms]),
        sum(term[1] for term in terms),
        sum(term[2] for term in terms),
    )


def scaled(term: tuple, factor) -> tuple:
    """Return the range function multiplied through by a scalar."""
    return (harmonic_scale(term[0], factor), term[1] * factor, term[2] * factor)


def across_the_range(term: tuple) -> list:
    """Return the range function as one harmonic series."""
    bulk, near, far = term
    return harmonic_add(
        [0.5 * (near + far), 0.5 * (far - near)],
        harmonic_multiply(_BOTH_ENDS, bulk),
    )


def sine_squared_times(series: list) -> tuple:
    """Return ``sin^2 phi`` times a harmonic series, as a range function.

    ``sin^2 phi = 4 x y`` vanishes at both ends, so the product's end values are
    exactly zero whatever the series is -- and a numerator carrying this factor
    puts no weight at all on either pole.  Which is why only the arctangent term,
    the one term without it, needs its end values from the geometry.
    """
    return (harmonic_scale(series, 4.0), 0.0 * series[0], 0.0 * series[0])


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
