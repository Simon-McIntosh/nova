"""Contract for Carlson's symmetric forms at a fixed trip count.

These are the machinery under the incomplete third kind and nothing else, so what
is asserted here is what that one caller depends on:

* **the values**, against ``scipy.special``'s own ``elliprf`` and ``elliprj``.
  That is a genuinely independent implementation -- Cephes-derived, iterated to a
  convergence test rather than run a fixed number of times -- so agreement pins
  the duplication and the closing series together.  Where the two disagree the
  tie is broken by the same algorithm run in ``np.longdouble``, which separates
  round-off in the double evaluation from truncation in either.
* **the fixed trip count**, in both directions.  A traced evaluation cannot
  iterate to a convergence test, so the count is a claim about the argument range
  and starving it must fail as surely as padding it must not help.
* **the three arrangements that were measured on the way in** -- the degenerate
  form's logarithm, the third kind's carried scale, and the gaps between the pole
  argument and the other three.  Each stands against an obvious transcription that
  fails by far more than the tolerances above, and each is run here in both forms
  so the claim is a measurement rather than a comment.  The gaps carry two exact
  statements with them -- the factorisation of the two roots' difference and the
  quartering under the step -- and both are asserted as identities here, since
  everything the argument buys follows from them.
"""

import numpy as np
import pytest
from scipy.special import elliprf, elliprj

from nova.biot.symmetricelliptic import TRIPS, _elementary, symmetric_kinds

# The arguments the incomplete third kind forms: ``x`` is the amplitude's squared
# cosine, ``y`` the modulus radical there, ``z`` one, and the pole argument the
# denominator at the amplitude.  Each spans what a ring geometry reaches.
SQUARED_COSINES = [1.0, 0.5, 4e-2, 1e-8, 0.0]
COMPLEMENTS = [1.0, 0.5, 1e-3, 1e-8, 1e-16, 1e-40, 1e-100, 1e-300]
POLES = [1.0, 0.5, 1e-3, 1e-10, 1e-40, 1e-200]


def longdouble_namespace():
    """Return an array namespace that runs the same code in ``np.longdouble``."""

    class Extended:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, term):
            return np.asarray(term, dtype=np.longdouble)

        def zeros_like(self, term):
            return np.zeros_like(term, dtype=np.longdouble)

    return Extended()


@pytest.mark.parametrize("squared_cosine", SQUARED_COSINES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_both_kinds_reproduce_the_reference_implementation(squared_cosine, complement):
    """Over the whole double range of complement and every pole the arc reaches."""
    radical = squared_cosine + complement * (1.0 - squared_cosine)
    for pole in POLES:
        weight = squared_cosine + pole * (1.0 - squared_cosine)
        if weight == 0.0:
            continue
        first, third = symmetric_kinds(squared_cosine, radical, 1.0, weight)
        assert float(first) == pytest.approx(
            float(elliprf(squared_cosine, radical, 1.0)), rel=2e-15
        )
        expected = float(elliprj(squared_cosine, radical, 1.0, weight))
        # the reference gives up at the extreme of its own range; where it does,
        # its own test below records what this one returns instead
        if np.isfinite(expected):
            assert float(third) == pytest.approx(expected, rel=2e-13)


def test_the_third_kind_holds_where_the_reference_implementation_gives_up():
    """Both smallnesses at once, and the tie broken by extended precision.

    ``x = 0`` with a modulus complement of 1e-300 and a pole 1e-200 past the range
    is the corner an iterated implementation cannot take: ``scipy`` returns
    ``nan``.  The value is 3.5e+202 and the same algorithm run in ``np.longdouble``
    puts it there, so what fails is the reference and not the arrangement.
    """
    extended = longdouble_namespace()
    squared_cosine, complement, pole = 0.0, 1e-300, 1e-200
    assert not np.isfinite(elliprj(squared_cosine, complement, 1.0, pole))
    _, third = symmetric_kinds(squared_cosine, complement, 1.0, pole)
    _, settled = symmetric_kinds(
        np.longdouble(squared_cosine),
        np.longdouble(complement),
        np.longdouble(1.0),
        np.longdouble(pole),
        xp=extended,
        trips=TRIPS + 8,
    )
    assert float(third) == pytest.approx(float(settled), rel=4e-15)


def test_the_equal_argument_values_are_exact():
    """``R_F(x,x,x) = x^-1/2`` and ``R_J(x,x,x,x) = x^-3/2``, from the definitions.

    Both are the confluent case the duplication has as its fixed point, and they
    are also where the degenerate form's two roots COINCIDE -- so this is the
    cheapest place a mis-taken branch there would show.
    """
    for value in (1.0, 0.25, 1e-6, 1e6):
        first, third = symmetric_kinds(value, value, value, value)
        assert float(first) == pytest.approx(value**-0.5, rel=4e-16)
        assert float(third) == pytest.approx(value**-1.5, rel=4e-15)


def test_the_carried_scale_is_the_value_it_would_have_been_multiplied_by():
    """Folding the weight into the accumulation must change nothing it can hold."""
    squared_cosine, radical, weight = 0.3, 0.31, 0.4
    for scale in (1.0, 1e-6, 3.5):
        _, plain = symmetric_kinds(squared_cosine, radical, 1.0, weight)
        _, scaled = symmetric_kinds(squared_cosine, radical, 1.0, weight, scale)
        assert float(scaled) == pytest.approx(scale * float(plain), rel=4e-15)


def test_the_carried_scale_is_what_keeps_the_near_pole_in_range():
    """And where it cannot be multiplied afterwards, because the value is not there.

    A root a squared aspect ratio past the start of the range, at a target a
    rounding error from the source ring, puts the third kind past 1e308 while the
    integral it belongs to is 1e-06.  The weight it is destined for is small by
    the same amount, so the product exists and the factors do not.
    """
    squared_cosine, complement, pole = 0.0, 1e-300, 1e-318
    with np.errstate(over="ignore"):
        _, unscaled = symmetric_kinds(squared_cosine, complement, 1.0, pole)
    _, carried = symmetric_kinds(squared_cosine, complement, 1.0, pole, pole / 3.0)
    assert not np.isfinite(float(unscaled))
    assert float(carried) == pytest.approx(1.5708e-09, rel=1e-4)


def accumulated(squared, pole, ratio_form):
    """Return the third kind's accumulated sum, in one arrangement or the other.

    Everything but the degenerate form is shared, so running the two side by side
    isolates exactly the step under test.  ``ratio_form`` takes the logarithm of
    ``(s + root)/t``, the direct transcription; otherwise the shipped
    :func:`nova.biot.symmetricelliptic._elementary` is called.
    """
    total, factor = 0.0, 1.0
    x, y, z, weight = 0.0, squared, 1.0, pole
    for _ in range(TRIPS):
        root_x, root_y, root_z = np.sqrt(x), np.sqrt(y), np.sqrt(z)
        step = root_x * root_y + root_y * root_z + root_z * root_x
        first = weight * (root_x + root_y + root_z) + root_x * root_y * root_z
        second = np.sqrt(weight) * (weight + step)
        if ratio_form:
            gap = (second - first) * (second + first)
            root = np.sqrt(abs(gap))
            if gap > 0.0:
                total += factor * np.arctan2(root, first) / root
            elif gap < 0.0:
                total += factor * np.log((first + root) / second) / root
            else:
                total += factor / first
        else:
            total += float(_elementary(first, second, factor, np))
        factor *= 0.25
        x, y, z, weight = (0.25 * (term + step) for term in (x, y, z, weight))
    return total


def test_the_degenerate_form_taken_as_a_ratio_is_what_this_repairs():
    """The logarithm's argument must be the small quantity, not a ratio near one.

    ``log((s + root)/t)`` is the direct transcription and it is exact in
    arithmetic; in floating point the ratio is one to within its own rounding
    wherever the two roots nearly meet -- which is every pole argument that nearly
    equals one of the other three, and the arc reaches that at any edge end level
    with the target.  Both arrangements are run here so the cost is a measurement:
    it reaches the ninth decimal, six decades outside the tolerance asserted above.
    """
    squared, pole = 1e-3, 1e-3
    extended = longdouble_namespace()
    _, settled = symmetric_kinds(
        np.longdouble(0.0),
        np.longdouble(squared),
        np.longdouble(1.0),
        np.longdouble(pole),
        xp=extended,
        trips=TRIPS + 6,
    )
    _, shipped = symmetric_kinds(0.0, squared, 1.0, pole)
    assert abs(float(shipped) - float(settled)) / float(settled) < 4e-15

    # only the accumulated sum differs between the two; the closing series is
    # common, so swapping one for the other is the whole difference
    strayed = float(shipped) + 3.0 * (
        accumulated(squared, pole, True) - accumulated(squared, pole, False)
    )
    assert abs(strayed - float(settled)) / float(settled) > 1e-9


def test_the_two_roots_differ_by_the_product_of_the_gaps_exactly():
    """The factorisation the carried gaps rest on, checked as an IDENTITY.

    The degenerate form's two arguments are, in the roots ``a, b, c`` of the three
    and ``q`` of the pole,

        s = q^2 (a + b + c) + a b c,   t = q (q^2 + a b + b c + c a)

    and ``t - s`` is ``(q - a)(q - b)(q - c)`` term for term -- a polynomial
    identity, not an approximation.  Its conjugate ``t + s`` is the same product
    with the signs flipped, so it is a SUM OF POSITIVES and cannot cancel, and
    multiplying the two gives ``t^2 - s^2 = (q^2 - a^2)(q^2 - b^2)(q^2 - c^2)``:
    the three gaps between the pole argument and the other three.

    Run over integers, where Python's arithmetic is exact, so what is asserted is
    the algebra rather than its round-off on some particular sample.
    """
    rng = np.random.default_rng(20260728)
    roots = rng.integers(1, 10**6, size=(4000, 4))
    for a, b, c, q in (tuple(int(term) for term in row) for row in roots):
        x, y, z, pole = a * a, b * b, c * c, q * q
        below = pole * (a + b + c) + a * b * c
        above = q * (pole + a * b + b * c + c * a)
        assert above - below == (q - a) * (q - b) * (q - c)
        assert above + below == (q + a) * (q + b) * (q + c)
        assert above**2 - below**2 == (pole - x) * (pole - y) * (pole - z)


def test_the_gaps_quarter_exactly_where_the_subtraction_loses_them():
    """Why the gaps are carried rather than re-formed at each trip.

    The step replaces all four of ``x, y, z, pole`` by ``(term + step)/4`` with the
    SAME ``step``, so every gap between two of them is exactly quartered -- and a
    quarter is exact in binary, so carrying costs nothing at all.  Re-subtracting
    the quartered terms instead keeps only the digits by which they still differ,
    and at the configuration the arc reaches -- a pole argument a part in 1e12 from
    the amplitude's squared cosine -- that is already the fourth digit at the FIRST
    trip and nothing at all by the eighth.  Both are asserted, so the carried form
    losing its exactness fails as surely as the subtraction regaining accuracy.
    """
    co_amplitude, pole, complement = 0.4, 1e-12, 1e-300
    sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
    squared_sine, squared_cosine = sine * sine, cosine * cosine
    x, y, z = squared_cosine, squared_cosine + complement * squared_sine, 1.0
    argument = squared_cosine + pole * squared_sine
    start = pole * squared_sine
    gap = start
    strayed = []
    for trip in range(1, 9):
        root_x, root_y, root_z = np.sqrt(x), np.sqrt(y), np.sqrt(z)
        step = root_x * root_y + root_y * root_z + root_z * root_x
        x, y, z, argument = (0.25 * (term + step) for term in (x, y, z, argument))
        gap = 0.25 * gap
        # the quartering is exact to the last bit, at every trip
        assert gap == start * 0.25**trip
        strayed.append(abs(gap - (argument - x)) / abs(gap))
    assert strayed[0] > 1e-4  # measured 1.7e-04 at the first trip
    assert strayed[-1] > 0.9  # measured 1.0 -- the subtraction returns zero


@pytest.mark.parametrize("squared_cosine", SQUARED_COSINES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_carried_gaps_reproduce_the_reference_implementation(
    squared_cosine, complement
):
    """Supplying the gaps must be the same function everywhere, not a near one.

    The argument exists for the configurations where the two roots nearly meet, but
    it changes the arithmetic at every argument, so it is held to the independent
    implementation over the same grid as the subtraction is.
    """
    radical = squared_cosine + complement * (1.0 - squared_cosine)
    for pole in POLES:
        weight = squared_cosine + pole * (1.0 - squared_cosine)
        if weight == 0.0:
            continue
        gaps = (weight - squared_cosine, weight - radical, weight - 1.0)
        first, third = symmetric_kinds(squared_cosine, radical, 1.0, weight, gaps=gaps)
        assert float(first) == pytest.approx(
            float(elliprf(squared_cosine, radical, 1.0)), rel=2e-15
        )
        expected = float(elliprj(squared_cosine, radical, 1.0, weight))
        if np.isfinite(expected):
            assert float(third) == pytest.approx(expected, rel=2e-13)


def test_the_trip_count_is_bounded_by_the_argument_range_in_both_directions():
    """Long enough that the whole double range has converged, short enough to bite.

    The duplication has the two phases the arithmetic-geometric mean has: the
    trips before the three arguments are of one size halve their exponent gap, and
    the ones after take the closing series' deviations down by four apiece.  Both
    are asserted, so padding the constant fails as surely as starving it.
    """
    complements = np.array([1e-300, 1e-300, 1e-100, 1e-40, 1e-16, 1e-3, 0.5])
    poles = np.array([1e-300, 1.0, 1e-40, 1e-10, 1e-3, 0.5, 1.0])
    taken = {
        trips: symmetric_kinds(0.0, complements, 1.0, poles, trips=trips)
        for trips in (TRIPS - 4, TRIPS, TRIPS + 6)
    }

    def strayed(trips, index):
        got, settled = taken[trips][index], taken[TRIPS + 6][index]
        return np.max(np.abs(np.asarray(got) - settled) / np.abs(settled))

    assert strayed(TRIPS, 0) < 4e-16
    assert strayed(TRIPS, 1) < 4e-15
    # four trips short leaves the widest exponent gap -- a target on the source
    # ring, at a pole three hundred decades past the range -- wrong in the ninth
    # decimal for the first kind and the tenth for the third
    assert strayed(TRIPS - 4, 0) > 1e-9
    assert strayed(TRIPS - 4, 1) > 1e-10


def test_the_traced_path_is_the_same_code_and_agrees_to_a_few_ulp():
    """One implementation, numpy on the host and a compiled kernel on a device."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    complements = np.array([1e-300, 1e-40, 1e-12, 1e-3, 0.4])
    poles = np.array([1e-30, 1e-8, 1e-2, 0.5, 1.0])

    @jax.jit
    def traced(complement, pole):
        return symmetric_kinds(
            jnp.asarray(0.0), complement, jnp.asarray(1.0), pole, xp=jnp
        )

    host = symmetric_kinds(0.0, complements, 1.0, poles)
    device = traced(jnp.asarray(complements), jnp.asarray(poles))
    for one, other in zip(host, device):
        assert np.max(np.abs(np.asarray(other) - one) / np.abs(one)) < 1e-14
