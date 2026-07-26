"""The representation both polygon-section reductions carry their numerators in.

:mod:`nova.biot.rangefunction` holds a polynomial on the reduction's angle range
as two exact end values plus a harmonic bulk, and the tests here pin the algebra
that moves between that form and a plain harmonic series -- in both directions,
since the finite arc needs the way back that the full turn never did.

Everything is checked by EVALUATION rather than by coefficient comparison: a
range function and a harmonic series are the same function of the angle, so
sampling both across the range is the statement that they agree, and it does not
depend on either one's internal shape.
"""

import numpy as np
import pytest

from nova.biot.rangefunction import (
    across_the_range,
    as_range_function,
    deflate,
    harmonic_add,
    harmonic_multiply,
    harmonic_scale,
    product,
    range_function,
    rising_integral,
    scaled,
    sine_squared_times,
    total,
)

# Angles spanning the quarter range, both ends included -- the ends are where a
# range function's two exact values live and where every degeneracy sits.
ANGLE = np.linspace(0.0, 0.5 * np.pi, 41)

SERIES = {
    "constant": [1.7],
    "linear": [0.3, -1.2],
    "quadratic": [0.0, 2.5, -0.75],
    "deep": [1.0, -0.5, 0.25, 0.125, -0.0625, 0.03125, -2.0, 0.5],
    "alternating": [1.0, -1.0, 1.0, -1.0, 1.0],
}


def harmonic_value(series, angle):
    """Return the harmonic series evaluated at the angle."""
    return sum(
        coefficient * np.cos(2.0 * order * angle)
        for order, coefficient in enumerate(series)
    )


def range_value(term, angle):
    """Return the range function evaluated at the angle."""
    bulk, near, far = term
    x, y = np.sin(angle) ** 2, np.cos(angle) ** 2
    return near * x + far * y + x * y * harmonic_value(bulk, angle)


@pytest.mark.parametrize("name", sorted(SERIES))
def test_a_series_and_its_range_function_are_the_same_function(name):
    """The inverse of across_the_range, checked by evaluation across the range."""
    series = SERIES[name]
    term = as_range_function(series)
    assert np.allclose(
        range_value(term, ANGLE), harmonic_value(series, ANGLE), rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("name", sorted(SERIES))
def test_the_round_trip_returns_the_series_it_started_from(name):
    """across_the_range and as_range_function invert each other on the series."""
    series = SERIES[name]
    returned = across_the_range(as_range_function(series))
    padded = series + [0.0] * (len(returned) - len(series))
    assert np.allclose(returned, padded, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("name", sorted(SERIES))
def test_the_round_trip_returns_the_range_function_it_started_from(name):
    """And the other way round, on a range function with unrelated end values."""
    term = range_function(SERIES[name], -0.4, 1.9)
    returned = as_range_function(across_the_range(term))
    assert np.allclose(
        range_value(returned, ANGLE), range_value(term, ANGLE), rtol=0.0, atol=1e-14
    )
    assert returned[1] == pytest.approx(term[1], abs=1e-14)
    assert returned[2] == pytest.approx(term[2], abs=1e-14)


@pytest.mark.parametrize("name", sorted(SERIES))
def test_the_rising_integral_is_the_integral_it_claims_to_be(name):
    """``integral_0^a sin 2s C(s) ds``, against a fine trapezium of the same."""
    series = SERIES[name]
    term = rising_integral(series)
    fine = np.linspace(0.0, 0.5 * np.pi, 200001)
    integrand = np.sin(2.0 * fine) * harmonic_value(series, fine)
    reference = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine))]
    )
    got = range_value(term, fine)
    assert np.max(np.abs(got - reference)) <= 1e-9 * max(1.0, np.max(np.abs(reference)))


@pytest.mark.parametrize("name", sorted(SERIES))
def test_the_rising_integral_vanishes_at_the_lower_limit_exactly(name):
    """Not to round-off -- exactly, which is what the divergence it multiplies needs.

    The transcendental an odd row's antiderivative multiplies runs to a logarithm
    at whichever end its own denominator vanishes on, and the pole family's seed
    runs there with it.  The two cancel; a far end value of a few ulp instead of
    zero would leave the difference of two logarithms weighted by round-off.
    """
    assert rising_integral(SERIES[name])[2] == 0.0


def test_the_rising_integral_reaches_the_whole_range_at_a_quarter_turn():
    """Its near end value is the weight's integral over the whole quarter range."""
    for name, series in SERIES.items():
        near = rising_integral(series)[1]
        # integral_0^(pi/2) sin 2a cos 2na da = -1/(n^2 - 1) for even n, 0 for odd
        reference = sum(
            -coefficient / (order * order - 1.0)
            for order, coefficient in enumerate(series)
            if order % 2 == 0
        )
        assert near == pytest.approx(reference, rel=1e-14), name


def test_the_helpers_carry_arrays_as_readily_as_scalars():
    """Every coefficient is a per-target column inside the reductions."""
    column = np.linspace(0.5, 2.0, 7)
    term = rising_integral([column, -0.5 * column, 0.25 * column])
    scalar = rising_integral([1.0, -0.5, 0.25])
    assert np.allclose(term[1], column * scalar[1])
    assert np.allclose(term[0][0], column * scalar[0][0])


# Roots the reductions actually hand the deflation: ``+/-(1 + 2 shift)`` with the
# shift below the switch at which the pole family takes over, so never further out
# than 1.5.
@pytest.mark.parametrize("root", [1.0, -1.0, 1.05, -1.5])
def test_deflation_reproduces_the_series_it_divided(root):
    """The synthetic division the two conversions above are built out of.

    What it costs is the quotient's OWN size: the division is exact in exact
    arithmetic, so the deviation is the rounding of a difference of terms that
    reach ``max|quotient|`` -- which is why the bound below is written against
    that rather than as a constant.
    """
    series = SERIES["deep"]
    quotient, value = deflate(series, root)
    t = np.cos(2.0 * ANGLE)
    rebuilt = (t - root) * harmonic_value(quotient, ANGLE) + value
    deviation = np.max(np.abs(rebuilt - harmonic_value(series, ANGLE)))
    assert deviation <= 1e-14 * max(np.max(np.abs(quotient)), 1.0)


def test_the_deflation_is_why_a_far_root_takes_the_other_route():
    """Measured in both directions, at the switch and four decades past it.

    A denominator's root sits at ``+/-(1 + 2 shift)`` and the reduction leaves the
    deflation for the pole family's own moments at a shift of a quarter.  The
    quotient's coefficients grow like the root's powers, and the reconstruction
    loses that factor: at the switch it is a few hundred ulp of the series, and by
    a root of seven it has thrown away eight decades.  Both bounds are asserted so
    a change that moved the switch would have to move this test too.
    """
    series = SERIES["deep"]
    t = np.cos(2.0 * ANGLE)

    def deviation(root):
        quotient, value = deflate(series, root)
        rebuilt = (t - root) * harmonic_value(quotient, ANGLE) + value
        return np.max(np.abs(rebuilt - harmonic_value(series, ANGLE)))

    assert 1e-14 <= deviation(-1.5) <= 1e-12
    assert 1e-10 <= deviation(-7.0) <= 1e-7


def test_the_algebra_keeps_end_values_out_of_the_series():
    """product, total and scaled form their ends from the factors' own."""
    left = range_function([0.5, -0.25], 1e-9, 3.0)
    right = range_function([2.0], -4e-10, 0.5)
    assert product(left, right)[1] == left[1] * right[1]
    assert product(left, right)[2] == left[2] * right[2]
    assert total(left, right)[1] == left[1] + right[1]
    assert scaled(left, 3.0)[2] == 3.0 * left[2]
    assert sine_squared_times([1.0, 2.0])[1] == 0.0
    assert sine_squared_times([1.0, 2.0])[2] == 0.0


def test_a_product_of_bounded_factors_keeps_bounded_coefficients():
    """The reason the bulk is harmonic: a monomial product does not do this."""
    left = [1.0, -0.5, 0.25, 0.125]
    right = [0.75, 0.5, -0.25]
    out = harmonic_multiply(left, right)
    bound = sum(np.abs(left)) * sum(np.abs(right))
    assert max(np.abs(out)) <= bound
    assert np.allclose(
        harmonic_value(out, ANGLE),
        harmonic_value(left, ANGLE) * harmonic_value(right, ANGLE),
        atol=1e-14,
    )


def test_the_series_helpers_add_and_scale_as_functions():
    """harmonic_add and harmonic_scale, checked the same way."""
    left, right = SERIES["deep"], SERIES["quadratic"]
    assert np.allclose(
        harmonic_value(harmonic_add(left, right), ANGLE),
        harmonic_value(left, ANGLE) + harmonic_value(right, ANGLE),
        atol=1e-14,
    )
    assert np.allclose(
        harmonic_value(harmonic_scale(left, -2.5), ANGLE),
        -2.5 * harmonic_value(left, ANGLE),
        atol=1e-14,
    )
