"""Contract for the quarter-period elliptic moments the Urankar reduction needs.

Every one of these is a closed form or a recursion standing in for an integral,
and every one of them is transcribed from a 1990 paper. A mis-transcribed
recursion is invisible once it is buried in a hundred-term field expression and
obvious against the integral it claims to equal, so each is pinned here against
direct numerical quadrature of its own definition, over the parameter range a
toroidal ring actually produces.

The quadrature references are deliberately naive: adaptive integration of the
literal definition, with no shared machinery with the thing under test.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy.integrate
import scipy.special

from nova.biot.elliptic import (
    complete_pi,
    pole_moment,
    sn_cn_moments,
    sn_moments,
)

# k^2 = 4 r r' / ((r + r')^2 + gamma^2) for a ring: bounded by 1, approaching it
# for a target on the source radius in the plane, and falling to zero far away.
PARAMETERS = [1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]

DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "286208998628034825342117067982148086513282306647093844609550582231725359"
    "408128481117450284102701938521105559644622948954930381964428810975665933"
    "446128475648233786783165271201909145648566923460348610454326648213393607"
    "260249141273724587006606315588174881520920962829254091715364367892590360"
    "011330530548820466521384146951941511609433057270365759591953092186117381"
)


def jacobi(u, parameter):
    """Return ``(sn, cn, dn)`` at ``u`` for parameter ``m = k^2``."""
    sn, cn, dn, _ = scipy.special.ellipj(u, parameter)
    return sn, cn, dn


def quarter_period(parameter):
    """Return ``K(k)``, the quarter period."""
    return float(scipy.special.ellipk(parameter))


def quadrature(integrand, parameter):
    """Return the integral of ``integrand(u)`` over one quarter period."""
    value, error = scipy.integrate.quad(
        integrand, 0.0, quarter_period(parameter), limit=200, epsabs=1e-13, epsrel=1e-13
    )
    assert error < 1e-9 * max(abs(value), 1e-12) + 1e-13
    return value


def decimal_sn_moment(parameter, order):
    """Return an even sn moment from its 120-digit hypergeometric series."""
    with localcontext() as context:
        context.prec = 120
        parameter = Decimal.from_float(float(parameter))
        half = Decimal(1) / 2
        rising = Decimal(1)
        factorial = Decimal(1)
        for index in range(1, order + 1):
            rising *= Decimal(index) - half
            factorial *= Decimal(index)

        first = half
        second = Decimal(order) + half
        third = Decimal(order + 1)
        term = total = Decimal(1)
        for index in range(1, 1000):
            step = Decimal(index - 1)
            term *= (
                (first + step)
                * (second + step)
                * parameter
                / ((third + step) * Decimal(index))
            )
            total += term
            if abs(term) < Decimal("1e-115"):
                break
        return float((DECIMAL_PI / 2) * rising / factorial * total)


def decimal_pole_moment(characteristic, parameter):
    """Return the defining pole integral from a 120-digit double series."""
    with localcontext() as context:
        context.prec = 120
        characteristic = Decimal.from_float(float(characteristic))
        parameter = Decimal.from_float(float(parameter))
        total = Decimal(0)
        characteristic_power = Decimal(1)
        for first_order in range(40):
            central = Decimal(1)
            parameter_power = Decimal(1)
            for second_order in range(40):
                total += (
                    characteristic_power
                    * central
                    * parameter_power
                    / Decimal(first_order + second_order + 1)
                )
                next_order = second_order + 1
                central *= Decimal(2 * next_order - 1) / Decimal(2 * next_order)
                parameter_power *= parameter
            characteristic_power *= characteristic
        return float(characteristic * total / 2)


def decimal_arctangent(value):
    """Return an arctangent from a 300-digit argument-reduced power series."""
    value = Decimal(value)
    if value < 0:
        return -decimal_arctangent(-value)
    if value > 1:
        return DECIMAL_PI / 2 - decimal_arctangent(1 / value)
    if value > Decimal("0.5"):
        return DECIMAL_PI / 4 + decimal_arctangent((value - 1) / (value + 1))
    square = value * value
    term = total = value
    for order in range(1, 2000):
        term = -term * square
        increment = term / Decimal(2 * order + 1)
        total += increment
        if abs(increment) < Decimal("1e-310"):
            return total
    raise AssertionError("the decimal arctangent series did not converge")


def decimal_complement_pole_moment(complement, parameter_complement):
    """Return the closed quarter-period value with 300-digit complement arithmetic."""
    with localcontext() as context:
        context.prec = 320
        pole_complement = Decimal.from_float(float(complement))
        modulus_complement = Decimal.from_float(float(parameter_complement))
        characteristic = Decimal(1) - pole_complement
        complementary = modulus_complement.sqrt()
        denominator = pole_complement + complementary
        difference = pole_complement - modulus_complement
        if difference == 0:
            return float(characteristic / denominator)
        root = (characteristic * abs(difference)).sqrt()
        if difference > 0:
            angle = decimal_arctangent(root / denominator)
        else:
            angle = (
                (denominator + root) / (pole_complement.sqrt() * (1 + complementary))
            ).ln()
        return float(characteristic / root * angle)


@pytest.mark.parametrize("parameter", PARAMETERS)
def test_sn_moments_reproduce_their_defining_integral(parameter):
    """El(2m) is the even sn moment over a quarter period."""
    moments = sn_moments(np.float64(parameter), 5)
    for order, moment in enumerate(moments):
        expected = quadrature(
            lambda u, order=order: jacobi(u, parameter)[0] ** (2 * order), parameter
        )
        assert abs(moment - expected) <= 2e-12 * max(abs(expected), 1.0), (
            f"order {order}, parameter {parameter}: {moment} vs {expected}"
        )


def test_sn_moments_keep_the_small_parameter_and_zero_limits():
    """The canonical recursion stays accurate where upward division cannot."""
    assert sn_moments(np.float64(0.0), 0) == []
    for parameter in (0.0, 1e-12, 1e-6):
        moments = sn_moments(np.float64(parameter), 5)
        for order, moment in enumerate(moments):
            expected = decimal_sn_moment(parameter, order)
            assert float(moment) == pytest.approx(expected, rel=3e-15)


@pytest.mark.parametrize("parameter", PARAMETERS)
def test_sn_cn_moments_reproduce_their_defining_integral(parameter):
    """I(2m+1) is the odd Jacobi moment, and stays accurate at small parameter."""
    moments = sn_cn_moments(np.float64(parameter), 4)
    for order, moment in enumerate(moments):
        expected = quadrature(
            lambda u, order=order: (
                (parameter**order)
                * jacobi(u, parameter)[0] ** (2 * order)
                * parameter
                * jacobi(u, parameter)[0]
                * jacobi(u, parameter)[1]
            ),
            parameter,
        )
        np.testing.assert_allclose(moment, expected, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("parameter", [1e-12, 1e-9, 1e-6])
def test_sn_cn_moments_stay_accurate_where_the_sn_moments_do_not(parameter):
    """The far field is the small-parameter case, so this family must survive it.

    Against Gauss-Legendre in ``v`` rather than quadrature of the Jacobi form:
    at these parameters the moments are down at 1e-13 and below, where an
    adaptive quadrature of the Jacobi integrand has no significant digits left
    either. The reference has to form the interval width ``1 - k'`` the stable
    way too -- writing it literally reintroduces exactly the cancellation the
    implementation exists to avoid, and the reference, not the implementation,
    is then the inaccurate one.
    """
    moments = sn_cn_moments(np.float64(parameter), 4)
    complementary = np.sqrt(1.0 - parameter)
    # substitute w = 1 - v: the interval becomes [0, 1 - k'] and the integrand
    # (1 - v^2)^m becomes (w (2 - w))^m, so nothing in the reference is a
    # difference of nearly equal numbers
    width = parameter / (1.0 + complementary)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    half = 0.5 * width
    w = half * (nodes + 1.0)
    for order, moment in enumerate(moments):
        expected = half * np.sum(weights * (w * (2.0 - w)) ** order)
        np.testing.assert_allclose(moment, expected, rtol=1e-12, atol=0.0)


def test_sn_cn_moments_match_the_printed_recursion_where_it_is_conditioned():
    """The stable route returns the paper's quantity, not a different one."""
    for parameter in (0.2, 0.5, 0.8):
        complementary = np.sqrt(1.0 - parameter)
        printed = [1.0 - complementary]
        for order in range(1, 4):
            printed.append(
                (-(parameter**order) * complementary + 2.0 * order * printed[order - 1])
                / (2 * order + 1)
            )
        np.testing.assert_allclose(
            sn_cn_moments(np.float64(parameter), 4), printed, rtol=1e-12
        )


@pytest.mark.parametrize("parameter", PARAMETERS)
@pytest.mark.parametrize("characteristic", [1e-4, 0.05, 0.25, 0.6, 0.95])
def test_pole_moment_reproduces_its_defining_integral(characteristic, parameter):
    """I(eta^2) in closed form, across all three branches of the substitution."""
    expected = quadrature(
        lambda u: (
            characteristic
            * jacobi(u, parameter)[0]
            * jacobi(u, parameter)[1]
            / (1.0 - characteristic * jacobi(u, parameter)[0] ** 2)
        ),
        parameter,
    )
    got = pole_moment(np.float64(characteristic), np.float64(parameter))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)


def test_pole_moment_handles_the_degenerate_branch():
    """eta^2 == k^2 collapses the quadratic; the closed form must not divide by zero."""
    parameter = 0.4
    expected = quadrature(
        lambda u: (
            parameter
            * jacobi(u, parameter)[0]
            * jacobi(u, parameter)[1]
            / (1.0 - parameter * jacobi(u, parameter)[0] ** 2)
        ),
        parameter,
    )
    got = pole_moment(np.float64(parameter), np.float64(parameter))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)


def test_pole_moment_keeps_the_value_immediately_below_equality():
    """A one-ulp parameter gap is resolved against the defining 120-digit series."""
    characteristic = np.float64(1e-12)
    parameter = np.nextafter(characteristic, np.float64(np.inf))
    expected = decimal_pole_moment(characteristic, parameter)
    got = pole_moment(characteristic, parameter)
    assert float(got) == pytest.approx(expected, rel=3e-15)


def test_pole_moment_keeps_exact_complements_and_elementary_limits():
    """Exact complements, broadcasting, zero, and the singular endpoint are held."""
    characteristics = np.array([0.0, 0.2, 1.0])
    with np.errstate(all="raise"):
        got = pole_moment(characteristics, np.float32(0.0))
    assert got.shape == characteristics.shape
    assert got.dtype == np.float64
    assert got[0] == 0.0
    assert got[1] == pytest.approx(-0.5 * np.log1p(-0.2), rel=2e-15)
    assert np.isposinf(got[2])

    complement = np.float64(1e-20)
    exact = pole_moment(
        np.float64(1.0),
        np.float64(0.0),
        complement=complement,
        parameter_complement=np.float64(1.0),
    )
    assert float(exact) == pytest.approx(-0.5 * np.log(complement), rel=2e-15)


def test_pole_moment_keeps_the_full_positive_complement_range_at_unit_parameter():
    """The finite endpoint is scaled before any reciprocal can overflow."""
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    complements = np.array([smallest, 1e-320, 1e-300, 1e-250, 1e-200])
    expected = np.array(
        [decimal_complement_pole_moment(term, 0.0) for term in complements]
    )
    with np.errstate(all="raise"):
        got = pole_moment(
            np.ones(complements.shape, dtype=np.float32),
            np.float32(1.0),
            complement=complements,
            parameter_complement=np.float32(0.0),
        )
    assert got.shape == complements.shape
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=4e-16, atol=0.0)


def test_pole_moment_branches_on_exact_neighbouring_complements():
    """Rounded principal values do not erase either side of complement equality."""
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    poles = np.array([smallest, 1e-300, 1e-250, 1e-200])
    pole_grid = np.repeat(poles, 3)
    modulus_grid = np.array(
        [
            neighbour
            for pole in poles
            for neighbour in (
                np.nextafter(pole, 0.0),
                pole,
                np.nextafter(pole, np.inf),
            )
        ]
    )
    expected = np.array(
        [
            decimal_complement_pole_moment(pole, modulus)
            for pole, modulus in zip(pole_grid, modulus_grid)
        ]
    )
    with np.errstate(all="raise"):
        got = pole_moment(
            np.ones_like(pole_grid),
            np.ones_like(modulus_grid),
            complement=pole_grid,
            parameter_complement=modulus_grid,
        )
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=0.0)


def test_pole_moment_extreme_lanes_are_held_before_eager_evaluation():
    """One broadcast covers zero, both finite branches, equality, and true infinity."""
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    poles = np.array([1.0, smallest, smallest, smallest, smallest, smallest, 0.0])
    moduli = np.array([1.0, 0.0, smallest, 2.0 * smallest, 0.25, 1.0, 0.25])
    characteristic = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    parameter = 1.0 - moduli
    with np.errstate(all="raise"):
        got = pole_moment(
            characteristic,
            parameter,
            complement=poles,
            parameter_complement=moduli,
        )
    assert got[0] == 0.0
    assert np.all(np.isfinite(got[1:-1]))
    assert np.isposinf(got[-1])

    references = np.array(
        [
            float(
                "7.0668772630353430919108272455989351906971060222382893819705684e161"
            ),
            float(
                "4.4989137945431963828105385076859818588696971183019166331007556e161"
            ),
            float(
                "3.9652237887882404081692413173009657591670080061570872317507195e161"
            ),
            float("7.4362914170516493355015127221515293583994316345598915453038230e2"),
            float("3.7222003596069063115705364922304081705654357215145707146280517e2"),
        ]
    )
    np.testing.assert_allclose(got[1:-1], references, rtol=4e-16, atol=0.0)


def test_pole_moment_retains_one_supplied_complement_and_small_equality():
    """Either paired complement and the principal characteristic carry information."""
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    with np.errstate(all="raise"):
        one_complement = pole_moment(1.0, 1.0, complement=smallest)
        explicit_pair = pole_moment(
            1.0,
            1.0,
            complement=smallest,
            parameter_complement=0.0,
        )
        small_equal = pole_moment(1e-20, 1e-20)
    assert float(one_complement) == float(explicit_pair)
    assert float(small_equal) == pytest.approx(5e-21, rel=2e-15)


@pytest.mark.parametrize("parameter", [0.05, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("characteristic", [-2.0, -0.3, 1e-3, 0.4, 0.8])
def test_complete_pi_reproduces_its_defining_integral(characteristic, parameter):
    """Pi(n | m) by Carlson forms against the Legendre integral."""
    expected, _ = scipy.integrate.quad(
        lambda theta: (
            1.0
            / (
                (1.0 - characteristic * np.sin(theta) ** 2)
                * np.sqrt(1.0 - parameter * np.sin(theta) ** 2)
            )
        ),
        0.0,
        np.pi / 2.0,
        limit=200,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    got = complete_pi(np.float64(characteristic), np.float64(parameter))
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-14)


def test_complete_pi_reduces_to_the_first_kind_at_zero_characteristic():
    """Pi(0 | m) == K(m) -- the cheapest possible transcription check."""
    parameter = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        complete_pi(np.zeros_like(parameter), parameter),
        scipy.special.ellipk(parameter),
        rtol=1e-13,
    )


def test_the_quarter_period_jacobi_values_the_reduction_relies_on():
    """sn K = 1, cn K = 0, dn K = k'.

    ``cn K = 0`` is load-bearing: it is what annihilates the radial component of
    the vector potential for a full turn, so the reduction reproduces
    axisymmetry rather than assuming it.
    """
    for parameter in (0.05, 0.5, 0.95):
        sn, cn, dn = jacobi(quarter_period(parameter), parameter)
        np.testing.assert_allclose(sn, 1.0, atol=1e-12)
        np.testing.assert_allclose(cn, 0.0, atol=1e-12)
        np.testing.assert_allclose(dn, np.sqrt(1.0 - parameter), atol=1e-12)


# --------------------------------------------------------------------------
# The moment families above are what the printed recursions give.  The three
# tests below pin the routes the polygon closed form actually evaluates, which
# differ in the direction they run and in what they are handed: an explicit
# complement, so a characteristic that sits a few 1e-9 from unity keeps its
# relative accuracy instead of inheriting the round-off of ``1 - n``.


@pytest.mark.parametrize("parameter", PARAMETERS + [1e-6, 1 - 1e-6])
def test_the_canonical_moments_agree_with_their_defining_integral_at_every_order(
    parameter,
):
    """The public route holds across both conditioning directions.

    The downward direction carries small parameters and the upward direction
    carries the near-unit limit; this test spans both sides of their switch.
    """
    moments = sn_moments(np.float64(parameter), 9)
    for order, moment in enumerate(moments):
        expected = quadrature(
            lambda u, order=order: jacobi(u, parameter)[0] ** (2 * order), parameter
        )
        np.testing.assert_allclose(moment, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("parameter", [0.05, 0.5, 0.95])
@pytest.mark.parametrize("characteristic", [-2.0e5, -0.3, 1e-3, 0.4, 0.95])
def test_pole_moments_reproduce_their_defining_integral(characteristic, parameter):
    """V_m, the moments the polygon edge denominators reduce to.

    ``V_m = int_0^{pi/2} sin^{2m} a / ((1 - n sin^2 a) sqrt(1 - k^2 sin^2 a)) da``.
    Both directions again: upward from ``Pi(n | k^2)`` divides by ``n`` per order
    and downward multiplies by it, so the switch is on ``|n|``, and the
    characteristics here straddle it.
    """
    from nova.biot.elliptic import pole_moments

    moments = pole_moments(np.float64(characteristic), np.float64(parameter), 9)
    for order, moment in enumerate(moments):
        expected, _ = scipy.integrate.quad(
            lambda angle, order=order: (
                np.sin(angle) ** (2 * order)
                / (
                    (1.0 - characteristic * np.sin(angle) ** 2)
                    * np.sqrt(1.0 - parameter * np.sin(angle) ** 2)
                )
            ),
            0.0,
            np.pi / 2.0,
            limit=200,
            epsabs=1e-14,
            epsrel=1e-14,
        )
        np.testing.assert_allclose(moment, expected, rtol=1e-11, atol=1e-15)


def test_complete_pi_keeps_its_accuracy_when_handed_the_complement():
    """A characteristic within round-off of unity is only usable via ``1 - n``.

    ``Pi(n | m)`` grows like ``(1 - n)^(-1/2)``, so forming the complement inside
    the routine caps its relative accuracy at ``eps / (1 - n)`` -- eight digits
    gone by ``1 - n = 1e-8``, which is what a target level with a polygon vertex
    produces.  Passing the complement, which the caller knows in closed form,
    costs nothing and keeps every digit.
    """
    parameter = np.float64(0.5)
    for complement in (1e-4, 1e-8, 1e-12):
        expected = third_kind_quadrature(1.0 - complement, complement, parameter)
        got = complete_pi(
            np.float64(1.0 - complement), parameter, complement=np.float64(complement)
        )
        np.testing.assert_allclose(got, expected, rtol=1e-11)


# --------------------------------------------------------------------------
# The complement basis.  Every moment above is a moment of ``sin^2 a``, which
# vanishes at ``a = 0``.  A pole sitting just past the OTHER end of the range
# needs the mirror family -- moments of ``cos^2 a`` -- so that a numerator's
# value AT that end is a coefficient of its expansion instead of an alternating
# sum of coefficients.  The polygon reduction has such a pole in both of its
# denominators, and their distance from the range end falls as the square of the
# section's aspect ratio, so the two families below are what its accuracy at a
# slender section rests on.


def graded_quadrature(integrand, floor, nodes=64):
    """Return the integral over ``theta`` in ``[0, pi/2]`` on graded panels.

    ``theta = pi/2 - a``, so the range end where the complement basis vanishes is
    at the origin -- and that is where the integrands below carry their narrow
    features: the pole's distance from the end, and the modulus complement, which
    are independent scales. An adaptive rule can step over either without
    noticing and report a small error estimate for a wrong answer. Panels graded
    geometrically from ``floor`` resolve both by construction: each spans one
    octave, on which the integrand is smooth, so a fixed high-order rule on every
    panel is spectrally accurate and no feature can hide between nodes.
    """
    node, weight = np.polynomial.legendre.leggauss(nodes)
    edges = [0.0]
    while edges[-1] < 0.5 * np.pi:
        edges.append(floor if edges[-1] == 0.0 else min(2.0 * edges[-1], 0.5 * np.pi))
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        half = 0.5 * (upper - lower)
        total += half * weight @ integrand(0.5 * (lower + upper) + half * node)
    return total


def third_kind_quadrature(
    characteristic, complement, parameter, parameter_complement=None, nodes=64
):
    """Return ``Pi(n | m)`` by quadrature that resolves both narrow scales.

    ``Pi = int_0^{pi/2} dtheta / ((1 - n + n sin^2 theta) sqrt(k'^2 + k^2 sin^2
    theta))`` in ``theta = pi/2 - a``: the characteristic's complement sets one
    scale at the origin and the parameter's the other, and a stretch fitted to
    either steps over the other. Both complements are taken as arguments because
    a float parameter cannot express its own complement below ``eps``, which is
    exactly the accuracy the routine under test claims to keep.
    """
    if parameter_complement is None:
        parameter_complement = 1.0 - parameter
    scales = [np.sqrt(complement / abs(characteristic))] if characteristic else [1.0]
    scales.append(np.sqrt(parameter_complement / max(parameter, 1e-300)))
    return graded_quadrature(
        lambda theta: (
            1.0
            / (
                (complement + characteristic * np.sin(theta) ** 2)
                * np.sqrt(parameter_complement + parameter * np.sin(theta) ** 2)
            )
        ),
        floor=0.01 * min(1.0, *scales),
        nodes=nodes,
    )


def complement_moment_quadrature(parameter, order, nodes=64):
    """Return ``A_m`` by quadrature of its own definition.

    ``A_m = int_0^{pi/2} cos^{2m} a / sqrt(1 - k^2 sin^2 a) da``, written in
    ``theta = pi/2 - a`` so the modulus complement's scale sits at the origin
    where the grading is.
    """
    complementary = 1.0 - parameter
    return graded_quadrature(
        lambda theta: (
            np.sin(theta) ** (2 * order)
            / np.sqrt(complementary + parameter * np.sin(theta) ** 2)
        ),
        floor=0.01 * min(1.0, np.sqrt(complementary / max(parameter, 1e-300))),
        nodes=nodes,
    )


def complement_pole_quadrature(shift, parameter, order, nodes=64):
    """Return ``T_m`` in the complement basis by quadrature of its definition.

    ``T_m = int_0^{pi/2} cos^{2m} a / ((cos^2 a + shift) sqrt(1 - k^2 sin^2 a)) da``.
    """
    complementary = 1.0 - parameter
    scale = min(
        np.sqrt(shift) if shift > 0.0 else 1.0,
        np.sqrt(complementary / max(parameter, 1e-300)),
    )
    return graded_quadrature(
        lambda theta: (
            np.sin(theta) ** (2 * order)
            / (
                (np.sin(theta) ** 2 + shift)
                * np.sqrt(complementary + parameter * np.sin(theta) ** 2)
            )
        ),
        floor=0.01 * min(1.0, scale),
        nodes=nodes,
    )


def plain_pole_quadrature(shift, parameter, order, nodes=64):
    """Return ``T_m`` in the plain basis by quadrature of its definition.

    ``T_m = int_0^{pi/2} sin^{2m} a / ((sin^2 a + shift) sqrt(1 - k^2 sin^2 a)) da``.
    Here the pole's scale sits at ``a = 0`` and the modulus complement's at
    ``a = pi/2``, so the grading runs from each end in turn.
    """
    complementary = 1.0 - parameter

    def integrand(angle):
        return np.sin(angle) ** (2 * order) / (
            (np.sin(angle) ** 2 + shift)
            * np.sqrt(complementary + parameter * np.cos(angle) ** 2)
        )

    # a = theta near the pole end, a = pi/2 - theta near the modulus end
    near = graded_quadrature(
        lambda theta: 0.5 * integrand(0.5 * theta),
        floor=0.01 * min(1.0, np.sqrt(shift) if shift > 0.0 else 1.0),
        nodes=nodes,
    )
    far = graded_quadrature(
        lambda theta: 0.5 * integrand(0.5 * (np.pi - theta)),
        floor=0.01 * min(1.0, np.sqrt(complementary / max(parameter, 1e-300))),
        nodes=nodes,
    )
    return near + far


def test_the_graded_reference_agrees_with_the_routines_already_pinned():
    """The new reference is checked against ones the suite already trusts.

    ``A_0`` is ``K`` and the plain shifted moment at ``m = 0`` is a third-kind
    integral, both of which this module pins independently above. Without this the
    tolerances below would be measuring the reference rather than the recursions.
    """
    for parameter in (1e-6, 0.3, 0.7, 1.0 - 1e-8):
        np.testing.assert_allclose(
            complement_moment_quadrature(parameter, 0),
            scipy.special.ellipk(parameter),
            rtol=1e-12,
        )
    for shift in (1e-8, 1e-3, 0.5):
        # sin^2 a + shift = (1 + shift)(1 - n sin^2 a) with n = 1/(1 + shift)
        characteristic = -1.0 / shift
        np.testing.assert_allclose(
            plain_pole_quadrature(shift, 0.7, 0),
            complete_pi(
                np.float64(characteristic),
                np.float64(0.7),
                complement=np.float64(1.0 + 1.0 / shift),
            )
            / shift,
            rtol=1e-11,
        )


# The recursion for the complement moments has branches 1 and -k'^2/k^2, so its
# upward direction holds where k^2 dominates and its downward one where the
# complement does. These straddle the switch from both sides.
COMPLEMENT_PARAMETERS = [1e-12, 1e-6, 0.1, 0.3, 0.33, 0.34, 0.5, 0.9, 1.0 - 1e-12]


@pytest.mark.parametrize("parameter", COMPLEMENT_PARAMETERS)
def test_the_complement_moments_agree_with_their_defining_integral(parameter):
    """A_m at every order the reduction needs, in both recursion directions."""
    from nova.biot.elliptic import stable_cn_moments

    moments = stable_cn_moments(
        np.float64(parameter), 9, complement=np.float64(1.0 - parameter)
    )
    for order, moment in enumerate(moments):
        expected = complement_moment_quadrature(parameter, order)
        np.testing.assert_allclose(moment, expected, rtol=2e-12, atol=1e-14)


def test_the_complement_moments_stay_bounded_by_the_quarter_period():
    """``cn <= 1`` makes every moment a decreasing sequence bounded by ``A0 = K``.

    A recursion that has drifted onto its parasitic branch breaks monotonicity
    long before it breaks the tolerance above, so this is the cheap sentinel.
    """
    from nova.biot.elliptic import stable_cn_moments

    for parameter in COMPLEMENT_PARAMETERS:
        moments = stable_cn_moments(np.float64(parameter), 12)
        assert moments[0] == pytest.approx(scipy.special.ellipk(parameter))
        for lower, upper in zip(moments[1:], moments[:-1]):
            assert 0.0 < lower <= upper * (1.0 + 1e-12), parameter


SHIFTS = [1e-10, 1e-6, 1e-3, 0.1, 1.0, 1.9, 2.1, 30.0]


@pytest.mark.parametrize("parameter", [1e-6, 0.3, 0.7, 1.0 - 1e-8])
@pytest.mark.parametrize("shift", SHIFTS)
def test_the_complement_pole_moments_agree_with_their_defining_integral(
    shift, parameter
):
    """T_m for a pole beyond the ``a = pi/2`` end, both recursion directions."""
    from nova.biot.elliptic import cn_pole_moments

    moments = cn_pole_moments(np.float64(shift), np.float64(parameter), 9)
    for order, moment in enumerate(moments):
        expected = complement_pole_quadrature(shift, parameter, order)
        np.testing.assert_allclose(moment, expected, rtol=2e-11, atol=1e-14)


@pytest.mark.parametrize("parameter", [1e-6, 0.3, 0.7, 1.0 - 1e-8])
@pytest.mark.parametrize("shift", SHIFTS)
def test_the_plain_pole_moments_agree_with_their_defining_integral(shift, parameter):
    """The same family for a pole beyond the ``a = 0`` end, in the plain basis."""
    from nova.biot.elliptic import sn_pole_moments

    moments = sn_pole_moments(np.float64(shift), np.float64(parameter), 9)
    for order, moment in enumerate(moments):
        expected = plain_pole_quadrature(shift, parameter, order)
        np.testing.assert_allclose(moment, expected, rtol=2e-11, atol=1e-14)


@pytest.mark.parametrize("parameter", [0.2, 0.8])
def test_a_pole_sitting_exactly_on_the_range_end_returns_the_plain_moments(parameter):
    """``shift = 0`` puts the pole ON the end, where the leading moment diverges.

    A numerator can only reach that configuration with its own leading
    coefficient exactly zero -- the pole factor and the numerator share the
    geometric quantity that vanishes -- so the divergent moment is multiplied by
    a hard zero and the useful convention is to return zero for it rather than an
    infinity that would poison the product. Every higher moment is finite and
    equals the plain moment one order down, which is what is asserted here.
    """
    from nova.biot.elliptic import cn_pole_moments, sn_pole_moments, stable_cn_moments

    plain = stable_cn_moments(np.float64(parameter), 9)
    complement = cn_pole_moments(np.float64(0.0), np.float64(parameter), 9)
    assert complement[0] == 0.0
    np.testing.assert_allclose(complement[1:], plain[:8], rtol=1e-13)

    plain = sn_moments(np.float64(parameter), 9)
    shifted = sn_pole_moments(np.float64(0.0), np.float64(parameter), 9)
    assert shifted[0] == 0.0
    np.testing.assert_allclose(shifted[1:], plain[:8], rtol=1e-13)


def test_the_third_kind_integral_keeps_its_digits_when_handed_both_complements():
    """``Pi`` is as sensitive to the parameter's complement as to the
    characteristic's.

    Forming ``1 - m`` from a float parameter caps the relative accuracy at
    ``eps / (1 - m)``, because the float cannot carry its own complement any
    finer -- and the polygon reduction drives ``1 - m`` to the square of the
    section aspect ratio, so at the aspect this exists for several digits are
    already gone before the routine starts. The caller knows the complement in
    closed form, from the target's offset from the source ring.
    """
    # the parameter is what a slender section produces: a rounded 1 - 1e-13, whose
    # complement is therefore known to only nine digits from the float alone
    parameter_complement = np.float64(1e-13)
    parameter = np.float64(1.0) - parameter_complement
    complement = np.float64(1e-8)
    characteristic = np.float64(1.0) - complement
    loose = complete_pi(characteristic, parameter, complement=complement)
    tight = complete_pi(
        characteristic,
        parameter,
        complement=complement,
        parameter_complement=parameter_complement,
    )
    exact = third_kind_quadrature(
        characteristic, complement, parameter, parameter_complement
    )
    np.testing.assert_allclose(tight, exact, rtol=1e-10)
    # and the difference is real rather than round-off: the rounded parameter puts
    # the complement out by a part in 1e4, which Pi carries as half of that
    assert abs(loose - exact) > 1e-6 * abs(exact)


# --------------------------------------------------------------------------
# The HARMONIC families. A numerator written in powers of a range variable reaches
# coefficients four decades above its own size by degree six -- that is what the
# monomial basis costs on a unit interval -- so contracting one against the power
# families above forms the answer out of terms that exceed it by as much. Cosines
# are bounded by one and their moments decay, so the same contraction adds nothing
# larger than the result. These three are what makes that possible: the plain
# family, the same weighted by the radical, and one per pole factor.
#
# What is wanted from them is ABSOLUTE accuracy of order ``eps K``, not relative:
# the high harmonics are small and the coefficients multiplying them are no larger
# than the low ones, so an error tiny beside ``P_0`` is tiny beside the contraction
# however large it is beside ``P_n`` itself. The tolerances below are written that
# way, against ``K`` rather than against each moment.

# Adaptive quadrature stops being a usable reference past this: at a complement of
# 1e-9 the integrand is 3e4 over a width of 3e-5 radians and ``quad`` reports its own
# error at 6e-09, an order above what is being measured. The confluence is pinned
# below instead, against values that are elementary there.
HARMONIC_PARAMETERS = [0.0, 1e-3, 0.3, 0.7, 0.95, 0.985, 0.995]


def harmonic_quadrature(order, parameter, weight):
    """Return ``int_0^(pi/2) cos(2 n a) weight(Delta) da`` by adaptive quadrature."""
    value, _ = scipy.integrate.quad(
        lambda angle: (
            np.cos(2 * order * angle)
            * weight(np.sqrt(1.0 - parameter * np.sin(angle) ** 2))
        ),
        0.0,
        np.pi / 2.0,
        limit=400,
        epsabs=1e-15,
        epsrel=1e-15,
    )
    return value


@pytest.mark.parametrize("parameter", HARMONIC_PARAMETERS)
def test_harmonic_moments_reproduce_their_defining_integral(parameter):
    """``P_n = int cos(2 n a)/Delta da``, over and under the recursion's switch.

    The family is the MINIMAL solution of its own three-term relation, so its
    downward direction is the stable one everywhere except at the confluence,
    where the two branches become degenerate; there it is run upward instead, on
    the difference from the divergence every order shares. Both sides are swept.
    """
    from nova.biot.elliptic import harmonic_moments

    complement = np.float64(1.0 - parameter)
    moments = harmonic_moments(np.float64(parameter), 9, complement=complement)
    scale = float(scipy.special.ellipk(parameter))
    for order, moment in enumerate(moments):
        expected = harmonic_quadrature(order, parameter, lambda delta: 1.0 / delta)
        assert abs(float(moment) - expected) < 1e-13 * scale


@pytest.mark.parametrize("parameter", HARMONIC_PARAMETERS)
def test_harmonic_root_moments_reproduce_their_defining_integral(parameter):
    """``D_n = int cos(2 n a) Delta da``, folded off the plain family.

    ``Delta^2`` is itself two harmonics, so no new special function is needed --
    which is the only reason the radical-weighted term costs nothing.
    """
    from nova.biot.elliptic import harmonic_moments, harmonic_root_moments

    complement = np.float64(1.0 - parameter)
    moments = harmonic_moments(np.float64(parameter), 10, complement=complement)
    root = harmonic_root_moments(moments, np.float64(parameter))
    scale = float(scipy.special.ellipk(parameter))
    for order, moment in enumerate(root):
        expected = harmonic_quadrature(order, parameter, lambda delta: delta)
        assert abs(float(moment) - expected) < 1e-13 * scale


@pytest.mark.parametrize("shift", [0.3, 1.0, 7.0, 1.0e3, 1.0e7])
@pytest.mark.parametrize("parameter", [0.05, 0.5, 0.95, 0.995])
@pytest.mark.parametrize("mirrored", [False, True])
def test_harmonic_pole_moments_reproduce_their_defining_integral(
    shift, parameter, mirrored
):
    """One pole factor, a root past either end, over seven decades of distance.

    The far shifts are the point: a nearly vertical polygon edge puts the root at
    the reciprocal of its squared slope, and the family's own decay is what has to
    separate its orders there. Near roots are excluded because a caller takes the
    weight on those out exactly instead -- the moments are then dominated by their
    seed and this contraction is not the route used.
    """
    from nova.biot.elliptic import (
        cn_pole_moment,
        harmonic_moments,
        harmonic_pole_moments,
        sn_pole_moment,
    )
    from nova.biot.elliptic import POLE_HEADROOM

    complement = np.float64(1.0 - parameter)
    count = 9
    moments = harmonic_moments(
        np.float64(parameter), count + POLE_HEADROOM + 1, complement=complement
    )
    seed = (sn_pole_moment if mirrored else cn_pole_moment)(
        np.float64(shift), np.float64(parameter), parameter_complement=complement
    )
    family = harmonic_pole_moments(
        np.float64(shift), seed, moments, count, mirrored=mirrored
    )
    scale = float(scipy.special.ellipk(parameter)) / (1.0 + shift)
    for order, moment in enumerate(family):
        expected, _ = scipy.integrate.quad(
            lambda angle, order=order: (
                np.cos(2 * order * angle)
                / (
                    ((np.sin(angle) if mirrored else np.cos(angle)) ** 2 + shift)
                    * np.sqrt(1.0 - parameter * np.sin(angle) ** 2)
                )
            ),
            0.0,
            np.pi / 2.0,
            limit=400,
            epsabs=1e-16,
            epsrel=1e-15,
        )
        assert abs(float(moment) - expected) < 1e-12 * scale


def test_the_harmonic_family_carries_the_confluence_as_a_finite_part():
    """At ``k^2 = 1`` every harmonic moment diverges with the SAME logarithm.

    ``cos 2 n a`` equals ``(-1)^n`` exactly where ``Delta`` vanishes, so the whole
    family shares one divergence with weights ``(-1)^n`` -- which is why a
    contraction against it puts the numerator's value at that end on the divergence,
    and why a reduction whose own weight there is zero can take the finite parts and
    be done. With ``K`` returned as zero those finite parts are elementary:
    ``int (cos 2 n a - (-1)^n)/cos a da`` collapses on a polynomial.

        P_0 = 0,  P_1 = 2,  P_2 = -8/3,  P_3 = 46/15

    and the recursion reproduces them, which is the check: the first two are the
    seeds, the rest are the relation.
    """
    from nova.biot.elliptic import harmonic_moments

    moments = harmonic_moments(np.float64(1.0), 4, complement=np.float64(0.0))
    np.testing.assert_allclose(
        [float(value) for value in moments], [0.0, 2.0, -8.0 / 3.0, 46.0 / 15.0]
    )


@pytest.mark.parametrize("complement", [1e-4, 1e-9, 1e-14])
def test_the_harmonic_family_satisfies_its_own_relation_near_the_confluence(complement):
    """Where no quadrature reference survives, the recursion is pinned on itself.

    ``(2n + 1) k^2 P(n+1) + 4n (1 + k'^2) P(n) + (2n - 1) k^2 P(n-1) = 0`` is what
    the family is built from, so satisfying it proves nothing about the seeds -- but
    ``P_0`` IS ``K``, taken from the complement, and the residual then says the
    upward direction has not lost the seeds on the way. Both together are what the
    quadrature would have said.
    """
    from nova.biot.elliptic import harmonic_moments

    parameter = np.float64(1.0) - np.float64(complement)
    moments = harmonic_moments(parameter, 10, complement=np.float64(complement))
    expected = scipy.special.elliprf(0.0, complement, 1.0)
    np.testing.assert_allclose(float(moments[0]), expected, rtol=1e-15)
    for order in range(1, len(moments) - 1):
        residual = (
            (2 * order + 1) * parameter * moments[order + 1]
            + 4.0 * order * (1.0 + complement) * moments[order]
            + (2 * order - 1) * parameter * moments[order - 1]
        )
        assert abs(float(residual)) < 1e-13 * order * expected
