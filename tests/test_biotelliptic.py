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


@pytest.mark.parametrize("parameter", PARAMETERS)
def test_sn_moments_reproduce_their_defining_integral(parameter):
    """El(2m) is the even sn moment over a quarter period."""
    moments = sn_moments(np.float64(parameter), 5)
    for order, moment in enumerate(moments):
        expected = quadrature(
            lambda u, order=order: jacobi(u, parameter)[0] ** (2 * order), parameter
        )
        # the recursion divides by k^2 once per order, so the attainable
        # tolerance degrades with order as the parameter falls
        tolerance = 1e-11 / parameter**order
        assert abs(moment - expected) <= tolerance * max(abs(expected), 1.0), (
            f"order {order}, parameter {parameter}: {moment} vs {expected}"
        )


def test_sn_moments_lose_a_digit_per_order_as_the_parameter_falls():
    """The conditioning limit is documented, so it is also asserted.

    At the smallest parameter a far target produces, the highest moment the
    vector potential needs has lost most of its digits. A far-field evaluation
    cannot go through this recursion, and a future transcription needs to know
    that before it debugs the wrong thing.
    """
    parameter = 1e-6
    moments = sn_moments(np.float64(parameter), 5)
    expected = quadrature(lambda u: jacobi(u, parameter)[0] ** 8, parameter)
    relative = abs(moments[4] - expected) / abs(expected)
    assert relative > 1e-6, "recursion is better conditioned than documented"


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
def test_the_stable_moments_agree_with_their_defining_integral_at_every_order(
    parameter,
):
    """The route used in anger has to hold where the printed recursion does not.

    ``sn_moments`` divides by ``k^2`` once per order, so at the small parameter a
    distant target produces it has lost most of its digits by the highest order
    the vector potential needs.  Running the same recursion DOWNWARD cures that,
    because the parasitic branch decays as ``k^(2m)`` -- but only while ``k^2``
    stays clear of unity, where the two branches become degenerate.  Neither
    direction covers the whole range, so the stable route picks per parameter and
    this test spans both sides of the switch.
    """
    from nova.biot.elliptic import stable_sn_moments

    moments = stable_sn_moments(np.float64(parameter), 9)
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
        expected = split_scale_pi(1.0 - complement, parameter, complement)
        got = complete_pi(
            np.float64(1.0 - complement), parameter, complement=np.float64(complement)
        )
        np.testing.assert_allclose(got, expected, rtol=1e-11)


def split_scale_pi(characteristic, parameter, complement, nodes=300):
    """Return Pi(n | m) by quadrature that resolves the ``1 - n`` scale.

    Adaptive integration of the Legendre form is useless here: the integrand
    carries a peak of width ``sqrt(1 - n)`` at ``a = pi/2``, so by ``1 - n =
    1e-12`` the reference, not the routine under test, is the inaccurate side.
    The interval is therefore split, and across the peak ``cos a`` is stretched
    by ``tan``, which absorbs it exactly and leaves a smooth integrand for a
    fixed rule.
    """
    node, weight = np.polynomial.legendre.leggauss(nodes)
    width = np.sqrt(complement / characteristic)
    # geometric mean of the two scales: wide enough that the flank's own rise is
    # resolved, narrow enough that the stretched peak integrand is near constant
    cut = min(0.5, np.sqrt(width))

    upper = np.arccos(cut)
    angle = 0.5 * upper * (node + 1.0)
    flank = (0.5 * upper * weight) @ (
        1.0
        / (
            (complement + characteristic * np.cos(angle) ** 2)
            * np.sqrt(1.0 - parameter * np.sin(angle) ** 2)
        )
    )

    upper = np.arctan(cut / width)
    cosine = width * np.tan(0.5 * upper * (node + 1.0))
    peak = (
        (width / complement)
        * (0.5 * upper * weight)
        @ (
            1.0
            / (
                np.sqrt(1.0 - parameter + parameter * cosine**2)
                * np.sqrt(1.0 - cosine**2)
            )
        )
    )
    return flank + peak
