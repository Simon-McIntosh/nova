"""Contract for the complement-native complete elliptic integrals.

The module under test exists because of one fact about floating point: a
parameter cannot carry its own complement. Everything here is therefore checked
in the regime where that matters -- ``k'^2`` from the smallest positive double up
to one -- and every reference is chosen so that no check depends on the very
convention being tested. Four of them, each used only where it is the strongest:

* ``np.longdouble``, three decimal digits beyond the routine under test, running
  the ordinary arithmetic-geometric mean from the complement. ``K = pi/(2 AGM)``
  has no cancellation anywhere in it, so this is exact to its own precision over
  the whole range -- but the companion ``c``-sequence for ``E`` does cancel, and
  is NOT used.
* ``E`` beyond ``k'^2 = 0.1`` from the hypergeometric series in longdouble, which
  converges in tens of terms there.
* ``E`` below ``k'^2 = 1e-6`` from ``1 + (k'^2/2)(log(4/k') - 1/2)``, whose error is
  ``O(k'^4 log k')`` -- under 1e-24 across that whole span, so it is effectively
  exact where a series would need a million terms.
* scipy's Carlson symmetric forms, which take the complement as an argument, for
  the decades between; validated against the two above where they overlap. For the
  third kind they are used in the arrangement that keeps both terms POSITIVE -- the
  other arrangement, an ordinary ``Pi`` at a hugely negative characteristic, is a
  difference of near-equal terms and is measured here to be the weaker reference.

Two exact identities need no reference at all: a vanishing modulus makes the third
kind elementary, ``Pi(n | 0) = (pi/2)/sqrt(1 - n)``, and ``Pi(m | m) = E/(1 - m)``.
Between them they pin the third kind across eighteen decades of pole, where every
numerical reference is itself losing digits.

The fixed trip count is the other thing under test. A traced evaluation cannot
iterate to a convergence test, so the descent runs a constant number of times;
that constant is a claim about the argument range and is asserted here in both
directions -- long enough that the whole double range has converged, and short
enough that it is not padding.
"""

import numpy as np
import pytest
import scipy.integrate
import scipy.special

from nova.biot.completeelliptic import TRIPS, complete_kind, complete_pole
from nova.jax.config import configure_dtypes


def _f64(value):
    """Construct an explicitly selected double-precision JAX value."""
    configure_dtypes()
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float64)


EPS = float(np.finfo(np.float64).eps)
LONG_PI = np.longdouble("3.14159265358979323846264338327950288")

# k'^2 = (u^2 + (r' - r)^2)/a^2 for a ring: the target's squared distance to the
# source point over the squared ring span. One at a target on the axis or at
# infinity, and driven to zero -- exactly zero, for a target ON the ring -- as the
# target approaches. The sweep spans the whole double range because the geometry
# does: a micron from a metre-scale ring is already 1e-12.
COMPLEMENTS = np.array(
    [1.0, 1.0 - 1e-9, 0.9, 0.5, 0.1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-16]
    + [1e-20, 1e-40, 1e-80, 1e-160, 1e-300]
)

# p = 1 - n, the pole factor's value at the far end of the range. Below one the
# root sits past the near end of the range and above one past the far end; the
# polygon reduction's own shifts reach a squared aspect ratio and its reciprocal.
POLES = np.array(
    [1e-18, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0, 1e3, 1e6, 1e12, 1e18]
)


def extended_first_kind(complement):
    """Return ``K`` in longdouble from the complement, iterated to convergence.

    ``K = pi/(2 AGM(1, k'))``. The mean is a sequence of arithmetic means and
    square roots of positive quantities, and the division at the end is by a
    quantity of order one, so there is no cancellation in it at any argument -- the
    reason this and not the companion ``E`` recursion is used as a reference.
    """
    complement = np.asarray(complement, dtype=np.longdouble)
    mean, geometric = np.ones_like(complement), np.sqrt(complement)
    for _ in range(80):
        mean, geometric = 0.5 * (mean + geometric), np.sqrt(mean * geometric)
    return LONG_PI / (2.0 * mean)


def extended_second_kind_series(complement):
    """Return ``E`` in longdouble from its hypergeometric series.

    ``E = (pi/2) 2F1(-1/2, 1/2; 1; k^2)``, whose terms fall like ``k^(2n)/n^2``: tens
    of them beyond ``k'^2 = 0.1`` and a million as the complement vanishes, which is
    why this reference is used at one end of the range only.
    """
    parameter = np.longdouble(1.0) - np.longdouble(complement)
    term = total = np.longdouble(1.0)
    for order in range(20000):
        index = np.longdouble(order)
        term = term * (index - 0.5) * (index + 0.5) / (index + 1.0) ** 2 * parameter
        total = total + term
        if abs(term) < np.longdouble("1e-27") * abs(total):
            return LONG_PI / 2.0 * total
    raise AssertionError("the series reference did not converge at this argument")


def near_ring_second_kind(complement):
    """Return ``E`` from its small-complement expansion, error ``O(k'^4 log k')``."""
    return 1.0 + 0.5 * complement * (np.log(4.0 / np.sqrt(complement)) - 0.5)


def carlson_pole(pole, complement):
    """Return the third kind from scipy, in its all-positive arrangement.

    ``cel(k', p, 1, 1)`` is ``Pi(n | m)`` at ``n = 1 - p``, and Carlson's form for
    that is ``R_F(0, k'^2, 1) + (n/3) R_J(0, k'^2, 1, p)`` -- fine while ``n`` stays
    of order one, a difference of near-equal terms once ``p`` is large. Reflecting
    the pole factor onto the other end of the range gives the same integral as two
    POSITIVE terms, which is the arrangement taken above one.
    """
    pole = np.asarray(pole, dtype=np.float64)
    complement = np.asarray(complement, dtype=np.float64) + np.zeros_like(pole)
    first = scipy.special.elliprf(0.0, complement, 1.0)
    direct = first + (1.0 - pole) / 3.0 * scipy.special.elliprj(
        0.0, complement, 1.0, pole
    )
    reflected = (
        first
        + (1.0 - 1.0 / pole)
        * complement
        / 3.0
        * scipy.special.elliprj(0.0, complement, 1.0, complement / pole)
    ) / pole
    return np.where(pole > 1.0, reflected, direct)


def quadrature(pole, complement, cosine_weight=1.0, sine_weight=1.0):
    """Return the defining integral by adaptive quadrature, in the pole's scale.

    The integrand is a needle of width ``1/sqrt(p)`` at one end of the range, so it
    is integrated in the variable that spreads that needle out; the second scale,
    the modulus complement at the other end, is left to the rule.
    """

    def integrand(x):
        angle = np.arctan(x / np.sqrt(pole))
        cosine, sine = np.cos(angle) ** 2, np.sin(angle) ** 2
        return (
            (cosine_weight * cosine + sine_weight * sine)
            / ((cosine + pole * sine) * np.sqrt(cosine + complement * sine))
            * np.sqrt(pole)
            / (pole + x * x)
        )

    value, _ = scipy.integrate.quad(
        integrand, 0.0, np.inf, limit=800, epsabs=1e-300, epsrel=1e-13
    )
    return value


def test_the_references_agree_with_each_other_where_they_overlap():
    """Nothing is measured against a reference nobody checked."""
    moderate = np.array([1.0, 0.9, 0.5, 0.1])
    assert np.allclose(
        np.asarray(extended_first_kind(moderate), dtype=np.float64),
        scipy.special.elliprf(0.0, moderate, 1.0),
        rtol=2.0 * EPS,
        atol=0.0,
    )
    series = np.array([float(extended_second_kind_series(c)) for c in moderate])
    assert np.allclose(
        series, 2.0 * scipy.special.elliprg(0.0, moderate, 1.0), rtol=EPS, atol=0.0
    )
    tiny = np.array([1e-6, 1e-9, 1e-12, 1e-16])
    assert np.allclose(
        near_ring_second_kind(tiny),
        2.0 * scipy.special.elliprg(0.0, tiny, 1.0),
        rtol=2e-12,
        atol=0.0,
    )
    # the two ends of the range in closed form
    quarter = 0.5 * np.pi
    assert float(extended_first_kind(np.array([1.0]))[0]) == pytest.approx(quarter)
    assert float(extended_second_kind_series(1.0)) == pytest.approx(quarter)


def test_first_kind_holds_to_a_few_ulp_over_the_whole_double_range():
    """``K`` from the complement, against the extended-precision mean."""
    reference = np.asarray(extended_first_kind(COMPLEMENTS), dtype=np.float64)
    assert np.abs(complete_kind(COMPLEMENTS)[0] / reference - 1.0).max() < 4.0 * EPS


def test_second_kind_holds_to_a_few_ulp_against_each_reference_in_its_own_span():
    """``E`` against the series, the near-ring expansion, and scipy between them.

    A few ulp rather than none: the descent accumulates the second kind as a
    weighted running pair whose answer is of order one, so a vanishing complement --
    where ``E`` is one to every digit -- lands within a handful of ulp of it rather
    than on it exactly. The first kind, which grows without bound there, keeps every
    digit.
    """
    got = complete_kind(COMPLEMENTS)[1]
    carlson = 2.0 * scipy.special.elliprg(0.0, COMPLEMENTS, 1.0)
    assert np.abs(got / carlson - 1.0).max() < 8.0 * EPS
    far = COMPLEMENTS >= 0.1
    series = np.array([float(extended_second_kind_series(c)) for c in COMPLEMENTS[far]])
    assert np.abs(got[far] / series - 1.0).max() < 2.0 * EPS
    near = COMPLEMENTS <= 1e-9
    expansion = near_ring_second_kind(COMPLEMENTS[near])
    assert np.abs(got[near] / expansion - 1.0).max() < 4.0 * EPS


def test_the_parameter_route_is_what_the_complement_route_repairs():
    """Passing the parameter to scipy loses the complement, as claimed.

    Not a test of the module: a test of the premise it exists for. ``1 - k'^2`` is
    all a float parameter carries, so ``K``, which grows like ``-log k'``, comes back
    wrong by about ``eps/k'^2`` -- and below ``k'^2 = eps`` the parameter rounds to
    one outright and the answer is infinite.
    """
    reference = np.asarray(extended_first_kind(COMPLEMENTS), dtype=np.float64)
    parameter_route = scipy.special.ellipk(1.0 - COMPLEMENTS)
    error = np.abs(parameter_route / reference - 1.0)
    assert error[COMPLEMENTS == 1e-9] > 1e-9
    assert error[COMPLEMENTS == 1e-12] > 1e-7
    assert error[COMPLEMENTS == 1e-16] > 1e-4
    assert not np.isfinite(parameter_route[COMPLEMENTS < 1e-16]).any()
    # and the complement route is exact over the same span
    assert np.abs(complete_kind(COMPLEMENTS)[0] / reference - 1.0).max() < 4.0 * EPS


@pytest.mark.parametrize(
    "pole", [1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0, 2.0, 10.0, 1e3, 1e6]
)
def test_third_kind_matches_carlson_in_its_positive_arrangement(pole):
    """The pole form against scipy, over the complements a ring produces."""
    complement = COMPLEMENTS[COMPLEMENTS >= 1e-20]
    got = complete_pole(pole, complement)
    assert np.abs(got / carlson_pole(pole, complement) - 1.0).max() < 1e-13


def test_third_kind_matches_its_elementary_limit_at_every_pole():
    """A vanishing modulus makes the third kind elementary, at any pole.

    ``k^2 = 0`` -- a target on the axis, or at infinity -- collapses the radical and
    leaves ``Pi(n | 0) = (pi/2)/sqrt(1 - n)``. That is a closed form at EVERY pole,
    including the eighteen-decade ends where a numerical reference is itself losing
    digits, so it is the only check that spans the argument range the polygon
    reduction actually reaches.
    """
    got = complete_pole(POLES, 1.0)
    assert np.abs(got / (0.5 * np.pi / np.sqrt(POLES)) - 1.0).max() < 4.0 * EPS


def test_third_kind_matches_the_identity_that_ties_it_to_the_second():
    """``Pi(m | m) = E(m)/(1 - m)``: the pole at the parameter, reference-free."""
    complement = COMPLEMENTS[COMPLEMENTS > 0.0]
    got = complete_pole(complement, complement)
    ratio = got * complement / complete_kind(complement)[1]
    assert np.abs(ratio - 1.0).max() < 8.0 * EPS


@pytest.mark.parametrize("pole", [1e-6, 1e-3, 0.5, 2.0, 1e3, 1e6])
@pytest.mark.parametrize("complement", [1.0, 0.5, 1e-3])
def test_third_kind_reproduces_its_defining_integral(pole, complement):
    """Adaptive quadrature of the definition, where its two scales are separable."""
    got = float(complete_pole(np.float64(pole), np.float64(complement)))
    assert got == pytest.approx(quadrature(pole, complement), rel=1e-12)


def test_a_target_on_the_source_ring_returns_the_finite_part():
    """``k'^2 = 0`` is a working configuration, not an excluded one.

    A target ON the source ring -- which a grid across a polygon section reaches by
    alignment with one of its corners -- makes the first kind diverge
    logarithmically. Every integral here carries that one divergence with a weight
    the reduction sums to zero, so each returns its FINITE PART: zero for ``K``, one
    for ``E``, and for the pole form the elementary integral left when the
    divergence is subtracted with its own coefficient ``b/p``.
    """
    first, second = complete_kind(np.array([0.0, 0.0]))
    assert first.tolist() == [0.0, 0.0]
    assert second.tolist() == [1.0, 1.0]
    for pole in (1e-6, 0.3, 1.0, 3.0, 1e6):
        got = float(complete_pole(np.float64(pole), np.float64(0.0)))
        expected, _ = scipy.integrate.quad(
            lambda angle, p=pole: (
                (1.0 - 1.0 / p)
                * np.cos(angle)
                / (np.cos(angle) ** 2 + p * np.sin(angle) ** 2)
            ),
            0.0,
            0.5 * np.pi,
            limit=800,
            epsrel=1e-13,
        )
        assert got == pytest.approx(expected, rel=1e-12)
    # a root ON the range end has no finite part; zero is the callers' convention
    assert float(complete_pole(np.float64(0.0), np.float64(0.5))) == 0.0


def test_the_trip_count_is_bounded_by_the_argument_range_in_both_directions():
    """The fixed count is a measured claim about the range, not a guess.

    Long enough: at ``TRIPS`` every argument agrees with a run four times as long to
    a few ulp, over the complement's whole double range -- denormals included -- and
    eighteen decades of pole. Short enough: four trips fewer and the smallest
    complements come back wrong in the sixth decimal, so the constant is not
    padding. Each trip takes a geometric mean, so the correct digits double once the
    running pair is of one size and the exponent gap halves before that, which is
    why a single constant covers three hundred decades.
    """
    from nova.biot.completeelliptic import _accumulate, _descent

    complement = np.concatenate([np.logspace(-308, 0, 600), [1.0, 5e-324]])

    def cel(pole, trips):
        radicals, arithmetic = _descent(complement, np, trips)
        return _accumulate(radicals, arithmetic, pole, 1.0, 1.0, np)

    shortfall = 0.0
    for pole in (1e-18, 1e-6, 0.5, 1.0, 2.0, 1e6, 1e18):
        converged = cel(pole, 4 * TRIPS)
        assert np.abs(cel(pole, TRIPS) / converged - 1.0).max() < 16.0 * EPS
        shortfall = max(shortfall, np.abs(cel(pole, TRIPS - 4) / converged - 1.0).max())
    assert shortfall > 1e-6


# --- the same code path, traced -----------------------------------------------


def traced_namespace():
    """Return ``jax.numpy`` with float64 on, or skip."""
    jax = pytest.importorskip("jax")
    configure_dtypes()
    import jax.numpy as jnp

    return jax, jnp


@pytest.mark.slow
def test_the_traced_path_is_the_same_code_and_agrees_to_a_few_ulp():
    """One implementation, two namespaces: the trace is not a second transcription.

    Parity to a few ulp rather than bit-for-bit -- a compiler is free to reassociate
    the descent's arithmetic and to contract a multiply-add, which moves the last
    bits without moving the answer.
    """
    jax, jnp = traced_namespace()
    complement = _f64(COMPLEMENTS)
    first, second = jax.jit(lambda c: complete_kind(c, xp=jnp))(complement)
    host_first, host_second = complete_kind(COMPLEMENTS)
    assert np.abs(np.asarray(first) / host_first - 1.0).max() < 8.0 * EPS
    assert np.abs(np.asarray(second) / host_second - 1.0).max() < 8.0 * EPS
    for pole in (1e-12, 1e-3, 0.5, 1.0, 2.0, 1e6, 1e18):
        kernel = jax.jit(lambda p, c: complete_pole(p, c, xp=jnp))
        got = np.asarray(kernel(_f64(pole), complement))
        assert np.abs(got / complete_pole(pole, COMPLEMENTS) - 1.0).max() < 16.0 * EPS


@pytest.mark.slow
def test_the_trace_is_compiled_once_however_many_tiles_it_serves():
    """A fixed trip count and no value-dependent branch means one compilation.

    The point of the fixed count: a convergence test would either retrace per
    argument set or need a ``while_loop``, and both defeat the tiled build's one
    compilation per build. The shapes are the only thing the kernel is a function of.
    """
    jax, jnp = traced_namespace()
    kernel = jax.jit(lambda p, c: complete_pole(p, c, xp=jnp))
    for shift in (1e-9, 1e-3, 0.5, 7.0, 1e9):
        kernel(_f64(shift), _f64(COMPLEMENTS))
    assert kernel._cache_size() == 1


@pytest.mark.slow
def test_a_batch_of_moduli_maps_over_the_same_kernel():
    """``vmap`` over the pair axis, which is how a tile presents itself."""
    jax, jnp = traced_namespace()
    poles = _f64(POLES)
    complement = _f64(np.linspace(1e-9, 1.0, POLES.size))
    kernel = jax.jit(jax.vmap(lambda p, c: complete_pole(p, c, xp=jnp)))
    mapped = kernel(poles, complement)
    direct = complete_pole(POLES, np.asarray(complement))
    assert np.abs(np.asarray(mapped) / direct - 1.0).max() < 16.0 * EPS


@pytest.mark.slow
def test_the_first_kind_differentiates_exactly_over_the_whole_range():
    """A derivative is wanted, not just a value: nova differentiates through solves.

    Against the closed form in the complement, ``dK/dk'^2 = -(E - k'^2 K)/(2 k^2
    k'^2)``, rather than against a difference: a difference of a quantity that grows
    like ``-log k'`` across a step small enough to be a derivative is the weaker
    reference, while the expression above needs only the two values, both of which
    are exact to a few ulp.
    """
    jax, jnp = traced_namespace()
    for complement in (0.9, 0.5, 1e-3, 1e-8, 1e-14, 1e-20):
        first, second = (float(v) for v in complete_kind(np.float64(complement)))
        expected = -(second - complement * first) / (
            2.0 * (1.0 - complement) * complement
        )
        gradient = float(
            jax.grad(lambda c: complete_kind(c, xp=jnp)[0])(_f64(complement))
        )
        assert np.isfinite(gradient)
        assert gradient == pytest.approx(expected, rel=8.0 * EPS)


@pytest.mark.slow
def test_the_third_kind_differentiates_exactly_in_the_pole_and_the_modulus():
    """Against the closed-form derivatives of the third kind, which need only values.

    ``dPi/dn`` and ``dPi/dm`` are rational in the three complete integrals, so a
    reference for them costs nothing once the values are trusted -- and the
    expressions are singular where the pole meets the complement (the case
    ``Pi(m | m)``), which is why that one pairing is left out here rather than
    measured against a badly conditioned reference.
    """
    jax, jnp = traced_namespace()
    for complement in (0.9, 0.5, 1e-3, 1e-8):
        first, second = (float(v) for v in complete_kind(np.float64(complement)))
        for pole in (1e-3, 0.5, 3.0, 100.0):
            if abs(pole - complement) < 0.1 * pole:
                continue
            value = float(complete_pole(np.float64(pole), np.float64(complement)))
            expected = (
                (
                    second
                    + (pole - complement) * first / (1.0 - pole)
                    + (pole * pole - 2.0 * pole + complement) * value / (1.0 - pole)
                )
                / (2.0 * pole * (pole - complement)),
                (value - second / complement) / (2.0 * (pole - complement)),
            )
            for index in (0, 1):
                gradient = float(
                    jax.grad(lambda p, c: complete_pole(p, c, xp=jnp), argnums=index)(
                        _f64(pole), _f64(complement)
                    )
                )
                assert np.isfinite(gradient)
                assert gradient == pytest.approx(expected[index], rel=1e-11)


@pytest.mark.slow
def test_the_second_kind_derivative_degrades_where_its_value_becomes_one():
    """The one derivative that is not exact, bounded here rather than left unstated.

    ``E`` approaches one as the complement vanishes, and its whole dependence on the
    complement is a variation of relative size ``k'^2 log k'`` on that one. The
    descent's arrangement recovers the value to a few ulp but forms the DERIVATIVE as
    a difference of larger terms, so it loses about one digit per decade of
    complement: exact to round-off at ``k'^2 = 1e-3``, 1e-10 at 1e-8, and worthless
    by 1e-20. A caller that needs it there has it exactly from the two values,
    ``dE/dk'^2 = (K - E)/(2 k^2)``, which is the reference used below.
    """
    jax, jnp = traced_namespace()
    bounds = {0.9: 4.0 * EPS, 0.5: 4.0 * EPS, 1e-3: 1e-14, 1e-6: 1e-11, 1e-8: 1e-9}
    for complement, bound in bounds.items():
        first, second = (float(v) for v in complete_kind(np.float64(complement)))
        expected = (first - second) / (2.0 * (1.0 - complement))
        gradient = float(
            jax.grad(lambda c: complete_kind(c, xp=jnp)[1])(_f64(complement))
        )
        assert np.isfinite(gradient)
        assert abs(gradient / expected - 1.0) < bound
