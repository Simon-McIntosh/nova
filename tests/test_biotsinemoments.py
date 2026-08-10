"""Contract for the sine-weighted moment family, which the full turn never forms.

The arc's azimuthal rows -- the vector potential's radial component and the
field's toroidal one -- carry ``sin phi`` where the other three rows carry
``cos phi``.  Over a full turn that odd weight integrates to nothing, which is
the parity argument that makes an axisymmetric ring carry no toroidal field, so
:mod:`nova.biot.elliptic` has no counterpart for any of this and there is nothing
above these functions to check them against.  Everything here is therefore held
to the defining integral directly.

Two claims are what the tests are really about, because between them they decide
how much the two new rows cost:

* the radical fold and the pole recursion are identities of the INTEGRAND, so
  :mod:`nova.biot.elliptic`'s own routines are correct on this family AS THEY
  STAND -- the same claim the cosine family's tests make, made again because the
  weight is different and the claim is easy to assume;
* the two pole SEEDS, which for the cosine family are incomplete integrals of
  the third kind, are ELEMENTARY here.  ``sin 2a`` is what the radical's own
  differential carries, so the integral collapses onto one inverse hyperbolic
  tangent in ``Delta``, and the tests pin that against quadrature over twelve
  decades of shift and both signs of the reflected pole.

The reference is a fixed Gauss rule over the actual range rather than an
adaptive one, for the reason ``tests/test_biotincompletemoments.py`` gives: only
a fixed rule can be shown to have converged by refining it.  Its panels are
graded geometrically into BOTH ends, which is what lets the sweeps here reach
two configurations that file has to leave out -- a quarter-turn amplitude, where
the radical's own layer of width ``k'`` sits ON the range end, and a pole root
within 1e-4 of the range start, where the integrand is a spike of that width.
Uniform panels are not a reference at either.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from nova.biot.elliptic import (
    POLE_HEADROOM,
    harmonic_pole_moments,
    harmonic_root_moments,
)
from nova.biot.incompletemoments import (
    HEADROOM,
    SWITCH,
    cn_pole_moment,
    harmonic_cosines,
    sine_cn_pole_moment,
    sine_moments,
    sine_sn_pole_moment,
    sn_pole_moment,
)
from nova.jax.config import configure_dtypes


def _f64(value):
    """Construct an explicitly selected double-precision JAX value."""
    configure_dtypes()
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float64)


# The amplitude held as its distance BELOW a quarter turn -- the arc's own half
# separation from one of its ends, and the quantity the accuracy depends on.
# Zero is the quarter turn itself, the configuration the arc closes onto.
CO_AMPLITUDES = [1.5, 1.0, 0.7, 0.4, 0.2, 5e-2, 1e-2, 1e-3, 0.0]

COMPLEMENTS = [0.99, 0.9, 0.5, 0.1, 3e-2, 1e-2, 6e-3, 3e-3, 1e-3, 1e-6, 1e-9, 1e-16]

ORDERS = 10

# The one parameter below the direction switch that the closure has to work
# hardest at: the tridiagonal solve's contraction ratio reaches one as the
# modulus does, so its slowest usable point is just under the switch.
SLOWEST_COMPLEMENT = 1.0 - SWITCH + 5e-4


def graded(amplitude, integrand, levels=60, ratio=0.6, nodes=48):
    """Return the integral over ``[0, amplitude]``, panels graded to both ends.

    Each half carries panels falling geometrically into its own end, so the
    smallest is around 1e-13 of the range and any endpoint layer wider than that
    is resolved -- whichever of the three produces it: the radical's ``k'``, a
    pole root's ``sqrt(shift)``, or the amplitude's own approach to a quarter
    turn.
    """
    if amplitude == 0.0:
        return 0.0
    half = 0.5 * amplitude
    tail = half * ratio ** np.arange(levels, 0, -1)
    lower = np.concatenate([[0.0], tail, [half]])
    edge = np.concatenate([lower, amplitude - lower[-2::-1]])
    node, weight = leggauss(nodes)
    span = 0.5 * np.diff(edge)[:, None]
    angle = edge[:-1, None] + span * (node[None, :] + 1.0)
    return float((span * integrand(angle) @ weight).sum())


def radical(angle, complement):
    """Return ``Delta``, built from the complement so it holds at the confluence."""
    return np.sqrt(np.cos(angle) ** 2 + complement * np.sin(angle) ** 2)


def reference_sine(amplitude, complement, order, **rule):
    """Return ``integral_0^a sin(2a) cos(2 n a)/Delta da``."""
    return graded(
        amplitude,
        lambda angle: (
            np.sin(2 * angle) * np.cos(2 * order * angle) / radical(angle, complement)
        ),
        **rule,
    )


def reference_root(amplitude, complement, order):
    """Return ``integral_0^a sin(2a) cos(2 n a) Delta da``."""
    return graded(
        amplitude,
        lambda angle: (
            np.sin(2 * angle) * np.cos(2 * order * angle) * radical(angle, complement)
        ),
    )


def reference_pole(amplitude, complement, shift, order, mirrored, **rule):
    """Return the sine pole family's own defining integral at one order."""

    def integrand(angle):
        base = np.sin(angle) ** 2 if mirrored else np.cos(angle) ** 2
        return (
            np.sin(2 * angle)
            * np.cos(2 * order * angle)
            / ((base + shift) * radical(angle, complement))
        )

    return graded(amplitude, integrand, **rule)


def family(co_amplitude, complement, count=ORDERS, **kwargs):
    """Return the sine moments, with the amplitude's pair formed exactly.

    ``sin(pi/2 - c) = cos c`` and ``cos(pi/2 - c) = sin c``, so holding the
    amplitude by its distance below a quarter turn gives a pair that is exact
    there -- which is the configuration the arc closes onto.
    """
    return sine_moments(
        0.5 * np.pi - co_amplitude,
        1.0 - complement,
        count,
        complement=complement,
        sine=np.cos(co_amplitude),
        cosine=np.sin(co_amplitude),
        **kwargs,
    )


def seed(co_amplitude, complement, shift, mirrored):
    """Return the family's seed from the routine, in whichever orientation."""
    routine = sine_sn_pole_moment if mirrored else sine_cn_pole_moment
    return float(
        routine(
            np.asarray(shift),
            1.0 - complement,
            np.cos(co_amplitude),
            np.sin(co_amplitude),
            complement=np.asarray(complement),
        )
    )


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_sine_moments_reproduce_their_defining_integral(co_amplitude, complement):
    """Over amplitude and parameter, every order, against graded quadrature.

    The tolerance is relative to the ZEROTH moment rather than to each order's
    own value, for the reason the cosine family's is: only absolute accuracy of
    that size is wanted, because the high harmonics are small and multiply
    coefficients no larger than the low ones.
    """
    amplitude = 0.5 * np.pi - co_amplitude
    got = family(co_amplitude, complement)
    scale = abs(float(got[0]))
    for order in range(ORDERS):
        expected = reference_sine(amplitude, complement, order)
        assert abs(float(got[order]) - expected) < 5e-13 * scale


@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_zeroth_moment_meets_its_closed_form_at_a_quarter_turn(complement):
    """``S_0 = 2 sin^2(phi)/(1 + Delta)``, which is the relation at order zero.

    Written as the printed ``2 (1 - Delta)/k^2`` instead it is a difference of
    two nearly-equal numbers over a small one, and a distant target -- where the
    parameter is what is small -- loses every digit it has.  The closed form is
    what seeds the upward direction; this checks that the SOLVE, which is the
    other direction and the one that carries most of the parameter range, lands
    on the same value at the quarter turn where it is known outright.
    """
    got = float(family(0.0, complement, count=1)[0])
    expected = 2.0 / (1.0 + np.sqrt(complement))
    assert abs(got - expected) <= 1e-15 * expected


def test_the_exact_quarter_confluence_has_analytic_moments_without_warnings():
    """At ``k'^2 = 0`` the radical cancels and every moment is elementary."""
    with np.errstate(all="raise"):
        got = family(0.0, 0.0)
    expected = [
        2.0 if order == 0 else 1.0 / (2 * order + 1) - 1.0 / (2 * order - 1)
        for order in range(ORDERS)
    ]
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=2e-15)


@pytest.mark.parametrize("co_amplitude", [1.0, 0.4, 5e-2])
@pytest.mark.parametrize("complement", [0.9, 1e-2, 1e-9])
def test_the_root_fold_holds_on_this_family_too(co_amplitude, complement):
    """``Delta^2`` folded back knows nothing about the weight the integrand carries.

    So :func:`nova.biot.elliptic.harmonic_root_moments` is correct on the sine
    family exactly as it stands, and the arc's azimuthal rows need no radical
    routine of their own.  This is the test that would fail if that were wrong.
    """
    amplitude = 0.5 * np.pi - co_amplitude
    plain = family(co_amplitude, complement, count=ORDERS + 1)
    got = harmonic_root_moments(plain, 1.0 - complement)
    scale = abs(float(plain[0]))
    for order in range(ORDERS):
        expected = reference_root(amplitude, complement, order)
        assert abs(float(got[order]) - expected) < 5e-13 * scale


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("shift", [0.25, 1.0, 4.0, 1e3])
@pytest.mark.parametrize("co_amplitude", [1.0, 0.4, 5e-2])
def test_the_pole_recursion_holds_on_this_family_too(mirrored, shift, co_amplitude):
    """The pole factor multiplied back is likewise an identity of the integrand.

    The shifts are the ones a caller routes through the family rather than
    through the exact-weight route -- a root far enough past the range for the
    closure error to decay -- and the seed comes from the routine, so a defect in
    the elementary seed reaches this test as well as its own.
    """
    complement = 1e-3
    amplitude = 0.5 * np.pi - co_amplitude
    plain = family(co_amplitude, complement, count=ORDERS + POLE_HEADROOM + 2)
    value = seed(co_amplitude, complement, shift, mirrored)
    got = harmonic_pole_moments(
        np.asarray(shift), np.asarray(value), plain, ORDERS, mirrored=mirrored
    )
    for order in range(ORDERS):
        expected = reference_pole(amplitude, complement, shift, order, mirrored)
        assert abs(float(got[order]) - expected) < 5e-13 * abs(value)


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("shift", [1e-6, 1e-4, 1e-2, 0.25, 1.0, 30.0, 1e4, 1e6])
@pytest.mark.parametrize("co_amplitude", [1.2, 0.5, 1e-2])
@pytest.mark.parametrize("complement", [0.9, 1e-2, 1e-8])
def test_the_pole_seed_is_elementary_and_holds_at_every_shift(
    mirrored, shift, co_amplitude, complement
):
    """RELATIVE accuracy, because a near root multiplies a numerator end value.

    The seed is the whole third-kind cost the cosine family pays, and here it is
    one inverse hyperbolic tangent.  Both orientations are swept over the shift
    range a slender section reaches: the ``cos^2`` one has its root past the far
    end of the range, the ``sin^2`` one past the near end -- which is where the
    range starts, so it is reached at every corner rather than occasionally.
    """
    amplitude = 0.5 * np.pi - co_amplitude
    got = seed(co_amplitude, complement, shift, mirrored)
    expected = reference_pole(amplitude, complement, shift, 0, mirrored)
    assert abs(got - expected) < 5e-14 * abs(expected)


@pytest.mark.parametrize("co_amplitude", [1.0, 0.3])
def test_the_seed_crosses_its_own_sign_change_without_noticing(co_amplitude):
    """``q^2 = k'^2 - shift k^2`` changes sign, and one even function covers both.

    Above the crossing the seed is an inverse hyperbolic tangent and below it an
    arctangent; they are the same series, so what the evaluation must not do is
    lose accuracy AT the crossing, where both arguments vanish.  Swept densely
    through it against quadrature.
    """
    complement = 0.2
    amplitude = 0.5 * np.pi - co_amplitude
    crossing = complement / (1.0 - complement)
    for factor in np.geomspace(1e-6, 1e6, 25):
        shift = crossing * factor
        got = seed(co_amplitude, complement, shift, False)
        expected = reference_pole(amplitude, complement, shift, 0, False)
        assert abs(got - expected) < 5e-14 * abs(expected), f"factor {factor:.1e}"


def test_a_root_on_the_range_end_returns_the_rings_own_convention():
    """A shift of exactly zero, which a target level with a corner produces.

    Zero, as :func:`nova.biot.incompletemoments.cn_pole_moment` and its mirror
    return it, so the arc and the ring agree in the limit the arc closes.  The
    callers that reach it carry a weight on such a pole that is itself exactly
    zero, so the convention is never the answer -- but a mismatch between the two
    families would put a spurious difference into the assembled arc.
    """
    for mirrored in (False, True):
        assert seed(0.4, 1e-3, 0.0, mirrored) == 0.0


def test_inactive_seed_branches_are_held_at_the_exact_confluence():
    """Zero shifts and positive shifts avoid every unselected root and quotient."""
    with np.errstate(all="raise"):
        for shift in (-1.0, 0.0):
            assert sine_cn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0) == 0.0
            assert sine_sn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0) == 0.0
            assert cn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0) == 0.0
            assert sn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0) == 0.0
        for shift in (1e-12, 0.5, 2.0):
            cn_value = sine_cn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0)
            sn_value = sine_sn_pole_moment(shift, 1.0, 1.0, 0.0, complement=0.0)
            cn_expected = 2.0 * np.arctan(1.0 / np.sqrt(shift)) / np.sqrt(shift)
            sn_root = np.sqrt(1.0 + shift)
            sn_gap = shift / (sn_root * (1.0 + sn_root))
            sn_expected = np.log1p(2.0 / (sn_root * sn_gap)) / sn_root
            assert float(cn_value) == pytest.approx(cn_expected, rel=3e-15)
            assert float(sn_value) == pytest.approx(sn_expected, rel=3e-15)


def test_the_seed_branch_crossing_has_one_value_and_one_jacobian():
    """The circular and hyperbolic forms share their analytic squared series."""
    jax = pytest.importorskip("jax")
    configure_dtypes()
    jnp = jax.numpy
    crossing = 0.25

    def value(shift):
        return sine_cn_pole_moment(
            shift,
            _f64(0.8),
            _f64(0.6),
            _f64(0.8),
            complement=_f64(0.2),
            xp=jnp,
        )

    samples = (
        np.nextafter(crossing, 0.0),
        crossing,
        np.nextafter(crossing, np.inf),
    )
    derivatives = []
    for shift in samples:
        forward = float(jax.jacfwd(value)(_f64(shift)))
        reverse = float(jax.grad(value)(_f64(shift)))
        assert np.isfinite(forward)
        assert forward == pytest.approx(reverse, rel=2e-15)
        derivatives.append(forward)
    np.testing.assert_allclose(derivatives, derivatives[1], rtol=2e-15, atol=0.0)

    step = 1e-5
    difference = (
        float(value(_f64(crossing + step))) - float(value(_f64(crossing - step)))
    ) / (2.0 * step)
    assert derivatives[1] == pytest.approx(difference, rel=2e-9)


def test_the_harmonic_cosines_are_exact_at_a_quarter_turn():
    """``(-1)^n`` to the last bit, which is what makes the arc close onto the ring.

    Taken from ``cos`` of an assembled right angle the first harmonic is
    ``-1 + 1e-16`` and the error grows linearly with the order; taken from the
    amplitude's own pair the doubled cosine is exactly ``-1`` and every harmonic
    after it is exact.
    """
    got = harmonic_cosines(np.float64(1.0), np.float64(0.0), 20)
    for order, value in enumerate(got):
        assert float(value) == (-1.0) ** order


def test_the_reference_rule_has_converged():
    """Refining it must not move it, or nothing above means anything."""
    for co_amplitude in (1e-2, 0.0):
        amplitude = 0.5 * np.pi - co_amplitude
        for complement in (0.5, 1e-9):
            for order in (0, 5, 9):
                coarse = reference_sine(amplitude, complement, order, levels=40)
                fine = reference_sine(amplitude, complement, order)
                assert abs(coarse - fine) < 1e-15 * max(abs(fine), 1.0)
    for shift, mirrored in ((1e-6, True), (1e-6, False)):
        coarse = reference_pole(1.0, 1e-8, shift, 0, mirrored, levels=40)
        fine = reference_pole(1.0, 1e-8, shift, 0, mirrored)
        assert abs(coarse - fine) < 1e-14 * abs(fine)


def test_the_closure_headroom_is_needed_at_the_shipped_depth():
    """Starving it must fail, or the constant is padding rather than a measurement.

    The family is differenced out of a system whose closure error decays by the
    contraction ratio per order, and that ratio reaches one as the modulus does
    -- so the depth is set just under the switch, where the solve is still the
    selected direction and is contracting as slowly as it ever has to.  Measured
    there: 8.6e-06 at a tenth of the shipped headroom against 7e-18 at the
    shipped depth, both against twice it.
    """
    co_amplitude = 0.4
    shipped = float(family(co_amplitude, SLOWEST_COMPLEMENT)[4])
    starved = float(
        family(co_amplitude, SLOWEST_COMPLEMENT, headroom=HEADROOM // 10)[4]
    )
    padded = float(family(co_amplitude, SLOWEST_COMPLEMENT, headroom=2 * HEADROOM)[4])
    assert abs(shipped - padded) < 1e-15
    assert abs(starved - padded) > 1e-7
