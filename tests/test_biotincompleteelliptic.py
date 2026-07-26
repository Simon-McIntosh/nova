"""Contract for the incomplete elliptic integrals at an interior amplitude.

The module under test is the finite arc's counterpart to the complete routine the
full turn uses, so everything here is checked against the two things it has to
agree with:

* **its own defining integral**, over the whole double range of modulus complement
  and over amplitudes from a rounding error to a quarter turn.  The reference is
  built in ``np.longdouble`` on a composite ``sinh`` map, and never by adaptive
  quadrature: the integrand turns over within ``k'`` of the far end of the range,
  which is exactly where the accuracy question is, and an adaptive rule is the
  weaker party there.  Both the map and its composition are load-bearing --
  :func:`_defining_integral` records what each of the two obvious shortcuts costs,
  measured, because each of them fails by more than the tolerance being asserted
  and so looks like an error in the routine.

* **the complete routine at a quarter-turn amplitude**, to the last bit.  That is
  the strongest single check available, because it is the limit in which the arc
  closes into the ring the shipped reduction already evaluates, and because it is
  the configuration a naive implementation gets catastrophically wrong -- the
  amplitude step goes through ``tan``, a quarter turn's tangent is 1.6e16 rather
  than infinity, and the descent then never doubles.  Measured on the way in:
  taking the step from the tangent saturates the first kind near 38 where the true
  value is 347, an 89 % error, at a complement of 1e-300.

Two exact limits need no reference at all.  A vanishing parameter makes both kinds
the amplitude itself, and a vanishing complement -- the target ON the source ring
-- makes them ``log((1 + sin a)/cos a)`` and ``sin a``.

The fixed trip count is the other thing under test, as it is for the complete
routine: a traced evaluation cannot iterate to a convergence test, so the descent
runs a constant number of times, and that constant is a claim about the argument
range asserted here in both directions.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from nova.biot.completeelliptic import complete_kind
from nova.biot.incompleteelliptic import TRIPS, incomplete_kind

LONG_PI = np.longdouble("3.14159265358979323846264338327950288")

# k'^2 for a ring is the target's squared distance to the source point over the
# squared ring span, so the sweep spans the double range because the geometry does.
COMPLEMENTS = np.array(
    [1.0, 0.9, 0.5, 0.1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-16, 1e-20, 1e-40, 1e-80, 1e-300]
)

# The arc's amplitude is (pi + psi)/2 for an azimuthal separation psi from one of
# its ends, so a quarter turn is the target ON that end and the range below it is
# everything else. Held as the COMPLEMENT of a quarter turn, because that is the
# quantity whose smallness the routine is sensitive to. A whole quarter turn --
# the amplitude vanishing outright -- has its own test, since the pair a caller
# supplies for it is exactly (0, 1) rather than a cosine of a right angle.
CO_AMPLITUDES = np.array([1.4, 1.2, 0.7, 0.2, 1e-2, 1e-4, 1e-8, 1e-12, 0.0])

# Where the two smallnesses meet the routine loses digits, and the boundary is
# measured rather than assumed: below this co-amplitude AND below the complement
# beside it, the amplitude's own step underflows before the descent has begun.
# Geometrically it is a target within a femtometre of the arc's END EDGE -- both
# on a section corner and on the end plane -- and the tests exclude only that.
FRAGILE_CO_AMPLITUDE = 1e-7
FRAGILE_COMPLEMENT = 1e-25


def pair(co_amplitude):
    """Return ``(amplitude, sin, cos)`` from the amplitude's distance below pi/2.

    The pair is what a caller supplies from its own geometry, and forming it from
    the co-amplitude is how a test supplies it exactly: the arc reduction's own
    half-angle does the same thing, and the point of the argument is that neither
    goes through ``cos`` of a quantity near a quarter turn.
    """
    return 0.5 * np.pi - co_amplitude, np.cos(co_amplitude), np.sin(co_amplitude)


def extended_complete(complement):
    """Return ``K`` in longdouble, ``pi/(2 AGM(1, k'))``, iterated to convergence."""
    complement = np.longdouble(complement)
    mean, geometric = np.longdouble(1.0), np.sqrt(complement)
    for _ in range(90):
        mean, geometric = 0.5 * (mean + geometric), np.sqrt(mean * geometric)
    return LONG_PI / (2.0 * mean)


def _defining_integral(co_amplitude, complement, radical):
    """Return either kind's defining integral in longdouble, resolved everywhere.

    Integrated in ``b = pi/2 - a``, measured from the FAR end of the range, so the
    range runs from the co-amplitude up to a quarter turn.  In that variable the
    modulus radical is ``sqrt(sin^2 b + k'^2 cos^2 b)``, which turns over within
    ``k'`` of ``b = 0`` -- just outside the range, and the closer to it the harder
    the configuration is.  ``b = k' sinh t`` is exact for that model, so both
    integrands become slowly-varying functions of ``t``.

    The rule over ``t`` is COMPOSITE, panels of unit width, and that is not
    padding.  The map stretches the range to ``arcsinh(1/k')`` -- three hundred
    units at the smallest complement -- and over it the second kind's integrand
    grows like ``e^(2t)``, which no single fixed rule follows.  A plain rule on the
    unmapped range fails the other way: it misses the layer entirely, which is
    worth ``k'^2 log(1/k')`` and so lands at 1e-8 for a mid-range complement --
    measured, six decades outside the tolerance being asserted, and looking for all
    the world like an error in the routine.

    Neither kind is built as a complete value less a tail.  For the first the mean
    would supply one without cancellation, but for the second the only longdouble
    route to the complete value is the difference sequence, which DOES cancel: at a
    complement of 1e-40 that costs the reference 1e-13 where the routine under test
    is at 3e-16.  One construction for both avoids the question.
    """
    complement = np.longdouble(complement)
    co_amplitude = np.longdouble(co_amplitude)
    root = np.sqrt(complement)
    if root == 0.0:
        # the target ON the source ring: the radical is sin b and both kinds are
        # elementary, so the reference is the closed form rather than a rule
        if radical < 0:
            return -np.log(np.tan(0.5 * co_amplitude))
        return np.cos(co_amplitude)
    node, weight = (np.longdouble(term) for term in leggauss(60))
    lower, upper = np.arcsinh(co_amplitude / root), np.arcsinh(LONG_PI / (2 * root))
    panels = max(int(np.ceil(float(upper - lower))), 1)
    edge = lower + (upper - lower) * np.arange(panels + 1, dtype=np.longdouble) / panels
    width = 0.5 * (edge[1] - edge[0])
    stretch = edge[:-1, None] + width * (node[None, :] + 1.0)
    offset = root * np.sinh(stretch)
    modulus = np.sqrt(np.sin(offset) ** 2 + complement * np.cos(offset) ** 2)
    integrand = root * np.cosh(stretch) * modulus ** np.longdouble(radical)
    return width * (integrand @ weight).sum()


def extended_first_kind(co_amplitude, complement):
    """Return ``F`` in longdouble from the composite map above."""
    return _defining_integral(co_amplitude, complement, -1)


def extended_second_kind(co_amplitude, complement):
    """Return ``E`` in longdouble from the composite map above."""
    return _defining_integral(co_amplitude, complement, 1)


@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_quarter_turn_reproduces_the_complete_routine_to_the_last_bit(complement):
    """The limit in which the arc closes onto the ring the full turn evaluates."""
    amplitude, sine, cosine = pair(0.0)
    first, second = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
    complete_first, complete_second = complete_kind(np.asarray(complement))
    assert float(first) == pytest.approx(float(complete_first), rel=4e-16, abs=1e-300)
    assert float(second) == pytest.approx(float(complete_second), rel=4e-16)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_first_kind_reproduces_its_defining_integral(co_amplitude, complement):
    """Over the whole double range of complement and amplitude but one corner."""
    if co_amplitude and co_amplitude < FRAGILE_CO_AMPLITUDE:
        if complement < FRAGILE_COMPLEMENT:
            pytest.skip("the arc end edge, whose boundary its own test measures")
    amplitude, sine, cosine = pair(co_amplitude)
    first, _ = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
    expected = float(extended_first_kind(co_amplitude, complement))
    assert float(first) == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_second_kind_reproduces_its_defining_integral(co_amplitude, complement):
    """Bounded everywhere, so it holds where the first kind is at its worst."""
    amplitude, sine, cosine = pair(co_amplitude)
    _, second = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
    expected = float(extended_second_kind(co_amplitude, complement))
    assert float(second) == pytest.approx(expected, rel=2e-14, abs=1e-300)


def test_the_amplitude_step_taken_from_the_tangent_is_what_this_repairs():
    """The saturation the ``arctan2`` form removes, measured in both arrangements.

    A quarter-turn amplitude has a tangent of 1.6e16 rather than infinity, so a
    step formed as ``arctan(k' tan a)`` returns nothing where the true step is a
    quarter turn.  The descent then never doubles and the first kind saturates on
    ``log(2/cos(pi/2))`` -- a number that depends on nothing but the floating-point
    representation of a right angle.
    """
    complement = 1e-300
    saturation = np.log(2.0 / np.cos(0.5 * np.pi))
    amplitude, sine, cosine = pair(0.0)
    got, _ = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
    complete, _ = complete_kind(np.asarray(complement))
    assert float(complete) > 300.0
    assert float(got) == pytest.approx(float(complete), rel=4e-16)
    # the arrangement being repaired, run here so the claim is measured and not
    # asserted: it lands on the representation's own ceiling, an 88 % error
    ratio, phase = np.sqrt(complement), amplitude
    mean, geometric = 1.0, np.sqrt(complement)
    for _ in range(TRIPS):
        principal = np.arctan(ratio * np.tan(phase))
        phase = phase + principal + np.pi * np.round((phase - principal) / np.pi)
        mean, geometric = 0.5 * (mean + geometric), np.sqrt(mean * geometric)
        ratio = geometric / mean
    saturated = phase / (2.0**TRIPS * mean)
    assert saturated == pytest.approx(saturation, rel=1e-6)
    assert saturated < 0.15 * float(complete)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
def test_a_vanishing_parameter_makes_both_kinds_the_amplitude(co_amplitude):
    """``k = 0`` is a target on the axis or at infinity; the integrand is one."""
    amplitude, sine, cosine = pair(co_amplitude)
    first, second = incomplete_kind(
        amplitude, 1.0, sine=sine, cosine=cosine, parameter=0.0
    )
    assert float(first) == pytest.approx(amplitude, rel=4e-16)
    assert float(second) == pytest.approx(amplitude, rel=4e-16)


@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_a_vanishing_amplitude_returns_nothing_at_every_modulus(complement):
    """The target diametrically opposite the arc's end: the range is empty.

    The pair is exactly ``(0, 1)`` here, which is what a caller forms from its own
    half-separation; the amplitude's own cosine would be a rounding error instead
    and the integrals would return one to match it.
    """
    first, second = incomplete_kind(0.0, complement, sine=0.0, cosine=1.0)
    assert float(first) == 0.0
    assert float(second) == 0.0


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
def test_a_target_on_the_source_ring_is_elementary(co_amplitude):
    """``k'^2 = 0`` collapses the descent, and the two kinds are closed forms.

    Unlike the complete case this is a divergence only where the amplitude reaches
    a quarter turn as well; short of that both kinds are finite and exact, and at
    it the complete routine's finite part is returned so the arc and the ring agree
    in the limit the arc closes.
    """
    amplitude, sine, cosine = pair(co_amplitude)
    first, second = incomplete_kind(amplitude, 0.0, sine=sine, cosine=cosine)
    assert float(second) == pytest.approx(float(sine), rel=4e-16, abs=1e-300)
    if co_amplitude == 0.0:
        assert float(first) == 0.0
        return
    assert float(first) == pytest.approx(np.log((1.0 + sine) / cosine), rel=4e-16)


def test_the_arc_end_edge_corner_degrades_by_a_measured_amount():
    """Where both smallnesses meet, and by how much -- asserted in BOTH directions.

    The amplitude's first step is ``arctan2(k' sin a, cos a)``, and where the
    co-amplitude and the complementary modulus are both tiny that step underflows
    before the descent has begun: the routine returns the confluence's own
    elementary value instead of the one a hair away from it.  The configuration is
    a target on a section CORNER and within a rounding error of the arc's end
    plane at once.  The bound below is what it costs, so a regression and an
    unexplained improvement both fail.
    """
    worst = 0.0
    for co_amplitude in (1e-8, 1e-10, 1e-12, 1e-14):
        for complement in (1e-30, 1e-40, 1e-80, 1e-300):
            amplitude, sine, cosine = pair(co_amplitude)
            first, _ = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
            expected = float(extended_first_kind(co_amplitude, complement))
            worst = max(worst, abs(float(first) - expected) / abs(expected))
    assert 1e-6 < worst < 1e-3


def test_the_trip_count_is_bounded_by_the_argument_range_in_both_directions():
    """Long enough that the whole double range has converged, short enough to bite.

    The claim the constant makes is about the arithmetic-geometric mean's two
    phases: the trips before the pair is of one size halve the exponent gap, and
    the ones after double the correct digits.  Both are asserted, so padding the
    constant fails as surely as starving it.
    """
    amplitude, sine, cosine = pair(0.3)
    complements = np.array([1e-300, 1e-80, 1e-16, 1e-3, 0.5, 1.0])
    settled, _ = incomplete_kind(
        amplitude, complements, sine=sine, cosine=cosine, trips=TRIPS + 6
    )
    at_count, _ = incomplete_kind(amplitude, complements, sine=sine, cosine=cosine)
    short, _ = incomplete_kind(
        amplitude, complements, sine=sine, cosine=cosine, trips=TRIPS - 4
    )
    assert np.max(np.abs(at_count - settled) / settled) < 4e-16
    assert np.max(np.abs(short - settled) / settled) > 1e-8


def traced_namespace():
    """Return ``jax.numpy`` and a jit, or skip where jax is not installed."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax, jax.numpy


def test_the_traced_path_is_the_same_code_and_agrees_to_a_few_ulp():
    """One implementation, numpy on the host and a compiled kernel on a device."""
    jax, jnp = traced_namespace()
    amplitude, sine, cosine = pair(0.4)
    complements = np.array([1e-300, 1e-40, 1e-12, 1e-3, 0.4, 1.0])

    @jax.jit
    def traced(complement):
        return incomplete_kind(
            jnp.asarray(amplitude),
            complement,
            sine=jnp.asarray(sine),
            cosine=jnp.asarray(cosine),
            xp=jnp,
        )

    host = incomplete_kind(
        np.full_like(complements, amplitude), complements, sine=sine, cosine=cosine
    )
    device = traced(jnp.asarray(complements))
    for one, other in zip(host, device):
        assert np.max(np.abs(np.asarray(other) - one) / np.abs(one)) < 1e-14


def test_the_first_kind_differentiates_in_the_amplitude_exactly():
    """``dF/da = 1/Delta``, which is the definition and costs nothing to check."""
    jax, jnp = traced_namespace()
    complement = 1e-6

    def first(amplitude):
        return incomplete_kind(
            amplitude,
            jnp.asarray(complement),
            sine=jnp.sin(amplitude),
            cosine=jnp.cos(amplitude),
            xp=jnp,
        )[0]

    for amplitude in (0.2, 0.8, 1.4):
        got = float(jax.grad(first)(jnp.asarray(amplitude)))
        expected = 1.0 / np.sqrt(
            np.cos(amplitude) ** 2 + complement * np.sin(amplitude) ** 2
        )
        assert got == pytest.approx(expected, rel=1e-12)
