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

The THIRD kind is held to the same two agreements, with its own reference: the
pole factor gives the integrand a SECOND layer, at the other end of the range and
of a width the pole sets, so :func:`_pole_integral` grades from both ends where
the first two kinds need only one.  Its own arrangement question is the reflection
of a near pole onto its far partner, and that too is run in both forms so the
cost of the printed one is a measurement -- it reaches the eleventh decimal at a
section a millionth as thick as it is wide, which is exactly the configuration
whose weight needs every digit.

The third kind is also the one place where the COMPILED path has to be held to the
reference in its own right rather than to the host.  A compiled kernel contracts
multiply-adds and reassociates sums, so the pole argument it forms can differ from
the host's in its last bit; where the symmetric forms' degenerate term rests on a
difference of two nearly-equal roots that bit is most of what the difference has,
and a host-against-device check cannot see the loss because it cannot tell which
of the two is right.  The sweep therefore holds both against
:func:`_pole_integral` absolutely, across the whole small-pole side.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import elliprf, elliprj

from nova.biot import incompleteelliptic
from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.incompleteelliptic import TRIPS, incomplete_kind, incomplete_pole

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


# The pole argument, which is the denominator's value at the FAR end of the range.
# Below one the root sits past that end -- outside a partial range altogether --
# and above it past the NEAR end, which is where the range starts, so that side is
# reached at every corner. Both are swept over the eighteen decades either side of
# one that the polygon reduction's own shifts reach.
POLES = [1e-12, 1e-8, 1e-3, 0.3, 1.0, 3.0, 1e3, 1e8, 1e12]

# Where the reflected pole k'^2/p stops being a normal double, past which the
# partner the reflection needs has fewer bits than a double carries. Reaching it
# takes a section a millionth as thick as it is wide AND a target within 1e-150
# ring spans of the source, at once.
SMALLEST_PARTNER = 2.3e-308


def _on_the_two_layers(co_amplitude, pole, complement, nodes, integrand):
    """Return ``integrand`` over the amplitude range, resolved everywhere.

    Two layers rather than one, so the range is halved and each half graded from
    its OWN end.  The pole factor turns over within ``1/sqrt(1 + p)`` of ``a = 0``
    and the whole denominator within ``k' sqrt(p)/sqrt(p + k'^2)`` of a quarter
    turn -- the second being ``k'`` where the modulus layer is the narrower of the
    two and ``sqrt(p)`` where the pole's is.  The two layers are properties of the
    DENOMINATORS, so the same pair grades the integral and its derivatives, which
    raise a denominator's power without moving where it turns over.

    Neither end angle is ever formed by subtracting from a quarter turn.  That is
    not tidiness: below the ulp of ``pi/2`` the subtraction loses the co-amplitude
    outright, and a reference built that way agrees with itself under refinement
    while being wrong in the third decimal -- measured, on the way to this one.

    ``integrand`` is called with the amplitude's ``(cosine, sine)`` and everything
    is longdouble, the caller having converted its own arguments.
    """
    node, weight = (np.longdouble(term) for term in leggauss(nodes))

    def graded(lower, top, layer, flipped):
        """Return the piece over ``[lower, top]``, mapped by sinh from ``lower``."""
        low, high = np.arcsinh(lower / layer), np.arcsinh(top / layer)
        panels = max(int(np.ceil(float(high - low))), 1)
        edge = low + (high - low) * np.arange(panels + 1, dtype=np.longdouble) / panels
        half = 0.5 * (edge[1] - edge[0])
        stretch = edge[:-1, None] + half * (node[None, :] + 1.0)
        angle = layer * np.sinh(stretch)
        pair = (np.sin(angle), np.cos(angle))
        if not flipped:
            pair = pair[::-1]
        return half * (layer * np.cosh(stretch) * integrand(*pair) @ weight).sum()

    half_amplitude = 0.5 * (LONG_PI / 2 - co_amplitude)
    return float(
        graded(np.longdouble(0.0), half_amplitude, 1.0 / np.sqrt(1.0 + pole), False)
        + graded(
            co_amplitude,
            LONG_PI / 4 + 0.5 * co_amplitude,
            np.sqrt(complement * pole / (pole + complement)),
            True,
        )
    )


def _pole_integral(co_amplitude, pole, complement, nodes=60):
    """Return the third kind's defining integral in longdouble."""
    pole = np.longdouble(pole)
    complement = np.longdouble(complement)
    co_amplitude = np.longdouble(co_amplitude)

    def integrand(cosine, sine):
        return 1.0 / (
            (cosine**2 + pole * sine**2) * np.sqrt(cosine**2 + complement * sine**2)
        )

    return _on_the_two_layers(co_amplitude, pole, complement, nodes, integrand)


def _pole_slope(co_amplitude, pole, complement, nodes=60):
    """Return d/d(pole) of the third kind's defining integral, in longdouble.

    Differentiated UNDER the integral sign rather than by differencing
    :func:`_pole_integral`, because a difference cannot reach the slope where it
    is most at issue.  The step has to sit inside the pole itself, so at a pole of
    1e-08 it is 1e-14 and the two values it separates are of order one -- the
    difference then keeps two digits of a float64 reference and the quotient is
    6e-04 out.  Measured, on the way to this one: differenced, the reference
    reports a routine that is right to a few ulp as being wrong in the fourth
    decimal, which is the opposite of the sign the test is looking for.

    The integrand is the same one with its pole factor squared and negated, so it
    turns over on the same two layers and takes the same grading.
    """
    pole = np.longdouble(pole)
    complement = np.longdouble(complement)
    co_amplitude = np.longdouble(co_amplitude)

    def integrand(cosine, sine):
        return -(sine**2) / (
            (cosine**2 + pole * sine**2) ** 2
            * np.sqrt(cosine**2 + complement * sine**2)
        )

    return _on_the_two_layers(co_amplitude, pole, complement, nodes, integrand)


@pytest.mark.parametrize("pole", POLES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_third_kind_reproduces_its_defining_integral(pole, complement):
    """Every pole the reduction reaches, over the whole double range of complement.

    Both orientations in one sweep: below a pole of one the root sits past the far
    end of the range and the symmetric forms are taken as printed, above it the
    reflection carries them, and the tolerance is the same on both sides because
    that is the claim.
    """
    for co_amplitude in CO_AMPLITUDES:
        amplitude, sine, cosine = pair(co_amplitude)
        got = incomplete_pole(np.asarray(pole), np.asarray(complement), sine, cosine)
        expected = _pole_integral(co_amplitude, pole, complement)
        assert float(got) == pytest.approx(expected, rel=1e-13)


def test_the_reference_has_converged_under_refinement():
    """Halving the rule must not move it, or nothing above means anything."""
    for co_amplitude in (0.7, 1e-4, 0.0):
        for pole in (1e-8, 1.0, 1e8):
            for complement in (0.5, 1e-16, 1e-300):
                coarse = _pole_integral(co_amplitude, pole, complement, nodes=30)
                fine = _pole_integral(co_amplitude, pole, complement, nodes=60)
                assert abs(coarse - fine) < 1e-15 * abs(fine)


@pytest.mark.parametrize("pole", POLES + [1e18])
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_quarter_turn_reproduces_the_complete_pole_to_a_few_ulp(pole, complement):
    """The limit in which the arc closes onto the ring, at every pole.

    Two entirely different algorithms have to meet here: the complete routine is
    one Bartky descent and this is a Carlson duplication with a reflection in
    front of it, so agreement to a few ulp is the strongest single statement
    available about either.
    """
    if complement / pole < SMALLEST_PARTNER:
        pytest.skip("the denormal partner, whose boundary its own test measures")
    amplitude, sine, cosine = pair(0.0)
    got = incomplete_pole(np.asarray(pole), np.asarray(complement), sine, cosine)
    expected = complete_pole(np.asarray(pole), np.asarray(complement))
    assert float(got) == pytest.approx(float(expected), rel=4e-15)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("pole", POLES)
def test_a_vanishing_parameter_leaves_the_elementary_integral(co_amplitude, pole):
    """``k = 0`` is a target on the axis or at infinity: the radical is one.

    What is left is ``arctan(sqrt(p) tan phi)/sqrt(p)``, which is the term the
    reflection puts in front -- so this is where a sign or a factor in it shows,
    with no elliptic integral left to hide behind.
    """
    amplitude, sine, cosine = pair(co_amplitude)
    got = incomplete_pole(np.asarray(pole), np.asarray(1.0), sine, cosine)
    root = np.sqrt(pole)
    expected = np.arctan2(root * sine, cosine) / root
    assert float(got) == pytest.approx(expected, rel=4e-15)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_a_unit_pole_is_the_first_kind(co_amplitude, complement):
    """The denominator is then one, and two unrelated algorithms must meet.

    Worth having beyond the reference checks because it crosses the two routes
    this module carries -- the amplitude-carrying descent and the symmetric forms
    -- at the one argument where they compute the same thing.

    The arc-end-edge corner is excluded here for the reason the descent's own test
    excludes it, and the exclusion is one-sided: measured against the longdouble
    reference there, the descent is 5.3e-06 out and the symmetric route 2.5e-16.
    The corner is the descent's and not the integral's.
    """
    if co_amplitude and co_amplitude < FRAGILE_CO_AMPLITUDE:
        if complement < FRAGILE_COMPLEMENT:
            pytest.skip("the arc end edge, whose boundary its own test measures")
    amplitude, sine, cosine = pair(co_amplitude)
    got = incomplete_pole(np.asarray(1.0), np.asarray(complement), sine, cosine)
    first, _ = incomplete_kind(amplitude, complement, sine=sine, cosine=cosine)
    assert float(got) == pytest.approx(float(first), rel=2e-13, abs=1e-300)


@pytest.mark.parametrize("pole", POLES)
def test_a_vanishing_amplitude_returns_nothing_at_every_pole(pole):
    """The target diametrically opposite the arc's end: the range is empty."""
    got = incomplete_pole(np.asarray(pole), np.asarray(1e-3), 0.0, 1.0)
    assert float(got) == 0.0


def test_a_root_on_the_range_end_keeps_the_complete_routine_s_convention():
    """Zero, so the arc and the ring agree in the limit the arc closes.

    At an interior amplitude the far end of the range is OUTSIDE it, so the
    integral is finite there and the convention is a choice rather than a
    necessity; it is the callers' choice, and their weight on such a pole is
    itself exactly zero.
    """
    for co_amplitude in (0.7, 1e-4, 0.0):
        _, sine, cosine = pair(co_amplitude)
        got = incomplete_pole(np.asarray(0.0), np.asarray(1e-3), sine, cosine)
        assert float(got) == 0.0
        assert float(complete_pole(np.asarray(0.0), np.asarray(1e-3))) == 0.0


def test_the_symmetric_forms_taken_as_printed_are_what_the_reflection_repairs():
    """The cancellation at a near pole, measured in both arrangements.

    ``F + (n/3) sin^3 phi R_J`` with ``n = 1 - p`` is the direct transcription.
    For a root just past the NEAR end of the range ``n`` is hugely negative, the
    two terms are of opposite sign and nearly equal, and their sum falls as
    ``1/sqrt(p)`` while each stays of order ``F``.  Run here so the loss is
    measured: half the decades of the pole, which at a shift of 1e-12 is six.
    """
    complement, co_amplitude = 1e-3, 0.7
    _, sine, cosine = pair(co_amplitude)
    squared_cosine = cosine * cosine
    radical = squared_cosine + complement * sine * sine
    worst = 0.0
    for pole in (1e8, 1e10, 1e12):
        expected = _pole_integral(co_amplitude, pole, complement)
        got = incomplete_pole(np.asarray(pole), np.asarray(complement), sine, cosine)
        assert float(got) == pytest.approx(expected, rel=4e-15)
        printed = sine * elliprf(squared_cosine, radical, 1.0) + (
            (1.0 - pole) / 3.0
        ) * sine**3 * elliprj(
            squared_cosine, radical, 1.0, squared_cosine + pole * sine * sine
        )
        worst = max(worst, abs(printed - expected) / abs(expected))
    # the printed form loses half the decades of the pole; the bound is two-sided
    # so an unexplained improvement fails as surely as a regression
    assert 1e-12 < worst < 1e-9


def test_the_denormal_partner_corner_degrades_by_a_measured_amount():
    """Where the reflected pole leaves the exponent range -- and by how much.

    The reflection needs ``k'^2/p``, and below the smallest normal double that
    partner carries fewer bits than a double does.  Geometrically it takes a
    section a millionth as thick as it is wide AND a target within 1e-150 ring
    spans of the source at once -- for a metre-scale ring, a separation a hundred
    and thirty decades below the Planck length.  The bound below is what the
    corner costs, asserted in both directions.
    """
    kept = max(
        abs(
            float(incomplete_pole(np.asarray(pole), np.asarray(complement), 1.0, 0.0))
            / float(complete_pole(np.asarray(pole), np.asarray(complement)))
            - 1.0
        )
        for pole, complement in ((1e6, 1e-300), (1e12, 1e-296), (1e18, 1e-290))
    )
    lost = max(
        abs(
            float(incomplete_pole(np.asarray(pole), np.asarray(complement), 1.0, 0.0))
            / float(complete_pole(np.asarray(pole), np.asarray(complement)))
            - 1.0
        )
        for pole, complement in ((1e6, 1e-310), (1e12, 1e-305), (1e18, 1e-300))
    )
    assert kept < 4e-15
    assert 1e-7 < lost < 1e-4


def test_the_third_kind_s_trip_count_is_bounded_in_both_directions():
    """The duplication's constant, seen through the integral that consumes it."""
    _, sine, cosine = pair(0.0)
    poles = np.array([1.0, 1e-12, 1e-3, 1e3, 1e12])
    complements = np.array([1e-300, 1e-300, 1e-40, 1e-16, 1e-3])
    settled = incomplete_pole(poles, complements, sine, cosine, trips=TRIPS + 6)
    at_count = incomplete_pole(poles, complements, sine, cosine)
    short = incomplete_pole(poles, complements, sine, cosine, trips=TRIPS - 4)
    assert np.max(np.abs(at_count - settled) / settled) < 4e-15
    assert np.max(np.abs(short - settled) / settled) > 1e-9


def test_the_third_kind_traces_and_agrees_to_a_few_ulp():
    """One implementation, numpy on the host and a compiled kernel on a device."""
    jax, jnp = traced_namespace()
    _, sine, cosine = pair(0.4)
    poles = np.array([1e-12, 1e-3, 1.0, 1e3, 1e12])
    complements = np.array([1e-300, 1e-40, 1e-12, 1e-3, 0.4])

    @jax.jit
    def traced(pole, complement):
        return incomplete_pole(
            pole, complement, jnp.asarray(sine), jnp.asarray(cosine), xp=jnp
        )

    host = incomplete_pole(poles, complements, sine, cosine)
    device = traced(jnp.asarray(poles), jnp.asarray(complements))
    assert np.max(np.abs(np.asarray(device) - host) / np.abs(host)) < 1e-14


# The poles the compiled path is held over. Below one the pole argument handed to
# the symmetric forms is within ``p`` of the amplitude's squared cosine, so it is
# the whole small-pole side -- not one corner of it -- that the degenerate form's
# two roots nearly coincide on.
FUSED_POLES = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 0.3, 1.0, 3.0, 1e3, 1e12]
FUSED_COMPLEMENTS = [1e-300, 1e-40, 1e-12, 1e-3, 0.4]


def test_the_compiled_third_kind_holds_the_extended_reference_over_the_sweep():
    """The compiled path against the defining integral DIRECTLY, not against numpy.

    Host-against-device says only that the two agree, which two paths that had
    drifted together would also say; this is the absolute statement.  It matters
    here because the arithmetic the two paths do is not the same: a compiled kernel
    contracts ``a b + c`` into one fused multiply-add and reassociates sums, so an
    argument the host rounds and the device does not enters the descent differing
    in its last bit -- and where the degenerate form's two roots are a part in
    1e12 apart, that last bit is most of what their difference has.

    Both the host and the compiled path are held to the same bound, over the whole
    small-pole side rather than one point of it, so the claim is that the region
    holds and not that a sample does.
    """
    jax, jnp = traced_namespace()

    @jax.jit
    def traced(pole, complement, sine, cosine):
        return incomplete_pole(pole, complement, sine, cosine, xp=jnp)

    worst_host = worst_device = 0.0
    for co_amplitude in (1.2, 0.4, 1e-4):
        _, sine, cosine = pair(co_amplitude)
        poles = np.array([pole for pole in FUSED_POLES for _ in FUSED_COMPLEMENTS])
        complements = np.array(
            [term for _ in FUSED_POLES for term in FUSED_COMPLEMENTS]
        )
        host = np.asarray(incomplete_pole(poles, complements, sine, cosine))
        device = np.asarray(
            traced(
                jnp.asarray(poles),
                jnp.asarray(complements),
                jnp.asarray(sine),
                jnp.asarray(cosine),
            )
        )
        for index, (pole, complement) in enumerate(zip(poles, complements)):
            if complement / pole < SMALLEST_PARTNER:
                continue  # the denormal partner, whose boundary its own test measures
            expected = _pole_integral(co_amplitude, pole, complement)
            worst_host = max(worst_host, abs(host[index] - expected) / abs(expected))
            worst_device = max(
                worst_device, abs(device[index] - expected) / abs(expected)
            )
    assert worst_host < 4e-15  # measured 2.2e-15
    assert worst_device < 4e-15  # measured 2.2e-15


# The small-pole side, and one reflected pole whose partner k'^2/p is DENORMAL.
# At the first four the gap between the pole argument and the amplitude's squared
# cosine is resolvable but small; at the last the pole argument rounds onto that
# cosine outright, which is the end of the ladder rather than a separate corner.
GAP_POLES = [1e-14, 1e-12, 1e-10, 1e-8]
DENORMAL_PARTNER_POLE = 1e12

# What subtracting the two leaves, as a fraction of the gap itself. The pole
# argument is ``cos^2 phi + partner sin^2 phi`` and the other three include
# ``cos^2 phi``, so their difference is quantised by the ulp of ``cos^2 phi`` and
# the subtraction keeps the gap only to ulp/gap -- 2.8e-17 over 8.5e-15 at the
# first of these. Round-to-nearest on a double is the whole mechanism, so these
# are floors on every machine rather than one host's measurement.
SUBTRACTED_GAP_FLOORS = [5e-4, 2e-6, 1e-8, 2e-10]  # measured 1.1e-3 to 5.2e-10


def _supplied_arguments(poles, complements, co_amplitude=0.4):
    """Return what :func:`incomplete_pole` hands the symmetric forms, as it forms it.

    The gaps are the point of the exercise, so they are taken from the entry point
    rather than rebuilt here: a routine that stopped supplying them, or supplied
    them by subtraction, is exactly what this has to see.
    """
    _, sine, cosine = pair(co_amplitude)
    seen = {}
    original = incompleteelliptic.symmetric_kinds

    def spy(x, y, z, pole, scale=1.0, *, gaps=None, **keywords):
        seen.update(x=x, y=y, z=z, pole=pole, gaps=gaps)
        return original(x, y, z, pole, scale, gaps=gaps, **keywords)

    incompleteelliptic.symmetric_kinds = spy
    try:
        incomplete_pole(np.asarray(poles), np.asarray(complements), sine, cosine)
    finally:
        incompleteelliptic.symmetric_kinds = original
    return seen


def _exact_gaps(pole, complement, co_amplitude=0.4):
    """Return the three gaps in longdouble, through the reflection the routine takes."""
    pole, complement = np.longdouble(pole), np.longdouble(complement)
    _, sine, _ = pair(co_amplitude)
    squared_sine = np.longdouble(sine) * np.longdouble(sine)
    partner = complement / pole if pole > 1.0 else pole
    return (
        partner * squared_sine,
        (partner - complement) * squared_sine,
        (partner - 1.0) * squared_sine,
    )


def test_the_gaps_are_carried_as_products_because_the_subtraction_has_no_digits():
    """The cancellation the carried gaps remove, measured where it happens.

    The pole argument the symmetric forms receive is
    ``cos^2 phi + partner sin^2 phi`` and one of the other three is ``cos^2 phi``
    itself, so the gap between them is ``partner sin^2 phi`` -- a PRODUCT of
    quantities the caller already holds, and exact.  Formed instead by subtracting
    the two, it keeps the gap only to the ulp of ``cos^2 phi``: three decades gone
    at a pole of 1e-14, and at a reflected pole whose partner is denormal the pole
    argument rounds onto ``cos^2 phi`` outright, so the subtraction returns zero
    where the product still carries its mantissa.  Nothing in that but
    round-to-nearest on a double, which is why it is asserted with a floor.

    Both arrangements are measured HERE, on the arguments the entry point itself
    formed, rather than through the assembled value, and that is deliberate.  The
    degenerate form is SECOND order in its root: ``R_C`` departs from ``1/s`` by
    ``root^2/3s^2``, and in the hyperbolic branch the ``root - difference`` under
    the logarithm is what cancels the log's own first-order term.  So a root
    overstated by eight decades -- which is what subtracting two roots that agree
    past their last bit gives -- still leaves the value at round-off, PROVIDED root
    and difference stay consistent with each other.  Carried, they are, because the
    difference is formed from the root.  Subtracted, they are two separately
    rounded derivations of a subtraction with no correct digits left, and a
    compiler free to recompute either one breaks the cancellation and lets those
    eight decades through: measured, the same kernel over the same poles strays
    1.2e-09 on one backend and holds 1.6e-15 on another.  The assembled comparison
    below therefore bounds the subtracted arrangement from ABOVE only, and the
    claim carrying a floor is the one about the gaps.
    """
    complements = [1e-300] * (len(GAP_POLES) + 1)
    poles = GAP_POLES + [DENORMAL_PARTNER_POLE]
    seen = _supplied_arguments(poles, complements)

    # the supply itself: a routine that stopped carrying the gaps leaves the
    # degenerate form to subtract its two roots, and there is nothing to measure
    assert seen["gaps"] is not None

    argument = np.asarray(seen["pole"])
    others = [np.broadcast_to(np.asarray(seen[name]), argument.shape) for name in "xyz"]
    for index, (pole, complement) in enumerate(zip(poles, complements)):
        exact = _exact_gaps(pole, complement)
        for which, other in enumerate(others):
            supplied = np.longdouble(seen["gaps"][which][index])
            subtracted = np.longdouble(argument[index] - other[index])
            carried_error = abs(supplied - exact[which]) / abs(exact[which])
            subtracted_error = abs(subtracted - exact[which]) / abs(exact[which])
            if pole == DENORMAL_PARTNER_POLE and which == 0:
                # the product is denormal, so its own mantissa is what limits it
                assert carried_error < 1e-11  # measured 4.4e-12
            else:
                assert carried_error < 1e-16  # measured 2.0e-18 to 5.6e-17
            if which == 2:
                continue  # the gap to z is of order one, and subtracts cleanly
            if pole == DENORMAL_PARTNER_POLE:
                # the pole argument rounds onto the other outright
                assert subtracted == 0.0
                assert supplied != 0.0
            else:
                assert subtracted_error > SUBTRACTED_GAP_FLOORS[index]

    # and the accuracy the products buy at the entry point, on both paths. The
    # subtracted arrangement is bounded from above alone: whether the loss above
    # reaches the value depends on how the kernel was compiled, not on the input.
    jax, jnp = traced_namespace()
    co_amplitude = 0.4
    _, sine, cosine = pair(co_amplitude)
    swept = np.array(GAP_POLES)
    complements = np.full_like(swept, 1e-300)

    @jax.jit
    def traced(pole, complement):
        return incomplete_pole(
            pole, complement, jnp.asarray(sine), jnp.asarray(cosine), xp=jnp
        )

    def strayed(got):
        return max(
            abs(got[index] - _pole_integral(co_amplitude, pole, complement))
            / abs(_pole_integral(co_amplitude, pole, complement))
            for index, (pole, complement) in enumerate(zip(swept, complements))
        )

    def compiled():
        traced.clear_cache()
        return np.asarray(traced(jnp.asarray(swept), jnp.asarray(complements)))

    host = np.asarray(incomplete_pole(swept, complements, sine, cosine))
    assert strayed(host) < 4e-15  # measured 1.3e-15
    assert strayed(compiled()) < 4e-15  # measured 1.2e-15 and 1.4e-15

    # the arrangement being held against, run so its cost is a measurement
    carried_kinds = incompleteelliptic.symmetric_kinds

    def without_gaps(*arguments, gaps=None, **keywords):
        return carried_kinds(*arguments, **keywords)

    incompleteelliptic.symmetric_kinds = without_gaps
    try:
        subtracted = strayed(compiled())
    finally:
        incompleteelliptic.symmetric_kinds = carried_kinds
    assert subtracted < 1e-7  # measured 1.6e-15 on one backend, 1.2e-09 on another


def test_the_third_kind_differentiates_where_a_gap_vanishes_exactly():
    """The pole argument landing ON one of the descent's other three.

    The gaps are supplied as products, so ``partner - k'^2`` is exactly zero when
    the two coincide rather than a rounding error from it, and the square root of a
    gap has an infinite derivative there.  The gap is therefore held at one and the
    product zeroed, which leaves the value at the elementary limit the confluence
    has and the derivative finite -- the root's own contribution is dropped, so it
    is finite rather than exact, and the nearby non-degenerate poles below pin that
    everywhere else it is the derivative itself.

    The confluence is reached only on the DIRECT side of the reflection, and that is
    geometry rather than a choice of sweep: the reflection runs above a pole of one,
    ``k'^2`` is a modulus complement and so never exceeds one, and the two can
    therefore not coincide above one at all.  Driven there ANYWAY -- with a
    ``complement`` above one, which no ring produces -- the reflected arrangement
    divides its three coefficients by ``pole - complement`` and returns nan for the
    value as well as for the slope; a hundredth of a part per trillion off it the
    value is an ordinary 0.6465.  So the sweep here stops below one because that is
    where the argument lives, not to avoid a corner.
    """
    jax, jnp = traced_namespace()
    _, sine, cosine = pair(0.4)

    def value(pole, complement):
        return incomplete_pole(
            pole, complement, jnp.asarray(sine), jnp.asarray(cosine), xp=jnp
        )

    slope = jax.jit(jax.grad(value, argnums=(0, 1)))
    # the confluence: the partner pole and the complement coincide exactly
    for pole in (1e-8, 1e-3, 0.3):
        for taken in slope(jnp.asarray(pole), jnp.asarray(pole)):
            assert np.isfinite(float(taken))

    # and away from it the slope is the derivative itself, against a central
    # difference -- taken where the difference is well conditioned, since a step
    # small enough to sit inside a tiny complement is all round-off
    for pole, complement in ((0.3, 0.1), (3.0, 0.1), (0.3, 1e-3)):
        taken = slope(jnp.asarray(pole), jnp.asarray(complement))
        for index, argument in enumerate((pole, complement)):
            step = 1e-5 * argument
            moved = [
                [pole, complement][other] + (step if other == index else 0.0)
                for other in (0, 1)
            ]
            backed = [
                [pole, complement][other] - (step if other == index else 0.0)
                for other in (0, 1)
            ]
            difference = (
                float(value(*(jnp.asarray(term) for term in moved)))
                - float(value(*(jnp.asarray(term) for term in backed)))
            ) / (2.0 * step)
            assert float(taken[index]) == pytest.approx(difference, rel=1e-7)


def test_the_third_kind_differentiates_forward_below_a_pole_of_one():
    """The unreflected side must survive FORWARD mode, not only reverse.

    The reflection's growth factor is zero wherever the pole sits below one, and
    a zero passed through its square root turns a zero TANGENT into nan -- while
    reverse mode survives, because the zero cotangent is discarded by the mask
    before it reaches the root.  Every pole a section's own denominators produce
    on the cn orientation is below one, so forward-mode geometry Jacobians hang
    on this case: the root's argument is held at one where no reflection is
    wanted and the result masked, and the slope is then the derivative itself in
    both modes.
    """
    jax, jnp = traced_namespace()
    _, sine, cosine = pair(0.4)

    def value(pole, complement):
        return incomplete_pole(
            pole, complement, jnp.asarray(sine), jnp.asarray(cosine), xp=jnp
        )

    for pole, complement in ((1e-8, 0.1), (0.3, 0.1), (3.0, 0.1)):
        forward = jax.jacfwd(value, argnums=(0, 1))(
            jnp.asarray(pole), jnp.asarray(complement)
        )
        reverse = jax.grad(value, argnums=(0, 1))(
            jnp.asarray(pole), jnp.asarray(complement)
        )
        for one, other in zip(forward, reverse):
            assert np.isfinite(float(one))
            assert float(one) == pytest.approx(float(other), rel=1e-12)


def test_the_slope_in_the_pole_holds_the_extended_reference_over_the_sweep():
    """The SLOPE against the derivative integral, not the two modes against each other.

    A geometry Jacobian is what the third kind is differentiated FOR, and the two
    modes agreeing does not establish that either is right -- below a pole of 1e-3
    they agree with each other an order of magnitude better than either agrees
    with the derivative itself.  So this holds both to :func:`_pole_slope`
    absolutely, across the small-pole side, exactly as the value's own sweep is
    held to :func:`_pole_integral`.

    What the sweep pins is the series the degenerate form leaves its
    transcendental for once the two roots are within a factor of ten --
    :data:`nova.biot.symmetricelliptic.SERIES_RATIO`.  ``arctan(v)/v`` is EVEN in
    ``v``, so through the series the slope flows into the gap product itself
    rather than into its square root, and nothing cancels; taken through the
    transcendental it loses a relative ``v^2``, and ``v^2`` falls with the pole.
    Measured against this reference with the transcendental alone: 3e-13 at a pole
    of 1e-04, 3e-09 at 1e-08 and 1.3e-05 at 1e-12, where the VALUE is a few ulp at
    every one of them -- so the tolerance here is what separates the two
    arrangements, and it is not a round number for that reason.

    The reflected side needs no series and is included to say so: above a pole of
    one the partner is the small quantity, and the same few ulp hold out to a
    partner of 1e-13 either way.
    """
    jax, jnp = traced_namespace()
    _, sine, cosine = pair(0.4)
    complement = 0.1

    def value(pole):
        return incomplete_pole(
            pole,
            jnp.asarray(complement),
            jnp.asarray(sine),
            jnp.asarray(cosine),
            xp=jnp,
        )

    forward = jax.jit(jax.jacfwd(value))
    reverse = jax.jit(jax.grad(value))
    for pole in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e4, 1e8, 1e12):
        expected = _pole_slope(0.4, pole, complement)
        coarse = _pole_slope(0.4, pole, complement, nodes=30)
        assert coarse == pytest.approx(expected, rel=1e-15)
        for taken in (forward(jnp.asarray(pole)), reverse(jnp.asarray(pole))):
            assert float(taken) == pytest.approx(expected, rel=1e-13)
