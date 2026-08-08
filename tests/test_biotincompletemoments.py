"""Contract for the harmonic moment families over a partial range.

Three families feed the polygon reduction and only one of them changes when the
range stops short of a quarter turn.  That claim is what these tests are mostly
about, because it is what decides how much of the shipped full-turn reduction an
arc can reuse:

* the plain moments gain a source term and a whole new stability argument, and
  are re-implemented here;
* the root moments and the pole moments are algebraic identities of the
  integrand, so :mod:`nova.biot.elliptic`'s own versions are checked AS THEY
  STAND against partial-range quadrature -- if either needed a counterpart, these
  tests are where it would show.

What the pole family DOES need is its SEED, and that is the one special function a
partial range adds: an incomplete integral of the third kind, at a pole argument
each orientation's shift supplies exactly.  The family's tests take it from
:func:`nova.biot.incompletemoments.cn_pole_moment` and its mirror rather than from
quadrature, so a defect in either reaches these tests as well as the seed's own.

Every reference is a panelled Gauss rule over the actual range rather than an
adaptive one: the integrands oscillate up to the tenth harmonic and carry a layer
of width ``k'`` at the far end, and a fixed panelled rule is the reference that
can be shown to have converged by refining it.  The rule is why the shifts swept
here stop at 1e-4: a root closer than that to the range start is a spike the
uniform panels do not resolve -- measured, the rule moves by 4e-06 between 1500
panels and 3000 at a shift of 1e-10 -- and the seed is held to the extended
reference of ``tests/test_biotincompleteelliptic.py`` over the rest.

The two constants the family is tuned by -- where it changes direction and how
far past the last wanted order it closes -- are asserted in both directions, so
starving either fails and padding either fails too.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from nova.biot.elliptic import cn_pole_moment as complete_cn_pole_moment
from nova.biot.elliptic import harmonic_moments as complete_harmonic_moments
from nova.biot.elliptic import harmonic_pole_moments, harmonic_root_moments
from nova.biot.elliptic import sn_pole_moment as complete_sn_pole_moment
from nova.biot.incompletemoments import (
    HEADROOM,
    SWITCH,
    cn_pole_moment,
    harmonic_moments,
    sn_pole_moment,
)
from nova.jax.config import enable_x64

# Held as the amplitude's distance BELOW a quarter turn, which is the arc's own
# half-separation from one of its ends, and the quantity the accuracy depends on.
# A quarter turn itself is absent by design: there the modulus radical's layer sits
# ON the range end, where a uniform panelled rule stops being a reference at all --
# and it needs none, the complete family being an exact one.
CO_AMPLITUDES = [1.5, 1.0, 0.7, 0.4, 0.2, 5e-2, 1e-2, 1e-3]

# k'^2 across the range a ring geometry reaches, dense where the family changes
# direction because that is where either route alone would fail.
COMPLEMENTS = [0.99, 0.9, 0.5, 0.1, 3e-2, 1e-2, 6e-3, 3e-3, 1e-3, 1e-6, 1e-9, 1e-16]

ORDERS = 10


def panelled(amplitude, integrand, panels=1500, nodes=48):
    """Return the integral over ``[0, amplitude]`` by a fixed panelled rule.

    Refinable, unlike an adaptive rule, so the reference can be shown to have
    converged rather than asserted to have; the accompanying convergence test
    halves the panels and checks the answer does not move.
    """
    node, weight = leggauss(nodes)
    edge = np.linspace(0.0, amplitude, panels + 1)
    half = 0.5 * (edge[1] - edge[0])
    angle = edge[:-1, None] + half * (node[None, :] + 1.0)
    return (half * (integrand(angle) @ weight)).sum()


def radical(angle, complement):
    """Return ``Delta``, built from the complement so it holds at the confluence."""
    return np.sqrt(np.cos(angle) ** 2 + complement * np.sin(angle) ** 2)


def reference_plain(amplitude, complement, order, **rule):
    """Return ``integral_0^a cos(2 n a)/Delta da``."""
    return panelled(
        amplitude,
        lambda angle: np.cos(2 * order * angle) / radical(angle, complement),
        **rule,
    )


def reference_root(amplitude, complement, order):
    """Return ``integral_0^a cos(2 n a) Delta da``."""
    return panelled(
        amplitude, lambda angle: np.cos(2 * order * angle) * radical(angle, complement)
    )


def reference_pole(amplitude, complement, shift, order, mirrored):
    """Return the pole family's own defining integral at one order."""

    def integrand(angle):
        base = np.sin(angle) ** 2 if mirrored else np.cos(angle) ** 2
        return np.cos(2 * order * angle) / ((base + shift) * radical(angle, complement))

    return panelled(amplitude, integrand)


def pole_seed(co_amplitude, complement, shift, mirrored):
    """Return the family's seed from the routine, in whichever orientation."""
    seed = sn_pole_moment if mirrored else cn_pole_moment
    return seed(
        np.asarray(shift),
        1.0 - complement,
        np.cos(co_amplitude),
        np.sin(co_amplitude),
        complement=np.asarray(complement),
    )


def family(co_amplitude, complement, count=ORDERS, **kwargs):
    """Return the partial moments, with the amplitude's pair formed exactly."""
    amplitude = 0.5 * np.pi - co_amplitude
    return harmonic_moments(
        amplitude,
        1.0 - complement,
        count,
        complement=complement,
        sine=np.cos(co_amplitude),
        cosine=np.sin(co_amplitude),
        **kwargs,
    )


@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_quarter_turn_reproduces_the_complete_family(complement):
    """The limit in which the arc closes onto the ring the full turn evaluates.

    Exact rather than close: the source term is generated from the amplitude's own
    pair, so at a quarter turn its seed ``2 sin a cos a`` is exactly zero, every
    harmonic after it is exactly zero, and the system the arc solves IS the
    homogeneous one the complete family satisfies.
    """
    got = family(0.0, complement, count=12)
    expected = complete_harmonic_moments(1.0 - complement, 12, complement=complement)
    scale = float(expected[0])
    for one, other in zip(got, expected):
        assert abs(float(one) - float(other)) < 4e-14 * max(scale, 1.0)


@pytest.mark.parametrize("co_amplitude", CO_AMPLITUDES)
@pytest.mark.parametrize("complement", COMPLEMENTS)
def test_the_plain_moments_reproduce_their_defining_integral(co_amplitude, complement):
    """Over amplitude and parameter, every order, against panelled quadrature.

    The tolerance is relative to the ZEROTH moment, not to each order's own value.
    Only absolute accuracy of that size is wanted -- the high harmonics are small
    and multiply coefficients no larger than the low ones, so what matters is what
    the contraction sees.
    """
    amplitude = 0.5 * np.pi - co_amplitude
    got = family(co_amplitude, complement)
    scale = abs(float(got[0]))
    for order in range(ORDERS):
        expected = reference_plain(amplitude, complement, order)
        assert abs(float(got[order]) - expected) < 5e-14 * scale


@pytest.mark.parametrize("co_amplitude", [1.0, 0.4, 5e-2])
@pytest.mark.parametrize("complement", [0.9, 1e-2, 1e-9])
def test_the_root_moments_need_no_counterpart(co_amplitude, complement):
    """``Delta^2`` folded back is an identity of the integrand, not of the range.

    So the shipped full-turn routine is correct at a partial limit as it stands,
    handed partial plain moments.  This is the test that would fail if it were
    not, and it is worth having precisely because the claim is easy to assume.
    """
    amplitude = 0.5 * np.pi - co_amplitude
    plain = family(co_amplitude, complement, count=ORDERS + 1)
    got = harmonic_root_moments(plain, 1.0 - complement)
    scale = abs(float(plain[0]))
    for order in range(ORDERS):
        expected = reference_root(amplitude, complement, order)
        assert abs(float(got[order]) - expected) < 5e-14 * scale


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("shift", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("co_amplitude", [1.0, 0.4, 5e-2])
def test_the_pole_family_needs_no_counterpart_either(mirrored, shift, co_amplitude):
    """Multiplying the pole factor back is also an identity of the integrand.

    Its CLOSURE is the part that needed re-arguing: the wanted solution no longer
    decays at a partial limit, but the closure error still does, at
    ``(1 + 2 shift) - sqrt((1 + 2 shift)^2 - 1)`` per order.  The shifts here are
    the ones a caller actually routes through this family -- a root far enough
    past the range for that decay to bite -- and the shipped headroom is what
    makes it hold.

    The seed comes from the routine, not from quadrature: it is the incomplete
    integral of the THIRD kind, and with it in place the whole moment stack the
    arc contracts against is closed.
    """
    complement = 1e-3
    amplitude = 0.5 * np.pi - co_amplitude
    plain = family(co_amplitude, complement, count=ORDERS + 40)
    seed = float(pole_seed(co_amplitude, complement, shift, mirrored))
    got = harmonic_pole_moments(
        np.asarray(shift), np.asarray(seed), plain, ORDERS, mirrored=mirrored
    )
    scale = abs(seed)
    for order in range(ORDERS):
        expected = reference_pole(amplitude, complement, shift, order, mirrored)
        assert abs(float(got[order]) - expected) < 5e-13 * scale


def test_the_reference_rule_has_converged():
    """Halving the panels must not move it, or nothing above means anything."""
    amplitude = 0.5 * np.pi - 1e-2
    for complement in (0.5, 1e-9):
        for order in (0, 5, 9):
            coarse = reference_plain(amplitude, complement, order, panels=750)
            fine = reference_plain(amplitude, complement, order, panels=1500)
            assert abs(coarse - fine) < 1e-15 * max(abs(fine), 1.0)


def test_the_switch_between_the_two_directions_is_needed_from_both_sides():
    """Each direction must fail where the other is used, or one would do.

    The solve stops contracting as the modulus reaches one and the recursion stops
    growing at the same point.  Forcing either to cover the whole range is what
    demonstrates that the pair is not redundant.
    """
    amplitude = 0.5 * np.pi - 0.2
    distant, near = 0.5, 1e-6

    def forced(complement, switch):
        return harmonic_moments(
            amplitude,
            1.0 - complement,
            ORDERS,
            complement=complement,
            sine=np.cos(0.2),
            cosine=np.sin(0.2),
            switch=switch,
        )

    for complement, wrong in ((distant, 0.0), (near, 1.0)):
        held = forced(complement, SWITCH)
        strayed = forced(complement, wrong)
        scale = abs(float(held[0]))
        worst = max(
            abs(float(one) - reference_plain(amplitude, complement, order)) / scale
            for order, one in enumerate(held)
        )
        strayed_worst = max(
            abs(float(one) - reference_plain(amplitude, complement, order)) / scale
            for order, one in enumerate(strayed)
        )
        assert worst < 5e-14
        assert strayed_worst > 1e3 * max(worst, 1e-16)


def test_the_headroom_is_bounded_by_the_closure_in_both_directions():
    """Long enough that the closure has died, short enough that it has not to spare.

    The closure's decay is the system's contraction ratio, which reaches one as
    the modulus does -- so this constant is set by the switch and not by a generic
    halving, and starving it shows up first just below the switch.
    """
    co_amplitude, complement = 5e-2, 1e-2
    amplitude = 0.5 * np.pi - co_amplitude

    def worst(headroom):
        got = family(co_amplitude, complement, headroom=headroom)
        scale = abs(float(got[0]))
        return max(
            abs(float(one) - reference_plain(amplitude, complement, order)) / scale
            for order, one in enumerate(got)
        )

    assert worst(HEADROOM) < 5e-14
    assert worst(HEADROOM + 40) == pytest.approx(worst(HEADROOM), abs=5e-14)
    # forty orders short costs two decades here, which is the closure showing
    # through rather than round-off, and is what pins the constant from below
    assert worst(HEADROOM - 40) > 1e-12


def test_the_source_term_is_what_the_partial_range_adds():
    """Drop it and the family is wrong; it is the whole difference from the ring.

    Stated as a measurement rather than a claim: the homogeneous system is what the
    complete family solves, and solving it at an interior amplitude returns the
    ring's moments instead of the arc's.
    """
    co_amplitude, complement = 0.7, 1e-3
    amplitude = 0.5 * np.pi - co_amplitude
    got = family(co_amplitude, complement)
    ring = complete_harmonic_moments(1.0 - complement, ORDERS, complement=complement)
    scale = abs(float(got[0]))
    assert (
        abs(float(got[0]) - reference_plain(amplitude, complement, 0)) < 5e-14 * scale
    )
    assert abs(float(ring[0]) - reference_plain(amplitude, complement, 0)) > 0.5 * scale


def test_the_traced_path_is_the_same_code_and_agrees_to_a_few_ulp():
    """One implementation, numpy on the host and a compiled kernel on a device."""
    jax = pytest.importorskip("jax")
    enable_x64()
    jnp = jax.numpy
    co_amplitude = 0.4
    complements = np.array([0.9, 1e-2, 1e-6, 1e-16])

    @jax.jit
    def traced(complement):
        return harmonic_moments(
            jnp.asarray(0.5 * np.pi - co_amplitude),
            1.0 - complement,
            ORDERS,
            complement=complement,
            sine=jnp.asarray(np.cos(co_amplitude)),
            cosine=jnp.asarray(np.sin(co_amplitude)),
            xp=jnp,
        )

    host = harmonic_moments(
        np.full_like(complements, 0.5 * np.pi - co_amplitude),
        1.0 - complements,
        ORDERS,
        complement=complements,
        sine=np.cos(co_amplitude),
        cosine=np.sin(co_amplitude),
    )
    device = traced(jnp.asarray(complements))
    scale = np.abs(host[0])
    for one, other in zip(host, device):
        assert np.max(np.abs(np.asarray(other) - one) / scale) < 1e-14


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("shift", [1e-4, 1e-2, 0.25, 1.0, 4.0, 1e4])
@pytest.mark.parametrize("co_amplitude", [1.0, 0.4, 5e-2])
def test_the_pole_seed_reproduces_its_defining_integral(mirrored, shift, co_amplitude):
    """The one special function the partial range adds, in both orientations.

    A root past the FAR end of the range is outside a partial one altogether, so
    that orientation is bounded whatever the shift; a root past the NEAR end is
    inside the layer the integral is concentrated in, because the range starts
    there, and it is the one every corner reaches.
    """
    complement = 1e-3
    amplitude = 0.5 * np.pi - co_amplitude
    got = float(pole_seed(co_amplitude, complement, shift, mirrored))
    expected = reference_pole(amplitude, complement, shift, 0, mirrored)
    assert got == pytest.approx(expected, rel=5e-14)


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("shift", [1e-12, 1e-4, 0.25, 4.0, 1e8])
@pytest.mark.parametrize("complement", [0.9, 1e-3, 1e-9, 1e-16])
def test_the_quarter_turn_reproduces_the_complete_seed(mirrored, shift, complement):
    """The limit in which the arc closes onto the ring, for the seed itself.

    Reached at every shift including the ones no panelled rule resolves, which is
    what makes it the check that spans the whole range: the complete routine is a
    Bartky descent and this is a Carlson duplication behind a reflection, so the
    two share no arithmetic at all.
    """
    seed = sn_pole_moment if mirrored else cn_pole_moment
    complete_seed = complete_sn_pole_moment if mirrored else complete_cn_pole_moment
    got = seed(
        np.asarray(shift),
        1.0 - complement,
        1.0,
        0.0,
        complement=np.asarray(complement),
    )
    expected = complete_seed(
        np.asarray(shift),
        1.0 - complement,
        parameter_complement=np.asarray(complement),
    )
    assert float(got) == pytest.approx(float(expected), rel=4e-15)


def test_the_seed_carries_the_family_where_quadrature_cannot():
    """A root a squared aspect ratio past the range start, which is the working case.

    The shift a slender section produces is far below anything a uniform panelled
    rule resolves, so this is checked against the seed's own growth instead: the
    near orientation's own moment is ``(pi/2)/sqrt(shift(1 + shift))`` as the shift
    vanishes, which is the elementary integral the modulus radical multiplies by
    one over the layer it is concentrated in.
    """
    complement = 1e-6
    for shift in (1e-8, 1e-10, 1e-12):
        got = float(pole_seed(0.4, complement, shift, True))
        # the radical is one to within k'^2 over the layer of width sqrt(shift)
        elementary = 0.5 * np.pi / np.sqrt(shift * (1.0 + shift))
        assert got == pytest.approx(elementary, rel=2e-4)
        assert got > elementary
