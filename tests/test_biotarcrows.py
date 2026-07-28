"""Conditioning contract for the finite arc's integration rows.

:class:`nova.biot.arc.Arc` is the ring reduction with its range stopped short, so
every elliptic integral it takes regains an amplitude and the arguments it hands
them are the ones :mod:`nova.biot.constants` already takes as exact geometric
complements.  Four quantities in the rows are reached either by subtracting from
one or by the geometry, and the two routes agree in exact arithmetic and not in
floats:

* the modulus radical ``dn^2 = 1 - k^2 sin^2 theta`` is
  ``cos^2 theta + k'^2 sin^2 theta``, and the printed form is additionally taken by
  INVERTING the first kind -- a numerical round trip whose answer is the amplitude's
  own sine and cosine;
* the pole gap ``k^2 - n`` is ``(1 - n) - k'^2``, a difference of two quantities
  that each vanish at a source corner rather than two of order one;
* the pole denominator ``1 - n sin^2 theta`` is
  ``cos^2 theta + (1 - n) sin^2 theta``, a sum of positives;
* the second kind's weight in the vertical row, ``2 r - b k^2``, is
  ``2 r (gamma^2 - b (rs - r))/a^2``.

and the first and second kinds themselves are taken from the modulus COMPLEMENT
rather than from the parameter, because the first grows like ``-log k'`` and no
parameter however exact carries ``1 - k^2`` to better than an absolute ``eps``.

So the tests come in pairs, as the constants' own do.  :class:`PrintedRows` holds
every one of those spellings in the arrangement that reaches its argument by
subtraction, and each conditioning test runs BOTH classes against the same
reference over the same approach -- because the size of the gap between the two
curves is what says the geometric arrangement is worth taking.

Three references, none of them the code under test:

* **the axisymmetric kernel.** An arc closed on itself is the complete ring, and
  the rows telescope onto ``-2 X(pi/2)`` -- the quarter-turn row, which is the row
  every odd row's fold picks up.  So :func:`nova.biot.greens.greens_psi` and
  :func:`nova.biot.greens.greens_bz_br` are exact oracles for the potential's
  azimuthal row and both field rows in that limit, and they are complement-native
  throughout with their own gate in ``test_biotgreens``.
* **the defining integrals in longdouble**, on the composite ``sinh`` map from
  ``test_biotincompleteelliptic``, which is where their convergence under
  refinement is asserted.  Imported rather than copied so the two cannot drift.
* **the same closed form at extended precision**, for the elementary coefficient
  and the two radicals, where the question is the arrangement rather than the
  function -- the pattern ``test_biotconstants`` uses for the radial row's weight.
"""

from functools import cached_property

import numpy as np
import pytest
import scipy.special

from nova.biot.arc import Arc
from nova.biot.constants import Constants
from nova.biot.greens import MU0, greens_bz_br, greens_psi
from nova.biot.incompleteelliptic import incomplete_pole
from tests.test_biotincompleteelliptic import (
    _pole_integral,
    extended_first_kind,
    extended_second_kind,
)

RADIUS = 6.2  # a metre-scale ring, where a face is centimetres from the axis

# The target's distance to the source, as a fraction of the ring radius.  The lower
# end is a target ON the conductor and the upper end is one clear of it.
RATIOS = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12])

# Where both spellings must agree, because the subtraction still has digits: half a
# radius clear of the source in both directions.
SOUND = 0.5

QUARTER_TURN = 0.5 * np.pi


class ArcRows(Arc):
    """The arc's rows over geometry supplied directly, without the frame plumbing.

    ``Arc.__post_init__`` reads the source and target frames to reach ``rs``, ``zs``,
    ``r`` and ``z``; those four are the dataclass' own fields, so supplying them and
    the folded limit stack ``alpha`` is the whole state the rows need.  What is
    measured is then the arguments the rows are assembled from rather than the frame
    machinery around them -- the same separation ``test_biotconstants`` makes for the
    complete filament.
    """

    def __post_init__(self):
        """Take the geometry as given."""


class PrintedRows(ArcRows):
    """Every row in the arrangement that reaches its argument by subtraction.

    One class per spelling rather than a flag, so a conditioning test can run the two
    over the same geometry and the same reference and the gap between them is the
    arrangement and nothing else.  Reverting any of the five arrangements in
    :mod:`nova.biot.arc` makes the shipped class agree with this one, and every
    assertion below that separates them then fails.
    """

    @cached_property
    def ellipj(self):
        """Return the Jacobian functions by INVERTING the first kind."""
        return dict(
            zip(["sn", "cn", "dn", "ph"], scipy.special.ellipj(self.Kinc, self.k2))
        )

    @cached_property
    def Kinc(self):
        """Return the first kind from the parameter, through Cephes."""
        return self.ellipkinc(self.theta, self.k2)

    @cached_property
    def Einc(self):
        """Return the second kind from the parameter, through Cephes."""
        return self.ellipeinc(self.theta, self.k2)

    @cached_property
    def Pi_inc(self) -> dict[int, np.ndarray]:
        """Return the third kind from the characteristic and the parameter alone."""
        return {
            p: self.ellippinc(self.np2[p], self.theta, self.k2) for p in range(1, 4)
        }

    def _pole_gap(self, p: int) -> np.ndarray:
        """Return ``k^2 - n`` as the printed difference."""
        return self.k2 - self.np2[p]

    def _pole_denominator(self, p: int) -> np.ndarray:
        """Return ``1 - n sin^2 theta`` as the printed difference.

        The amplitude's sine directly rather than through the Jacobian round trip the
        printed row took it from, so this isolates the subtraction from the radical's
        own arrangement -- the two are the same quantity wherever the round trip
        returns anything at all.
        """
        return 1 - self.np2[p] * np.sin(self.theta) ** 2

    @property
    def _Bz_hat(self):
        """Return the vertical row with the weight taken from the modulus."""
        Bz_hat = (
            self.sign_theta
            * (
                self.r * self.ck2 * self.Kinc
                - (self.r - self.b * self.k2 / 2) * self.Winc
            )
        ) / self.rack2
        return self.sign_alpha * self._exterior(Bz_hat)


def arc_rows(rs, gamma, r, limits, rows=ArcRows):
    """Return an :class:`ArcRows` over broadcast geometry and a limit stack.

    ``limits`` is the leading axis the reduction evaluates its antiderivative on:
    the two arc ENDS first and then the two integration limits, a vanishing
    amplitude and a quarter turn, which is the order every fold and every
    ``_exterior`` operator here assumes.
    """
    geometry = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(term, dtype=float)) for term in (rs, gamma, r))
    )
    rs, gamma, r = (term.reshape(-1, 1).copy() for term in geometry)
    element = rows(rs=rs, zs=gamma, r=r, z=np.zeros_like(r))
    element.__dict__["alpha"] = np.stack(
        [np.full_like(r, limit) for limit in limits + [0.0, QUARTER_TURN]]
    )
    element.__dict__["_phi"] = np.zeros_like(r)
    return element


def approaching_the_filament(ratios, rows=ArcRows, limits=None):
    """Return an arc whose target is driven onto its own filament, diagonally.

    Both the standoff and the radial gap shrink together, so the modulus complement
    falls as the SQUARE of the distance and both field rows diverge.  ``limits``
    defaults to the closed arc: an end at one radian and the other a half turn from
    it, which is the configuration in which the reduction is the complete ring.
    """
    gap = np.atleast_1d(ratios * RADIUS / np.sqrt(2.0)).reshape(-1, 1)
    limits = [1.0, 1.0 - np.pi] if limits is None else limits
    return arc_rows(RADIUS, gap, RADIUS + gap, limits, rows=rows), gap


def on_the_end_plane(ratios, rows=ArcRows, co_amplitude=0.0):
    """Return an arc whose target sits on its own end plane, at the same approach.

    A target on the end plane puts the amplitude at a quarter turn, where the sine's
    square is one and nothing dilutes a pole or a characteristic formed by
    subtraction.  ``co_amplitude`` steps the first end back off the plane; the
    quarter-turn LIMIT row is a right angle by construction either way.
    """
    return approaching_the_filament(
        ratios, rows=rows, limits=[QUARTER_TURN - co_amplitude, 0.3]
    )


def extended(element):
    """Return ``(k'^2, k^2, {1 - n})`` in longdouble, from the geometry.

    Spelled out here term for term against the identities rather than read off the
    class, so the reference for an arrangement question is independent of the
    arrangement: ``r - c = -gamma^2/(r + c)`` for ``c = sqrt(gamma^2 + r^2)``, and
    the two radicals ``a^2`` and ``a^2 - 4 r rs`` differ by exactly ``4 r rs``.
    """
    rs, gamma, r = (
        np.longdouble(term) for term in (element.rs, element.gamma, element.r)
    )
    b, c = rs + r, np.sqrt(gamma**2 + r**2)
    a2 = gamma**2 + b**2
    return (
        (gamma**2 + (rs - r) ** 2) / a2,
        4 * r * rs / a2,
        {
            1: ((r + c) / gamma) ** 2,
            2: (gamma / (r + c)) ** 2,
            3: ((rs - r) / b) ** 2,
        },
    )


def extended_pair(element):
    """Return the amplitude's ``(sine, cosine)`` in longdouble, exact where it is.

    Spelled out here rather than promoted from the class, so the reference does not
    inherit the arrangement it checks.  An amplitude on a right angle has a cosine of
    exactly zero -- the last limit row by construction, and any arc end whose target
    sits on that end's own plane -- and ``cos`` of a float right angle is 6.1e-17,
    which at a small modulus complement is the larger of the two terms under the
    radical.
    """
    amplitude = np.longdouble(element.theta)
    right_angle = element.theta >= QUARTER_TURN
    return (
        np.where(right_angle, np.longdouble(1.0), np.sin(amplitude)),
        np.where(right_angle, np.longdouble(0.0), np.cos(amplitude)),
    )


def extended_radical(element):
    """Return ``dn`` in longdouble, as the sum of positives it identically is."""
    complement, _, _ = extended(element)
    sine, cosine = extended_pair(element)
    return np.sqrt(cosine**2 + complement * sine**2)


def extended_pole_coefficient(element, p):
    """Return the elementary coefficient ``I(n)`` in longdouble, branch by branch.

    Every difference is taken in the arrangement that does not cancel -- the pole gap
    as a difference of poles, the denominator as a sum of positives, and the far
    root's logarithm through
    ``(k^2 - n) - |n| dn^2 = k^2 (cos^2 theta + (1 - n) sin^2 theta)`` -- so the
    reference is the closed form at extended precision and not the printed
    arrangement at extended precision, which loses the same digits the float one
    does.
    """
    complement, parameter, poles = extended(element)
    pole = poles[p]
    characteristic = 1.0 - pole
    sine, cosine = extended_pair(element)
    radical = np.sqrt(cosine**2 + complement * sine**2)
    denominator = cosine**2 + pole * sine**2
    gap = pole - complement
    if p == 1:  # the root past the near end of the range: n <= 0
        span = np.sqrt(gap) + np.sqrt(-characteristic) * radical
        return (
            -np.sqrt(-characteristic)
            / (2 * np.sqrt(gap))
            * np.log(parameter**2 * denominator / span**2)
        )
    if np.all(gap < 0):  # n > k^2
        span = np.sqrt(-gap) + np.sqrt(characteristic) * radical
        return (
            np.sqrt(characteristic)
            / (2 * np.sqrt(-gap))
            * np.log(span**2 / denominator)
        )
    # one branch for the whole sweep, so the sign of the gap has to be uniform over
    # it -- a mixed sweep would take the wrong closed form for half its elements
    assert np.all(gap > 0)  # n < k^2
    return (
        -np.sqrt(characteristic)
        / (2 * np.sqrt(gap))
        * np.arcsin(
            2
            * np.sqrt(characteristic)
            * radical
            * np.sqrt(gap)
            / (parameter * abs(denominator))
        )
    )


def ring(element, gap):
    """Return ``(psi, Br, Bz)`` for the closed arc, and the kernel's own values.

    The arc's two ends a half turn apart is the complete ring, so what the rows
    assemble to is the axisymmetric kernel -- which is where the reference comes
    from and why it is independent of every argument the arc's rows form.
    """
    filament = (element.r, np.zeros_like(gap), np.full_like(gap, RADIUS), gap)
    want_bz, want_br = greens_bz_br(*filament)
    return (
        (
            2.0 * np.pi * MU0 * element.r * element._intergrate(element._Aphi_hat),
            MU0 * element._intergrate(element._Br_hat),
            MU0 * element._intergrate(element._Bz_hat),
        ),
        (greens_psi(*filament), want_br, want_bz),
    )


def worst(got, want):
    """Return the largest relative departure of ``got`` from ``want``."""
    return float(np.max(abs(np.asarray(got, dtype=float) / np.float64(want) - 1.0)))


# ---------------------------------------------------------------------------
# The Jacobian functions, which are the amplitude's own pair and one radical.


def test_the_jacobian_pair_is_the_amplitude_s_own_sine_and_cosine():
    """``ellipj(F(theta, m), m)`` returns ``(sin theta, cos theta)`` by definition.

    So the inversion is a numerical round trip with an exact answer, and this is the
    check that the two are the same thing: run over geometry well clear of the source
    ring, where the round trip is sound and the agreement is therefore a statement
    about the identity rather than about either party's conditioning.
    """
    clear = np.array([SOUND, 0.2, 0.05])
    element, _ = approaching_the_filament(clear)
    printed, _ = approaching_the_filament(clear, rows=PrintedRows)
    sine, cosine = element.theta_pair
    assert np.max(abs(printed.ellipj["sn"] - sine)) < 8e-16  # measured 2.2e-16
    assert np.max(abs(printed.ellipj["cn"] - cosine)) < 8e-16  # measured 2.2e-16
    assert np.max(abs(printed.ellipj["ph"] - element.theta)) < 8e-16  # measured 3.3e-16


def test_the_round_trip_returns_nothing_once_the_parameter_rounds_to_one():
    """A target within 1e-08 radii of the ring, which is a grid node on a face.

    The parameter is ``4 r rs/a^2`` and reaches one exactly there, so the complete
    first kind the round trip inverts is infinite and every Jacobian function comes
    back ``nan`` -- not a lost decimal but the whole reduction.  The amplitude's own
    pair is elementary and the radical is a sum of positives, so all three stay
    finite on the same geometry.
    """
    element, _ = approaching_the_filament(RATIOS[RATIOS <= 1e-8])
    printed, _ = approaching_the_filament(RATIOS[RATIOS <= 1e-8], rows=PrintedRows)
    for name in ("sn", "cn", "dn"):
        # the quarter-turn row, whose first kind is the complete one and infinite
        assert np.all(np.isnan(printed.ellipj[name][-1]))
        assert np.all(np.isfinite(element.ellipj[name]))


@pytest.mark.parametrize("co_amplitude", [0.0, 1e-4])
def test_the_modulus_radical_is_a_sum_of_positives_at_the_end_plane(co_amplitude):
    """What the two printed radicals cost, measured against the same reference.

    ``dn`` is ``sqrt(1 - k^2 sin^2 theta)`` printed, and both terms are one to every
    digit at a quarter-turn amplitude and a target on the source ring -- so the
    subtraction returns the modulus complement with whatever digits the parameter
    had, and the round trip through the first kind returns even less.  The geometric
    form is the same quantity with no subtraction in it.
    """
    element, _ = on_the_end_plane(RATIOS, co_amplitude=co_amplitude)
    printed, _ = on_the_end_plane(RATIOS, rows=PrintedRows, co_amplitude=co_amplitude)
    want = extended_radical(element)
    sine = element.theta_pair[0]
    subtracted = np.sqrt(1.0 - element.k2 * sine**2)
    assert not np.all(np.isfinite(printed.ellipj["dn"]))  # the round trip, at the ring
    # the subtraction returns exactly zero once the parameter rounds to one, so the
    # radical has no digits at all rather than fewer of them
    assert worst(subtracted, want) == 1.0
    assert worst(element.ellipj["dn"], want) < 4e-16  # measured 2.2e-16


def test_the_rows_the_radical_divides_carry_whatever_it_loses():
    """The radial potential row and the toroidal field row, end to end.

    Both are the radical times a factor the geometry supplies exactly -- ``a/r`` and
    ``-gamma k'^2/(r a k'^2)`` -- so the row's relative error IS the radical's, which
    is the point: there is no small weight anywhere to damp it.  Run at a target on
    the end plane, where the printed radical is at its worst.
    """
    element, _ = on_the_end_plane(RATIOS)
    printed, _ = on_the_end_plane(RATIOS, rows=PrintedRows)
    radical = extended_radical(element)
    want_radial = np.longdouble(element.a) / np.longdouble(element.r) * radical
    want_toroidal = (
        -np.longdouble(element.gamma)
        * np.longdouble(element.ck2)
        / radical
        / np.longdouble(element.rack2)
    )
    sine = element.theta_pair[0]
    subtracted = np.sqrt(1.0 - element.k2 * sine**2)
    assert worst(element.a / element.r * subtracted, want_radial) == 1.0
    assert worst(element._Ar_hat, want_radial) < 4e-16  # measured 2.2e-16
    assert worst(element._Bphi_hat, want_toroidal) < 4e-16  # measured 2.2e-16
    assert not np.all(np.isfinite(printed._Ar_hat))
    assert not np.all(np.isfinite(printed._Bphi_hat))


# ---------------------------------------------------------------------------
# The first and second kinds, from the complement rather than the parameter.


@pytest.mark.parametrize("co_amplitude", [0.7, 1e-2, 1e-4, 1e-8, 0.0])
def test_the_first_and_second_kinds_hold_at_every_amplitude(co_amplitude):
    """Against their defining integrals, over the whole approach to the ring.

    The first kind grows like ``-log k'``, so an absolute ``eps`` on the parameter is
    a relative ``eps/k'^2`` on the answer -- and Cephes re-derives ``1 - m``
    internally, so supplying the parameter exactly buys a factor of six and no more.
    The second kind is bounded and holds either way, which is why only one of the two
    separates the routes.
    """
    element, _ = on_the_end_plane(RATIOS, co_amplitude=co_amplitude)
    complement, _, _ = extended(element)
    want_first = np.array(
        [float(extended_first_kind(co_amplitude, term)) for term in complement.ravel()]
    )
    want_second = np.array(
        [float(extended_second_kind(co_amplitude, term)) for term in complement.ravel()]
    )
    # 2.5e-12 is the amplitude's own accuracy showing through, at a co-amplitude of
    # 1e-08 and a target 1e-08 radii off the ring: alpha is assembled as (pi - psi)/2
    # and the separation's digits are gone before any row sees the angle
    assert worst(element.Kinc[0].ravel(), want_first) < 1e-11  # measured 2.5e-12
    assert worst(element.Einc[0].ravel(), want_second) < 1e-14  # measured 2.6e-15
    if co_amplitude < 1e-3:  # where the parameter's own eps reaches the answer
        # only over the approach the parameter route can be evaluated on at all
        sound = RATIOS >= 1e-6
        printed, _ = on_the_end_plane(
            RATIOS[sound], rows=PrintedRows, co_amplitude=co_amplitude
        )
        assert worst(printed.Kinc[0].ravel(), want_first[sound]) > 1e-11  # 3.9e-10 up


def test_the_quarter_turn_limit_row_is_a_right_angle_by_construction():
    """Its cosine is exactly zero, and a float right angle's cosine is 6.1e-17.

    That row is the value every odd row's fold picks up, so what it loses the whole
    reduction inherits.  Both spellings run against the complete routine the ring
    already uses, which is the limit the arc closes onto.
    """
    element, _ = approaching_the_filament(RATIOS)
    sine, cosine = element.theta_pair
    assert np.array_equal(cosine[-1], np.zeros_like(cosine[-1]))
    assert np.array_equal(sine[-1], np.ones_like(sine[-1]))
    complement, _, _ = extended(element)
    want = np.array(
        [float(extended_first_kind(0.0, term)) for term in complement.ravel()]
    )
    from nova.biot.incompleteelliptic import incomplete_kind

    assembled, _ = incomplete_kind(element.theta[-1], element.ck2, parameter=element.k2)
    assert worst(assembled.ravel(), want) > 1e-08  # measured 4.1e-06
    assert worst(element.Kinc[-1].ravel(), want) < 1e-14  # measured 4.4e-16


def test_the_end_edge_corner_is_a_target_within_picometres_of_the_end_edge():
    """The one corner the descent does not cover, sized against the geometry.

    Its boundary -- an amplitude within 1e-07 of a quarter turn AND a complement
    below 1e-25 -- is measured in ``test_biotincompleteelliptic``.  For a metre-scale
    ring the complement is the squared distance to the filament over the squared ring
    span, so 1e-25 is a target four PICOMETRES off it, and the whole sweep here stops
    an order of magnitude short of that.  The quarter-turn limit row is exempt at any
    complement, because its cosine is exactly zero rather than nearly so.
    """
    element, _ = approaching_the_filament(RATIOS)
    complement, _, _ = extended(element)
    assert np.min(complement) > 1e-25  # measured 2.5e-25 at the sweep's own end
    corner = 2.0 * RADIUS * np.sqrt(1e-25)  # k'^2 = (distance/a)^2, and a is 2 r here
    assert corner < 1e-11  # measured 3.9e-12 metres off a 6.2 m filament
    element, _ = on_the_end_plane(np.array([1e-13]), co_amplitude=1e-9)
    assert np.all(np.isfinite(element.Kinc))


# ---------------------------------------------------------------------------
# The rows the closed arc assembles, against the axisymmetric kernel.


def test_the_closed_arc_reproduces_the_axisymmetric_kernel():
    """The arc's two ends a half turn apart, driven onto its own filament.

    Every interior contribution telescopes between the two ends and what is left is
    the quarter-turn row, so the kernel pins the potential's azimuthal row and both
    field rows at once -- through the first and second kinds, the composite second
    kind, the radical inside it and the vertical row's own weight.  The kernel is
    complement-native and carries the divergence rather than a standoff, so it is the
    reference the whole way in.
    """
    element, gap = approaching_the_filament(RATIOS)
    (psi, br, bz), (want_psi, want_br, want_bz) = ring(element, gap)
    assert psi == pytest.approx(want_psi, rel=8e-15, abs=0)  # measured 7.8e-16
    assert br == pytest.approx(want_br, rel=8e-15, abs=0)  # measured 1.8e-15
    assert bz == pytest.approx(want_bz, rel=8e-15, abs=0)  # measured 1.9e-15


def test_the_parameter_route_loses_the_closed_arc_and_then_returns_nothing():
    """What the first kind from the parameter costs the assembled rows, measured.

    The flux is the row that shows it plainly: the parameter route is 1.9e-05 out a
    micrometre off a metre-scale filament and infinite a hundredth of that, where the
    complement route is on round-off the whole way.  Both fields go ``nan`` at the
    same place, through the Jacobian round trip the same class takes.
    """
    printed, gap = approaching_the_filament(RATIOS, rows=PrintedRows)
    (psi, br, bz), (want_psi, want_br, want_bz) = ring(printed, gap)
    finite = RATIOS >= 1e-6
    assert worst(psi[finite], want_psi[finite]) > 1e-06  # measured 1.9e-05
    assert worst(br[finite], want_br[finite]) > 1e-11  # measured 1.2e-10
    assert not np.any(np.isfinite(psi[~finite]))
    assert not np.any(np.isfinite(br[~finite]))
    assert not np.any(np.isfinite(bz[~finite]))


def test_the_vertical_row_weight_cancels_where_it_is_taken_from_the_modulus():
    """``2 r - b k^2`` reaches zero as the target reaches the ring, measured.

    Two quantities of order the ring span coming back at order the distance to it,
    and the difference stands over the modulus complement in the row it weights.
    Isolated from the first kind's own arrangement by running the printed weight
    inside the shipped class, so the only thing that moves between the two curves is
    the weight.
    """

    class PrintedWeight(ArcRows):
        """The shipped rows with the vertical weight taken from the modulus."""

        _Bz_hat = PrintedRows._Bz_hat

    element, gap = approaching_the_filament(RATIOS)
    printed, _ = approaching_the_filament(RATIOS, rows=PrintedWeight)
    _, (_, _, want_bz) = ring(element, gap)
    assert worst(MU0 * printed._intergrate(printed._Bz_hat), want_bz) > 1e-09  # 7.1e-09
    assert worst(MU0 * element._intergrate(element._Bz_hat), want_bz) < 8e-15  # 1.9e-15


# ---------------------------------------------------------------------------
# The pole gap and the pole denominator, which the elementary coefficient divides.


def test_the_pole_gap_and_denominator_agree_with_the_subtraction_off_the_source():
    """One algebra, not two: a target well clear of the source separates neither."""
    element = arc_rows(RADIUS * (1.0 + SOUND), RADIUS * SOUND, RADIUS, [1.0, 0.3])
    printed = arc_rows(
        RADIUS * (1.0 + SOUND), RADIUS * SOUND, RADIUS, [1.0, 0.3], rows=PrintedRows
    )
    for p in (1, 2, 3):
        assert element._pole_gap(p) == pytest.approx(printed._pole_gap(p), rel=1e-11)
        assert element._pole_denominator(p) == pytest.approx(
            printed._pole_denominator(p), rel=1e-11
        )


@pytest.mark.parametrize("p", [1, 2, 3])
def test_the_pole_gap_is_a_difference_of_poles_at_the_end_plane(p):
    """``k^2 - n`` as ``(1 - n) - k'^2``, whose two terms both vanish at a corner.

    Taken from the parameter and the characteristic, both terms are of order one and
    the difference is known only to an absolute ``eps`` -- while the gap itself falls
    with the squared distance to the source for two of the three characteristics.  It
    divides every branch of the elementary coefficient.
    """
    element, _ = on_the_end_plane(RATIOS)
    printed, _ = on_the_end_plane(RATIOS, rows=PrintedRows)
    complement, _, poles = extended(element)
    want = poles[p] - complement
    assert worst(element._pole_gap(p), want) < 8e-16  # measured 6.7e-16
    if p != 1:  # the far root's pole reaches 1e20 and the subtraction survives it
        # the printed difference returns exactly zero at the sweep's own end
        assert worst(printed._pole_gap(p), want) == 1.0


@pytest.mark.parametrize("p", [2, 3])
def test_the_pole_denominator_stays_positive_at_the_end_plane(p):
    """``1 - n sin^2 theta`` at a quarter turn is the characteristic's complement.

    The sine's square is one there, so nothing is left to dilute the subtraction --
    and for a characteristic within an ``eps`` of one the printed form reaches zero
    and goes NEGATIVE, which puts the logarithms it sits under at ``nan``.  Both
    bounded characteristics reach one as the target reaches the ring.
    """
    element, _ = on_the_end_plane(RATIOS)
    printed, _ = on_the_end_plane(RATIOS, rows=PrintedRows)
    _, _, poles = extended(element)
    sine, cosine = extended_pair(element)
    want = cosine**2 + poles[p] * sine**2
    assert np.all(element._pole_denominator(p) > 0.0)
    assert worst(element._pole_denominator(p), want) < 8e-16  # measured 2.2e-16
    assert np.min(printed._pole_denominator(p)) == 0.0  # and one ulp the other way
    assert worst(printed._pole_denominator(p)[0], want[0]) == 1.0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_the_elementary_coefficient_holds_at_the_end_plane(p):
    """The coefficient the section rows weight, in both arrangements.

    Its reference is the same closed form at extended precision, because what is
    under test is the arrangement of three differences rather than the function -- so
    the reference takes each of them the way that does not cancel.

    The bound is 1e-09 rather than round-off because ONE loss is left in the closed
    form and is not one of the differences under test: the logarithm's argument
    approaches one at a quarter-turn amplitude, where the denominator and the squared
    span are both the pole, so the logarithm returns a value of order ``k'`` out of an
    argument that differs from one by the same amount.  It sits in both arrangements
    identically, which is what
    :func:`test_the_logarithm_of_a_near_unit_argument_is_the_loss_that_is_left`
    measures -- and it is four decades better than what the two subtractions cost.
    """
    ratios = RATIOS[RATIOS >= 1e-6]
    element, _ = on_the_end_plane(ratios)
    printed, _ = on_the_end_plane(ratios, rows=PrintedRows)
    want = extended_pole_coefficient(element, p)
    assert worst(element.Ip[p], want) < 1e-09  # measured 4.3e-10
    end_plane = [0, -1]  # the two rows whose amplitude is a quarter turn
    # measured 1.3e-04 to 4.0e-04 over the three roots
    assert worst(printed.Ip[p][end_plane], want[end_plane]) > 1e-05


def test_the_far_root_s_logarithm_collapses_where_it_is_taken_as_a_difference():
    """``sqrt(k^2 - n) - sqrt(|n|) dn`` agrees to every digit at a far root.

    The gap is ``|n|`` plus ``k^2`` and the two roots then differ in the last bits,
    so the printed difference returns exactly zero past a pole of 1e20 and the
    logarithm of its square is ``-inf``.  Their product with their sum is
    ``k^2 (cos^2 theta + (1 - n) sin^2 theta)`` identically, which is the denominator
    the ratio divides by, so the quotient is one of positives.  Reached by a target a
    tenth of a nanometre off the plane of a source corner -- the first characteristic
    diverges as ``gamma^-2``.
    """
    element = arc_rows(RADIUS * 1.001, RADIUS * 1e-11, RADIUS, [1.0, 0.3])
    gap, radical = element._pole_gap(1), element.ellipj["dn"]
    far = np.sqrt(abs(element.np2[1]))
    difference = np.sqrt(gap) - far * radical
    span = np.sqrt(gap) + far * radical
    quotient = element.k2 * element._pole_denominator(1) / span
    assert np.any(difference == 0.0)  # the zero-amplitude limit row, where dn is one
    assert np.all(quotient > 0.0)
    assert np.all(np.isfinite(element.Ip[1]))
    want = extended_pole_coefficient(element, 1)
    assert worst(element.Ip[1], want) < 1e-12  # measured 2.4e-13


def test_the_logarithm_of_a_near_unit_argument_is_the_loss_that_is_left():
    """What the far root's branch still costs, measured, and why it is not a spelling.

    At a quarter-turn amplitude the denominator and the squared span are both the pole
    to leading order, so the logarithm's argument approaches one from below by of
    order the modulus ``k'`` -- and its value is of that order too, which is a
    relative ``eps/k'``.  Nothing in the two arrangements under test moves it: the
    printed spelling and the geometric one go through the same logarithm and lose the
    same amount.  Closing it needs the argument's DEPARTURE from one, assembled as a
    sum of positives and taken through ``log1p``, which is a fourth arrangement and
    is not made here -- so the residual is pinned instead, and this test fails if it
    grows.
    """
    element, _ = on_the_end_plane(RATIOS[RATIOS >= 1e-6])
    gap, radical = element._pole_gap(1), element.ellipj["dn"]
    far = np.sqrt(abs(element.np2[1]))
    span = np.sqrt(gap) + far * radical
    argument = element.k2**2 * element._pole_denominator(1) / span**2
    # smallest over the sweep is the tightest geometry, where the loss is worst
    assert np.min(abs(argument[0] - 1.0)) < 1e-05
    want = extended_pole_coefficient(element, 1)
    assert worst(element.Ip[1], want) > 1e-11  # measured 4.2e-10
    assert worst(element.Ip[1], want) < 1e-09  # and no worse than that


# ---------------------------------------------------------------------------
# The third kind, at the pole and the complement the geometry supplies.


@pytest.mark.parametrize("p", [1, 2, 3])
def test_the_third_kind_holds_at_the_end_plane(p):
    """Against its defining integral, in both routes.

    Supplying the pole and the complement routes the evaluation through the descent
    that reflects a near pole onto its far partner; without them the symmetric forms
    have ``1 - n`` and ``1 - m`` to work from, and at a quarter-turn amplitude the
    sine's square is one and nothing dilutes either.
    """
    ratios = RATIOS[RATIOS >= 1e-6]
    element, _ = on_the_end_plane(ratios)
    printed, _ = on_the_end_plane(ratios, rows=PrintedRows)
    complement, _, poles = extended(element)
    want = np.array(
        [
            _pole_integral(0.0, float(pole), float(term))
            for pole, term in zip(poles[p].ravel(), complement.ravel())
        ]
    )
    assert worst(element.Pi_inc[p][0].ravel(), want) < 1e-14  # measured 3.3e-16
    # 9.4e-09 at the far root, whose pole reaches 1e12 and dilutes its own
    # subtraction, and 2.5e-04 and 8.2e-04 at the two that approach the range end
    assert worst(printed.Pi_inc[p][0].ravel(), want) > 1e-09


def test_the_parameter_route_cannot_be_taken_once_the_parameter_reaches_one():
    """A target within 1e-08 radii of the ring, where ``1 - m`` is nothing at all.

    The general entry point forms the modulus complement itself and asserts it has
    something to form it from, so the printed route does not silently return a wrong
    number there -- it stops.  Supplying the complement is what makes those
    geometries evaluable at all.
    """
    printed, _ = approaching_the_filament(RATIOS[RATIOS <= 1e-8], rows=PrintedRows)
    with pytest.raises(AssertionError):
        printed.Pi_inc  # noqa: B018
    element, _ = approaching_the_filament(RATIOS[RATIOS <= 1e-8])
    for p in (1, 2, 3):
        assert np.all(np.isfinite(element.Pi_inc[p]))


def weighted(element, family, integral):
    """Return the ``p_sum`` a section row carries, and the scale it is formed on.

    The coefficient families come from the class either way, because what is under
    test is the INTEGRAL each one multiplies.  The scale is the largest of the three
    terms: the families themselves cancel between the roots -- by up to 3e05 over this
    sweep for the vertical row -- so a departure measured against the sum would report
    the family's own conditioning rather than the integral's.
    """
    weights = getattr(element, family)
    scale = np.max([abs(weights[p] * integral[p]) for p in range(1, 4)], axis=0)
    return element.p_sum(weights, integral), scale


def departure(got, want, scale):
    """Return the largest departure of ``got`` from ``want``, against ``scale``."""
    return float(np.max(abs(np.float64(got) - np.float64(want)) / np.float64(scale)))


@pytest.mark.parametrize("family", ["Qr", "Qz", "Pphi"])
def test_the_weighted_third_kind_sums_the_section_rows_carry(family):
    """The rows are ``p_sum`` over the three characteristics, not one of them.

    So the end-to-end claim is about the weighted sum, held against the same defining
    integral substituted for the class' own -- and measured against the terms being
    summed rather than the sum, for the reason :func:`weighted` gives.
    """
    ratios = RATIOS[RATIOS >= 1e-6]
    element, _ = on_the_end_plane(ratios)
    printed, _ = on_the_end_plane(ratios, rows=PrintedRows)
    complement, _, poles = extended(element)
    reference = {
        p: np.array(
            [
                _pole_integral(0.0, float(pole), float(term))
                for pole, term in zip(poles[p].ravel(), complement.ravel())
            ]
        ).reshape(element.ck2.shape)
        for p in range(1, 4)
    }
    want, scale = weighted(element, family, reference)
    got, _ = weighted(element, family, {p: element.Pi_inc[p][0] for p in range(1, 4)})
    lost, _ = weighted(printed, family, {p: printed.Pi_inc[p][0] for p in range(1, 4)})
    assert departure(got, want, scale) < 8e-15  # measured 5.3e-16
    assert departure(lost, want, scale) > 1e-05  # measured 1.3e-04 .. 5.3e-04


@pytest.mark.parametrize("family", ["Pr", "Qphi"])
def test_the_weighted_elementary_coefficient_sums_the_section_rows_carry(family):
    """The same claim for the elementary coefficient, over the same three roots."""
    ratios = RATIOS[RATIOS >= 1e-6]
    element, _ = on_the_end_plane(ratios)
    printed, _ = on_the_end_plane(ratios, rows=PrintedRows)
    reference = {
        p: np.float64(extended_pole_coefficient(element, p)[0]) for p in range(1, 4)
    }
    want, scale = weighted(element, family, reference)
    got, _ = weighted(element, family, {p: element.Ip[p][0] for p in range(1, 4)})
    lost, _ = weighted(printed, family, {p: printed.Ip[p][0] for p in range(1, 4)})
    assert departure(got, want, scale) < 8e-15  # measured 3.1e-16
    assert departure(lost, want, scale) > 1e-05  # measured 1.3e-04 .. 4.0e-04


def test_the_domain_selection_answers_on_either_side_of_a_unit_characteristic():
    """Which route runs is decided by the POLE'S DOMAIN, and both sides are covered.

    Below a characteristic of one the pole is positive -- the ring denominator's root
    sits past the far end of the range -- and the descent applies; the arc supplies
    the pole and takes it.  Above one the root falls INSIDE the range, the integral is
    a principal value, and the descent has no branch for it: the general entry point
    on the characteristic and the parameter is the one that can answer, and its sign
    is the assertion because a positive result would be the descent's zero-pole
    convention leaking into a domain it does not cover.
    """
    amplitude, complement = np.array([1.2]), np.array([0.2])
    sine, cosine = np.sin(amplitude), np.cos(amplitude)
    for characteristic in (0.9, 1.0 - 4e-16):
        pole = np.array([1.0 - characteristic])
        descent = Constants.ellippinc(
            np.array([characteristic]),
            amplitude,
            1.0 - complement,
            pole=pole,
            complement=complement,
        )
        assert descent == pytest.approx(
            incomplete_pole(pole, complement, sine, cosine), rel=4e-16
        )
        assert np.all(descent > 0.0)
    for characteristic in (1.0 + 4e-16, 1.3):
        pole = np.array([1.0 - characteristic])
        general = Constants.ellippinc(
            np.array([characteristic]), amplitude, 1.0 - complement
        )
        assert np.all(np.isfinite(general)) and np.all(general != 0.0)
        assert np.all(incomplete_pole(pole, complement, sine, cosine) == 0.0)
    # far enough past one that the principal value has changed sign, which is the
    # unmistakable statement that the descent's zero is not an answer here
    assert Constants.ellippinc(np.array([1.3]), np.array([5.5]), 1.0 - complement) < 0.0


def test_every_characteristic_the_arc_reaches_stays_inside_the_descent_s_domain():
    """Which is why the arc can always supply the pole, at every one of its limits."""
    for co_amplitude in (0.0, 0.3, 1.4):
        element, _ = on_the_end_plane(RATIOS, co_amplitude=co_amplitude)
        for p in (1, 2, 3):
            assert np.all(element.np2_pole[p] >= 0.0)
            assert np.all(element.np2[p] <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
