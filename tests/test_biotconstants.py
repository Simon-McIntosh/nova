"""Conditioning contract for the arc reduction's integration constants.

:class:`nova.biot.constants.Constants` holds the ring quantities the arc elements
are built on -- the modulus, the three ring characteristics, and the coefficient
families that weight them.  Every one of those is a quantity whose SMALLNESS is
the answer, and each is reached by two routes that agree in exact arithmetic and
not in floats:

* the modulus complement is the target's squared distance to the source corner
  over the squared ring span, and ``1 - k^2`` knows it only to ``eps``;
* a ring characteristic's own complement -- the denominator's value at the far end
  of the range, which is what the third kind's growth is set by -- is an exact
  square of the geometry, and ``1 - n`` knows it only to ``eps``;
* the near pole's numerator ``rs - c`` cancels to nothing on a corner's own
  radius, where it is ``(rs - r)`` less ``gamma^2/(r + c)``.

So the tests here come in pairs.  One half pins the IDENTITIES -- that the
geometric spelling and the subtraction agree wherever the subtraction is sound,
which is what makes the two the same algebra rather than two formulas.  The other
half drives the geometry at the configuration that separates them: a target
approaching the plane of a source corner (``gamma -> 0``) and a target
approaching a corner's own radius (``rs -> r``), which between them are every
grid node that lands on a conductor face.

References are the complement-native routines
(:mod:`nova.biot.completeelliptic`, :mod:`nova.biot.incompleteelliptic`) fed the
exact geometric complements, and -- for the ring itself --
:func:`nova.biot.greens.greens_psi` and :func:`nova.biot.greens.greens_bz_br`,
whose own accuracy gates live in ``test_biotcompleteelliptic``,
``test_biotincompleteelliptic`` and ``test_biotgreens``.  Nothing here re-measures
a special function; what is measured is the ARGUMENT each one is handed.
"""

import numpy as np
import pytest

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.constants import Constants
from nova.biot.greens import MU0, greens_bz_br, greens_psi
from nova.biot.incompleteelliptic import incomplete_pole

RADIUS = 6.2  # a metre-scale ring, where a face is centimetres from the axis

# The target's distance to a source corner, as a fraction of the ring radius, from
# a tenth of the radius down to the resolution a float radius has at all.  A
# section face is centimetres from a metre-scale ring's axis, so the upper end of
# this range is a target OFF the section and the lower end is one on its face.
RATIOS = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12])

# Where the two spellings must agree, because the subtraction still has digits:
# half a radius clear of the corner in BOTH directions puts every complement at
# order 1e-02 and the naive route within a couple of hundred ulp of the geometric
# one -- which is a loss already, and the largest one an identity check can be
# built on.
SOUND = 0.5

QUARTER_TURN = 0.5 * np.pi


def constants(rs, gamma, r, z=0.0):
    """Return a Constants over broadcast geometry arrays."""
    arrays = np.broadcast_arrays(
        *(np.asarray(term, dtype=float) for term in (rs, gamma, r, z))
    )
    return Constants(*(term.copy() for term in arrays))


def exact(rs, gamma, r):
    """Return the modulus complement and the three poles, from the geometry.

    Spelled out independently of the class under test, term for term against the
    identities: ``r - c`` is ``-gamma^2/(r + c)`` exactly for
    ``c = sqrt(gamma^2 + r^2)``, so the first characteristic's complement is
    ``((r + c)/gamma)^2`` and the second's its reciprocal; the two radicals
    ``a^2`` and ``a^2 - 4 r rs`` differ by exactly ``4 r rs``, so the modulus
    complement is a sum of squares over one of them.
    """
    b = rs + r
    a2 = gamma**2 + b**2
    c = np.sqrt(gamma**2 + r**2)
    return (
        (gamma**2 + (rs - r) ** 2) / a2,
        {
            1: ((r + c) / gamma) ** 2,
            2: (gamma / (r + c)) ** 2,
            3: ((rs - r) / b) ** 2,
        },
    )


def corner_plane(ratios, offset=1e-3):
    """Return geometry driving the target onto the plane of a source corner.

    ``offset`` is the corner's radial distance from the target, held fixed, so
    only ``gamma`` shrinks and the modulus complement and the two reciprocal ring
    poles are what move.
    """
    return dict(rs=RADIUS * (1.0 + offset), gamma=RADIUS * ratios, r=RADIUS)


def corner_radius(ratios, ratio=1e-3):
    """Return geometry driving the target onto a source corner's own radius.

    The complement of the THIRD characteristic is ``((rs - r)/b)^2`` and carries no
    ``gamma`` at all, so this is the sweep that separates it -- and the numerator
    ``rs - c`` cancels along it.
    """
    return dict(rs=RADIUS * (1.0 + ratios), gamma=RADIUS * ratio, r=RADIUS)


# ---------------------------------------------------------------------------
# The identities, where both spellings are sound.


def test_the_geometric_complements_agree_with_the_subtraction_off_the_corner():
    """One algebra, not two: a target well off the corner separates neither.

    Every complement is order 1e-02 here, which takes both a HALF radius of
    ``gamma`` and a half radius of radial gap -- the third characteristic's pole is
    ``((rs - r)/b)^2`` and carries no ``gamma``, so a target off the corner PLANE
    alone leaves it small and the subtraction already lossy.
    """
    con = constants(rs=RADIUS * (1.0 + SOUND), gamma=RADIUS * SOUND, r=RADIUS)
    complement, pole = exact(con.rs, con.gamma, con.r)
    assert con.ck2 == pytest.approx(complement, rel=1e-14, abs=0)
    assert (1.0 - con.k2) == pytest.approx(complement, rel=1e-12, abs=0)
    for p in (1, 2, 3):
        assert con.np2_pole[p] == pytest.approx(pole[p], rel=1e-14, abs=0)
        assert (1.0 - con.np2[p]) == pytest.approx(pole[p], rel=1e-11, abs=0)


def test_the_first_two_ring_poles_are_reciprocal():
    """``(1 - n1)(1 - n2) = 1`` -- the ring denominator's two roots, symmetric.

    Exact in the mathematics and a check on the spelling: both are formed from the
    same ``r + c``, so the product is one to round-off however small ``gamma`` is,
    where a pair formed by subtraction loses the identity with the poles.
    """
    con = constants(**corner_plane(RATIOS))
    product = con.np2_pole[1] * con.np2_pole[2]
    assert product == pytest.approx(np.ones_like(product), rel=4e-16, abs=0)


def test_nothing_is_held_back_from_unity():
    """The modulus and the two bounded characteristics are the geometry, exactly.

    All three approach one at a corner, and a factor just under one keeps them off
    it -- which caps the complement at that factor's own distance from unity
    however small the geometry makes it, and leaves the value biased by the same
    amount everywhere else.  Asserted bit for bit against the expression the
    geometry gives, because any factor at all is visible there and nowhere else:
    the complements now come from the geometry and no longer need the modulus held
    off its own limit.
    """
    con = constants(**corner_plane(RATIOS))
    assert np.array_equal(con.k2, 4 * con.r * con.rs / con.a2)
    assert np.array_equal(con.np2[2], 2 * con.r / (con.r + con.c))
    assert np.array_equal(con.np2[3], 4 * con.r * con.rs / con.b**2)


def test_the_near_pole_numerator_is_the_radial_gap_less_the_level_term():
    """``rs - c`` as ``(rs - r) - gamma^2/(r + c)``, which keeps its digits."""
    con = constants(**corner_radius(RATIOS))
    want = (con.rs - con.r) - con.gamma**2 / (con.r + con.c)
    assert con.edge[2] == pytest.approx(want, rel=4e-16, abs=0)


# ---------------------------------------------------------------------------
# The conditioning, where they part.


def test_the_modulus_complement_holds_onto_a_corner_plane():
    """``k'^2`` from the geometry, not from one less the modulus.

    ``1 - k^2`` carries an absolute ``eps``, so its relative accuracy is ``eps``
    over the complement itself -- and the complement is the target's squared
    distance to the corner over the squared ring span, which a centimetre off a
    metre-scale ring is already 1e-06.
    """
    con = constants(**corner_plane(RATIOS))
    complement, _ = exact(con.rs, con.gamma, con.r)
    assert con.ck2 == pytest.approx(complement, rel=8e-16, abs=0)


def test_the_modulus_complement_holds_onto_a_corner_radius():
    """The other approach to the same confluence, along the radius."""
    con = constants(**corner_radius(RATIOS))
    complement, _ = exact(con.rs, con.gamma, con.r)
    assert con.ck2 == pytest.approx(complement, rel=8e-16, abs=0)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_each_ring_pole_holds_at_a_corner(p, geometry):
    """Every characteristic's complement is an exact square of the geometry.

    The third kind grows like the inverse root of its pole, so a pole formed as
    ``1 - n`` caps the integral at ``eps`` over it -- and all three poles vanish
    at a corner, the first two as ``gamma^2`` and the third as ``(rs - r)^2``.
    """
    con = constants(**geometry(RATIOS))
    _, pole = exact(con.rs, con.gamma, con.r)
    assert con.np2_pole[p] == pytest.approx(pole[p], rel=8e-16, abs=0)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_each_characteristic_holds_at_a_corner(p, geometry):
    """The characteristic itself, which the coefficient families weight.

    The first diverges as ``gamma^-2`` and is taken as ``-2 r (r + c)/gamma^2``
    rather than through ``r - c``, whose two terms agree to every digit on the
    corner's own plane.
    """
    con = constants(**geometry(RATIOS))
    _, pole = exact(con.rs, con.gamma, con.r)
    assert con.np2[p] == pytest.approx(1.0 - pole[p], rel=8e-16, abs=0)


def extended_radial_weight(con):
    """Return ``v`` in longdouble from the collapsed numerator.

    ``1 + k^2 (gamma^2 - b r)/(2 r rs)`` is ``(3 gamma^2 + (rs - r)(rs + r))/a^2``
    identically, and every term of the second is a product of exact differences.
    """
    gamma = np.longdouble(con.gamma)
    gap = np.longdouble(con.rs) - np.longdouble(con.r)
    span = np.longdouble(con.rs) + np.longdouble(con.r)
    return (3 * gamma**2 + gap * span) / (gamma**2 + span**2)


@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_the_radial_row_weight_holds_at_a_corner(geometry):
    """``v`` vanishes at a corner, and the collapsed numerator holds onto it."""
    con = constants(**geometry(RATIOS))
    ratio = con.v / extended_radial_weight(con)
    assert np.max(abs(ratio - 1.0)) < 1e-15  # measured 2.5e-16


def test_the_radial_row_weight_cancels_where_it_is_taken_from_one():
    """What the printed ``v`` costs, measured, on the sweep that drives it to zero.

    ``v`` is one plus a term that reaches minus one, so it comes back at order the
    squared distance to the corner out of two quantities of order one -- and it
    vanishes only when BOTH the standoff and the radial gap do, which is why this
    runs along the radius at a corner already within a micrometre of the plane.
    Both spellings go against the same extended-precision value, so the gap between
    the two curves is the arrangement and nothing else.
    """
    con = constants(**corner_radius(RATIOS, ratio=1e-6))
    want = extended_radial_weight(con)
    printed = 1 + con.k2 * (con.gamma**2 - con.b * con.r) / (2 * con.r * con.rs)
    assert np.max(abs(printed / want - 1.0)) > 1e-06  # measured 6.5e-05
    assert np.max(abs(con.v / want - 1.0)) < 1e-15  # measured 1.9e-16


@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_the_complete_third_kind_holds_at_a_corner(geometry):
    """The integral the poles are handed to, over all three characteristics."""
    con = constants(**geometry(RATIOS))
    complement, pole = exact(con.rs, con.gamma, con.r)
    for p in (1, 2, 3):
        want = complete_pole(pole[p], complement)
        assert con.Pi[p] == pytest.approx(want, rel=1e-14, abs=0)


@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_the_first_and_second_kinds_hold_at_a_corner(geometry):
    """``K`` grows like ``-log k'``, so it is only as good as its argument."""
    con = constants(**geometry(RATIOS))
    complement, _ = exact(con.rs, con.gamma, con.r)
    want_first, want_second = complete_kind(complement)
    assert con.K == pytest.approx(want_first, rel=1e-14, abs=0)
    assert con.E == pytest.approx(want_second, rel=1e-14, abs=0)


# ---------------------------------------------------------------------------
# Through the third kind's own entry points, at an exact pole.


def test_the_general_entry_points_keep_the_wider_domain():
    """A characteristic ABOVE one, which the descent has no branch for.

    The pole is then negative -- the ring denominator's root falls INSIDE the
    range and the integral is a principal value -- and the descent, whose whole
    arrangement is a sum of positives over a pole above zero, returns zero there.
    Every characteristic a source ring produces is below one, so supplying the pole
    is always available to the reduction; the entry points also serve callers whose
    characteristic is not, and this is the case that separates the two routes.

    The value is the principal value of the integral, so the sign is the assertion:
    a positive result would be the descent's zero-pole convention leaking into a
    domain it does not cover.
    """
    n, m = np.array([1.3]), np.array([0.8])
    assert Constants.ellipp(n, m) < 0.0
    assert Constants.ellippinc(n, np.array([5.5]), m) < 0.0


def test_every_ring_characteristic_is_below_one():
    """Which is why the reduction can always supply the pole to the descent.

    ``c >= r`` makes the near characteristic ``2 r/(r + c)`` at most one, ``4 r rs``
    is at most ``b^2`` by the arithmetic-geometric mean so the third is too, and the
    far one is negative outright.  All three poles are therefore non-negative, which
    is the descent's domain.
    """
    for geometry in (corner_plane, corner_radius):
        con = constants(**geometry(RATIOS))
        assert np.all(con.np2[1] < 0.0)
        assert np.all(con.np2[2] <= 1.0)
        assert np.all(con.np2[3] <= 1.0)
        for p in (1, 2, 3):
            assert np.all(con.np2_pole[p] >= 0.0)


@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_the_complete_entry_point_takes_an_exact_pole(geometry):
    """``ellipp`` at a supplied pole is the complement-native routine."""
    con = constants(**geometry(RATIOS))
    complement, pole = exact(con.rs, con.gamma, con.r)
    for p in (1, 2, 3):
        got = con.ellipp(con.np2[p], con.k2, pole=pole[p], complement=complement)
        assert got == pytest.approx(
            complete_pole(pole[p], complement), rel=4e-16, abs=0
        )


@pytest.mark.parametrize("co_amplitude", [0.9, 1e-4, 1e-9])
@pytest.mark.parametrize("geometry", [corner_plane, corner_radius])
def test_the_incomplete_entry_point_takes_an_exact_pole(co_amplitude, geometry):
    """The arc's own amplitude, which reaches a quarter turn on its end plane.

    A quarter-turn amplitude is where the characteristic's own cancellation
    reaches the answer undiminished: the third kind's denominator at the
    amplitude is ``cos^2 + (1 - n) sin^2``, and the sine's square is one there,
    so nothing is left to dilute a pole formed by subtraction.
    """
    con = constants(**geometry(RATIOS))
    complement, pole = exact(con.rs, con.gamma, con.r)
    amplitude = QUARTER_TURN - co_amplitude
    sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
    for p in (1, 2, 3):
        got = con.ellippinc(
            con.np2[p],
            amplitude,
            con.k2,
            sine=np.full_like(complement, sine),
            cosine=np.full_like(complement, cosine),
            pole=pole[p],
            complement=complement,
        )
        want = incomplete_pole(
            pole[p],
            complement,
            np.full_like(complement, sine),
            np.full_like(complement, cosine),
        )
        assert got == pytest.approx(want, rel=4e-16, abs=0)


# ---------------------------------------------------------------------------
# The ring these constants assemble, against the axisymmetric kernel.


def approaching_the_filament():
    """Return a Constants driving the target onto a ring along the diagonal.

    Both ``gamma`` and the radial gap shrink together, so the modulus complement
    falls as the square of the distance and the two field components diverge.
    """
    gap = RATIOS * RADIUS / np.sqrt(2.0)
    return constants(rs=RADIUS, gamma=gap, r=RADIUS + gap), gap


def ring_rows(con, axial_weight):
    """Return ``(Psi, Br, Bz)`` for a circular filament, from the constants.

    The three expressions :class:`nova.biot.circle.Circle` builds, term for term,
    so what is measured is the constants they are assembled from rather than the
    frame plumbing around them.  ``Br`` and ``Bz`` divide by the modulus
    complement, which is the whole reason it cannot be a subtraction.

    ``axial_weight`` is the second kind's own weight in the vertical row, ``2 r``
    less ``b k^2``, supplied by the caller because it is the one term of the three
    whose arrangement is still a question -- see
    :func:`test_the_vertical_row_weight_cancels_where_it_is_taken_from_the_modulus`.
    """
    aphi = 1.0 / (2.0 * np.pi) * con.a / con.r * ((1.0 - con.k2 / 2.0) * con.K - con.E)
    psi = 2.0 * np.pi * MU0 * con.r * aphi
    br = (
        MU0
        / (2.0 * np.pi)
        * con.gamma
        * (con.K - (2.0 - con.k2) / (2.0 * con.ck2) * con.E)
        / (con.a * con.r)
    )
    bz = (
        MU0
        / (2.0 * np.pi)
        * (con.r * con.K - axial_weight / (2.0 * con.ck2) * con.E)
        / (con.a * con.r)
    )
    return psi, br, bz


def geometric_axial_weight(con):
    """Return ``2 r - b k^2`` as the geometry gives it, without the subtraction.

    ``a^2 - 2 rs b`` is ``gamma^2 + b(r - rs)``, so

        2 r - b k^2 = 2 r (gamma^2 - b (rs - r))/a^2

    exactly, and neither term of that numerator approaches the other as the target
    reaches the ring.
    """
    return 2.0 * con.r * (con.gamma**2 - con.b * (con.rs - con.r)) / con.a2


def test_the_ring_reproduces_the_axisymmetric_kernel_near_the_filament():
    """A target driven onto a circular filament, both radially and vertically.

    The kernel is complement-native throughout and carries the divergence rather
    than a standoff, so it is the reference the whole way in; the flux is bounded
    there and the two field components are not, which is why all three are held
    relatively.
    """
    con, gap = approaching_the_filament()
    psi, br, bz = ring_rows(con, geometric_axial_weight(con))
    want_psi = greens_psi(con.r, np.zeros_like(gap), RADIUS, gap)
    want_bz, want_br = greens_bz_br(con.r, np.zeros_like(gap), RADIUS, gap)
    assert psi == pytest.approx(want_psi, rel=1e-13, abs=0)
    assert br == pytest.approx(want_br, rel=1e-13, abs=0)
    assert bz == pytest.approx(want_bz, rel=1e-13, abs=0)


def test_the_vertical_row_weight_cancels_where_it_is_taken_from_the_modulus():
    """What the printed ``2 r - b k^2`` costs the vertical row, measured.

    The complements above put the flux and the radial row on round-off the whole
    way onto the filament, and leave the vertical row on a subtraction of its own:
    ``b k^2`` reaches ``2 r`` as the target reaches the ring, so the printed
    difference of two quantities of order the ring span comes back at order the
    distance, for a relative ``eps b/(2 r - b k^2)``.  Both arrangements run here
    against the same reference, over the same approach, because the size of the gap
    between them is what says the geometric one is worth taking -- four decades,
    at a target a hundredth of a micrometre off a metre-scale ring, where the
    complement route it stands beside is already exact.
    """
    con, gap = approaching_the_filament()
    want_bz, _ = greens_bz_br(con.r, np.zeros_like(gap), RADIUS, gap)
    printed = ring_rows(con, 2.0 * con.r - con.b * con.k2)[2]
    geometric = ring_rows(con, geometric_axial_weight(con))[2]
    assert np.max(abs(printed / want_bz - 1.0)) > 1e-09  # measured 7.1e-09
    assert np.max(abs(geometric / want_bz - 1.0)) < 1e-14  # measured 1.1e-15
