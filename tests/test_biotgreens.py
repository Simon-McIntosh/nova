"""Tests for the canonical axisymmetric Green's-function library.

These run on plain numpy arrays with no frame machinery.  They pin:

* the single corner antiderivative ``corner_fields`` against the Urankar
  Part III rectangular-section formula it was extracted from -- the dataclass
  cylinder kernel and the functional ``cylinder_greens`` both drive this one
  routine, so the guard protects against the two paths silently diverging;
* continuity of the finite-area rectangle kernel across the section's OWN corner
  planes, where the antiderivative's branch terms change sign -- twice over, since
  an oracle comparison bounds the VALUE either side and a ladder of difference
  quotients is what separates a discontinuity from the field's own slope;
* far-field agreement of the finite-area rectangle kernel with the point
  circular-loop kernel;
* psi<->B consistency of the point-loop kernel (Jackson forms).
"""

from __future__ import annotations

import numpy as np

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.greens import (
    MU0,
    corner_fields,
    cylinder_greens,
    greens_bz_br,
    greens_psi,
)
from nova.biot.zeta import zeta

# rectangular section + off-section targets (inside and outside)
_A, _Z0, _DA, _DZ = 0.90, 0.10, 0.12, 0.18
_TR = np.array([1.02, 1.20, 1.50, 0.93, 0.30, 1.90])
_TZ = np.array([0.10, 0.40, 0.00, 0.13, -1.20, 1.50])
_ULP = np.finfo(np.float64).eps


def _corner_stacks(a, z0, da, dz, tr, tz):
    """Return (rs, zs, r, z) corner arrays shaped (T, 4), matching the kernel."""
    rs = np.broadcast_to(
        np.array([a - da / 2, a + da / 2, a + da / 2, a - da / 2]), tr.shape + (4,)
    ).copy()
    zs = np.broadcast_to(
        np.array([z0 - dz / 2, z0 - dz / 2, z0 + dz / 2, z0 + dz / 2]), tr.shape + (4,)
    ).copy()
    r4 = np.repeat(tr[..., None], 4, axis=-1)
    z4 = np.repeat(tz[..., None], 4, axis=-1)
    return rs, zs, r4, z4


def _reference_corner_fields(rs, zs, r4, z4):
    """The Urankar Part III corner antiderivative, spelled out independently.

    Written out term by term against the paper, in the paper's own grouping: the
    ``P`` and ``Q`` coefficient families each carrying their own factor of gamma,
    and one ``Pi`` per characteristic.  The kernel instead folds a factor of gamma
    out of the two ring poles and carries the bounded product, which is what makes
    it finite and continuous on a corner plane; off those planes the two groupings
    are the same algebra and this pins that they are.

    What this reference does NOT do is form a pole or the modulus by subtraction.
    The paper's printed ``2r/(r - c)`` and ``1 - 4 r rs/a^2`` cost the branch
    cancellation a relative ``eps r^2/gamma^2``, which is already 3e-09 in the flux
    at a tenth of the section height off a corner level -- so a reference spelled
    that way disagrees with an accurate kernel by four decades more than the pin
    allows, and would be measuring its own conditioning rather than the algebra.
    Both spellings therefore take the same exact complements, the same ``zeta`` and
    the same complete-elliptic descent, leaving only the grouping between them;
    the quadrature and the special functions have their own accuracy gates in
    ``test_biotzeta`` and ``test_biotcompleteelliptic``.

    Returns ``(values, widest)``: the three antiderivative coefficients, and the
    magnitude of the largest term each of them is assembled from.  The second is
    what a rounding is proportional to and the first is not -- the corner value is a
    cancellation of those terms -- so a comparison of the two spellings that means
    anything about the ALGEBRA is taken over the widest term.
    """
    gamma = zs - z4
    b = rs + r4
    a = np.sqrt(gamma**2 + b**2)
    c2 = gamma**2 + r4**2
    c = np.sqrt(c2)
    k2 = 4 * r4 * rs / (gamma**2 + b**2)
    complement = (gamma**2 + (rs - r4) ** 2) / (gamma**2 + b**2)
    big_k, big_e = complete_kind(complement)
    v = 1 + k2 * (gamma**2 - b * r4) / (2 * r4 * rs)
    u_coef = k2 * (4 * gamma**2 + 3 * rs**2 - 5 * r4**2) / (4 * r4)
    zt = zeta(rs, r4, gamma, (np.pi / 2) * np.ones_like(rs))

    # r - c is -gamma^2/(r + c) exactly, which is what the first characteristic and
    # its own pole are built from rather than from r less c
    np2 = {
        1: -2 * r4 * (r4 + c) / gamma**2,
        2: 2 * r4 / (r4 + c),
        3: 4 * r4 * rs / b**2,
    }
    pole = {
        1: ((r4 + c) / gamma) ** 2,
        2: (gamma / (r4 + c)) ** 2,
        3: ((rs - r4) / b) ** 2,
    }
    pi3 = {p: complete_pole(pole[p], complement) for p in (1, 2, 3)}
    # rs -/+ c: the minus sign cancels to nothing on a corner's own radius, where it
    # is (rs - r) less gamma^2/(r + c) instead
    edge = {1: rs + c, 2: (rs - r4) - gamma**2 / (r4 + c)}
    qr = {p: edge[p] * np2[p] * gamma**2 * c / r4 for p in (1, 2)}
    qr[3] = np.zeros_like(r4)
    qz = {p: edge[p] * -2 * gamma * c * np2[p] for p in (1, 2)}
    qz[3] = gamma * b * (rs - r4) * np2[3]
    pphi = {p: edge[p] * np2[p] * c * (3 * r4**2 - c2) / (2 * r4) for p in (1, 2)}
    pphi[3] = -rs / b * (rs - r4) * (3 * r4**2 - rs**2)

    def p_sum(coef):
        return sum((-1) ** p * coef[p] * pi3[p] for p in (1, 2, 3))

    cphi = -1 / 3 * r4**2 * np.pi / 2 * np.sign(gamma) * (np.sign(rs - r4) + 1)
    aphi = (
        cphi
        + gamma * r4 * zt
        + gamma * a / (6 * r4) * (u_coef * big_k - 2 * rs * big_e)
        + gamma / (6 * a * r4) * p_sum(pphi)
    )
    br = (
        r4 * zt - a / (2 * r4) * rs * (big_e - v * big_k) - 1 / (4 * a * r4) * p_sum(qr)
    )
    bz = (
        3 / r4 * cphi
        + 2 * gamma * zt
        - a / (2 * r4) * 3 / 2 * gamma * k2 * big_k
        - 1 / (4 * a * r4) * p_sum(qz)
    )

    def widest(*terms):
        return np.max(np.abs(np.array(terms)), axis=0)

    # every addend that carries its own rounding, the inner differences and the
    # third-kind sum broken back out: those two are where the cancellation is, so
    # the terms and not the bracketed groups are what the envelope is taken over
    envelope = (
        widest(
            cphi,
            gamma * r4 * zt,
            gamma * a / (6 * r4) * u_coef * big_k,
            gamma * a / (6 * r4) * 2 * rs * big_e,
            *[gamma / (6 * a * r4) * pphi[p] * pi3[p] for p in (1, 2, 3)],
        ),
        widest(
            r4 * zt,
            a / (2 * r4) * rs * big_e,
            a / (2 * r4) * rs * v * big_k,
            *[qr[p] * pi3[p] / (4 * a * r4) for p in (1, 2, 3)],
        ),
        widest(
            3 / r4 * cphi,
            2 * gamma * zt,
            a / (2 * r4) * 3 / 2 * gamma * k2 * big_k,
            *[qz[p] * pi3[p] / (4 * a * r4) for p in (1, 2, 3)],
        ),
    )
    return (aphi, br, bz), envelope


def test_corner_fields_matches_reference_formula():
    """The single corner antiderivative reproduces the Part III formula exactly.

    Held at a few ulp rather than a physics tolerance: with the quadrature shared
    between the two spellings, any drift here is the algebra diverging, and that
    should never cost more than the last couple of bits.  The targets are all well
    off the section's corner planes, which is where the paper's own arrangement of
    the characteristics is still faithful to its algebra.

    A few ulp OF WHAT is the whole question, because the corner value is a
    cancellation.  The widest term the antiderivative is assembled from reaches 206
    times the four-corner stack's own scale on these targets and four decades of it
    on targets they do not visit, and the third-kind sum inside cancels by a further
    800 -- so one rounding of one term's last bit arrives as a couple of THOUSAND
    ulp of the corner value.  Counting ulp of the value measures that cancellation
    rather than the algebra, and it is the target's cancellation, not the formula's.

    What the algebra answers for is the last bit of each TERM, so that is the pin.
    The scaled bound below is the cruder statement of the same thing, kept because
    it is the difference the corner rule actually forms.
    """
    stacks = _corner_stacks(_A, _Z0, _DA, _DZ, _TR, _TZ)
    got = corner_fields(*stacks)
    ref, envelope = _reference_corner_fields(*stacks)
    for one, other, widest in zip(got, ref, envelope, strict=True):
        # 2.7 ulp of the widest term here, and no worse than 3.9 across several
        # hundred random target and section geometries, over which the cancellation
        # factor spans three decades and this does not move -- the two groupings
        # differ by rounding and nothing else.  Twice the measured worst.
        assert np.max(np.abs(one - other) / widest) < 8 * _ULP
        # the same disagreement against the corner STACK's scale, which is what the
        # corner rule differences the four values against: 3.7e-14 measured, the
        # term bound multiplied by this target set's own cancellation, so 2.7 times
        # the measured worst is as tight as it goes.  Tightening it towards the term
        # bound would only make it a statement about which targets were chosen.
        scale = np.abs(other).max(axis=-1, keepdims=True)
        assert np.max(np.abs(one - other) / scale) < 1e-13


def test_cylinder_far_field_matches_point_loop():
    """Far from a small section the rectangle kernel tends to the point loop."""
    # targets several section-widths away, where the finite-area correction is
    # a sub-percent second-moment term
    tr = np.array([2.5, 0.4, 1.8])
    tz = np.array([1.5, -1.4, 1.2])
    psi_c, br_c, bz_c = cylinder_greens(tr, tz, _A, _Z0, 0.02, 0.02)
    psi_p = greens_psi(tr, tz, _A, _Z0)
    bz_p, br_p = greens_bz_br(tr, tz, _A, _Z0)
    np.testing.assert_allclose(psi_c, psi_p, rtol=1e-3)
    np.testing.assert_allclose(bz_c, bz_p, rtol=1e-3)
    np.testing.assert_allclose(br_c, br_p, rtol=1e-3)


def test_point_loop_psi_b_consistency():
    """B_Z = (1/2piR) dpsi/dR and B_R = -(1/2piR) dpsi/dZ for the point loop."""
    a, z0 = 0.9, 0.1
    h = 1e-6
    for r, z in [(1.3, 0.3), (0.6, -0.2), (1.1, 0.5)]:
        psi_rp = greens_psi(np.array([r + h]), np.array([z]), a, z0)[0]
        psi_rm = greens_psi(np.array([r - h]), np.array([z]), a, z0)[0]
        psi_zp = greens_psi(np.array([r]), np.array([z + h]), a, z0)[0]
        psi_zm = greens_psi(np.array([r]), np.array([z - h]), a, z0)[0]
        bz, br = greens_bz_br(np.array([r]), np.array([z]), a, z0)
        np.testing.assert_allclose(
            bz[0], (psi_rp - psi_rm) / (2 * h) / (2 * np.pi * r), rtol=1e-5
        )
        np.testing.assert_allclose(
            br[0], -(psi_zp - psi_zm) / (2 * h) / (2 * np.pi * r), rtol=1e-5
        )


def test_units_constant():
    """The kernel carries the SI vacuum permeability."""
    assert MU0 == 4.0e-7 * np.pi


# --- continuity across the section's own corner planes -----------------------
#
# The corner antiderivative carries branch terms that are ODD in the target's
# offset from a corner plane: an arctangent boundary term whose limit is a signed
# right angle, and two third-kind integrals whose poles collapse onto the ends of
# the integration range there.  Those jumps cancel each other exactly, so the
# antiderivative is continuous -- but only if every characteristic and the modulus
# complement are formed from the geometry.  A mis-set branch, or a characteristic
# reached by subtraction, leaves an offset-INDEPENDENT jump; a ladder of standoffs
# over nine decades is what separates that from the smooth variation across the
# plane, since only the jump fails to fall with the standoff.

_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ = 4.0, 0.0, 0.1, 0.08
_LADDER = 10.0 ** -np.arange(2.0, 11.0)
_STRADDLE = np.concatenate([-_LADDER, [0.0], _LADDER[::-1]])
# a target radius strictly inside the section's radial span, off its mid-plane and
# off both corner radii, so the two lower corners straddle it in radius
_INSIDE_SPAN = _FACE_A - 0.0435

# The four ways a target crosses a corner plane, as ``(radius, level, across)``:
# ``across`` selects which coordinate the standoff is added to, and the other one
# is offset onto the corner.  The first two cross a face LEVEL, at a radius inside
# the section's radial span and at one outside it; the last two cross a corner
# RADIUS, at a level inside the section's vertical span and at one below it.  The
# pair that stays outside the span never enters the conductor, so the branch terms
# there change sign without the field's own kink to hide behind.
_CROSSINGS = (
    (_INSIDE_SPAN, 0.0, 1.0),
    (_FACE_A + 0.021, 0.0, 1.0),
    (0.0, _FACE_Z0 + 0.017, 0.0),
    (0.0, _FACE_Z0 - _FACE_DZ / 2 - 0.05, 0.0),
)


def _crossing_centres(radius, level, across):
    """Yield the two corner-plane centres of a crossing, one per corner."""
    for corner in (-1.0, 1.0):
        yield (
            radius + (1.0 - across) * (_FACE_A + corner * _FACE_DA / 2),
            level + across * (_FACE_Z0 + corner * _FACE_DZ / 2),
        )


def _rectangle(a, z0, da, dz):
    """Return the four ``(r, z)`` corners of an axis-aligned rectangular section."""
    return np.array(
        [
            [a - da / 2, z0 - dz / 2],
            [a + da / 2, z0 - dz / 2],
            [a + da / 2, z0 + dz / 2],
            [a - da / 2, z0 + dz / 2],
        ]
    )


def _worst_against_the_polygon(target_r, target_z):
    """Return the worst relative ``(psi, B_R, B_Z)`` deviation from the oracle.

    The oracle is the fully analytic polygon reduction, not the quadrature one: a
    target ON a face sits on the quadrature integrand's own boundary layer, where
    the shipped 16x48 rule is six decades off in ``B_R``, so the quadrature form
    cannot referee the singular configuration.  Away from the plane the two agree,
    which :func:`test_the_two_polygon_oracles_agree_off_the_face` pins.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    got = cylinder_greens(target_r, target_z, _FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ)
    exact = polygon_analytic_greens(
        target_r, target_z, _rectangle(_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ)
    )
    # the field is scaled by its own magnitude rather than per component: B_R
    # passes through zero on the section's mid-plane, where a per-component
    # relative measure has no meaning
    field = np.hypot(exact[1], exact[2])
    local = (np.abs(exact[0]), field, field)
    return [
        float(np.max(np.abs(one - other) / scale))
        for one, other, scale in zip(got, exact, local, strict=True)
    ]


def test_the_two_polygon_oracles_agree_off_the_face():
    """Off the plane the quadrature and analytic polygon kernels are one oracle.

    Which is what licenses the analytic one as the referee ON the plane, where the
    quadrature rule's panels straddle the integrand's boundary layer.
    """
    from nova.biot.polygon import polygon_greens
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = _rectangle(_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ)
    target_r = np.full(6, _INSIDE_SPAN)
    target_z = (
        _FACE_Z0 - _FACE_DZ / 2 + np.array([-0.03, -0.01, -3e-3, 3e-3, 0.01, 0.03])
    )
    quadrature = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
    analytic = polygon_analytic_greens(target_r, target_z, vertices)
    for one, other in zip(quadrature, analytic, strict=True):
        np.testing.assert_allclose(one, other, rtol=2e-9)


def test_the_rectangle_kernel_is_continuous_across_a_horizontal_face():
    """Approaching a section face from either side reaches the same field.

    The configuration a plasma bundle produces by alignment: a grid row level with
    the cell's own lower or upper face, at a radius inside the cell's radial span,
    so the two corners on that face straddle the target in radius and the
    arctangent boundary term is live at one of them and dead at the other.
    """
    for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
        worst = _worst_against_the_polygon(
            np.full(_STRADDLE.shape, _INSIDE_SPAN), level + _STRADDLE
        )
        assert max(worst) < 1e-10, worst


def test_the_rectangle_kernel_is_continuous_across_a_source_radius_plane():
    """Crossing a corner's own radius leaves the field unchanged.

    The other pair of planes the branch terms change sign across: ``rs = r`` flips
    the arctangent's limit at the far end of the angle range, and it also drives
    the third pole and the modulus complement to their own confluence.
    """
    for radius in (_FACE_A - _FACE_DA / 2, _FACE_A + _FACE_DA / 2):
        worst = _worst_against_the_polygon(
            radius + _STRADDLE, np.full(_STRADDLE.shape, _FACE_Z0 + 0.017)
        )
        assert max(worst) < 1e-10, worst


def test_a_target_on_the_source_radius_line_survives_the_corner():
    """Both confluences at once -- the corner itself -- stays on the oracle.

    Sliding along ``rs = r`` through a corner drives the modulus complement, the
    near pole and the third pole to zero together, where the first kind diverges
    logarithmically and the reduction's total weight on it is zero.  It is reached
    rather than approached: the middle row is the vertex exactly.  This is also the
    line the near pole's own numerator ``rs - c`` vanishes on, which is why it is
    formed as ``(rs - r)`` less ``gamma^2/(r + c)``; taken as a subtraction it costs
    the flux 2e-07 a tenth of a micrometre out.
    """
    for radius in (_FACE_A - _FACE_DA / 2, _FACE_A + _FACE_DA / 2):
        for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
            worst = _worst_against_the_polygon(
                np.full(_STRADDLE.shape, radius), level + _STRADDLE
            )
            assert max(worst) < 1e-10, worst


def test_every_corner_plane_crossing_holds_the_oracle():
    """All four ways of crossing a corner plane land on the analytic polygon kernel.

    The oracle half of the crossing set, and it reaches two configurations the
    straddle tests above do not: a radius OUTSIDE the section's radial span crossing
    a face level, and a level below the section crossing a corner radius.  In both
    the target stays outside the conductor throughout, so the branch terms change
    sign with no kink in the field to absorb an error, and neither the flux nor
    either field component may move.

    The middle row of the straddle is the plane exactly.  On the two level crossings
    that is ``gamma == 0`` to the bit, where the kernel masks the pole moments onto
    the mean of their two one-sided limits instead of forming a reciprocal
    characteristic -- an assignment no difference quotient reaches, because every
    quotient stands off the plane on both sides.
    """
    for radius, level, across_the_level in _CROSSINGS:
        for centre in _crossing_centres(radius, level, across_the_level):
            worst = _worst_against_the_polygon(
                centre[0] + (1.0 - across_the_level) * _STRADDLE,
                centre[1] + across_the_level * _STRADDLE,
            )
            # 1.5e-12 measured worst, over the whole ladder including the plane
            # itself; sevenfold reserve.  What holds the floor down is the same
            # thing the sibling straddle tests rest on -- every characteristic and
            # the modulus complement formed from the geometry rather than by
            # subtraction -- so a mis-masked branch on the plane is a full
            # ``pi r^2/3`` here, nine decades clear of the bound.
            assert max(worst) < 1e-11, worst


def test_the_corner_planes_carry_no_jump_of_their_own():
    """The difference quotient across a corner plane converges instead of blowing up.

    Oracle-free, and the sharpest statement of the defect this guards.  A branch
    term left uncancelled is a jump ``J`` that does not depend on the standoff, so
    the quotient ``|f(+d) - f(-d)|/2d`` grows as ``J/2d`` -- five decades of it by
    the time ``d`` is a micrometre -- where a continuous field's quotient settles on
    its own derivative.  Comparing the rungs against each other rather than against
    a bound is what makes this independent of both the field's magnitude and the
    section's geometry.

    A CONVERGENCE statement, not an error against a reference: nothing here is a
    tolerance on an accuracy.  It says the quotient has reached the derivative
    instead of growing as ``J/2d``, which is why the number below is a spread across
    rungs rather than a bound on any one of them.  The oracle statement about the
    same crossings is :func:`test_every_corner_plane_crossing_holds_the_oracle` and
    the two are complementary: an oracle catches a wrong VALUE, this catches a
    DISCONTINUITY, and a jump of a part in 1e11 is far below what any oracle
    comparison at these standoffs resolves.
    """
    for radius, level, across_the_level in _CROSSINGS:
        for centre in _crossing_centres(radius, level, across_the_level):
            quotients = []
            for standoff in _LADDER:
                step = np.array([-standoff, standoff])
                got = cylinder_greens(
                    centre[0] + (1.0 - across_the_level) * step,
                    centre[1] + across_the_level * step,
                    _FACE_A,
                    _FACE_Z0,
                    _FACE_DA,
                    _FACE_DZ,
                )
                quotients.append(
                    [
                        abs(part[1] - part[0]) / (2.0 * standoff) / np.abs(part).max()
                        for part in got
                    ]
                )
            # Every rung below ten micrometres is on the limit already, and they
            # agree with each other to four parts in a thousand.  What sets that
            # floor is the quotient itself and not the kernel's accuracy: the last
            # rung differences two field values a tenth of a nanometre apart, so the
            # kernel's own relative round-off arrives divided by ``2d``, and the
            # earlier rungs carry the quotient's second-order truncation instead --
            # which is why the ladder starts where it does.  The gate is 2.3 times
            # the 4.3e-03 measured, and the last rung is kept even though it sets
            # that floor, because it is the widest lever on the defect: the gate
            # fires on a standoff-independent jump of 1e-11 of the field, ten decades
            # below the ``pi r^2/3`` an uncancelled branch term leaves, and dropping
            # the rung costs a decade of that reach to buy a decade of headroom.
            settled = np.array(quotients[3:])
            spread = np.ptp(settled, axis=0) / settled.mean(axis=0)
            assert np.max(spread) < 1e-2, spread


# --- quadrupole-corrected filament ------------------------------------------


def _hexagon(r0=6.2, z0=0.0, radius=0.06):
    """Return a regular hexagon section of the given circumradius."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def _trapezium(r0=6.2, z0=0.0):
    """Return an asymmetric section whose cross moment does not vanish."""
    return np.array(
        [
            [r0 - 0.05, z0 - 0.03],
            [r0 + 0.04, z0 - 0.03],
            [r0 + 0.06, z0 + 0.02],
            [r0 - 0.02, z0 + 0.04],
        ]
    )


def _sampled_moments(vertices, count=1400):
    """Return area-normalised second central moments by dense grid sampling.

    An independent oracle for the shoelace formulae: uniform points on the
    section's bounding box, kept by a crossing-number test against the polygon.
    """
    v = np.asarray(vertices, dtype=np.float64)
    grid_r, grid_z = np.meshgrid(
        np.linspace(v[:, 0].min(), v[:, 0].max(), count),
        np.linspace(v[:, 1].min(), v[:, 1].max(), count),
    )
    r, z = grid_r.ravel(), grid_z.ravel()
    inside = np.zeros(r.shape, dtype=bool)
    for (ra, za), (rb, zb) in zip(v, np.roll(v, -1, axis=0)):
        if zb == za:  # a horizontal ray never crosses a horizontal edge
            continue
        straddles = (za > z) != (zb > z)
        crossing = ra + (z - za) * (rb - ra) / (zb - za)
        inside ^= straddles & (r < crossing)
    r, z = r[inside], z[inside]
    r, z = r - r.mean(), z - z.mean()
    return float(np.mean(r * r)), float(np.mean(z * z)), float(np.mean(r * z))


def test_the_shoelace_second_moments_match_a_sampled_section():
    """The closed-form moments reproduce a dense sampling of the same polygon."""
    from nova.biot.greens import second_moments

    for vertices in (_hexagon(), _trapezium()):
        got = second_moments(vertices)
        expected = _sampled_moments(vertices)
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=1e-9)


def test_the_moment_filament_removes_the_bare_filament_floor():
    """Far out the bare filament stalls on a floor; the corrected one does not.

    The floor is the section second moment weighted by a curvature the MAJOR
    radius sets, so it does not decay with target distance. Measured against the
    exact polygon kernel at ten section radii, relative to the local magnitude.
    """
    from nova.biot.greens import moment_filament
    from nova.biot.polygon import polygon_greens

    vertices = _hexagon()
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    target_r = 6.2 + 0.6 * np.cos(angle)
    target_z = 0.6 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = moment_filament(target_r, target_z, vertices)
    bare_psi = greens_psi(target_r, target_z, 6.2, 0.0)
    bare_bz, bare_br = greens_bz_br(target_r, target_z, 6.2, 0.0)

    field = np.hypot(exact[1], exact[2])
    local = (np.abs(exact[0]), field, field)
    bare = (bare_psi, bare_br, bare_bz)
    for index in range(3):
        got = np.max(np.abs(corrected[index] - exact[index]) / local[index])
        floor = np.max(np.abs(bare[index] - exact[index]) / local[index])
        assert got < 1e-6
        assert floor > 20.0 * got


def test_the_quadrupole_correction_scales_as_the_section_area():
    """Shrinking the section by ten shrinks its correction by a hundred.

    The correction is a second moment, so it is quadratic in the section size and
    a section small against the ring reduces to the bare filament. Testing the
    SCALING rather than an absolute floor is what separates the physics from the
    difference scheme's round-off: the step scales with the section, which holds
    the correction's relative accuracy section-size independent.
    """
    from nova.biot.greens import moment_filament

    target_r = np.array([6.6, 5.9, 6.2])
    target_z = np.array([0.3, -0.4, 0.5])
    point = np.array(
        [
            greens_psi(target_r, target_z, 6.2, 0.0),
            greens_bz_br(target_r, target_z, 6.2, 0.0)[1],
            greens_bz_br(target_r, target_z, 6.2, 0.0)[0],
        ]
    )
    correction = {}
    for radius in (0.06, 0.006):
        got = np.array(moment_filament(target_r, target_z, _hexagon(radius=radius)))
        correction[radius] = np.abs(got - point) / np.abs(point)
    np.testing.assert_allclose(correction[0.06] / correction[0.006], 100.0, rtol=0.03)
    assert np.max(correction[0.06]) > 1e-4  # real at the shipped plasma cell size


def test_the_moment_filament_carries_the_cross_moment():
    """An asymmetric section needs the off-diagonal moment, so it is applied.

    Dropping the cross term is the cheap five-evaluation path, valid only when the
    section is symmetric about one of its own axes. On a section that is not, the
    term measurably closes the gap to the exact kernel, which is checked here by
    zeroing it: an axis-aligned diagonal-only correction is the worse model.
    """
    from nova.biot.greens import moment_filament, second_moments
    from nova.biot.polygon import polygon_greens

    vertices = _trapezium()
    assert abs(second_moments(vertices)[2]) > 1e-5
    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target_r = 6.2 + 1.0 * np.cos(angle)
    target_z = 1.0 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = moment_filament(target_r, target_z, vertices)
    diagonal = moment_filament(target_r, target_z, vertices, cross_moment=False)
    for index in range(3):
        scale = np.max(np.abs(exact[index]))
        full = np.max(np.abs(corrected[index] - exact[index])) / scale
        without = np.max(np.abs(diagonal[index] - exact[index])) / scale
        assert full < 1e-5
        assert without > 5.0 * full


def _wall_clipped(r0=6.2, z0=0.0, radius=0.06):
    """Return a hexagon with one corner cut by a straight wall.

    Simple (non-self-intersecting) and mildly asymmetric, which is what a plasma
    cell clipped by the first wall looks like: its third moments do not vanish,
    where a regular hexagon's do by symmetry.
    """
    v = list(_hexagon(r0, z0, radius))
    return np.array(
        v[:2]
        + [[r0 - 0.35 * radius, z0 + 0.75 * radius]]
        + [[r0 - 0.95 * radius, z0 + 0.30 * radius]]
        + v[3:]
    )


def _sampled_third_moments(vertices, count=1400):
    """Return area-normalised third central moments by dense grid sampling."""
    v = np.asarray(vertices, dtype=np.float64)
    grid_r, grid_z = np.meshgrid(
        np.linspace(v[:, 0].min(), v[:, 0].max(), count),
        np.linspace(v[:, 1].min(), v[:, 1].max(), count),
    )
    r, z = grid_r.ravel(), grid_z.ravel()
    inside = np.zeros(r.shape, dtype=bool)
    for (ra, za), (rb, zb) in zip(v, np.roll(v, -1, axis=0)):
        if zb == za:
            continue
        straddles = (za > z) != (zb > z)
        crossing = ra + (z - za) * (rb - ra) / (zb - za)
        inside ^= straddles & (r < crossing)
    r, z = r[inside], z[inside]
    r, z = r - r.mean(), z - z.mean()
    return (
        float(np.mean(r**3)),
        float(np.mean(r * r * z)),
        float(np.mean(r * z * z)),
        float(np.mean(z**3)),
    )


def test_the_third_moments_match_a_sampled_section():
    """The triangulated moments reproduce a dense sampling of the same polygon."""
    from nova.biot.greens import third_moments

    vertices = _wall_clipped()
    got = third_moments(vertices)
    expected = _sampled_third_moments(vertices)
    scale = max(abs(value) for value in expected)
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=1e-3 * scale)


def test_a_section_symmetric_about_its_centroid_has_no_third_moment():
    """A regular hexagon's odd moments vanish, so it pays nothing for them."""
    from nova.biot.greens import moment_filament, third_moments

    vertices = _hexagon()
    assert np.max(np.abs(third_moments(vertices))) < 1e-16
    target_r = np.array([6.8, 5.7, 6.2])
    target_z = np.array([0.4, -0.5, 0.6])
    for quadrupole, full in zip(
        moment_filament(target_r, target_z, vertices, order=2),
        moment_filament(target_r, target_z, vertices, order=3),
    ):
        np.testing.assert_array_equal(quadrupole, full)


def test_the_section_skew_dominates_an_asymmetric_cell_far_field():
    """On a wall-clipped cell the third moment is the leading far-field residual.

    Carrying it wins better than an order of magnitude at every distance. What it
    leaves behind is the fourth moment, so the bound is reached a little further
    out rather than never -- where the quadrupole form alone leaves a third-order
    residual that no practical band width recovers.
    """
    from nova.biot.greens import moment_filament
    from nova.biot.polygon import polygon_greens

    vertices = _wall_clipped()
    radius = float(np.max(np.hypot(*(vertices - vertices.mean(axis=0)).T)))
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    worst = {}
    for reach in (7.8, 16.0):
        target_r = 6.2 + reach * radius * np.cos(angle)
        target_z = reach * radius * np.sin(angle)
        exact = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
        field = np.hypot(exact[1], exact[2])
        local = (np.abs(exact[0]), field, field)
        for order in (2, 3):
            got = moment_filament(target_r, target_z, vertices, order=order)
            worst[reach, order] = max(
                float(np.max(np.abs(got[index] - exact[index]) / local[index]))
                for index in range(3)
            )
    for reach in (7.8, 16.0):
        assert worst[reach, 3] < 0.2 * worst[reach, 2]
    assert worst[7.8, 3] > 1e-6
    assert worst[16.0, 3] < 1e-6


# A filament model cannot answer for a point ON the filament, and the one thing it
# must not do is answer for a point NEAR it instead. Both halves are pinned here.


def test_a_target_on_the_filament_returns_the_divergence():
    """No phantom standoff: the flux diverges and the field has no limit at all.

    ``psi`` grows without bound from every direction, so ``inf`` is its limit.
    ``B_Z`` does not have one -- approached radially in the plane its sign follows
    which side the target is on -- so ``nan`` is the honest answer rather than either
    infinity. What neither may be is a finite number: an absolute floor on the
    squared distance to the filament answers for a target 1.4 um away instead, with
    nothing in the result to say so, and it does that at every ring radius because
    the ring radius does not appear in it.
    """
    for radius in (0.9, 1.0, 6.2):
        target = np.array([radius])
        level = np.array([0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = np.asarray(greens_psi(target, level, radius, 0.0))
            bz, br = (
                np.asarray(part) for part in greens_bz_br(target, level, radius, 0.0)
            )
        assert psi[0] == np.inf
        assert np.isnan(bz[0])
        assert np.isnan(br[0]) or br[0] == 0.0


def test_one_ulp_off_the_filament_is_finite_at_every_ring_radius():
    """Adjacent is not coincident, and only the complement tells the two apart.

    The two radicals are formed independently, so their ratio -- the parameter --
    can land an ulp ABOVE one this close, and whether it does depends on the ring
    radius: a unit ring trips it one ULP off the filament where a 6.2 m ring does
    not. The second kind is held at one for that reason; the first kind never sees
    the parameter, taking the complement from the geometry instead, which is what
    makes these values right rather than merely finite.
    """
    for radius in (0.9, 1.0, 6.2, 12.0):
        target = np.array([np.nextafter(radius, np.inf)])
        level = np.array([0.0])
        psi = np.asarray(greens_psi(target, level, radius, 0.0))
        bz, br = (np.asarray(part) for part in greens_bz_br(target, level, radius, 0.0))
        assert np.isfinite(psi).all() and psi[0] > 0.0
        assert np.isfinite(bz).all() and bz[0] < 0.0
        assert np.isfinite(br).all()
        # the flux of a loop against itself one ulp away is 2 mu0 a log(8a/d)-ish:
        # bounded by the ring's own scale rather than by a floor
        assert 1e-5 < psi[0] / radius < 1e-3
