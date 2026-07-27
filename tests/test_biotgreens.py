"""Tests for the canonical axisymmetric Green's-function library.

These run on plain numpy arrays with no frame machinery.  They pin:

* the single corner antiderivative ``corner_fields`` against the Urankar
  Part III rectangular-section formula it was extracted from -- the dataclass
  cylinder kernel and the functional ``cylinder_greens`` both drive this one
  routine, so the guard protects against the two paths silently diverging;
* continuity of the finite-area rectangle kernel across the section's OWN corner
  planes, where the antiderivative's branch terms change sign;
* far-field agreement of the finite-area rectangle kernel with the point
  circular-loop kernel;
* psi<->B consistency of the point-loop kernel (Jackson forms).
"""

from __future__ import annotations

import numpy as np

from nova.biot.constants import Constants
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

    Uses the shared coefficient primitives in :mod:`nova.biot.constants` plus
    the ``zeta`` integral -- the exact formula the kernel replaced.

    Both spellings drive the same ``zeta``, so what this reference isolates is
    the algebra of the antiderivative, not the quadrature inside it; the
    quadrature has its own accuracy gate in ``test_biotzeta``.
    """
    c = Constants(rs, zs, r4, z4)
    zt = zeta(c.rs, c.r, c.gamma, (np.pi / 2) * np.ones_like(rs))
    cphi = -1 / 3 * c.r**2 * np.pi / 2 * c.sign(c.gamma) * (c.sign(c.rs - c.r) + 1)
    aphi = (
        cphi
        + c.gamma * c.r * zt
        + c.gamma * c.a / (6 * c.r) * (c.U * c.K - 2 * c.rs * c.E)
        + c.gamma / (6 * c.a * c.r) * c.p_sum(c.Pphi, c.Pi)
    )
    br = (
        c.r * zt
        - c.a / (2 * c.r) * c.rs * (c.E - c.v * c.K)
        - 1 / (4 * c.a * c.r) * c.p_sum(c.Qr, c.Pi)
    )
    bz = (
        3 / c.r * cphi
        + 2 * c.gamma * zt
        - c.a / (2 * c.r) * 3 / 2 * c.gamma * c.k2 * c.K
        - 1 / (4 * c.a * c.r) * c.p_sum(c.Qz, c.Pi)
    )
    return aphi, br, bz


def test_corner_fields_matches_reference_formula():
    """The single corner antiderivative reproduces the Part III formula exactly.

    Held at a few ulp rather than a physics tolerance: with the quadrature
    shared between the two spellings, any drift here is the algebra diverging,
    and that should never cost more than the last couple of bits.
    """
    stacks = _corner_stacks(_A, _Z0, _DA, _DZ, _TR, _TZ)
    got = corner_fields(*stacks)
    ref = _reference_corner_fields(*stacks)
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_array_max_ulp(g, r, maxulp=4)


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
        assert max(worst) < 1e-9, worst


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
        assert max(worst) < 1e-9, worst


def test_a_target_on_the_source_radius_line_survives_the_corner():
    """Both confluences at once -- the corner itself -- stays on the oracle.

    Sliding along ``rs = r`` through a corner drives the modulus complement, the
    near pole and the third pole to zero together, where the first kind diverges
    logarithmically and the reduction's total weight on it is zero.  Two decades
    looser than the plane tests because the antiderivative is a cancellation of
    terms of order the squared major radius here, and it is reached rather than
    approached: the last row is the vertex exactly.
    """
    for radius in (_FACE_A - _FACE_DA / 2, _FACE_A + _FACE_DA / 2):
        for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
            worst = _worst_against_the_polygon(
                np.full(_STRADDLE.shape, radius), level + _STRADDLE
            )
            assert max(worst) < 1e-7, worst


def test_the_corner_planes_carry_no_jump_of_their_own():
    """The two one-sided limits meet each other, and meet the value ON the plane.

    Oracle-free, and the sharpest statement of the defect this guards: the branch
    terms are odd in the offset, so a mis-set branch survives as a jump that does
    NOT fall with the standoff.  Differencing the kernel against itself over a
    ladder of standoffs is what exposes that without reference to any other kernel
    -- and the value on the plane must be the meeting point rather than either
    side, since a discontinuous antiderivative evaluated at its own jump lands on
    the mean of the two limits and passes a two-sided comparison on its own.
    """
    for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
        for radius in (_INSIDE_SPAN, _FACE_A + 0.021):
            got = cylinder_greens(
                np.full(_STRADDLE.shape, radius),
                level + _STRADDLE,
                _FACE_A,
                _FACE_Z0,
                _FACE_DA,
                _FACE_DZ,
            )
            middle = len(_LADDER)
            for component in got:
                scale = np.max(np.abs(component))
                closest = np.abs(component[middle - 1] - component[middle + 1])
                on_plane = np.abs(component[middle] - component[middle + 1])
                assert closest < 1e-9 * scale
                assert on_plane < 1e-9 * scale


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
    infinity. What both replace is a finite number: capping the modulus and flooring
    the squared distance used to return the kernel's value for a target 1.4 um away,
    with nothing in the result to say so.
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
