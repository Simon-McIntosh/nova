"""Tests for the canonical axisymmetric Green's-function library.

These run on plain numpy arrays with no frame machinery.  They pin:

* the single corner antiderivative ``corner_fields`` against the Urankar
  Part III rectangular-section formula it was extracted from -- the dataclass
  cylinder kernel and the functional ``cylinder_greens`` both drive this one
  routine, so the guard protects against the two paths silently diverging;
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


def test_the_quadrupole_filament_removes_the_bare_filament_floor():
    """Far out the bare filament stalls on a floor; the corrected one does not.

    The floor is the section second moment weighted by a curvature the MAJOR
    radius sets, so it does not decay with target distance. Measured against the
    exact polygon kernel at ten section radii, relative to the local magnitude.
    """
    from nova.biot.greens import quadrupole_filament
    from nova.biot.polygon import polygon_greens

    vertices = _hexagon()
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    target_r = 6.2 + 0.6 * np.cos(angle)
    target_z = 0.6 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = quadrupole_filament(target_r, target_z, vertices)
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
    from nova.biot.greens import quadrupole_filament

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
        got = np.array(quadrupole_filament(target_r, target_z, _hexagon(radius=radius)))
        correction[radius] = np.abs(got - point) / np.abs(point)
    np.testing.assert_allclose(correction[0.06] / correction[0.006], 100.0, rtol=0.03)
    assert np.max(correction[0.06]) > 1e-4  # real at the shipped plasma cell size


def test_the_quadrupole_filament_carries_the_cross_moment():
    """An asymmetric section needs the off-diagonal moment, so it is applied.

    Dropping the cross term is the cheap five-evaluation path, valid only when the
    section is symmetric about one of its own axes. On a section that is not, the
    term measurably closes the gap to the exact kernel, which is checked here by
    zeroing it: an axis-aligned diagonal-only correction is the worse model.
    """
    from nova.biot.greens import quadrupole_filament, second_moments
    from nova.biot.polygon import polygon_greens

    vertices = _trapezium()
    assert abs(second_moments(vertices)[2]) > 1e-5
    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target_r = 6.2 + 1.0 * np.cos(angle)
    target_z = 1.0 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = quadrupole_filament(target_r, target_z, vertices)
    diagonal = quadrupole_filament(target_r, target_z, vertices, cross_moment=False)
    for index in range(3):
        scale = np.max(np.abs(exact[index]))
        full = np.max(np.abs(corrected[index] - exact[index])) / scale
        without = np.max(np.abs(diagonal[index] - exact[index])) / scale
        assert full < 1e-5
        assert without > 5.0 * full
