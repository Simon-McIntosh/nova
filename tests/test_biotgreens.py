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
    the midpoint ``zeta`` integral -- the exact formula the kernel replaced.
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
    """The single corner antiderivative reproduces the Part III formula exactly."""
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
