"""Canonical axisymmetric Green's functions (numpy + scipy.special only).

This is the single source of the analytic axisymmetric Biot-Savart kernels the
spine evaluates:

* point circular filament -- :func:`greens_psi`, :func:`greens_bz_br` (loop of
  unit current, log-singular at the source);
* rectangular finite section -- :func:`cylinder_greens` (uniform current spread
  over a rectangular cross-section, smooth everywhere including inside the
  conductor);
* near/far blend -- :func:`hybrid_greens` (finite-area form near the section,
  cheap point-filament form beyond a standoff band).

The arbitrary-polygon section generalisation lives in :mod:`nova.biot.polygon`
(:func:`~nova.biot.polygon.polygon_greens`) and shares the sign/units contract
below.  All kernels return quantities per ampere of TOTAL conductor current in
SI units.

Why finite-area: a point-filament Green's function is log-singular at the
source, so any evaluation grid that approaches a conductor -- in-vessel PF
winding packs, or the plasma current cells a Grad-Shafranov solve distributes
current over -- inherits a spurious near-field spike.  The finite-area kernel
spreads unit current uniformly over the cross-section and is smooth everywhere,
which is what a psi field read for topology (axis / X-points / LCFS) requires.

Formulation (rectangular section): closed-form antiderivatives of the
uniformly-distributed ring current -- complete elliptic integrals K, E, Pi
(Carlson forms) plus a 1-D ``zeta`` quadrature (midpoint rule on an arcsinh
integrand, L. K. Urankar Part III) -- evaluated at the four cross-section
corners and combined with alternating signs (the standard definite-double-
integral corner rule), normalised per ampere of total conductor current:

    psi = 2 pi mu0 R * Aphi_corner / (2 pi A)     [Wb/A]
    B   = mu0 * {Br,Bz}_corner / (2 pi A)          [T/A]

with A the cross-section area.  The far-field limit of ``cylinder_greens``
matches the point-filament ``greens_psi``/``greens_bz_br`` (pinned by test).
"""

from __future__ import annotations

import numpy as np
import scipy.special  # type: ignore[import-untyped]

MU0 = 4.0e-7 * np.pi
"""Vacuum permeability [T.m/A]."""

_EPS = 2.0 * np.finfo(float).eps
# Numerical floor so on-axis / coincident points don't divide by zero.
_R_FLOOR = 1.0e-9


# --- point circular filament ------------------------------------------


def greens_psi(rs: np.ndarray, zs: np.ndarray, ar: float, az: float) -> np.ndarray:
    """Total poloidal flux ``Phi`` [Wb per A] at targets from a loop at ``(ar, az)``.

    Axisymmetric circular filament of unit current -- the flux *threading* the
    observation loop, ``Phi = 2 pi R A_phi`` with the standard vector potential

        A_phi(R, Z) = (mu0 / (pi k)) sqrt(a / R) [(1 - k^2/2) K(k^2) - E(k^2)]

    so

        Phi(R, Z) = (2 mu0 / k) sqrt(a R) [(1 - k^2/2) K(k^2) - E(k^2)]

    with ``k^2 = 4 a R / ((a + R)^2 + (Z - az)^2)``.  This is the TOTAL flux (Wb)
    threading the observation loop -- NOT the stream function ``Phi/2 pi`` -- and
    it is consistent with :func:`greens_bz_br` via ``B_Z = (1/(2 pi R)) dPhi/dR``
    and ``B_R = -(1/(2 pi R)) dPhi/dZ``.
    """
    r = np.asarray(rs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    k2 = 4.0 * ar * r / np.maximum(denom, _R_FLOOR)
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-12)
    k = np.sqrt(k2)
    big_k = scipy.special.ellipk(k2)
    big_e = scipy.special.ellipe(k2)
    pref = 2.0 * MU0 * np.sqrt(ar * np.maximum(r, _R_FLOOR)) / np.maximum(k, _R_FLOOR)
    psi = pref * ((1.0 - 0.5 * k2) * big_k - big_e)
    # at R->0 the loop encloses no flux at the axis target -> Phi->0
    return np.where(r < _R_FLOOR, 0.0, psi)


def greens_bz_br(
    rs: np.ndarray, zs: np.ndarray, ar: float, az: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_Z, B_R)`` [T per A] at targets ``(rs, zs)`` from a loop at ``(ar, az)``.

    Standard axisymmetric forms (Jackson section 5.5).  On-axis (``R->0``)
    ``B_R->0`` and ``B_Z`` reduces to the textbook
    ``mu0 a^2 / (2 (a^2 + dz^2)^{3/2})``.
    """
    r = np.asarray(rs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    sq = np.sqrt(np.maximum(denom, _R_FLOOR))
    k2 = 4.0 * ar * r / np.maximum(denom, _R_FLOOR)
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-12)
    big_k = scipy.special.ellipk(k2)
    big_e = scipy.special.ellipe(k2)
    d2 = (ar - r) ** 2 + dz**2
    pre = MU0 / (2.0 * np.pi)
    bz = pre / sq * (big_k + (ar**2 - r**2 - dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
    br_full = (
        pre
        * dz
        / (np.maximum(r, _R_FLOOR) * sq)
        * (-big_k + (ar**2 + r**2 + dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
    )
    br = np.where(r < _R_FLOOR, 0.0, br_full)
    return bz, br


# --- rectangular finite section ---------------------------------------


def _sign(x: np.ndarray) -> np.ndarray:
    """Sign with a dead-band: 0 within numerical noise of zero."""
    return np.where(np.abs(x) > 1e4 * _EPS, np.sign(x), 0.0)


def _ellipp(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Complete elliptic integral of the 3rd kind via Carlson symmetric forms."""
    x = np.zeros_like(n)
    y = 1.0 - m
    z = np.ones_like(n)
    p = 1.0 - n
    rf = scipy.special.elliprf(x, y, z)
    rj = scipy.special.elliprj(x, y, z, p)
    return rf + rj * n / 3.0


def _zeta(
    rs: np.ndarray, r: np.ndarray, gamma: np.ndarray, *, points: int = 785
) -> np.ndarray:
    """The zeta integral: midpoint quadrature of the arcsinh integrand.

    zeta = integral arcsinh((rs - r cos phi)/sqrt(gamma^2 + r^2 sin^2 phi)) dalpha
    over alpha in [0, pi/2] with phi = pi - 2 alpha -- the one non-closed-form
    piece of the cylinder antiderivative.  ``points`` matches the reference
    resolution (500 per unit alpha -> 785 for pi/2).
    """
    alpha_max = np.pi / 2.0
    dalpha = alpha_max / (points - 1)
    alpha = np.linspace(0.0, alpha_max, points)[:-1] + dalpha / 2.0  # midpoints
    phi = np.pi - 2.0 * alpha  # (Q,)
    sin2 = np.sin(phi) ** 2
    cosphi = np.cos(phi)
    # broadcast: (..., 1) against (Q,)
    g2 = gamma[..., None] ** 2 + r[..., None] ** 2 * sin2
    integrand = np.arcsinh((rs[..., None] - r[..., None] * cosphi) / np.sqrt(g2))
    return dalpha * integrand.sum(axis=-1)


def corner_fields(
    rs: np.ndarray, zs: np.ndarray, r: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Antiderivative coefficients (Aphi_hat, Br_hat, Bz_hat) at one corner set.

    All inputs broadcast to the same shape ``(..., 4)`` -- target coordinates
    repeated over the four source-section corners.  This is the shared
    axisymmetric corner antiderivative: :func:`cylinder_greens` combines it
    over one ring's four corners, and the dataclass cylinder kernel drives it
    with per-source corner stacks.
    """
    gamma = zs - z
    a2 = gamma**2 + (rs + r) ** 2
    a = np.sqrt(a2)
    b = rs + r
    c2 = gamma**2 + r**2
    c = np.sqrt(c2)
    k2 = (1.0 - _EPS) * 4.0 * r * rs / a2
    v = 1.0 + k2 * (gamma**2 - b * r) / (2.0 * r * rs)
    ellip_k = scipy.special.ellipk(k2)
    ellip_e = scipy.special.ellipe(k2)
    u_coef = k2 * (4.0 * gamma**2 + 3.0 * rs**2 - 5.0 * r**2) / (4.0 * r)

    np2 = {
        1: 2.0 * r / (r - c - _EPS),
        2: (1.0 - _EPS) * 2.0 * r / (r + c),
        3: (1.0 - _EPS) * 4.0 * r * rs / b**2,
    }
    pi3 = {p: _ellipp(np2[p], k2) for p in (1, 2, 3)}

    qr = {p: (rs - (-1.0) ** p * c) * np2[p] * gamma**2 * c / r for p in (1, 2)}
    qr[3] = np.zeros_like(r)
    qz = {p: (rs - (-1.0) ** p * c) * -2.0 * gamma * c * np2[p] for p in (1, 2)}
    qz[3] = gamma * b * (rs - r) * np2[3]
    pphi = {
        p: (rs - (-1.0) ** p * c) * np2[p] * c * (3.0 * r**2 - c2) / (2.0 * r)
        for p in (1, 2)
    }
    pphi[3] = -rs / b * (rs - r) * (3.0 * r**2 - rs**2)

    def p_sum(coef: dict[int, np.ndarray]) -> np.ndarray:
        out = np.zeros_like(coef[1])
        for p in (1, 2, 3):
            out += (-1.0) ** p * coef[p] * pi3[p]
        return out

    cphi = -1.0 / 3.0 * r**2 * np.pi / 2.0 * _sign(gamma) * (_sign(rs - r) + 1.0)
    dz_coef = 3.0 / r * cphi
    zeta = _zeta(rs, r, gamma)

    aphi_hat = (
        cphi
        + gamma * r * zeta
        + gamma * a / (6.0 * r) * (u_coef * ellip_k - 2.0 * rs * ellip_e)
        + gamma / (6.0 * a * r) * p_sum(pphi)
    )
    br_hat = (
        r * zeta
        - a / (2.0 * r) * rs * (ellip_e - v * ellip_k)
        - 1.0 / (4.0 * a * r) * p_sum(qr)
    )
    bz_hat = (
        dz_coef
        + 2.0 * gamma * zeta
        - a / (2.0 * r) * 1.5 * gamma * k2 * ellip_k
        - 1.0 / (4.0 * a * r) * p_sum(qz)
    )
    return aphi_hat, br_hat, bz_hat


def cylinder_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    a: float,
    z0: float,
    da: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere at targets, from a rectangular-section ring.

    ``a, z0`` -- section centroid [m]; ``da, dz`` -- radial/vertical extents [m].
    Returns arrays shaped like ``target_r``: total poloidal flux psi [Wb/A] and
    field components [T/A], smooth everywhere including inside the section.
    """
    tr = np.asarray(target_r, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    # corner order (matching the reference): (-,-), (+,-), (+,+), (-,+)
    rs = np.stack(
        [np.full(tr.shape, a + d * da / 2.0) for d in (-1, 1, 1, -1)], axis=-1
    )
    zs = np.stack(
        [np.full(tr.shape, z0 + d * dz / 2.0) for d in (-1, -1, 1, 1)], axis=-1
    )
    r4 = np.repeat(tr[..., None], 4, axis=-1)
    z4 = np.repeat(tz[..., None], 4, axis=-1)

    aphi_hat, br_hat, bz_hat = corner_fields(rs, zs, r4, z4)
    area = da * dz

    def corner(data: np.ndarray) -> np.ndarray:
        return (
            1.0
            / (2.0 * np.pi * area)
            * ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0]))
        )

    aphi = corner(aphi_hat)
    psi = 2.0 * np.pi * MU0 * tr * aphi
    br = MU0 * corner(br_hat)
    bz = MU0 * corner(bz_hat)
    return psi, br, bz


def hybrid_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    a: float,
    z0: float,
    da: float,
    dz: float,
    *,
    switch: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere: cylinder near the section, point filament far.

    Beyond ``switch * max(da, dz)`` from the section centroid the finite-area
    correction is the constant second-moment term (approx +da^2/12a0^2 relative
    in psi -- sub-0.2% for typical section sizes, far below measurement noise),
    so the cheap point-filament loop formulas are used there; within the band
    the full cylinder form keeps the field smooth and finite through the
    conductor.  Cost scales with the (small) number of near-band targets.
    """
    tr = np.asarray(target_r, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    psi = greens_psi(tr, tz, a, z0)
    bz, br = greens_bz_br(tr, tz, a, z0)
    near = np.hypot(tr - a, tz - z0) < switch * max(da, dz)
    if near.any():
        psi_n, br_n, bz_n = cylinder_greens(tr[near], tz[near], a, z0, da, dz)
        psi = psi.copy()
        br = br.copy()
        bz = bz.copy()
        psi[near] = psi_n
        br[near] = br_n
        bz[near] = bz_n
    return psi, br, bz


__all__ = [
    "MU0",
    "greens_psi",
    "greens_bz_br",
    "corner_fields",
    "cylinder_greens",
    "hybrid_greens",
]
