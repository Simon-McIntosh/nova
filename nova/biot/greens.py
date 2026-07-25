"""Canonical axisymmetric Green's functions (numpy + scipy.special only).

This is the single source of the analytic axisymmetric Biot-Savart kernels the
spine evaluates:

* point circular filament -- :func:`greens_psi`, :func:`greens_bz_br` (loop of
  unit current, log-singular at the source);
* rectangular finite section -- :func:`cylinder_greens` (uniform current spread
  over a rectangular cross-section, smooth everywhere including inside the
  conductor);
* second-moment corrected filament -- :func:`quadrupole_filament` (centroid
  filament plus the section's quadrupole term, the only far-field form that
  converges to a finite section for a full ring);
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
(Carlson forms) plus a 1-D ``zeta`` quadrature (fixed-node rule on an arcsinh
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

from nova.biot.zeta import zeta

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


def _zeta(rs: np.ndarray, r: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """The zeta integral over the full arc half-angle range.

    zeta = integral arcsinh((rs - r cos phi)/sqrt(gamma^2 + r^2 sin^2 phi)) dalpha
    over alpha in [0, pi/2] with phi = pi - 2 alpha -- the one non-closed-form
    piece of the cylinder antiderivative.  Delegates to the shared fixed-node
    quadrature so the cylinder and bow kernels evaluate one and the same rule.
    """
    return zeta(rs, r, gamma, np.pi / 2.0)


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


# --- second-moment corrected filament ---------------------------------


def section_centroid(vertices: np.ndarray) -> np.ndarray:
    """Return the polygon's ``(r, z)`` AREA centroid [m].

    The area centroid, not the mean of the vertices: the two coincide only for a
    section with a symmetry that pairs its corners, and a filament placed at the
    vertex mean of a trapezoidal or wall-clipped section carries a first-moment
    (dipole) error that no second-moment correction can absorb.
    """
    v = np.asarray(vertices, dtype=np.float64)
    r, z = v[:, 0], v[:, 1]
    r_next, z_next = np.roll(r, -1), np.roll(z, -1)
    cross = r * z_next - r_next * z
    area = 0.5 * cross.sum()
    return np.array(
        [
            float(np.sum((r + r_next) * cross) / (6.0 * area)),
            float(np.sum((z + z_next) * cross) / (6.0 * area)),
        ]
    )


def second_moments(vertices: np.ndarray) -> tuple[float, float, float]:
    """Return the area-normalised second central moments ``(Irr, Izz, Irz)`` [m^2].

    Closed polygon (shoelace) formulae about the section's area centroid, not a
    sampled approximation: they feed a correction term that is itself only parts
    in ten thousand of the coupling, so a percent-level moment would swamp the
    thing it corrects.
    """
    v = np.asarray(vertices, dtype=np.float64)
    v = v - section_centroid(v)
    r, z = v[:, 0], v[:, 1]
    r_next, z_next = np.roll(r, -1), np.roll(z, -1)
    cross = r * z_next - r_next * z
    area = 0.5 * cross.sum()
    irr = float(np.sum((r**2 + r * r_next + r_next**2) * cross) / 12.0 / area)
    izz = float(np.sum((z**2 + z * z_next + z_next**2) * cross) / 12.0 / area)
    irz = float(
        np.sum((r * z_next + 2.0 * r * z + 2.0 * r_next * z_next + r_next * z) * cross)
        / 24.0
        / area
    )
    return irr, izz, irz


# Source-position step for the curvature difference, in section radii.  The
# truncation error of the second difference falls as step^2 and the round-off
# amplification rises as step^-2; balanced against the smallest scale the ring
# Green's function varies over, the optimum sits a few parts in a thousand of the
# section size, and the resulting correction bottoms out near 1e-10 relative.  A
# step tied to the section rather than fixed keeps that true for a millimetre
# section as well as a metre one.
_MOMENT_STEP = 2.0e-3


def quadrupole_filament(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    step: float | None = None,
    cross_moment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere: centroid filament plus the section quadrupole.

    ``vertices`` -- ``(n, 2)`` polygon ``(r, z)`` corners, either orientation, no
    repeated closing vertex.  Spreading unit current over a finite section shifts
    the coupling by the section's second moment contracted with the curvature of
    the ring Green's function in the SOURCE position,

        f_section = f(centroid) + (1/2) sum_ij m_ij d2f/dx_i dx_j,

    with ``m_ij`` the area-normalised second central moments.  For a full
    toroidal ring that curvature is set by the MAJOR radius rather than by the
    distance to the target, so the bare centroid filament does not converge to
    the section at any standoff -- its relative error flattens onto a floor of
    order ``(a / R0)^2``.  Carrying the quadrupole term removes that floor at
    point-filament cost: five Green's-function evaluations for a section
    symmetric about one of its own axes, nine when the cross moment survives.
    ``cross_moment=False`` forces the five-evaluation diagonal-only form even on
    an asymmetric section, which is how the cross term's contribution is
    isolated.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    centre = section_centroid(v)
    irr, izz, irz = second_moments(v)
    if step is None:
        radius = float(np.max(np.hypot(*(v - centre).T)))
        step = _MOMENT_STEP * radius

    def at(dr: float, dz: float) -> np.ndarray:
        psi = greens_psi(target_r, target_z, centre[0] + dr, centre[1] + dz)
        bz, br = greens_bz_br(target_r, target_z, centre[0] + dr, centre[1] + dz)
        return np.array([psi, br, bz])

    value = at(0.0, 0.0)
    curvature = irr * (at(step, 0.0) - 2.0 * value + at(-step, 0.0)) + izz * (
        at(0.0, step) - 2.0 * value + at(0.0, -step)
    )
    if cross_moment and abs(irz) > 1e-12 * max(abs(irr), abs(izz)):
        curvature += (
            0.5
            * irz
            * (at(step, step) - at(step, -step) - at(-step, step) + at(-step, -step))
        )
    corrected = value + 0.5 * curvature / step**2
    return corrected[0], corrected[1], corrected[2]


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
    "section_centroid",
    "second_moments",
    "quadrupole_filament",
]
