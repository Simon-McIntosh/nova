"""Canonical axisymmetric Green's functions (numpy + scipy.special only).

This is the single source of the analytic axisymmetric Biot-Savart kernels the
spine evaluates:

* point circular filament -- :func:`greens_psi`, :func:`greens_bz_br` (loop of
  unit current, log-singular at the source);
* rectangular finite section -- :func:`cylinder_greens` (uniform current spread
  over a rectangular cross-section, smooth everywhere including inside the
  conductor);
* second-moment corrected filament -- :func:`moment_filament` (centroid
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
# Numerical floor so an ON-AXIS point does not divide by zero.  It bounds the ring
# SPAN ``(a + R)^2 + dz^2``, the target radius, and the modulus -- all three of order
# the machine, and all three reached only at the axis, where what they guard is
# masked to its own limit anyway.  It is NOT applied to the target's distance to the
# FILAMENT: an absolute floor on a squared length there engages 32 micrometres out at
# any ring radius -- the radius does not appear -- and answers, silently, for a target
# that far away.
_R_FLOOR = 1.0e-9


# --- point circular filament ------------------------------------------


def _filament_gap(r: np.ndarray, dz: np.ndarray, ar: float) -> np.ndarray:
    """Return ``d^2 = (a - R)^2 + dz^2``, the squared distance to the filament.

    Both kernels below want it: the flux as the modulus COMPLEMENT
    ``k'^2 = d^2/((a + R)^2 + dz^2)``, and the field components as that AND as the
    pole they carry over it.

    The complement is the point.  ``k^2 = 4 a R/((a + R)^2 + dz^2)``, and the two
    radicals differ by exactly ``4 a R``, so ``k'^2`` follows from the geometry with
    no subtraction from one and keeps every digit however close the target sits --
    where a float PARAMETER cannot carry its own complement at all.  Once ``k'^2`` is
    below the spacing of the numbers next to one, ``1 - k'^2`` has rounded it away,
    and ``K``, which grows like ``-log k'``, comes back wrong by about
    ``eps/2 k'^2``.  Measured against an extended-precision reference
    (``benchmarks/near_field_elliptic.py``): the parameter route loses 1e-6 of the
    flux at 2.2e-06 ring radii and 1e-9 at 8.5e-05, and a couple of micrometres from
    a metre-scale ring it is wrong in the second decimal.
    """
    return (ar - r) ** 2 + dz**2


def _held_parameter(k2: np.ndarray) -> np.ndarray:
    """Return the parameter bounded at one, for the SECOND kind alone.

    ``4 a R`` and ``(a + R)^2 + dz^2`` are formed independently, so their ratio can
    land an ulp ABOVE one for a target within about ``1e-8`` ring radii of the
    filament -- and whether it does depends on the radius, which is why a unit ring
    trips it one ULP off the filament and a 6.2 m one does not.  ``E`` is bounded and
    smooth through ``m = 1``, where it is exactly one, so holding it there costs
    nothing anywhere; the FIRST kind, which is the sensitive one, never sees the
    parameter at all.
    """
    return np.minimum(k2, 1.0)


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
    span = np.maximum(denom, _R_FLOOR)
    k2 = 4.0 * ar * r / span
    complement = _filament_gap(r, dz, ar) / span
    k = np.sqrt(k2)
    big_k = scipy.special.ellipkm1(complement)
    big_e = scipy.special.ellipe(_held_parameter(k2))
    pref = 2.0 * MU0 * np.sqrt(ar * np.maximum(r, _R_FLOOR)) / np.maximum(k, _R_FLOOR)
    psi = pref * (0.5 * (1.0 + complement) * big_k - big_e)
    # at R->0 the loop encloses no flux at the axis target -> Phi->0
    return np.where(r < _R_FLOOR, 0.0, psi)


def greens_bz_br(
    rs: np.ndarray, zs: np.ndarray, ar: float, az: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_Z, B_R)`` [T per A] at targets ``(rs, zs)`` from a loop at ``(ar, az)``.

    Standard axisymmetric forms (Jackson section 5.5).  On-axis (``R->0``)
    ``B_R->0`` and ``B_Z`` reduces to the textbook
    ``mu0 a^2 / (2 (a^2 + dz^2)^{3/2})``.

    A target ON the filament returns the divergence, not a number.  The field there
    is genuinely singular, and a filament model asked for it has been asked a
    question it cannot answer -- the finite-section kernels (:func:`cylinder_greens`,
    :func:`~nova.biot.polygon.polygon_greens`, :func:`moment_filament`) exist for
    exactly that target and are smooth through the conductor.  What is NOT acceptable
    is answering for a phantom standoff: a floor on the squared distance, or a cap on
    the parameter, returns the kernel's value some micrometres away with nothing to
    say so, and that is the one failure a caller cannot detect.  One ULP off the
    filament these return the true finite values.
    """
    r = np.asarray(rs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    span = np.maximum(denom, _R_FLOOR)
    sq = np.sqrt(span)
    k2 = 4.0 * ar * r / span
    gap = _filament_gap(r, dz, ar)
    big_k = scipy.special.ellipkm1(gap / span)
    big_e = scipy.special.ellipe(_held_parameter(k2))
    pre = MU0 / (2.0 * np.pi)
    # Both brackets are printed with the pole's numerator as a difference of terms of
    # order a^2 whose value is of order a d -- so it arrives with relative error
    # eps a/2d, which beats the modulus as the near-field limit.  Splitting the pole
    # off removes it: a^2 - R^2 - dz^2 is -d^2 + 2 a (a - R) and a^2 + R^2 + dz^2 is
    # d^2 + 2 a R, both exact, and what is left over d^2 is the divergence itself.
    bz = pre / sq * (big_k - big_e + 2.0 * ar * (ar - r) / gap * big_e)
    br_full = (
        pre
        * dz
        / (np.maximum(r, _R_FLOOR) * sq)
        * (big_e - big_k + 2.0 * ar * r / gap * big_e)
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


# Degree-3-exact barycentric rule on a triangle (weights summing to one), used to
# integrate the cubic monomials of the third moments over a fan triangulation of
# the section.  Exact for the integrand, so the moments carry no quadrature error.
_TRIANGLE_RULE = (
    ((1.0 / 3.0, 1.0 / 3.0), -27.0 / 48.0),
    ((0.2, 0.2), 25.0 / 48.0),
    ((0.6, 0.2), 25.0 / 48.0),
    ((0.2, 0.6), 25.0 / 48.0),
)


def third_moments(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """Return the area-normalised third central moments, ``(Irrr, Irrz, Irzz,
    Izzz)`` [m^3].

    The section's skew.  It vanishes for any section symmetric about its own
    centroid -- a regular hexagonal plasma cell -- and survives for one clipped by
    the first wall, where it is the leading residual of a quadrupole-corrected
    filament.

    Integrated over a fan triangulation from the centroid with a degree-3-exact
    triangle rule, so the result is exact for the cubic integrand (the signed fan
    decomposition is valid for any simple polygon, concave included).
    """
    v = np.asarray(vertices, dtype=np.float64)
    v = v - section_centroid(v)
    start, end = v, np.roll(v, -1, axis=0)
    signed_area = 0.5 * (start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1])
    moment = np.zeros(4)
    for (towards_start, towards_end), weight in _TRIANGLE_RULE:
        point = towards_start * start + towards_end * end
        r, z = point[:, 0], point[:, 1]
        monomial = np.array([r**3, r * r * z, r * z * z, z**3])
        moment += weight * (monomial * signed_area).sum(axis=1)
    return tuple(moment / signed_area.sum())


# Source-position step for the curvature difference, in section radii.  The
# truncation error of the second difference falls as step^2 and the round-off
# amplification rises as step^-2; balanced against the smallest scale the ring
# Green's function varies over, the optimum sits a few parts in a thousand of the
# section size, and the resulting correction bottoms out near 1e-10 relative.  A
# step tied to the section rather than fixed keeps that true for a millimetre
# section as well as a metre one.
_MOMENT_STEP = 2.0e-3


def moment_filament(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    step: float | None = None,
    order: int = 3,
    cross_moment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere: centroid filament corrected by the section moments.

    ``vertices`` -- ``(n, 2)`` polygon ``(r, z)`` corners, either orientation, no
    repeated closing vertex.  Spreading unit current over a finite section shifts
    the coupling by the section's moments contracted with the derivatives of the
    ring Green's function in the SOURCE position,

        f_section = f(centroid) + (1/2) sum_ij m_ij d2f + (1/6) sum_ijk m_ijk d3f,

    with ``m`` the area-normalised central moments.  ``order=2`` keeps the
    quadrupole term alone, ``order=3`` adds the section's skew.

    The quadrupole term is what makes a far-field filament viable at all.  For a
    full toroidal ring the curvature it multiplies is set by the MAJOR radius
    rather than by the distance to the target, so a BARE centroid filament does
    not converge to the section at any standoff -- its relative error flattens
    onto a floor of order ``(a / R0)^2``.  Carrying the quadrupole removes that
    floor for five Green's-function evaluations, nine when the cross moment
    survives.

    The skew term matters only for a section without centroid symmetry.  A
    regular hexagonal plasma cell has no third moments and pays nothing for the
    request; a cell clipped by the first wall does, and there the quadrupole form
    alone stalls two orders above one part in a million at the distances a
    far-field band starts at -- a third-order residual whose decay is too slow for
    a wider band to recover.  Thirteen evaluations with it.

    ``cross_moment=False`` forces the diagonal-only quadrupole even on an
    asymmetric section, which is how the cross term's contribution is isolated.
    Derivatives are central differences on a step tied to the section scale; see
    ``_MOMENT_STEP``.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    centre = section_centroid(v)
    irr, izz, irz = second_moments(v)
    radius = float(np.max(np.hypot(*(v - centre).T)))
    if step is None:
        step = _MOMENT_STEP * radius

    evaluated: dict[tuple[int, int], np.ndarray] = {}

    def at(dr: int, dz: int) -> np.ndarray:
        """Return ``(psi, Br, Bz)`` with the source offset by whole steps."""
        if (dr, dz) not in evaluated:
            source_r = centre[0] + dr * step
            source_z = centre[1] + dz * step
            psi = greens_psi(target_r, target_z, source_r, source_z)
            bz, br = greens_bz_br(target_r, target_z, source_r, source_z)
            evaluated[dr, dz] = np.array([psi, br, bz])
        return evaluated[dr, dz]

    value = at(0, 0)
    scale = max(abs(irr), abs(izz))
    correction = 0.5 * (
        irr * (at(1, 0) - 2.0 * value + at(-1, 0))
        + izz * (at(0, 1) - 2.0 * value + at(0, -1))
    )
    if cross_moment and abs(irz) > 1e-12 * scale:
        correction += 0.25 * irz * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
    correction /= step**2

    if order >= 3:
        mrrr, mrrz, mrzz, mzzz = third_moments(v)
        # a section symmetric about its centroid has none, to round-off; asking
        # for the term then costs nothing rather than adding difference noise
        if max(abs(mrrr), abs(mrrz), abs(mrzz), abs(mzzz)) > 1e-12 * radius**3:
            skew = (
                mrrr * (at(2, 0) - 2.0 * at(1, 0) + 2.0 * at(-1, 0) - at(-2, 0))
                + mzzz * (at(0, 2) - 2.0 * at(0, 1) + 2.0 * at(0, -1) - at(0, -2))
                + 3.0
                * mrrz
                * (
                    at(1, 1)
                    - 2.0 * at(0, 1)
                    + at(-1, 1)
                    - at(1, -1)
                    + 2.0 * at(0, -1)
                    - at(-1, -1)
                )
                + 3.0
                * mrzz
                * (
                    at(1, 1)
                    - 2.0 * at(1, 0)
                    + at(1, -1)
                    - at(-1, 1)
                    + 2.0 * at(-1, 0)
                    - at(-1, -1)
                )
            )
            correction += skew / (12.0 * step**3)

    corrected = value + correction
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
    "moment_filament",
]
