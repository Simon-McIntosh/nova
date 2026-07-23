"""Finite-area Green's functions: complete toroidal conductors of POLYGON section.

Generalises the rectangular-section kernel in :mod:`nova.biot.greens`
(:func:`~nova.biot.greens.cylinder_greens`, Urankar Part III) to an arbitrary
polygon cross-section, from L. K. Urankar, *"Vector potential and magnetic
field of current-carrying finite arc segment in analytical form -- Part V:
polygon cross section,"* IEEE Trans. Magn. 26(3), 1171-1180 (1990).

Why: a slanted (parallelogram) or trapezoidal conductor -- the vessel end
crowns and stability-plate arms, a non-rectangular coil pack -- is neither a
filled axis-aligned box (the rectangle kernel's assumption) nor cheaply
represented by a multi-filament tiling (O(N) cost, Riemann-limited accuracy).
Urankar converts the cross-section surface integral into a CONTOUR sum over
the polygon's edges (Stokes), does the edge-parameter integral in closed form,
and leaves -- for the axisymmetric full turn -- a single smooth 1-D integral
over the arc angle phi per edge.

The vector potential.  Per edge nu with endpoints (r'v1, z'v1) -> (r'v2, z'v2)
[paper eqs (7)-(9a)]: parametrise the edge r'(u) = r1 + b1 u with u = z' - z,
slope b1 = dr/dz, intercept r1 = r'v1 - b1 (z'v1 - z).  The u-integral of eq
(8) has the closed antiderivative (eq 9, with a0^2 = 1 + b1^2, G^2 = u^2 +
r^2 sin^2 phi, B^2 = (r1 - r cos phi)^2 + a0^2 r^2 sin^2 phi, D^2 = G^2 +
(r' - r cos phi)^2, Gamma = u + b1 (r' - r cos phi), beta1 = (r' - r cos phi)/G,
beta2 = Gamma/B, beta3 = [u(r' - r cos phi) - b1 G^2]/(r sin phi D)):

    g(u, phi) = Gamma D/(2 a0^2) + u r cos phi arsinh beta1
              + [B^2 + 2 a0^2 r cos phi (r1 - r cos phi)]/(2 a0^3) arsinh beta2
              - (r^2/2) sin 2phi arctan beta3

NOTE the 1990 typesetting trap: the printed "Gamma(phi)/2a0^2 D(phi)" means
Gamma D/(2 a0^2) -- D(phi) is a NUMERATOR factor.  ``g`` reproduces the raw
cross-section integral of r'/D dr'dz' edge-by-edge to machine precision
(regression-pinned against a dense 2-D quadrature and against the rectangular
kernel).

Then A_phi(r, z) = -sum_nu integral cos phi [g] dphi (eqs 3b, 10a/b, j = phi),
and the axisymmetric flux psi = 2 pi mu0 R A_phi / (4 pi A) per ampere of total
current.

The field.  Rather than transcribe the paper's closed B integrands (eq 11b --
a longer, more error-prone form), the field is the EXACT curl of the verified
vector potential, B_Z = (1/2piR) dpsi/dR and B_R = -(1/2piR) dpsi/dZ, evaluated
by COMPLEX-STEP differentiation of psi.  Complex-step (df/dx = Im f(x + ih)/h
with h ~ 1e-30) is exact to machine precision -- it has none of the subtractive
cancellation of a real finite difference -- so this is analytic
differentiation, not a numerical approximation, and it guarantees psi<->B
consistency by construction.  For a rectangle it reproduces
``cylinder_greens``' B to ~1e-15.

For the FULL TURN (axisymmetric ring, arc = 2pi) the phi-integrand is even
about phi = pi, so it is evaluated on [0, pi] and doubled, with composite
Gauss-Legendre panels.  The integrand is analytic for every target off the
section boundary -- including INSIDE the conductor -- because D^2 >= r^2 sin^2
phi > 0 at the interior quadrature nodes; convergence is spectral.  With the
default 16x48 rule the field is machine-precise (~1e-13 relative) for targets
more than ~1 cm off the section boundary and holds to <=1e-6 down to ~1 mm;
sub-millimetre standoffs (finer than any physical sensor) recover full accuracy
by raising ``n_panels`` / ``n_nodes`` (both exposed).  A target lying exactly on
an edge or vertex stays finite because the complex-step increment nudges the
evaluation off the real singularity.  This mirrors the rectangle kernel, which
carries a 785-point arcsinh (zeta) quadrature inside its "closed"
antiderivative, so a smooth bounded 1-D quadrature per edge is the established
cost model.

Sign/units conventions match :func:`nova.biot.greens.cylinder_greens` (and
hence the point-filament ``greens_psi``/``greens_bz_br``), per ampere of TOTAL
conductor current, with uniform azimuthal current density J_phi = 1/A:

    psi [Wb/A] = 2 pi mu0 R A_phi / (4 pi A)
    B   [T/A]  = curl of psi.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

MU0 = 4.0e-7 * np.pi

# Composite Gauss-Legendre rule on phi in [0, pi] (doubled by even symmetry):
# the integrand is analytic on the open interval, so a modest panel count
# converges past 1e-12.  16 panels x 48 nodes reproduce the rectangular kernel
# to ~1e-11.
_N_PANELS = 16
_N_NODES = 48
_CSTEP = 1e-30  # complex-step increment (d via Im part; no cancellation)


def _phi_rule(n_panels: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n_nodes)
    edges = np.linspace(0.0, np.pi, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    phi = (0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)).ravel()
    wts = (0.5 * (hi - lo) * w[None, :]).ravel()
    return phi, wts


def _psi_hat(
    r: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    cosp: np.ndarray,
    sinp: np.ndarray,
    sin2p: np.ndarray,
    w_cos: np.ndarray,
    sign: float,
    area: float,
) -> np.ndarray:
    """Complex-analytic psi(r, z) per ampere from the verified edge antiderivative.

    ``r, z`` are ``(T, 1)`` (possibly complex, for the complex-step curl); the
    phi node arrays are ``(Q,)``.  Returns ``(T,)`` -- real for real inputs.
    """
    n = len(v)
    a_hat = np.zeros(r.shape[0], dtype=np.result_type(r.dtype, z.dtype))
    z_scale = max(float(np.ptp(v[:, 1])), 1e-6)
    for i in range(n):
        ra, za = v[i]
        rb, zb = v[(i + 1) % n]
        dz = zb - za
        if abs(dz) < 1e-12 * z_scale:
            continue  # horizontal edge: f_nu(phi) vanishes (paper eq 7a)
        b1 = (rb - ra) / dz
        a02 = 1.0 + b1 * b1
        a03 = a02 * np.sqrt(a02)
        r1 = ra - b1 * (za - z)  # (T, 1) -- depends on z
        for u, s_lim in ((zb - z, 1.0), (za - z, -1.0)):
            rp = r1 + b1 * u
            rmc = rp - r * cosp
            r1mc = r1 - r * cosp
            g2 = u * u + (r * sinp) ** 2
            b2 = r1mc * r1mc + a02 * (r * sinp) ** 2
            d = np.sqrt(g2 + rmc * rmc)
            cap_gamma = u + b1 * rmc
            ash1 = np.arcsinh(rmc / np.sqrt(g2))
            ash2 = np.arcsinh(cap_gamma / np.sqrt(b2))
            at3 = np.arctan((u * rmc - b1 * g2) / (r * sinp * d))
            g = (
                cap_gamma * d / (2.0 * a02)
                + u * r * cosp * ash1
                + (b2 + 2.0 * a02 * r * cosp * r1mc) / (2.0 * a03) * ash2
                - 0.5 * r * r * sin2p * at3
            )
            # -[g] over u limits ua..ub  ->  -g(ub)(+1) - g(ua)(-1); fold the
            # +/-1 into s_lim.
            a_hat += -s_lim * (g @ w_cos)
    a_hat *= 2.0  # [0, pi] half-turn x2
    norm = sign * MU0 / (4.0 * np.pi * area)
    return 2.0 * np.pi * r[:, 0] * norm * a_hat


def polygon_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    n_panels: int = _N_PANELS,
    n_nodes: int = _N_NODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere at targets, from a polygon-section ring.

    ``vertices`` -- (n, 2) array of the section's (r, z) corners, either
    orientation, no repeated closing vertex.  Returns arrays shaped like
    ``target_r``: total poloidal flux psi [Wb/A] and field components [T/A],
    smooth everywhere including inside the conductor.  Horizontal edges
    (dz = 0) contribute nothing (paper eq 7a) and are skipped.
    """
    v = np.asarray(vertices, dtype=np.float64)
    tr = np.asarray(target_r, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    shape = tr.shape
    r = tr.ravel()[:, None]
    z = tz.ravel()[:, None]

    rolled = np.roll(v, -1, axis=0)
    signed_area2 = float(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1]))
    area = 0.5 * abs(signed_area2)
    # the counter-clockwise edge sum yields -f(phi); one orientation sign fixes
    # all three components at once (pinned by the rectangle-reduction and
    # filament oracles in tests/test_biotpolygon.py).
    sign = -np.sign(signed_area2)

    phi, wts = _phi_rule(n_panels, n_nodes)
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    sin2p = np.sin(2.0 * phi)
    w_cos = wts * cosp

    def psi_at(rr: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return _psi_hat(rr, zz, v, cosp, sinp, sin2p, w_cos, sign, area)

    # one complex-step pass in r gives psi (real part) and dpsi/dR (imag/h ->
    # B_Z); one in z gives dpsi/dZ (imag/h -> B_R).  Exact-to-machine-precision
    # curl.
    h = _CSTEP
    psi_r = psi_at(r + 1j * h, z)
    dpsi_dz = psi_at(r, z + 1j * h).imag / h
    psi = psi_r.real
    dpsi_dr = psi_r.imag / h
    two_pi_r = 2.0 * np.pi * r[:, 0]
    bz = dpsi_dr / two_pi_r
    br = -dpsi_dz / two_pi_r
    return psi.reshape(shape), br.reshape(shape), bz.reshape(shape)


__all__ = ["polygon_greens", "MU0"]
