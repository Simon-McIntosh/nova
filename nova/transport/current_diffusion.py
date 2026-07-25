"""1D flux-surface-averaged resistive current diffusion -- the temporal prior.

The poloidal flux on a flux-surface-averaged radial coordinate obeys

    toc dpsi/dt = d/drho [ (g2 g3 / rho) dpsi/drho ],
    toc = sigma_parallel mu0 16 pi^2 Phi_b^2 rho / F^2,

with ``psi`` the TOTAL poloidal flux [Wb] (the spine's ``Phi = 2 pi R A_phi``
convention), ``rho`` the normalised toroidal-flux coordinate
``sqrt(Phi_tor / Phi_tor,boundary)``, a regularity condition on axis, and the
measured plasma current as the edge gradient condition

    dpsi/drho|_1 = Ip 16 pi^3 mu0 Phi_b / (D_1 F_1).

The formulation follows the TORAX finite-volume psi equation, which is a
reference implementation this module is verified against in form and never a
runtime dependency.

What the module provides:

* :class:`EtaProfile` -- the bounded low-degree-of-freedom parallel resistivity
  profile, the equation's only genuine unknown, fitted ACROSS shots and never
  per-slice;
* :class:`FluxSurfaceGeometry` -- the 1D metrics of one equilibrium on a uniform
  ``rho`` face grid, plus the Ampere identity that reads an enclosed-current
  profile back out of a flux profile;
* :func:`diffuse_psi` -- the implicit theta-scheme flux-diffusion step, a jitted
  fp64 Thomas solve per sub-step;
* :func:`predicted_current` -- the evolved state's flux-surface-averaged toroidal
  current density ``dI/dS`` and its ohmic parallel current
  ``<J.B> = sigma_parallel <E.B>``;
* :func:`basis_projection_images` / :func:`project_coefficients` -- the predicted
  profiles projected back onto a profile-coefficient ladder as a non-negative
  least squares.  The two current targets weight the pressure-gradient and
  diamagnetic families DIFFERENTLY, so the parallel Ohm's law -- pinned by the
  measured flux evolution -- carries split information no single-slice magnetics
  fit has;
* :func:`flux_budget` / :func:`ejima_coefficient` -- the consumption ledger.  The
  surface flux swing decomposes structurally into RESISTIVE consumption (the axis
  swing: Ohm's law at the axis, since no flux is stored inside a zero-volume
  surface) plus INDUCTIVE internal storage, and the Ejima coefficient falls out.

All inputs are raw measurements or the spine's own fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nova.biot.greens import MU0
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

_TWO_PI = 2.0 * np.pi
_16PI3 = 16.0 * np.pi**3
_16PI2 = 16.0 * np.pi**2


@dataclass(frozen=True)
class EtaProfile:
    """Bounded monotone parallel-resistivity profile ``eta(psi_n)`` [Ohm m].

    ``eta(psi_n) = eta0 exp(contrast psi_n**shape)`` -- an axis value, a
    non-negative edge/axis log-contrast, and a shape exponent.  The family
    brackets the Spitzer ``eta ~ T_e**-1.5`` form for peaked-to-broad temperature
    profiles while staying finite at the separatrix, where the
    temperature-derived form diverges.  Cross-shot parameters; never per-slice.
    """

    eta0: float = 5.0e-8
    contrast: float = 2.0
    shape: float = 2.0

    BOUNDS = ((1.0e-9, 3.0e-6), (0.0, 8.0), (0.3, 4.0))

    def __call__(self, psi_n: np.ndarray) -> np.ndarray:
        """Return the resistivity at the given normalised poloidal flux."""
        normalised = np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0)
        return self.eta0 * np.exp(self.contrast * normalised**self.shape)

    def as_vector(self) -> np.ndarray:
        """Return the parameters as an optimiser vector (log axis value)."""
        return np.array(
            [np.log10(self.eta0), self.contrast, self.shape], dtype=np.float64
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> EtaProfile:
        """Build a profile from an optimiser vector, clipped into the bounds."""
        low, high = zip(*cls.BOUNDS, strict=True)
        return cls(
            eta0=float(np.clip(10.0 ** float(vector[0]), low[0], high[0])),
            contrast=float(np.clip(vector[1], low[1], high[1])),
            shape=float(np.clip(vector[2], low[2], high[2])),
        )


@dataclass(frozen=True)
class FluxSurfaceGeometry:
    """1D metrics of one equilibrium on a uniform ``rho`` grid.

    Face arrays sit on ``rho`` in [0, 1] inclusive (``n_rho + 1`` of them); cell
    arrays midway between them.  ``psi_face`` is the TOTAL poloidal flux [Wb] on
    the faces -- the initial condition of a diffusion step -- and ``phi_b`` the
    boundary toroidal flux [Wb].  ``psi_n_face`` maps ``rho`` back to the
    equilibrium's normalised poloidal flux, which is what the resistivity profile
    and the coefficient basis shapes are keyed to.

    ``flux_sign`` is +1 when psi increases outward and -1 on the opposite
    convention (axis flux above boundary flux with a positive toroidal current).
    It normalises the Ampere identity and the current boundary condition, so a
    caller always passes and reads a POSITIVE plasma current whichever convention
    the equilibrium carries.
    """

    rho_face: np.ndarray
    rho_cell: np.ndarray
    psi_face: np.ndarray
    psi_n_face: np.ndarray
    psi_n_cell: np.ndarray
    vpr_face: np.ndarray
    vpr_cell: np.ndarray
    g2_face: np.ndarray
    g3_face: np.ndarray
    g3_cell: np.ndarray
    f_face: np.ndarray
    f_cell: np.ndarray
    b2_cell: np.ndarray
    inv_r_cell: np.ndarray
    phi_b: float
    r0: float
    ip_amperes: float
    axis_psi: float
    boundary_psi: float
    volume: float
    q_face: np.ndarray
    flux_sign: float = 1.0

    @property
    def d_face(self) -> np.ndarray:
        """Diffusion face coefficient ``g2 g3 / rho`` (regular limit 0 on axis)."""
        coefficient = np.zeros_like(self.rho_face)
        coefficient[1:] = self.g2_face[1:] * self.g3_face[1:] / self.rho_face[1:]
        return coefficient

    def ip_edge_gradient(self, ip_amperes: float) -> float:
        """``dpsi/drho`` at the edge carrying the given total current.

        ``ip_amperes`` is the measured (positive-normalised) current; the stored
        ``flux_sign`` orients the gradient to the equilibrium's own convention.
        """
        return (
            self.flux_sign
            * ip_amperes
            * (_16PI3 * MU0 * self.phi_b)
            / (self.d_face[-1] * self.f_face[-1])
        )

    def enclosed_current(self, psi_face: np.ndarray) -> np.ndarray:
        """Enclosed current on the faces from the flux gradient (Ampere).

        Built from the SCHEME'S OWN stable discrete fluxes: the midpoint gradients
        with midpoint-averaged diffusion and F metrics give the enclosed current at
        cell centres, averaged back onto interior faces (axis exactly zero, edge
        linearly extrapolated).  A pointwise gradient inversion is exact for a
        constructed initial state but amplifies odd-even components of an evolved
        state into current oscillations; the midpoint read is second-order and
        unconditionally stable.
        """
        psi = np.asarray(psi_face, dtype=np.float64)
        drho = float(self.rho_face[1] - self.rho_face[0])
        d_mid = 0.5 * (self.d_face[:-1] + self.d_face[1:])
        f_mid = 0.5 * (self.f_face[:-1] + self.f_face[1:])
        i_mid = (
            self.flux_sign
            * d_mid
            * (np.diff(psi) / drho)
            * f_mid
            / (self.phi_b * _16PI3 * MU0)
        )
        i_face = np.empty_like(psi)
        i_face[0] = 0.0
        i_face[1:-1] = 0.5 * (i_mid[:-1] + i_mid[1:])
        i_face[-1] = 1.5 * i_mid[-1] - 0.5 * i_mid[-2]
        return i_face


def poloidal_field_energy_li(geometry: FluxSurfaceGeometry) -> float:
    """Internal inductance ``li3 = 4 Wpol / (mu0 Ip^2 R0)`` from the 1D state."""
    dpsi = np.gradient(geometry.psi_face, geometry.rho_face, edge_order=2)
    vpr = np.clip(geometry.vpr_face, 1e-30, None)
    b_poloidal2 = (dpsi / _TWO_PI) ** 2 * geometry.g2_face / vpr**2
    b_poloidal2[0] = 0.0
    energy = float(np.trapezoid(b_poloidal2 * geometry.vpr_face, geometry.rho_face)) / (
        2.0 * MU0
    )
    return 4.0 * energy / (MU0 * geometry.ip_amperes**2 * geometry.r0)


# ---------------------------------------------------------------------------
# the diffusion step
# ---------------------------------------------------------------------------
def _thomas_solve(sub, diag, sup, rhs):
    """Solve a tridiagonal system by the Thomas algorithm (fixed-shape scans).

    ``sub[i]`` multiplies ``x[i-1]`` and ``sup[i]`` multiplies ``x[i+1]``, so
    ``sub[0]`` and ``sup[-1]`` are unused.  Two ``lax.scan`` sweeps replace the
    dense factorisation: O(n) instead of O(n^3), and jit / vmap / grad-safe.
    """

    def eliminate(carry, row):
        diag_prev, rhs_prev, sup_prev = carry
        sub_i, diag_i, sup_i, rhs_i = row
        factor = sub_i / diag_prev
        diag_i = diag_i - factor * sup_prev
        rhs_i = rhs_i - factor * rhs_prev
        return (diag_i, rhs_i, sup_i), (diag_i, rhs_i)

    _carry, swept = jax.lax.scan(
        eliminate,
        (diag[0], rhs[0], sup[0]),
        (sub[1:], diag[1:], sup[1:], rhs[1:]),
    )
    diag_swept = jnp.concatenate([diag[:1], swept[0]])
    rhs_swept = jnp.concatenate([rhs[:1], swept[1]])

    def substitute(x_next, row):
        diag_i, sup_i, rhs_i = row
        x_i = (rhs_i - sup_i * x_next) / diag_i
        return x_i, x_i

    x_edge = rhs_swept[-1] / diag_swept[-1]
    _carry, x_rest = jax.lax.scan(
        substitute,
        x_edge,
        (diag_swept[:-1], sup[:-1], rhs_swept[:-1]),
        reverse=True,
    )
    return jnp.concatenate([x_rest, x_edge[jnp.newaxis]])


def _diffusion_step(psi_in, d_face, d_mid, toc_face, drho, dt, grad_edge, theta):
    """One implicit theta-scheme sub-step of the flux diffusion.

    Vertex-centred finite volume on the face grid: an interior point owns a cell
    of width ``drho`` with fluxes ``D dpsi/drho`` at the midpoints, while the END
    points own HALF cells so the physical boundary flux sits exactly at ``rho = 0``
    (regularity: zero) and ``rho = 1`` (the prescribed enclosed current AT the
    boundary, not half a cell outside it).
    """
    lam = dt / (toc_face * drho * drho)
    zero = jnp.zeros(1, dtype=psi_in.dtype)

    sub = jnp.concatenate(
        [zero, -theta * lam[1:-1] * d_mid[:-1], -theta * 2.0 * lam[-1:] * d_mid[-1:]]
    )
    sup = jnp.concatenate(
        [-theta * 2.0 * lam[:1] * d_mid[:1], -theta * lam[1:-1] * d_mid[1:], zero]
    )
    diag = jnp.concatenate(
        [
            1.0 + theta * 2.0 * lam[:1] * d_mid[:1],
            1.0 + theta * lam[1:-1] * (d_mid[:-1] + d_mid[1:]),
            1.0 + theta * 2.0 * lam[-1:] * d_mid[-1:],
        ]
    )
    rhs = psi_in.at[-1].add(2.0 * lam[-1] * d_face[-1] * grad_edge * drho)
    explicit = jnp.concatenate(
        [
            2.0 * d_mid[:1] * (psi_in[1:2] - psi_in[:1]),
            d_mid[1:] * (psi_in[2:] - psi_in[1:-1])
            - d_mid[:-1] * (psi_in[1:-1] - psi_in[:-2]),
            -2.0 * d_mid[-1:] * (psi_in[-1:] - psi_in[-2:-1]),
        ]
    )
    rhs = rhs + (1.0 - theta) * lam * explicit
    return _thomas_solve(sub, diag, sup, rhs)


def _diffuse_scan(psi0, d_face, d_mid, toc_face, drho, intervals, grad_edge, theta):
    """Scan the diffusion step over the sub-step intervals, recording voltages."""

    def step(psi, row):
        dt, gradient = row
        advanced = _diffusion_step(
            psi, d_face, d_mid, toc_face, drho, dt, gradient, theta
        )
        # a repeated sample carries no evolution and no loop voltage
        moving = dt > 0.0
        psi_next = jnp.where(moving, advanced, psi)
        safe_dt = jnp.where(moving, dt, 1.0)
        voltage = jnp.where(moving, (psi_next - psi) / safe_dt, 0.0)
        return psi_next, (psi_next, voltage[0], voltage[-1])

    _final, history = jax.lax.scan(step, psi0, (intervals, grad_edge))
    return history


def diffuse_psi(
    geometry: FluxSurfaceGeometry,
    eta,
    *,
    t_grid: np.ndarray,
    ip_of_t: np.ndarray,
    psi0_face: np.ndarray | None = None,
    theta: float = 1.0,
) -> dict:
    """Integrate the flux diffusion over ``t_grid`` with the current condition.

    ``t_grid`` ``(n_t,)`` are the sub-step times [s] and ``ip_of_t`` the measured
    plasma current at those times [A].  ``eta`` is any callable mapping
    normalised poloidal flux to resistivity [Ohm m] -- an :class:`EtaProfile`, or
    a table when an external conductivity model supplies one.

    The geometry is FROZEN over the interval (metrics from the starting slice); a
    recorded approximation the soft prior downstream absorbs.  Implicit
    theta-scheme (``theta = 1`` backward Euler), one jitted fp64 tridiagonal solve
    per sub-step.

    Returns ``psi_face`` ``(n_t, n_rho + 1)`` [Wb] plus the flux-budget traces:
    ``v_axis`` and ``v_bdry``, the axis and boundary ``dpsi/dt`` [V], and
    ``psidot_face`` at the final step for the ohmic parallel current.
    """
    times = np.asarray(t_grid, dtype=np.float64)
    ip = np.asarray(ip_of_t, dtype=np.float64)
    n_face = geometry.rho_face.size
    drho = float(geometry.rho_face[1] - geometry.rho_face[0])

    sigma_cell = 1.0 / np.asarray(eta(geometry.psi_n_cell), dtype=np.float64)
    toc_cell = (
        sigma_cell
        * MU0
        * _16PI2
        * geometry.phi_b**2
        * geometry.rho_cell
        / geometry.f_cell**2
    )
    # face-centred state: interpolate the time coefficient onto the faces
    toc_face = np.empty(n_face)
    toc_face[1:-1] = 0.5 * (toc_cell[:-1] + toc_cell[1:])
    toc_face[0] = toc_cell[0]
    toc_face[-1] = toc_cell[-1]

    d_face = geometry.d_face
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])
    psi0 = np.asarray(
        geometry.psi_face if psi0_face is None else psi0_face, dtype=np.float64
    )
    intervals = np.diff(times)
    gradients = np.array([geometry.ip_edge_gradient(float(value)) for value in ip[1:]])

    if intervals.size:
        psi_history, v_axis_history, v_bdry_history = _COMPILED_SCAN(
            jnp.asarray(psi0),
            jnp.asarray(d_face),
            jnp.asarray(d_mid),
            jnp.asarray(toc_face),
            drho,
            jnp.asarray(intervals),
            jnp.asarray(gradients),
            float(theta),
        )
        psi_face = np.vstack([psi0, np.asarray(psi_history)])
        v_axis = np.concatenate([[0.0], np.asarray(v_axis_history)])
        v_bdry = np.concatenate([[0.0], np.asarray(v_bdry_history)])
    else:
        psi_face = psi0[np.newaxis, :]
        v_axis = np.zeros(1)
        v_bdry = np.zeros(1)

    if times.size > 1 and (times[-1] - times[-2]) > 0:
        psidot_face = (psi_face[-1] - psi_face[-2]) / (times[-1] - times[-2])
    else:
        psidot_face = np.zeros(n_face)
    return {
        "t": times,
        "psi_face": psi_face,
        "v_axis": v_axis,
        "v_bdry": v_bdry,
        "psidot_face": psidot_face,
    }


with skip_import("jax"):
    #: the traced/compiled diffusion scan; ``drho`` and ``theta`` are static so a
    #: change of scheme retraces rather than silently reusing a stale kernel
    _COMPILED_SCAN = jax.jit(_diffuse_scan, static_argnums=(4, 7))


# ---------------------------------------------------------------------------
# predicted profiles + basis projection
# ---------------------------------------------------------------------------
def predicted_current(
    geometry: FluxSurfaceGeometry,
    psi_face: np.ndarray,
    psidot_face: np.ndarray,
    eta,
) -> dict:
    """Evolved-state current profiles on the cell grid.

    ``j_tor`` is ``dI/dS`` with ``dS = vpr <1/R> drho / 2pi``, the
    flux-surface-averaged toroidal current density.  ``j_par_b`` is the OHMIC
    ``<J.B> = sigma_parallel <E.B>`` with
    ``<E.B> = flux_sign psidot F <1/R^2> / 2pi``.  The ``flux_sign`` factor keeps
    the dissipative channel POSITIVE for a positive normalised current in either
    flux convention -- without it the ohmic target flips sign on one convention and
    a non-negative projection can only reach it by annihilating the profile.
    """
    i_face = geometry.enclosed_current(psi_face)
    surface_per_rho = geometry.vpr_cell * geometry.inv_r_cell / _TWO_PI
    j_tor = np.diff(i_face) / (
        np.diff(geometry.rho_face) * np.clip(surface_per_rho, 1e-30, None)
    )
    psidot_cell = 0.5 * (psidot_face[:-1] + psidot_face[1:])
    sigma_cell = 1.0 / np.asarray(eta(geometry.psi_n_cell), dtype=np.float64)
    e_dot_b = (
        geometry.flux_sign * psidot_cell * geometry.f_cell * geometry.g3_cell / _TWO_PI
    )
    return {"i_face": i_face, "j_tor": j_tor, "j_par_b": sigma_cell * e_dot_b}


#: exponents of the non-negative monomial profile ladder ``(1 - psi_n)**e``
NONNEGATIVE_EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def profile_shapes(psi_n: np.ndarray, n_terms: int, *, nonneg: bool) -> np.ndarray:
    """Profile-ladder shape functions on a normalised-flux sample.

    The non-negative arm is the monomial family ``(1 - psi_n)**e``, each term
    edge-vanishing and sign-definite; the general arm is the Legendre family on
    the unit interval times the edge-vanishing factor.  Returns
    ``(psi_n.size, n_terms)``.
    """
    normalised = np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0)
    if nonneg:
        return np.column_stack(
            [
                (1.0 - normalised) ** NONNEGATIVE_EXPONENTS[term]
                for term in range(n_terms)
            ]
        )
    from numpy.polynomial import legendre

    scaled = 2.0 * normalised - 1.0
    edge = 1.0 - normalised
    return np.column_stack(
        [
            legendre.legval(scaled, [0.0] * term + [1.0]) * edge
            for term in range(n_terms)
        ]
    )


def basis_projection_images(
    geometry: FluxSurfaceGeometry,
    coefficient_scale: np.ndarray,
    *,
    n_pressure: int,
    n_diamagnetic: int,
    nonneg: bool = True,
) -> dict:
    """Per-unit-coefficient current images on the geometry's cell grid.

    ``coefficient_scale`` holds the source fit's per-column amplitude scales, so
    a stored normalised coefficient vector maps back to a physical current.  For
    the pressure-gradient family (drive ``R/R0``) the images are
    ``j_tor = s phi / (R0 <1/R>)`` and ``<J.B> = F s phi / R0``; for the
    diamagnetic family (drive ``R0/R``) they are
    ``j_tor = s phi R0 <1/R^2> / <1/R>`` and ``<J.B> = <B^2> R0 s phi / F``.  The
    differing F weights are the split leverage the parallel Ohm's law provides.
    Both matrices are ``(n_rho, n_pressure + n_diamagnetic)``.
    """
    psi_n = geometry.psi_n_cell
    phi_pressure = profile_shapes(psi_n, n_pressure, nonneg=nonneg)
    phi_diamagnetic = profile_shapes(psi_n, n_diamagnetic, nonneg=nonneg)
    scale = np.asarray(coefficient_scale, dtype=np.float64)
    scale_pressure = scale[:n_pressure]
    scale_diamagnetic = scale[n_pressure:]
    inv_r = np.clip(geometry.inv_r_cell, 1e-9, None)
    f_cell = geometry.f_cell
    a_tor = np.hstack(
        [
            phi_pressure
            * scale_pressure[np.newaxis, :]
            / (geometry.r0 * inv_r[:, np.newaxis]),
            phi_diamagnetic
            * scale_diamagnetic[np.newaxis, :]
            * geometry.r0
            * (geometry.g3_cell / inv_r)[:, np.newaxis],
        ]
    )
    a_par = np.hstack(
        [
            phi_pressure
            * scale_pressure[np.newaxis, :]
            * (f_cell / geometry.r0)[:, np.newaxis],
            phi_diamagnetic
            * scale_diamagnetic[np.newaxis, :]
            * (geometry.b2_cell * geometry.r0 / f_cell)[:, np.newaxis],
        ]
    )
    return {"a_tor": a_tor, "a_par": a_par}


def project_coefficients(
    geometry: FluxSurfaceGeometry,
    images: dict,
    j_tor_predicted: np.ndarray,
    j_par_b_predicted: np.ndarray,
    *,
    nonneg: bool = True,
    parallel_weight: float = 1.0,
    ridge: float = 1e-6,
) -> np.ndarray | None:
    """Predicted profile-ladder coefficients from the evolved 1D profiles.

    Volume-weighted least squares over the cell grid with both current targets
    stacked, each block normalised to a unit root-mean-square target so
    ``parallel_weight`` is a dimensionless balance, and non-negativity per the
    source fit's arm.  Returns ``None`` on a degenerate solve -- the caller then
    skips the prior for that slice rather than fabricating one.
    """
    from scipy import optimize

    weight = np.sqrt(np.clip(geometry.vpr_cell, 0.0, None))
    weight = weight / max(np.linalg.norm(weight), 1e-30)

    def block(matrix, target, block_weight):
        scale = max(float(np.sqrt(np.mean((weight * target) ** 2))), 1e-30)
        gain = np.sqrt(block_weight) / scale
        return gain * (matrix * weight[:, np.newaxis]), gain * (target * weight)

    a_tor, y_tor = block(
        images["a_tor"], np.asarray(j_tor_predicted, dtype=np.float64), 1.0
    )
    a_par, y_par = block(
        images["a_par"],
        np.asarray(j_par_b_predicted, dtype=np.float64),
        parallel_weight,
    )
    n_terms = a_tor.shape[1]
    matrix = np.vstack([a_tor, a_par, np.sqrt(ridge) * np.eye(n_terms)])
    target = np.concatenate([y_tor, y_par, np.zeros(n_terms)])
    try:
        if nonneg:
            coefficients = optimize.lsq_linear(
                matrix,
                target,
                bounds=(np.zeros(n_terms), np.full(n_terms, np.inf)),
            ).x
        else:
            coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    except ValueError, np.linalg.LinAlgError:
        return None
    return coefficients if np.isfinite(coefficients).all() else None


# ---------------------------------------------------------------------------
# flux-consumption ledger
# ---------------------------------------------------------------------------
def flux_budget(step: dict, geometry: FluxSurfaceGeometry) -> dict:
    """Inductive/resistive decomposition of an interval's flux consumption.

    Working in the equilibrium's own sign, over the integration window:

    * ``d_psi_bdry`` -- the total surface flux swing [Wb];
    * ``d_psi_axis`` -- the RESISTIVE consumption [Wb].  The axis loop voltage is
      purely resistive: no flux is stored inside a zero-volume surface;
    * ``d_psi_internal`` -- the INDUCTIVE storage change [Wb].

    The identity ``d_psi_bdry = d_psi_axis + d_psi_internal`` holds by
    construction; reporting all three keeps both channels explicitly accounted.
    """
    psi = step["psi_face"]
    d_axis = float(psi[-1, 0] - psi[0, 0])
    d_bdry = float(psi[-1, -1] - psi[0, -1])
    multi_step = psi.shape[0] > 1
    return {
        "d_psi_bdry": d_bdry,
        "d_psi_axis": d_axis,
        "d_psi_internal": d_bdry - d_axis,
        "v_axis_mean": float(np.mean(step["v_axis"][1:])) if multi_step else 0.0,
        "v_bdry_mean": float(np.mean(step["v_bdry"][1:])) if multi_step else 0.0,
        "r0": geometry.r0,
    }


def ejima_coefficient(d_psi_resistive: float, d_ip: float, r0: float) -> float:
    """Windowed Ejima coefficient ``|dPsi_res| / (mu0 R0 |dIp|)``.

    The resistive poloidal-flux consumption normalised by ``mu0 R0 dIp`` over the
    same window.  The classical definition runs from breakdown, so a chained
    window starting at the first reconstructed slice gives the INCREMENTAL
    coefficient over the covered ramp -- comparable across shots, and comparable
    to literature values when the window covers most of the ramp.
    """
    return float(abs(d_psi_resistive) / (MU0 * abs(r0) * max(abs(d_ip), 1e-30)))


@dataclass
class CurrentDiffusion:
    """Resistive current-diffusion solver over one frozen-geometry interval.

    The stateful face over this module: hold the interval's 1D metrics and the
    resistivity profile, then :meth:`evolve` the flux, :meth:`predict` the current
    profiles, and :meth:`project` them back onto a coefficient ladder.
    """

    geometry: FluxSurfaceGeometry
    eta: EtaProfile
    theta: float = 1.0

    def evolve(
        self,
        t_grid: np.ndarray,
        ip_of_t: np.ndarray,
        psi0_face: np.ndarray | None = None,
    ) -> dict:
        """Diffuse the flux across the interval."""
        return diffuse_psi(
            self.geometry,
            self.eta,
            t_grid=t_grid,
            ip_of_t=ip_of_t,
            psi0_face=psi0_face,
            theta=self.theta,
        )

    def predict(self, step: dict) -> dict:
        """Current profiles of the evolved state at the end of the interval."""
        return predicted_current(
            self.geometry, step["psi_face"][-1], step["psidot_face"], self.eta
        )

    def project(
        self,
        prediction: dict,
        coefficient_scale: np.ndarray,
        *,
        n_pressure: int,
        n_diamagnetic: int,
        nonneg: bool = True,
        parallel_weight: float = 1.0,
    ) -> np.ndarray | None:
        """Project the predicted profiles onto the profile-ladder coefficients."""
        images = basis_projection_images(
            self.geometry,
            coefficient_scale,
            n_pressure=n_pressure,
            n_diamagnetic=n_diamagnetic,
            nonneg=nonneg,
        )
        return project_coefficients(
            self.geometry,
            images,
            prediction["j_tor"],
            prediction["j_par_b"],
            nonneg=nonneg,
            parallel_weight=parallel_weight,
        )

    def budget(self, step: dict) -> dict:
        """Flux-consumption ledger of the interval."""
        return flux_budget(step, self.geometry)


__all__ = [
    "NONNEGATIVE_EXPONENTS",
    "CurrentDiffusion",
    "EtaProfile",
    "FluxSurfaceGeometry",
    "basis_projection_images",
    "diffuse_psi",
    "ejima_coefficient",
    "flux_budget",
    "poloidal_field_energy_li",
    "predicted_current",
    "profile_shapes",
    "project_coefficients",
]
