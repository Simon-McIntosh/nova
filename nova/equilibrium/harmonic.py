"""Source-free toroidal-harmonic reconstruction in the vacuum annulus.

In the vacuum annulus between the plasma and the sensors :math:`j_\\phi = 0`, so
the poloidal flux solves the *homogeneous* Grad-Shafranov operator exactly,

.. math::
    \\Delta^* \\psi = \\frac{\\partial^2 \\psi}{\\partial R^2}
        - \\frac{1}{R}\\frac{\\partial \\psi}{\\partial R}
        + \\frac{\\partial^2 \\psi}{\\partial Z^2} = 0 .

A free interior cell current, and equally a low-order current-moment fit, both
reconstruct a current whose :math:`j_\\phi` is non-zero in the vacuum region and
so *violate* the premise the boundary read rests on. This module instead
represents the plasma-produced flux directly in a basis that cannot commit any
current to the annulus: the toroidal harmonics (ring functions) that separate
:math:`\\Delta^* \\psi = 0` about a fixed pole placed near the magnetic axis.

Representation
--------------
About a toroidal-coordinate pole (a focal ring of radius ``pole_r`` at height
``pole_z``) the homogeneous operator separates, and its solutions carry the
**order-1** half-integer-degree Legendre functions — order 1, not the scalar
Laplace order 0, because of the :math:`-(1/R)\\,\\partial_R` term — with a
:math:`\\sqrt{\\cosh\\eta - \\cos\\theta}` prefactor:

.. math::
    \\psi_{n}(\\eta, \\theta) = \\sqrt{\\cosh\\eta - \\cos\\theta}\\;
        P^1_{n-1/2}(\\cosh\\eta)\\;\\{\\cos n\\theta,\\ \\sin n\\theta\\} .

The plasma current sits *inside* the pole and the flux is observed outward in the
annulus toward the sensors, so the physical radial set is the one that stays
regular in the source-free region out to infinity (:math:`\\eta \\to 0`). That
selects :math:`P` over :math:`Q`, which is pinned here by the filament-recovery
test: the :math:`P` set reproduces the exact exterior flux of current filaments
inside the pole at *held-out* annulus points, which a set diverging at infinity
cannot do.

The expansion is physical only in the annulus. Toward the focal ring the ring
functions diverge, so the reconstructed flux blows up inside the plasma where the
expansion does not hold; :meth:`ReconstructHarmonic.mask_invalid_interior`
replaces that disc with a confined-side plateau so a boundary read sees a clean
interior and the physical field outside. The interior itself belongs to whichever
model carries the plasma current.

Gauge
-----
The absolute level of the read is only weakly pinned — there is no constant
column and a handful of flux loops carry the DC — so two gauge tools are
provided. The poloidal-circulation anchor ties
:math:`\\oint \\mathbf{B}_{pol}\\cdot d\\mathbf{l} = \\mu_0 I_p`, which fixes the
monopole without leaning only on the flux loops, and the annulus soft prior can
be written in gradient form, which is manifestly invariant to an additive
constant on :math:`\\psi`.

Conventions: every column carries the total poloidal flux
:math:`\\Phi = 2 \\pi R A_\\phi` [Wb], so a flux loop reads :math:`\\psi` and a
field probe reads the projected field from
:math:`B_R = -(1/2\\pi R)\\,\\partial_Z \\Phi`,
:math:`B_Z = +(1/2\\pi R)\\,\\partial_R \\Phi`. The known conductor field is added
separately by the caller through the finite-area cylinder kernels in
:mod:`nova.biot.greens`; no point-filament conductor term is ever introduced
here.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk

from nova.equilibrium.measurement import (
    Magnetics,
    SliceMeasurement,
    whitened_solve,
)

_DISTANCE_FLOOR = 1.0e-12
"""Numerical floor on the product of focal distances (the pole itself)."""


def toroidal_coordinates(
    r: np.ndarray, z: np.ndarray, pole_r: float, pole_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return toroidal coordinates about a focal ring at ``(pole_r, pole_z)``.

    The forward transform is

    .. math::
        R = \\frac{a \\sinh\\eta}{\\cosh\\eta - \\cos\\theta}, \\quad
        Z = z_0 + \\frac{a \\sin\\theta}{\\cosh\\eta - \\cos\\theta},
        \\quad a = \\mathrm{pole\\_r},

    whose self-consistent inverse, in terms of the two focal distances
    :math:`d_1^2 = (R - a)^2 + (Z - z_0)^2` and
    :math:`d_2^2 = (R + a)^2 + (Z - z_0)^2` with :math:`D = d_1 d_2` and
    :math:`P = R^2 + (Z - z_0)^2`, is

    .. math::
        \\cosh\\eta = \\frac{P + a^2}{D}, \\quad
        \\cos\\theta = \\frac{P - a^2}{D}, \\quad
        \\sin\\theta = \\frac{2 a (Z - z_0)}{D}, \\quad
        \\cosh\\eta - \\cos\\theta = \\frac{2 a^2}{D} .

    The focal ring is :math:`\\eta \\to \\infty` (:math:`D \\to 0`); the symmetry
    axis and spatial infinity are both :math:`\\eta \\to 0`.

    Returns ``(cosh_eta, cos_theta, sin_theta, theta, cosh_eta_minus_cos_theta)``.
    """
    a = float(pole_r)
    r = np.asarray(r, dtype=np.float64)
    dz = np.asarray(z, dtype=np.float64) - float(pole_z)
    near = (r - a) ** 2 + dz**2
    far = (r + a) ** 2 + dz**2
    distance = np.sqrt(np.maximum(near * far, _DISTANCE_FLOOR**2))
    squared = r**2 + dz**2
    cosh_eta = np.maximum((squared + a**2) / distance, 1.0)
    cos_theta = (squared - a**2) / distance
    sin_theta = 2.0 * a * dz / distance
    return (
        cosh_eta,
        cos_theta,
        sin_theta,
        np.arctan2(sin_theta, cos_theta),
        2.0 * a**2 / distance,
    )


def ring_legendre_p1(order: int, x: np.ndarray) -> np.ndarray:
    """Return :math:`P^1_{n-1/2}(x)` for ``n = 0..order`` at ``x = cosh eta >= 1``.

    Built from the order-0 ring-function elliptic-integral seeds

    .. math::
        P_{-1/2}(x) = \\frac{2}{\\pi}\\sqrt{\\frac{2}{x+1}}\\,K(m), \\quad
        P_{1/2}(x) = \\frac{2}{\\pi}\\left[\\sqrt{2(x+1)}\\,E(m)
                     - \\sqrt{\\frac{2}{x+1}}\\,K(m)\\right],
        \\quad m = \\frac{x-1}{x+1}

    (:func:`scipy.special.ellipk` and :func:`~scipy.special.ellipe` take the
    parameter :math:`m = k^2`, not the modulus), climbed in degree by the stable
    forward recurrence :math:`(\\nu+1)P_{\\nu+1} = (2\\nu+1) x P_\\nu - \\nu
    P_{\\nu-1}` — :math:`P` is the dominant, forward-stable solution — then raised
    to order 1 by :math:`P^1_\\nu = (x^2-1)^{-1/2}\\nu(x P_\\nu - P_{\\nu-1})` with
    the degree reflection :math:`P_{-3/2} = P_{1/2}`.

    Closed-form elliptic integrals rather than a general hypergeometric
    evaluation are what make a per-slice pole affordable: the whole ladder comes
    out of one vectorised call.

    Returns ``(order + 1, x.size)``.
    """
    x = np.asarray(x, dtype=np.float64)
    # x = 1 is the axis / spatial infinity, where P^1 vanishes; clamp just off it
    # so the (x^2 - 1)^{-1/2} order raise stays finite (the far field is masked).
    x = np.maximum(x, 1.0 + 1e-12)
    parameter = (x - 1.0) / (x + 1.0)
    complete_k = ellipk(parameter)
    complete_e = ellipe(parameter)
    low = np.sqrt(2.0 / (x + 1.0))
    high = np.sqrt(2.0 * (x + 1.0))
    degree = [
        2.0 / np.pi * low * complete_k,
        2.0 / np.pi * (high * complete_e - low * complete_k),
    ]
    for index in range(1, order + 1):
        nu = index - 0.5
        degree.append(
            ((2.0 * nu + 1.0) * x * degree[index] - nu * degree[index - 1]) / (nu + 1.0)
        )
    inverse_sinh = 1.0 / np.sqrt(x * x - 1.0)
    out = np.empty((order + 1, x.size), dtype=np.float64)
    for n in range(order + 1):
        nu = n - 0.5
        previous = degree[1] if n == 0 else degree[n - 1]  # P_{-3/2} = P_{1/2}
        out[n] = nu * (x * degree[n] - previous) * inverse_sinh
    return out


@dataclass(frozen=True)
class HarmonicConfig:
    """Configuration of the source-free toroidal-harmonic annulus fit."""

    pole_r: float = 0.9
    """Focal-ring radius [m], placed near the nominal magnetic axis."""

    pole_z: float = 0.0
    """Focal-ring height [m]."""

    order: int = 3
    """Highest harmonic index ``n``; the basis holds ``2 * order + 1`` columns."""

    ridge: float = 1e-8
    """Numerical floor on the column-normalised normal equations."""

    ip_anchor: bool = False
    """Add the poloidal-circulation gauge tie to the total plasma current."""

    ip_anchor_weight: float = 1.0
    """Strength of the gauge tie in the relative-whitened frame."""

    sobolev_exponent: float = 0.0
    """Graded ridge exponent: the penalty on a degree-``n`` coefficient scales as
    ``(1 + n) ** sobolev_exponent``.

    This is the machine-agnostic cure for truncation ringing. A generous
    ``order`` resolves the boundary shape — elongation, triangularity, X-point
    sharpness — while the graded penalty suppresses the high-mode ripple an
    unregularised high-order fit rings with, so neither order truncation nor
    boundary-curve smoothing (both of which round X-points) is needed.
    """


def harmonic_labels(order: int) -> list[str]:
    """Return the column labels of the harmonic basis of a given order."""
    labels = ["h0"]
    for n in range(1, order + 1):
        labels += [f"h{n}c", f"h{n}s"]
    return labels


def mode_penalty(order: int, exponent: float) -> np.ndarray:
    """Return the per-column ridge multiplier ``(1 + degree) ** exponent``.

    Column order matches :func:`harmonic_labels`: the degree-0 term, then the
    cosine and sine of each degree. ``exponent = 0`` gives a uniform ridge.
    """
    degree = [0] + [n for n in range(1, order + 1) for _ in range(2)]
    return (1.0 + np.asarray(degree, dtype=np.float64)) ** float(exponent)


def harmonic_columns(
    r: np.ndarray, z: np.ndarray, config: HarmonicConfig
) -> tuple[np.ndarray, list[str]]:
    """Return the source-free flux columns ``(n_point, 2 * order + 1)``.

    Column ``k`` is a flux :math:`\\psi_k(R, Z)` with
    :math:`\\Delta^* \\psi_k = 0` away from the pole: the
    :math:`\\sqrt{\\cosh\\eta - \\cos\\theta}` prefactor times the order-1
    half-integer Legendre radial function times the angular factor.

    The total-flux convention puts an explicit :math:`R` factor on each column:
    the order-1 toroidal harmonic is :math:`A_\\phi`, and the measured quantity
    is :math:`\\Phi = 2 \\pi R A_\\phi` (the :math:`2\\pi` is absorbed into the
    fitted coefficient). Without that factor the columns solve the
    :math:`A_\\phi` equation rather than the flux equation and cannot represent a
    flux-loop signature at all.
    """
    r = np.asarray(r, dtype=np.float64)
    cosh_eta, _cos_theta, _sin_theta, theta, prefactor = toroidal_coordinates(
        r, z, config.pole_r, config.pole_z
    )
    prefactor = r * np.sqrt(np.maximum(prefactor, 0.0))
    radial = ring_legendre_p1(config.order, cosh_eta)
    columns, labels = [], []
    for n in range(config.order + 1):
        base = prefactor * radial[n]
        columns.append(base * np.cos(n * theta))
        labels.append("h0" if n == 0 else f"h{n}c")
        if n >= 1:
            columns.append(base * np.sin(n * theta))
            labels.append(f"h{n}s")
    return np.stack(columns, axis=1), labels


def harmonic_field_columns(
    r: np.ndarray, z: np.ndarray, config: HarmonicConfig, *, step: float = 1.0e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(B_R, B_Z)`` columns of each harmonic flux column.

    Central differences of the analytic, smooth, source-free flux columns give
    the field under the total-flux convention
    :math:`B_R = -(1/2\\pi R)\\,\\partial_Z\\Phi`,
    :math:`B_Z = +(1/2\\pi R)\\,\\partial_R\\Phi`.
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    dpsi_dr = (
        harmonic_columns(r + step, z, config)[0]
        - harmonic_columns(r - step, z, config)[0]
    ) / (2.0 * step)
    dpsi_dz = (
        harmonic_columns(r, z + step, config)[0]
        - harmonic_columns(r, z - step, config)[0]
    ) / (2.0 * step)
    radius = np.maximum(r[:, None], _DISTANCE_FLOOR)
    return -dpsi_dz / (2.0 * np.pi * radius), dpsi_dr / (2.0 * np.pi * radius)


def grad_shafranov_residual(
    psi: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    """Return :math:`\\Delta^* \\psi` on a ``(n_z, n_r)`` field, edges as NaN.

    Second-order central differences on the uniform raster. This is the
    correctness oracle for the basis: every harmonic column must drive it to the
    truncation floor, independently of where the analytic formula came from.
    """
    psi = np.asarray(psi, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    out = np.full_like(psi, np.nan)
    psi_rr = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / dr**2
    psi_r = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dr)
    psi_zz = (psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / dz**2
    out[1:-1, 1:-1] = (
        psi_rr[1:-1, :] - psi_r[1:-1, :] / r[1:-1][None, :] + psi_zz[:, 1:-1]
    )
    return out


@dataclass
class HarmonicInversion:
    """One slice's toroidal-harmonic annulus fit."""

    coefficients: np.ndarray
    labels: list[str]
    misfit: float
    """Whitened mean-square sensor residual over the trusted rows; the gauge tie
    never enters it."""

    config: HarmonicConfig
    covariance: np.ndarray | None = field(default=None, repr=False)

    def flux_on_grid(self, grid_r: np.ndarray, grid_z: np.ndarray) -> np.ndarray:
        """Return the plasma harmonic flux ``(n_z, n_r)`` [Wb], no conductor term."""
        return harmonic_flux_on_grid(self.config, self.coefficients, grid_r, grid_z)


def harmonic_flux_on_grid(
    config: HarmonicConfig,
    coefficients: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    """Return the plasma harmonic flux ``(n_z, n_r)`` [Wb] on a grid raster.

    Row index is Z and column index is R, the raster every other flux map on the
    spine uses.
    """
    radius, height = np.meshgrid(
        np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    )
    columns, _labels = harmonic_columns(radius.ravel(), height.ravel(), config)
    return (columns @ np.asarray(coefficients, dtype=np.float64)).reshape(height.shape)


def harmonic_grad_flux_on_grid(
    config: HarmonicConfig,
    coefficients: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gauge-free flux gradient ``(dPhi/dR, dPhi/dZ)`` [Wb/m].

    The absolute level of a harmonic read is only weakly pinned, so an interior
    soft prior matches the *gradient* of the flux — equivalently the poloidal
    field, the best-measured quantity — which is invariant to any additive
    constant on :math:`\\psi`. Undoes the field convention of
    :func:`harmonic_field_columns`:
    :math:`\\partial_R\\Phi = +2\\pi R B_Z`,
    :math:`\\partial_Z\\Phi = -2\\pi R B_R`.
    """
    radius, height = np.meshgrid(
        np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    )
    br, bz = harmonic_field_columns(radius.ravel(), height.ravel(), config)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    two_pi_r = 2.0 * np.pi * np.maximum(radius.ravel(), _DISTANCE_FLOOR)
    return (
        (two_pi_r * (bz @ coefficients)).reshape(height.shape),
        (-two_pi_r * (br @ coefficients)).reshape(height.shape),
    )


def circulation_row(
    config: HarmonicConfig,
    loop_radius: float,
    *,
    loop_z: float | None = None,
    n_loop: int = 512,
) -> np.ndarray:
    """Return the poloidal-circulation gauge-tie row of the basis.

    Ampere's law for the axisymmetric poloidal field says the circulation around
    any closed poloidal loop enclosing the plasma is
    :math:`\\oint \\mathbf{B}_{pol}\\cdot d\\mathbf{l} = \\mu_0 I_p`. The harmonic
    columns are source-free everywhere except the focal ring — where the plasma
    current sits — so the circulation is path-independent for any pole-enclosing
    loop and the tie is a single well-posed equation on the coefficients. It acts
    on the cosine family: the sine columns are odd about the pole height and
    enclose no net current, so their circulation vanishes (pinned by test).

    The loop is traversed clockwise, the orientation for which a positive current
    gives ``row @ coefficients = +mu_0 * Ip`` under the field convention of
    :func:`harmonic_field_columns`.
    """
    height = config.pole_z if loop_z is None else float(loop_z)
    angle = np.linspace(0.0, 2.0 * np.pi, int(n_loop), endpoint=False)
    step = 2.0 * np.pi / int(n_loop)
    br, bz = harmonic_field_columns(
        config.pole_r + loop_radius * np.cos(angle),
        height + loop_radius * np.sin(angle),
        config,
    )
    d_r = loop_radius * np.sin(angle) * step
    d_z = -loop_radius * np.cos(angle) * step
    return (br * d_r[:, None] + bz * d_z[:, None]).sum(axis=0)


@dataclass
class ReconstructHarmonic:
    """Reconstruct the plasma flux in the vacuum annulus on a ring-function basis.

    Holds the fixed geometry — the pole placement in ``config`` and the sensor
    set — and fits one slice at a time. The design matrix is pure geometry, so it
    is built once and reused across every slice of a campaign.
    """

    magnetics: Magnetics
    config: HarmonicConfig = field(default_factory=HarmonicConfig)
    sensor_matrix: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Build the sensor design matrix unless the caller supplied one."""
        if self.sensor_matrix is None:
            self.sensor_matrix = self.build_sensor_matrix()

    def build_sensor_matrix(self) -> np.ndarray:
        """Return each harmonic's signature per sensor row, ``(n_sensor, n_term)``.

        Flux-loop rows read the flux column; field-probe rows read the
        orientation-projected field.
        """
        psi, _labels = harmonic_columns(
            self.magnetics.r, self.magnetics.z, self.config
        )
        br, bz = harmonic_field_columns(
            self.magnetics.r, self.magnetics.z, self.config
        )
        return self.magnetics.project(psi, br, bz)

    def gauge_anchor(
        self, measurement: SliceMeasurement
    ) -> tuple[np.ndarray, float, float] | None:
        """Return the circulation gauge tie, or ``None`` when it is disabled.

        The loop is a circle among the sensors, well inside the sensor ring and
        enclosing the pole; the circulation is path-independent, so the exact
        radius is immaterial. The tie is whitened to a relative circulation so
        its target is of order one and its weight is comparable with the
        whitened sensor rows.
        """
        if not self.config.ip_anchor:
            return None
        distance = np.hypot(
            np.asarray(self.magnetics.r, dtype=np.float64) - self.config.pole_r,
            np.asarray(self.magnetics.z, dtype=np.float64) - self.config.pole_z,
        )
        row = circulation_row(self.config, 0.5 * float(np.median(distance)))
        target = mu_0 * float(measurement.plasma_current)
        scale = abs(target) if abs(target) > 0.0 else 1.0
        return row / scale, target / scale, float(self.config.ip_anchor_weight)

    def fit(self, measurement: SliceMeasurement) -> HarmonicInversion:
        """Fit the harmonic amplitudes of one slice by whitened least squares."""
        penalty = (
            mode_penalty(self.config.order, self.config.sobolev_exponent)
            if self.config.sobolev_exponent > 0
            else None
        )
        coefficients, covariance = whitened_solve(
            self.sensor_matrix,
            measurement.signature,
            measurement.weight,
            self.config.ridge,
            penalty=penalty,
            anchor=self.gauge_anchor(measurement),
        )
        keep = np.asarray(measurement.mask, dtype=bool)
        residual = (
            self.sensor_matrix @ coefficients - measurement.signature
        ) * measurement.weight
        return HarmonicInversion(
            coefficients=coefficients,
            labels=harmonic_labels(self.config.order),
            misfit=float((residual[keep] ** 2).sum() / max(int(keep.sum()), 1)),
            config=self.config,
            covariance=covariance,
        )

    def flux_on_grid(
        self, coefficients: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
    ) -> np.ndarray:
        """Return the plasma harmonic flux ``(n_z, n_r)`` [Wb]."""
        return harmonic_flux_on_grid(self.config, coefficients, grid_r, grid_z)

    def grad_flux_on_grid(
        self, coefficients: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the gauge-free flux gradient on the grid [Wb/m]."""
        return harmonic_grad_flux_on_grid(self.config, coefficients, grid_r, grid_z)

    def mask_invalid_interior(
        self,
        psi: np.ndarray,
        grid_r: np.ndarray,
        grid_z: np.ndarray,
        radius: float,
        *,
        axis: tuple[float, float] | None = None,
        spread_factor: float = 20.0,
    ) -> np.ndarray:
        """Fill the near-pole disc with the confined-side extreme.

        The harmonics are physical only in the source-free annulus; toward the
        focal ring the ring functions diverge, so the reconstructed flux blows up
        inside the plasma where the expansion does not hold. Reading a boundary
        directly off that field puts spurious saddles and early ray-cast
        crossings in the invalid interior. Replacing the disc with a single value
        ``spread_factor`` standard deviations past the annulus median on the
        confined side makes the interior read as deeper than any boundary, so the
        boundary read sees a clean confined plateau inside and the physical field
        outside.

        Returns a copy; the annulus outside ``radius`` is untouched.
        """
        psi = np.asarray(psi, dtype=np.float64)
        grid_r = np.asarray(grid_r, dtype=np.float64)
        grid_z = np.asarray(grid_z, dtype=np.float64)
        radial, vertical = np.meshgrid(grid_r, grid_z)
        inside = (
            np.hypot(radial - self.config.pole_r, vertical - self.config.pole_z)
            < radius
        )
        if not inside.any() or inside.all():
            return psi.copy()
        annulus = psi[~inside]
        median = float(np.median(annulus))
        spread = float(np.std(annulus)) or 1.0
        if axis is not None:
            row = int(np.argmin(np.abs(grid_z - axis[1])))
            column = int(np.argmin(np.abs(grid_r - axis[0])))
            sign = np.sign(psi[row, column] - median)
        else:
            near = psi[inside]
            sign = np.sign(near[int(np.argmax(np.abs(near - median)))] - median)
        out = psi.copy()
        out[inside] = median + (sign or 1.0) * spread_factor * spread
        return out


# --- the annulus soft prior -------------------------------------------------


def annulus_points(grid, *, flux, axis_flux, boundary_flux) -> np.ndarray:
    """Return the flat grid indices of the shared vacuum annulus.

    The annulus is the set of points inside the limiter but outside the confined
    region, the confined region being the flux deeper than ``boundary_flux``
    toward ``axis_flux``. Both references must come from the boundary read's own
    frozen values, never from an evolving interior solve, so the penalty domain
    stays fixed across a Picard loop.
    """
    flux = np.asarray(flux, dtype=np.float64).ravel()
    inside = np.asarray(grid.inside_limiter, dtype=bool).ravel()
    if flux.shape != inside.shape:
        raise ValueError(
            f"flux ravel {flux.shape} does not match the limiter mask {inside.shape}"
        )
    confined = (flux - boundary_flux) * np.sign(axis_flux - boundary_flux) > 0.0
    return np.where(inside & ~confined)[0]


def huber_weights(residual, clip: float | None) -> np.ndarray:
    """Return Huber-style per-row weights from a residual proxy.

    Points within ``clip`` robust sigma of the median-centred bulk keep weight
    one; beyond that a point of standardised deviation ``z`` is down-weighted to
    ``clip / z``. The proxy is centred by its median first, so a uniform additive
    shift — an absolute-flux gauge change — leaves the weights invariant. The
    robust scale is the MAD times the Gaussian-consistent 1.4826; a degenerate
    scale returns uniform weights.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.size == 0 or clip is None:
        return np.ones(residual.shape, dtype=np.float64)
    centred = residual - np.median(residual)
    scale = 1.4826 * np.median(np.abs(centred))
    weight = np.ones(residual.shape, dtype=np.float64)
    if not scale > 0.0:
        return weight
    standardised = np.abs(centred) / scale
    hot = standardised > clip
    weight[hot] = clip / standardised[hot]
    return weight


def _design_block(block, n_row: int, n_column: int) -> np.ndarray:
    """Coerce a possibly missing design block to its declared shape."""
    if n_column == 0:
        return np.zeros((n_row, 0), dtype=np.float64)
    if block is None:
        raise ValueError(
            "design block is missing but its coefficient count is non-zero"
        )
    block = np.asarray(block, dtype=np.float64)
    if block.shape != (n_row, n_column):
        raise ValueError(f"design block {block.shape} != expected {(n_row, n_column)}")
    return block


def annulus_penalty_rows(
    *,
    form: str,
    basis: np.ndarray | None = None,
    passive: np.ndarray | None = None,
    fixed: np.ndarray,
    target: np.ndarray,
    n_profile: int,
    n_passive: int,
    weight: float,
    uncertainty: float | None = None,
    robust_clip: float | None = None,
    gauge_offset: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return least-squares rows penalising annulus disagreement with this read.

    This is the harmonic read's coupling into an interior solve: appending the
    returned rows to the solve's whitened design, and the returned targets under
    its data, adds the annulus soft prior. Rows act on the interior solve's
    coefficient vector — ``n_profile`` profile coefficients followed by
    ``n_passive`` passive-mode amplitudes — extended by a trailing free offset
    column in absolute-flux form when ``gauge_offset`` is set. Each row drives

    * ``form="grad-psi"``: ``basis @ x + fixed - target`` on the flux gradient,
      equivalently the poloidal field, which is invariant to an additive
      constant on the flux — the gauge-safe form;
    * ``form="abs-psi"``: the same on absolute flux, minus the free offset, which
      absorbs the read's weakly-pinned DC level.

    Rows are weighted by ``weight / uncertainty`` times the Huber weight, so a
    read's own uncertainty sets the prior strength and individual annulus points
    whose mismatch is an outlier are down-weighted rather than allowed to drag
    the solve.
    """
    if form not in ("grad-psi", "abs-psi"):
        raise ValueError(f"form must be 'grad-psi' or 'abs-psi', got {form!r}")
    uncertainty = 1.0 if uncertainty is None else float(uncertainty)
    if not uncertainty > 0.0:
        raise ValueError("uncertainty must be positive")

    target = np.asarray(target, dtype=np.float64).ravel()
    fixed = np.asarray(fixed, dtype=np.float64).ravel()
    n_row = target.shape[0]
    design = np.hstack(
        [
            _design_block(basis, n_row, n_profile),
            _design_block(passive, n_row, n_passive),
        ]
    )
    residual = target - fixed
    if form == "abs-psi" and gauge_offset:
        # the free shared offset enters as a -1 column: design @ x - offset
        # reproduces target - fixed, so the offset absorbs any constant shift
        design = np.hstack([design, -np.ones((n_row, 1))])

    row_weight = float(weight) / uncertainty * huber_weights(residual, robust_clip)
    return row_weight[:, np.newaxis] * design, row_weight * residual
