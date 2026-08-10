"""Exact rotating Grad-Shafranov equilibria under an isothermal-surface closure.

This module is a self-contained analytic benchmark: it imports numpy and scipy
only, so it stays a fixed target while a solver implementation changes
underneath it. Nothing here solves an equilibrium; every quantity is a closed
form or a one-dimensional quadrature of a closed form.

Force balance with purely toroidal flow
---------------------------------------
Take ideal MHD, axisymmetry, and a purely toroidal velocity
``v = R Omega e_phi``. The convective acceleration is centripetal,
``(v.grad) v = -R Omega^2 e_R``, so the momentum balance reads

    grad(p) = J x B + rho R Omega^2 e_R.

Write the field with the poloidal flux per radian ``psi`` and the
toroidal-field function ``F = R B_phi``:

    B = grad(psi) x grad(phi) + F grad(phi),      grad(phi) = e_phi / R,
    B_R = -psi_Z / R,   B_Z = psi_R / R,   B_phi = F / R,
    mu0 J = grad(F) x grad(phi) - DeltaStar(psi) grad(phi),
    DeltaStar(psi) = psi_RR - psi_R / R + psi_ZZ,
    mu0 J_phi = -DeltaStar(psi) / R.

The toroidal component of the momentum balance is empty on the left and forces
``grad(F) x grad(psi) = 0``, so ``F = F(psi)`` exactly as in the static case.
The poloidal component splits on the two independent directions ``grad(psi)``
and ``grad(R)``:

    dp/dR at fixed psi = rho R Omega^2,                       (radial balance)
    DeltaStar(psi) = -mu0 R^2 (dp/dpsi at fixed R) - F dF/dpsi.  (rotating GS)

Pressure is no longer a flux function; only the derivative *at fixed major
radius* enters the toroidal-current source.

Isothermal-surface closure
--------------------------
The closure fixes what ``rho`` is. Take an ideal gas ``p = rho T / mbar`` with
temperature in energy units and ``mbar`` the mean mass per pressure-carrying
particle, and let both the temperature and the angular frequency be flux
functions, ``T = T(psi)`` and ``Omega = Omega(psi)``. Radial balance becomes a
linear equation for ``ln p`` in ``R^2``:

    d ln p / dR at fixed psi = mbar Omega^2(psi) R / T(psi) = 2 theta(psi) R,
    theta(psi) = mbar Omega^2(psi) / (2 T(psi)),

so that, with ``u = R^2 - R_0^2`` measured from a reference major radius,

    p(R, psi)   = p_0(psi) exp[theta(psi) u],
    rho(R, psi) = rho_0(psi) exp[theta(psi) u],   rho_0 = mbar p_0 / T.

Both pressure and mass density are centrifugally piled up on the outboard side
of every flux surface, by the same exponential factor.

Why the exactly solvable branch has one uniform Mach number
-----------------------------------------------------------
The Solov'ev-type branch is the one whose Grad-Shafranov source does not depend
on ``psi``, which is what makes the equation linear and solvable in closed form.
Differentiating the closure at fixed ``R``,

    dp/dpsi at fixed R = [p_0'(psi) + p_0(psi) theta'(psi) u] exp[theta(psi) u].

Requiring this to be independent of ``psi`` for every ``u`` says that ``p`` is
affine in ``psi`` at fixed ``u``, i.e. ``p_0(psi) exp[theta(psi) u]
= A(u) + psi B(u)``. Taking ``d/du`` of the logarithm gives
``theta(psi) = (A' + psi B') / (A + psi B)``, which is free of ``u`` only when
``A`` and ``B`` share a single exponential rate -- that is, only when ``theta``
is constant and ``p_0`` is affine in ``psi``.

Constant ``theta`` is not an arbitrary restriction but a physical statement.
With the thermal speed ``v_th = sqrt(2 T / mbar)`` the thermal Mach number is

    M(psi) = Omega(psi) R_0 / v_th(psi) = R_0 sqrt(theta),

so the exactly solvable isothermal family is exactly the **uniform thermal Mach
number** family. Temperature and angular frequency may both vary across the
plasma provided ``Omega`` tracks ``sqrt(T)``; the reference cases below use a
falling temperature and the ``sqrt(T)`` rotation profile it implies.

The closed form
---------------
Put ``p_0(psi) = p_1 psi`` and ``F dF/dpsi = 2 k_z`` (both constants) into the
rotating Grad-Shafranov equation. The right-hand side then depends on ``R``
alone,

    DeltaStar(psi) = -mu0 p_1 R^2 exp(theta u) - 2 k_z,

and the two identities ``DeltaStar[exp(theta u)] = 4 theta^2 R^2 exp(theta u)``
and ``DeltaStar[Z^2] = 2`` integrate it exactly. With ``4 k_p = mu0 p_1``,

    psi(R, Z) = psi_a - k_p u^2 phi2(theta u) - k_z Z^2,
    phi2(x) = (exp(x) - 1 - x) / x^2,   phi2(0) = 1/2,

which is the Maschke-Perrin isothermal rotating equilibrium in the branch whose
static limit is a Solov'ev solution. Nothing is expanded or truncated: ``phi2``
is entire, so the solution is exact at every Mach number.

Three properties of that form drive the tests:

* ``theta -> 0`` gives ``phi2 -> 1/2`` and recovers the Solov'ev particular
  integral ``-k_p u^2 / 2`` identically, with no separate code path;
* the low-Mach expansion ``phi2(theta u) = 1/2 + theta u / 6 + ...`` turns the
  source into ``-mu0 p_1 R^2 - mu0 p_1 theta R^2 (R^2 - R_0^2) + O(theta^2)``,
  the familiar quartic-radius rotating-Solov'ev term -- but with its
  coefficient *fixed* by ``mbar Omega^2 / 2 T`` rather than free, which is why
  fitting a quartic coefficient is not by itself evidence of rotation;
* the flux surfaces stay closed and nested around ``(R_0, 0)`` because
  ``grad(psi)`` vanishes only there, so the family has no X-point and its
  boundary is the smooth contour ``psi = 0``.

Orientation and signs
---------------------
``(R, phi, Z)`` is right-handed, ``e_R x e_phi = e_Z``. ``psi`` is the poloidal
flux per radian; the total poloidal flux is ``2 pi psi``. The relations above
correspond to ``sigma_Bp = +1`` (poloidal field written as
``grad(psi) x grad(phi)``), ``sigma_Rphiz = +1`` and ``e_Bp = 0`` in the
Sauter-Medvedev parametrisation -- the label usually attached to that triple is
COCOS 1, and COCOS 11 for the total-flux variant. A caller pinning a convention
should assert against the equations, not the label.

Every reference case takes ``k_p > 0`` and ``k_z > 0``, which fixes the
orientation completely:

* ``psi`` is maximal on the magnetic axis, ``psi_axis = psi_a > 0``, and falls
  to ``0`` on the plasma boundary;
* ``mu0 J_phi = 4 k_p R exp(theta u) + 2 k_z / R > 0`` everywhere, so the
  toroidal current flows along ``+e_phi``;
* ``F > 0``, so ``B_phi > 0`` and the plasma current is co-directed with the
  toroidal field;
* on the outboard midplane ``B_Z = psi_R / R < 0``, and on the inboard midplane
  ``B_Z > 0``, which is the field of a positive toroidal current loop.

Derived invariants
------------------
The following are closed forms, derived here and asserted in the companion
tests.

*Magnetic axis.* ``grad(psi) = (-2 R G'(u), -2 k_z Z)`` with
``G'(u) = k_p u phi1(theta u)`` and ``phi1(x) = (exp(x) - 1) / x``, which
vanishes only at ``u = 0``, ``Z = 0``. The axis therefore sits exactly at
``(R_0, 0)`` for every Mach number, and ``psi`` there is exactly ``psi_a``.

*Midplane boundary.* On ``Z = 0`` the boundary solves
``exp(w) - 1 - w = W`` with ``w = theta u`` and ``W = theta^2 psi_a / k_p``.
Writing ``c = 1 + W`` this is ``exp(w) = c + w``, whose two real roots are
``w = -c - LambertW_k(-exp(-c))`` on branches ``k = 0`` (inboard) and
``k = -1`` (outboard). Because ``phi2`` is increasing, the outboard root has
the smaller ``|u|``: rotation pulls the outboard boundary in more than the
inboard one, so the geometric centre moves inward while the axis stays at
``R_0`` and the outward Shafranov shift grows. To first order in ``theta`` both
roots shift by ``-theta psi_a / (3 k_p)`` in ``u``, so the shift grows linearly
in ``theta``, i.e. quadratically in the Mach number.

*Safety factor on axis.* Near the axis ``psi = psi_a - a dR^2 - b Z^2`` with
``a = 2 k_p R_0^2`` and ``b = k_z``, independently of ``theta`` because
``phi2(0) = 1/2`` exactly. The elliptic contour integral
``q = (1 / 2 pi) * closed_integral[F dl / (R^2 B_pol)]`` collapses because
``dl / |grad(psi)|`` has the constant integrand ``1 / sqrt(a b)``, giving

    q_axis = F(psi_a) / (2 R_0^2 sqrt(2 k_p k_z)),
    kappa_axis = R_0 sqrt(2 k_p / k_z).

Rotation leaves both unchanged at fixed source coefficients, because the
exponential is unity at the axis.

*Contour quadratures.* Every surface ``psi = psi_s`` satisfies
``G(u) + k_z Z^2 = s`` with ``s = psi_a - psi_s``, so the whole family of
surface integrals shares one parametrisation: ``R(t) = R_mid - R_half cos t``
maps ``t`` in ``[0, pi]`` onto the surface with a half-height proportional to
``sin t``, which makes every integrand below a smooth even ``2 pi``-periodic
function of ``t`` and the midpoint rule spectrally accurate. The inner
``Z``-integrals are done analytically first, leaving

    V             = 4 pi   * integral[ Z_b R dR ],
    I_p           = (2/mu0) * integral[ Z_b (4 k_p R e^{theta u}
                                              + 2 k_z / R) dR ],
    integral p dV = (8 pi / 3) p_1 k_z * integral[ Z_b^3 e^{theta u} R dR ],
    integral B_pol^2 dV = 2 pi * integral[ (8 Z_b R^2 G'(u)^2
                                            + (8/3) k_z^2 Z_b^3) / R dR ],
    L_p           = integral[ |grad psi| / (k_z Z_b) dR ],
    mu0 I_p       = integral[ |grad psi|^2 / (k_z Z_b R) dR ],
    q(psi_s)      = F(psi_s) / (2 pi k_z) * integral[ dR / (Z_b R) ],

the last two being Ampere's law and the safety factor, independent routes to
quantities the first list already provides.

*Beta and internal inductance.* All three are tied to the same poloidal field
scale ``B_pa = mu0 I_p / L_p`` built from the boundary perimeter, so the trio is
internally consistent:

    beta_t = 2 mu0 <p> / B_0^2 with B_0 = F_b / R_0 the vacuum field at R_0,
    beta_p = 2 mu0 <p> / B_pa^2,
    l_i    = <B_pol^2> / B_pa^2,

with ``<.>`` a plasma-volume average.
"""

import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import scipy.constants
import scipy.special
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

MU_0: Final[float] = scipy.constants.mu_0
"""Vacuum permeability, carried explicitly through every source relation."""

ELEMENTARY_CHARGE: Final[float] = scipy.constants.elementary_charge
"""Conversion from electronvolts to joules for temperatures in energy units."""

DEUTERON_MASS: Final[float] = scipy.constants.physical_constants["deuteron mass"][0]
"""Mean particle mass used by the reference cases, a pure deuterium plasma."""

# phi1 and phi2 are the first two phi-functions of exponential integrators,
# entire at the origin where their quotient forms cancel. Below the cutoff the
# Taylor series is used, above it the expm1 form; the crossover is placed where
# the cancellation in expm1(x) - x is still only a few ulp.
_SERIES_CUTOFF: Final[float] = 0.5
_SERIES_ORDER: Final[int] = 18
_PHI_ONE_COEFFS: Final[Array] = np.array(
    [1.0 / math.factorial(order + 1) for order in range(_SERIES_ORDER)][::-1]
)
_PHI_TWO_COEFFS: Final[Array] = np.array(
    [1.0 / math.factorial(order + 2) for order in range(_SERIES_ORDER)][::-1]
)

# Fixed iteration counts: the flux offset G is globally convex with a single
# turning point at u = 0, so Newton from the static root converges
# quadratically and monotonically on each side.
_NEWTON_STEPS: Final[int] = 8
_QUADRATURE_NODES: Final[int] = 512


def _phi_one(argument: ArrayLike) -> Array:
    """Return ``(exp(x) - 1) / x``, entire with value one at the origin."""
    value = np.asarray(argument, dtype=float)
    near_origin = np.abs(value) < _SERIES_CUTOFF
    safe = np.where(near_origin, 1.0, value)
    return np.where(
        near_origin,
        np.polyval(_PHI_ONE_COEFFS, value),
        np.expm1(safe) / safe,
    )


def _phi_two(argument: ArrayLike) -> Array:
    """Return ``(exp(x) - 1 - x) / x**2``, entire with value one half at zero."""
    value = np.asarray(argument, dtype=float)
    near_origin = np.abs(value) < _SERIES_CUTOFF
    safe = np.where(near_origin, 1.0, value)
    return np.where(
        near_origin,
        np.polyval(_PHI_TWO_COEFFS, value),
        (np.expm1(safe) - safe) / safe**2,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RotatingEquilibrium:
    """One exact member of the isothermal rotating Grad-Shafranov family.

    The solution is fixed by four coefficients -- ``axis_flux``,
    ``pressure_coefficient``, ``field_coefficient`` and ``rotation_parameter``
    -- plus the boundary value of the toroidal-field function. The
    thermodynamic primitives ``temperature`` and ``angular_frequency`` are
    additionally parametrised by the temperature profile; the rotation
    parameter constrains only their ratio, so a temperature profile picks the
    angular frequency uniquely.
    """

    name: str
    major_radius: float
    """Reference major radius ``R_0``; also the exact magnetic axis radius."""

    axis_flux: float
    """Poloidal flux per radian on the axis, with zero on the boundary."""

    pressure_coefficient: float
    """``k_p``, the pressure drive; ``mu0 dp_0/dpsi = 4 k_p``."""

    field_coefficient: float
    """``k_z``, the diamagnetic drive; ``F dF/dpsi = 2 k_z``."""

    rotation_parameter: float
    """``theta = mbar Omega^2 / (2 T)``, uniform across the plasma."""

    boundary_f: float
    """``F`` on the plasma boundary, equal to the vacuum ``R_0 B_0``."""

    axis_temperature: float
    """Temperature on the axis, in energy units."""

    boundary_temperature: float
    """Temperature on the plasma boundary, in energy units."""

    mean_particle_mass: float
    """Mean mass per pressure-carrying particle."""

    # ------------------------------------------------------------------
    # source amplitudes and dimensionless rotation measures
    # ------------------------------------------------------------------
    @property
    def pressure_flux_gradient(self) -> float:
        """Return ``dp_0/dpsi``, the constant amplitude of the pressure drive."""
        return 4.0 * self.pressure_coefficient / MU_0

    @property
    def f_f_prime(self) -> float:
        """Return the constant ``F dF/dpsi``."""
        return 2.0 * self.field_coefficient

    @property
    def thermal_mach_number(self) -> float:
        """Return ``Omega R_0 / sqrt(2 T / mbar)``, uniform over the plasma."""
        return self.major_radius * math.sqrt(self.rotation_parameter)

    @property
    def axis_pressure(self) -> float:
        """Return the pressure on the magnetic axis."""
        return self.pressure_flux_gradient * self.axis_flux

    @property
    def vacuum_field(self) -> float:
        """Return the vacuum toroidal field at the reference major radius."""
        return self.boundary_f / self.major_radius

    # ------------------------------------------------------------------
    # the closed-form solution and its derivatives
    # ------------------------------------------------------------------
    def _flux_label(self, radius: ArrayLike) -> Array:
        """Return ``u = R^2 - R_0^2``, the natural radial variable of the family."""
        return np.asarray(radius, dtype=float) ** 2 - self.major_radius**2

    def _flux_offset(self, label: ArrayLike) -> Array:
        """Return ``G(u) = k_p u^2 phi2(theta u)``, the flux drop from the axis."""
        value = np.asarray(label, dtype=float)
        return self.pressure_coefficient * value**2 * _phi_two(
            self.rotation_parameter * value
        )

    def _flux_offset_derivative(self, label: ArrayLike) -> Array:
        """Return ``dG/du = k_p u phi1(theta u)``."""
        value = np.asarray(label, dtype=float)
        return (
            self.pressure_coefficient
            * value
            * _phi_one(self.rotation_parameter * value)
        )

    def flux(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return the poloidal flux per radian at ``(R, Z)``."""
        label = self._flux_label(radius)
        height_value = np.asarray(height, dtype=float)
        return (
            self.axis_flux
            - self._flux_offset(label)
            - self.field_coefficient * height_value**2
        )

    def flux_gradient(
        self, radius: ArrayLike, height: ArrayLike
    ) -> tuple[Array, Array]:
        """Return ``(dpsi/dR, dpsi/dZ)`` at ``(R, Z)``."""
        radius_value, height_value = np.broadcast_arrays(
            np.asarray(radius, dtype=float), np.asarray(height, dtype=float)
        )
        label = self._flux_label(radius_value)
        radial = -2.0 * radius_value * self._flux_offset_derivative(label)
        vertical = -2.0 * self.field_coefficient * height_value
        return radial, vertical

    def delta_star(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return the Grad-Shafranov operator applied to the exact flux.

        The second radial derivative cancels the first exactly, leaving a form
        free of the phi-functions: ``DeltaStar(psi) = -4 k_p R^2 exp(theta u)
        - 2 k_z``.
        """
        radius_value, _ = np.broadcast_arrays(
            np.asarray(radius, dtype=float), np.asarray(height, dtype=float)
        )
        label = self._flux_label(radius_value)
        exponential = np.exp(self.rotation_parameter * label)
        drive = -4.0 * self.pressure_coefficient * radius_value**2 * exponential
        return drive - 2.0 * self.field_coefficient

    # ------------------------------------------------------------------
    # flux functions and the thermodynamic primitives
    # ------------------------------------------------------------------
    def f_function(self, flux_value: ArrayLike) -> Array:
        """Return ``F(psi) = sqrt(F_b^2 + 4 k_z psi)``.

        Outside the plasma the flux function is held at its boundary value,
        which is the vacuum toroidal-field function.
        """
        inside_flux = np.maximum(np.asarray(flux_value, dtype=float), 0.0)
        return np.sqrt(self.boundary_f**2 + 4.0 * self.field_coefficient * inside_flux)

    def temperature(self, flux_value: ArrayLike) -> Array:
        """Return the flux-surface temperature, linear in the flux label."""
        fraction = np.asarray(flux_value, dtype=float) / self.axis_flux
        return self.boundary_temperature + (
            self.axis_temperature - self.boundary_temperature
        ) * fraction

    def angular_frequency(self, flux_value: ArrayLike) -> Array:
        """Return ``Omega(psi) = sqrt(2 theta T(psi) / mbar)``.

        A uniform rotation parameter ties the angular frequency to the square
        root of the temperature; that pairing is what keeps the thermal Mach
        number the same on every flux surface.
        """
        return np.sqrt(
            2.0
            * self.rotation_parameter
            * self.temperature(flux_value)
            / self.mean_particle_mass
        )

    def pressure_flux_derivative(self, radius: ArrayLike) -> Array:
        """Return ``dp/dpsi`` at fixed major radius.

        Constant amplitude times the centrifugal factor; independent of ``psi``
        in this family, which is what makes the equation linear.
        """
        return self.pressure_flux_gradient * np.exp(
            self.rotation_parameter * self._flux_label(radius)
        )

    def pressure_at(self, flux_value: ArrayLike, radius: ArrayLike) -> Array:
        """Return ``p(psi, R) = p_1 psi exp[theta (R^2 - R_0^2)]``."""
        return (
            self.pressure_flux_gradient
            * np.asarray(flux_value, dtype=float)
            * np.exp(self.rotation_parameter * self._flux_label(radius))
        )

    def pressure(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return the pressure at ``(R, Z)``."""
        return self.pressure_at(self.flux(radius, height), radius)

    def mass_density_at(self, flux_value: ArrayLike, radius: ArrayLike) -> Array:
        """Return ``rho = mbar p / T`` at a flux label and major radius."""
        return (
            self.mean_particle_mass
            * self.pressure_at(flux_value, radius)
            / self.temperature(flux_value)
        )

    def mass_density(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return ``rho = mbar p / T``, outboard-peaked on every flux surface."""
        return self.mass_density_at(self.flux(radius, height), radius)

    def number_density(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return the particle density ``p / T``."""
        return self.mass_density(radius, height) / self.mean_particle_mass

    # ------------------------------------------------------------------
    # the toroidal current source
    # ------------------------------------------------------------------
    def toroidal_current_density(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return ``J_phi = -DeltaStar(psi) / (mu0 R)``."""
        return -self.delta_star(radius, height) / (
            MU_0 * np.asarray(radius, dtype=float)
        )

    def toroidal_current_source(self, radius: ArrayLike) -> Array:
        """Return the source form ``R dp/dpsi + F dF/dpsi / (mu0 R)``.

        This is the same current density written from the flux-function
        primitives instead of from the operator, and the two must agree.
        """
        radius_value = np.asarray(radius, dtype=float)
        return (
            radius_value * self.pressure_flux_derivative(radius_value)
            + self.f_f_prime / (MU_0 * radius_value)
        )

    def magnetic_field(
        self, radius: ArrayLike, height: ArrayLike
    ) -> tuple[Array, Array, Array]:
        """Return ``(B_R, B_phi, B_Z)`` at ``(R, Z)``."""
        radius_value = np.asarray(radius, dtype=float)
        radial_flux, vertical_flux = self.flux_gradient(radius_value, height)
        toroidal = self.f_function(self.flux(radius_value, height)) / radius_value
        return -vertical_flux / radius_value, toroidal, radial_flux / radius_value

    def grad_shafranov_residual(self, radius: ArrayLike, height: ArrayLike) -> Array:
        """Return the rotating Grad-Shafranov residual, analytically zero.

        ``DeltaStar(psi) + mu0 R^2 (dp/dpsi at fixed R) + F dF/dpsi`` cancels
        term by term, so any departure from zero is floating-point noise.
        """
        radius_value = np.asarray(radius, dtype=float)
        return (
            self.delta_star(radius_value, height)
            + MU_0 * radius_value**2 * self.pressure_flux_derivative(radius_value)
            + self.f_f_prime
        )

    def source_scale(self, radius: ArrayLike) -> Array:
        """Return the magnitude scale of the two source terms.

        Residuals are reported relative to this so that a tolerance means the
        same thing at reactor and compact scale.
        """
        radius_value = np.asarray(radius, dtype=float)
        return (
            MU_0 * radius_value**2 * np.abs(self.pressure_flux_derivative(radius_value))
            + abs(self.f_f_prime)
        )

    # ------------------------------------------------------------------
    # domain, axis, and flux-surface geometry
    # ------------------------------------------------------------------
    def contains(self, radius: ArrayLike, height: ArrayLike) -> NDArray[np.bool_]:
        """Return whether ``(R, Z)`` lies inside the plasma boundary."""
        return self.flux(radius, height) > 0.0

    @property
    def magnetic_axis(self) -> tuple[float, float]:
        """Return the exact magnetic axis, independent of the Mach number."""
        return self.major_radius, 0.0

    @property
    def axis_elongation(self) -> float:
        """Return the elongation of the vanishing flux surfaces at the axis."""
        return self.major_radius * math.sqrt(
            2.0 * self.pressure_coefficient / self.field_coefficient
        )

    @property
    def axis_safety_factor(self) -> float:
        """Return the closed-form ``q`` on axis, unchanged by rotation."""
        return float(self.f_function(self.axis_flux)) / (
            2.0
            * self.major_radius**2
            * math.sqrt(2.0 * self.pressure_coefficient * self.field_coefficient)
        )

    def boundary_flux_labels_closed_form(self) -> tuple[float, float]:
        """Return the midplane boundary labels ``(u_in, u_out)`` in closed form.

        Solves ``exp(w) - 1 - w = W`` through the two real Lambert branches.
        Near a vanishing ``W`` the branch point costs about half the available
        digits, so this is exposed for verification while the refined radii come
        from :meth:`surface_midplane_radii`.
        """
        if self.rotation_parameter == 0.0:
            half_width = math.sqrt(2.0 * self.axis_flux / self.pressure_coefficient)
            return -half_width, half_width
        drive = (
            self.rotation_parameter**2 * self.axis_flux / self.pressure_coefficient
        )
        offset = 1.0 + drive
        argument = -math.exp(-offset)
        inboard = -offset - scipy.special.lambertw(argument, 0).real
        outboard = -offset - scipy.special.lambertw(argument, -1).real
        return inboard / self.rotation_parameter, outboard / self.rotation_parameter

    def _surface_flux_labels(self, offset: ArrayLike) -> tuple[Array, Array]:
        """Return ``(u_in, u_out)`` where ``G(u)`` equals the given flux drop.

        The static root is an exact seed as either the rotation or the offset
        vanishes, and is within roughly a tenth of the root otherwise; ``G`` is
        globally convex with its only turning point at the origin, so Newton
        converges monotonically on each branch.
        """
        drop = np.asarray(offset, dtype=float)
        seed = np.sqrt(2.0 * drop / self.pressure_coefficient)
        labels = []
        for root in (-seed, seed):
            label = root
            for _ in range(_NEWTON_STEPS):
                label = label - (self._flux_offset(label) - drop) / (
                    self._flux_offset_derivative(label)
                )
            labels.append(label)
        return labels[0], labels[1]

    def surface_midplane_radii(self, flux_value: float) -> tuple[float, float]:
        """Return the inboard and outboard midplane radii of a flux surface."""
        inboard, outboard = self._surface_flux_labels(self.axis_flux - flux_value)
        return (
            float(np.sqrt(self.major_radius**2 + inboard)),
            float(np.sqrt(self.major_radius**2 + outboard)),
        )

    def boundary_midplane_radii(self) -> tuple[float, float]:
        """Return the inboard and outboard midplane radii of the boundary."""
        return self.surface_midplane_radii(0.0)

    @property
    def geometric_axis_radius(self) -> float:
        """Return the midpoint of the boundary midplane extent."""
        inboard, outboard = self.boundary_midplane_radii()
        return 0.5 * (inboard + outboard)

    @property
    def minor_radius(self) -> float:
        """Return half the boundary midplane extent."""
        inboard, outboard = self.boundary_midplane_radii()
        return 0.5 * (outboard - inboard)

    @property
    def shafranov_shift(self) -> float:
        """Return the outward displacement of the axis from the geometric axis."""
        return self.major_radius - self.geometric_axis_radius

    # ------------------------------------------------------------------
    # surface quadratures
    # ------------------------------------------------------------------
    def _surface_nodes(
        self, flux_value: float, nodes: int
    ) -> tuple[Array, Array, Array, float]:
        """Return midpoint quadrature nodes on one flux surface.

        The cosine map turns the square-root endpoint behaviour into a factor
        of ``sin t``, so every integrand assembled from these nodes is a smooth
        even periodic function of ``t`` and the midpoint rule converges
        spectrally. Returns the radii, the half-heights, the ``dR`` weights and
        the flux-drop offset.
        """
        offset = self.axis_flux - flux_value
        inboard, outboard = self.surface_midplane_radii(flux_value)
        centre = 0.5 * (inboard + outboard)
        half_width = 0.5 * (outboard - inboard)
        angle = (np.arange(nodes) + 0.5) * np.pi / nodes
        radius = centre - half_width * np.cos(angle)
        label = self._flux_label(radius)
        half_height = np.sqrt(
            np.maximum(offset - self._flux_offset(label), 0.0) / self.field_coefficient
        )
        weight = half_width * np.sin(angle) * np.pi / nodes
        return radius, half_height, weight, offset

    def plasma_volume(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the plasma volume."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        return float(4.0 * np.pi * np.sum(half_height * radius * weight))

    def plasma_current(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the toroidal plasma current from the current-density integral."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        exponential = np.exp(self.rotation_parameter * self._flux_label(radius))
        density = (
            4.0 * self.pressure_coefficient * radius * exponential
            + 2.0 * self.field_coefficient / radius
        )
        return float(2.0 * np.sum(half_height * density * weight) / MU_0)

    def plasma_current_from_ampere_law(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the plasma current from the boundary loop integral of ``B_pol``."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        gradient_squared = self._boundary_gradient_squared(radius, half_height)
        integrand = gradient_squared / (self.field_coefficient * half_height * radius)
        return float(np.sum(integrand * weight) / MU_0)

    def _boundary_gradient_squared(self, radius: Array, half_height: Array) -> Array:
        """Return ``|grad psi|^2`` on the upper branch of a surface."""
        label = self._flux_label(radius)
        radial = 2.0 * radius * self._flux_offset_derivative(label)
        vertical = 2.0 * self.field_coefficient * half_height
        return radial**2 + vertical**2

    def boundary_perimeter(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the poloidal circumference of the plasma boundary."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        gradient = np.sqrt(self._boundary_gradient_squared(radius, half_height))
        integrand = gradient / (self.field_coefficient * half_height)
        return float(np.sum(integrand * weight))

    def pressure_volume_integral(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the volume integral of the pressure."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        exponential = np.exp(self.rotation_parameter * self._flux_label(radius))
        integrand = half_height**3 * exponential * radius
        return float(
            8.0
            * np.pi
            * self.pressure_flux_gradient
            * self.field_coefficient
            * np.sum(integrand * weight)
            / 3.0
        )

    def poloidal_field_volume_integral(self, nodes: int = _QUADRATURE_NODES) -> float:
        """Return the volume integral of the squared poloidal field."""
        radius, half_height, weight, _ = self._surface_nodes(0.0, nodes)
        label = self._flux_label(radius)
        radial = 2.0 * radius * self._flux_offset_derivative(label)
        integrand = (
            2.0 * half_height * radial**2
            + 8.0 * self.field_coefficient**2 * half_height**3 / 3.0
        ) / radius
        return float(2.0 * np.pi * np.sum(integrand * weight))

    def safety_factor(
        self, flux_value: float, nodes: int = _QUADRATURE_NODES
    ) -> float:
        """Return the safety factor of one flux surface by contour quadrature."""
        radius, half_height, weight, _ = self._surface_nodes(flux_value, nodes)
        integrand = 1.0 / (half_height * radius)
        toroidal = float(self.f_function(flux_value))
        return float(
            toroidal
            * np.sum(integrand * weight)
            / (2.0 * np.pi * self.field_coefficient)
        )

    # ------------------------------------------------------------------
    # integral moments
    # ------------------------------------------------------------------
    @property
    def mean_pressure(self) -> float:
        """Return the volume-averaged pressure."""
        return self.pressure_volume_integral() / self.plasma_volume()

    @property
    def poloidal_field_scale(self) -> float:
        """Return ``mu0 I_p / L_p``, the perimeter-averaged poloidal field."""
        return MU_0 * self.plasma_current() / self.boundary_perimeter()

    @property
    def beta_toroidal(self) -> float:
        """Return ``2 mu0 <p> / B_0^2`` against the vacuum field at ``R_0``."""
        return 2.0 * MU_0 * self.mean_pressure / self.vacuum_field**2

    @property
    def beta_poloidal(self) -> float:
        """Return ``2 mu0 <p>`` over the squared perimeter-averaged field."""
        return 2.0 * MU_0 * self.mean_pressure / self.poloidal_field_scale**2

    @property
    def internal_inductance(self) -> float:
        """Return ``<B_pol^2>`` over the squared perimeter-averaged field."""
        mean_square = self.poloidal_field_volume_integral() / self.plasma_volume()
        return mean_square / self.poloidal_field_scale**2

    # ------------------------------------------------------------------
    # family relatives
    # ------------------------------------------------------------------
    def with_rotation_parameter(
        self, rotation_parameter: float
    ) -> "RotatingEquilibrium":
        """Return the same source coefficients at another rotation parameter.

        Only the centrifugal factor moves; the pressure and toroidal-field
        gradients and the axis flux are untouched, which is the comparison that
        isolates what rotation does to a solution at a fixed drive.
        """
        return replace(self, rotation_parameter=rotation_parameter)

    def static_limit(self) -> "RotatingEquilibrium":
        """Return the same source amplitudes with the rotation switched off.

        The flux collapses to the Solov'ev form
        ``psi_a - k_p u^2 / 2 - k_z Z^2``.
        """
        return replace(self, name=f"{self.name}-static", rotation_parameter=0.0)

    def with_thermal_mach_number(self, mach_number: float) -> "RotatingEquilibrium":
        """Return the same shape anchors and temperature at another Mach number.

        Holding the outboard midplane radius, the boundary half-height and the
        axis pressure fixed isolates what rotation costs in drive: the geometry
        anchors are recovered and the coefficients move instead.
        """
        _, outboard = self.boundary_midplane_radii()
        return rotating_equilibrium(
            name=self.name,
            major_radius=self.major_radius,
            outboard_radius=outboard,
            half_height=math.sqrt(self.axis_flux / self.field_coefficient),
            vacuum_field=self.vacuum_field,
            axis_pressure=self.axis_pressure,
            thermal_mach_number=mach_number,
            axis_temperature=self.axis_temperature,
            boundary_temperature=self.boundary_temperature,
            mean_particle_mass=self.mean_particle_mass,
        )


def rotating_equilibrium(
    *,
    name: str,
    major_radius: float,
    outboard_radius: float,
    half_height: float,
    vacuum_field: float,
    axis_pressure: float,
    thermal_mach_number: float,
    axis_temperature: float,
    boundary_temperature: float,
    mean_particle_mass: float = DEUTERON_MASS,
) -> RotatingEquilibrium:
    """Build a member from geometry and physical amplitudes.

    The construction inverts the closed form without a solve. The rotation
    parameter follows from the Mach number as ``theta = M^2 / R_0^2``; the
    outboard midplane radius fixes the boundary flux label ``u_out``; the axis
    pressure ``p_1 psi_a = 4 k_p^2 u_out^2 phi2(theta u_out) / mu0`` then
    determines ``k_p`` in closed form; and the boundary half-height at the
    reference radius fixes ``k_z = psi_a / Z_max^2``.
    """
    rotation_parameter = thermal_mach_number**2 / major_radius**2
    label = outboard_radius**2 - major_radius**2
    if label <= 0.0:
        raise ValueError("outboard_radius must exceed major_radius")
    shape = float(_phi_two(rotation_parameter * label))
    pressure_coefficient = math.sqrt(MU_0 * axis_pressure) / (
        2.0 * label * math.sqrt(shape)
    )
    axis_flux = pressure_coefficient * label**2 * shape
    return RotatingEquilibrium(
        name=name,
        major_radius=major_radius,
        axis_flux=axis_flux,
        pressure_coefficient=pressure_coefficient,
        field_coefficient=axis_flux / half_height**2,
        rotation_parameter=rotation_parameter,
        boundary_f=major_radius * vacuum_field,
        axis_temperature=axis_temperature,
        boundary_temperature=boundary_temperature,
        mean_particle_mass=mean_particle_mass,
    )


def _kilo_electronvolt(value: float) -> float:
    """Return a temperature in joules from a value in kiloelectronvolts."""
    return value * 1.0e3 * ELEMENTARY_CHARGE


def reference_cases() -> dict[str, RotatingEquilibrium]:
    """Return the three reference members, spanning the rotation range.

    All three carry metre, tesla and pascal scales a tokamak would recognise,
    and differ mainly in the thermal Mach number. The inboard boundary is the
    binding geometric constraint: the family measures radius through
    ``u = R^2 - R_0^2``, so the boundary reaches the symmetry axis once
    ``u_out`` approaches ``R_0^2``, which keeps these cases at conventional
    aspect ratio.

    weak-rotation-reactor
        Reactor scale at a thermal Mach number of 0.1 -- roughly a hundred
        kilometres per second of toroidal flow at ten kiloelectronvolts, where
        the centrifugal factor moves pressure across a surface by a few per
        cent and the quartic-radius source term is a small correction.
    moderate-rotation-conventional
        Metre-scale at a thermal Mach number of 0.35, the range a
        beam-heated conventional tokamak reaches, where the outboard-inboard
        pressure ratio on a surface is around fifteen per cent.
    strong-rotation-compact
        Compact and strongly driven at a thermal Mach number of 0.9, where the
        exponential closure and its low-Mach truncation visibly part company
        and the Shafranov shift carries a clear rotation contribution.
    """
    return {
        case.name: case
        for case in (
            rotating_equilibrium(
                name="weak-rotation-reactor",
                major_radius=6.2,
                outboard_radius=7.8,
                half_height=3.6,
                vacuum_field=5.3,
                axis_pressure=8.0e5,
                thermal_mach_number=0.10,
                axis_temperature=_kilo_electronvolt(10.0),
                boundary_temperature=_kilo_electronvolt(0.2),
            ),
            rotating_equilibrium(
                name="moderate-rotation-conventional",
                major_radius=1.70,
                outboard_radius=2.15,
                half_height=1.00,
                vacuum_field=2.0,
                axis_pressure=1.0e5,
                thermal_mach_number=0.35,
                axis_temperature=_kilo_electronvolt(2.0),
                boundary_temperature=_kilo_electronvolt(0.05),
            ),
            rotating_equilibrium(
                name="strong-rotation-compact",
                major_radius=0.90,
                outboard_radius=1.08,
                half_height=0.40,
                vacuum_field=0.60,
                axis_pressure=1.5e4,
                thermal_mach_number=0.90,
                axis_temperature=_kilo_electronvolt(0.8),
                boundary_temperature=_kilo_electronvolt(0.02),
            ),
        )
    }


def interior_sample(
    case: RotatingEquilibrium, count: int = 21, margin: float = 0.15
) -> tuple[Array, Array]:
    """Return a grid of points strictly inside the plasma boundary.

    The margin keeps the sample away from the boundary, where a finite
    difference stencil would reach outside the plasma and where the flux itself
    is a difference of two comparable numbers.
    """
    inboard, outboard = case.boundary_midplane_radii()
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    centre = 0.5 * (inboard + outboard)
    half_width = 0.5 * (outboard - inboard)
    radius = np.linspace(
        centre - (1.0 - margin) * half_width,
        centre + (1.0 - margin) * half_width,
        count,
    )
    height = np.linspace(
        -(1.0 - margin) * half_height, (1.0 - margin) * half_height, count
    )
    mesh_radius, mesh_height = np.meshgrid(radius, height, indexing="ij")
    inside = case.contains(mesh_radius, mesh_height)
    return mesh_radius[inside], mesh_height[inside]


def delta_star_finite_difference(
    case: RotatingEquilibrium, radius: Array, height: Array, spacing: float
) -> Array:
    """Return ``DeltaStar(psi)`` from a fourth-order centred stencil.

    Only the flux is sampled, so agreement with the analytic source exercises
    the whole chain -- solution, pressure closure and toroidal-field function --
    rather than one algebraic rearrangement of it.
    """
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    first = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    second = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    radial_first = np.zeros_like(radius)
    radial_second = np.zeros_like(radius)
    vertical_second = np.zeros_like(radius)
    for offset, first_weight, second_weight in zip(offsets, first, second, strict=True):
        shifted_radius = case.flux(radius + offset * spacing, height)
        shifted_height = case.flux(radius, height + offset * spacing)
        radial_first = radial_first + first_weight * shifted_radius
        radial_second = radial_second + second_weight * shifted_radius
        vertical_second = vertical_second + second_weight * shifted_height
    return (
        radial_second / spacing**2
        - radial_first / (spacing * radius)
        + vertical_second / spacing**2
    )
