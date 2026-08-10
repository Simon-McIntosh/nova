r"""Toroidal rotation as a typed force-balance closure on the source.

With a purely toroidal flow :math:`v = R \Omega e_\phi` the convective
acceleration is centripetal, so ideal-MHD momentum balance carries one extra
term,

.. math::
    \nabla p = J \times B + \rho R \Omega^2 e_R .

Its toroidal component is still empty, so :math:`F` remains a flux function
exactly as in the static case and ``ff_prime`` is untouched by everything
here. The poloidal component splits on the two independent directions and
leaves a radial balance beside the Grad-Shafranov equation:

.. math::
    \left(\frac{\partial p}{\partial R}\right)_\psi = \rho R \Omega^2,
    \qquad
    \Delta^\star \Phi = 4 \pi^2 \left[ \mu_0 R^2
        \left(-\frac{\partial p}{\partial \Phi}\right)_R + FF' \right].

**Pressure is no longer a flux function, and the source needs the derivative
at fixed major radius.** That single sentence is the whole difference: a
rotating closure is not a new solver, it is a different function reaching the
one source-evaluation seam in :class:`~nova.equilibrium.source.DomainProfile`.

The isothermal-surface closure
------------------------------
The closure is what fixes :math:`\rho`. Take an ideal gas
:math:`p = \rho T / \bar{m}` with the temperature in energy units, and let
temperature and angular frequency both be flux functions. Radial balance is
then linear in :math:`R^2`,

.. math::
    \left(\frac{\mathrm{d} \ln p}{\mathrm{d} R}\right)_\psi = 2 \theta R,
    \qquad
    \theta(\psi_N) = \frac{\bar{m}\, \Omega^2(\psi_N)}{2\, T(\psi_N)},

so with :math:`u = R^2 - R_0^2` measured from a declared reference radius the
pressure and mass density pile up on the outboard side of every surface by
the same exponential factor,

.. math::
    p(\psi_N, R) = p_0(\psi_N)\, e^{\theta u}, \qquad
    \rho(\psi_N, R) = \frac{\bar{m} p_0}{T}\, e^{\theta u} .

Differentiating at fixed :math:`R` gives the source this module publishes,

.. math::
    \left(-\frac{\partial p}{\partial \Phi}\right)_R
        = \left[ p_0'(\psi_N) + p_0(\psi_N)\, \theta'(\psi_N)\, u \right]
          e^{\theta u},

with :math:`p_0'` and :math:`\theta'` derivatives with respect to the NEGATED
total poloidal flux, the sense every gradient in
:mod:`nova.equilibrium.convention` is written in. Both terms are physical:
:math:`p_0'` is the pressure gradient the static closure already carried, and
the :math:`\theta'` term is the flux gradient of the rotation itself. Dropping
it silently is a different equilibrium wherever the Mach number varies.

The mean particle mass is a declared convention
-----------------------------------------------
:math:`\bar{m}` is the mass per pressure-carrying particle, defined by
:math:`\rho = \bar{m} p / T`, and it is required rather than defaulted
because two conventional readings of the same deuterium plasma differ by a
factor of two: a single fluid at ion temperature carrying only the ion
pressure gives :math:`\bar{m} = m_D`, while a quasineutral plasma whose
pressure is :math:`n(T_i + T_e)` at a common temperature gives
:math:`\bar{m} = m_D / 2`. That factor lands directly in :math:`\theta` and
therefore in every rotational effect on the equilibrium, so the source
declares it and the solve publishes it in its receipt.

A free quartic-radius coefficient is not this closure
-----------------------------------------------------
Expanding for a small centrifugal exponent,
:math:`e^{\theta u} = 1 + \theta u + O(\theta^2 u^2)`, turns the pressure
drive into :math:`\mu_0 R^2 p_0' + \mu_0 p_0' \theta (R^4 - R_0^2 R^2)`,
which is the familiar quartic-radius term of a rotating Solov'ev source. The
coefficient of that quartic is *fixed* by :math:`\bar{m}\Omega^2 / 2T`; it is
not a free column. A generic radial-power basis fitted to a source can
therefore reproduce the term without any of the physics that produces it,
which is why this module exposes the thermodynamic primitives and never a
power expansion. The truncation is a diagnostic of a solution, not a way to
declare one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.source import (
    DomainProfile,
    RotationClosure,
    RotationRecord,
    _validate_flux_function,
)

__all__ = ["IsothermalRotation", "RotatingDomainProfile"]

#: Relative agreement demanded between a declared boundary pressure and the
#: reference-radius pressure the rotating closure carries at the boundary.
BOUNDARY_PRESSURE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class IsothermalRotation:
    """Isothermal-surface thermodynamics of one rotating plasma.

    Temperature and angular frequency are flux functions of normalised flux;
    ``temperature_gradient`` and ``angular_frequency_gradient`` are their
    derivatives with respect to the negated total poloidal flux, the sense
    ``p_prime`` and ``ff_prime`` already use. They are required rather than
    differentiated from the primitives because the caller producing a
    transport or inferred state knows them exactly, and a difference of an
    interpolant would put a quadrature error inside the Grad-Shafranov
    source.

    The temperature is in energy units and the mean particle mass is the mass
    per pressure-carrying particle; together they set the centrifugal
    exponent, and neither has a defensible default.
    """

    temperature: Callable
    angular_frequency: Callable
    temperature_gradient: Callable
    angular_frequency_gradient: Callable
    mean_particle_mass: float
    reference_radius: float

    def __post_init__(self):
        """Refuse a sampled image or an undeclared species convention."""
        for name in (
            "temperature",
            "angular_frequency",
            "temperature_gradient",
            "angular_frequency_gradient",
        ):
            object.__setattr__(
                self, name, _validate_flux_function(getattr(self, name), name)
            )
        for name in ("mean_particle_mass", "reference_radius"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")

    def centrifugal_exponent(self, psi_norm: jax.Array) -> jax.Array:
        """Return ``theta = mbar Omega^2 / (2 T)`` [1/m^2] on each surface."""
        return (
            self.mean_particle_mass
            * self.angular_frequency(psi_norm) ** 2
            / (2.0 * self.temperature(psi_norm))
        )

    def centrifugal_exponent_gradient(self, psi_norm: jax.Array) -> jax.Array:
        """Return the flux gradient of the exponent, in the negated sense.

        The quotient rule on ``theta`` carries the declared gradients through
        unchanged, so a closure whose angular frequency tracks the square
        root of the temperature returns exactly zero here — the uniform
        thermal Mach number branch, where the whole rotation effect sits in
        the exponential and none of it in this term.
        """
        temperature = self.temperature(psi_norm)
        frequency = self.angular_frequency(psi_norm)
        spin = 2.0 * frequency * self.angular_frequency_gradient(psi_norm)
        return (
            self.mean_particle_mass
            * (spin * temperature - frequency**2 * self.temperature_gradient(psi_norm))
            / (2.0 * temperature**2)
        )

    def squared_radius_offset(self, radius: jax.Array) -> jax.Array:
        """Return ``u = R^2 - R_0^2``, the radial variable of the closure."""
        return radius**2 - self.reference_radius**2

    def centrifugal_factor(self, radius: jax.Array, psi_norm: jax.Array) -> jax.Array:
        """Return ``exp(theta u)``, the pile-up of pressure along a surface."""
        return jnp.exp(
            self.centrifugal_exponent(psi_norm) * self.squared_radius_offset(radius)
        )

    def log_pressure_radial_gradient(
        self, radius: jax.Array, psi_norm: jax.Array
    ) -> jax.Array:
        """Return ``d ln p / dR`` at fixed flux, pinned by the closure.

        Radial balance under isothermal surfaces leaves no freedom here: the
        logarithmic derivative is ``2 theta R`` exactly.
        """
        return 2.0 * self.centrifugal_exponent(psi_norm) * radius

    def thermal_mach_number(self, psi_norm: jax.Array) -> jax.Array:
        """Return ``Omega R_0 / sqrt(2 T / mbar)``, equal to ``R_0 sqrt(theta)``."""
        return self.reference_radius * jnp.sqrt(self.centrifugal_exponent(psi_norm))

    def mass_density(self, psi_norm: jax.Array, pressure: jax.Array) -> jax.Array:
        """Return ``rho = mbar p / T`` [kg/m^3] from the local pressure."""
        return self.mean_particle_mass * pressure / self.temperature(psi_norm)


@dataclass(frozen=True, kw_only=True)
class RotatingDomainProfile(DomainProfile):
    """Absolute flux-function gradients under a toroidal-rotation closure.

    ``reference_pressure`` is :math:`p_0(\\psi_N)`, the pressure on a surface
    at the closure's reference major radius, and ``p_prime`` is its gradient
    :math:`-\\mathrm{d}p_0/\\mathrm{d}\\Phi` in the same negated-flux sense a
    static profile uses. The primitive is required, not integrated from the
    gradient: the source needs it at every evaluation of the fixed point,
    where the flux span the integration would need is not yet known, and a
    producer of a rotating state has it in hand.

    ``ff_prime`` keeps its static meaning exactly. Rotation leaves the
    toroidal-field function a flux function, so nothing on the diamagnetic
    side of the source changes.
    """

    reference_pressure: Callable
    rotation: IsothermalRotation

    def __post_init__(self):
        """Validate the pressure primitive and the declared closure."""
        super().__post_init__()
        object.__setattr__(
            self,
            "reference_pressure",
            _validate_flux_function(self.reference_pressure, "reference_pressure"),
        )
        if not isinstance(self.rotation, IsothermalRotation):
            raise TypeError("rotation must be an IsothermalRotation closure")

    def validate_boundary_pressure(self, boundary_pressure) -> None:
        """Refuse a boundary primitive the pressure profile contradicts.

        A rotating plasma has no single boundary pressure: the centrifugal
        factor varies along the boundary contour, so the declared value is
        the one AT THE REFERENCE RADIUS and has to agree with the pressure
        profile there. Catching the disagreement at construction stops a
        solve from reporting an integral pressure moment built on one
        primitive and a force balance built on another.
        """
        sampled = jnp.asarray(self.reference_pressure(jnp.asarray([0.0, 1.0])))
        axis, edge = float(sampled[0]), float(sampled[-1])
        scale = max(abs(axis), abs(edge), abs(float(boundary_pressure)), 1.0)
        if abs(edge - float(boundary_pressure)) > BOUNDARY_PRESSURE_TOLERANCE * scale:
            raise ValueError(
                f"the reference-radius pressure at the boundary ({edge:.6g} Pa) "
                f"contradicts the declared boundary pressure "
                f"({float(boundary_pressure):.6g} Pa); a rotating closure "
                "measures the boundary primitive at its reference radius"
            )

    def pressure_gradient(self, radius: jax.Array, psi_norm: jax.Array) -> jax.Array:
        """Return the pressure flux gradient [Pa/Wb] at fixed major radius.

        Both terms of the closure reach the source: the reference-radius
        gradient and the flux gradient of the rotation itself. With no
        rotation declared anywhere the exponent and its gradient are zero and
        this returns the declared ``p_prime`` unchanged, bit for bit.
        """
        offset = self.rotation.squared_radius_offset(radius)
        return (
            self.p_prime(psi_norm)
            + self.reference_pressure(psi_norm)
            * self.rotation.centrifugal_exponent_gradient(psi_norm)
            * offset
        ) * self.rotation.centrifugal_factor(radius, psi_norm)

    def pressure(
        self,
        radius: jax.Array,
        psi_norm: jax.Array,
        boundary_pressure,
        flux_span,
    ) -> jax.Array:
        """Return ``p_0 exp(theta u)``, the pressure on every cell.

        The declared primitive is used directly rather than integrated back
        from its gradient; the boundary check at construction is what ties
        the two together.
        """
        return self.reference_pressure(psi_norm) * self.rotation.centrifugal_factor(
            radius, psi_norm
        )

    def radial_body_force(
        self, radius: jax.Array, psi_norm: jax.Array, pressure: jax.Array
    ) -> jax.Array:
        """Return the centrifugal force density ``rho R Omega^2`` [N/m^3].

        This is the term that makes the outboard pressure pile-up a force
        balance rather than an unbalanced gradient, so the force receipt of a
        rotating solve is read against the same residual a static one is.
        """
        return (
            self.rotation.mass_density(psi_norm, pressure)
            * radius
            * self.rotation.angular_frequency(psi_norm) ** 2
        )

    def rotation_record(self, radius: jax.Array, masks: DomainMasks) -> RotationRecord:
        """Return the rotation receipt of one labelled flux map."""
        dtype = jnp.asarray(masks.psi_norm).dtype
        factor = jnp.where(
            masks.core,
            self.rotation.centrifugal_factor(radius, masks.psi_norm),
            1.0,
        )
        return RotationRecord(
            closure=jnp.asarray(
                int(RotationClosure.ISOTHERMAL_SURFACE), dtype=jnp.int8
            ),
            reference_radius=jnp.asarray(self.rotation.reference_radius, dtype=dtype),
            mean_particle_mass=jnp.asarray(
                self.rotation.mean_particle_mass, dtype=dtype
            ),
            axis_mach_number=jnp.asarray(
                self.rotation.thermal_mach_number(jnp.zeros((), dtype=dtype)),
                dtype=dtype,
            ),
            minimum_centrifugal_factor=jnp.min(factor),
            maximum_centrifugal_factor=jnp.max(factor),
        )
