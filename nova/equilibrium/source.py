r"""Immutable prescribed source state for the forward equilibrium solve.

The state is a physical flux-function state, not a fitted one: it carries the
absolute pressure gradient :math:`p'` and diamagnetic gradient :math:`FF'`
that a transport model or an upstream causal estimator produced, together
with the boundary primitives :math:`p_b` and :math:`F_b` that turn those
gradients back into pressure and toroidal-field function. Nothing in this
module reads a magnetic measurement, forms a whitened residual or updates a
coefficient; the forward solve consumes the state exactly as supplied.

Normalisation is declared, never inferred. Under
:attr:`NormalisationPolicy.ABSOLUTE` the supplied gradients reach the source
evaluation unchanged, the cell-current image is never rescaled to hit a net
current, and :class:`NormalisationRecord` reports a unit scale so the receipt
is explicit rather than empty. The alternative policy is present as a typed
seam for one declared scalar amplitude closed by a target plasma current; it
is not part of the shipped closure and constructing it fails loudly.

Source support is domain qualified. A closure declared on
:class:`~nova.equilibrium.domain.PlasmaDomain.CORE` drives current on the
axis-connected closed surfaces and nowhere else, so no current can appear in
the scrape-off layer, the private-flux region or the excluded material.

Every profile evaluation is written against major radius as well as
normalised flux. Under a static closure pressure is a flux function and the
radius does not enter, so the declared gradient reaches the source unchanged;
under a force-balance closure that makes pressure vary along a surface — the
toroidal-rotation closure in :mod:`nova.equilibrium.rotation` is the first —
the Grad-Shafranov source needs the pressure derivative at FIXED major
radius, which is a different function of two arguments. Carrying the radius
through the one evaluation seam keeps that a typed override rather than a
branch inside the solve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium.convention import (
    flux_function_pressure,
    toroidal_current_density,
)
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.observation import gradient_tail

__all__ = [
    "DomainProfile",
    "ForwardSource",
    "NormalisationPolicy",
    "NormalisationRecord",
    "RotationClosure",
    "RotationRecord",
]


class NormalisationPolicy(IntEnum):
    """How the supplied source amplitude is fixed."""

    ABSOLUTE = 0
    DECLARED_SCALAR_CURRENT = 1


class RotationClosure(IntEnum):
    """Thermodynamic closure the toroidal-rotation term is formed under.

    ``STATIC`` is the absence of a rotation closure, not a rotation of zero
    frequency: it names a source that never declared temperature, angular
    frequency or a species convention. A polytropic-surface closure is a
    further member of this enumeration and not a variant of the isothermal
    one; adding it is a new typed profile, not a flag.
    """

    STATIC = 0
    ISOTHERMAL_SURFACE = 1


def _validate_flux_function(value, name: str) -> Callable:
    """Reject anything but a callable flux function of normalised flux.

    A sampled array reaching this argument would be a measurement, a sensor
    image or a cell-current image standing in for a physical closure. None of
    those is a force-balanced source, so they are refused at the boundary
    rather than silently interpolated.
    """
    if callable(value):
        return value
    if hasattr(value, "shape") or isinstance(value, list | tuple):
        raise TypeError(
            f"{name} must be a callable flux function of normalised flux; "
            f"an array of shape {getattr(value, 'shape', (len(value),))} is not "
            "a force-balanced source"
        )
    raise TypeError(f"{name} must be a callable flux function of normalised flux")


@dataclass(frozen=True)
class DomainProfile:
    """Absolute flux-function gradients declared on one plasma domain.

    Both callables take normalised flux and return SI gradients with respect
    to the negated total poloidal flux, the sense pinned in
    :mod:`nova.equilibrium.convention`.
    """

    p_prime: Callable
    ff_prime: Callable

    def __post_init__(self):
        """Refuse a sampled image standing in for a flux function."""
        object.__setattr__(
            self, "p_prime", _validate_flux_function(self.p_prime, "p_prime")
        )
        object.__setattr__(
            self, "ff_prime", _validate_flux_function(self.ff_prime, "ff_prime")
        )

    def pressure_gradient(self, radius: jax.Array, psi_norm: jax.Array) -> jax.Array:
        """Return the pressure flux gradient [Pa/Wb] at fixed major radius.

        A static closure leaves pressure a flux function, so the radius does
        not enter and the declared gradient is returned unchanged. The
        argument is present because the Grad-Shafranov source is defined by
        the derivative at fixed major radius, and a closure that piles
        pressure up along a surface makes that derivative radius dependent.
        """
        return self.p_prime(psi_norm)

    def current_density(self, radius: jax.Array, psi_norm: jax.Array) -> jax.Array:
        """Return the toroidal current density [A/m^2] this closure drives."""
        return toroidal_current_density(
            radius, self.pressure_gradient(radius, psi_norm), self.ff_prime(psi_norm)
        )

    def pressure(
        self,
        radius: jax.Array,
        psi_norm: jax.Array,
        boundary_pressure,
        flux_span,
    ) -> jax.Array:
        """Return the pressure [Pa] this closure implies on every cell.

        Recovered from the declared gradient and the boundary primitive by
        the inward integration pinned in :mod:`nova.equilibrium.convention`.
        """
        return flux_function_pressure(
            boundary_pressure, flux_span, gradient_tail(self.p_prime, psi_norm)
        )

    def radial_body_force(
        self, radius: jax.Array, psi_norm: jax.Array, pressure: jax.Array
    ) -> jax.Array:
        """Return the non-magnetic radial force density [N/m^3] of the closure.

        Static force balance stands between the Lorentz force and the
        pressure gradient alone, so this vanishes identically. A closure with
        a body force returns it here, and the force receipt in
        :mod:`nova.equilibrium.conservation` reads the same balance for both.
        """
        return jnp.zeros_like(pressure)

    def validate_boundary_pressure(self, boundary_pressure) -> None:
        """Check the declared boundary primitive against the closure's own.

        A static closure carries no pressure primitive of its own — the
        boundary value is what makes its gradient integrable — so there is
        nothing to contradict.
        """

    def rotation_record(
        self, radius: jax.Array, masks: DomainMasks
    ) -> "RotationRecord":
        """Return the receipt of a closure that declared no rotation."""
        return static_rotation_record(jnp.asarray(masks.psi_norm).dtype)


class NormalisationRecord(NamedTuple):
    """Every normalisation action the solve took on the supplied source."""

    policy: jax.Array
    amplitude: jax.Array
    rescaled: jax.Array

    @property
    def policy_name(self) -> str:
        """Return the declared policy name."""
        return NormalisationPolicy(int(self.policy)).name.lower()


def absolute_normalisation_record(dtype=jnp.float64) -> NormalisationRecord:
    """Return the receipt of a solve that changed nothing about the source."""
    return NormalisationRecord(
        policy=jnp.asarray(int(NormalisationPolicy.ABSOLUTE), dtype=jnp.int8),
        amplitude=jnp.asarray(1.0, dtype=dtype),
        rescaled=jnp.asarray(False),
    )


class RotationRecord(NamedTuple):
    """The rotation closure a solve ran under and what it cost the solution.

    ``reference_radius`` and ``mean_particle_mass`` are the two conventions a
    rotating source cannot be read without: the radius the pressure pile-up
    is measured from, and the mass per pressure-carrying particle that turns
    an angular frequency into a centrifugal exponent. A single-species and a
    quasineutral reading of the same plasma differ by a factor of two in that
    mass and therefore in the exponent, so it is published rather than
    assumed. Both are not-a-number under a static closure, which declares
    neither.

    The centrifugal factors bracket :math:`\\exp[\\theta (R^2 - R_0^2)]` over
    the labelled core, so the receipt states the outboard-to-inboard pressure
    ratio the closure actually produced instead of the one it was asked for.
    """

    closure: jax.Array
    reference_radius: jax.Array
    mean_particle_mass: jax.Array
    axis_mach_number: jax.Array
    minimum_centrifugal_factor: jax.Array
    maximum_centrifugal_factor: jax.Array

    @property
    def closure_name(self) -> str:
        """Return the declared rotation-closure name."""
        return RotationClosure(int(self.closure)).name.lower()

    @property
    def active(self) -> jax.Array:
        """Return whether a rotation closure drove the source."""
        return jnp.not_equal(self.closure, int(RotationClosure.STATIC))


def static_rotation_record(dtype=jnp.float64) -> RotationRecord:
    """Return the receipt of a solve whose source declared no rotation."""
    return RotationRecord(
        closure=jnp.asarray(int(RotationClosure.STATIC), dtype=jnp.int8),
        reference_radius=jnp.asarray(jnp.nan, dtype=dtype),
        mean_particle_mass=jnp.asarray(jnp.nan, dtype=dtype),
        axis_mach_number=jnp.asarray(0.0, dtype=dtype),
        minimum_centrifugal_factor=jnp.asarray(1.0, dtype=dtype),
        maximum_centrifugal_factor=jnp.asarray(1.0, dtype=dtype),
    )


@dataclass(frozen=True)
class ForwardSource:
    """Immutable absolute flux-function state of one equilibrium.

    ``boundary_pressure`` and ``boundary_field_function`` are the primitives
    at the plasma boundary, :math:`p_b` in Pa and :math:`F_b = R B_\\phi` in
    T m. They are required because a gradient pair alone cannot produce the
    pressure and toroidal-field function that the integral observations and
    the force residual are defined on.

    ``common_sol`` and ``private_flux`` name the open-region closures. They
    are typed here so a caller cannot smuggle an open-region source through
    the core argument, but a static absolute solve declares neither: the
    continuation policy, its separatrix continuity class and its outer
    support are a separate physical contract.
    """

    core: DomainProfile
    boundary_pressure: float = 0.0
    boundary_field_function: float = 0.0
    common_sol: DomainProfile | None = None
    private_flux: DomainProfile | None = None
    normalisation: NormalisationPolicy = NormalisationPolicy.ABSOLUTE

    def __post_init__(self):
        """Validate the declared closure set and the normalisation policy."""
        if not isinstance(self.core, DomainProfile):
            raise TypeError("core must be a DomainProfile")
        self.core.validate_boundary_pressure(self.boundary_pressure)
        if self.normalisation is not NormalisationPolicy.ABSOLUTE:
            raise NotImplementedError(
                "the shipped forward closure preserves the supplied source "
                "exactly; a declared scalar amplitude closed by a target "
                "plasma current is a separate normalisation policy"
            )
        for name in ("common_sol", "private_flux"):
            if getattr(self, name) is not None:
                raise NotImplementedError(
                    f"a {name} closure needs a declared continuity class, "
                    "branch policy and outer support; the static absolute "
                    "source drives the core alone"
                )

    @property
    def closure_degrees(self) -> int:
        """Return the number of scalar unknowns the closure may solve for."""
        return 0

    def rotation_record(self, radius: jax.Array, masks: DomainMasks) -> RotationRecord:
        """Return the rotation receipt of the declared core closure."""
        return self.core.rotation_record(radius, masks)

    def cell_current(
        self, radius: jax.Array, area: jax.Array, masks: DomainMasks
    ) -> jax.Array:
        """Return the per-cell toroidal current [A] on the declared support.

        The current density is evaluated from the supplied gradients without
        any amplitude change and is then selected by domain label, so a cell
        outside the declared support carries exactly zero.

        The evaluation point is held on the declared support as well. Selecting
        the result alone would leave the same numbers, but a closure is only
        required to be a physical function of flux where it was declared, and a
        value the selection discards still poisons a reverse-mode derivative of
        everything downstream of it.
        """
        psi_norm = jnp.where(masks.core, masks.psi_norm, 0.0)
        density = self.core.current_density(radius, psi_norm)
        return jnp.where(masks.core, density * area, 0.0)
