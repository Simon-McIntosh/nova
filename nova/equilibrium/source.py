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
is explicit rather than empty. A forward caller may instead supply a scalar
plasma-current target; the map records
:attr:`NormalisationPolicy.DECLARED_SCALAR_CURRENT` and the eliminated common
amplitude without changing these immutable profile functions.

Source support is domain qualified. A closure declared on
:class:`~nova.equilibrium.domain.PlasmaDomain.CORE` drives current on the
axis-connected closed surfaces and nowhere else, so no current can appear in
the scrape-off layer, the private-flux region or the excluded material. An
open region carries current only when a continuation is declared on it, and
each open domain carries its own: the common scrape-off layer and the
private-flux region are separate physical contracts, never one extrapolation
applied twice. :mod:`nova.equilibrium.continuation` builds those closures;
this module holds the declaration, the receipt and the domain-selected
evaluation.

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
    flux_function_toroidal_field,
    toroidal_current_density,
)
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.observation import gradient_tail
from nova.equilibrium.stencil_mesh import CellCurrentMoments

__all__ = [
    "ContinuationForm",
    "ContinuationLedger",
    "ContinuationRecord",
    "DomainProfile",
    "ForwardSource",
    "NormalisationPolicy",
    "NormalisationRecord",
    "CurrentNormalisationError",
    "SCALAR_CURRENT_AMPLITUDE_BAND",
    "RotationClosure",
    "RotationRecord",
    "SeparatrixContinuity",
    "undeclared_continuation_ledger",
    "undeclared_continuation_record",
]

#: Open domains a continuation may be declared on, in argument order.
OPEN_DOMAINS: tuple[tuple[str, PlasmaDomain], ...] = (
    ("common_sol", PlasmaDomain.COMMON_SOL),
    ("private_flux", PlasmaDomain.PRIVATE_FLUX),
)


class NormalisationPolicy(IntEnum):
    """How the supplied source amplitude is fixed."""

    ABSOLUTE = 0
    DECLARED_SCALAR_CURRENT = 1


#: Inclusive amplitude range admitted by declared-current normalisation.
SCALAR_CURRENT_AMPLITUDE_BAND = (1.0e-6, 1.0e6)


class CurrentNormalisationError(RuntimeError):
    """Report an inadmissible declared-current source amplitude."""

    def __init__(self, amplitude: float):
        self.amplitude = float(amplitude)
        super().__init__(
            f"source amplitude {self.amplitude:.12g} is outside "
            f"{SCALAR_CURRENT_AMPLITUDE_BAND}"
        )


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


class ContinuationForm(IntEnum):
    """Functional family a bounded open-region continuation is carried on.

    ``UNDECLARED`` is the absence of a continuation, which is what an open
    domain carries unless the caller declares one; it is not a continuation of
    zero amplitude. The two declared families differ in what happens at the
    outer support bound: the polynomial vanishes there to the same order it
    matches at the separatrix, while the exponential is truncated and the
    receipt reports the amplitude the truncation discarded.
    """

    UNDECLARED = -1
    HERMITE_POLYNOMIAL = 0
    EXPONENTIAL_DECAY = 1


class SeparatrixContinuity(IntEnum):
    """Orders a continuation matches the core closure at the separatrix.

    The member value IS the number of matched orders, so
    ``VALUE_AND_GRADIENT`` matches two — the value and the first derivative —
    and is the minimum a continuation may declare. The two members below it
    leave a jump in the source at the separatrix, which is carried by a
    surface current or a thin current layer; declaring one fails until such a
    layer is modelled, because a jump that arises numerically is not a
    physical sheet.
    """

    UNDECLARED = -1
    VALUE_JUMP = 0
    GRADIENT_JUMP = 1
    VALUE_AND_GRADIENT = 2
    VALUE_GRADIENT_AND_CURVATURE = 3

    @property
    def matched_orders(self) -> int:
        """Return how many orders the class matches at the separatrix."""
        return max(int(self), 0)


class ContinuationRecord(NamedTuple):
    """The continuation one open domain was solved under.

    ``support`` is the outer bound in separatrix distance — normalised flux
    measured away from the separatrix on the domain's own branch — beyond
    which the source is exactly zero. ``truncated_fraction`` is the amplitude
    left at that bound relative to the separatrix value, zero for a family
    that vanishes there, so a truncated continuation cannot hide the step it
    takes to zero.

    The two separatrix gradients are published because a continuation is only
    readable against the closure it continues: they are the values the
    continuity class pinned, and a private-flux continuation carrying a
    different pair would be a value jump rather than an independent policy.
    """

    domain: jax.Array
    form: jax.Array
    continuity: jax.Array
    support: jax.Array
    decay_width: jax.Array
    separatrix_pressure_gradient: jax.Array
    separatrix_diamagnetic_gradient: jax.Array
    truncated_fraction: jax.Array

    @property
    def active(self) -> jax.Array:
        """Return whether a continuation was declared on this domain."""
        return jnp.not_equal(self.form, int(ContinuationForm.UNDECLARED))

    @property
    def form_name(self) -> str:
        """Return the declared functional-family name."""
        return ContinuationForm(int(self.form)).name.lower()

    @property
    def continuity_name(self) -> str:
        """Return the declared separatrix-continuity class name."""
        return SeparatrixContinuity(int(self.continuity)).name.lower()

    @property
    def domain_name(self) -> str:
        """Return the domain the continuation is declared on."""
        if not bool(self.active):
            return "undeclared"
        return PlasmaDomain(int(self.domain)).name.lower()


class ContinuationLedger(NamedTuple):
    """The continuation declared on each open domain of one solve."""

    common_sol: ContinuationRecord
    private_flux: ContinuationRecord

    @property
    def active(self) -> jax.Array:
        """Return whether either open domain carries a declared continuation."""
        return self.common_sol.active | self.private_flux.active


def undeclared_continuation_record(dtype=jnp.float64) -> ContinuationRecord:
    """Return the receipt of an open domain no closure was declared on."""
    return ContinuationRecord(
        domain=jnp.asarray(int(ContinuationForm.UNDECLARED), dtype=jnp.int8),
        form=jnp.asarray(int(ContinuationForm.UNDECLARED), dtype=jnp.int8),
        continuity=jnp.asarray(int(SeparatrixContinuity.UNDECLARED), dtype=jnp.int8),
        support=jnp.asarray(0.0, dtype=dtype),
        decay_width=jnp.asarray(jnp.nan, dtype=dtype),
        separatrix_pressure_gradient=jnp.asarray(0.0, dtype=dtype),
        separatrix_diamagnetic_gradient=jnp.asarray(0.0, dtype=dtype),
        truncated_fraction=jnp.asarray(0.0, dtype=dtype),
    )


def undeclared_continuation_ledger(dtype=jnp.float64) -> ContinuationLedger:
    """Return the receipt of a solve whose source drives the core alone."""
    record = undeclared_continuation_record(dtype)
    return ContinuationLedger(common_sol=record, private_flux=record)


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

    def field_function_squared(
        self, psi_norm: jax.Array, boundary_field_function, flux_span
    ) -> jax.Array:
        """Return the squared toroidal-field function [T^2 m^2] of the closure.

        The same boundary-inward integration the pressure primitive uses,
        applied to the diamagnetic gradient. A closure spanning a domain the
        core integration does not reach owns its own primitive and overrides
        this.
        """
        return flux_function_toroidal_field(
            boundary_field_function, flux_span, gradient_tail(self.ff_prime, psi_norm)
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

    @property
    def rotation_closure(self) -> RotationClosure:
        """Return the force-balance closure the source is formed under.

        A continuation reads its separatrix anchor off this profile, so it has
        to know whether the anchor is a flux function at all: a rotating
        closure's pressure gradient depends on major radius and cannot be
        continued without continuing the thermodynamic primitives that produce
        it.
        """
        return RotationClosure.STATIC

    def continuation_record(self, dtype=jnp.float64) -> ContinuationRecord:
        """Return the receipt of a closure that continues nothing.

        A profile carrying no continuation policy is not admissible on an open
        domain: the declaration a continuation needs — continuity class,
        functional form and outer support — is exactly what this record
        reports as undeclared.
        """
        return undeclared_continuation_record(dtype)


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


def declared_scalar_current_record(
    amplitude: jax.Array, dtype=jnp.float64
) -> NormalisationRecord:
    """Return the receipt of an exact declared-current amplitude elimination."""
    value = jnp.asarray(amplitude, dtype=dtype)
    return NormalisationRecord(
        policy=jnp.asarray(
            int(NormalisationPolicy.DECLARED_SCALAR_CURRENT), dtype=jnp.int8
        ),
        amplitude=value,
        rescaled=jnp.not_equal(value, jnp.asarray(1.0, dtype=dtype)),
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

    ``common_sol`` and ``private_flux`` name the open-region closures, and
    each is an independent declaration: a bare
    :class:`DomainProfile` is refused on either argument because it carries no
    continuity class, functional form or outer support, and a continuation
    built for one branch is refused on the other. Neither may be defaulted
    from the other or from the core.
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
        for name, domain in OPEN_DOMAINS:
            profile = getattr(self, name)
            if profile is None:
                continue
            if not isinstance(profile, DomainProfile):
                raise TypeError(f"{name} must be a DomainProfile")
            record = profile.continuation_record()
            if not bool(record.active):
                raise NotImplementedError(
                    f"a {name} closure needs a declared continuity class, "
                    "functional form and outer support; the supplied profile "
                    "declares none, and an absolute source without a "
                    "continuation drives the core alone"
                )
            if int(record.domain) != int(domain):
                raise ValueError(
                    f"the {name} argument carries a continuation declared on "
                    f"{record.domain_name}; the separatrix distance runs the "
                    "other way there, so each open domain needs its own"
                )
            profile.validate_separatrix_match(self.core)

    @property
    def closure_degrees(self) -> int:
        """Return the number of scalar unknowns the closure may solve for."""
        return 0

    def normalisation_record(
        self, dtype=jnp.float64, *, amplitude: jax.Array | None = None
    ) -> NormalisationRecord:
        """Return the absolute or declared-current action taken on this source."""
        if amplitude is None:
            return absolute_normalisation_record(dtype)
        return declared_scalar_current_record(amplitude, dtype)

    @property
    def open_profiles(self) -> tuple[tuple[PlasmaDomain, DomainProfile], ...]:
        """Return the declared open-region closures with their domains."""
        return tuple(
            (domain, getattr(self, name))
            for name, domain in OPEN_DOMAINS
            if getattr(self, name) is not None
        )

    def rotation_record(self, radius: jax.Array, masks: DomainMasks) -> RotationRecord:
        """Return the rotation receipt of the declared core closure."""
        return self.core.rotation_record(radius, masks)

    def continuation_ledger(self, dtype=jnp.float64) -> ContinuationLedger:
        """Return the continuation receipt of both open domains."""
        return ContinuationLedger(
            **{
                name: (
                    undeclared_continuation_record(dtype)
                    if getattr(self, name) is None
                    else getattr(self, name).continuation_record(dtype)
                )
                for name, _ in OPEN_DOMAINS
            }
        )

    def declared_support(self, masks: DomainMasks) -> jax.Array:
        """Return the mask of cells any declared closure drives current on."""
        support = masks.core
        for domain, _ in self.open_profiles:
            support = support | masks.mask(domain)
        return support

    def current_density(self, radius: jax.Array, masks: DomainMasks) -> jax.Array:
        """Return the toroidal current density [A/m^2] the declared closures drive.

        Each closure is evaluated on its own domain and summed, so the
        partition the topology read publishes decides which gradient pair
        reaches which cell and a cell no closure was declared on carries
        exactly zero.

        The evaluation point is held on the declared support as well. Selecting
        the result alone would leave the same numbers, but a closure is only
        required to be a physical function of flux where it was declared, and a
        value the selection discards still poisons a reverse-mode derivative of
        everything downstream of it. The open closures are held at the
        separatrix, which is inside the support of either branch.
        """
        density = jnp.where(
            masks.core,
            self.core.current_density(
                radius, jnp.where(masks.core, masks.psi_norm, 0.0)
            ),
            0.0,
        )
        for domain, profile in self.open_profiles:
            selection = masks.mask(domain)
            density = density + jnp.where(
                selection,
                profile.current_density(
                    radius, jnp.where(selection, masks.psi_norm, 1.0)
                ),
                0.0,
            )
        return density

    def cell_current(
        self, radius: jax.Array, area: jax.Array, masks: DomainMasks
    ) -> jax.Array:
        """Return per-cell toroidal current [A] on profile-owned support.

        Domain labels choose the declared closure while profile participation
        is the sole geometric support. A participating cell with no declared
        closure receives zero from :meth:`current_density`; shadow and
        excluded-material cells are removed explicitly.
        """
        return jnp.where(
            masks.profile_participation,
            self.current_density(radius, masks) * area,
            0.0,
        )

    def current_moments(
        self,
        masks: DomainMasks,
        support_moments,
        profile_support,
        *_,
        sample_flux=None,
    ) -> CellCurrentMoments:
        """Return profile-owned current moments without a boundary clip.

        The static material boundary and the saddle-aware private-flux shadow
        are the only geometric participation decisions. Achieved domain labels
        select the core or declared common-SOL flux function, and the latter's
        declared support owns where its exterior current becomes exactly zero.
        """

        core = support_moments(
            self.core,
            masks.psi_norm,
            sample_flux,
            profile_support,
        )
        total = jnp.stack(
            CellCurrentMoments(*(jnp.where(masks.core, entry, 0.0) for entry in core))
        )
        if self.common_sol is None:
            return CellCurrentMoments(*total)

        common = support_moments(
            self.common_sol,
            masks.psi_norm,
            sample_flux,
            profile_support,
        )
        record = self.common_sol.continuation_record(masks.psi_norm.dtype)
        common_distance = jnp.maximum(masks.psi_norm - 1.0, 0.0)
        common_selection = masks.common_sol & (common_distance <= record.support)
        total = total + jnp.stack(
            CellCurrentMoments(
                *(jnp.where(common_selection, entry, 0.0) for entry in common)
            )
        )
        return CellCurrentMoments(*total)
