r"""Eich-scaled, topology-qualified toroidal-current closure for the common SOL.

The empirical H-mode multi-machine scaling used here is

.. math::
    \lambda_q[\mathrm{mm}] = 0.63\,B_{p,\mathrm{OMP}}[\mathrm{T}]^{-1.19}.

It is evaluated from the solved poloidal-field components at the outboard
midplane. Nova carries total poloidal flux, so

.. math::
    |\nabla\Phi| = 2\pi R B_p,
    \qquad
    \lambda_{\psi_N} =
    \frac{2\pi R_{\mathrm{OMP}} B_{p,\mathrm{OMP}}\lambda_q}
         {|\Phi_b-\Phi_a|}.

The resulting exponential continues the confined pressure and diamagnetic
gradients with their value and first derivative intact. Consequently their
toroidal current density has the same value and first derivative at fixed
major radius. Evaluation selects points with normalised flux above one inside
the material boundary, while the topological private mask remains an explicit
veto. A reported support extent is a measurement of where the current becomes
insignificant, never an input to the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.continuation import ContinuedDomainProfile, SeparatrixContinuation
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.source import (
    ContinuationForm,
    DomainProfile,
    SeparatrixContinuity,
)

__all__ = [
    "EichSolClosure",
    "EichWidth",
    "SolDecayVariant",
    "eich_width",
]

EICH_WIDTH_COEFFICIENT_M = 0.63e-3
EICH_FIELD_EXPONENT = -1.19


class SolDecayVariant(IntEnum):
    """Exponential channels included in one common-SOL comparison."""

    SINGLE_LENGTH = 0
    DUAL_LENGTH = 1


@dataclass(frozen=True)
class EichWidth:
    """Physical and normalised-flux forms of one evaluated Eich width."""

    outboard_midplane_poloidal_field_t: float
    outboard_midplane_radius_m: float
    flux_span_wb: float
    heat_flux_width_m: float
    normalized_flux_width: float


def _positive_finite(value, name: str) -> float:
    """Return a positive finite scalar or reject an invalid equilibrium read."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def eich_width(
    *,
    outboard_midplane_radius_m,
    radial_poloidal_field_t,
    vertical_poloidal_field_t,
    flux_span_wb,
) -> EichWidth:
    """Evaluate the Eich width and map it into Nova's normalised total flux.

    The two field components must come from the equilibrium at its outboard
    midplane separatrix. Their magnitude, rather than a global or vacuum-field
    proxy, is the field entering the scaling.
    """
    radius = _positive_finite(outboard_midplane_radius_m, "outboard radius")
    flux_span = _positive_finite(abs(float(flux_span_wb)), "flux span")
    radial = float(radial_poloidal_field_t)
    vertical = float(vertical_poloidal_field_t)
    if not math.isfinite(radial) or not math.isfinite(vertical):
        raise ValueError("outboard-midplane poloidal-field components must be finite")
    poloidal_field = _positive_finite(
        math.hypot(radial, vertical), "outboard-midplane poloidal field"
    )
    physical_width = EICH_WIDTH_COEFFICIENT_M * poloidal_field**EICH_FIELD_EXPONENT
    normalized_width = (
        TOTAL_FLUX_FACTOR * radius * poloidal_field * physical_width / flux_span
    )
    return EichWidth(
        outboard_midplane_poloidal_field_t=poloidal_field,
        outboard_midplane_radius_m=radius,
        flux_span_wb=flux_span,
        heat_flux_width_m=physical_width,
        normalized_flux_width=normalized_width,
    )


@dataclass(frozen=True)
class EichSolClosure:
    """Build single- and dual-length common-SOL source continuations.

    ``spreading_length_m`` is the physical outboard-midplane width of the
    broader channel. It is compared with the Eich width rather than replacing
    it. Both channels independently match the confined separatrix value and
    derivative, so their weighted sum does too.
    """

    width: EichWidth
    spreading_length_m: float
    spreading_fraction: float = 0.5

    def __post_init__(self):
        """Validate the broader channel without defining any support cutoff."""
        spreading = _positive_finite(self.spreading_length_m, "spreading length")
        if spreading <= self.width.heat_flux_width_m:
            raise ValueError("the spreading length must be broader than lambda_q")
        if not 0.0 < float(self.spreading_fraction) < 1.0:
            raise ValueError(
                "the spreading fraction must lie strictly between zero and one"
            )

    @property
    def spreading_normalized_flux_width(self) -> float:
        """Return the broader physical length mapped through the same equilibrium."""
        return (
            self.width.normalized_flux_width
            * float(self.spreading_length_m)
            / self.width.heat_flux_width_m
        )

    def policy(self, variant: SolDecayVariant) -> SeparatrixContinuation:
        """Return a material-bounded continuation for one measured variant."""
        variant = SolDecayVariant(variant)
        dual = variant is SolDecayVariant.DUAL_LENGTH
        return SeparatrixContinuation(
            form=ContinuationForm.EXPONENTIAL_DECAY,
            continuity=SeparatrixContinuity.VALUE_AND_GRADIENT,
            support=None,
            decay_width=self.width.normalized_flux_width,
            spreading_width=(self.spreading_normalized_flux_width if dual else None),
            spreading_fraction=float(self.spreading_fraction) if dual else 0.0,
        )

    def domain_profile(
        self, confined: DomainProfile, variant: SolDecayVariant
    ) -> ContinuedDomainProfile:
        """Continue a confined source onto the material-bounded common SOL."""
        return self.policy(variant).extend_open_field_line(confined)

    def current_density(
        self,
        confined: DomainProfile,
        radius,
        masks: DomainMasks,
        variant: SolDecayVariant,
    ):
        """Return SOL toroidal current selected by live flux and topology."""
        profile = self.domain_profile(confined, variant)
        selection = masks.open_field_line
        safe_flux = jnp.where(selection, masks.psi_norm, 1.0)
        return jnp.where(
            selection,
            profile.current_density(jnp.asarray(radius), safe_flux),
            0.0,
        )

    def support_extent(
        self,
        confined: DomainProfile,
        variant: SolDecayVariant,
        *,
        insignificant_fraction: float = 1.0e-6,
    ) -> float:
        r"""Measure where outboard-midplane current falls below a given fraction.

        The returned value is :math:`\psi_N`, not a distance. The last crossing
        is used, so an anchored prefactor that changes sign cannot make an
        earlier zero look like the asymptotic support extent.
        """
        floor = float(insignificant_fraction)
        if not 0.0 < floor < 1.0:
            raise ValueError(
                "the insignificance fraction must lie between zero and one"
            )
        profile = self.domain_profile(confined, variant)
        radius = self.width.outboard_midplane_radius_m
        separatrix = abs(float(profile.current_density(radius, jnp.asarray(1.0))))
        if separatrix == 0.0:
            raise ValueError("support extent is undefined for zero separatrix current")

        widest = self.width.normalized_flux_width
        if SolDecayVariant(variant) is SolDecayVariant.DUAL_LENGTH:
            widest = self.spreading_normalized_flux_width
        maximum_distance = 64.0 * widest
        distance = np.linspace(0.0, maximum_distance, 16385)
        amplitude = np.abs(
            np.asarray(profile.current_density(radius, jnp.asarray(1.0 + distance)))
        )
        above = np.flatnonzero(amplitude >= floor * separatrix)
        if above.size == 0 or above[-1] == distance.size - 1:
            raise RuntimeError(
                "the exponential tail did not resolve its support extent"
            )

        lower = float(distance[above[-1]])
        upper = float(distance[above[-1] + 1])
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            ratio = (
                abs(float(profile.current_density(radius, jnp.asarray(1.0 + middle))))
                / separatrix
            )
            if ratio >= floor:
                lower = middle
            else:
                upper = middle
        return 1.0 + 0.5 * (lower + upper)
