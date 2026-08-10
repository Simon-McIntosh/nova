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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium.convention import toroidal_current_density
from nova.equilibrium.domain import DomainMasks

__all__ = [
    "DomainProfile",
    "ForwardSource",
    "NormalisationPolicy",
    "NormalisationRecord",
]


class NormalisationPolicy(IntEnum):
    """How the supplied source amplitude is fixed."""

    ABSOLUTE = 0
    DECLARED_SCALAR_CURRENT = 1


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

    def current_density(self, radius: jax.Array, psi_norm: jax.Array) -> jax.Array:
        """Return the toroidal current density [A/m^2] this closure drives."""
        return toroidal_current_density(
            radius, self.p_prime(psi_norm), self.ff_prime(psi_norm)
        )


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

    def cell_current(
        self, radius: jax.Array, area: jax.Array, masks: DomainMasks
    ) -> jax.Array:
        """Return the per-cell toroidal current [A] on the declared support.

        The current density is evaluated from the supplied gradients without
        any amplitude change and is then selected by domain label, so a cell
        outside the declared support carries exactly zero.
        """
        density = self.core.current_density(radius, masks.psi_norm)
        return jnp.where(masks.core, density * area, 0.0)
