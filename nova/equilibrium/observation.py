r"""Convention-pinned integral observations of a converged equilibrium.

An absolute source state predicts its own integral quantities; it does not
receive them. This module is therefore an observation operator: it maps a
converged flux map and the source that produced it to plasma current,
poloidal beta and internal inductance, and to the residuals of any targets a
caller wants to validate against. It never changes a profile.

All three follow from volume integrals over the labelled core, so none of
them needs a traced boundary contour and all of them differentiate cleanly:

.. math::
    I_p = \sum_{\mathrm{core}} j_\phi\, \Delta A, \qquad
    \mathrm{d}V = 2 \pi R\, \Delta A, \qquad
    R_{\mathrm{ax}} = \frac{1}{V} \int_{\mathrm{core}} R\, \mathrm{d}V,

.. math::
    \beta_p = \frac{4 \int p\, \mathrm{d}V}{\mu_0 R_{\mathrm{ax}} I_p^2},
    \qquad
    l_i = \frac{2 \int B_p^2\, \mathrm{d}V}{\mu_0^2 R_{\mathrm{ax}} I_p^2}.

The reference radius is pinned as the volume-averaged major radius of the
labelled core rather than a machine constant or a boundary extremum, so it
moves with the solved geometry and stays smooth in the flux map. Pressure is
asked of the declared closure at the node radius and the converged flux span
— a static closure integrates it inward from the boundary primitive by the
rule pinned in :mod:`nova.equilibrium.convention`, a rotating one returns its
own major-radius-dependent primitive — so the observation is a property of
the solve and not of a frozen grid.

All three are read on the labelled CORE, whatever else the source declares.
A continuation beyond the separatrix adds current on the open branches, and
that current is published — per domain, and as a total — by
:class:`CurrentLedger`; folding it into :math:`I_p` would silently change the
denominator both :math:`\beta_p` and :math:`l_i` are defined against and make
two solves with different scrape-off policies incomparable in quantities that
are supposed to describe the confined plasma. The ledger is where the open
current is read.

Enforcement is separate from observation. A caller may ask the solve to
enforce moments only up to the number of scalar degrees of freedom the
declared closure carries; an absolute source carries none, so any enforcement
request fails through :class:`MomentEnforcementError` before the source is
evaluated and without touching the profiles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from scipy.constants import mu_0

from nova.equilibrium.domain import DomainMasks, PlasmaDomain

__all__ = [
    "MOMENT_NAMES",
    "CurrentLedger",
    "IntegralObservation",
    "MomentEnforcementError",
    "MomentTargets",
    "declared_body_force",
    "declared_field_function_squared",
    "declared_pressure",
    "gradient_tail",
    "moment_residual",
    "observe_moments",
    "reject_unsupported_enforcement",
]

#: Observation names in residual-vector column order.
MOMENT_NAMES: tuple[str, ...] = (
    "plasma_current",
    "poloidal_beta",
    "internal_inductance",
)

#: Nodes the flux-function gradients are integrated on to recover primitives.
PROFILE_NODES = 257


class MomentEnforcementError(ValueError):
    """Raised when more moments are enforced than the closure can solve."""


@dataclass(frozen=True)
class MomentTargets:
    """Optional validation targets for the integral observations."""

    plasma_current: float | None = None
    poloidal_beta: float | None = None
    internal_inductance: float | None = None

    @property
    def requested(self) -> tuple[str, ...]:
        """Return the declared target names in column order."""
        return tuple(name for name in MOMENT_NAMES if getattr(self, name) is not None)


class CurrentLedger(NamedTuple):
    """Toroidal current [A] integrated over each domain label."""

    core: jax.Array
    common_sol: jax.Array
    private_flux: jax.Array
    excluded_material: jax.Array
    total: jax.Array


class IntegralObservation(NamedTuple):
    """Integral quantities observed from one converged equilibrium."""

    plasma_current: jax.Array
    poloidal_beta: jax.Array
    internal_inductance: jax.Array
    volume: jax.Array
    major_radius: jax.Array
    pressure_integral: jax.Array
    poloidal_field_integral: jax.Array
    flux_span: jax.Array

    def value(self, name: str) -> jax.Array:
        """Return one named observation."""
        if name not in MOMENT_NAMES:
            raise KeyError(f"unknown integral observation {name!r}")
        return getattr(self, name)

    def stack(self) -> jax.Array:
        """Return the observations in residual-vector column order."""
        return jnp.stack([getattr(self, name) for name in MOMENT_NAMES])


def gradient_tail(gradient, psi_norm: jax.Array, nodes: int = PROFILE_NODES):
    """Return the boundary-inward integral of one flux-function gradient.

    The tail is :math:`\\int_{\\psi_N}^{1} g(s)\\,\\mathrm{d}s` on a fixed
    trapezoidal node set, so the primitive recovered from it has the same
    fixed shape for every slice and differentiates through the solve.
    """
    grid = jnp.linspace(0.0, 1.0, nodes)
    values = jnp.asarray(gradient(grid))
    step = grid[1] - grid[0]
    segment = 0.5 * step * (values[1:] + values[:-1])
    tail = jnp.concatenate(
        [jnp.cumsum(segment[::-1])[::-1], jnp.zeros(1, dtype=segment.dtype)]
    )
    return jnp.interp(psi_norm, grid, tail)


def core_pressure(
    source, masks: DomainMasks, radius: jax.Array, flux_span: jax.Array
) -> jax.Array:
    """Return the pressure [Pa] the core closure implies on every cell.

    The major radius is passed because pressure is a flux function only under
    a static closure; a rotation closure varies it along a surface and owns
    its own primitive.
    """
    return source.core.pressure(
        radius, masks.psi_norm, source.boundary_pressure, flux_span
    )


def core_field_function_squared(
    source, masks: DomainMasks, flux_span: jax.Array
) -> jax.Array:
    """Return the squared toroidal-field function [T^2 m^2] on every cell."""
    return source.core.field_function_squared(
        masks.psi_norm, source.boundary_field_function, flux_span
    )


def _select_open(source, masks: DomainMasks, core_value: jax.Array, evaluate):
    """Return one cell field with each declared open closure's own value.

    The core value fills the map and a declared continuation overwrites its
    own domain, so a solve that declares nothing beyond the core reaches the
    identical expression it always did. The evaluation point is held on the
    open domain the same way the current image holds it, at the separatrix,
    which is inside the support of either branch.
    """
    value = core_value
    for domain, profile in source.open_profiles:
        selection = masks.mask(domain)
        value = jnp.where(
            selection,
            evaluate(profile, jnp.where(selection, masks.psi_norm, 1.0)),
            value,
        )
    return value


def declared_pressure(
    source, masks: DomainMasks, radius: jax.Array, flux_span: jax.Array
) -> jax.Array:
    """Return the pressure [Pa] every declared closure implies on its domain."""
    return _select_open(
        source,
        masks,
        core_pressure(source, masks, radius, flux_span),
        lambda profile, psi_norm: profile.pressure(
            radius, psi_norm, source.boundary_pressure, flux_span
        ),
    )


def declared_field_function_squared(
    source, masks: DomainMasks, flux_span: jax.Array
) -> jax.Array:
    """Return the squared toroidal-field function [T^2 m^2] of every closure."""
    return _select_open(
        source,
        masks,
        core_field_function_squared(source, masks, flux_span),
        lambda profile, psi_norm: profile.field_function_squared(
            psi_norm, source.boundary_field_function, flux_span
        ),
    )


def declared_body_force(
    source, masks: DomainMasks, radius: jax.Array, pressure: jax.Array
) -> jax.Array:
    """Return the non-magnetic radial force density [N/m^3] of every closure."""
    return _select_open(
        source,
        masks,
        source.core.radial_body_force(radius, masks.psi_norm, pressure),
        lambda profile, psi_norm: profile.radial_body_force(radius, psi_norm, pressure),
    )


def current_ledger(cell_current: jax.Array, masks: DomainMasks) -> CurrentLedger:
    """Return the toroidal current integrated over each domain label."""
    per_domain = [
        jnp.sum(jnp.where(masks.mask(domain), cell_current, 0.0))
        for domain in PlasmaDomain
    ]
    return CurrentLedger(
        core=per_domain[PlasmaDomain.CORE],
        common_sol=per_domain[PlasmaDomain.COMMON_SOL],
        private_flux=per_domain[PlasmaDomain.PRIVATE_FLUX],
        excluded_material=per_domain[PlasmaDomain.EXCLUDED_MATERIAL],
        total=jnp.sum(cell_current),
    )


def observe_moments(
    source,
    masks: DomainMasks,
    radius: jax.Array,
    area: jax.Array,
    cell_current: jax.Array,
    poloidal_field_squared: jax.Array,
    flux_span: jax.Array,
) -> IntegralObservation:
    """Return the integral observations of one converged equilibrium."""
    volume_element = jnp.where(masks.core, 2.0 * jnp.pi * radius * area, 0.0)
    volume = jnp.sum(volume_element)
    safe_volume = jnp.where(volume > 0.0, volume, 1.0)
    plasma_current = jnp.sum(jnp.where(masks.core, cell_current, 0.0))

    major_radius = jnp.sum(radius * volume_element) / safe_volume
    # the closure is declared on the core and need not evaluate anywhere else,
    # so the pressure is selected by label rather than weighted by an area
    # that vanishes off it: a zero weight does not neutralise a non-finite
    # value, and one such cell would carry the whole integral away
    pressure_integral = jnp.sum(
        jnp.where(masks.core, core_pressure(source, masks, radius, flux_span), 0.0)
        * volume_element
    )
    field_integral = jnp.sum(poloidal_field_squared * volume_element)

    reference = mu_0 * major_radius * plasma_current**2
    safe_reference = jnp.where(
        jnp.abs(reference) > 0.0, reference, jnp.ones_like(reference)
    )
    return IntegralObservation(
        plasma_current=plasma_current,
        poloidal_beta=4.0 * pressure_integral / safe_reference,
        internal_inductance=2.0 * field_integral / (mu_0 * safe_reference),
        volume=volume,
        major_radius=major_radius,
        pressure_integral=pressure_integral,
        poloidal_field_integral=field_integral,
        flux_span=flux_span,
    )


def moment_residual(
    observation: IntegralObservation, targets: MomentTargets
) -> jax.Array:
    """Return the scale-normalised target residuals in column order.

    An observation with no declared target contributes an exact zero, so the
    residual vector keeps one fixed shape whatever the caller asked for.
    """
    residual = []
    for name in MOMENT_NAMES:
        target = getattr(targets, name)
        observed = observation.value(name)
        if target is None:
            residual.append(jnp.zeros_like(observed))
            continue
        scale = max(abs(float(target)), 1.0e-30)
        residual.append((observed - target) / scale)
    return jnp.stack(residual)


def reject_unsupported_enforcement(
    enforce: Sequence[str], closure_degrees: int
) -> tuple[str, ...]:
    """Validate an enforcement request before any source is evaluated.

    The request is checked against the declared closure's scalar degrees of
    freedom, so an over-determined request fails while the supplied profiles
    are still untouched.
    """
    requested = tuple(enforce)
    unknown = [name for name in requested if name not in MOMENT_NAMES]
    if unknown:
        raise MomentEnforcementError(
            f"unknown integral observation(s) {', '.join(sorted(unknown))}; "
            f"available: {', '.join(MOMENT_NAMES)}"
        )
    if len(set(requested)) != len(requested):
        raise MomentEnforcementError("an integral observation was enforced twice")
    if len(requested) > closure_degrees:
        raise MomentEnforcementError(
            f"enforcing {len(requested)} moment(s) "
            f"({', '.join(requested)}) needs {len(requested)} scalar closure "
            f"degrees of freedom; the declared source carries "
            f"{closure_degrees}. Reshaping profiles to hit a moment is a "
            "conditioning or reconstruction problem, not a forward solve"
        )
    return requested
