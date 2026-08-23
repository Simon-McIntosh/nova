r"""Convention-pinned integral observations of a converged equilibrium.

An absolute source state predicts its own integral quantities; it does not
receive them. This module is therefore an observation operator: it maps a
converged flux map and the source that produced it to plasma current,
poloidal beta and internal inductance, and to the residuals of any targets a
caller wants to validate against. It never changes a profile.

All three follow from volume integrals over the traced core support.  The
same fixed-capacity clip partition that attributes current supplies the
integration domain, so boundary cells contribute their exact clipped polygon
and topology-zero cells contribute exactly zero:

.. math::
    I_p = \sum_c \int_{A_c\cap\mathrm{core}} j_\phi\, \mathrm{d}A, \qquad
    \mathrm{d}V = 2 \pi R\, \Delta A, \qquad
    R_{\mathrm{ax}} = \frac{1}{V} \int_{\mathrm{core}} R\, \mathrm{d}V,

.. math::
    \beta_p = \frac{4 \int p\, \mathrm{d}V}{\mu_0 R_{\mathrm{ax}} I_p^2},
    \qquad
    l_i = \frac{2 \int B_p^2\, \mathrm{d}V}{\mu_0^2 R_{\mathrm{ax}} I_p^2}.

The forward-moment reference radius is pinned as the volume-averaged major
radius of the clipped core rather than a machine constant or a boundary
extremum, so it moves with the solved geometry and stays smooth in the flux
map.  Its internal-inductance observation uses the same current-radius algebra
as ``li_3``, but its explicitly reported ``major_radius`` remains this
differentiable volume average.  It is therefore a forward-moment convention,
not the IMAS egress value unless that radius coincides with the LCFS geometric
radius.  The pulse-design egress reads the LCFS geometric-radius value from
:class:`nova.biot.plasma.Plasma`. Pressure is
asked of the declared closure at the node radius and the converged flux span
— a static closure integrates it inward from the boundary primitive by the
rule pinned in :mod:`nova.equilibrium.convention`, a rotating one returns its
own major-radius-dependent primitive — so the observation is a property of
the solve and not of a frozen grid.

All three are read on the topology-qualified core support, whatever else the
source declares.
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
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.domain import DomainMasks, PlasmaDomain

__all__ = [
    "MOMENT_NAMES",
    "CurrentLedger",
    "ClippedIntegralMeasure",
    "IntegralObservation",
    "ConstraintPinSet",
    "ConstraintViolationError",
    "CurrentMomentObservation",
    "IsofluxPin",
    "MomentIntegralSupport",
    "MomentPin",
    "MomentEnforcementError",
    "MomentTargets",
    "PinUncertainty",
    "declared_body_force",
    "declared_field_function_squared",
    "declared_pressure",
    "gradient_tail",
    "clipped_support_quadrature",
    "moment_residual",
    "observe_current_moments",
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

# Eight Gauss nodes in each Duffy coordinate integrate polynomial content
# through degree fifteen in either coordinate and leave ample headroom for the
# smooth pressure closure and the 1/R field factor.
_GAUSS_NODE, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(8)
_UNIT_NODE = 0.5 * (_GAUSS_NODE + 1.0)
_UNIT_WEIGHT = 0.5 * _GAUSS_WEIGHT


class MomentEnforcementError(ValueError):
    """Raised when more moments are enforced than the closure can solve."""


class ConstraintViolationError(ValueError):
    """Raised when a solved state lies outside a trusted pin interval."""


class MomentIntegralSupport(str, Enum):
    """Current integration domain carried by a deterministic moment pin."""

    ALL_DOMAIN = "all_domain"
    CONFINED_CORE = "confined_core"


@dataclass(frozen=True)
class PinUncertainty:
    """Absolute deterministic interval attached to one trusted pin.

    The interval is metadata and an acceptance scale.  It is not interpreted
    as a probability distribution or used to form a likelihood.
    """

    absolute: float
    unit: str
    statement: str

    def __post_init__(self) -> None:
        """Require a finite positive interval and its provenance statement."""
        if not np.isfinite(self.absolute) or self.absolute <= 0.0:
            raise ValueError("pin uncertainty must be positive and finite")
        if not self.unit.strip():
            raise ValueError("pin uncertainty must declare a unit")
        if not self.statement.strip():
            raise ValueError("pin uncertainty must declare its basis")


@dataclass(frozen=True)
class IsofluxPin:
    """Two R-Z points trusted to share one normalised-flux value."""

    first_coordinate: tuple[float, float]
    second_coordinate: tuple[float, float]
    psi_norm: float
    uncertainty: PinUncertainty

    def __post_init__(self) -> None:
        """Require finite physical coordinates and a finite target."""
        first = np.asarray(self.first_coordinate, dtype=np.float64)
        second = np.asarray(self.second_coordinate, dtype=np.float64)
        if first.shape != (2,) or second.shape != (2,):
            raise ValueError("isoflux coordinates must be R-Z pairs")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError("isoflux coordinates must be finite")
        if first[0] <= 0.0 or second[0] <= 0.0:
            raise ValueError("isoflux major radii must be positive")
        if not np.isfinite(self.psi_norm):
            raise ValueError("isoflux target must be finite")


@dataclass(frozen=True)
class MomentPin:
    """One current amplitude or centroid pin with explicit support."""

    name: str
    target: float
    uncertainty: PinUncertainty
    support: MomentIntegralSupport

    def __post_init__(self) -> None:
        """Require a supported moment, finite target, and typed support."""
        if self.name not in ("plasma_current", "centroid_r", "centroid_z"):
            raise ValueError(f"unknown current moment {self.name!r}")
        if not np.isfinite(self.target):
            raise ValueError("moment target must be finite")
        if not isinstance(self.support, MomentIntegralSupport):
            raise TypeError("moment support must be a MomentIntegralSupport")


@dataclass(frozen=True)
class ConstraintPinSet:
    """Trusted deterministic pins supplied by an upstream evidence owner."""

    isoflux: tuple[IsofluxPin, ...] = ()
    moments: tuple[MomentPin, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one typed pin and reject duplicate moment claims."""
        if not self.isoflux and not self.moments:
            raise ValueError("a constraint pin set must contain at least one pin")
        if not all(isinstance(pin, IsofluxPin) for pin in self.isoflux):
            raise TypeError("isoflux pins must be IsofluxPin values")
        if not all(isinstance(pin, MomentPin) for pin in self.moments):
            raise TypeError("moment pins must be MomentPin values")
        keys = [(pin.name, pin.support) for pin in self.moments]
        if len(keys) != len(set(keys)):
            raise ValueError("a current moment was pinned twice on one support")


class CurrentMomentObservation(NamedTuple):
    """Net toroidal current and current centroid on one declared support."""

    plasma_current: jax.Array
    centroid_r: jax.Array
    centroid_z: jax.Array
    support: MomentIntegralSupport

    def stack(self) -> jax.Array:
        """Return amplitude and centroid in the public map's fixed order."""
        return jnp.stack((self.plasma_current, self.centroid_r, self.centroid_z))

    def value(self, name: str) -> jax.Array:
        """Return one named current moment."""
        if name not in ("plasma_current", "centroid_r", "centroid_z"):
            raise KeyError(f"unknown current moment {name!r}")
        return getattr(self, name)


def observe_current_moments(
    cell_current,
    coordinate,
    *,
    core_mask,
    support: MomentIntegralSupport,
) -> CurrentMomentObservation:
    """Integrate current amplitude and centroid on the declared domain.

    ``ALL_DOMAIN`` consumes every authored plasma cell. ``CONFINED_CORE``
    consumes only the topology-qualified core mask. The support therefore
    stays visible beside every returned moment and cannot be silently mixed.
    """
    current = jnp.asarray(cell_current)
    point = jnp.asarray(coordinate)
    if current.ndim != 1 or point.shape != (current.size, 2):
        raise ValueError("cell current and R-Z coordinates must align")
    if support is MomentIntegralSupport.CONFINED_CORE:
        selected = jnp.asarray(core_mask, dtype=bool)
        current = jnp.where(selected, current, 0.0)
    elif support is not MomentIntegralSupport.ALL_DOMAIN:
        raise TypeError("support must be a MomentIntegralSupport")
    total = jnp.sum(current)
    safe_total = jnp.where(jnp.abs(total) > 0.0, total, 1.0)
    centroid = jnp.sum(current[:, None] * point, axis=0) / safe_total
    return CurrentMomentObservation(total, centroid[0], centroid[1], support)


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


class ClippedIntegralMeasure(NamedTuple):
    """Per-cell integrals over the topology-qualified clipped core support.

    ``area`` is the exact polygon area receipt.  The volume, radial-volume,
    pressure and field entries are fixed high-order quadratures over that same
    polygon.  ``cell_current`` is the closed-form integral of the production
    clip-independent current density, including its in-cell variation.

    Measure survey for the equilibrium path:

    * volume and major radius sum ``volume`` and ``radial_volume``;
    * plasma current sums ``cell_current``;
    * pressure integral and poloidal beta sum ``pressure_volume``;
    * field integral and internal inductance sum ``field_volume``;
    * :class:`CurrentLedger` consumes already-integrated cell current but keeps
      domain labels because it is a receipt taxonomy, not a geometric integral;
    * conservation residuals remain centre-point finite-difference norms whose
      eroded stencil domain is part of their discretisation, not a volume
      observation;
    * flux-surface geometry uses independently traced contour line integrals,
      while flux-surface connectivity uses a multi-level coarea estimator.
      Neither has the single separatrix support represented here.
    * the structured-lattice compatibility constructor has no direct pre-clip
      sample targets and therefore cannot form the own-node polynomial.  It
      retains its labelled finite-volume measure; the default hex construction
      supplies direct samples and always takes the clipped route.
    """

    area: jax.Array
    volume: jax.Array
    radial_volume: jax.Array
    cell_current: jax.Array
    pressure_volume: jax.Array
    field_volume: jax.Array
    masks: DomainMasks

    def with_current_amplitude(self, amplitude: jax.Array) -> ClippedIntegralMeasure:
        """Return this measure with its exact cell-current integral rescaled."""
        return self._replace(cell_current=amplitude * self.cell_current)


def clipped_support_quadrature(support, selection):
    """Return fixed-shape degree-fifteen Duffy quadrature on each support.

    Convex clipped cells are fanned from their first vertex.  Padding and
    topology qualification enter only through zero weights; points on dead
    triangles are moved to the authored cell centroid so discarded closure
    evaluations remain finite under differentiation.
    """
    vertices = jnp.asarray(support.support_vertices)
    count = jnp.asarray(support.vertex_count)
    selected = jnp.asarray(selection, dtype=bool)
    capacity = vertices.shape[1]
    triangle_slot = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:, :1], (len(vertices), capacity - 2, 2))
    second = vertices[:, triangle_slot]
    third = vertices[:, triangle_slot + 1]
    radial = jnp.asarray(_UNIT_NODE, dtype=vertices.dtype)
    vertical = jnp.asarray(_UNIT_NODE, dtype=vertices.dtype)
    radial_weight = jnp.asarray(_UNIT_WEIGHT, dtype=vertices.dtype)
    vertical_weight = jnp.asarray(_UNIT_WEIGHT, dtype=vertices.dtype)
    u, v = jnp.meshgrid(radial, vertical, indexing="ij")
    wu, wv = jnp.meshgrid(radial_weight, vertical_weight, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    rule_weight = (wu * wv).reshape(-1)
    edge_first = second - first
    edge_second = third - first
    points = (
        first[:, :, None, :]
        + u[None, None, :, None] * edge_first[:, :, None, :]
        + (1.0 - u)[None, None, :, None]
        * v[None, None, :, None]
        * edge_second[:, :, None, :]
    )
    cross = jnp.abs(
        edge_first[..., 0] * edge_second[..., 1]
        - edge_first[..., 1] * edge_second[..., 0]
    )
    live = (triangle_slot[None, :] + 1 < count[:, None]) & selected[:, None]
    weights = cross[:, :, None] * (1.0 - u)[None, None, :] * rule_weight[None, None, :]
    weights = jnp.where(live[:, :, None], weights, 0.0)
    points = points.reshape(len(vertices), -1, 2)
    weights = weights.reshape(len(vertices), -1)
    points = jnp.where(
        (weights > 0.0)[..., None], points, jnp.asarray(support.centroids)[:, None, :]
    )
    return points, weights


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
    support_integrals: ClippedIntegralMeasure,
    flux_span: jax.Array,
) -> IntegralObservation:
    """Return integral observations from one clipped-support measure."""
    volume = jnp.sum(support_integrals.volume)
    safe_volume = jnp.where(volume > 0.0, volume, 1.0)
    plasma_current = jnp.sum(support_integrals.cell_current)
    major_radius = jnp.sum(support_integrals.radial_volume) / safe_volume
    pressure_integral = jnp.sum(support_integrals.pressure_volume)
    field_integral = jnp.sum(support_integrals.field_volume)

    beta_reference = mu_0 * major_radius * plasma_current**2
    safe_beta_reference = jnp.where(
        jnp.abs(beta_reference) > 0.0,
        beta_reference,
        jnp.ones_like(beta_reference),
    )
    inductance_normaliser = 0.5 * mu_0**2 * plasma_current**2 * major_radius
    safe_inductance_normaliser = jnp.where(
        jnp.abs(inductance_normaliser) > 0.0,
        inductance_normaliser,
        jnp.ones_like(inductance_normaliser),
    )
    return IntegralObservation(
        plasma_current=plasma_current,
        poloidal_beta=4.0 * pressure_integral / safe_beta_reference,
        internal_inductance=field_integral / safe_inductance_normaliser,
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
