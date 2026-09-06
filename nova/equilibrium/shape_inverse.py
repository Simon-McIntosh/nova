"""Inverse shape control: a commanded bounding-box target to circuit currents.

The pulse-design app solves the inverse problem (:meth:`InverseDesign.
solve_current` in ``nova/equilibrium/inverse.py``): a Tikhonov-regularised
least squares over the free coil currents against the boundary-flux and field
targets at the control points, with the plasma column's own contribution moved
to the right-hand side.  This module poses the same step on the forward
operator's prescribed-current carrier, so a playable solve can drive the coil
currents straight toward a commanded shape and let the forward solve answer,
instead of compensating one constrained row at a time.

The rows are the bounding-box set the constraint module already reads: the
boundary flux at the four turning points (outer, upper, inner, lower), zero
radial field at the outer and inner points, zero vertical field at the upper
and lower points, and the two-gradient null at the commanded X-point.  Every
row is evaluated by the same lattice interpolation the constraint rows use
(``sample_lattice_flux`` and the field reads of ``FieldComponentConstraint``),
and its response to every drivable circuit current is the observation
Jacobian contracted with the operator's response carrier.  With the observed
values taken at the current plasma iterate, the linear system the
pseudo-inverse solves is ``response @ delta = target - observed`` — the plasma's
own contribution enters subtracted, exactly as the inverse design moves it to
the right-hand side.  Field rows are weighted by ``sqrt(field_weight)`` and
the Tikhonov ``gamma`` scales with the plasma current, so the two solve paths
agree dimensionally as the current waveform ramps.

An unmoved command — the achieved boundary's own turning points read back
through :meth:`BoundingBoxTarget.from_boundary` — reproduces the carrier's own
currents to the interpolation discretisation, and a moved command is achieved
by iterating the inverse step against the forward solve: solve currents from
the present iterate, solve the reduced route forward on those currents
warm-started from the previous frame, and re-inverse at the achieved plasma
when the turning-point error exceeds the stated tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import (
    BoundingBoxTarget,
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    FieldComponentConstraint,
    IsofluxConstraint,
    XPointConstraint,
    constraint_row_slices,
    sample_lattice_flux,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.observation import MomentIntegralSupport
from nova.linalg.regression import MoorePenrose

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardProfile

#: Field-row weight whose square root scales the field and flux rows onto
#: comparable residuals, the same factor the inverse design carries.
FIELD_WEIGHT = 50.0
#: Tikhonov factor at unit plasma current [per ampere]; the inverse step
#: regularises with ``GAMMA * |Ip|`` exactly as the inverse design does.
GAMMA = 1.0e-12

IsofluxReference = Literal["boundary", "reference_point"]


@dataclass(frozen=True)
class ShapeInverseResult:
    """The currents the inverse step commands and the system it solved."""

    currents: np.ndarray
    delta: np.ndarray
    free_circuits: np.ndarray
    response: np.ndarray
    observed: np.ndarray
    target: np.ndarray
    row_kinds: tuple[str, ...]
    plasma_current: float
    gamma: float
    field_weight: float
    least_squares_residual: float


def reference_point(profile: ForwardProfile, flux) -> np.ndarray:
    """Return the point whose flux stands in for the boundary level.

    A diverted equilibrium's read saddle is on the separatrix; a limited
    equilibrium has no saddle, so its wall point is the boundary anchor.  Both
    are read through the same interpolation the flux rows use, keeping every
    row's derivative exact rather than carrying the boundary-level tangent.
    The read saddle is only trusted on a resolved diverted class: on a limited
    machine the null search can return a finite lattice artifact inside the
    core, whose flux is far off the boundary.
    """
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    saddle = np.asarray(topology.x_point, dtype=float)
    if bool(np.asarray(topology.diverted)) and np.all(np.isfinite(saddle)):
        return saddle
    return np.asarray(topology.wall_point, dtype=float)


def boundary_polygon(profile: ForwardProfile, flux, *, angles: int = 181) -> np.ndarray:
    """Ray-cast the achieved boundary outward from the magnetic axis.

    Each ray from the axis is bisected onto the boundary-flux level with the
    same lattice interpolation the control rows read, so a receipt that
    ray-casts the achieved boundary measures the same surface the inverse step
    was driven onto.  A control point that has left the grid returns the cubic
    extension of the edge cell, which shows up downstream as a row that does
    not close rather than as a shape error.
    """
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    level = float(np.asarray(topology.boundary_flux))
    polarity = float(profile.operator.polarity)
    lattice = profile.lattice
    grid = jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)
    reach = 0.5 * min(
        float(lattice.radius[-1] - lattice.radius[0]),
        float(lattice.height[-1] - lattice.height[0]),
    )
    theta = 2.0 * np.pi * np.arange(angles) / angles
    points = []
    for angle in theta:
        ray = np.asarray([np.cos(angle), np.sin(angle)])
        low, high = 0.0, reach
        for _step in range(48):
            middle = 0.5 * (low + high)
            value = float(
                np.asarray(
                    sample_lattice_flux(lattice, grid, jnp.asarray(axis + middle * ray))
                )
            )
            if polarity * (value - level) > 0.0:
                low = middle
            else:
                high = middle
        points.append(axis + 0.5 * (low + high) * ray)
    return np.asarray(points)


def _turning_point_residual(
    lattice,
    grid: jax.Array,
    level: jax.Array,
    point: jax.Array,
    *,
    radial: bool,
) -> jax.Array:
    """Return the two conditions one turning-point extremum must meet.

    A radial turning point (outer or inner) is an extremum of the radius along
    the boundary, where the boundary tangent is vertical and therefore the
    vertical flux gradient — the radial field — vanishes; a vertical turning
    point (upper or lower) is an extremum of the height, where the radial flux
    gradient vanishes.  Both are ``(psi - level, one component of grad psi)``
    rooted through the same lattice interpolation the control rows read.
    """
    psi = sample_lattice_flux(lattice, grid, point) - level
    gradient = jax.grad(lambda position: sample_lattice_flux(lattice, grid, position))(
        point
    )
    return jnp.stack((psi, gradient[1] if radial else gradient[0]))


def _refine_turning_point(
    profile: ForwardProfile, grid: jax.Array, level: float, start, *, radial: bool
) -> np.ndarray:
    """Newton-refine a rough boundary extremum onto the exact turning point.

    The ray-cast extrema of a coarse contour sit a few millimetres off the
    true extrema of the flux surface, which leaves the field rows of a
    commanded target carrying a fictive field offset that the least squares
    then wastes current cancelling.  Rooting ``psi == level`` beside the
    matching flux-gradient component on the same interpolation the rows read
    nails the point to interpolation precision, so an unmoved command starts
    with every field row at zero.
    """
    point = jnp.asarray(start, dtype=jnp.float64)
    for _ in range(20):
        residual = _turning_point_residual(
            profile.lattice, grid, level, point, radial=radial
        )
        if float(np.asarray(jnp.linalg.norm(residual))) < 1.0e-12:
            break
        jacobian = jax.jacfwd(
            lambda position: _turning_point_residual(
                profile.lattice, grid, level, position, radial=radial
            )
        )(point)
        point = point + jnp.linalg.solve(jacobian, -residual)
    return np.asarray(point)


def achieved_target(profile: ForwardProfile, flux) -> BoundingBoxTarget:
    """Return the bounding-box target an achieved flux state already meets.

    This is the unmoved command: read the achieved boundary's four turning
    points as the exact extrema of its flux surface, anchor the flux rows on
    its own boundary point, and its field rows vanish where the boundary
    starts, so a solver that reproduces these rows leaves the plasma where it
    is.  A diverted equilibrium carries its read X-point here so the
    two-gradient null row joins the set; a limited equilibrium has no null
    row.
    """
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    level = float(np.asarray(topology.boundary_flux))
    lattice = profile.lattice
    grid = jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)
    poly = boundary_polygon(profile, flux)
    starts = (
        poly[int(np.argmax(poly[:, 0]))],
        poly[int(np.argmax(poly[:, 1]))],
        poly[int(np.argmin(poly[:, 0]))],
        poly[int(np.argmin(poly[:, 1]))],
    )
    outer = _refine_turning_point(profile, grid, level, starts[0], radial=True)
    upper = _refine_turning_point(profile, grid, level, starts[1], radial=False)
    inner = _refine_turning_point(profile, grid, level, starts[2], radial=True)
    lower = _refine_turning_point(profile, grid, level, starts[3], radial=False)
    saddle = np.asarray(topology.x_point, dtype=float)
    x_point = (
        saddle
        if bool(np.asarray(topology.diverted)) and np.all(np.isfinite(saddle))
        else None
    )
    return BoundingBoxTarget(
        flux_points=jnp.asarray(np.stack((outer, upper, inner, lower))),
        radial_field_points=jnp.asarray(np.stack((outer, inner))),
        vertical_field_points=jnp.asarray(np.stack((upper, lower))),
        x_point=x_point,
        # A refined turning point is on the same cubic-interpolated surface as
        # every other row. A limited topology's wall vertex is the physical
        # boundary anchor but its direct node value and the interpolated value
        # differ slightly, which would manufacture a current edit for an
        # otherwise unmoved command.
        reference_point=outer,
    )


def _row_scale(profile: ForwardProfile, span: float, kind: str) -> float:
    """Return the residual scale one bounding-box row kind is measured in."""
    if kind == "flux":
        return span
    if kind == "field":
        lattice = profile.lattice
        return span / (
            TOTAL_FLUX_FACTOR * float(lattice.radius[0]) * float(lattice.radial_step)
        )
    return span / float(profile.lattice.radial_step)


def bounding_box_pairs(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    *,
    span: float,
    reference: IsofluxReference = "reference_point",
    ampere_scale: float = 1.0e3,
) -> tuple[ConstraintPair, ...]:
    """Assemble the isoflux, field and X-point pairs one target produces.

    The pairs are shaped exactly like the shape-row receipts the constraint
    module already validates: one isoflux pair per turning point (the payload
    appends the reference point when the flux rows are anchored on one), one
    combined radial-plus-vertical field pair, and one two-gradient null pair
    when the target commands an X-point.  The unknowns are placeholders — the
    inverse step reads only the functional's observation rows and their
    response, never a compensating current.
    """
    flux_points = np.asarray(target.flux_points, dtype=float)
    radial = np.asarray(target.radial_field_points, dtype=float)
    vertical = np.asarray(target.vertical_field_points, dtype=float)
    x_point = np.asarray(target.x_point, dtype=float)
    circuits = int(profile.operator.prescribed_current_field.circuit_count)

    def pair(functional, points, kind):
        rows = functional.row_count
        scale = _row_scale(profile, span, kind)
        return ConstraintPair(
            functional=functional,
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, rows)).at[0].set(1.0),
                ampere_scale=jnp.full((rows,), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(rows),
                tolerance=jnp.full((rows,), 1.0e-6 * scale),
                scale=jnp.full((rows,), scale),
                initial_unknown=jnp.zeros(rows),
                payload=jnp.asarray(points, dtype=jnp.float64),
            ),
        )

    flux_payload = jnp.concatenate(
        (
            jnp.asarray(flux_points, dtype=jnp.float64),
            jnp.asarray(target.reference_point, dtype=jnp.float64)[None],
        )
    )
    pairs = [
        ConstraintPair(
            functional=IsofluxConstraint(
                point_count=flux_points.shape[0], reference=reference
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, flux_points.shape[0])).at[0].set(1.0),
                ampere_scale=jnp.full((flux_points.shape[0],), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(flux_points.shape[0]),
                tolerance=jnp.full(
                    (flux_points.shape[0],), 1.0e-6 * _row_scale(profile, span, "flux")
                ),
                scale=jnp.full((flux_points.shape[0],), span),
                initial_unknown=jnp.zeros(flux_points.shape[0]),
                payload=flux_payload,
            ),
        ),
        pair(
            FieldComponentConstraint(
                components=("radial",) * radial.shape[0]
                + ("vertical",) * vertical.shape[0]
            ),
            np.concatenate((radial, vertical)),
            "field",
        ),
    ]
    if x_point.shape == (2,):
        pairs.append(pair(XPointConstraint(), x_point[None, :], "xpoint"))
    return tuple(pairs)


def _flux_span(profile: ForwardProfile, flux) -> float:
    """Return the read flux span magnitude used to scale the row set."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    return abs(float(np.asarray(topology.flux_span)))


def observed_values(
    profile: ForwardProfile,
    pairs: Sequence[ConstraintPair],
    flux: jax.Array,
    *,
    requested_class=None,
    target_current=None,
) -> np.ndarray:
    """Return every registered observation at one flux state, concatenated."""
    from nova.equilibrium.constraint import ConstraintContext

    state = jnp.ravel(jnp.asarray(flux))
    context = ConstraintContext(state, requested_class, target_current, None)
    rows = []
    for pair in pairs:
        rows.append(
            jnp.atleast_1d(
                pair.functional.observed(profile, context, pair.binding.payload)
            )
        )
    return np.asarray(jnp.concatenate(rows))


def response_matrix(
    profile: ForwardProfile,
    pairs: Sequence[ConstraintPair],
    flux: jax.Array,
    *,
    requested_class=None,
    target_current=None,
) -> np.ndarray:
    """Return each observed row's sensitivity to every circuit current.

    The prescribed circuits enter the flux state linearly through the
    operator's response carrier, so one reverse-mode pass per row contracted
    with that carrier is the whole matrix — the exact read the shape-row
    receipts use, which is what keeps this module's rows and the constraint
    module's rows on one surface.
    """
    response = profile.constraint_response_matrix(
        pairs,
        jnp.asarray(flux),
        requested_class=requested_class,
        target_current=target_current,
    )
    return np.asarray(response)


def plasma_current(profile: ForwardProfile, flux, *, target_current=None) -> float:
    """Return the plasma current the Tikhonov scale keys to."""
    if target_current is not None:
        return abs(float(np.asarray(target_current)))
    observation = profile.current_moment_observation(
        jnp.asarray(flux), support=MomentIntegralSupport.ALL_DOMAIN
    )
    return abs(float(np.asarray(observation.plasma_current)))


def solve_shape_inverse(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    prescribed_current=None,
    requested_class=None,
    target_current=None,
    free_circuits: Sequence[int] | None = None,
    reference: IsofluxReference = "reference_point",
    gamma: float = GAMMA,
    field_weight: float = FIELD_WEIGHT,
) -> ShapeInverseResult:
    """Return the prescribed currents that drive the flux map to one target.

    The plasma's own contribution is the observed value at the supplied flux
    state; subtracting it from the target leaves the current delta the
    response matrix must supply.  Free circuits default to every prescribed
    circuit; a caller that must not drive passive structure passes the
    drivable indices explicitly.  Field rows are weighted by
    ``sqrt(field_weight)`` and the Tikhonov factor is ``gamma * |Ip|``, the
    exact scaling the pulse-design inverse step carries.
    """
    state = jnp.asarray(flux)
    span = _flux_span(profile, state)
    pairs = bounding_box_pairs(profile, target, span=span, reference=reference)
    observed = observed_values(
        profile,
        pairs,
        state,
        requested_class=requested_class,
        target_current=target_current,
    )
    response = response_matrix(
        profile,
        pairs,
        state,
        requested_class=requested_class,
        target_current=target_current,
    )
    if free_circuits is None:
        free = np.arange(response.shape[1])
    else:
        free = np.unique(np.asarray(free_circuits, dtype=int))
    if free.size == 0:
        raise ValueError("the shape-inverse step needs at least one free circuit")

    # Field rows sit in every pair after the flux rows; weight exactly those
    # so the field and flux residuals trade off on comparable terms.
    slices = constraint_row_slices(pairs)
    field_weight_rows = []
    for row_slice in slices[1:]:
        field_weight_rows.extend(range(row_slice.start, row_slice.stop))
    weight = np.ones(observed.shape[0])
    weight[field_weight_rows] = np.sqrt(field_weight)
    weighted = response[:, free] * weight[:, None]
    rhs = (np.zeros(observed.shape[0]) - observed) * weight

    ip = plasma_current(profile, state, target_current=target_current)
    regularisation = gamma * ip
    delta_free = MoorePenrose(weighted, gamma=regularisation) / rhs

    if prescribed_current is None:
        current = np.zeros(response.shape[1])
    else:
        current = np.asarray(prescribed_current, dtype=float).copy()
        if current.shape != (response.shape[1],):
            raise ValueError(
                "prescribed_current must match the response column count "
                f"{response.shape[1]}"
            )
    current[free] = current[free] + delta_free

    row_kinds = tuple(
        kind
        for _pair, row_slice, kind in zip(
            pairs, slices, ("flux", "field", "xpoint"), strict=False
        )
        for _ in range(row_slice.stop - row_slice.start)
    )
    return ShapeInverseResult(
        currents=current,
        delta=delta_free,
        free_circuits=free,
        response=response,
        observed=observed,
        target=np.zeros(observed.shape[0]),
        row_kinds=row_kinds,
        plasma_current=float(ip),
        gamma=float(regularisation),
        field_weight=field_weight,
        least_squares_residual=float(np.linalg.norm(weighted @ delta_free - rhs)),
    )


def turning_point_error(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
) -> float:
    """Return the largest distance from a commanded point to the achieved one.

    Each of the four principal turning points the target commands is compared
    with the achieved boundary's own extrema, read through the same ray-cast
    interpolation the target was built from, so the error measures boundary
    motion in metres rather than row residuals.
    """
    achieved = np.asarray(achieved_target(profile, jnp.asarray(flux)).flux_points)
    commanded = np.asarray(target.flux_points, dtype=float)
    return float(np.max(np.linalg.norm(achieved - commanded, axis=1)))


__all__ = [
    "FIELD_WEIGHT",
    "GAMMA",
    "ShapeInverseResult",
    "achieved_target",
    "boundary_polygon",
    "bounding_box_pairs",
    "observed_values",
    "plasma_current",
    "reference_point",
    "response_matrix",
    "solve_shape_inverse",
    "turning_point_error",
]
