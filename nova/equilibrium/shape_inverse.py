"""Inverse shape control: a commanded bounding-box target to circuit currents.

The pulse-design app solves the inverse problem (:meth:`InverseDesign.
solve_current` in ``nova/equilibrium/inverse.py``): a Tikhonov-regularised
least squares over the free coil currents against the boundary-flux and field
targets at the control points, with the plasma column's own contribution moved
to the right-hand side.  This module poses the same step on the forward
operator's prescribed-current carrier, so a playable solve can drive the coil
currents straight toward a commanded shape and let the forward solve answer,
instead of compensating one constrained row at a time.

The rows are the full boundary polygon continuously deformed by the commanded
bounding box, with its four turning points retained explicitly; zero radial
field at the outer and inner points; zero vertical field at the upper and
lower points; and both field components at the commanded X-point. Every row
is evaluated by the same lattice interpolation the constraint rows use
(``sample_lattice_flux`` and the field reads of ``FieldComponentConstraint``),
and its response to every drivable circuit current is the observation
Jacobian contracted with the operator's response carrier. The unknowns are
free-circuit current changes about the equilibrium's original seed currents.
The fixed-conductor, plasma and seed-current contributions are moved to the
right-hand side. Each row is weighted by the reciprocal of its seed-level
consistency floor, field rows retain the ``sqrt(field_weight)`` priority, and
the Tikhonov ``gamma`` is scaled by the plasma current.

Proposal-only calls use the placement map to select a current vector without
claiming that the resulting nonlinear equilibrium has achieved the command.
When a nonlinear forward referee is supplied, every admitted result is measured
through its achieved turning points. A bounded outer Newton loop refreshes the
local current-to-turning-point tangent at that state, corrects the physical
shape residual, and sends every correction through the same axis-admission
line search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
    sample_lattice_flux,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import NoQualifiedAxisError
from nova.linalg.regression import MoorePenrose

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardProfile

#: Field-row weight whose square root scales the field and flux rows onto
#: comparable residuals, the same factor the inverse design carries.
FIELD_WEIGHT = 50.0
#: Tikhonov factor multiplied by the absolute plasma current [A].
GAMMA = 1.0e-12
#: Number of plasma-placement updates before the final current solve.
PICARD_ROUNDS = 3
#: Relative physical turning-point error accepted by the achieved-shape loop.
TURNING_POINT_RELATIVE_TOLERANCE = 0.1
#: Symmetric current perturbation used for the local turning-point tangent [A].
TURNING_POINT_TANGENT_STEP_A = 100.0

IsofluxReference = Literal["boundary", "reference_point"]


class NoAdmissibleShapeStepError(ValueError):
    """Every nonzero fraction of a proposed shape-current step was refused."""

    def __init__(
        self,
        refusal_sequence: Sequence[float],
        proposed_delta: np.ndarray,
    ) -> None:
        self.refusal_sequence = tuple(float(value) for value in refusal_sequence)
        self.proposed_delta = np.asarray(proposed_delta, dtype=float).copy()
        super().__init__("no axis-admissible current fraction remains")


@dataclass(frozen=True)
class ShapeInverseIteration:
    """One proposed current correction and its admitted achieved shape."""

    commanded_turning_points: np.ndarray
    achieved_turning_points: np.ndarray
    turning_point_residual: np.ndarray
    turning_point_residual_norm: float
    tangent_singular_values: np.ndarray
    tangent_numerical_rank: int
    proposed_delta: np.ndarray
    admitted_delta: np.ndarray
    accepted_fraction: float
    admissibility_trials: int


@dataclass(frozen=True)
class ShapeInverseResult:
    """The currents the inverse step commands and the system it solved."""

    currents: np.ndarray
    delta: np.ndarray
    uncapped_delta: np.ndarray
    free_circuits: np.ndarray
    response: np.ndarray
    observed: np.ndarray
    target: np.ndarray
    right_hand_side: np.ndarray
    linear_prediction: np.ndarray
    row_kinds: tuple[str, ...]
    plasma_current: float
    gamma: float
    delta_regularisation: float
    delta_current_scale: np.ndarray
    field_weight: float
    singular_values: np.ndarray
    numerical_rank: int
    right_null_space: np.ndarray
    picard_currents: np.ndarray
    picard_boundary_flux: np.ndarray
    current_step_fraction: float | None
    current_step_limited: bool
    accepted_fraction: float
    admissibility_trials: int
    least_squares_residual: float
    uncapped_least_squares_residual: float
    flux_points: np.ndarray
    previous_flux_points: np.ndarray
    consistency_floor: np.ndarray
    row_weight: np.ndarray
    iterations: tuple[ShapeInverseIteration, ...]
    achieved_flux_points: np.ndarray
    turning_point_residual: np.ndarray
    turning_point_residual_norm: float
    turning_point_tolerance: float
    converged: bool


def _cap_current_delta(
    delta: np.ndarray,
    reference_current: np.ndarray,
    fraction: float | None,
) -> tuple[np.ndarray, bool]:
    """Bound each circuit update by a fraction of its seed-current magnitude."""
    update = np.asarray(delta, dtype=float)
    reference = np.asarray(reference_current, dtype=float)
    if update.shape != reference.shape:
        raise ValueError("the current-step reference must match the free-circuit count")
    if fraction is None:
        return update.copy(), False
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("current_step_fraction must be finite and positive")
    limit = fraction * np.abs(reference)
    capped = np.clip(update, -limit, limit)
    return capped, bool(np.any(capped != update))


def _delta_current_scale(
    scale,
    circuit_count: int,
    free_circuits: np.ndarray,
    regularisation: float,
) -> np.ndarray:
    """Return positive free-circuit scales for dimensionless delta penalties."""
    if not np.isfinite(regularisation) or regularisation < 0.0:
        raise ValueError("delta_regularisation must be finite and non-negative")
    if regularisation == 0.0:
        return np.ones(free_circuits.size)
    if scale is None:
        raise ValueError(
            "delta_current_scale is required when delta_regularisation is non-zero"
        )
    values = np.asarray(scale, dtype=float)
    if values.ndim == 0:
        values = np.full(free_circuits.size, float(values))
    elif values.shape == (free_circuits.size,):
        values = values.copy()
    elif values.shape == (circuit_count,):
        values = values[free_circuits]
    else:
        raise ValueError(
            "delta_current_scale must be a scalar, one value per circuit, "
            "or one value per free circuit"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("delta_current_scale values must be finite and positive")
    return values


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


def turning_point_response_matrix(
    profile: ForwardProfile,
    flux: jax.Array,
    free_circuits: Sequence[int],
    *,
    current_step: float = TURNING_POINT_TANGENT_STEP_A,
) -> np.ndarray:
    """Return the local achieved-turning-point response in metres per ampere.

    The prescribed-current carrier is linear in flux, while the boundary read
    and extremum refinement are deliberately evaluated afresh on both sides of
    every perturbation. This differentiates the physical coordinates used by
    the outer residual rather than treating flux and field rows as if they were
    distances. If a perturbation crosses a topology boundary, progressively
    smaller symmetric steps retain the same local derivative authority.
    """
    if not np.isfinite(current_step) or current_step <= 0.0:
        raise ValueError("current_step must be finite and positive")
    field = profile.operator.prescribed_current_field
    if field is None:
        raise ValueError("the shape inverse needs a prescribed current field")
    state = jnp.ravel(jnp.asarray(flux))
    response = jnp.asarray(field.response)
    free = np.asarray(free_circuits, dtype=int)
    columns = []
    for circuit in free:
        direction = response[:, int(circuit)]
        step = float(current_step)
        for _attempt in range(12):
            try:
                plus = np.asarray(
                    achieved_target(profile, state + step * direction).flux_points,
                    dtype=float,
                )[:4]
                minus = np.asarray(
                    achieved_target(profile, state - step * direction).flux_points,
                    dtype=float,
                )[:4]
            except NoQualifiedAxisError:
                step *= 0.5
                continue
            columns.append(((plus - minus) / (2.0 * step)).reshape(-1))
            break
        else:
            raise NoQualifiedAxisError(
                "no topology-preserving perturbation for turning-point tangent"
            )
    return np.column_stack(columns)


def _secant_refreshed_tangent(
    tangent: np.ndarray,
    current_delta: np.ndarray,
    achieved_delta: np.ndarray,
) -> np.ndarray:
    """Impose the admitted nonlinear shape secant on a refreshed local tangent."""
    step = np.asarray(current_delta, dtype=float)
    denominator = float(step @ step)
    if denominator <= np.finfo(float).tiny:
        return np.asarray(tangent, dtype=float)
    matrix = np.asarray(tangent, dtype=float)
    mismatch = np.asarray(achieved_delta, dtype=float).reshape(-1) - matrix @ step
    return matrix + np.outer(mismatch, step) / denominator


def _turning_point_current_update(
    tangent: np.ndarray,
    residual: np.ndarray,
    *,
    regularisation: float,
    delta_regularisation: float,
    delta_scale: np.ndarray,
) -> np.ndarray:
    """Solve one regularised physical turning-point Newton correction."""
    matrix = np.asarray(tangent, dtype=float)
    right_hand_side = np.asarray(residual, dtype=float).reshape(-1)
    if delta_regularisation == 0.0:
        return np.asarray(MoorePenrose(matrix, gamma=regularisation) / right_hand_side)
    scaled = matrix * delta_scale[np.newaxis, :]
    penalty = np.vstack(
        (
            regularisation * np.diag(delta_scale),
            np.sqrt(delta_regularisation) * np.eye(delta_scale.size),
        )
    )
    design = np.vstack((scaled, penalty))
    target = np.concatenate((right_hand_side, np.zeros(penalty.shape[0])))
    return delta_scale * np.asarray(MoorePenrose(design) / target)


def _deform_boundary_polygon(
    boundary: np.ndarray,
    previous_turning_points: np.ndarray,
    commanded_turning_points: np.ndarray,
) -> np.ndarray:
    """Map a measured separatrix onto its commanded bounding box.

    The four cardinal points carry the low-dimensional command, while the
    complete measured polygon supplies the isoflux rows.  Radius is an affine
    box map with upper and lower triangularity corrections; height is an
    affine box map.  The construction maps all four cardinal points exactly
    and gives every intervening boundary point a continuous commanded place.
    """
    previous = np.asarray(previous_turning_points, dtype=float)
    commanded = np.asarray(commanded_turning_points, dtype=float)
    if previous.shape != (4, 2) or commanded.shape != (4, 2):
        raise ValueError("shape steering needs four ordered turning points")
    source = np.asarray(boundary, dtype=float)
    radial_centre = 0.5 * (previous[0, 0] + previous[2, 0])
    commanded_radial_centre = 0.5 * (commanded[0, 0] + commanded[2, 0])
    radial_half_width = 0.5 * (previous[0, 0] - previous[2, 0])
    commanded_radial_half_width = 0.5 * (commanded[0, 0] - commanded[2, 0])
    height_centre = 0.5 * (previous[1, 1] + previous[3, 1])
    commanded_height_centre = 0.5 * (commanded[1, 1] + commanded[3, 1])
    height_half_span = 0.5 * (previous[1, 1] - previous[3, 1])
    commanded_height_half_span = 0.5 * (commanded[1, 1] - commanded[3, 1])
    if radial_half_width <= 0.0 or height_half_span <= 0.0:
        raise ValueError("the measured boundary must have nonzero box spans")
    height_coordinate = (source[:, 1] - height_centre) / height_half_span
    mapped_radius = (
        commanded_radial_centre
        + (source[:, 0] - radial_centre)
        * commanded_radial_half_width
        / radial_half_width
    )
    mapped_height = (
        commanded_height_centre
        + (source[:, 1] - height_centre) * commanded_height_half_span / height_half_span
    )
    upper_affine = (
        commanded_radial_centre
        + (previous[1, 0] - radial_centre)
        * commanded_radial_half_width
        / radial_half_width
    )
    lower_affine = (
        commanded_radial_centre
        + (previous[3, 0] - radial_centre)
        * commanded_radial_half_width
        / radial_half_width
    )
    mapped_radius += np.maximum(height_coordinate, 0.0) * (
        commanded[1, 0] - upper_affine
    )
    mapped_radius += np.maximum(-height_coordinate, 0.0) * (
        commanded[3, 0] - lower_affine
    )
    return np.column_stack((mapped_radius, mapped_height))


def shape_steering_target(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux,
) -> tuple[BoundingBoxTarget, np.ndarray]:
    """Expand four commanded extrema into a full boundary-polygon row set.

    The first four rows stay at the exact commanded turning points, retaining
    the explicit cardinal control semantics.  The remaining rows are every
    point of the measured boundary polygon after its continuous bounding-box
    deformation.  X-point rows remain attached to ``target.x_point`` so a
    moved null is read at its commanded location rather than at the seed. A
    non-null command also appends the uncommanded extrema at their prior
    locations, including their matching field rows, so the least-squares step
    must trade the desired motion against drift at every other turning point.
    """
    previous = achieved_target(profile, flux)
    source_polygon = boundary_polygon(profile, flux)
    commanded_turning_points = np.asarray(target.flux_points, dtype=float)[:4]
    if commanded_turning_points.shape != (4, 2):
        raise ValueError("shape steering targets must start with four turning points")
    transformed_polygon = _deform_boundary_polygon(
        source_polygon,
        np.asarray(previous.flux_points, dtype=float),
        commanded_turning_points,
    )
    previous_points = np.concatenate(
        (np.asarray(previous.flux_points, dtype=float), source_polygon)
    )
    command_delta = commanded_turning_points - np.asarray(previous.flux_points)
    command_scale = max(
        1.0,
        float(np.max(np.abs(commanded_turning_points))),
        float(np.max(np.abs(previous.flux_points))),
    )
    uncommanded = np.flatnonzero(
        np.max(np.abs(command_delta), axis=1)
        <= 64.0 * np.finfo(float).eps * command_scale
    )
    moved = bool(
        np.any(np.abs(command_delta) > 64.0 * np.finfo(float).eps * command_scale)
    )
    held_points = (
        np.asarray(previous.flux_points, dtype=float)[uncommanded]
        if moved
        else np.empty((0, 2))
    )
    commanded_points = np.concatenate(
        (commanded_turning_points, transformed_polygon, held_points)
    )
    radial_held = (
        np.asarray(previous.flux_points, dtype=float)[
            np.intersect1d(uncommanded, (0, 2))
        ]
        if moved
        else np.empty((0, 2))
    )
    vertical_held = (
        np.asarray(previous.flux_points, dtype=float)[
            np.intersect1d(uncommanded, (1, 3))
        ]
        if moved
        else np.empty((0, 2))
    )
    return (
        replace(
            target,
            flux_points=jnp.asarray(commanded_points, dtype=jnp.float64),
            radial_field_points=jnp.asarray(
                np.concatenate((target.radial_field_points, radial_held)),
                dtype=jnp.float64,
            ),
            vertical_field_points=jnp.asarray(
                np.concatenate((target.vertical_field_points, vertical_held)),
                dtype=jnp.float64,
            ),
        ),
        previous_points,
    )


def _admits_axis(
    profile: ForwardProfile,
    state: jax.Array,
    current: np.ndarray,
    *,
    requested_class=None,
    target_current=None,
    forward_solve: Callable[[np.ndarray], object] | None = None,
) -> bool:
    """Return whether a tentative prescribed-current forward state has an axis.

    Callers with an established nonlinear solve pass it through ``forward_solve``.
    The inverse itself otherwise projects one forward map before reading topology,
    which keeps a direct inverse call self-contained while still refusing a trial
    state that has already lost its axis.
    """
    try:
        trial = (
            forward_solve(current)
            if forward_solve is not None
            else profile.flux_map(
                requested_class=requested_class,
                target_current=target_current,
                prescribed_current=jnp.asarray(current),
            )(state)
        )
        trial_flux = jnp.asarray(getattr(trial, "flux", trial))
        profile.operator.read(trial_flux, requested_class=requested_class)
    except NoQualifiedAxisError:
        return False
    return True


def _admissible_delta(
    profile: ForwardProfile,
    state: jax.Array,
    initial_current: np.ndarray,
    free: np.ndarray,
    delta: np.ndarray,
    *,
    requested_class=None,
    target_current=None,
    forward_solve: Callable[[np.ndarray], object] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Contract one current update until its forward state admits an axis."""
    fraction = 1.0
    trials = 0
    refusals = []
    while fraction >= 2.0**-20:
        candidate = initial_current.copy()
        candidate[free] += fraction * delta
        trials += 1
        if _admits_axis(
            profile,
            state,
            candidate,
            requested_class=requested_class,
            target_current=target_current,
            forward_solve=forward_solve,
        ):
            return fraction * delta, fraction, trials
        refusals.append(fraction)
        fraction *= 0.5
    raise NoAdmissibleShapeStepError(refusals, delta)


def _consistency_floor(
    profile: ForwardProfile,
    flux,
    observed: np.ndarray,
    response: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    *,
    flux_rows: int,
) -> np.ndarray:
    """Return each row's seed-extraction floor in its own physical units.

    A symmetry-exact seed can make a field component numerically zero on both
    sides of the observation split.  The constraint extraction floor prevents
    such a row acquiring infinite authority merely because its measured scale
    happened to vanish.
    """
    active_image = response @ current
    fixed_and_plasma = observed - active_image
    characteristic = np.maximum.reduce(
        (np.abs(target), np.abs(active_image), np.abs(fixed_and_plasma))
    )
    span = _flux_span(profile, flux)
    extraction_floor = np.full(target.shape, 1.0e-6 * span)
    extraction_floor[flux_rows:] = 1.0e-6 * _row_scale(profile, span, "field")
    return np.maximum(0.01 * characteristic, extraction_floor)


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


def _shape_values(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    requested_class=None,
    target_current=None,
) -> jax.Array:
    """Return traced absolute flux and field values at commanded points.

    Unlike an isoflux residual, the flux block retains the absolute boundary
    level. This is the row layout used by the pulse-design inverse: four psi
    rows followed by Br and Bz turning-point rows and, for a diverted target,
    Br and Bz at the X-point.
    """
    from nova.equilibrium.constraint import ConstraintContext

    state = jnp.ravel(jnp.asarray(flux))
    context = ConstraintContext(state, requested_class, target_current, None)
    grid = jnp.reshape(state[: profile.lattice.node_count], profile.lattice.shape)
    flux_points = jnp.asarray(target.flux_points, dtype=jnp.float64)
    psi = jax.vmap(lambda point: sample_lattice_flux(profile.lattice, grid, point))(
        flux_points
    )
    field_points = jnp.concatenate(
        (
            jnp.asarray(target.radial_field_points, dtype=jnp.float64),
            jnp.asarray(target.vertical_field_points, dtype=jnp.float64),
        )
    )
    field = FieldComponentConstraint(
        components=("radial",) * int(jnp.shape(target.radial_field_points)[0])
        + ("vertical",) * int(jnp.shape(target.vertical_field_points)[0])
    ).observed(profile, context, field_points)
    if target.x_point is not None and np.shape(target.x_point) == (2,):
        x_point = jnp.asarray(target.x_point, dtype=jnp.float64)
        x_field = FieldComponentConstraint(components=("radial", "vertical")).observed(
            profile, context, jnp.repeat(x_point[None, :], 2, axis=0)
        )
        field = jnp.concatenate((field, x_field))
    return jnp.concatenate((psi, field))


def shape_values(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    requested_class=None,
    target_current=None,
) -> np.ndarray:
    """Return absolute flux and field values at the commanded points."""
    return np.asarray(
        _shape_values(
            profile,
            target,
            flux,
            requested_class=requested_class,
            target_current=target_current,
        )
    )


def shape_response_matrix(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    requested_class=None,
    target_current=None,
) -> np.ndarray:
    """Return the absolute shape rows' coil coupling in Wb/A and T/A."""
    field = profile.operator.prescribed_current_field
    if field is None:
        raise ValueError("the shape inverse needs a prescribed current field")
    state = jnp.ravel(jnp.asarray(flux))

    def rows(value):
        return _shape_values(
            profile,
            target,
            value,
            requested_class=requested_class,
            target_current=target_current,
        )

    jacobian = jax.jacrev(rows)(state)
    return np.asarray(jacobian @ jnp.asarray(field.response))


def shape_row_target(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    requested_class=None,
) -> np.ndarray:
    """Return boundary-flux and zero-field targets in forward conventions."""
    _masks, topology = profile.operator.read(
        jnp.asarray(flux), requested_class=requested_class
    )
    flux_rows = int(jnp.shape(target.flux_points)[0])
    field_rows = int(jnp.shape(target.radial_field_points)[0]) + int(
        jnp.shape(target.vertical_field_points)[0]
    )
    if target.x_point is not None and np.shape(target.x_point) == (2,):
        field_rows += 2
    # ForwardProfile already carries COCOS-17 total flux. This value is the
    # sign-adjusted boundary psi that the legacy inverse supplied through
    # update_constraints, so no second convention flip belongs here.
    boundary = float(np.asarray(topology.boundary_flux))
    return np.concatenate((np.full(flux_rows, boundary), np.zeros(field_rows)))


def plasma_current(profile: ForwardProfile, flux, *, target_current=None) -> float:
    """Return the plasma current the Tikhonov scale keys to."""
    if target_current is not None:
        return abs(float(np.asarray(target_current)))
    observation = profile.current_moment_observation(
        jnp.asarray(flux), support=MomentIntegralSupport.ALL_DOMAIN
    )
    return abs(float(np.asarray(observation.plasma_current)))


def _same_points(left, right) -> bool:
    """Return whether two commanded point sets are equal to roundoff."""
    left_points = np.asarray(left, dtype=float)
    right_points = np.asarray(right, dtype=float)
    if left_points.shape != right_points.shape:
        return False
    scale = max(
        1.0,
        float(np.max(np.abs(left_points))),
        float(np.max(np.abs(right_points))),
    )
    return bool(
        np.max(np.abs(left_points - right_points)) <= 64.0 * np.finfo(float).eps * scale
    )


def _is_unmoved_command(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux,
    previous_flux_points: np.ndarray,
    *,
    requested_class=None,
) -> bool:
    """Return whether every commanded location is the seed extraction."""
    previous_turning_points = np.asarray(previous_flux_points, dtype=float)[:4]
    if not _same_points(target.flux_points[:4], previous_turning_points):
        return False
    if not _same_points(target.radial_field_points, previous_turning_points[[0, 2]]):
        return False
    if not _same_points(target.vertical_field_points, previous_turning_points[[1, 3]]):
        return False
    _masks, topology = profile.operator.read(
        jnp.asarray(flux), requested_class=requested_class
    )
    seed_x_point = np.asarray(topology.x_point, dtype=float)
    has_seed_x_point = bool(np.asarray(topology.diverted)) and np.all(
        np.isfinite(seed_x_point)
    )
    if target.x_point is None:
        return not has_seed_x_point
    return has_seed_x_point and _same_points(target.x_point, seed_x_point)


def solve_shape_inverse(
    profile: ForwardProfile,
    target: BoundingBoxTarget,
    flux: jax.Array,
    *,
    prescribed_current=None,
    requested_class=None,
    target_current=None,
    free_circuits: Sequence[int] | None = None,
    gamma: float = GAMMA,
    field_weight: float = FIELD_WEIGHT,
    picard_rounds: int = PICARD_ROUNDS,
    current_step_fraction: float | None = None,
    current_step_reference=None,
    delta_regularisation: float = 0.0,
    delta_current_scale=None,
    forward_solve: Callable[[np.ndarray], object] | None = None,
    turning_point_relative_tolerance: float = TURNING_POINT_RELATIVE_TOLERANCE,
) -> ShapeInverseResult:
    """Solve seed-anchored free-circuit changes against achieved shape.

    At each round, the coil coupling is solved for a current change about the
    fixed seed after the fixed-conductor, plasma and seed-current images
    have been subtracted from the target. Every round remains anchored to that
    same seed rather than accumulating changes from the prior round. Between
    solves one forward-map evaluation re-evaluates the plasma profile inside
    the boundary produced by those currents. No nonlinear equilibrium solve
    is run here. When ``delta_regularisation`` is non-zero, its Tikhonov term
    is applied to each delta divided by ``delta_current_scale``. The scale is
    therefore a rated-current vector or caller-stated current ceiling, and
    the penalty is dimensionless. Without ``forward_solve`` this remains a
    proposal-only placement calculation. With a referee, every admitted
    nonlinear result is measured in turning-point coordinates and corrected
    until its largest point error is within the stated fraction of the largest
    commanded motion, or the bounded iteration count is exhausted.
    """
    if picard_rounds < 0:
        raise ValueError("picard_rounds must be non-negative")
    if (
        not np.isfinite(turning_point_relative_tolerance)
        or turning_point_relative_tolerance <= 0.0
    ):
        raise ValueError("turning_point_relative_tolerance must be finite and positive")
    field = profile.operator.prescribed_current_field
    if field is None:
        raise ValueError("the shape inverse needs a prescribed current field")
    initial_current = np.asarray(
        field.current if prescribed_current is None else prescribed_current,
        dtype=float,
    )
    if initial_current.shape != (field.circuit_count,):
        raise ValueError(
            "prescribed_current must match the response column count "
            f"{field.circuit_count}"
        )
    state = jnp.asarray(flux)
    current = initial_current.copy()
    if free_circuits is None:
        free = np.arange(field.circuit_count)
    else:
        free = np.unique(np.asarray(free_circuits, dtype=int))
    if free.size == 0:
        raise ValueError("the shape-inverse step needs at least one free circuit")
    delta_scale = _delta_current_scale(
        delta_current_scale,
        field.circuit_count,
        free,
        delta_regularisation,
    )
    if current_step_reference is None:
        step_reference = initial_current
    else:
        step_reference = np.asarray(current_step_reference, dtype=float)
    if step_reference.shape != (field.circuit_count,):
        raise ValueError(
            "current_step_reference must match the response column count "
            f"{field.circuit_count}"
        )
    row_target, previous_flux_points = shape_steering_target(profile, target, state)
    unmoved_command = _is_unmoved_command(
        profile,
        row_target,
        state,
        previous_flux_points,
        requested_class=requested_class,
    )
    target_rows = shape_row_target(
        profile, row_target, state, requested_class=requested_class
    )
    initial_observed = shape_values(
        profile,
        row_target,
        state,
        requested_class=requested_class,
        target_current=target_current,
    )
    if unmoved_command:
        # Ray-cast boundary points and the topology saddle are extraction
        # coordinates. On a coarse diverted lattice their interpolated rows
        # need not equal the scalar boundary level or exact zero field. A null
        # command targets those same extracted values, making its delta system
        # identically zero without weakening moved-point targets.
        target_rows = initial_observed.copy()
    initial_response = shape_response_matrix(
        profile,
        row_target,
        state,
        requested_class=requested_class,
        target_current=target_current,
    )
    flux_rows = int(jnp.shape(row_target.flux_points)[0])
    consistency_floor = _consistency_floor(
        profile,
        state,
        initial_observed,
        initial_response,
        initial_current,
        target_rows,
        flux_rows=flux_rows,
    )
    row_weight = 1.0 / consistency_floor
    row_weight[flux_rows:] *= np.sqrt(field_weight)
    initial_right_hand_side = target_rows - initial_observed
    numerical_zero = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.max(np.abs(target_rows))),
            float(np.max(np.abs(initial_observed))),
        )
    )
    placement_rounds = (
        0
        if float(np.max(np.abs(initial_right_hand_side))) <= numerical_zero
        else picard_rounds
    )
    if forward_solve is not None:
        placement_rounds = 0
    picard_current_history = []
    picard_boundary_history = []
    current_step_limited = False
    accepted_fraction = 1.0
    admissibility_trials = 0
    admitted_forward: dict[str, object] = {}

    def capture_forward(candidate: np.ndarray):
        if forward_solve is None:
            raise RuntimeError("a nonlinear forward referee was not supplied")
        result = forward_solve(candidate)
        admitted_forward["result"] = result
        return result

    for iteration in range(placement_rounds + 1):
        _masks, topology = profile.operator.read(state, requested_class=requested_class)
        picard_boundary_history.append(float(np.asarray(topology.boundary_flux)))
        if iteration == 0:
            response = initial_response
            seed_observed = initial_observed
        else:
            full_observed = shape_values(
                profile,
                row_target,
                state,
                requested_class=requested_class,
                target_current=target_current,
            )
            response = shape_response_matrix(
                profile,
                row_target,
                state,
                requested_class=requested_class,
                target_current=target_current,
            )
            # Removing the current free-circuit image leaves the plasma plus
            # every fixed conductor. The original seed image is then held on
            # that side of the equation so regularisation selects the smallest
            # steering change, not the smallest absolute machine-current state.
            base = full_observed - response[:, free] @ current[free]
            seed_observed = base + response[:, free] @ initial_current[free]
        right_hand_side = target_rows - seed_observed
        weighted = response[:, free] * row_weight[:, None]
        weighted_rhs = right_hand_side * row_weight
        ip = plasma_current(profile, state, target_current=target_current)
        regularisation = gamma * abs(ip)
        if delta_regularisation == 0.0:
            solved_delta = MoorePenrose(weighted, gamma=regularisation) / weighted_rhs
        else:
            scaled_response = weighted * delta_scale[np.newaxis, :]
            penalty_rows = np.vstack(
                (
                    regularisation * np.diag(delta_scale),
                    np.sqrt(delta_regularisation) * np.eye(free.size),
                )
            )
            scaled_design = np.vstack((scaled_response, penalty_rows))
            scaled_rhs = np.concatenate((weighted_rhs, np.zeros(penalty_rows.shape[0])))
            solved_delta = delta_scale * (MoorePenrose(scaled_design) / scaled_rhs)
        applied_round_delta, limited = _cap_current_delta(
            solved_delta,
            step_reference[free],
            current_step_fraction,
        )
        current_step_limited = current_step_limited or limited
        final_without_referee = iteration == placement_rounds and forward_solve is None
        if final_without_referee:
            fraction = 1.0
            trials = 0
        else:
            applied_round_delta, fraction, trials = _admissible_delta(
                profile,
                state,
                initial_current,
                free,
                applied_round_delta,
                requested_class=requested_class,
                target_current=target_current,
                forward_solve=(
                    capture_forward
                    if forward_solve is not None and iteration == placement_rounds
                    else None
                ),
            )
        accepted_fraction = fraction
        admissibility_trials += trials
        current[free] = initial_current[free] + applied_round_delta
        picard_current_history.append(current.copy())
        if iteration < placement_rounds:
            state = profile.flux_map(
                requested_class=requested_class,
                target_current=target_current,
                prescribed_current=jnp.asarray(current),
            )(state)

    commanded_turning_points = np.asarray(target.flux_points, dtype=float)[:4]
    previous_turning_points = np.asarray(previous_flux_points, dtype=float)[:4]
    commanded_motion = commanded_turning_points - previous_turning_points
    command_norm = float(np.max(np.linalg.norm(commanded_motion, axis=1)))
    turning_point_tolerance = turning_point_relative_tolerance * command_norm
    iteration_history: list[ShapeInverseIteration] = []
    achieved_turning_points = previous_turning_points.copy()
    physical_residual = commanded_turning_points - achieved_turning_points
    physical_residual_norm = float(np.max(np.linalg.norm(physical_residual, axis=1)))
    converged = command_norm == 0.0
    last_uncapped_total = np.asarray(solved_delta, dtype=float).copy()

    if forward_solve is not None:
        admitted_result = admitted_forward.get("result")
        if admitted_result is None:
            raise RuntimeError("the axis-admission referee returned no forward result")
        state = jnp.asarray(getattr(admitted_result, "flux", admitted_result))
        achieved_turning_points = np.asarray(
            achieved_target(profile, state).flux_points, dtype=float
        )[:4]
        physical_residual = commanded_turning_points - achieved_turning_points
        physical_residual_norm = float(
            np.max(np.linalg.norm(physical_residual, axis=1))
        )
        tangent = turning_point_response_matrix(profile, flux, free)
        tangent = _secant_refreshed_tangent(
            tangent,
            current[free] - initial_current[free],
            achieved_turning_points - previous_turning_points,
        )
        tangent_singular_values = np.linalg.svd(tangent, compute_uv=False)
        iteration_history.append(
            ShapeInverseIteration(
                commanded_turning_points=commanded_turning_points.copy(),
                achieved_turning_points=achieved_turning_points.copy(),
                turning_point_residual=physical_residual.copy(),
                turning_point_residual_norm=physical_residual_norm,
                tangent_singular_values=tangent_singular_values,
                tangent_numerical_rank=int(np.linalg.matrix_rank(tangent)),
                proposed_delta=np.asarray(solved_delta, dtype=float).copy(),
                admitted_delta=np.asarray(applied_round_delta, dtype=float).copy(),
                accepted_fraction=float(accepted_fraction),
                admissibility_trials=int(trials),
            )
        )
        converged = physical_residual_norm <= turning_point_tolerance
        maximum_iterations = max(1, picard_rounds)
        prior_current = initial_current[free].copy()
        prior_achieved = previous_turning_points.copy()
        while not converged and len(iteration_history) < maximum_iterations:
            tangent = turning_point_response_matrix(profile, state, free)
            tangent = _secant_refreshed_tangent(
                tangent,
                current[free] - prior_current,
                achieved_turning_points - prior_achieved,
            )
            proposed_update = _turning_point_current_update(
                tangent,
                physical_residual,
                regularisation=regularisation,
                delta_regularisation=delta_regularisation,
                delta_scale=delta_scale,
            )
            capped_update, limited = _cap_current_delta(
                proposed_update,
                step_reference[free],
                current_step_fraction,
            )
            current_step_limited = current_step_limited or limited
            base_current = current.copy()
            prior_current = current[free].copy()
            prior_achieved = achieved_turning_points.copy()
            admitted_forward.clear()
            admitted_update, fraction, trials = _admissible_delta(
                profile,
                state,
                base_current,
                free,
                capped_update,
                requested_class=requested_class,
                target_current=target_current,
                forward_solve=capture_forward,
            )
            current[free] += admitted_update
            last_uncapped_total = (
                base_current[free] + proposed_update - initial_current[free]
            )
            accepted_fraction = fraction
            admissibility_trials += trials
            current_step_limited = current_step_limited or fraction < 1.0
            admitted_result = admitted_forward.get("result")
            if admitted_result is None:
                raise RuntimeError(
                    "the axis-admission referee returned no forward result"
                )
            state = jnp.asarray(getattr(admitted_result, "flux", admitted_result))
            achieved_turning_points = np.asarray(
                achieved_target(profile, state).flux_points, dtype=float
            )[:4]
            physical_residual = commanded_turning_points - achieved_turning_points
            physical_residual_norm = float(
                np.max(np.linalg.norm(physical_residual, axis=1))
            )
            tangent_singular_values = np.linalg.svd(tangent, compute_uv=False)
            iteration_history.append(
                ShapeInverseIteration(
                    commanded_turning_points=commanded_turning_points.copy(),
                    achieved_turning_points=achieved_turning_points.copy(),
                    turning_point_residual=physical_residual.copy(),
                    turning_point_residual_norm=physical_residual_norm,
                    tangent_singular_values=tangent_singular_values,
                    tangent_numerical_rank=int(np.linalg.matrix_rank(tangent)),
                    proposed_delta=np.asarray(proposed_update, dtype=float).copy(),
                    admitted_delta=np.asarray(admitted_update, dtype=float).copy(),
                    accepted_fraction=float(fraction),
                    admissibility_trials=int(trials),
                )
            )
            picard_current_history.append(current.copy())
            _masks, achieved_topology = profile.operator.read(
                state, requested_class=requested_class
            )
            picard_boundary_history.append(
                float(np.asarray(achieved_topology.boundary_flux))
            )
            converged = physical_residual_norm <= turning_point_tolerance

    singular_values = np.linalg.svd(weighted, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(weighted))
    right_vectors_h = np.linalg.svd(weighted, full_matrices=True)[2]
    uncapped_current = current.copy()
    uncapped_current[free] = initial_current[free] + last_uncapped_total
    delta_free = current[free] - initial_current[free]
    uncapped_delta = uncapped_current[free] - initial_current[free]
    linear_prediction = response[:, free] @ delta_free
    row_kinds = ("flux",) * flux_rows + ("field",) * (target_rows.size - flux_rows)
    return ShapeInverseResult(
        currents=current,
        delta=delta_free,
        uncapped_delta=uncapped_delta,
        free_circuits=free,
        response=response,
        observed=seed_observed,
        target=target_rows,
        right_hand_side=right_hand_side,
        linear_prediction=linear_prediction,
        row_kinds=row_kinds,
        plasma_current=float(ip),
        gamma=float(regularisation),
        delta_regularisation=float(delta_regularisation),
        delta_current_scale=delta_scale,
        field_weight=field_weight,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        right_null_space=right_vectors_h[numerical_rank:],
        picard_currents=np.asarray(picard_current_history),
        picard_boundary_flux=np.asarray(picard_boundary_history),
        current_step_fraction=current_step_fraction,
        current_step_limited=current_step_limited,
        accepted_fraction=accepted_fraction,
        admissibility_trials=admissibility_trials,
        least_squares_residual=float(
            np.linalg.norm(weighted @ delta_free - weighted_rhs)
        ),
        uncapped_least_squares_residual=float(
            np.linalg.norm(weighted @ solved_delta - weighted_rhs)
        ),
        flux_points=np.asarray(row_target.flux_points, dtype=float),
        previous_flux_points=previous_flux_points,
        consistency_floor=consistency_floor,
        row_weight=row_weight,
        iterations=tuple(iteration_history),
        achieved_flux_points=achieved_turning_points,
        turning_point_residual=physical_residual,
        turning_point_residual_norm=physical_residual_norm,
        turning_point_tolerance=turning_point_tolerance,
        converged=converged,
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
    "PICARD_ROUNDS",
    "TURNING_POINT_RELATIVE_TOLERANCE",
    "TURNING_POINT_TANGENT_STEP_A",
    "NoAdmissibleShapeStepError",
    "ShapeInverseIteration",
    "ShapeInverseResult",
    "achieved_target",
    "boundary_polygon",
    "bounding_box_pairs",
    "observed_values",
    "plasma_current",
    "reference_point",
    "response_matrix",
    "shape_response_matrix",
    "shape_steering_target",
    "shape_row_target",
    "shape_values",
    "solve_shape_inverse",
    "turning_point_error",
    "turning_point_response_matrix",
]
