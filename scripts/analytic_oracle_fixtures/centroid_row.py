"""Centroid and flux-level constraint pairs for the coil-less analytic fixture.

Three compensating columns carry the fixture's constraint family.  The two field
columns shift the plasma position; the level column adds a uniform flux offset,
which carries no poloidal field and therefore no force.  Both solenoidal columns
vanish at the magnetic axis, so only the level column can move the flux level
with the position pinned -- the three columns together are what make an authored
fixed point reachable.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from nova.equilibrium.constraint import (
    BoundedExteriorFieldUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
    FluxLevelConstraint,
)
from nova.equilibrium.observation import MomentIntegralSupport
from scripts.analytic_oracle_fixtures.measure import (
    EXTERIOR_COMPENSATION_COLUMNS,
    EXTERIOR_FIELD_COMPONENTS,
    EXTERIOR_LEVEL_COMPONENT,
)


DEFAULT_FIELD_SCALE_T = 1.0e-3
DEFAULT_FIELD_BOUND_T = 2.5e-1
DEFAULT_STEP_LIMIT = 1.0
DEFAULT_LEVEL_SCALE_WB = 1.0
CENTROID_COMPONENTS = ("centroid_r", "centroid_z")
LEVEL_COMPONENTS = (EXTERIOR_LEVEL_COMPONENT,)


def _field_direction(components: Sequence[str]) -> jnp.ndarray:
    """Select one compensating column per constraint row.

    A radial-position row is *driven* by a uniform vertical field and a
    height row by a uniform radial field; a flux-level row is driven by the
    uniform flux offset alone.  The returned matrix has shape
    ``(column_count, row_count)`` over :data:`EXTERIOR_COMPENSATION_COLUMNS`, so
    a row that the level column cannot move carries a zero in its slot.
    """
    columns = []
    for component in components:
        if component == "centroid_r":
            columns.append((1.0, 0.0, 0.0))
        elif component == "centroid_z":
            columns.append((0.0, 1.0, 0.0))
        elif component == EXTERIOR_LEVEL_COMPONENT:
            columns.append((0.0, 0.0, 1.0))
        else:
            raise ValueError(f"unsupported fixture component {component!r}")
    return jnp.asarray(columns, dtype=jnp.float64).T


def centroid_constraint_pair(
    target,
    *,
    pitch: float,
    components: Sequence[str] = CENTROID_COMPONENTS,
    field_scale_t: float = DEFAULT_FIELD_SCALE_T,
    field_bound_t: float = DEFAULT_FIELD_BOUND_T,
    step_limit: float = DEFAULT_STEP_LIMIT,
    initial_field_t=None,
) -> ConstraintPair:
    """Bind analytic centroid targets to bounded uniform exterior fields.

    The pair's direction spans the two solenoidal columns and leaves the level
    column at zero: a centroid row is gauge blind, so a uniform flux offset has
    no leverage on it.
    """
    selected = tuple(components)
    rows = len(selected)
    if not rows:
        raise ValueError("at least one centroid component is required")
    if any(component == EXTERIOR_LEVEL_COMPONENT for component in selected):
        raise ValueError("a centroid row cannot be driven by the level column")
    target_value = jnp.atleast_1d(jnp.asarray(target))
    if target_value.shape != (rows,):
        raise ValueError("centroid target must have one value per selected component")
    scale = jnp.full(rows, float(pitch))
    field_scale = jnp.full(rows, float(field_scale_t))
    initial = (
        jnp.zeros(rows)
        if initial_field_t is None
        else jnp.atleast_1d(jnp.asarray(initial_field_t)) / field_scale
    )
    return ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=selected,
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=BoundedExteriorFieldUnknown(
            direction=_field_direction(selected),
            field_scale=field_scale,
            field_bound=jnp.full(rows, float(field_bound_t)),
            step_limit=jnp.full(rows, float(step_limit)),
        ),
        binding=ConstraintBinding(
            target=target_value,
            tolerance=scale * 1.0e-12,
            scale=scale,
            initial_unknown=initial,
            payload=None,
            policy="imposed",
        ),
    )


def level_constraint_pair(
    point,
    target,
    *,
    level_scale_wb: float = DEFAULT_LEVEL_SCALE_WB,
    step_limit: float = DEFAULT_STEP_LIMIT,
    initial_level_wb=None,
) -> ConstraintPair:
    """Bind one flux-level target at a fixed point to an unbounded level column.

    The row reads the total flux the map interpolates at ``point``, and the
    compensating term is the level column: a constant added identically at every
    target, in weber.  The level amplitude is reported beside the two field
    amplitudes and is not subject to their tesla bound -- it carries no poloidal
    field, so no field magnitude can describe it.  Only the per-trip step cap
    limits it, and that cap is on the normalized unknown rather than a physical
    bound.
    """
    coordinates = jnp.reshape(jnp.asarray(point, dtype=jnp.float64), (1, 2))
    target_value = jnp.atleast_1d(jnp.asarray(target, dtype=jnp.float64))
    if target_value.shape != (1,):
        raise ValueError("a flux-level row needs one target value")
    scale = jnp.full(1, float(level_scale_wb))
    initial = (
        jnp.zeros(1)
        if initial_level_wb is None
        else jnp.atleast_1d(jnp.asarray(initial_level_wb, dtype=jnp.float64)) / scale
    )
    return ConstraintPair(
        functional=FluxLevelConstraint(point_count=1),
        unknown=BoundedExteriorFieldUnknown(
            direction=_field_direction(LEVEL_COMPONENTS),
            field_scale=jnp.full(1, float(level_scale_wb)),
            field_bound=jnp.full(1, jnp.inf),
            step_limit=jnp.full(1, float(step_limit)),
        ),
        binding=ConstraintBinding(
            target=target_value,
            tolerance=scale * 1.0e-12,
            scale=scale,
            initial_unknown=initial,
            payload=coordinates,
            policy="imposed",
        ),
    )


def fixture_constraint_pairs(
    centroid_target,
    *,
    level_point,
    level_target,
    pitch: float,
    components: Sequence[str] = CENTROID_COMPONENTS,
    field_scale_t: float = DEFAULT_FIELD_SCALE_T,
    field_bound_t: float = DEFAULT_FIELD_BOUND_T,
    level_scale_wb: float = DEFAULT_LEVEL_SCALE_WB,
    step_limit: float = DEFAULT_STEP_LIMIT,
    initial_field_t=None,
    initial_level_wb=None,
) -> tuple[ConstraintPair, ConstraintPair]:
    """Assembling builder: the centroid pair and the flux-level pair.

    The returned pair list is ordered as the solve consumes it -- the position
    rows first, then the level row -- and each pair carries its own unknown and
    binding, so the solve reports three amplitudes for this fixture: two in
    tesla against the declared field bound and one in weber outside it.
    """
    return (
        centroid_constraint_pair(
            centroid_target,
            pitch=pitch,
            components=components,
            field_scale_t=field_scale_t,
            field_bound_t=field_bound_t,
            step_limit=step_limit,
            initial_field_t=initial_field_t,
        ),
        level_constraint_pair(
            level_point,
            level_target,
            level_scale_wb=level_scale_wb,
            step_limit=step_limit,
            initial_level_wb=initial_level_wb,
        ),
    )


def reader_identity() -> dict[str, object]:
    """Return what the level row's point read is built from, with its sources.

    The statement is static because the row does not choose its reader by
    value: a cell-carried carrier is always read through the owning cell's
    own-node quadratic, whose weights are solved once on the host when the mesh
    is built and are then fixed arrays in the traced read.
    """
    return {
        "carrier": "cell-carried mesh (per-cell centroid coordinate)",
        "reader": "own-node quadratic of the owning cell, evaluated at the point",
        "fit": "one local 6-coefficient quadratic per node ring, host-solved",
        "interpolation": "local per node, not a global spline",
        "neighbourhood": "the node's sampling polygon, centre-first and complete",
        "normalisation": "ring centred on its own node and scaled to unit width",
        "ownership": "the cell whose centroid is nearest the point",
        "weights_source": "nova/equilibrium/stencil_mesh.py:817",
        "weights_expression": "np.linalg.pinv(_quadratic_design(ring_local))",
        "design_source": "nova/equilibrium/stencil_mesh.py:433",
        "device_read_source": "nova/equilibrium/stencil_mesh.py:245",
        "operator_read_source": "nova/equilibrium/forward_operator.py:2682",
        "row_read_source": "nova/equilibrium/constraint.py:_mesh_carried_point_flux",
        "host_solved_at_build": True,
        "global_spline": False,
    }


def exterior_field_identity() -> dict[str, object]:
    """Return the semantic direction and finite-bound identity for receipts."""
    return {
        "response_columns": list(EXTERIOR_COMPENSATION_COLUMNS),
        "field_columns": list(EXTERIOR_FIELD_COMPONENTS),
        "level_column": EXTERIOR_LEVEL_COMPONENT,
        "centroid_r_compensator": "uniform vertical field",
        "centroid_z_compensator": "uniform radial field",
        "level_compensator": "uniform flux offset in weber",
        "field_scale_t": DEFAULT_FIELD_SCALE_T,
        "field_bound_t": DEFAULT_FIELD_BOUND_T,
        "level_scale_wb": DEFAULT_LEVEL_SCALE_WB,
        "level_bound_wb": None,
        "level_bound_is_field_bound": False,
        "step_limit": DEFAULT_STEP_LIMIT,
        "bound_route": "damped step control with a recorded refusal",
    }
