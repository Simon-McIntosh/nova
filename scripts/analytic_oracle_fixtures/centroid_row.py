"""Centroid constraint pairs for the coil-less analytic fixture."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from nova.equilibrium.constraint import (
    BoundedExteriorFieldUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.observation import MomentIntegralSupport
from scripts.analytic_oracle_fixtures.measure import EXTERIOR_FIELD_COMPONENTS


DEFAULT_FIELD_SCALE_T = 1.0e-3
DEFAULT_FIELD_BOUND_T = 2.5e-1
CENTROID_COMPONENTS = ("centroid_r", "centroid_z")


def _field_direction(components: Sequence[str]) -> jnp.ndarray:
    """Select vertical field for radial position and radial field for height."""
    columns = []
    for component in components:
        if component == "centroid_r":
            columns.append((1.0, 0.0))
        elif component == "centroid_z":
            columns.append((0.0, 1.0))
        else:
            raise ValueError(f"unsupported centroid component {component!r}")
    return jnp.asarray(columns).T


def centroid_constraint_pair(
    target,
    *,
    pitch: float,
    components: Sequence[str] = CENTROID_COMPONENTS,
    field_scale_t: float = DEFAULT_FIELD_SCALE_T,
    field_bound_t: float = DEFAULT_FIELD_BOUND_T,
    initial_field_t=None,
) -> ConstraintPair:
    """Bind analytic centroid targets to bounded uniform exterior fields."""
    selected = tuple(components)
    rows = len(selected)
    if not rows:
        raise ValueError("at least one centroid component is required")
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


def exterior_field_identity() -> dict[str, object]:
    """Return the semantic direction and finite-bound identity for receipts."""
    return {
        "response_columns": list(EXTERIOR_FIELD_COMPONENTS),
        "centroid_r_compensator": "uniform vertical field",
        "centroid_z_compensator": "uniform radial field",
        "field_scale_t": DEFAULT_FIELD_SCALE_T,
        "field_bound_t": DEFAULT_FIELD_BOUND_T,
    }
