"""Fixed-shape state-space geometry for topology-manifold advance.

The topology predicate is deliberately absent from this module.  Callers own
that physical classification and use these operations only to construct a
state-space secant, advance along it, and constrain a corrector to the normal
space.  Every operation has a fixed output shape and uses array selection
rather than value-dependent host control flow.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "ManifoldAdvanceQualification",
    "SecantFrame",
    "normal_component",
    "oriented_secant",
]


class ManifoldAdvanceQualification(IntEnum):
    """Host-readable outcome of one attempted manifold advance."""

    NOT_APPLICABLE = 0
    ACCEPTED = 1
    DEGENERATE_SECANT = 2
    KRYLOV_ACTION_REFUSED = 3
    NONFINITE_CORRECTED_STATE = 4
    INADMISSIBLE_CORRECTED_STATE = 5
    ZERO_MATERIAL_ADVANCE = 6


class SecantFrame(NamedTuple):
    """One oriented unit secant and its fixed-shape qualification."""

    tangent: jax.Array
    length: jax.Array
    qualification: jax.Array


def oriented_secant(
    previous: jax.Array,
    current: jax.Array,
    orientation: jax.Array,
) -> SecantFrame:
    """Return the admitted-state secant oriented with a reference direction.

    A secant is refused when its length is indistinguishable from zero at the
    scale of the two states.  The returned tangent is then exactly zero, so a
    caller that accidentally consumes a refused frame still cannot advance.
    A negative inner product flips the unit secant; this removes the arbitrary
    sign of a one-dimensional tangent basis without changing its span.
    """
    previous = jnp.asarray(previous)
    current = jnp.asarray(current)
    orientation = jnp.asarray(orientation)
    if previous.shape != current.shape or current.shape != orientation.shape:
        raise ValueError("secant states and orientation must have identical shapes")
    if current.ndim != 1:
        raise ValueError("manifold states must be flat one-dimensional arrays")

    difference = current - previous
    length = jnp.linalg.norm(difference)
    state_scale = jnp.maximum(
        jnp.maximum(jnp.linalg.norm(previous), jnp.linalg.norm(current)),
        jnp.asarray(1.0, dtype=current.dtype),
    )
    floor = 32.0 * jnp.finfo(current.dtype).eps * state_scale
    material = jnp.isfinite(length) & (length > floor)
    safe_length = jnp.maximum(length, jnp.finfo(current.dtype).tiny)
    tangent = jnp.where(material, difference / safe_length, jnp.zeros_like(difference))
    orientation_finite = jnp.all(jnp.isfinite(orientation))
    flip = material & orientation_finite & (jnp.vdot(tangent, orientation) < 0.0)
    tangent = jnp.where(flip, -tangent, tangent)
    qualification = jnp.where(
        material,
        ManifoldAdvanceQualification.ACCEPTED,
        ManifoldAdvanceQualification.DEGENERATE_SECANT,
    )
    return SecantFrame(tangent, length, qualification.astype(jnp.int32))


def normal_component(vector: jax.Array, tangent: jax.Array) -> jax.Array:
    """Project ``vector`` normal to a unit state-space tangent."""
    vector = jnp.asarray(vector)
    tangent = jnp.asarray(tangent)
    if vector.shape != tangent.shape:
        raise ValueError("vector and tangent must have identical shapes")
    return vector - jnp.vdot(vector, tangent) * tangent
