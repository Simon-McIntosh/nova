"""Prototype a full-jet Bell read with a fresh stationary-root census."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import topology
from scripts.prototypes.lean_map.quadrature import (
    cut_moments,
    interior_moments,
)
from scripts.prototypes.read_robustness.census import axis_identity, census


class RobustState(NamedTuple):
    moments: jax.Array
    raw_current: jax.Array
    roots: jax.Array
    root_valid: jax.Array
    axis_identity: jax.Array
    axis_distance: jax.Array
    candidate_count: jax.Array
    axis_count: jax.Array
    saddle_count: jax.Array
    census_overflow: jax.Array
    boundary_flux: jax.Array
    boundary_class: jax.Array
    axis_flux: jax.Array
    wall_candidate: jax.Array
    saddle_candidates: jax.Array
    booking_status: jax.Array
    topology_reason: jax.Array
    topology_valid: jax.Array
    topology_qualified: jax.Array
    valid: jax.Array


def read(
    moments,
    values,
    jets,
    rule,
    geometry,
    profile,
    previous,
    expected_axis,
    convention,
    policy,
    target_current,
):
    """Read every jet row, census roots, then book the selected support."""
    field = jets.field(moments)
    roots = census(field, geometry, convention, policy, previous)
    topology_state = topology.read_at_stationary_points(
        field, geometry, convention, policy, roots.roots
    )
    pitch = jnp.median(geometry.pitch)
    identity = axis_identity(topology_state.axis, expected_axis, pitch)
    gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
    full = (topology_state.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    interior = interior_moments(
        rule,
        gauss_flux,
        profile,
        topology_state.axis_flux,
        topology_state.boundary_flux,
        full,
    )
    samples = field.jets[field.sample_index, 0]
    cut = cut_moments(
        samples,
        topology_state,
        geometry,
        profile,
        convention.sigma,
        len(geometry.centre),
    )
    raw = interior + cut.moments
    current = jnp.sum(raw[:, 0])
    valid = (
        topology_state.valid
        & topology_state.qualified
        & identity
        & jnp.any(roots.roots.valid)
        & ~roots.overflow
        & (cut.status == 0)
        & jnp.isfinite(current)
        & (current != 0)
    )
    normalized = raw * target_current / jnp.where(current != 0, current, 1)
    normalized = jnp.where(valid, normalized, jnp.nan)
    _, wall_flux = topology._wall_events(field, geometry, policy, convention.sigma)
    candidate_wall = jnp.max(convention.sigma * wall_flux) * convention.sigma
    candidate_saddles = jnp.where(
        roots.roots.valid[1:], roots.roots.jet.value[1:], jnp.nan
    )
    return RobustState(
        normalized,
        current,
        roots.roots.position,
        roots.roots.valid,
        identity,
        roots.axis_distance / pitch,
        roots.candidates,
        roots.axes,
        roots.saddles,
        roots.overflow,
        topology_state.boundary_flux,
        topology_state.boundary_class,
        topology_state.axis_flux,
        candidate_wall,
        candidate_saddles,
        cut.status,
        topology_state.reason,
        topology_state.valid,
        topology_state.qualified,
        valid,
    )


def jet_bytes(jets):
    """The full stored source response footprint in bytes."""
    return jets.rows.size * jets.rows.dtype.itemsize
