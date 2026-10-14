"""Fixed-shape candidate forward map with dense interior and bounded cut rows."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import fixed_points, topology
from scripts.prototypes.lean_map.quadrature import (
    BookingStatus,
    cut_moments,
    interior_moments,
)


class LeanState(NamedTuple):
    moments: jax.Array
    roots: jax.Array
    root_status: jax.Array
    cut_count: jax.Array
    booking_status: jax.Array
    topology_valid: jax.Array
    topology_qualified: jax.Array


def apply(
    moments,
    values,
    jets,
    window,
    rule,
    geometry,
    profile,
    seeds,
    previous,
    convention,
    policy,
    target_current,
    cut_capacity,
    axis_sign,
):
    """Return one live connected-support map evaluation at fixed carrier shape."""
    nodal = jets.gather(moments)
    field = fixed_points.field_from_jets(nodal, jets.sample_index, jets.mesh)
    roots = fixed_points.solve(
        window,
        moments,
        seeds,
        previous,
        jnp.median(geometry.pitch),
        axis_sign,
    )
    reading = topology.read_at_stationary_points(
        field, geometry, convention, policy, roots
    )
    gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
    full = reading.membership >= 1 - 1e-14
    interior = interior_moments(
        rule, gauss_flux, profile, reading.axis_flux, reading.boundary_flux, full
    )
    samples = nodal[jets.sample_index, 0]
    centre = jax.vmap(field.cell_value)(
        jnp.arange(len(geometry.centre)), geometry.centre
    )
    samples = samples.at[:, 0].set(centre)
    cut = cut_moments(
        samples, reading, geometry, profile, convention.sigma, cut_capacity
    )
    unscaled = interior + cut.moments
    raw_current = jnp.sum(unscaled[:, 0])
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & (cut.status == int(BookingStatus.VALID))
        & jnp.isfinite(raw_current)
        & (raw_current != 0)
    )
    normalized = unscaled * (
        target_current / jnp.where(raw_current != 0, raw_current, 1)
    )
    normalized = jnp.where(valid, normalized, jnp.nan)
    return LeanState(
        normalized,
        roots.position,
        roots.status,
        cut.cut_count,
        cut.status,
        reading.valid,
        reading.qualified,
    )


def residual(*args):
    """Residual in source moments, preserving typed read and booking refusals."""
    current = args[0]
    return apply(*args).moments - current
