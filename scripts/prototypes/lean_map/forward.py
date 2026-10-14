"""Fixed-shape candidate forward map with dense interior and bounded cut rows."""

from dataclasses import dataclass
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
    band_status: jax.Array
    topology_valid: jax.Array
    topology_qualified: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MixedField:
    """Bell jets only in the gathered window, plus cellwise Gauss coefficients."""

    cached: fixed_points._CachedField
    coefficient: jax.Array

    def evaluate(self, point):
        return self.cached.evaluate(point)

    def sampled_coefficient(self, geometry):
        return self.coefficient


def apply(
    moments,
    values,
    jets,
    window,
    rule,
    fit_inverse,
    sample_design,
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
    cached = fixed_points._CachedField(
        window.mesh, window.lookup, window.rows, moments, window.exterior
    )
    roots = fixed_points.solve(
        window,
        moments,
        seeds,
        previous,
        jnp.median(geometry.pitch),
        axis_sign,
    )
    gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
    gauss_coefficient = jnp.einsum("nkq,nq->nk", fit_inverse, gauss_flux)
    slots = window.lookup[jets.sample_index]
    complete = jnp.all(slots >= 0, axis=1)
    safe_slots = jnp.maximum(slots, 0)
    band_values = window.exterior[safe_slots, 0] + jnp.einsum(
        "nscm,cm->ns", window.rows[safe_slots, ..., 0], moments
    )
    band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
    coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
    field = MixedField(cached, coefficient)
    reading = topology.read_at_stationary_points(
        field, geometry, convention, policy, roots
    )
    full = reading.membership >= 1 - 1e-14
    interior = interior_moments(
        rule, gauss_flux, profile, reading.axis_flux, reading.boundary_flux, full
    )
    samples = jnp.where(
        complete[:, None],
        band_values,
        jnp.einsum("nsk,nk->ns", sample_design, coefficient),
    )
    cut = cut_moments(
        samples, reading, geometry, profile, convention.sigma, cut_capacity
    )
    unscaled = interior + cut.moments
    raw_current = jnp.sum(unscaled[:, 0])
    cut_cells = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
    missing_band = jnp.any(cut_cells & ~complete)
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & (cut.status == int(BookingStatus.VALID))
        & ~missing_band
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
        jnp.where(missing_band, 1, 0),
        reading.valid,
        reading.qualified,
    )


def residual(*args):
    """Residual in source moments, preserving typed read and booking refusals."""
    current = args[0]
    return apply(*args).moments - current
