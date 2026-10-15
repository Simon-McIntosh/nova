"""Fixed-shape connected moment map and device Newton correction."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium import current, fixed_points
from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from scripts.prototypes.lean_map.quadrature import interior_moments
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fast_read.kernel import read


class MapState(NamedTuple):
    moments: jax.Array
    moment_residual: jax.Array
    constraint_rows: jax.Array
    roots: jax.Array
    boundary_class: jax.Array
    boundary_flux: jax.Array
    topology_valid: jax.Array
    booking_valid: jax.Array
    graph_cells: jax.Array
    saddle_cells: jax.Array


class Correction(NamedTuple):
    moments: jax.Array
    residual: jax.Array
    converged: jax.Array
    linear_residual: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FusedField:
    cached: fixed_points._CachedField
    coefficient: jax.Array

    def evaluate(self, point):
        return self.cached.evaluate(point)

    def sampled_coefficient(self, geometry):
        return self.coefficient


def _density_coefficients(geometry, reading, profile):
    """Fit the same quadratic density used by ordinary Bell polygons."""
    centre, pitch = geometry.centre, geometry.pitch
    polynomial = reading.field_coefficients
    span = reading.boundary_flux - reading.axis_flux

    def value(x, y):
        flux = (
            polynomial[:, 0]
            + polynomial[:, 1] * x
            + polynomial[:, 2] * y
            + polynomial[:, 3] * x * x
            + polynomial[:, 4] * x * y
            + polynomial[:, 5] * y * y
        )
        return profile.current_density(
            centre[:, 0] + pitch * x, (flux - reading.axis_flux) / span
        )

    origin = value(0, 0)
    radial_positive, radial_negative = value(1, 0), value(-1, 0)
    vertical_positive, vertical_negative = value(0, 1), value(0, -1)
    radial = (radial_positive - radial_negative) / 2
    vertical = (vertical_positive - vertical_negative) / 2
    return jnp.stack(
        (
            origin,
            radial,
            vertical,
            (radial_positive + radial_negative) / 2 - origin,
            value(1, 1) - radial_positive - vertical_positive + origin,
            (vertical_positive + vertical_negative) / 2 - origin,
        ),
        axis=1,
    )


def _saddle_moments(geometry, reading, profile, capacity=8, ordinary_valid=None):
    """Book selected saddle sectors with four chords per cubic ray."""
    active = reading.normal_form_cells & (reading.membership > 0)
    if ordinary_valid is not None:
        active = active & ~ordinary_valid
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    coefficients = _density_coefficients(geometry, reading, profile)

    def one(index):
        rings, sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            4,
        )
        origin = geometry.centre[index]
        pitch = jnp.broadcast_to(geometry.pitch[index], (2,))
        values, first = padded_polynomial_current_moments(
            rings,
            sizes,
            jnp.broadcast_to(origin, (len(rings), 2)),
            jnp.broadcast_to(pitch, (len(rings), 2)),
            jnp.broadcast_to(coefficients[index], (len(rings), 6)),
        )
        return jnp.stack((jnp.sum(values), *jnp.sum(first, axis=0)), axis=0)

    packed = jax.vmap(one)(indices)
    live = jnp.arange(capacity) < count
    scattered = (
        jnp.zeros((len(geometry.centre), 3), dtype=packed.dtype)
        .at[indices]
        .add(jnp.where(live[:, None], packed, 0.0))
    )
    return scattered, count <= capacity


def map_moments(
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
    convention,
    policy,
    target_current,
    axis_sign,
    *,
    frame_gate=True,
):
    """Return the connected three-moment image and read qualifications."""
    cached = fixed_points._CachedField(
        window.mesh, window.lookup, window.rows, moments, window.exterior
    )
    roots = fixed_points.solve(
        window, moments, seeds, seeds, jnp.median(geometry.pitch), axis_sign
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
    field = FusedField(cached, coefficient)
    reading = read(field, roots, geometry, convention, policy, frame_gate=frame_gate)
    selected = reading.selected_cells
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    interior = interior_moments(
        rule, gauss_flux, profile, reading.axis_flux, reading.boundary_flux, full
    )
    ordinary, ordinary_valid = bell_moments(
        geometry, reading, profile, convention.sigma
    )
    cut = (reading.membership > 0) & ~full
    saddle_cut = cut & reading.normal_form_cells & ~ordinary_valid
    ordinary_cut = cut & ~saddle_cut
    saddle, saddle_valid = _saddle_moments(
        geometry, reading, profile, ordinary_valid=ordinary_valid
    )
    raw = interior + jnp.where(ordinary_cut[:, None], ordinary, 0.0) + saddle
    total = jnp.sum(raw[:, 0])
    missing = jnp.any(cut & ~complete)
    valid = (
        jnp.all(roots.valid)
        & reading.valid
        & reading.qualified
        & jnp.all(~ordinary_cut | ordinary_valid)
        & saddle_valid
        & ~missing
        & jnp.isfinite(total)
        & (total != 0)
    )
    normalized = raw * (target_current / jnp.where(total != 0, total, 1))
    normalized = jnp.where(valid, normalized, jnp.nan)
    constraint_scale = jnp.maximum(jnp.abs(target_current), 1.0)
    current_row = (jnp.sum(normalized[:, 0]) - target_current) / constraint_scale
    return MapState(
        normalized,
        moments - normalized,
        jnp.atleast_1d(current_row),
        roots.position,
        reading.boundary_class,
        reading.boundary_flux,
        reading.valid & reading.qualified,
        valid,
        jnp.sum(selected),
        jnp.sum(saddle_cut),
    )


def warm_slice(start, scale, *map_operands):
    """Run four device JVP Krylov products and check the corrected map."""

    def equation(scaled):
        varied = scaled * scale
        return (map_moments(varied, *map_operands).moment_residual / scale).reshape(-1)

    scaled = start / scale
    residual, factor = jax.linearize(equation, scaled)
    rhs = -residual
    length = jnp.linalg.norm(rhs)
    basis = (
        jnp.zeros((5, rhs.size), dtype=rhs.dtype)
        .at[0]
        .set(rhs / jnp.where(length > 0, length, 1))
    )
    hessenberg = jnp.zeros((5, 4), dtype=rhs.dtype)
    for column in range(4):
        vector = factor(basis[column].reshape(start.shape))
        projections = jnp.einsum("ij,j->i", basis, vector)
        occupied = jnp.arange(5) <= column
        projection = jnp.where(occupied, projections, 0)
        vector = vector - jnp.einsum("i,ij->j", projection, basis)
        next_length = jnp.linalg.norm(vector)
        basis = basis.at[column + 1].set(
            vector / jnp.where(next_length > 0, next_length, 1)
        )
        hessenberg = hessenberg.at[:, column].set(
            projection.at[column + 1].set(next_length)
        )
    target = jnp.zeros((5,), dtype=rhs.dtype).at[0].set(length)
    coefficients = jnp.linalg.lstsq(hessenberg, target, rcond=None)[0]
    step = jnp.einsum("i,ij->j", coefficients, basis[:4]).reshape(start.shape)
    candidate = (scaled + step) * scale
    candidate_residual = jnp.max(jnp.abs(equation(candidate / scale)))
    initial_residual = jnp.max(jnp.abs(residual))
    accepted = jnp.isfinite(candidate_residual) & (
        candidate_residual < initial_residual
    )
    state = jnp.where(accepted, candidate, start)
    final_residual = jnp.where(accepted, candidate_residual, initial_residual)
    linear_residual = jnp.linalg.norm(hessenberg @ coefficients - target)
    return Correction(
        state,
        final_residual,
        jnp.isfinite(final_residual) & (final_residual <= 1e-12),
        linear_residual,
    )
