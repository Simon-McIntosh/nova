"""Fixed-shape interior Gauss points and bounded exact cut-cell booking."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InteriorRule:
    points: jax.Array
    weights: jax.Array
    centre: jax.Array


class BookingStatus(IntEnum):
    VALID = 0
    NEEDS_CUT_ROWS = 1
    INVALID = 2


class LeanBooking(NamedTuple):
    moments: jax.Array
    cut_count: jax.Array
    status: jax.Array


def triangle_rule(geometry) -> InteriorRule:
    """Three-point Gaussian triangle fan of every polygon, padded to one shape."""
    centre = np.asarray(geometry.centre)
    vertices = np.asarray(geometry.vertices)
    count = np.asarray(geometry.vertex_count)
    slots = np.arange(vertices.shape[1])
    following = np.where(slots[None] + 1 < count[:, None], slots[None] + 1, 0)
    ends = np.take_along_axis(vertices, following[..., None], axis=1)
    first = vertices - centre[:, None]
    second = ends - centre[:, None]
    area = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
    area = np.where(slots[None] < count[:, None], area, 0.0)
    if np.any(area < -1e-13):
        raise ValueError("polygon fan must be counterclockwise")
    barycentric = np.asarray(((2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 6, 1 / 6)))
    points = (
        centre[:, None, None]
        + barycentric[None, None, :, 0, None] * first[:, :, None]
        + barycentric[None, None, :, 1, None] * second[:, :, None]
    ).reshape(len(centre), -1, 2)
    weights = np.repeat(area / 3, 3, axis=1)
    points = np.where(weights[..., None] > 0, points, centre[:, None])
    np.testing.assert_allclose(
        weights.sum(axis=1), np.asarray(geometry.full_area), rtol=1e-10, atol=1e-13
    )
    np.testing.assert_allclose(
        np.einsum("nq,nqd->nd", weights, points - centre[:, None]),
        0,
        rtol=0,
        atol=1e-11,
    )
    return InteriorRule(jnp.asarray(points), jnp.asarray(weights), jnp.asarray(centre))


def interior_moments(rule, flux, profile, axis_flux, boundary_flux, full):
    """Book density and first moments on full cells only."""
    normalized = (flux - axis_flux) / (boundary_flux - axis_flux)
    normalized = jnp.where(full[:, None], normalized, 0.5)
    density = profile.current_density(rule.points[..., 0], normalized)
    current_weight = rule.weights * density * full[:, None]
    displacement = rule.points - rule.centre[:, None]
    return jnp.stack(
        (
            jnp.sum(current_weight, axis=1),
            jnp.sum(current_weight * displacement[..., 0], axis=1),
            jnp.sum(current_weight * displacement[..., 1], axis=1),
        ),
        axis=-1,
    )


def cut_moments(psi, reading, geometry, profile, sigma, capacity):
    """Gather one fixed cut batch and refuse any state exceeding its capacity."""
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    count = jnp.sum(cut, dtype=jnp.int32)
    indices = jnp.nonzero(cut, size=capacity, fill_value=0)[0]
    active = jnp.arange(capacity) < count
    geometry_names = (
        "vertices",
        "vertex_count",
        "centre",
        "pitch",
        "sample_points",
        "fit_inverse",
        "full_area",
    )
    geometry_updates = {
        name: getattr(geometry, name)[indices] for name in geometry_names
    }
    compact_geometry = replace(geometry, **geometry_updates)
    read_names = (
        "membership",
        "fragment_area",
        "fragment_component",
        "fragment_selected",
        "edge_interval",
        "field_coefficients",
        "normal_form_cells",
    )
    read_updates = {name: getattr(reading, name)[indices] for name in read_names}
    compact_read = reading._replace(**read_updates)
    compact_read = compact_read._replace(
        membership=jnp.where(active, compact_read.membership, 0.0)
    )
    booked = current.book(
        psi[indices],
        compact_read,
        compact_geometry,
        profile,
        current.Quadrature(sigma=sigma),
    )
    packed = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    scatter = (
        jnp.zeros((len(psi), 3), dtype=psi.dtype)
        .at[indices]
        .add(jnp.where(active[:, None], packed, 0.0))
    )
    status = jnp.where(
        count > capacity,
        int(BookingStatus.NEEDS_CUT_ROWS),
        jnp.where(booked.valid, int(BookingStatus.VALID), int(BookingStatus.INVALID)),
    )
    return LeanBooking(scatter, count, status)
