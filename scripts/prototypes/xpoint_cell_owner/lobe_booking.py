"""Book the saddle lobe over the normal-form rings with one shared integral.

The region is the intended owner region: the cell clipped against the
separatrix rays the read carries as ``fragment_selected`` and ``edge_interval``,
so no flux level root bounds it and the curvature and support terms of the
normal form hold the boundary on the separatrix across a whole cell. The rings
arrive as a signed fan, so opposing edges cancel and no polygon union is needed.

Area and moments come from the same fan, so the read's membership and the booked
current share one integral at every carrier size. The quadrature and the
triangle reduction are the clip-quadrature owner's own
(``clip_quadrature._quadrature_from_arrays`` with ``signed=True``), so the rule
that books an owner cell here is the rule that books one there, and a signed row
lets a region be removed from another without a polygon union.

The remainder of an owner cell, the cell with the lobe removed, is the same
integral over the same builder: the cell polygon enters as the star fan
``current._normal_rings`` emits for a fully selected cell, the lobe fan enters
negative, and one call to the owner's rule evaluates both.
"""

import jax
import jax.numpy as jnp

from nova.equilibrium import current
from nova.equilibrium.clip_quadrature import _quadrature_from_arrays


def _quadratic(coefficients, x, y):
    return (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
    )


def _ring_moments(rings, sizes, signs, origin, density_at):
    """Area, current and both first moments of a signed ring fan.

    The triangles and their quadrature are the owner's: one ring per row, the
    row's leading vertex as the fan apex, a row with fewer than three live
    vertices contributing nothing. ``signs`` lets a fan enter negatively, and
    the density is evaluated only at points the owner already emitted, so no
    quadrature is restated here.
    """
    counts = jnp.asarray(sizes)
    selection = jnp.ones(rings.shape[0], dtype=bool)
    points, weights = _quadrature_from_arrays(
        rings, counts, rings[:, 0, :], selection, signed=True
    )
    weights = weights * signs[:, None]
    weighted = weights * density_at(points)
    displacement = points - origin
    return jnp.stack(
        (
            jnp.sum(weights),
            jnp.sum(weighted),
            jnp.sum(weighted * displacement[..., 0]),
            jnp.sum(weighted * displacement[..., 1]),
        )
    )


def _density_of(geometry, reading, profile, index, density, span, sigma, density_mode):
    """A callable evaluating the booked density at points in the cell's frame."""
    origin = geometry.centre[index]
    pitch = geometry.pitch[index]

    def density_at(points):
        local = (points - origin) / pitch
        x, y = local[..., 0], local[..., 1]
        if density_mode == "profile":
            field = (
                reading.field_coefficients[index].at[0].add(-reading.boundary_flux)
                * sigma
            )
            normalized = 1.0 + _quadratic(field, x, y) / (sigma * span)
            return profile.current_density(points[..., 0], normalized)
        return _quadratic(density[index], x, y)

    return density_at


def _whole_cell_rings(geometry, reading, index, segments):
    """The owner cell's own polygon as the star fan the ring builder emits.

    Every fragment slot is selected and one identity interval is placed on the
    first strip of every edge, so ``current._normal_rings`` emits the cell's
    straight-edge star fan. The separatrix-arc rows follow the straight rows in
    that return value and are not part of the cell polygon, so they are dropped
    by the static straight-row count. Both fans therefore come from the same
    builder the lobe uses, and no vertex list is walked here.
    """
    interval = reading.edge_interval[index]
    identity = jnp.zeros_like(interval)
    identity = identity.at[..., 0, 0].set(0.0)
    identity = identity.at[..., 0, 1].set(1.0)
    selected = jnp.ones(reading.fragment_selected.shape[1], dtype=bool)
    rings, sizes = current._normal_rings(
        geometry.vertices[index],
        geometry.vertex_count[index],
        geometry.centre[index],
        reading.saddle_form,
        selected,
        identity,
        segments,
    )
    straight = (
        reading.fragment_selected.shape[1]
        * geometry.vertices.shape[1]
        * interval.shape[-2]
    )
    return rings[:straight], sizes[:straight]


def lobe_moments(
    geometry,
    reading,
    profile,
    indices,
    density,
    span,
    sigma,
    density_mode="fitted",
    segments=4,
):
    """Return the lobe's area, current and two first moments for each index.

    Columns are the traced integral's: area in square metres, then current and
    the two first current moments about the cell centre, so the booking is
    interchangeable with the other owner integrals.
    """

    def one(index):
        origin = geometry.centre[index]
        rings, sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            origin,
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            segments,
        )
        density_at = _density_of(
            geometry, reading, profile, index, density, span, sigma, density_mode
        )
        return _ring_moments(
            rings, sizes, jnp.ones(sizes.shape[0], rings.dtype), origin, density_at
        )

    integrals = jax.vmap(one)(indices)
    valid = jnp.all(jnp.isfinite(integrals), axis=-1)
    return integrals, valid


def remaining_moments(
    geometry,
    reading,
    profile,
    indices,
    density,
    span,
    sigma,
    density_mode="fitted",
    segments=4,
):
    """Return each owner cell's remainder: the cell fan less the lobe fan.

    The cell's own polygon enters positive and the lobe enters negative in one
    signed fan, so the two sides of the partition are evaluated by one call to
    one rule on the same points and their sum is the cell integral exactly, up
    to the fan's own chord approximation of the separatrix rays.
    """

    def one(index):
        origin = geometry.centre[index]
        cell_rings, cell_sizes = _whole_cell_rings(geometry, reading, index, segments)
        lobe_rings, lobe_sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            origin,
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            segments,
        )
        rings = jnp.concatenate((cell_rings, lobe_rings), axis=0)
        sizes = jnp.concatenate((cell_sizes, lobe_sizes), axis=0)
        signs = jnp.concatenate(
            (
                jnp.ones(cell_sizes.shape[0], cell_rings.dtype),
                -jnp.ones(lobe_sizes.shape[0], lobe_rings.dtype),
            )
        )
        density_at = _density_of(
            geometry, reading, profile, index, density, span, sigma, density_mode
        )
        return _ring_moments(rings, sizes, signs, origin, density_at)

    integrals = jax.vmap(one)(indices)
    valid = jnp.all(jnp.isfinite(integrals), axis=-1)
    return integrals, valid
