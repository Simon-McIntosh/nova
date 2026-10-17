"""Book the saddle lobe over the normal-form rings with one shared integral.

The region is the intended owner region: the cell clipped against the
separatrix rays the read carries as ``fragment_selected`` and ``edge_interval``,
so no flux level root bounds it and the curvature and support terms of the
normal form hold the boundary on the separatrix across a whole cell. The rings
arrive as a signed fan, so opposing edges cancel and no polygon union is needed.

Area and moments come from the same fan, so the read's membership and the booked
current share one integral at every carrier size. The rule is a degree-five
triangle rule on the fan's triangles, so the fitted density quadratic and the
displacement moments enter exactly and the profile representation is limited
only by the profile itself.

The remainder of an owner cell, the cell with the lobe removed, is the same
integral: the cell's own polygon fan enters positive and the lobe fan enters
negative, and one call to the same rule evaluates both on the same points.
"""

import jax
import jax.numpy as jnp

from nova.equilibrium import current

# Degree-five rule on the reference triangle; the weights sum to one.
_DUNAVANT = jnp.asarray(
    (
        (1 / 3, 1 / 3, 0.225),
        (0.0597158717, 0.4701420641, 0.1323941527),
        (0.4701420641, 0.0597158717, 0.1323941527),
        (0.4701420641, 0.4701420641, 0.1323941527),
        (0.7974269853, 0.1012865073, 0.1259391805),
        (0.1012865073, 0.7974269853, 0.1259391805),
        (0.1012865073, 0.1012865073, 0.1259391805),
    )
)


def _quadratic(coefficients, x, y):
    return (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
    )


def _fan_moments(rings, sizes, signs, origin, density_at):
    """One degree-five rule over every triangle of a signed fan.

    ``rings`` carries one row per (fragment, edge) interval, each row a ring of
    points about ``origin``; a row beyond its own size is masked out, and
    ``signs`` lets a fan enter negatively so a region can be removed from
    another without a polygon union.
    """
    weights = _DUNAVANT[:, 2]
    fractions = _DUNAVANT[:, :2]
    slot = jnp.arange(rings.shape[1])
    following = jnp.where(slot[None] + 1 < sizes[:, None], slot[None] + 1, 0)
    end = jnp.take_along_axis(rings, following[..., None], axis=1)
    first = rings - origin
    second = end - origin
    area = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
    live = (slot[None] < sizes[:, None]).astype(area.dtype)
    area = area * live
    points = (
        origin
        + fractions[:, 0, None] * first[..., None, :]
        + fractions[:, 1, None] * second[..., None, :]
    )
    weight = area[..., None] * signs[:, None, None] * weights[None, :]
    weighted = weight * density_at(points)
    displacement = points - origin
    return jnp.stack(
        (
            jnp.sum(weight * jnp.ones_like(weighted)),
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


def _cell_rings(geometry, index, segments):
    """The cell's own polygon as a centre fan, one row per subdivided edge."""
    count = int(geometry.vertex_count[index])
    vertices = geometry.vertices[index, :count]
    following = jnp.concatenate((vertices[1:], vertices[:1]))
    fractions = jnp.linspace(0.0, 1.0, segments + 1)
    edge = (
        vertices[None, :, :]
        + fractions[:, None, None] * (following - vertices)[None, :, :]
    )
    return jnp.transpose(edge, (1, 0, 2)), jnp.full((count,), segments + 1)


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
        return _fan_moments(
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
        cell_rings, cell_sizes = _cell_rings(geometry, index, segments)
        lobe_rings, lobe_sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            origin,
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            segments,
        )
        width = max(cell_rings.shape[1], lobe_rings.shape[1])
        if lobe_rings.shape[1] < width:
            padding = jnp.zeros(
                (lobe_rings.shape[0], width - lobe_rings.shape[1], 2),
                lobe_rings.dtype,
            )
            lobe_rings = jnp.concatenate((lobe_rings, padding), axis=1)
        if cell_rings.shape[1] < width:
            padding = jnp.zeros(
                (cell_rings.shape[0], width - cell_rings.shape[1], 2),
                cell_rings.dtype,
            )
            cell_rings = jnp.concatenate((cell_rings, padding), axis=1)
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
        return _fan_moments(rings, sizes, signs, origin, density_at)

    integrals = jax.vmap(one)(indices)
    valid = jnp.all(jnp.isfinite(integrals), axis=-1)
    return integrals, valid
