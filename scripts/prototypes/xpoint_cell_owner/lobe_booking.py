"""Book the saddle lobe over the normal-form rings with one shared integral.

The region is the intended owner region: the cell clipped against the
separatrix rays the read carries as ``fragment_selected`` and ``edge_interval``,
so no flux level root bounds it and the curvature and cubic terms of the normal
form hold the boundary on the separatrix across a whole cell. The rings arrive
as a signed fan, so opposing edges cancel and no polygon union is needed.

Area and moments come from the same fan, so the read's membership and the booked
current share one integral at every carrier size. The rule is a degree-five
triangle rule on the fan's triangles, so the fitted density quadratic and the
displacement moments enter exactly and the profile representation is limited
only by the profile itself.
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
    weights = _DUNAVANT[:, 2]
    fractions = _DUNAVANT[:, :2]

    def one(index):
        origin = geometry.centre[index]
        pitch = geometry.pitch[index]
        rings, sizes = current._normal_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            origin,
            reading.saddle_form,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            segments,
        )
        slot = jnp.arange(rings.shape[1])
        following = jnp.where(slot[None] + 1 < sizes[:, None], slot[None] + 1, 0)
        end = jnp.take_along_axis(rings, following[..., None], axis=1)
        first = rings - origin
        second = end - origin
        area = 0.5 * (
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        )
        area = jnp.where(slot[None] < sizes[:, None], area, 0.0)
        points = (
            origin
            + fractions[:, 0, None] * first[..., None, :]
            + fractions[:, 1, None] * second[..., None, :]
        )
        local = (points - origin) / pitch
        x, y = local[..., 0], local[..., 1]
        if density_mode == "profile":
            field = (
                reading.field_coefficients[index].at[0].add(-reading.boundary_flux)
                * sigma
            )
            normalized = 1.0 + _quadratic(field, x, y) / (sigma * span)
            rho = profile.current_density(points[..., 0], normalized)
        else:
            rho = _quadratic(density[index], x, y)
        weighted = area[..., None] * weights[None, :] * rho
        displacement = points - origin
        return jnp.stack(
            (
                jnp.sum(area[..., None] * weights[None, :] * jnp.ones_like(rho)),
                jnp.sum(weighted),
                jnp.sum(weighted * displacement[..., 0]),
                jnp.sum(weighted * displacement[..., 1]),
            )
        )

    integrals = jax.vmap(one)(indices)
    valid = jnp.all(jnp.isfinite(integrals), axis=-1)
    return integrals, valid