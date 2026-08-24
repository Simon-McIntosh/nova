"""Fixed-shape second-moment flux columns for polygonal source cells.

The first three columns are the area means of the ring flux kernel weighted by
``1``, ``R - R_c`` and ``Z - Z_c``.  The remaining columns use the quadratic
weights ``(R - R_c)^2``, ``(R - R_c)(Z - Z_c)`` and ``(Z - Z_c)^2``.  A static
tensor Gauss rule on a triangle fan keeps every cell at six columns without a
data-dependent cut-cell path.

The quadratic weighted rows are deliberately identified as quadrature rows.
The closed Part V reduction currently exposes exact constant and linear
weighted edge antiderivatives, but no quadratic endpoint primitives.  This
module does not silently label a fixed quadrature as a closed form.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from nova.biot.greens import traced_filament_greens

__all__ = [
    "flux_density_columns",
    "monopole_taylor_columns",
    "taylor_displaced_flux",
]


@lru_cache(maxsize=8)
def _unit_interval_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a Gauss-Legendre rule mapped to the unit interval."""
    if order < 2:
        raise ValueError("quadrature order must be at least two")
    node, weight = np.polynomial.legendre.leggauss(order)
    return 0.5 * (node + 1.0), 0.5 * weight


def _area_centroid(xp, vertices):
    """Return the polygon area centroid using a traced shoelace reduction."""
    following = xp.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    area_twice = xp.sum(cross)
    return xp.stack(
        (
            xp.sum((vertices[:, 0] + following[:, 0]) * cross) / (3.0 * area_twice),
            xp.sum((vertices[:, 1] + following[:, 1]) * cross) / (3.0 * area_twice),
        )
    )


def _triangle_fan(xp, vertices):
    """Triangulate a convex cell without changing its static vertex shape."""
    return xp.stack(
        (
            xp.broadcast_to(vertices[0], (vertices.shape[0] - 2, 2)),
            vertices[1:-1],
            vertices[2:],
        ),
        axis=1,
    )


def flux_density_columns(
    xp,
    target_r,
    target_z,
    vertices,
    *,
    expansion_point=None,
    order: int = 8,
    columns: int = 6,
):
    """Return six area-normalised flux columns for one convex source cell.

    The result shape is ``broadcast(target_r, target_z).shape + (6,)``.  The
    tensor-product Duffy rule has a compile-time order and the triangle count is
    fixed by ``vertices.shape[0]``, so the same function can be used under
    :func:`jax.jit` and :func:`jax.vmap`.
    """
    if columns not in (3, 6):
        raise ValueError("columns must select the linear three or quadratic six")
    vertices = xp.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        raise ValueError("vertices must have fixed shape (corners >= 3, 2)")
    centre = (
        _area_centroid(xp, vertices)
        if expansion_point is None
        else xp.asarray(expansion_point)
    )
    if centre.shape != (2,):
        raise ValueError("expansion_point must have shape (2,)")

    unit_node, unit_weight = _unit_interval_rule(order)
    first = xp.asarray(unit_node)[:, None]
    second = xp.asarray(unit_node)[None, :]
    first_weight = xp.asarray(unit_weight)[:, None]
    second_weight = xp.asarray(unit_weight)[None, :]

    triangles = _triangle_fan(xp, vertices)
    origin = triangles[:, 0]
    radial_edge = triangles[:, 1] - origin
    vertical_edge = triangles[:, 2] - origin
    point = (
        origin[:, None, None, :]
        + first[None, ..., None] * radial_edge[:, None, None, :]
        + (1.0 - first)[None, ..., None]
        * second[None, ..., None]
        * vertical_edge[:, None, None, :]
    )
    determinant = xp.abs(
        radial_edge[:, 0] * vertical_edge[:, 1]
        - radial_edge[:, 1] * vertical_edge[:, 0]
    )
    area_weight = (
        determinant[:, None, None]
        * (1.0 - first)[None, ...]
        * first_weight[None, ...]
        * second_weight[None, ...]
    ).reshape(-1)
    point = point.reshape(-1, 2)
    local = point - centre
    basis = xp.stack(
        (
            xp.ones_like(local[:, 0]),
            local[:, 0],
            local[:, 1],
            local[:, 0] ** 2,
            local[:, 0] * local[:, 1],
            local[:, 1] ** 2,
        ),
        axis=1,
    )[:, :columns]

    target_r, target_z = xp.broadcast_arrays(xp.asarray(target_r), xp.asarray(target_z))
    kernel = traced_filament_greens(
        xp,
        target_r[..., None],
        target_z[..., None],
        point[:, 0],
        point[:, 1],
    )[0]
    area = xp.sum(area_weight)
    return xp.einsum("...q,qc,q->...c", kernel, basis, area_weight) / area


def monopole_taylor_columns(target_r, target_z, source_centre):
    """Return value, gradient and Hessian columns of a source monopole.

    The six columns are ``(G, G_R, G_Z, G_RR, G_RZ, G_ZZ)`` and are obtained by
    automatic differentiation of the same traced closed-form filament kernel
    used for the direct displaced evaluation.
    """
    import jax
    import jax.numpy as jnp

    target_r, target_z = jnp.broadcast_arrays(
        jnp.asarray(target_r), jnp.asarray(target_z)
    )
    source_centre = jnp.asarray(source_centre)
    if source_centre.shape != (2,):
        raise ValueError("source_centre must have shape (2,)")

    def one_target(one_r, one_z):
        def coupling(source):
            return traced_filament_greens(jnp, one_r, one_z, source[0], source[1])[0]

        value, gradient = jax.value_and_grad(coupling)(source_centre)
        hessian = jax.hessian(coupling)(source_centre)
        return jnp.stack(
            (
                value,
                gradient[0],
                gradient[1],
                hessian[0, 0],
                hessian[0, 1],
                hessian[1, 1],
            )
        )

    flat = jax.vmap(one_target)(target_r.reshape(-1), target_z.reshape(-1))
    return flat.reshape(target_r.shape + (6,))


def taylor_displaced_flux(columns, displacement):
    """Contract monopole columns through quadratic order in displacement."""
    namespace = getattr(columns, "__array_namespace__", lambda: np)()
    columns = namespace.asarray(columns)
    displacement = namespace.asarray(displacement)
    if columns.shape[-1] != 6 or displacement.shape != (2,):
        raise ValueError("six columns and one (R, Z) displacement are required")
    radial, vertical = displacement
    return (
        columns[..., 0]
        + radial * columns[..., 1]
        + vertical * columns[..., 2]
        + 0.5
        * (
            radial**2 * columns[..., 3]
            + 2.0 * radial * vertical * columns[..., 4]
            + vertical**2 * columns[..., 5]
        )
    )
