"""Traced array construction of one cold current-moment seed branch.

The limited cold seed is the external conductor field plus the flux image of a
uniform current disc about a declared centroid, and every operation in that
construction is elementwise or a reduction over fixed-shape device arrays.  The
kernels here hold that construction once, in device-array form, so the host
entry point (:meth:`nova.equilibrium.forward.ForwardProfile.cold_seed_portfolio`)
and a catalog ``vmap`` enclosing the solve run the same code rather than two
implementations that agree by inspection.

Host-only work is confined to argument validation of concrete values.  A value
that is being traced carries no Python truth value, so the kernels return the
supported-cell count instead of raising on it and leave the caller to refuse a
disc that is too small once the count is known.

The exact support clip in
:meth:`~nova.equilibrium.forward.ForwardProfile._clip_*` is a host bisection over
a bracket search and is not part of this construction; it returns its input
unchanged in every other clip mode, which is the mode these kernels construct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.moment import UnsupportedSlice
from nova.equilibrium.stencil_mesh import CellCurrentMoments

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardFluxOperator, SaddleSeedGeometry

__all__ = [
    "ColdSeedBranch",
    "current_centroid_seed",
    "is_traced",
    "limiter_radial_extent",
    "limiter_supported_distance",
    "require_concrete",
    "require_supported_disc",
    "saddle_geometry_seed",
    "uniform_disc_current",
]


class ColdSeedBranch(NamedTuple):
    """One constructed branch: its flux, its disc radius and its support size."""

    flux: jax.Array
    radius: jax.Array
    supported_cells: jax.Array


def is_traced(value: object) -> bool:
    """Return whether a value is a trace of a device computation.

    A traced value has no Python truth value, so a validation that depends on
    one raises nowhere and is deferred to the host caller.
    """

    return isinstance(value, jax.core.Tracer)


def require_concrete(condition, message: str) -> None:
    """Raise ``ValueError`` for a concrete failure and defer a traced one."""

    if is_traced(condition):
        return
    if not bool(condition):
        raise ValueError(message)


def require_supported_disc(
    branch: ColdSeedBranch,
    centroid,
    minimum_cells: int,
) -> None:
    """Refuse a concrete disc that supports too few cells to spread its current.

    A traced count has no Python value, so a traced construction cannot be
    refused here; the host caller checks the count it receives instead.
    """

    if is_traced(branch.supported_cells):
        return
    count = int(branch.supported_cells)
    if count < minimum_cells:
        centre = np.asarray(centroid, dtype=np.float64)
        raise UnsupportedSlice(
            "seed-disc-insufficient-supported-cells",
            supported_cell_count=count,
            minimum_cell_count=int(minimum_cells),
            centroid_r_m=float(centre[0]),
            centroid_z_m=float(centre[1]),
            seed_radius_m=float(branch.radius),
        )


def limiter_radial_extent(wall, height) -> tuple[jax.Array, jax.Array]:
    """Return the inboard and outboard limiter radii at one traced height [m].

    The wall is a closed polyline, so the crossings of the horizontal line at
    ``height`` are the segment interpolants whose endpoints straddle it.  A
    height that clears every segment falls back to the polygon's bounding
    radii, which keeps a seed sized for a centroid that has left the vessel.
    """

    coordinate = jnp.asarray(wall, dtype=jnp.float64)
    radius = coordinate[:, 0]
    vertical = coordinate[:, 1]
    next_radius = jnp.roll(radius, -1)
    next_vertical = jnp.roll(vertical, -1)
    spans = ((vertical - height) * (next_vertical - height) <= 0.0) & (
        vertical != next_vertical
    )
    denominator = jnp.where(spans, next_vertical - vertical, 1.0)
    crossing = radius + (height - vertical) / denominator * (next_radius - radius)
    any_crossing = jnp.any(spans)
    crossing_inboard = jnp.min(jnp.where(spans, crossing, jnp.inf))
    crossing_outboard = jnp.max(jnp.where(spans, crossing, -jnp.inf))
    return (
        jnp.where(any_crossing, crossing_inboard, jnp.min(radius)),
        jnp.where(any_crossing, crossing_outboard, jnp.max(radius)),
    )


def limiter_supported_distance(wall, centroid) -> jax.Array:
    """Return the inboard/outboard clearance of a centroid from the limiter [m]."""

    centre = jnp.asarray(centroid, dtype=jnp.float64)
    inboard, outboard = limiter_radial_extent(wall, centre[1])
    return jnp.minimum(centre[0] - inboard, outboard - centre[0])


def uniform_disc_current(
    coordinate,
    material,
    centroid,
    radius,
    plasma_current,
) -> tuple[jax.Array, jax.Array]:
    """Spread a total current evenly over the supported cells of a disc.

    A cell carries current when its centre lies inside the disc and the
    material flag admits it.  The returned count is a device scalar, so a
    caller checks it against its minimum support after the trace rather than
    inside it.
    """

    coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
    centre = jnp.asarray(centroid, dtype=jnp.float64)
    inside = (
        jnp.hypot(coordinate[:, 0] - centre[0], coordinate[:, 1] - centre[1]) < radius
    ) & (jnp.asarray(material) > 0.0)
    supported_cells = jnp.count_nonzero(inside)
    share = jnp.asarray(plasma_current, dtype=jnp.float64) / jnp.maximum(
        supported_cells, 1
    ).astype(jnp.float64)
    return jnp.where(inside, share, 0.0), supported_cells


def current_centroid_seed(
    operator: ForwardFluxOperator,
    centroid,
    plasma_current,
    *,
    radius_fraction: float,
    current=None,
) -> ColdSeedBranch:
    """Return the neutral limited cold seed of a uniform current disc.

    The disc radius is ``radius_fraction`` of the smaller inboard/outboard
    clearance of the centroid from the limiter.  Its zeroth moment is the only
    admissible current moment: the disc carries no radial or vertical moment,
    so the operator's moment coupling is exactly the identity on it.
    """

    coordinate = jnp.asarray(operator.grid.coordinate, dtype=jnp.float64)
    material = jnp.asarray(operator.inside_material, dtype=jnp.float64)
    wall = jnp.asarray(operator.wall.coordinate, dtype=jnp.float64)
    radius = float(radius_fraction) * limiter_supported_distance(wall, centroid)
    cell_current, supported_cells = uniform_disc_current(
        coordinate, material, centroid, radius, plasma_current
    )
    zero = jnp.zeros_like(cell_current)
    moments = CellCurrentMoments(cell_current, zero, zero)
    if operator.moment_geometry is None:
        coefficients = moments
    else:
        coefficients = operator.coupling_current_moments(moments)
    flux = operator.external(current) + operator.current_moment_image(coefficients)
    return ColdSeedBranch(flux=flux, radius=radius, supported_cells=supported_cells)


def saddle_geometry_seed(
    operator: ForwardFluxOperator,
    flux,
    geometry: SaddleSeedGeometry,
) -> jax.Array:
    """Return a cold field normalized at declared axis and saddle points.

    The seed is a cubic whose only stationary points are the declared magnetic
    axis and saddle.  Its axis-to-saddle flux span comes from the independent
    moment seed supplied as ``flux``, never from a reference flux map.
    """

    axis = jnp.asarray(geometry.axis, dtype=jnp.float64)
    saddle = jnp.asarray(geometry.saddle, dtype=jnp.float64)
    displacement = saddle - axis
    distance = jnp.linalg.norm(displacement)
    along = displacement / distance
    across = jnp.asarray((-along[1], along[0]))
    grid = jnp.asarray(operator.grid.coordinate, dtype=jnp.float64)
    parts = [grid, jnp.asarray(operator.wall.coordinate, dtype=jnp.float64)]
    if operator.sample is not None:
        parts.append(jnp.asarray(operator.sample.coordinate, dtype=jnp.float64))
    coordinates = jnp.concatenate(parts)
    axis_index = jnp.argmin(jnp.linalg.norm(grid - axis, axis=1))
    saddle_index = jnp.argmin(jnp.linalg.norm(grid - saddle, axis=1))
    neutral = jnp.asarray(flux, dtype=jnp.float64)
    polarity = float(operator.polarity)
    flux_span = polarity * (neutral[axis_index] - neutral[saddle_index])
    resolution = jnp.finfo(jnp.float64).eps * jnp.maximum(
        jnp.max(jnp.abs(neutral)), 1.0
    )
    flux_span = jnp.maximum(flux_span, resolution)
    local = coordinates - axis
    axial = local @ along / distance
    transverse = local @ across / distance
    potential = -0.5 * axial**2 + axial**3 / 3.0 - 0.5 * transverse**2
    normalized = 6.0 * (potential + 1.0 / 6.0)
    return neutral[saddle_index] + polarity * flux_span * normalized
