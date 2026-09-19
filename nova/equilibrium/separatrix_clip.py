"""Conservative polygonal supports cut by a moving separatrix.

The fixed cell mesh is atomised once: every endpoint lying on another cell's
edge splits that edge, so neighbouring cells address the same smallest shared
segment.  A flux crossing is then interpolated once per undirected atomic edge
and reused byte-for-byte by both cells.  This is essential on staggered meshes,
where one long edge otherwise meets two shorter edges at a junction.

The moving result is padded to capacities fixed by the atomised mesh.  Besides
the clipped vertices it carries area, first area moments and second area
moments about each cell's fixed geometric centroid.  These are sufficient to
form the zeroth and first moments of any cellwise-linear current density with
no quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AtomicCellMesh",
    "ClippedSupports",
    "LinearCurrentMoments",
    "SaddleCellWedges",
    "TracedClippedSupports",
    "complete_polynomial_powers",
    "padded_linear_current_moments",
    "padded_polynomial_current_moments",
]


def complete_polynomial_powers(degree: int) -> tuple[tuple[int, int], ...]:
    """Return a complete two-dimensional monomial basis through ``degree``."""
    if degree < 0:
        raise ValueError("polynomial degree must be non-negative")
    return tuple(
        (radial, total - radial)
        for total in range(degree + 1)
        for radial in range(total, -1, -1)
    )


POLYNOMIAL_POWERS = complete_polynomial_powers(3)

_CURVED_BOUNDARY_SEGMENTS = 512
"""Fixed chord count used to carry a traced quadratic level-set arc."""

_SPLINE_BOUNDARY_SEGMENTS = 128
"""Fixed sample count used by each spline boundary chain."""

_MINIMUM_TRACED_POLYGON_VERTICES = 3
"""Fewest vertices a traced polygon can enclose area with."""


def traced_polygon_vertex_capacity(straight_vertex_capacity: int) -> int:
    """Return the maximal live vertex count of one traced clipped polygon.

    A clipped polygon is its traced level arc joined to a straight chain of
    cell-boundary edges. The arc enters the polygon as its fixed
    ``_SPLINE_BOUNDARY_SEGMENTS`` samples, and every other vertex is a straight
    cell-boundary vertex: the compact traced polygon from which the arc is
    expanded carries at most ``straight_vertex_capacity`` vertices, and the arc
    replaces one of them. One arc plus that straight chain is the realised
    layout, so the capacity is their sum; counting the arc's slot once as a
    straight vertex as well only widens the bound. Reserving the straight
    sample count for every slot instead, as the expansion once did, multiplies
    the capacity by the arc sample count for a layout that spends it on one
    arc and a handful of straight edges.
    """
    if straight_vertex_capacity < _MINIMUM_TRACED_POLYGON_VERTICES:
        raise ValueError("a traced polygon carries at least three vertices")
    return _SPLINE_BOUNDARY_SEGMENTS + int(straight_vertex_capacity)


class TracedCapacityRefusalError(RuntimeError):
    """Raised when a clipped layout exceeds the derived traced-polygon capacity.

    The refusal is carried rather than dropped: a cell whose live vertex count
    passes :func:`traced_polygon_vertex_capacity` cannot be packed into the
    fixed shape, so :meth:`TracedClippedSupports.assert_no_refusal` makes the
    count a failure the solve can surface instead of an included flag that
    reads as an empty cell.
    """


def _signed_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    following = np.roll(vertices, -1, axis=0)
    return 0.5 * math.fsum(
        vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    )


def _centroid(vertices: np.ndarray) -> np.ndarray:
    """Return a polygon centroid without subtracting nearby large coordinates."""
    reference = np.mean(vertices, axis=0)
    local = vertices - reference
    following = np.roll(local, -1, axis=0)
    cross = local[:, 0] * following[:, 1] - following[:, 0] * local[:, 1]
    area_twice = math.fsum(cross)
    if area_twice == 0.0:
        raise ValueError("a cell polygon must have non-zero area")
    return reference + np.asarray(
        [
            math.fsum((local[:, 0] + following[:, 0]) * cross),
            math.fsum((local[:, 1] + following[:, 1]) * cross),
        ]
    ) / (3.0 * area_twice)


def _area_moments(
    vertices: np.ndarray, origin: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return area and first two area moments about ``origin``."""
    if len(vertices) < 3:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    local = vertices - origin
    following = np.roll(local, -1, axis=0)
    x, y = local.T
    next_x, next_y = following.T
    cross = x * next_y - next_x * y
    area_twice = math.fsum(cross)
    if area_twice == 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    orientation = math.copysign(1.0, area_twice)
    area = 0.5 * orientation * area_twice
    first = orientation * np.asarray(
        [
            math.fsum((x + next_x) * cross) / 6.0,
            math.fsum((y + next_y) * cross) / 6.0,
        ]
    )
    radial_squared = (
        orientation * math.fsum((x * x + x * next_x + next_x * next_x) * cross) / 12.0
    )
    vertical_squared = (
        orientation * math.fsum((y * y + y * next_y + next_y * next_y) * cross) / 12.0
    )
    cross_moment = (
        orientation
        * math.fsum(
            (2.0 * x * y + x * next_y + next_x * y + 2.0 * next_x * next_y) * cross
        )
        / 24.0
    )
    second = np.asarray(
        [[radial_squared, cross_moment], [cross_moment, vertical_squared]]
    )
    return area, first, second


def _line_key(start: np.ndarray, end: np.ndarray) -> tuple[float, float, float]:
    """Return an orientation-independent key for one supporting line."""
    direction = end - start
    direction /= np.linalg.norm(direction)
    if direction[0] < -1.0e-14 or (abs(direction[0]) <= 1.0e-14 and direction[1] < 0.0):
        direction = -direction
    normal = np.asarray([-direction[1], direction[0]])
    offset = float(normal @ start)
    return tuple(np.round([direction[0], direction[1], offset], 12))


def _point_key(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    return tuple(np.rint(point / tolerance).astype(np.int64))


def _normalise_cells(cells: Iterable[np.ndarray]) -> list[np.ndarray]:
    normalised = []
    for cell in cells:
        vertices = np.ascontiguousarray(cell, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError(
                "each cell must have shape (vertices, 2) with at least three vertices"
            )
        if not np.all(np.isfinite(vertices)):
            raise ValueError("cell vertices must be finite")
        if np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        if _signed_area(vertices) == 0.0:
            raise ValueError("a cell polygon must have non-zero area")
        if _signed_area(vertices) < 0.0:
            vertices = vertices[::-1]
        normalised.append(vertices)
    if not normalised:
        raise ValueError("at least one cell is required")
    return normalised


@dataclass(frozen=True)
class LinearCurrentMoments:
    """Per-cell zeroth and first moments of a linear current density."""

    current: np.ndarray
    first: np.ndarray

    @property
    def radial(self) -> np.ndarray:
        """Return the radial first-current moment."""
        return self.first[:, 0]

    @property
    def vertical(self) -> np.ndarray:
        """Return the vertical first-current moment."""
        return self.first[:, 1]


class TracedClippedSupports(NamedTuple):
    """JAX arrays describing fixed-capacity supports for one flux map.

    ``vertex_capacity`` is the derived traced-polygon bound the layout is
    packed against, and ``refused_cell_count`` counts the cells whose live
    vertex count passed the bound. A refused cell carries zero geometry, so
    the geometry fields alone cannot distinguish it from a cell the clip
    legitimately left outside; the count is what reports it.
    """

    support_vertices: object
    vertex_count: object
    centroids: object
    included: object
    boundary: object
    area: object
    full_area: object
    first_area_moment: object
    second_area_moment: object
    contour_area: object
    patch_area_sum: object
    branch_support_vertices: object
    branch_vertex_count: object
    branch_area: object
    branch_first_area_moment: object
    branch_second_area_moment: object
    saddle: object
    saddle_vertex: object
    vertex_capacity: object = 0
    refused_cell_count: object = 0

    def refused_cells(self) -> int:
        """Return how many cells were refused for exceeding the capacity."""
        return int(np.asarray(self.refused_cell_count))

    def assert_no_refusal(self) -> None:
        """Fail closed when any cell exceeded the derived vertex capacity.

        A refused cell is packed as an empty inclusion, which is
        indistinguishable from a cell the clip placed outside the plasma and
        would otherwise be dropped from the support in silence. A caller that
        must not lose a cell calls this and gets the count in the message.
        """
        refused = self.refused_cells()
        if refused:
            raise TracedCapacityRefusalError(
                f"{refused} clipped cell(s) exceeded the derived traced-polygon "
                f"vertex capacity {int(np.asarray(self.vertex_capacity))}"
            )

    def qualify(self, participation):
        """Zero every geometric measure outside a topology participation mask."""
        selected = jnp.asarray(participation, dtype=bool)
        area = jnp.where(selected, self.area, 0.0)
        return self._replace(
            vertex_count=jnp.where(selected, self.vertex_count, 0),
            included=self.included & selected,
            boundary=self.boundary & selected,
            area=area,
            first_area_moment=jnp.where(selected[:, None], self.first_area_moment, 0.0),
            second_area_moment=jnp.where(
                selected[:, None, None], self.second_area_moment, 0.0
            ),
            patch_area_sum=jnp.sum(area),
            branch_vertex_count=jnp.where(
                selected[:, None], self.branch_vertex_count, 0
            ),
            branch_area=jnp.where(selected[:, None], self.branch_area, 0.0),
            branch_first_area_moment=jnp.where(
                selected[:, None, None], self.branch_first_area_moment, 0.0
            ),
            branch_second_area_moment=jnp.where(
                selected[:, None, None, None],
                self.branch_second_area_moment,
                0.0,
            ),
        )

    def linear_current_moments(self, density, gradient):
        """Contract a cellwise-linear current over the traced supports."""
        return padded_linear_current_moments(
            self.support_vertices,
            self.vertex_count,
            self.centroids,
            density,
            gradient,
        )

    def branch_linear_current_moments(self, density, gradient):
        """Contract independently declared densities over both saddle lobes."""
        density = jnp.asarray(density)
        gradient = jnp.asarray(gradient)
        cell_count = self.branch_support_vertices.shape[0]
        if density.shape != (cell_count, 2):
            raise ValueError("density must have shape (cells, 2)")
        if gradient.shape != (cell_count, 2, 2):
            raise ValueError("gradient must have shape (cells, 2, 2)")
        current, first = padded_linear_current_moments(
            self.branch_support_vertices.reshape(
                2 * cell_count, self.branch_support_vertices.shape[2], 2
            ),
            self.branch_vertex_count.reshape(2 * cell_count),
            jnp.broadcast_to(self.centroids[:, None, :], (cell_count, 2, 2)).reshape(
                2 * cell_count, 2
            ),
            density.reshape(2 * cell_count),
            gradient.reshape(2 * cell_count, 2),
        )
        return current.reshape(cell_count, 2), first.reshape(cell_count, 2, 2)


class SaddleCellWedges(NamedTuple):
    """Four fixed-capacity regions meeting at a separatrix saddle.

    Wedges are ordered as core, private flux, and the two common scrape-off-layer
    regions.  The core reference supplied to :meth:`AtomicCellMesh.traced_saddle_wedges`
    selects the first of the two same-sign lobes; the remaining same-sign lobe is
    private flux.  The opposite-sign lobes are ordered counter-clockwise about the
    core direction.  Non-saddle cells carry zero counts and exact-zero geometry.
    """

    support_vertices: object
    vertex_count: object
    centroids: object
    area: object
    full_area: object
    first_area_moment: object
    second_area_moment: object
    saddle: object
    saddle_vertex: object

    def linear_current_moments(self, density, gradient):
        """Integrate one independently declared linear profile per wedge."""
        density = jnp.asarray(density)
        gradient = jnp.asarray(gradient)
        cell_count = self.support_vertices.shape[0]
        if density.shape != (cell_count, 4):
            raise ValueError("density must have shape (cells, 4)")
        if gradient.shape != (cell_count, 4, 2):
            raise ValueError("gradient must have shape (cells, 4, 2)")
        current, first = padded_linear_current_moments(
            self.support_vertices.reshape(
                4 * cell_count, self.support_vertices.shape[2], 2
            ),
            self.vertex_count.reshape(4 * cell_count),
            jnp.broadcast_to(self.centroids[:, None, :], (cell_count, 4, 2)).reshape(
                4 * cell_count, 2
            ),
            density.reshape(4 * cell_count),
            gradient.reshape(4 * cell_count, 2),
        )
        return current.reshape(cell_count, 4), first.reshape(cell_count, 4, 2)


def padded_linear_current_moments(
    support_vertices,
    vertex_count,
    centroids,
    density,
    gradient,
):
    """Contract fixed-capacity clipped supports into current moments.

    Every argument keeps the mesh's fixed leading cell dimension while
    ``support_vertices`` retains its fixed vertex capacity. ``vertex_count``
    masks padding and closes each live polygon back to its first vertex. The
    resulting contraction is therefore compatible with one JAX trace across
    moving separatrices even when the number of live support vertices changes.
    """
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax.numpy as jnp

    vertices = jnp.asarray(support_vertices)
    count = jnp.asarray(vertex_count)
    centre = jnp.asarray(centroids)
    density = jnp.asarray(density)
    gradient = jnp.asarray(gradient)
    if vertices.ndim != 3 or vertices.shape[2] != 2:
        raise ValueError("support_vertices must have shape (cells, capacity, 2)")
    cell_count, capacity, _coordinate = vertices.shape
    if count.shape != (cell_count,):
        raise ValueError("vertex_count must carry one value per cell")
    if centre.shape != (cell_count, 2):
        raise ValueError("centroids must have shape (cells, 2)")
    if density.shape != (cell_count,):
        raise ValueError("density must carry one value per cell")
    if gradient.shape != (cell_count, 2):
        raise ValueError("gradient must have shape (cells, 2)")

    slot = jnp.arange(capacity)
    valid = slot[jnp.newaxis, :] < count[:, jnp.newaxis]
    following_slot = jnp.where(
        slot[jnp.newaxis, :] + 1 < count[:, jnp.newaxis],
        slot[jnp.newaxis, :] + 1,
        0,
    )
    following = jnp.take_along_axis(vertices, following_slot[..., jnp.newaxis], axis=1)
    local = vertices - centre[:, jnp.newaxis, :]
    following_local = following - centre[:, jnp.newaxis, :]
    radial = local[..., 0]
    vertical = local[..., 1]
    following_radial = following_local[..., 0]
    following_vertical = following_local[..., 1]
    cross = radial * following_vertical - following_radial * vertical
    cross = jnp.where(valid, cross, 0.0)
    area_twice = jnp.sum(cross, axis=1)
    orientation = jnp.where(area_twice < 0.0, -1.0, 1.0)
    area = 0.5 * orientation * area_twice
    first_area = orientation[:, jnp.newaxis] * jnp.stack(
        [
            jnp.sum((radial + following_radial) * cross, axis=1) / 6.0,
            jnp.sum((vertical + following_vertical) * cross, axis=1) / 6.0,
        ],
        axis=1,
    )
    radial_squared = (
        orientation
        * jnp.sum(
            (
                radial * radial
                + radial * following_radial
                + following_radial * following_radial
            )
            * cross,
            axis=1,
        )
        / 12.0
    )
    vertical_squared = (
        orientation
        * jnp.sum(
            (
                vertical * vertical
                + vertical * following_vertical
                + following_vertical * following_vertical
            )
            * cross,
            axis=1,
        )
        / 12.0
    )
    cross_area = (
        orientation
        * jnp.sum(
            (
                2.0 * radial * vertical
                + radial * following_vertical
                + following_radial * vertical
                + 2.0 * following_radial * following_vertical
            )
            * cross,
            axis=1,
        )
        / 24.0
    )
    second_area = jnp.stack(
        [
            jnp.stack([radial_squared, cross_area], axis=1),
            jnp.stack([cross_area, vertical_squared], axis=1),
        ],
        axis=1,
    )
    current = density * area + jnp.einsum("ni,ni->n", gradient, first_area)
    first_current = density[:, jnp.newaxis] * first_area + jnp.einsum(
        "nij,nj->ni", second_area, gradient
    )
    return current, first_current


def padded_polynomial_current_moments(
    support_vertices,
    vertex_count,
    centroids,
    coordinate_scale,
    coefficients,
    powers=None,
):
    """Integrate one complete polynomial and its first moments over each support.

    ``coefficients`` multiply total-degree monomials in coordinates centred on
    ``centroids`` and divided by ``coordinate_scale``. If ``powers`` is omitted,
    the complete basis is inferred from the static coefficient width. The edge
    reductions are closed simplex moments, so moving clip vertices remain
    traced values and no quadrature nodes or data-dependent shapes enter an
    iteration.
    """
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax.numpy as jnp

    vertices = jnp.asarray(support_vertices)
    count = jnp.asarray(vertex_count)
    centre = jnp.asarray(centroids)
    scale = jnp.asarray(coordinate_scale)
    coefficient = jnp.asarray(coefficients)
    if powers is None:
        column_count = coefficient.shape[-1]
        degree = math.isqrt(8 * column_count + 1)
        degree = (degree - 3) // 2
        powers = complete_polynomial_powers(degree)
    else:
        powers = tuple(powers)
    if vertices.ndim != 3 or vertices.shape[2] != 2:
        raise ValueError("support_vertices must have shape (cells, capacity, 2)")
    cell_count, capacity, _coordinate = vertices.shape
    if count.shape != (cell_count,):
        raise ValueError("vertex_count must carry one value per cell")
    if centre.shape != (cell_count, 2) or scale.shape != (cell_count, 2):
        raise ValueError("centroids and coordinate_scale must have shape (cells, 2)")
    if coefficient.shape != (cell_count, len(powers)):
        raise ValueError(
            "coefficients must carry one complete polynomial basis per cell"
        )

    local = (vertices - centre[:, None, :]) / scale[:, None, :]
    slot = jnp.arange(capacity)
    valid = slot[None, :] < count[:, None]
    following_slot = jnp.where(slot[None, :] + 1 < count[:, None], slot[None, :] + 1, 0)
    following = jnp.take_along_axis(local, following_slot[..., None], axis=1)
    cross = local[..., 0] * following[..., 1] - following[..., 0] * local[..., 1]
    cross = jnp.where(valid, cross, 0.0)
    orientation = jnp.where(jnp.sum(cross, axis=1) < 0.0, -1.0, 1.0)
    area_scale = scale[:, 0] * scale[:, 1]

    def monomial_moment(radial_power, vertical_power):
        edge_moment = jnp.zeros((cell_count, capacity), dtype=vertices.dtype)
        total_degree = radial_power + vertical_power
        for radial_first in range(radial_power + 1):
            radial_factor = (
                math.comb(radial_power, radial_first)
                * local[..., 0] ** radial_first
                * following[..., 0] ** (radial_power - radial_first)
            )
            for vertical_first in range(vertical_power + 1):
                first_degree = radial_first + vertical_first
                simplex = (
                    math.factorial(first_degree)
                    * math.factorial(total_degree - first_degree)
                    / math.factorial(total_degree + 2)
                )
                vertical_factor = (
                    math.comb(vertical_power, vertical_first)
                    * local[..., 1] ** vertical_first
                    * following[..., 1] ** (vertical_power - vertical_first)
                )
                edge_moment = edge_moment + simplex * radial_factor * vertical_factor
        return orientation * area_scale * jnp.sum(cross * edge_moment, axis=1)

    maximum_degree = max(radial + vertical for radial, vertical in powers)
    required_powers = tuple(
        (radial, vertical)
        for degree in range(maximum_degree + 2)
        for radial in range(degree, -1, -1)
        for vertical in (degree - radial,)
    )
    moments = {power: monomial_moment(*power) for power in required_powers}
    current = sum(
        coefficient[:, column] * moments[power] for column, power in enumerate(powers)
    )
    radial = scale[:, 0] * sum(
        coefficient[:, column] * moments[(power[0] + 1, power[1])]
        for column, power in enumerate(powers)
    )
    vertical = scale[:, 1] * sum(
        coefficient[:, column] * moments[(power[0], power[1] + 1)]
        for column, power in enumerate(powers)
    )
    included = count >= 3
    return (
        jnp.where(included, current, 0.0),
        jnp.where(included[:, None], jnp.stack([radial, vertical], axis=1), 0.0),
    )


def _pack_traced_vertices(vertices, valid, capacity):
    """Compact masked vertices without data-dependent array shapes."""
    import jax.numpy as jnp

    rank = jnp.cumsum(valid, axis=1) - 1
    safe_rank = jnp.where(valid, rank, 0)
    cell = jnp.broadcast_to(jnp.arange(vertices.shape[0])[:, None], safe_rank.shape)
    packed = jnp.zeros((vertices.shape[0], capacity, vertices.shape[2]), vertices.dtype)
    packed = packed.at[cell, safe_rank].add(jnp.where(valid[..., None], vertices, 0.0))
    return packed, jnp.sum(valid, axis=1)


def _pack_traced_values(values, valid, capacity):
    """Compact masked scalar values with the same ordering as vertices."""
    import jax.numpy as jnp

    rank = jnp.cumsum(valid, axis=1) - 1
    safe_rank = jnp.where(valid, rank, 0)
    cell = jnp.broadcast_to(jnp.arange(values.shape[0])[:, None], safe_rank.shape)
    packed = jnp.zeros((values.shape[0], capacity), dtype=jnp.int32)
    packed = packed.at[cell, safe_rank].add((valid & values).astype(jnp.int32))
    return packed > 0


def _cross_2d(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _traced_quadratic_value(points, coefficient, centre, scale):
    """Evaluate one cell-local quadratic at fixed-shape point rows."""
    local = (points - centre[:, None, :]) / scale[:, None, :]
    radial = local[..., 0]
    vertical = local[..., 1]
    basis = jnp.stack(
        (
            jnp.ones_like(radial),
            radial,
            vertical,
            radial * radial,
            radial * vertical,
            vertical * vertical,
        ),
        axis=-1,
    )
    return jnp.einsum("...i,...i->...", basis, coefficient[:, None, :])


def _traced_quadratic_segment_root(
    start, end, start_value, end_value, coefficient, centre, scale
):
    """Return the quadratic root on each sign-changing cell edge."""
    midpoint = 0.5 * (start + end)
    midpoint_value = _traced_quadratic_value(midpoint, coefficient, centre, scale)
    quadratic = 2.0 * (end_value + start_value - 2.0 * midpoint_value)
    linear = end_value - start_value - quadratic
    constant = start_value
    linear_fraction = start_value / (start_value - end_value)
    dtype = start.dtype
    floor = 64.0 * jnp.finfo(dtype).eps
    safe_quadratic = jnp.where(jnp.abs(quadratic) > floor, quadratic, 1.0)
    discriminant = jnp.maximum(linear * linear - 4.0 * quadratic * constant, 0.0)
    root_scale = jnp.sqrt(discriminant)
    first_root = (-linear - root_scale) / (2.0 * safe_quadratic)
    second_root = (-linear + root_scale) / (2.0 * safe_quadratic)
    first_valid = (first_root >= 0.0) & (first_root <= 1.0)
    second_valid = (second_root >= 0.0) & (second_root <= 1.0)
    quadratic_root = jnp.where(
        first_valid & second_valid,
        jnp.where(
            jnp.abs(first_root - linear_fraction)
            <= jnp.abs(second_root - linear_fraction),
            first_root,
            second_root,
        ),
        jnp.where(first_valid, first_root, second_root),
    )
    return jnp.where(jnp.abs(quadratic) > floor, quadratic_root, linear_fraction)


def _traced_level_segment_root(start, end, start_value, end_value, evaluator):
    """Bisect one shared level-set crossing on every cell edge."""
    lower = jnp.zeros_like(start_value)
    upper = jnp.ones_like(start_value)
    lower_value = start_value

    def bisect(_iteration, state):
        low, high, low_value = state
        midpoint = 0.5 * (low + high)
        point = start + midpoint[..., None] * (end - start)
        value = evaluator(point)
        same_side = (value > 0.0) == (low_value > 0.0)
        return (
            jnp.where(same_side, midpoint, low),
            jnp.where(same_side, high, midpoint),
            jnp.where(same_side, value, low_value),
        )

    lower, upper, _value = jax.lax.fori_loop(0, 48, bisect, (lower, upper, lower_value))
    return 0.5 * (lower + upper)


def _traced_quadratic_arc(start, end, coefficient, centre, scale):
    """Sample the connected quadratic level-set arc between two crossings."""
    parameter = jnp.linspace(
        0.0,
        1.0,
        _CURVED_BOUNDARY_SEGMENTS + 1,
        dtype=start.dtype,
    )
    chord = start[:, None, :] + parameter[None, :, None] * (end - start)[:, None, :]
    delta = end - start
    normal = jnp.stack((-delta[:, 1], delta[:, 0]), axis=1)
    zero_value = _traced_quadratic_value(chord, coefficient, centre, scale)
    positive_value = _traced_quadratic_value(
        chord + normal[:, None, :], coefficient, centre, scale
    )
    negative_value = _traced_quadratic_value(
        chord - normal[:, None, :], coefficient, centre, scale
    )
    quadratic = 0.5 * (positive_value + negative_value) - zero_value
    linear = 0.5 * (positive_value - negative_value)
    dtype = start.dtype
    floor = 64.0 * jnp.finfo(dtype).eps
    safe_quadratic = jnp.where(jnp.abs(quadratic) > floor, quadratic, 1.0)
    safe_linear = jnp.where(jnp.abs(linear) > floor, linear, 1.0)
    discriminant = jnp.maximum(linear * linear - 4.0 * quadratic * zero_value, 0.0)
    root_scale = jnp.sqrt(discriminant)
    first_root = (-linear - root_scale) / (2.0 * safe_quadratic)
    second_root = (-linear + root_scale) / (2.0 * safe_quadratic)
    curved_root = jnp.where(
        jnp.abs(first_root) <= jnp.abs(second_root), first_root, second_root
    )
    root = jnp.where(
        jnp.abs(quadratic) > floor,
        curved_root,
        -zero_value / safe_linear,
    )
    root = root.at[:, 0].set(0.0)
    root = root.at[:, -1].set(0.0)
    return chord + root[..., None] * normal[:, None, :]


def _traced_level_arc(start, end, evaluator, inside_vertex):
    """Trace the nearest level-set arc on the retained polygon's side."""
    parameter = jnp.linspace(
        0.0,
        1.0,
        _SPLINE_BOUNDARY_SEGMENTS + 1,
        dtype=start.dtype,
    )
    chord = start[:, None, :] + parameter[None, :, None] * (end - start)[:, None, :]
    delta = end - start
    normal = jnp.stack((-delta[:, 1], delta[:, 0]), axis=1)
    squared_length = jnp.sum(delta**2, axis=1)
    safe_squared_length = jnp.maximum(squared_length, jnp.finfo(start.dtype).tiny)
    chord_midpoint = 0.5 * (start + end)
    inside_side = jnp.sum((inside_vertex - chord_midpoint) * normal, axis=1)
    side = jnp.where(inside_side < 0.0, 1.0, -1.0)
    local_extent = jnp.minimum(
        jnp.linalg.norm(inside_vertex - chord_midpoint, axis=1)
        / jnp.sqrt(safe_squared_length),
        1.0,
    )
    signed_extent = side * jnp.maximum(local_extent, 32.0 * jnp.finfo(start.dtype).eps)
    lower = jnp.minimum(signed_extent, 0.0)[:, None]
    upper = jnp.maximum(signed_extent, 0.0)[:, None]
    root = jnp.zeros(chord.shape[:-1], dtype=start.dtype)
    difference_step = jnp.asarray(1.0e-5, dtype=start.dtype)

    def polish(_iteration, current):
        point = chord + current[..., None] * normal[:, None, :]
        offset = difference_step * normal[:, None, :]
        value = evaluator(point)
        derivative = (evaluator(point + offset) - evaluator(point - offset)) / (
            2.0 * difference_step
        )
        safe_derivative = jnp.where(
            jnp.abs(derivative) > jnp.finfo(start.dtype).tiny, derivative, 1.0
        )
        candidate = jnp.clip(current - value / safe_derivative, lower, upper)
        return jnp.where(
            jnp.abs(derivative) > jnp.finfo(start.dtype).tiny,
            candidate,
            current,
        )

    root = jax.lax.fori_loop(0, 12, polish, root)
    root = root.at[:, 0].set(0.0)
    root = root.at[:, -1].set(0.0)
    return chord + root[..., None] * normal[:, None, :]


def _project_arc_to_cell(points, edge_start, edge_end, valid_edge):
    """Project samples outside a convex atomic cell onto its nearest edge."""
    import jax.numpy as jnp

    inside = jnp.ones(points.shape[:-1], dtype=bool)
    best_distance = jnp.full(points.shape[:-1], jnp.inf, dtype=points.dtype)
    best_point = points
    floor = 128.0 * jnp.finfo(points.dtype).eps

    def project_edge(carry, edge_geometry):
        current_inside, current_distance, current_point = carry
        start, end, valid = edge_geometry
        edge = end - start
        relative = points - start[:, None, :]
        squared_length = jnp.sum(edge**2, axis=1)
        safe_squared_length = jnp.maximum(squared_length, jnp.finfo(points.dtype).tiny)
        cross = (
            edge[:, None, 0] * relative[..., 1] - edge[:, None, 1] * relative[..., 0]
        )
        scale = jnp.maximum(
            1.0,
            jnp.sqrt(safe_squared_length)[:, None]
            * jnp.maximum(
                jnp.max(jnp.abs(points), axis=-1),
                jnp.max(jnp.abs(start), axis=-1)[:, None],
            ),
        )
        current_inside = current_inside & (~valid[:, None] | (cross >= -floor * scale))
        fraction = jnp.clip(
            jnp.sum(relative * edge[:, None, :], axis=-1)
            / safe_squared_length[:, None],
            0.0,
            1.0,
        )
        candidate = start[:, None, :] + fraction[..., None] * edge[:, None, :]
        distance = jnp.sum((points - candidate) ** 2, axis=-1)
        nearer = valid[:, None] & (distance < current_distance)
        return (
            current_inside,
            jnp.where(nearer, distance, current_distance),
            jnp.where(nearer[..., None], candidate, current_point),
        ), None

    (inside, _distance, projected), _unused = jax.lax.scan(
        project_edge,
        (inside, best_distance, best_point),
        (
            jnp.moveaxis(edge_start, 1, 0),
            jnp.moveaxis(edge_end, 1, 0),
            jnp.moveaxis(valid_edge, 1, 0),
        ),
    )
    return jnp.where(inside[..., None], points, projected)


def _traced_polygon_moments(vertices, count, centroids):
    """Evaluate fixed-capacity polygon moments with traced reductions."""
    import jax.numpy as jnp

    capacity = vertices.shape[1]
    slot = jnp.arange(capacity)
    valid = slot[jnp.newaxis, :] < count[:, jnp.newaxis]
    following_slot = jnp.where(
        slot[jnp.newaxis, :] + 1 < count[:, jnp.newaxis],
        slot[jnp.newaxis, :] + 1,
        0,
    )
    following = jnp.take_along_axis(vertices, following_slot[..., None], axis=1)
    local = vertices - centroids[:, None, :]
    following_local = following - centroids[:, None, :]
    radial, vertical = local[..., 0], local[..., 1]
    following_radial = following_local[..., 0]
    following_vertical = following_local[..., 1]
    cross = radial * following_vertical - following_radial * vertical
    cross = jnp.where(valid, cross, 0.0)
    area_twice = jnp.sum(cross, axis=1)
    orientation = jnp.where(area_twice < 0.0, -1.0, 1.0)
    area = 0.5 * orientation * area_twice
    first = orientation[:, None] * jnp.stack(
        [
            jnp.sum((radial + following_radial) * cross, axis=1) / 6.0,
            jnp.sum((vertical + following_vertical) * cross, axis=1) / 6.0,
        ],
        axis=1,
    )
    radial_squared = (
        orientation
        * jnp.sum(
            (radial**2 + radial * following_radial + following_radial**2) * cross,
            axis=1,
        )
        / 12.0
    )
    vertical_squared = (
        orientation
        * jnp.sum(
            (vertical**2 + vertical * following_vertical + following_vertical**2)
            * cross,
            axis=1,
        )
        / 12.0
    )
    cross_moment = (
        orientation
        * jnp.sum(
            (
                2.0 * radial * vertical
                + radial * following_vertical
                + following_radial * vertical
                + 2.0 * following_radial * following_vertical
            )
            * cross,
            axis=1,
        )
        / 24.0
    )
    second = jnp.stack(
        [
            jnp.stack([radial_squared, cross_moment], axis=1),
            jnp.stack([cross_moment, vertical_squared], axis=1),
        ],
        axis=1,
    )
    nonzero = area > 0.0
    return (
        jnp.where(nonzero, area, 0.0),
        jnp.where(nonzero[:, None], first, 0.0),
        jnp.where(nonzero[:, None, None], second, 0.0),
    )


def _traced_clip(
    node_coordinates,
    cell_nodes,
    cell_vertex_count,
    centroids,
    support_capacity,
    signed_flux,
    saddle_vertex=None,
    curve_coefficient=None,
    curve_centre=None,
    curve_scale=None,
    curve_evaluator=None,
    participating_cell=None,
    arc_tracer: Callable | None = None,
):
    """Clip fixed atomic cells using only traced fixed-shape operations.

    ``arc_tracer`` is the level-root tracer used on each gap crossing, called as
    ``arc_tracer(start, end, curve_evaluator, inside_vertex)``. It defaults to
    the fixed twelve-step polish of :func:`_traced_level_arc`; a caller that
    needs a different derivative, such as an implicit-function tangent, passes
    its own tracer explicitly rather than rebinding a module global.
    """
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax.numpy as jnp

    coordinates = jnp.asarray(node_coordinates)
    nodes = jnp.asarray(cell_nodes)
    count = jnp.asarray(cell_vertex_count)
    centre = jnp.asarray(centroids)
    flux = jnp.asarray(signed_flux)
    cell_count, width = nodes.shape
    if flux.shape != (coordinates.shape[0],):
        raise ValueError("signed_flux must carry one value per atomic node")

    slot = jnp.arange(width)
    valid_edge = slot[None, :] < count[:, None]
    following_slot = jnp.where(slot[None, :] + 1 < count[:, None], slot[None, :] + 1, 0)
    following_nodes = jnp.take_along_axis(nodes, following_slot, axis=1)
    cell_start_point = coordinates[nodes]
    cell_end_point = coordinates[following_nodes]
    start_point = cell_start_point
    end_point = cell_end_point
    original_vertex = valid_edge
    curved = curve_coefficient is not None or curve_evaluator is not None
    if curve_evaluator is not None:
        coefficient = jnp.zeros((cell_count, 6), dtype=coordinates.dtype)
        curve_origin = centre
        curve_extent = jnp.ones_like(centre)
        start_flux = curve_evaluator(start_point)
        end_flux = curve_evaluator(end_point)
    elif curved:
        if curve_centre is None or curve_scale is None:
            raise ValueError(
                "curve coefficients require cell centres and coordinate scales"
            )
        coefficient = jnp.asarray(curve_coefficient, dtype=coordinates.dtype)
        curve_origin = jnp.asarray(curve_centre, dtype=coordinates.dtype)
        curve_extent = jnp.asarray(curve_scale, dtype=coordinates.dtype)
        if coefficient.shape != (cell_count, 6):
            raise ValueError("curve_coefficient must have shape (cells, 6)")
        if curve_origin.shape != (cell_count, 2) or curve_extent.shape != (
            cell_count,
            2,
        ):
            raise ValueError("curve centres and scales must have shape (cells, 2)")
        start_flux = _traced_quadratic_value(
            start_point, coefficient, curve_origin, curve_extent
        )
        end_flux = _traced_quadratic_value(
            end_point, coefficient, curve_origin, curve_extent
        )
    else:
        coefficient = jnp.zeros((cell_count, 6), dtype=coordinates.dtype)
        curve_origin = centre
        curve_extent = jnp.ones_like(centre)
        start_flux = flux[nodes]
        end_flux = flux[following_nodes]
    if participating_cell is None:
        participating = jnp.ones(cell_count, dtype=bool)
    else:
        participating = jnp.asarray(participating_cell, dtype=bool)
        if participating.shape != (cell_count,):
            raise ValueError("participating_cell must carry one flag per cell")
    chord_capacity = support_capacity
    start_inside = start_flux > 0.0
    end_inside = end_flux > 0.0
    crossing_edge = valid_edge & participating[:, None] & (start_inside != end_inside)
    denominator = start_flux - end_flux
    linear_fraction = start_flux / denominator
    curved_fraction = (
        _traced_level_segment_root(
            start_point,
            end_point,
            start_flux,
            end_flux,
            curve_evaluator,
        )
        if curve_evaluator is not None
        else _traced_quadratic_segment_root(
            start_point,
            end_point,
            start_flux,
            end_flux,
            coefficient,
            curve_origin,
            curve_extent,
        )
    )
    fraction = jnp.where(
        crossing_edge, jnp.where(curved, curved_fraction, linear_fraction), 0.0
    )
    crossing_point = start_point + fraction[..., None] * (end_point - start_point)

    previous_crossing = jnp.roll(crossing_point, 1, axis=1)
    previous_crossing_edge = jnp.roll(crossing_edge, 1, axis=1)
    duplicate = (
        crossing_edge
        & previous_crossing_edge
        & jnp.all(crossing_point == previous_crossing, axis=2)
    )
    unique_crossing = crossing_edge & ~duplicate
    crossing, crossing_count = _pack_traced_vertices(
        crossing_point, unique_crossing, width
    )
    packed_leaving = _pack_traced_values(
        crossing_edge & start_inside, unique_crossing, width
    )
    supplied_saddle = saddle_vertex is not None
    saddle = (crossing_count == 4) & jnp.asarray(
        curve_evaluator is None or supplied_saddle
    )
    first_line = crossing[:, 2] - crossing[:, 0]
    second_line = crossing[:, 3] - crossing[:, 1]
    denominator = _cross_2d(first_line, second_line)
    safe_denominator = jnp.where(saddle, denominator, 1.0)
    inferred_saddle = (
        crossing[:, 0]
        + first_line
        * (_cross_2d(crossing[:, 1] - crossing[:, 0], second_line) / safe_denominator)[
            :, None
        ]
    )
    if saddle_vertex is None:
        saddle_point = inferred_saddle
    else:
        supplied = jnp.asarray(saddle_vertex, dtype=coordinates.dtype)
        if supplied.shape == (2,):
            supplied = jnp.broadcast_to(supplied, (cell_count, 2))
        if supplied.shape != (cell_count, 2):
            raise ValueError("saddle_vertex must have shape (2,) or (cells, 2)")
        saddle_point = supplied
    saddle_point = jnp.where(saddle[:, None], saddle_point, 0.0)

    saddle_candidate = jnp.broadcast_to(saddle_point[:, None, :], start_point.shape)
    candidates = jnp.stack(
        [start_point, crossing_point, saddle_candidate], axis=2
    ).reshape(cell_count, 3 * width, 2)
    candidate_valid = jnp.stack(
        [
            valid_edge & original_vertex & start_inside,
            crossing_edge,
            saddle[:, None] & crossing_edge & start_inside,
        ],
        axis=2,
    ).reshape(cell_count, 3 * width)
    candidate_saddle = jnp.stack(
        [
            jnp.zeros_like(valid_edge),
            jnp.zeros_like(valid_edge),
            saddle[:, None] & crossing_edge & start_inside,
        ],
        axis=2,
    ).reshape(cell_count, 3 * width)
    candidate_leaving = jnp.stack(
        [
            jnp.zeros_like(valid_edge),
            crossing_edge & start_inside,
            jnp.zeros_like(valid_edge),
        ],
        axis=2,
    ).reshape(cell_count, 3 * width)
    candidate_crossing = jnp.stack(
        [
            jnp.zeros_like(valid_edge),
            crossing_edge,
            jnp.zeros_like(valid_edge),
        ],
        axis=2,
    ).reshape(cell_count, 3 * width)
    compact, compact_count = _pack_traced_vertices(
        candidates, candidate_valid, support_capacity
    )
    compact_saddle = _pack_traced_values(
        candidate_saddle, candidate_valid, support_capacity
    )
    compact_leaving = _pack_traced_values(
        candidate_leaving, candidate_valid, support_capacity
    )
    compact_crossing = _pack_traced_values(
        candidate_crossing, candidate_valid, support_capacity
    )
    compact_slot = jnp.arange(support_capacity)
    compact_valid = compact_slot[None, :] < compact_count[:, None]
    previous = jnp.roll(compact, 1, axis=1)
    distinct = jnp.any(compact != previous, axis=2)
    keep = compact_valid & ((compact_slot[None, :] == 0) | distinct)
    support, vertex_count = _pack_traced_vertices(compact, keep, support_capacity)
    support_saddle = _pack_traced_values(compact_saddle, keep, support_capacity)
    support_leaving = _pack_traced_values(compact_leaving, keep, support_capacity)
    support_crossing = _pack_traced_values(compact_crossing, keep, support_capacity)
    last_slot = jnp.maximum(vertex_count - 1, 0)
    last = jnp.take_along_axis(support, last_slot[:, None, None], axis=1)[:, 0]
    repeated_closure = (vertex_count > 1) & jnp.all(last == support[:, 0], axis=1)
    vertex_count = vertex_count - repeated_closure.astype(vertex_count.dtype)
    support = jnp.where(
        compact_slot[None, :, None] < vertex_count[:, None, None], support, 0.0
    )

    first_saddle = jnp.argmax(support_saddle, axis=1)
    first_leaving = jnp.argmax(support_leaving, axis=1)
    simple_boundary = crossing_count == 2
    evaluator_boundary = (
        (crossing_count >= 2)
        & (crossing_count % 2 == 0)
        & jnp.asarray(curve_evaluator is not None)
    )
    rotation = jnp.where(
        saddle,
        first_saddle,
        jnp.where((simple_boundary & curved) | evaluator_boundary, first_leaving, 0),
    )
    live_count = jnp.maximum(vertex_count, 1)
    rotated_slot = (compact_slot[None, :] + rotation[:, None]) % live_count[:, None]
    support = jnp.take_along_axis(support, rotated_slot[..., None], axis=1)
    support_saddle = jnp.take_along_axis(support_saddle, rotated_slot, axis=1)
    support_leaving = jnp.take_along_axis(support_leaving, rotated_slot, axis=1)
    support_crossing = jnp.take_along_axis(support_crossing, rotated_slot, axis=1)
    support = jnp.where(
        compact_slot[None, :, None] < vertex_count[:, None, None], support, 0.0
    )
    support_saddle = support_saddle & (compact_slot[None, :] < vertex_count[:, None])

    overflow = jnp.zeros(cell_count, dtype=bool)
    if curve_evaluator is not None:
        base_valid = compact_slot[None, :] < vertex_count[:, None]
        next_slot = jnp.where(
            compact_slot[None, :] + 1 < vertex_count[:, None],
            compact_slot[None, :] + 1,
            0,
        )
        following_crossing = jnp.take_along_axis(support_crossing, next_slot, axis=1)
        outside_gap = base_valid & support_leaving & following_crossing
        inside_slot = jnp.where(next_slot + 1 < vertex_count[:, None], next_slot + 1, 0)
        following_vertex = jnp.take_along_axis(support, next_slot[..., None], axis=1)
        inside_vertex = jnp.take_along_axis(support, inside_slot[..., None], axis=1)

        tracer = _traced_level_arc if arc_tracer is None else arc_tracer

        def trace_gap(_carry, gap_geometry):
            gap_start, gap_end, gap_inside = gap_geometry
            traced = tracer(
                gap_start,
                gap_end,
                curve_evaluator,
                gap_inside,
            )
            return None, _project_arc_to_cell(
                traced,
                cell_start_point,
                cell_end_point,
                valid_edge,
            )

        _carry, scanned_arc = jax.lax.scan(
            trace_gap,
            None,
            (
                jnp.moveaxis(support, 1, 0),
                jnp.moveaxis(following_vertex, 1, 0),
                jnp.moveaxis(inside_vertex, 1, 0),
            ),
        )
        arc = jnp.moveaxis(scanned_arc, 0, 1)
        expanded_candidate = jnp.concatenate(
            (support[:, :, None, :], arc[:, :, 1:-1, :]), axis=2
        ).reshape(cell_count, chord_capacity * _SPLINE_BOUNDARY_SEGMENTS, 2)
        expanded_valid = jnp.concatenate(
            (
                base_valid[:, :, None],
                jnp.broadcast_to(
                    outside_gap[:, :, None],
                    (cell_count, chord_capacity, _SPLINE_BOUNDARY_SEGMENTS - 1),
                ),
            ),
            axis=2,
        ).reshape(cell_count, chord_capacity * _SPLINE_BOUNDARY_SEGMENTS)
        expanded_saddle = jnp.concatenate(
            (
                support_saddle[:, :, None],
                jnp.zeros(
                    (cell_count, chord_capacity, _SPLINE_BOUNDARY_SEGMENTS - 1),
                    dtype=bool,
                ),
            ),
            axis=2,
        ).reshape(cell_count, chord_capacity * _SPLINE_BOUNDARY_SEGMENTS)
        live_vertex_count = jnp.sum(expanded_valid, axis=1)
        support_capacity = traced_polygon_vertex_capacity(chord_capacity)
        overflow = live_vertex_count > support_capacity
        support, vertex_count = _pack_traced_vertices(
            expanded_candidate, expanded_valid, support_capacity
        )
        support_saddle = _pack_traced_values(
            expanded_saddle, expanded_valid, support_capacity
        )
        compact_slot = jnp.arange(support_capacity)
    elif curved:
        arc = _traced_quadratic_arc(
            support[:, 0],
            support[:, 1],
            coefficient,
            curve_origin,
            curve_extent,
        )
        expanded = jnp.concatenate(
            (support[:, :1], arc[:, 1:-1], support[:, 1:chord_capacity]), axis=1
        )
        expanded_count = vertex_count + (_CURVED_BOUNDARY_SEGMENTS - 1)
        use_arc = simple_boundary & (vertex_count >= 3)
        padded_support = jnp.pad(
            support, ((0, 0), (0, _CURVED_BOUNDARY_SEGMENTS - 1), (0, 0))
        )
        padded_saddle = jnp.pad(
            support_saddle, ((0, 0), (0, _CURVED_BOUNDARY_SEGMENTS - 1))
        )
        support = jnp.where(use_arc[:, None, None], expanded, padded_support)
        support_saddle = padded_saddle
        vertex_count = jnp.where(use_arc, expanded_count, vertex_count)
        support_capacity = chord_capacity + _CURVED_BOUNDARY_SEGMENTS - 1
        compact_slot = jnp.arange(support_capacity)

    branch_number = jnp.cumsum(support_saddle, axis=1) - 1
    branch_vertices = []
    branch_counts = []
    for branch in range(2):
        branch_valid = compact_slot[None, :] < vertex_count[:, None]
        branch_valid = branch_valid & jnp.where(
            saddle[:, None], branch_number == branch, branch == 0
        )
        vertices, counts = _pack_traced_vertices(
            support, branch_valid, support_capacity
        )
        branch_vertices.append(vertices)
        branch_counts.append(counts)
    branch_support = jnp.stack(branch_vertices, axis=1)
    branch_vertex_count = jnp.stack(branch_counts, axis=1)

    if curve_evaluator is not None:
        expanded_branches = []
        expanded_branch_counts = []
        branch_overflow = jnp.zeros(cell_count, dtype=bool)
        half_arc_slot = jnp.arange(0, _SPLINE_BOUNDARY_SEGMENTS + 1, 2)
        second_half_slot = half_arc_slot[1:-1]
        middle_slot = jnp.arange(2, chord_capacity)

        for branch in range(2):
            branch_polygon = branch_support[:, branch]
            branch_count = branch_vertex_count[:, branch]
            last_slot = jnp.maximum(branch_count - 1, 0)
            previous_slot = jnp.maximum(branch_count - 2, 0)
            first_root = branch_polygon[:, 1]
            last_root = jnp.take_along_axis(
                branch_polygon, last_slot[:, None, None], axis=1
            )[:, 0]
            first_inside = branch_polygon[:, 2]
            last_inside = jnp.take_along_axis(
                branch_polygon, previous_slot[:, None, None], axis=1
            )[:, 0]
            first_half = tracer(
                saddle_point,
                first_root,
                curve_evaluator,
                first_inside,
            )[:, half_arc_slot]
            second_half = tracer(
                last_root,
                saddle_point,
                curve_evaluator,
                last_inside,
            )[:, second_half_slot]
            middle = branch_polygon[:, middle_slot]
            expanded_candidate = jnp.concatenate(
                (first_half, middle, second_half), axis=1
            )
            expanded_valid = jnp.concatenate(
                (
                    jnp.broadcast_to(
                        saddle[:, None], (cell_count, first_half.shape[1])
                    ),
                    saddle[:, None] & (middle_slot[None, :] < branch_count[:, None]),
                    jnp.broadcast_to(
                        saddle[:, None], (cell_count, second_half.shape[1])
                    ),
                ),
                axis=1,
            )
            expanded_count = jnp.sum(expanded_valid, axis=1)
            branch_overflow = branch_overflow | (expanded_count > support_capacity)
            expanded_support, expanded_count = _pack_traced_vertices(
                expanded_candidate, expanded_valid, support_capacity
            )
            expanded_branches.append(
                jnp.where(saddle[:, None, None], expanded_support, branch_polygon)
            )
            expanded_branch_counts.append(
                jnp.where(saddle, expanded_count, branch_count)
            )

        branch_support = jnp.stack(expanded_branches, axis=1)
        branch_vertex_count = jnp.stack(expanded_branch_counts, axis=1)
        overflow = overflow | branch_overflow

    full_area, _full_first, _full_second = _traced_polygon_moments(
        cell_start_point, count, centre
    )
    flat_branch_support = branch_support.reshape(2 * cell_count, support_capacity, 2)
    flat_branch_count = branch_vertex_count.reshape(2 * cell_count)
    flat_centre = jnp.broadcast_to(centre[:, None, :], (cell_count, 2, 2)).reshape(
        2 * cell_count, 2
    )
    branch_area, branch_first, branch_second = _traced_polygon_moments(
        flat_branch_support, flat_branch_count, flat_centre
    )
    branch_area = branch_area.reshape(cell_count, 2)
    branch_first = branch_first.reshape(cell_count, 2, 2)
    branch_second = branch_second.reshape(cell_count, 2, 2, 2)
    area = jnp.sum(branch_area, axis=1)
    first = jnp.sum(branch_first, axis=1)
    second = jnp.sum(branch_second, axis=1)
    included = area > 0.0
    included = included & ~overflow
    vertex_count = jnp.where(included, vertex_count, 0)
    support = jnp.where(included[:, None, None], support, 0.0)
    area = jnp.where(included, area, 0.0)
    first = jnp.where(included[:, None], first, 0.0)
    second = jnp.where(included[:, None, None], second, 0.0)
    branch_area = jnp.where(included[:, None], branch_area, 0.0)
    branch_first = jnp.where(included[:, None, None], branch_first, 0.0)
    branch_second = jnp.where(included[:, None, None, None], branch_second, 0.0)

    boundary = included & jnp.where(
        jnp.asarray(curve_evaluator is not None),
        (crossing_count >= 2) & (crossing_count % 2 == 0),
        (crossing_count == 2) | saddle,
    )
    crossing_slot = jnp.arange(width)
    crossing_valid = crossing_slot[None, :] < crossing_count[:, None]
    next_slot = jnp.where(
        crossing_slot[None, :] + 1 < crossing_count[:, None],
        crossing_slot[None, :] + 1,
        0,
    )
    next_crossing = jnp.take_along_axis(crossing, next_slot[..., None], axis=1)
    direct_cross = _cross_2d(crossing, next_crossing)
    saddle_cross = _cross_2d(crossing, saddle_point[:, None, :]) + _cross_2d(
        saddle_point[:, None, :], next_crossing
    )
    contour_cross = jnp.sum(
        jnp.where(
            crossing_valid & packed_leaving,
            jnp.where(saddle[:, None], saddle_cross, direct_cross),
            0.0,
        ),
        axis=1,
    )
    contour_area = 0.5 * jnp.abs(jnp.sum(contour_cross))
    patch_area_sum = jnp.sum(area)
    return TracedClippedSupports(
        support_vertices=support,
        vertex_count=vertex_count,
        centroids=centre,
        included=included,
        boundary=boundary,
        area=area,
        full_area=full_area,
        first_area_moment=first,
        second_area_moment=second,
        contour_area=contour_area,
        patch_area_sum=patch_area_sum,
        branch_support_vertices=branch_support,
        branch_vertex_count=branch_vertex_count,
        branch_area=branch_area,
        branch_first_area_moment=branch_first,
        branch_second_area_moment=branch_second,
        saddle=saddle,
        saddle_vertex=saddle_point,
        vertex_capacity=jnp.asarray(support_capacity, dtype=jnp.int32),
        refused_cell_count=jnp.sum(overflow).astype(jnp.int32),
    )


def _ordered_saddle_wedges(positive, negative, core_reference):
    """Combine opposite clip signs into one deterministic four-wedge carrier."""
    reference = jnp.asarray(core_reference, dtype=positive.support_vertices.dtype)
    cell_count = positive.support_vertices.shape[0]
    if reference.shape == (2,):
        reference = jnp.broadcast_to(reference, (cell_count, 2))
    if reference.shape != (cell_count, 2):
        raise ValueError("core_reference must have shape (2,) or (cells, 2)")

    saddle = positive.saddle & negative.saddle
    saddle_vertex = positive.saddle_vertex
    core_direction = reference - saddle_vertex

    def branch_centroids(support):
        safe_area = jnp.where(support.branch_area > 0.0, support.branch_area, 1.0)
        return support.centroids[:, None, :] + (
            support.branch_first_area_moment / safe_area[..., None]
        )

    def gather(value, order):
        index = order[(...,) + (None,) * (value.ndim - 2)]
        return jnp.take_along_axis(value, jnp.broadcast_to(index, value.shape), axis=1)

    positive_centroids = branch_centroids(positive)
    positive_score = jnp.einsum(
        "nbi,ni->nb", positive_centroids - saddle_vertex[:, None, :], core_direction
    )
    positive_swap = positive_score[:, 1] > positive_score[:, 0]
    positive_order = jnp.stack(
        (
            jnp.where(positive_swap, 1, 0),
            jnp.where(positive_swap, 0, 1),
        ),
        axis=1,
    )

    negative_centroids = branch_centroids(negative)
    negative_direction = negative_centroids - saddle_vertex[:, None, :]
    negative_cross = (
        core_direction[:, None, 0] * negative_direction[..., 1]
        - core_direction[:, None, 1] * negative_direction[..., 0]
    )
    negative_swap = negative_cross[:, 1] > negative_cross[:, 0]
    negative_order = jnp.stack(
        (
            jnp.where(negative_swap, 1, 0),
            jnp.where(negative_swap, 0, 1),
        ),
        axis=1,
    )

    vertices = jnp.concatenate(
        (
            gather(positive.branch_support_vertices, positive_order),
            gather(negative.branch_support_vertices, negative_order),
        ),
        axis=1,
    )
    vertex_count = jnp.concatenate(
        (
            gather(positive.branch_vertex_count, positive_order),
            gather(negative.branch_vertex_count, negative_order),
        ),
        axis=1,
    )
    area = jnp.concatenate(
        (
            gather(positive.branch_area, positive_order),
            gather(negative.branch_area, negative_order),
        ),
        axis=1,
    )
    first = jnp.concatenate(
        (
            gather(positive.branch_first_area_moment, positive_order),
            gather(negative.branch_first_area_moment, negative_order),
        ),
        axis=1,
    )
    second = jnp.concatenate(
        (
            gather(positive.branch_second_area_moment, positive_order),
            gather(negative.branch_second_area_moment, negative_order),
        ),
        axis=1,
    )
    slot = jnp.arange(vertices.shape[2])
    live = slot[None, None, :] < vertex_count[..., None]
    selected = saddle[:, None]
    vertex_count = jnp.where(selected, vertex_count, 0)
    vertices = jnp.where((selected[..., None] & live)[..., None], vertices, 0.0)
    return SaddleCellWedges(
        support_vertices=vertices,
        vertex_count=vertex_count,
        centroids=positive.centroids,
        area=jnp.where(selected, area, 0.0),
        full_area=jnp.where(saddle, positive.full_area, 0.0),
        first_area_moment=jnp.where(selected[..., None], first, 0.0),
        second_area_moment=jnp.where(selected[..., None, None], second, 0.0),
        saddle=saddle,
        saddle_vertex=jnp.where(saddle[:, None], saddle_vertex, 0.0),
    )


def _traced_saddle_wedges_from_edge_roots(
    node_coordinates,
    cell_nodes,
    cell_vertex_count,
    centroids,
    support_capacity,
    signed_flux,
    saddle_vertex,
    core_reference,
    edge_root_fraction,
    edge_root_count,
    edge_root_positive_after,
):
    """Build four cyclic saddle wedges from at most two roots per cell edge."""
    coordinates = jnp.asarray(node_coordinates)
    nodes = jnp.asarray(cell_nodes)
    count = jnp.asarray(cell_vertex_count)
    centre = jnp.asarray(centroids)
    flux = jnp.asarray(signed_flux)
    fraction = jnp.asarray(edge_root_fraction, dtype=coordinates.dtype)
    root_count = jnp.asarray(edge_root_count)
    positive_after = jnp.asarray(edge_root_positive_after, dtype=bool)
    cell_count, width = nodes.shape
    expected_root_shape = (cell_count, width, 2)
    if fraction.shape != expected_root_shape:
        raise ValueError("edge_root_fraction must have shape (cells, edges, 2)")
    if positive_after.shape != expected_root_shape:
        raise ValueError("edge_root_positive_after must have shape (cells, edges, 2)")
    if root_count.shape != (cell_count, width):
        raise ValueError("edge_root_count must have shape (cells, edges)")
    if flux.shape != (coordinates.shape[0],):
        raise ValueError("signed_flux must carry one value per atomic node")

    supplied_saddle = jnp.asarray(saddle_vertex, dtype=coordinates.dtype)
    if supplied_saddle.shape == (2,):
        supplied_saddle = jnp.broadcast_to(supplied_saddle, (cell_count, 2))
    if supplied_saddle.shape != (cell_count, 2):
        raise ValueError("saddle_vertex must have shape (2,) or (cells, 2)")
    reference = jnp.asarray(core_reference, dtype=coordinates.dtype)
    if reference.shape == (2,):
        reference = jnp.broadcast_to(reference, (cell_count, 2))
    if reference.shape != (cell_count, 2):
        raise ValueError("core_reference must have shape (2,) or (cells, 2)")

    edge_slot = jnp.arange(width)
    root_slot = jnp.arange(2)
    valid_edge = edge_slot[None, :] < count[:, None]
    valid_root = valid_edge[..., None] & (
        root_slot[None, None, :] < root_count[..., None]
    )
    following_edge = jnp.where(
        edge_slot[None, :] + 1 < count[:, None], edge_slot[None, :] + 1, 0
    )
    following_nodes = jnp.take_along_axis(nodes, following_edge, axis=1)
    start = coordinates[nodes]
    end = coordinates[following_nodes]
    root_coordinate = (
        start[..., None, :] + fraction[..., None] * (end - start)[..., None, :]
    )
    perimeter = edge_slot[None, :, None] + fraction
    packed_root, total_root_count = _pack_traced_vertices(
        root_coordinate.reshape(cell_count, 2 * width, 2),
        valid_root.reshape(cell_count, 2 * width),
        4,
    )
    packed_parameter, _parameter_count = _pack_traced_vertices(
        perimeter[..., None].reshape(cell_count, 2 * width, 1),
        valid_root.reshape(cell_count, 2 * width),
        4,
    )
    packed_parameter = packed_parameter[..., 0]
    packed_positive_after = _pack_traced_values(
        positive_after.reshape(cell_count, 2 * width),
        valid_root.reshape(cell_count, 2 * width),
        4,
    )
    saddle = total_root_count == 4

    next_root = jnp.roll(packed_root, -1, axis=1)
    next_parameter = jnp.roll(packed_parameter, -1, axis=1)
    perimeter_length = count[:, None].astype(coordinates.dtype)
    next_parameter = jnp.where(
        next_parameter < packed_parameter,
        next_parameter + perimeter_length,
        next_parameter,
    )
    start_edge = jnp.floor(packed_parameter).astype(jnp.int32) % count[:, None]
    boundary_slot = jnp.arange(width)
    boundary_index = (start_edge[..., None] + 1 + boundary_slot[None, None, :]) % count[
        :, None, None
    ]
    cell_index = jnp.arange(cell_count)[:, None, None]
    boundary_node = nodes[cell_index, boundary_index]
    boundary_vertex = coordinates[boundary_node]
    boundary_parameter = boundary_index.astype(coordinates.dtype)
    boundary_parameter = jnp.where(
        boundary_parameter <= packed_parameter[..., None],
        boundary_parameter + perimeter_length[..., None],
        boundary_parameter,
    )
    between_roots = (boundary_parameter > packed_parameter[..., None]) & (
        boundary_parameter < next_parameter[..., None]
    )

    saddle_candidate = jnp.broadcast_to(
        supplied_saddle[:, None, None, :], (cell_count, 4, 1, 2)
    )
    cyclic_candidate = jnp.concatenate(
        (
            saddle_candidate,
            packed_root[:, :, None, :],
            boundary_vertex,
            next_root[:, :, None, :],
        ),
        axis=2,
    )
    cyclic_valid = (
        jnp.concatenate(
            (
                jnp.ones((cell_count, 4, 2), dtype=bool),
                between_roots,
                jnp.ones((cell_count, 4, 1), dtype=bool),
            ),
            axis=2,
        )
        & saddle[:, None, None]
    )
    flat_vertices, flat_count = _pack_traced_vertices(
        cyclic_candidate.reshape(cell_count * 4, width + 3, 2),
        cyclic_valid.reshape(cell_count * 4, width + 3),
        support_capacity,
    )
    cyclic_vertices = flat_vertices.reshape(cell_count, 4, support_capacity, 2)
    cyclic_count = flat_count.reshape(cell_count, 4)
    flat_centre = jnp.broadcast_to(centre[:, None, :], (cell_count, 4, 2)).reshape(
        cell_count * 4, 2
    )
    cyclic_area, cyclic_first, cyclic_second = _traced_polygon_moments(
        flat_vertices,
        flat_count,
        flat_centre,
    )
    cyclic_area = cyclic_area.reshape(cell_count, 4)
    cyclic_first = cyclic_first.reshape(cell_count, 4, 2)
    cyclic_second = cyclic_second.reshape(cell_count, 4, 2, 2)
    safe_area = jnp.where(cyclic_area > 0.0, cyclic_area, 1.0)
    wedge_centroid = centre[:, None, :] + cyclic_first / safe_area[..., None]
    core_direction = reference - supplied_saddle
    core_score = jnp.einsum(
        "nwi,ni->nw", wedge_centroid - supplied_saddle[:, None, :], core_direction
    )
    wedge_slot = jnp.arange(4)[None, :]
    core_index = jnp.argmax(
        jnp.where(packed_positive_after, core_score, -jnp.inf), axis=1
    )
    private_index = jnp.argmax(
        jnp.where(
            packed_positive_after & (wedge_slot != core_index[:, None]),
            1,
            0,
        ),
        axis=1,
    )
    negative_direction = wedge_centroid - supplied_saddle[:, None, :]
    negative_cross = (
        core_direction[:, None, 0] * negative_direction[..., 1]
        - core_direction[:, None, 1] * negative_direction[..., 0]
    )
    first_sol = jnp.argmax(
        jnp.where(~packed_positive_after, negative_cross, -jnp.inf), axis=1
    )
    second_sol = jnp.argmax(
        jnp.where(
            ~packed_positive_after & (wedge_slot != first_sol[:, None]),
            1,
            0,
        ),
        axis=1,
    )
    order = jnp.stack((core_index, private_index, first_sol, second_sol), axis=1)

    def gather(value):
        index = order[(...,) + (None,) * (value.ndim - 2)]
        return jnp.take_along_axis(value, jnp.broadcast_to(index, value.shape), axis=1)

    vertices = gather(cyclic_vertices)
    vertex_count = gather(cyclic_count)
    area = gather(cyclic_area)
    first = gather(cyclic_first)
    second = gather(cyclic_second)
    live = jnp.arange(support_capacity)[None, None, :] < vertex_count[..., None]
    selected = saddle[:, None]
    vertices = jnp.where((selected[..., None] & live)[..., None], vertices, 0.0)
    vertex_count = jnp.where(selected, vertex_count, 0)
    full_area, _full_first, _full_second = _traced_polygon_moments(
        start,
        count,
        centre,
    )
    return SaddleCellWedges(
        support_vertices=vertices,
        vertex_count=vertex_count,
        centroids=centre,
        area=jnp.where(selected, area, 0.0),
        full_area=jnp.where(saddle, full_area, 0.0),
        first_area_moment=jnp.where(selected[..., None], first, 0.0),
        second_area_moment=jnp.where(selected[..., None, None], second, 0.0),
        saddle=saddle,
        saddle_vertex=jnp.where(saddle[:, None], supplied_saddle, 0.0),
    )


def _sampled_spline_edge_roots(
    node_coordinates,
    cell_nodes,
    cell_vertex_count,
    curve_evaluator,
    participating_cell,
):
    """Locate at most two ordered spline roots on every atomic-cell edge."""
    coordinates = jnp.asarray(node_coordinates)
    nodes = jnp.asarray(cell_nodes)
    count = jnp.asarray(cell_vertex_count)
    cell_count, width = nodes.shape
    edge_slot = jnp.arange(width)
    valid_edge = edge_slot[None, :] < count[:, None]
    following_slot = jnp.where(
        edge_slot[None, :] + 1 < count[:, None], edge_slot[None, :] + 1, 0
    )
    following_nodes = jnp.take_along_axis(nodes, following_slot, axis=1)
    start = coordinates[nodes]
    end = coordinates[following_nodes]
    parameter = jnp.linspace(0.0, 1.0, 257, dtype=coordinates.dtype)
    sampled_point = (
        start[:, :, None, :]
        + parameter[None, None, :, None] * (end - start)[:, :, None, :]
    )
    sampled_value = curve_evaluator(
        sampled_point.reshape(cell_count, width * parameter.size, 2)
    ).reshape(cell_count, width, parameter.size)
    sign_change = (sampled_value[..., :-1] > 0.0) != (sampled_value[..., 1:] > 0.0)
    sign_change = sign_change & valid_edge[..., None]
    root_count = jnp.sum(sign_change, axis=2)
    supported = jnp.all(root_count <= 2, axis=1)
    if participating_cell is not None:
        participation = jnp.asarray(participating_cell, dtype=bool)
        if participation.shape != (cell_count,):
            raise ValueError("participating_cell must carry one flag per cell")
        supported = supported & participation

    rank = jnp.cumsum(sign_change, axis=2) - 1
    bracket_index = []
    bracket_valid = []
    for root_slot in range(2):
        selected = sign_change & (rank == root_slot)
        bracket_index.append(jnp.argmax(selected, axis=2))
        bracket_valid.append(jnp.any(selected, axis=2))
    bracket_index = jnp.stack(bracket_index, axis=2)
    bracket_valid = jnp.stack(bracket_valid, axis=2) & supported[:, None, None]
    lower_parameter = parameter[bracket_index]
    upper_parameter = parameter[bracket_index + 1]
    lower_point = (
        start[..., None, :] + lower_parameter[..., None] * (end - start)[..., None, :]
    )
    upper_point = (
        start[..., None, :] + upper_parameter[..., None] * (end - start)[..., None, :]
    )
    lower_value = jnp.take_along_axis(sampled_value, bracket_index, axis=2)
    upper_value = jnp.take_along_axis(sampled_value, bracket_index + 1, axis=2)
    flat_shape = (cell_count, 2 * width)
    local_fraction = _traced_level_segment_root(
        lower_point.reshape(*flat_shape, 2),
        upper_point.reshape(*flat_shape, 2),
        lower_value.reshape(flat_shape),
        upper_value.reshape(flat_shape),
        curve_evaluator,
    ).reshape(cell_count, width, 2)
    fraction = lower_parameter + local_fraction * (upper_parameter - lower_parameter)
    fraction = jnp.where(bracket_valid, fraction, 0.0)
    packed_count = jnp.where(supported[:, None], jnp.minimum(root_count, 2), 0).astype(
        jnp.int32
    )
    return fraction, packed_count, bracket_valid & (upper_value > 0.0)


def _expand_spline_saddle_wedges(
    wedges,
    curve_evaluator,
    arc_tracer,
    straight_capacity,
):
    """Replace both saddle-to-root chords by one fixed sampled chain per wedge."""
    vertices = jnp.asarray(wedges.support_vertices)
    count = jnp.asarray(wedges.vertex_count)
    cell_count = vertices.shape[0]
    capacity = traced_polygon_vertex_capacity(straight_capacity)
    half_arc_slot = jnp.arange(0, _SPLINE_BOUNDARY_SEGMENTS + 1, 2)
    second_half_slot = half_arc_slot[1:-1]
    middle_slot = jnp.arange(2, straight_capacity)
    expanded_vertices = []
    expanded_counts = []
    tracer = _traced_level_arc if arc_tracer is None else arc_tracer

    for wedge in range(4):
        polygon = vertices[:, wedge]
        polygon_count = count[:, wedge]
        last_slot = jnp.maximum(polygon_count - 1, 0)
        previous_slot = jnp.maximum(polygon_count - 2, 0)
        first_root = polygon[:, 1]
        last_root = jnp.take_along_axis(polygon, last_slot[:, None, None], axis=1)[:, 0]
        first_inside = polygon[:, 2]
        last_inside = jnp.take_along_axis(
            polygon, previous_slot[:, None, None], axis=1
        )[:, 0]
        first_half = tracer(
            wedges.saddle_vertex,
            first_root,
            curve_evaluator,
            first_inside,
        )[:, half_arc_slot]
        second_half = tracer(
            last_root,
            wedges.saddle_vertex,
            curve_evaluator,
            last_inside,
        )[:, second_half_slot]
        middle = polygon[:, middle_slot]
        candidate = jnp.concatenate((first_half, middle, second_half), axis=1)
        valid = jnp.concatenate(
            (
                jnp.broadcast_to(
                    wedges.saddle[:, None], (cell_count, first_half.shape[1])
                ),
                wedges.saddle[:, None]
                & (middle_slot[None, :] < polygon_count[:, None]),
                jnp.broadcast_to(
                    wedges.saddle[:, None], (cell_count, second_half.shape[1])
                ),
            ),
            axis=1,
        )
        expanded, expanded_count = _pack_traced_vertices(candidate, valid, capacity)
        expanded_vertices.append(expanded)
        expanded_counts.append(expanded_count)

    vertices = jnp.stack(expanded_vertices, axis=1)
    count = jnp.stack(expanded_counts, axis=1)
    flat_vertices = vertices.reshape(4 * cell_count, capacity, 2)
    flat_count = count.reshape(4 * cell_count)
    flat_centre = jnp.broadcast_to(
        wedges.centroids[:, None, :], (cell_count, 4, 2)
    ).reshape(4 * cell_count, 2)
    area, first, second = _traced_polygon_moments(
        flat_vertices, flat_count, flat_centre
    )
    area = area.reshape(cell_count, 4)
    first = first.reshape(cell_count, 4, 2)
    second = second.reshape(cell_count, 4, 2, 2)
    selected = wedges.saddle[:, None]
    live = jnp.arange(capacity)[None, None, :] < count[..., None]
    return SaddleCellWedges(
        support_vertices=jnp.where(
            (selected[..., None] & live)[..., None], vertices, 0.0
        ),
        vertex_count=jnp.where(selected, count, 0),
        centroids=wedges.centroids,
        area=jnp.where(selected, area, 0.0),
        full_area=wedges.full_area,
        first_area_moment=jnp.where(selected[..., None], first, 0.0),
        second_area_moment=jnp.where(selected[..., None, None], second, 0.0),
        saddle=wedges.saddle,
        saddle_vertex=wedges.saddle_vertex,
    )


@dataclass(frozen=True)
class ClippedSupports:
    """Fixed-shape supports and exact polygon moments for one flux map."""

    support_vertices: np.ndarray
    vertex_count: np.ndarray
    included: np.ndarray
    boundary: np.ndarray
    area: np.ndarray
    first_area_moment: np.ndarray
    second_area_moment: np.ndarray
    contour_vertices: np.ndarray
    contour_vertex_count: int
    contour_area: float
    contour_closed: bool
    patch_area_sum: float

    def linear_current_moments(
        self, density: np.ndarray, gradient: np.ndarray
    ) -> LinearCurrentMoments:
        """Integrate a cellwise-linear current over every clipped support.

        ``density`` is the value at each fixed cell centroid and ``gradient``
        holds its radial and vertical derivatives there.
        """
        density = np.asarray(density, dtype=np.float64)
        gradient = np.asarray(gradient, dtype=np.float64)
        if density.shape != self.area.shape:
            raise ValueError("density must carry one value per cell")
        if gradient.shape != (len(self.area), 2):
            raise ValueError("gradient must have shape (cells, 2)")
        current = density * self.area + np.einsum(
            "ni,ni->n", gradient, self.first_area_moment
        )
        first = density[:, np.newaxis] * self.first_area_moment + np.einsum(
            "nij,nj->ni", self.second_area_moment, gradient
        )
        return LinearCurrentMoments(current=current, first=first)


@dataclass(frozen=True)
class AtomicCellMesh:
    """A fixed cell mesh whose shared edges have identical subdivisions."""

    node_coordinates: np.ndarray
    cell_nodes: np.ndarray
    cell_vertex_count: np.ndarray
    centroids: np.ndarray
    tolerance: float
    support_capacity: int
    contour_capacity: int

    @classmethod
    def from_cells(
        cls,
        cells: Iterable[np.ndarray],
        *,
        centroids: np.ndarray | None = None,
        tolerance: float | None = None,
    ) -> AtomicCellMesh:
        """Atomise shared edges and return the fixed topology.

        Vertices from every collinear overlapping edge become split points on
        each other.  This converts long-edge/short-edge junctions into the same
        undirected atomic edges before any flux interpolation occurs.
        """
        polygons = _normalise_cells(cells)
        all_points = np.vstack(polygons)
        scale = max(float(np.max(np.abs(all_points))), float(np.ptp(all_points)), 1.0)
        if tolerance is None:
            tolerance = 128.0 * np.finfo(np.float64).eps * scale
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")

        line_points: dict[tuple[float, float, float], list[np.ndarray]] = {}
        for polygon in polygons:
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                if np.linalg.norm(end - start) <= tolerance:
                    raise ValueError("cell edges must have non-zero length")
                line_points.setdefault(_line_key(start, end), []).extend((start, end))

        node_lookup: dict[tuple[int, int], int] = {}
        nodes: list[np.ndarray] = []

        def node_index(point: np.ndarray) -> int:
            key = _point_key(point, tolerance)
            if key not in node_lookup:
                node_lookup[key] = len(nodes)
                nodes.append(np.asarray(point, dtype=np.float64))
            return node_lookup[key]

        cell_node_rows: list[list[int]] = []
        for polygon in polygons:
            row: list[int] = []
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                delta = end - start
                length_squared = float(delta @ delta)
                candidates = line_points[_line_key(start, end)]
                split: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
                for point in candidates:
                    fraction = float((point - start) @ delta / length_squared)
                    perpendicular = abs(
                        delta[0] * (point[1] - start[1])
                        - delta[1] * (point[0] - start[0])
                    ) / math.sqrt(length_squared)
                    if (
                        -tolerance <= fraction <= 1.0 + tolerance
                        and perpendicular <= tolerance
                    ):
                        split[_point_key(point, tolerance)] = (fraction, point)
                ordered = sorted(split.values(), key=lambda item: item[0])
                row.extend(node_index(point) for _fraction, point in ordered[:-1])
            if len(row) < 3:
                raise ValueError(
                    "atomisation left a cell with fewer than three vertices"
                )
            cell_node_rows.append(row)

        counts = np.asarray([len(row) for row in cell_node_rows], dtype=np.intp)
        width = int(np.max(counts))
        packed = np.zeros((len(polygons), width), dtype=np.intp)
        for packed_row, row in zip(packed, cell_node_rows, strict=True):
            packed_row[: len(row)] = row

        if centroids is None:
            centre = np.asarray([_centroid(polygon) for polygon in polygons])
        else:
            centre = np.ascontiguousarray(centroids, dtype=np.float64)
            if centre.shape != (len(polygons), 2) or not np.all(np.isfinite(centre)):
                raise ValueError("centroids must have shape (cells, 2) and be finite")
        return cls(
            node_coordinates=np.asarray(nodes),
            cell_nodes=packed,
            cell_vertex_count=counts,
            centroids=centre,
            tolerance=tolerance,
            support_capacity=2 * width,
            contour_capacity=int(np.sum(counts)),
        )

    def sample(
        self, level: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Evaluate a scalar field once at every shared atomic node."""
        values = np.asarray(
            level(self.node_coordinates[:, 0], self.node_coordinates[:, 1]),
            dtype=np.float64,
        )
        if values.shape != (len(self.node_coordinates),):
            raise ValueError("the sampled level field must return one value per node")
        return values

    def traced_clip(
        self,
        signed_flux,
        *,
        saddle_vertex=None,
        curve_coefficient=None,
        curve_centre=None,
        curve_scale=None,
        curve_evaluator=None,
        participating_cell=None,
        arc_tracer: Callable | None = None,
    ) -> TracedClippedSupports:
        """Clip this fixed topology inside a JAX transformation."""
        return _traced_clip(
            self.node_coordinates,
            self.cell_nodes,
            self.cell_vertex_count,
            self.centroids,
            self.support_capacity,
            signed_flux,
            saddle_vertex,
            curve_coefficient,
            curve_centre,
            curve_scale,
            curve_evaluator,
            participating_cell,
            arc_tracer,
        )

    def traced_saddle_wedges(
        self,
        signed_flux,
        *,
        saddle_vertex,
        core_reference,
        participating_cell=None,
        curve_evaluator=None,
        arc_tracer: Callable | None = None,
        edge_root_fraction=None,
        edge_root_count=None,
        edge_root_positive_after=None,
    ) -> SaddleCellWedges:
        """Return the four branch-paired regions of every saddle cell.

        ``signed_flux`` must be positive on the core side. ``core_reference``
        is normally the magnetic axis and distinguishes the core wedge from the
        same-sign private-flux wedge. All non-saddle cells are exact-zero rows.
        """
        explicit_roots = edge_root_fraction is not None
        if explicit_roots != (edge_root_count is not None) or explicit_roots != (
            edge_root_positive_after is not None
        ):
            raise ValueError(
                "edge root fractions, counts, and following signs must be supplied "
                "together"
            )
        if explicit_roots:
            if participating_cell is not None:
                raise ValueError(
                    "explicit saddle roots do not accept a separate participation mask"
                )
            if curve_evaluator is not None or arc_tracer is not None:
                raise ValueError(
                    "explicit saddle roots do not accept a separate spline tracer"
                )
            return _traced_saddle_wedges_from_edge_roots(
                self.node_coordinates,
                self.cell_nodes,
                self.cell_vertex_count,
                self.centroids,
                self.support_capacity,
                signed_flux,
                saddle_vertex,
                core_reference,
                edge_root_fraction,
                edge_root_count,
                edge_root_positive_after,
            )
        if curve_evaluator is not None:
            root_fraction, root_count, positive_after = _sampled_spline_edge_roots(
                self.node_coordinates,
                self.cell_nodes,
                self.cell_vertex_count,
                curve_evaluator,
                participating_cell,
            )
            straight = _traced_saddle_wedges_from_edge_roots(
                self.node_coordinates,
                self.cell_nodes,
                self.cell_vertex_count,
                self.centroids,
                self.support_capacity,
                signed_flux,
                saddle_vertex,
                core_reference,
                root_fraction,
                root_count,
                positive_after,
            )
            return _expand_spline_saddle_wedges(
                straight,
                curve_evaluator,
                arc_tracer,
                self.support_capacity,
            )
        positive = self.traced_clip(
            signed_flux,
            saddle_vertex=saddle_vertex,
            participating_cell=participating_cell,
        )
        negative = self.traced_clip(
            -jnp.asarray(signed_flux),
            saddle_vertex=saddle_vertex,
            participating_cell=participating_cell,
        )
        return _ordered_saddle_wedges(positive, negative, core_reference)

    def clip(self, signed_flux: np.ndarray) -> ClippedSupports:
        """Clip every cell to ``signed_flux > 0`` using shared crossings."""
        flux = np.asarray(signed_flux, dtype=np.float64)
        if flux.shape != (len(self.node_coordinates),):
            raise ValueError("signed_flux must carry one value per atomic node")
        if not np.all(np.isfinite(flux)):
            raise ValueError("signed_flux must be finite")

        cell_count = len(self.cell_nodes)
        support = np.zeros((cell_count, self.support_capacity, 2), dtype=np.float64)
        vertex_count = np.zeros(cell_count, dtype=np.intp)
        included = np.zeros(cell_count, dtype=bool)
        boundary = np.zeros(cell_count, dtype=bool)
        area = np.zeros(cell_count)
        first = np.zeros((cell_count, 2))
        second = np.zeros((cell_count, 2, 2))
        crossing_point: dict[tuple[int, ...], np.ndarray] = {}
        segments: set[frozenset[tuple[int, ...]]] = set()

        for cell_index, (packed_nodes, count, origin) in enumerate(
            zip(self.cell_nodes, self.cell_vertex_count, self.centroids, strict=True)
        ):
            indices = packed_nodes[:count]
            polygon: list[np.ndarray] = []
            crossing_keys: list[tuple[int, ...]] = []
            for start_index, end_index in zip(
                indices, np.roll(indices, -1), strict=True
            ):
                start_inside = flux[start_index] > 0.0
                end_inside = flux[end_index] > 0.0
                if start_inside:
                    polygon.append(self.node_coordinates[start_index])
                if start_inside != end_inside:
                    if flux[start_index] == 0.0:
                        crossing = (int(start_index),)
                    elif flux[end_index] == 0.0:
                        crossing = (int(end_index),)
                    else:
                        crossing = tuple(sorted((int(start_index), int(end_index))))
                    if crossing not in crossing_point:
                        start_flux = flux[start_index]
                        fraction = start_flux / (start_flux - flux[end_index])
                        crossing_point[crossing] = self.node_coordinates[
                            start_index
                        ] + fraction * (
                            self.node_coordinates[end_index]
                            - self.node_coordinates[start_index]
                        )
                    polygon.append(crossing_point[crossing])
                    crossing_keys.append(crossing)

            vertices = []
            for point in polygon:
                if not vertices or not np.array_equal(point, vertices[-1]):
                    vertices.append(point)
            if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
                vertices.pop()
            array = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
            if len(array) > self.support_capacity:
                raise RuntimeError("clipped support exceeded its fixed mesh capacity")
            cell_area, cell_first, cell_second = _area_moments(array, origin)
            if cell_area > 0.0:
                included[cell_index] = True
                support[cell_index, : len(array)] = array
                vertex_count[cell_index] = len(array)
                area[cell_index] = cell_area
                first[cell_index] = cell_first
                second[cell_index] = cell_second

            unique_crossings = list(dict.fromkeys(crossing_keys))
            if len(unique_crossings) > 2:
                raise ValueError(
                    "a cell has more than two separatrix crossings; refine the mesh "
                    "so one linear contour segment crosses each boundary cell"
                )
            if len(unique_crossings) == 2:
                segments.add(frozenset(unique_crossings))
                boundary[cell_index] = cell_area > 0.0

        contour, closed = self._contour(crossing_point, segments)
        contour_count = len(contour)
        padded_contour = np.zeros((self.contour_capacity, 2), dtype=np.float64)
        padded_contour[:contour_count] = contour
        contour_area = abs(_signed_area(contour)) if closed else math.nan
        return ClippedSupports(
            support_vertices=support,
            vertex_count=vertex_count,
            included=included,
            boundary=boundary,
            area=area,
            first_area_moment=first,
            second_area_moment=second,
            contour_vertices=padded_contour,
            contour_vertex_count=contour_count,
            contour_area=contour_area,
            contour_closed=closed,
            patch_area_sum=math.fsum(area),
        )

    def _contour(
        self,
        crossing_point: dict[tuple[int, ...], np.ndarray],
        segments: set[frozenset[tuple[int, ...]]],
    ) -> tuple[np.ndarray, bool]:
        """Traverse the cell-local contour segments into one closed polygon."""
        adjacency: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        for segment in segments:
            if len(segment) != 2:
                continue
            first, second = tuple(segment)
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        if not adjacency:
            points = np.asarray(list(crossing_point.values()), dtype=np.float64)
            return points.reshape(-1, 2), False
        if any(len(neighbours) != 2 for neighbours in adjacency.values()):
            points = np.asarray(
                [crossing_point[key] for key in adjacency], dtype=np.float64
            )
            return points, False

        start = min(adjacency)
        ordered: list[tuple[int, ...]] = []
        previous: tuple[int, ...] | None = None
        current = start
        while True:
            ordered.append(current)
            choices = adjacency[current] - (
                {previous} if previous is not None else set()
            )
            following = min(choices)
            if following == start:
                break
            if following in ordered:
                return np.asarray([crossing_point[key] for key in ordered]), False
            previous, current = current, following
        if set(ordered) != set(adjacency):
            raise ValueError("the sampled flux contains more than one closed contour")
        contour = np.asarray([crossing_point[key] for key in ordered])
        if _signed_area(contour) < 0.0:
            contour = contour[::-1]
        return contour, True
