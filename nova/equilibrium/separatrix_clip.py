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

import numpy as np

__all__ = [
    "AtomicCellMesh",
    "ClippedSupports",
    "LinearCurrentMoments",
    "TracedClippedSupports",
    "padded_linear_current_moments",
]


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
    """JAX arrays describing fixed-capacity supports for one flux map."""

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

    def linear_current_moments(self, density, gradient):
        """Contract a cellwise-linear current over the traced supports."""
        return padded_linear_current_moments(
            self.support_vertices,
            self.vertex_count,
            self.centroids,
            density,
            gradient,
        )

    def evaluation_weights(self, epsilon):
        """Return C1 participation and clipped-evaluation weights.

        These weights choose between already-evaluated current paths. They do
        not alter the exact clipped polygon or any of its area moments.
        """
        import jax.numpy as jnp

        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must lie between zero and one")
        if epsilon == 0.0:
            return self.included.astype(self.area.dtype), self.boundary.astype(
                self.area.dtype
            )
        fraction = jnp.clip(self.area / self.full_area, 0.0, 1.0)

        def smoothstep(value):
            scaled = jnp.clip(value / epsilon, 0.0, 1.0)
            return scaled * scaled * (3.0 - 2.0 * scaled)

        return smoothstep(fraction), smoothstep(1.0 - fraction)


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


def _pack_traced_vertices(vertices, valid, capacity):
    """Compact masked vertices without data-dependent array shapes."""
    import jax.numpy as jnp

    rank = jnp.cumsum(valid, axis=1) - 1
    destination = jnp.arange(capacity)
    selector = valid[:, :, jnp.newaxis] & (
        rank[:, :, jnp.newaxis] == destination[jnp.newaxis, jnp.newaxis, :]
    )
    packed = jnp.einsum("cvs,cvd->csd", selector, vertices)
    return packed, jnp.sum(valid, axis=1)


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
):
    """Clip fixed atomic cells using only traced fixed-shape operations."""
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
    start_flux = flux[nodes]
    end_flux = flux[following_nodes]
    start_inside = start_flux > 0.0
    end_inside = end_flux > 0.0
    crossing_edge = valid_edge & (start_inside != end_inside)
    denominator = start_flux - end_flux
    fraction = jnp.where(crossing_edge, start_flux / denominator, 0.0)
    start_point = coordinates[nodes]
    end_point = coordinates[following_nodes]
    crossing_point = start_point + fraction[..., None] * (end_point - start_point)

    candidates = jnp.stack([start_point, crossing_point], axis=2).reshape(
        cell_count, 2 * width, 2
    )
    candidate_valid = jnp.stack(
        [valid_edge & start_inside, crossing_edge], axis=2
    ).reshape(cell_count, 2 * width)
    compact, compact_count = _pack_traced_vertices(
        candidates, candidate_valid, support_capacity
    )
    compact_slot = jnp.arange(support_capacity)
    compact_valid = compact_slot[None, :] < compact_count[:, None]
    previous = jnp.roll(compact, 1, axis=1)
    distinct = jnp.any(compact != previous, axis=2)
    keep = compact_valid & ((compact_slot[None, :] == 0) | distinct)
    support, vertex_count = _pack_traced_vertices(compact, keep, support_capacity)
    last_slot = jnp.maximum(vertex_count - 1, 0)
    last = jnp.take_along_axis(support, last_slot[:, None, None], axis=1)[:, 0]
    repeated_closure = (vertex_count > 1) & jnp.all(last == support[:, 0], axis=1)
    vertex_count = vertex_count - repeated_closure.astype(vertex_count.dtype)
    support = jnp.where(
        compact_slot[None, :, None] < vertex_count[:, None, None], support, 0.0
    )

    full_area, _full_first, _full_second = _traced_polygon_moments(
        start_point, count, centre
    )
    area, first, second = _traced_polygon_moments(support, vertex_count, centre)
    included = area > 0.0
    vertex_count = jnp.where(included, vertex_count, 0)
    support = jnp.where(included[:, None, None], support, 0.0)

    edge_number = jnp.arange(width)
    same_crossing = jnp.all(
        crossing_point[:, :, None, :] == crossing_point[:, None, :, :], axis=3
    )
    earlier = edge_number[None, None, :] < edge_number[None, :, None]
    duplicate = jnp.any(same_crossing & crossing_edge[:, None, :] & earlier, axis=2)
    unique_crossing = crossing_edge & ~duplicate
    crossing, crossing_count = _pack_traced_vertices(
        crossing_point, unique_crossing, width
    )
    boundary = included & (crossing_count == 2)
    leaving, leaving_count = _pack_traced_vertices(
        crossing_point, crossing_edge & start_inside, width
    )
    entering, entering_count = _pack_traced_vertices(
        crossing_point, crossing_edge & end_inside, width
    )
    contour_cross = (
        leaving[:, 0, 0] * entering[:, 0, 1] - entering[:, 0, 0] * leaving[:, 0, 1]
    )
    contour_area = 0.5 * jnp.abs(
        jnp.sum(
            jnp.where((leaving_count > 0) & (entering_count > 0), contour_cross, 0.0)
        )
    )
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

    def traced_clip(self, signed_flux) -> TracedClippedSupports:
        """Clip this fixed topology inside a JAX transformation."""
        return _traced_clip(
            self.node_coordinates,
            self.cell_nodes,
            self.cell_vertex_count,
            self.centroids,
            self.support_capacity,
            signed_flux,
        )

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
