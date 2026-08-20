r"""Differential receipts on an unstructured cell mesh from neighbour rings.

:class:`~nova.equilibrium.conservation.FluxLattice` reads its derivatives by
central differences, which needs a tensor-product raster of one cell size. The
plasma mesh the package actually ships is not one: cells are hexagons on a
half-offset tiling, trimmed where the first wall cuts them, so rows are offset
by half a pitch, the domain stops at a curved boundary and a clipped cell
carries both a smaller area and a displaced centroid.

This class carries the same differential contract on that mesh. The geometry it
needs is what the grid solve already produces — cell centroids, cell areas, and
the centre-first neighbour rings the tessellation recovers from a Delaunay
triangulation of those centroids, packed exactly as
:func:`nova.geometry.hexstencil.hex_stencil` packs a structured one, so the
null search and the derivative operator read the same rings.

Derivatives come from a least-squares quadratic fitted on each ring. Six
neighbours and the centre give seven samples for the six coefficients of

.. math::
    f \simeq c_0 + c_1 u + c_2 v + c_3 u^2 + c_4 u v + c_5 v^2,

so the fit is overdetermined by one and the derivative at the centre is a fixed
linear functional of the ring values. The coordinates are centred on the ring
centre and scaled by the ring half-width before the fit, the normalisation
:class:`~nova.biot.null.Null2D` already applies, which is what keeps the design
matrix conditioned at a metre-scale major radius: on a regular hexagonal ring
its condition number is 5.5, and a quarter-pitch centroid displacement — far
more than first-wall clipping produces — leaves it under twenty.

A hexagonal ring determines the full quadratic rather than the Laplacian alone.
Restricted to the six ring points the quadratic basis spans the angular modes
:math:`1, \cos\theta, \sin\theta, \cos 2\theta, \sin 2\theta` and the centre
supplies the sixth degree of freedom, so :math:`\partial^2/\partial R^2` and
:math:`\partial^2/\partial Z^2` are separately resolved and not just their sum.
The first mode the ring cannot see is :math:`\cos 3\theta`, which is where the
truncation error lives; both the gradient and the elliptic operator converge at
second order in the pitch.

Cells without a ring — the hull of the tessellation, and any cell a caller
withheld — carry no derivative. They are reported as zero and excluded by
:meth:`StencilMesh.interior`, so a receipt never reads a value the mesh could
not form, exactly as the lattice border is trimmed before a residual is
reported.

``SharedNodeFluxStencil`` remains only as the atomic-node reconstruction used
by exact clip tracing and atomic-node topology labels. Current attribution
uses direct pre-clip sample rows and never consumes that reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import second_moments, section_centroid
from nova.equilibrium.conservation import STENCIL_MARGIN
from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    complete_polynomial_powers,
)

__all__ = [
    "RING_CONDITION_LIMIT",
    "CellCurrentMoments",
    "InteriorCurrentMomentStencil",
    "MomentGeometry",
    "PROFILE_DENSITY_POWERS",
    "SharedNodeFluxStencil",
    "StencilMesh",
    "fixed_profile_current_moments",
    "ring_condition",
]

#: Largest normalised design-matrix condition number a ring may carry. A
#: regular hexagonal ring sits at 5.5 and irregular ones climb slowly, so the
#: limit is far above any tiling and still catches the failure it exists for: a
#: cluster whose points are collinear, coincident or otherwise unable to
#: determine a quadratic, which pinv would answer with a plausible-looking
#: least-norm fit rather than an error.
RING_CONDITION_LIMIT = 1.0e3

#: Total-degree basis used for the clip-independent profile density. The
#: weighted full-cell design is projected once at geometry construction time;
#: exact support moments then require monomials through one degree higher.
PROFILE_DENSITY_POWERS = complete_polynomial_powers(9)

_DENSITY_QUADRATURE_NODE, _DENSITY_QUADRATURE_WEIGHT = np.polynomial.legendre.leggauss(
    8
)
_DENSITY_UNIT_NODE = 0.5 * (_DENSITY_QUADRATURE_NODE + 1.0)
_DENSITY_UNIT_WEIGHT = 0.5 * _DENSITY_QUADRATURE_WEIGHT

#: Coefficients of the fitted quadratic, in design-matrix column order.
_VALUE, _RADIAL, _VERTICAL, _RADIAL_CURVATURE, _CROSS, _VERTICAL_CURVATURE = range(6)


class CellCurrentMoments(NamedTuple):
    """Current and first moments about each fixed cell centroid."""

    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array


@dataclass(frozen=True)
class InteriorCurrentMomentStencil:
    """Fixed own-node projection and exact-support moment geometry."""

    cell_count: int
    ring_centre: np.ndarray | None = field(default=None, repr=False)
    ring_gather_index: np.ndarray | None = field(default=None, repr=False)
    ring_flux_weight: np.ndarray | None = field(default=None, repr=False)
    ring_coordinate_scale: np.ndarray | None = field(default=None, repr=False)
    ring_sampling_centre: np.ndarray | None = field(default=None, repr=False)
    ring_sample_node_count: int = 0

    def __post_init__(self):
        """Store compact contiguous geometry arrays for repeated contractions."""
        for name in (
            "ring_flux_weight",
            "ring_coordinate_scale",
            "ring_sampling_centre",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, np.ascontiguousarray(value, dtype=np.float64)
                )
        for name in ("ring_centre", "ring_gather_index"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, np.ascontiguousarray(value, dtype=np.intp)
                )

    def support_flux_moments(
        self,
        profile,
        centroid_flux,
        sample_flux,
        support,
    ) -> CellCurrentMoments:
        """Integrate one own-node profile polynomial for every carried cell."""
        if self.ring_centre is None or len(self.ring_centre) == 0:
            raise ValueError("own-node profile geometry was not built")

        sample_value = jnp.asarray(sample_flux)
        if sample_value.shape != (self.ring_sample_node_count,):
            raise ValueError("one flux value is needed per direct sampling node")
        centroid_value = jnp.asarray(centroid_flux)
        value_pool = jnp.concatenate([centroid_value, sample_value])
        gathered = value_pool[self.ring_gather_index]
        flux_coefficient = jnp.einsum(
            "rps,rs->rp",
            jnp.asarray(self.ring_flux_weight, dtype=value_pool.dtype),
            gathered,
        )
        ring = self.ring_centre
        current, first = fixed_profile_current_moments(
            profile,
            support.support_vertices[ring],
            support.vertex_count[ring],
            support.centroids[ring],
            jnp.asarray(self.ring_sampling_centre, dtype=value_pool.dtype),
            jnp.asarray(self.ring_coordinate_scale, dtype=value_pool.dtype),
            flux_coefficient,
        )
        entries = jnp.stack([current, first[:, 0], first[:, 1]])
        vectors = jnp.zeros((3, self.cell_count), dtype=entries.dtype)
        vectors = vectors.at[:, ring].set(entries)
        return CellCurrentMoments(*vectors)

    def sample_flux_field(self, centroid_flux, sample_flux, points):
        """Evaluate the own-node quadratic and its gradient at fixed points."""
        if self.ring_centre is None or len(self.ring_centre) == 0:
            raise ValueError("own-node profile geometry was not built")
        sample_value = jnp.asarray(sample_flux)
        centroid_value = jnp.asarray(centroid_flux)
        value_pool = jnp.concatenate([centroid_value, sample_value])
        gathered = value_pool[self.ring_gather_index]
        coefficient = jnp.einsum(
            "rps,rs->rp",
            jnp.asarray(self.ring_flux_weight, dtype=value_pool.dtype),
            gathered,
        )
        ring = self.ring_centre
        query = jnp.asarray(points)[ring]
        centre = jnp.asarray(self.ring_sampling_centre, dtype=value_pool.dtype)
        scale = jnp.asarray(self.ring_coordinate_scale, dtype=value_pool.dtype)
        local = (query - centre[:, None, :]) / scale[:, None, :]
        design = _quadratic_flux_design(local)
        value = jnp.einsum("nqi,ni->nq", design, coefficient)
        radial, vertical = local[..., 0], local[..., 1]
        radial_gradient = (
            coefficient[:, None, 1]
            + 2.0 * coefficient[:, None, 3] * radial
            + coefficient[:, None, 4] * vertical
        ) / scale[:, None, 0]
        vertical_gradient = (
            coefficient[:, None, 2]
            + coefficient[:, None, 4] * radial
            + 2.0 * coefficient[:, None, 5] * vertical
        ) / scale[:, None, 1]
        shape = (self.cell_count, points.shape[1])
        values = jnp.zeros(shape, dtype=value.dtype).at[ring].set(value)
        radial_values = (
            jnp.zeros(shape, dtype=value.dtype).at[ring].set(radial_gradient)
        )
        vertical_values = (
            jnp.zeros(shape, dtype=value.dtype).at[ring].set(vertical_gradient)
        )
        return values, radial_values, vertical_values


@dataclass(frozen=True)
class SharedNodeFluxStencil:
    """Reconstruct atomic-node flux only for clipping and topology labels.

    Current attribution never samples this reconstruction. Its sole role is
    the atomic-node representation shared by exact clip tracing and the
    corresponding atomic-node topology labels.
    """

    gather_index: np.ndarray = field(repr=False)
    weight: np.ndarray = field(repr=False)
    cell_count: int

    def __post_init__(self):
        """Store compact immutable interpolation arrays."""
        gather = np.ascontiguousarray(self.gather_index, dtype=np.intp)
        weight = np.ascontiguousarray(self.weight, dtype=np.float64)
        if gather.shape != weight.shape:
            raise ValueError("shared-node gather indices and weights must align")
        object.__setattr__(self, "gather_index", gather)
        object.__setattr__(self, "weight", weight)

    def __call__(self, cell_flux) -> jax.Array:
        """Evaluate cell-centred flux on every shared atomic node."""
        flux = jnp.asarray(cell_flux)
        if flux.shape != (self.cell_count,):
            raise ValueError("cell_flux must carry one value per cell centroid")
        return jnp.sum(
            jnp.asarray(self.weight, dtype=flux.dtype) * flux[self.gather_index],
            axis=1,
        )


@dataclass(frozen=True)
class MomentGeometry:
    """Fixed polygon topology and interpolation used by current moments."""

    polygons: tuple[np.ndarray, ...] = field(repr=False)
    atomic_mesh: AtomicCellMesh = field(repr=False)
    second_moment: np.ndarray = field(repr=False)
    shared_flux_stencil: SharedNodeFluxStencil = field(repr=False)
    sampling_vertices: tuple[np.ndarray, ...] = field(repr=False)
    sample_node_coordinates: np.ndarray = field(repr=False)
    cell_sample_nodes: np.ndarray = field(repr=False)
    sample_vertex_count: np.ndarray = field(repr=False)

    @classmethod
    def from_cells(
        cls, mesh: StencilMesh, cells, *, sampling_vertices=None
    ) -> MomentGeometry:
        """Build all geometry-dependent current-moment state once per mesh."""
        polygons = []
        for cell in cells:
            vertices = np.asarray(cell, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] < 2:
                raise ValueError(
                    "cell polygons must have shape (vertices, coordinates)"
                )
            vertices = vertices[:, :2]
            if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
                vertices = vertices[:-1]
            scale = max(float(np.max(np.abs(vertices))), float(np.ptp(vertices)), 1.0)
            tolerance = 128.0 * np.finfo(np.float64).eps * scale
            distinct = [vertices[0]]
            for vertex in vertices[1:]:
                if np.linalg.norm(vertex - distinct[-1]) > tolerance:
                    distinct.append(vertex)
            if (
                len(distinct) > 1
                and np.linalg.norm(distinct[-1] - distinct[0]) <= tolerance
            ):
                distinct.pop()
            vertices = np.asarray(distinct)
            if len(vertices) < 3:
                raise ValueError("a moment polygon must retain at least three vertices")
            polygons.append(np.ascontiguousarray(vertices))
        if len(polygons) != mesh.node_count:
            raise ValueError("one polygon is required per mesh cell")
        centroids = np.asarray([section_centroid(cell) for cell in polygons])
        atomic = AtomicCellMesh.from_cells(polygons, centroids=centroids)
        moments = np.asarray([second_moments(cell) for cell in polygons])
        if sampling_vertices is None:
            sampling = tuple(np.asarray(polygon) for polygon in polygons)
        else:
            sampling = tuple(
                np.ascontiguousarray(vertices, dtype=np.float64)
                for vertices in sampling_vertices
            )
            if len(sampling) != mesh.node_count:
                raise ValueError("one sampling polygon is required per mesh cell")
            if any(
                vertices.ndim != 2
                or vertices.shape[1] != 2
                or len(vertices) not in (4, 6)
                for vertices in sampling
            ):
                raise ValueError(
                    "sampling polygons must have four or six two-dimensional vertices"
                )
        sample_nodes: list[np.ndarray] = []
        sample_lookup: dict[tuple[int, int], int] = {}
        sample_count = np.asarray(
            [len(vertices) for vertices in sampling], dtype=np.intp
        )
        sample_width = int(np.max(sample_count))
        cell_sample_nodes = np.zeros((mesh.node_count, sample_width), dtype=np.intp)
        for cell, vertices in enumerate(sampling):
            for corner, vertex in enumerate(vertices):
                key = tuple(np.rint(vertex / atomic.tolerance).astype(np.int64))
                if key not in sample_lookup:
                    sample_lookup[key] = len(sample_nodes)
                    sample_nodes.append(vertex)
                cell_sample_nodes[cell, corner] = sample_lookup[key]
        sample_coordinate = np.asarray(sample_nodes)
        return cls(
            polygons=tuple(polygons),
            atomic_mesh=atomic,
            second_moment=moments,
            shared_flux_stencil=mesh.shared_node_flux_stencil(atomic.node_coordinates),
            sampling_vertices=sampling,
            sample_node_coordinates=sample_coordinate,
            cell_sample_nodes=cell_sample_nodes,
            sample_vertex_count=sample_count,
        )

    def shared_node_flux(self, cell_flux) -> jax.Array:
        """Evaluate one cell-centred flux map on the atomic shared nodes."""
        return self.shared_flux_stencil(cell_flux)


def _quadratic_design(local: np.ndarray) -> np.ndarray:
    """Return the quadratic design matrix of normalised ring coordinates."""
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def _quadratic_flux_design(local):
    """Evaluate the complete quadratic basis with the input array namespace."""
    radial, vertical = local[..., 0], local[..., 1]
    return jnp.stack(
        [
            jnp.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def fixed_profile_current_moments(
    profile,
    support_vertices,
    vertex_count,
    moment_centre,
    sampling_centre,
    coordinate_scale,
    flux_coefficient,
):
    """Integrate density moments by the fixed degree-fifteen Duffy rule."""
    return _direct_profile_current_moments(
        profile,
        support_vertices,
        vertex_count,
        moment_centre,
        sampling_centre,
        coordinate_scale,
        flux_coefficient,
    )


def _direct_profile_current_moments(
    profile,
    support_vertices,
    vertex_count,
    moment_centre,
    sampling_centre,
    coordinate_scale,
    flux_coefficient,
):
    """Integrate density and first moments with a fixed Duffy product rule."""
    vertices = jnp.asarray(support_vertices)
    count = jnp.asarray(vertex_count)
    capacity = vertices.shape[1]
    triangle_slot = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:, :1], (len(vertices), capacity - 2, 2))
    second = vertices[:, triangle_slot]
    third = vertices[:, triangle_slot + 1]
    node = jnp.asarray(_DENSITY_UNIT_NODE, dtype=vertices.dtype)
    node_weight = jnp.asarray(_DENSITY_UNIT_WEIGHT, dtype=vertices.dtype)
    radial, vertical = jnp.meshgrid(node, node, indexing="ij")
    radial_weight, vertical_weight = jnp.meshgrid(
        node_weight, node_weight, indexing="ij"
    )
    radial = radial.reshape(-1)
    vertical = vertical.reshape(-1)
    rule_weight = (radial_weight * vertical_weight).reshape(-1)
    edge_first = second - first
    edge_second = third - first
    points = (
        first[:, :, None, :]
        + radial[None, None, :, None] * edge_first[:, :, None, :]
        + (1.0 - radial)[None, None, :, None]
        * vertical[None, None, :, None]
        * edge_second[:, :, None, :]
    )
    cross = jnp.abs(
        edge_first[..., 0] * edge_second[..., 1]
        - edge_first[..., 1] * edge_second[..., 0]
    )
    live = triangle_slot[None, :] + 1 < count[:, None]
    weights = (
        cross[:, :, None] * (1.0 - radial)[None, None, :] * rule_weight[None, None, :]
    )
    weights = jnp.where(live[:, :, None], weights, 0.0)
    points = points.reshape(len(vertices), -1, 2)
    weights = weights.reshape(len(vertices), -1)
    points = jnp.where(
        (weights > 0.0)[..., None], points, jnp.asarray(sampling_centre)[:, None, :]
    )
    local = (points - jnp.asarray(sampling_centre)[:, None, :]) / jnp.asarray(
        coordinate_scale
    )[:, None, :]
    flux = jnp.einsum("nqi,ni->nq", _quadratic_flux_design(local), flux_coefficient)
    density = profile.current_density(points[..., 0], flux)
    weighted_density = density * weights
    current = jnp.sum(weighted_density, axis=1)
    first = jnp.sum(
        weighted_density[..., None] * (points - jnp.asarray(moment_centre)[:, None, :]),
        axis=1,
    )
    included = count >= 3
    return (
        jnp.where(included, current, 0.0),
        jnp.where(included[:, None], first, 0.0),
    )


def _polygon_monomial_integral(
    vertices: np.ndarray, radial_power: int, vertical_power: int
) -> float:
    """Return one exact signed-fan monomial integral on a local polygon."""
    total_degree = radial_power + vertical_power
    total = 0.0
    area_twice = 0.0
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        cross = first[0] * second[1] - second[0] * first[1]
        area_twice += cross
        edge = 0.0
        for radial_first in range(radial_power + 1):
            radial = (
                math.comb(radial_power, radial_first)
                * first[0] ** radial_first
                * second[0] ** (radial_power - radial_first)
            )
            for vertical_first in range(vertical_power + 1):
                first_degree = radial_first + vertical_first
                simplex = (
                    math.factorial(first_degree)
                    * math.factorial(total_degree - first_degree)
                    / math.factorial(total_degree + 2)
                )
                vertical = (
                    math.comb(vertical_power, vertical_first)
                    * first[1] ** vertical_first
                    * second[1] ** (vertical_power - vertical_first)
                )
                edge += simplex * radial * vertical
        total += cross * edge
    return math.copysign(1.0, area_twice) * total


def _normalised_ring(coordinate: np.ndarray, stencil: np.ndarray):
    """Return every ring centred on its own centre and scaled to unit width."""
    cluster = coordinate[stencil]
    offset = cluster - cluster[:, :1]
    scale = np.max(np.abs(offset), axis=1)
    if np.any(scale <= 0.0):
        raise ValueError("every ring must span both coordinate axes")
    return offset / scale[:, np.newaxis, :], scale, cluster


def ring_condition(coordinate, stencil) -> np.ndarray:
    """Return the conditioning of the quadratic fit on every ring.

    A caller that assembles rings from a tessellation reads this to decide
    which ones to hand over. :class:`StencilMesh` REFUSES a ring it cannot fit
    rather than answering with a least-norm one, so the selection has to be
    made deliberately, and this is what it is made on: the shape of the
    neighbourhood, measured after the centring and scaling the fit applies,
    and therefore independent of the major radius the ring sits at.
    """
    local, _scale, _cluster = _normalised_ring(
        np.ascontiguousarray(coordinate, dtype=np.float64), np.asarray(stencil)
    )
    return np.linalg.cond(_quadratic_design(local))


@dataclass(frozen=True)
class StencilMesh:
    """Cell mesh whose derivatives are fitted on centre-first neighbour rings.

    ``coordinate`` holds the ``(radius, height)`` centroid of every cell,
    ``area`` its poloidal cross-section, and ``stencil`` the neighbour rings:
    one row per cell that carries a derivative, its own index in column zero
    and its neighbours after it. A cell may appear in any number of rings but
    may centre only one.
    """

    coordinate: np.ndarray
    stencil: np.ndarray
    area: np.ndarray
    radial_weight: np.ndarray = field(init=False, repr=False)
    vertical_weight: np.ndarray = field(init=False, repr=False)
    elliptic_weight: np.ndarray = field(init=False, repr=False)
    ring_condition: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Validate the mesh and fit the derivative weights of every ring."""
        coordinate = np.ascontiguousarray(self.coordinate, dtype=np.float64)
        if coordinate.ndim != 2 or coordinate.shape[1] != 2:
            raise ValueError("mesh coordinates must have shape (cells, 2)")
        if np.any(coordinate[:, 0] <= 0.0):
            raise ValueError("cell radius must be strictly positive")
        area = np.ascontiguousarray(self.area, dtype=np.float64)
        if area.shape != (coordinate.shape[0],):
            raise ValueError("one area is needed per cell")
        stencil = np.ascontiguousarray(self.stencil, dtype=np.intp)
        if stencil.ndim != 2 or stencil.shape[1] < 6:
            raise ValueError(
                "a quadratic fit needs rings of at least six cells, centre first"
            )
        if stencil.size and (stencil.min() < 0 or stencil.max() >= len(coordinate)):
            raise ValueError("a ring indexes a cell the mesh does not carry")
        centre = stencil[:, 0]
        if len(np.unique(centre)) != len(centre):
            raise ValueError("a cell may centre at most one ring")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "stencil", stencil)
        self._fit_rings()

    def _fit_rings(self):
        """Solve the normalised quadratic fit of every ring for its weights."""
        local, scale, cluster = _normalised_ring(self.coordinate, self.stencil)
        design = _quadratic_design(local)
        condition = np.linalg.cond(design)
        if np.any(condition > RING_CONDITION_LIMIT):
            worst = int(np.argmax(condition))
            raise ValueError(
                f"ring {worst} centred on cell {self.stencil[worst, 0]} cannot "
                f"determine a quadratic (condition {condition[worst]:.3e})"
            )
        inverse = np.linalg.pinv(design)
        radial = inverse[:, _RADIAL] / scale[:, :1]
        vertical = inverse[:, _VERTICAL] / scale[:, 1:]
        curvature = (
            2.0 * inverse[:, _RADIAL_CURVATURE] / scale[:, :1] ** 2
            + 2.0 * inverse[:, _VERTICAL_CURVATURE] / scale[:, 1:] ** 2
        )
        object.__setattr__(self, "radial_weight", radial)
        object.__setattr__(self, "vertical_weight", vertical)
        object.__setattr__(
            self, "elliptic_weight", curvature - radial / cluster[:, :1, 0]
        )
        object.__setattr__(self, "ring_condition", condition)

    @property
    def node_count(self) -> int:
        """Return the cell count."""
        return len(self.coordinate)

    @property
    def node_radius(self) -> np.ndarray:
        """Return the major radius [m] of every cell centroid."""
        return self.coordinate[:, 0]

    @property
    def cell_area(self) -> np.ndarray:
        """Return the poloidal cross-section [m^2] of every cell."""
        return self.area

    @property
    def centre(self) -> np.ndarray:
        """Return the cells that carry a derivative, in ring order."""
        return self.stencil[:, 0]

    def _scatter(self, ring_value) -> jax.Array:
        """Return one per-ring value placed on its centre cell, zero elsewhere."""
        return (
            jnp.zeros(self.node_count, dtype=ring_value.dtype)
            .at[self.centre]
            .set(ring_value)
        )

    def _apply(self, weight: np.ndarray, values) -> jax.Array:
        """Return one fitted derivative of the ring values."""
        return jnp.sum(jnp.asarray(weight, dtype=values.dtype) * values, axis=1)

    def gradient(self, field) -> tuple[jax.Array, jax.Array]:
        """Return the radial and vertical derivative of one cell field."""
        values = jnp.asarray(field)[self.stencil]
        return (
            self._scatter(self._apply(self.radial_weight, values)),
            self._scatter(self._apply(self.vertical_weight, values)),
        )

    def shared_node_flux_stencil(self, coordinates) -> SharedNodeFluxStencil:
        """Fit fixed weights that reconstruct flux at arbitrary shared nodes.

        Each node uses the nearest complete neighbour ring. The reconstruction
        evaluates the same normalised quadratic fit as the mesh derivatives,
        so it is exact on that fitted polynomial space and remains one fixed
        gather and reduction while the flux map changes.
        """
        query = np.ascontiguousarray(coordinates, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("shared-node coordinates must have shape (nodes, 2)")
        centre_coordinate = self.coordinate[self.centre]
        distance_squared = np.sum(
            (query[:, np.newaxis, :] - centre_coordinate[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        owner = np.argmin(distance_squared, axis=1)
        local, scale, cluster = _normalised_ring(self.coordinate, self.stencil)
        inverse = np.linalg.pinv(_quadratic_design(local))
        query_local = (query - cluster[owner, 0]) / scale[owner]
        basis = _quadratic_design(query_local)
        weight = np.einsum("ni,nij->nj", basis, inverse[owner])
        return SharedNodeFluxStencil(
            gather_index=self.stencil[owner],
            weight=weight,
            cell_count=self.node_count,
        )

    def current_moment_stencil(
        self,
        *,
        support_centre=None,
        sampling_node_coordinate=None,
        sampling_cell_node=None,
    ) -> InteriorCurrentMomentStencil:
        """Build the fixed own-node density projection and support integrator."""
        ring_centre = None
        ring_gather_index = None
        ring_flux_weight = None
        ring_coordinate_scale = None
        ring_sampling_centre = None
        ring_sample_node_count = 0
        if support_centre is not None:
            if sampling_node_coordinate is None or sampling_cell_node is None:
                raise ValueError(
                    "support centres require sampling coordinates and cell-node indices"
                )
            support_centre = np.ascontiguousarray(support_centre, dtype=np.intp)
            ring_centre = support_centre
            sampling_node_coordinate = np.ascontiguousarray(
                sampling_node_coordinate, dtype=np.float64
            )
            sampling_cell_node = np.ascontiguousarray(sampling_cell_node, dtype=np.intp)
            if sampling_cell_node.shape[0] != self.node_count:
                raise ValueError("sampling cell-node rows must match the mesh")
            ring_sample_node_count = len(sampling_node_coordinate)
            ring_gather_index = np.column_stack(
                [
                    ring_centre,
                    self.node_count + sampling_cell_node[ring_centre],
                ]
            )
            ring_sampling_centre = self.coordinate[ring_centre]
            ring_vertices = sampling_node_coordinate[sampling_cell_node[ring_centre]]
            ring_offset = ring_vertices - ring_sampling_centre[:, None, :]
            ring_coordinate_scale = np.max(np.abs(ring_offset), axis=1)
            if np.any(ring_coordinate_scale <= 0.0):
                raise ValueError("every sampling polygon must span both coordinates")
            ring_points = np.concatenate(
                [ring_sampling_centre[:, None, :], ring_vertices], axis=1
            )
            ring_local = (
                ring_points - ring_sampling_centre[:, None, :]
            ) / ring_coordinate_scale[:, None, :]
            ring_flux_weight = np.linalg.pinv(_quadratic_design(ring_local))
        return InteriorCurrentMomentStencil(
            cell_count=self.node_count,
            ring_centre=ring_centre,
            ring_gather_index=ring_gather_index,
            ring_flux_weight=ring_flux_weight,
            ring_coordinate_scale=ring_coordinate_scale,
            ring_sampling_centre=ring_sampling_centre,
            ring_sample_node_count=ring_sample_node_count,
        )

    def delta_star(self, flux) -> jax.Array:
        """Return the elliptic operator value [Wb/m^2] of one flux map.

        The radial, vertical and curvature weights of a ring act on the same
        seven values, so the whole operator is carried by one weight vector
        rather than by three fits of the same quadratic.
        """
        values = jnp.asarray(flux)[self.stencil]
        return self._scatter(self._apply(self.elliptic_weight, values))

    def erode(self, mask, margin: int) -> jax.Array:
        """Return a cell mask shrunk by ``margin`` successive ring erosions."""
        eroded = jnp.asarray(mask, dtype=bool)
        for _ in range(margin):
            eroded = self._scatter(jnp.all(eroded[self.stencil], axis=1))
        return eroded

    def interior(self, margin: int = STENCIL_MARGIN) -> jax.Array:
        """Return the mask of cells whose ring neighbourhood is complete."""
        return self.erode(jnp.ones(self.node_count, dtype=bool), margin)
