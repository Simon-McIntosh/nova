r"""Traced free-boundary flux map behind the forward equilibrium solve.

This is the implementation-level operator of
:class:`nova.equilibrium.forward.ForwardProfile`, not a second public
equilibrium problem. It carries one write-then-read cycle of the
free-boundary map: a trial total poloidal flux is read for its topology, the
prescribed flux functions are evaluated on the domains that read labels, and
the resulting cell currents are mapped back to flux through the precomputed
coupling operators. A root of :meth:`ForwardFluxOperator.residual` is a
free-boundary equilibrium.

Without a declared target the supplied source reaches the current image
unchanged. A caller may instead declare a scalar plasma current; the operator
then eliminates one common profile amplitude from the exact clipped current
moments and applies it to every component before forming the flux image.

The operator is immutable host-and-device state that the traced maps close
over rather than a traced argument, so a flux function may be any callable —
an interpolant carrying device arrays or a closed-form profile. Everything
that varies across a batch (the trial flux, the conductor currents) is an
explicit argument, which is what ``jit``, ``vmap`` and ``grad`` need.

All fluxes are total poloidal fluxes, :math:`\Phi = 2 \pi R A_\phi` in Wb,
concatenated over the plasma grid nodes, wall nodes and, when present, the
direct pre-clip sample nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.cell_partition import cell_partition_geometry
from nova.equilibrium.connectivity_boundary import (
    traced_boundary_read,
    wall_height_shadow_mask,
)
from nova.equilibrium.observation import (
    ClippedIntegralMeasure,
    clipped_support_quadrature,
)
from nova.equilibrium.source import (
    SCALAR_CURRENT_AMPLITUDE_BAND,
    CurrentNormalisationError,
    ForwardSource,
)
from nova.equilibrium.stencil_mesh import (
    CellCurrentMoments,
    MomentGeometry,
    StencilMesh,
)
from nova.equilibrium.topology import (
    Topology,
    TopologyState,
    require_qualified_axis,
)

__all__ = [
    "axis_cell_seed",
    "ForwardFluxOperator",
    "ForwardTopologyState",
    "PrescribedCurrentField",
]

_PRODUCTION_STATIONARY_POINT_CAPACITY = 30
"""Candidate slots retained by the topology reader used by forward solves."""


@jax.jit
def axis_cell_seed(coordinate, axis, inside_material):
    """Return the hex cell owning a continuous axis and its usable material mask.

    The topology grid stores one coordinate per plasma cell. Cell ownership is
    the nearest-centre Voronoi partition of the centre-first hex mesh, including
    at a wall-trimmed support whose stored material flag was decided from the
    cell centre. A continuous magnetic axis inside that support must remain an
    occupiable flood seed even when the centre-only material flag is false.
    Exactly that owning cell is admitted; no neighbouring material flag changes.
    """
    point = jnp.asarray(coordinate)
    continuous_axis = jnp.asarray(axis, dtype=point.dtype)
    material = jnp.asarray(inside_material, dtype=bool)
    distance_squared = jnp.sum((point - continuous_axis) ** 2, axis=1)
    owner = jnp.argmin(distance_squared)
    seed = jnp.arange(point.shape[0]) == owner
    return seed, material | seed


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ForwardTopologyState:
    """Topology landmarks plus one achieved connectivity classification."""

    axis: jax.Array
    axis_flux: jax.Array
    boundary: jax.Array
    boundary_flux: jax.Array
    x_point: jax.Array
    x_point_flux: jax.Array
    wall_point: jax.Array
    wall_point_flux: jax.Array
    _class_margin_read: Callable[[], jax.Array] = field(repr=False)

    @cached_property
    def _class_margin(self) -> jax.Array:
        """Cache the single saddle-aware connectivity comparator read."""

        return self._class_margin_read()

    @property
    def class_margin(self) -> jax.Array:
        """Return the continuous saddle-aware connectivity margin."""

        return self._class_margin

    @property
    def class_determinate(self) -> jax.Array:
        """Return whether the connectivity comparator resolved a class."""

        return jnp.logical_not(jnp.isnan(self.class_margin))

    @property
    def diverted(self) -> jax.Array:
        """Return the achieved saddle-aware class.

        Positive infinity is a resolved diverted result. A NaN margin remains
        explicitly indeterminate through :attr:`class_determinate`; the
        Boolean is false so an unresolved class cannot qualify a diverted
        branch.
        """

        return self.class_determinate & (self.class_margin >= 0)

    @property
    def flux_span(self) -> jax.Array:
        """Return the total poloidal flux [Wb] from the axis to the boundary."""
        return self.boundary_flux - self.axis_flux

    def tree_flatten(self):
        """Return landmarks and both values from one comparator read."""

        class_margin = self.class_margin
        return (
            (
                self.axis,
                self.axis_flux,
                self.boundary,
                self.boundary_flux,
                self.x_point,
                self.x_point_flux,
                self.wall_point,
                self.wall_point_flux,
                self.class_determinate & (class_margin >= 0),
                class_margin,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a topology state without carrying a callable as a leaf."""
        del aux_data
        *landmarks, _diverted, class_margin = children
        return cls(*landmarks, lambda: class_margin)


class _FrozenTopologyPartition(NamedTuple):
    """Discrete topology authority retained through one Newton solve."""

    label: jax.Array
    topology: TopologyState
    profile_support: object
    residual_shadow: jax.Array


def _structured_grid_axes(coordinate) -> tuple[np.ndarray, np.ndarray]:
    """Recover the tensor-product axes carried by a forward grid."""
    points = np.asarray(coordinate, dtype=np.float64)
    radius = np.unique(points[:, 0])
    height = np.unique(points[:, 1])
    expected = np.c_[
        np.repeat(radius, height.size),
        np.tile(height, radius.size),
    ]
    if points.shape != expected.shape or not np.allclose(
        points, expected, rtol=0.0, atol=0.0
    ):
        raise ValueError("connectivity topology requires a tensor-product forward grid")
    return radius, height


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _FixedDesignNull2D:
    """Locate grid nulls without differentiating a fixed SVD factorisation.

    The quadratic design depends only on immutable grid geometry.  Applying its
    host-built pseudoinverse as a matrix is algebraically the same least-squares
    fit while keeping JAX differentiation on the sampled flux alone.  This
    avoids the SVD JVP's undefined singular-vector derivative when a symmetric
    stencil has repeated singular values.
    """

    locator: Null2D
    fit_weight: jax.Array = field(repr=False)

    @classmethod
    def from_locator(cls, locator: Null2D) -> _FixedDesignNull2D:
        """Precompute one quadratic least-squares operator per grid stencil."""
        local = np.asarray(locator.local_coordinate_stencil, dtype=np.float64)
        radial = local[..., 0]
        vertical = local[..., 1]
        design = np.stack(
            (
                radial**2,
                vertical**2,
                radial,
                vertical,
                radial * vertical,
                np.ones_like(radial),
            ),
            axis=-1,
        )
        weight = np.linalg.pinv(design)
        return cls(locator=locator, fit_weight=jnp.asarray(weight, locator.fit_dtype))

    @property
    def coordinate(self):
        """Return the physical grid-node coordinates."""
        return self.locator.coordinate

    @property
    def node_number(self):
        """Return the physical grid-node count."""
        return self.locator.node_number

    @property
    def fit_dtype(self):
        """Return the dtype of the local quadratic fits."""
        return self.locator.fit_dtype

    @jax.jit
    def _candidate_census(self, psi):
        """Fit every stencil and return supported stationary-point candidates."""
        sampled = jnp.asarray(psi, dtype=self.fit_dtype)[self.locator.stencil]
        coefficient = jnp.einsum("...ij,...j->...i", self.fit_weight, sampled)
        determinant = (
            4.0 * coefficient[..., 0] * coefficient[..., 1] - coefficient[..., 4] ** 2
        )
        determinant_floor = jnp.asarray(1.0e-12, coefficient.dtype)
        nonsingular = jnp.abs(determinant) >= determinant_floor
        safe_determinant = jnp.where(nonsingular, determinant, 1.0)
        local_radial = (
            coefficient[..., 4] * coefficient[..., 3]
            - 2.0 * coefficient[..., 1] * coefficient[..., 2]
        ) / safe_determinant
        local_vertical = (
            coefficient[..., 4] * coefficient[..., 2]
            - 2.0 * coefficient[..., 0] * coefficient[..., 3]
        ) / safe_determinant
        local_flux = (
            coefficient[..., 0] * local_radial**2
            + coefficient[..., 1] * local_vertical**2
            + coefficient[..., 2] * local_radial
            + coefficient[..., 3] * local_vertical
            + coefficient[..., 4] * local_radial * local_vertical
            + coefficient[..., 5]
        )
        support = jnp.max(jnp.abs(self.locator.local_coordinate_stencil), axis=1)
        supported = (
            nonsingular
            & (jnp.abs(local_radial) <= support[:, 0])
            & (jnp.abs(local_vertical) <= support[:, 1])
        )
        kind = jnp.where(
            determinant < 0.0,
            0.0,
            jnp.where(
                (coefficient[..., 0] > 0.0) & (coefficient[..., 1] > 0.0),
                -1.0,
                jnp.where(
                    (coefficient[..., 0] < 0.0) & (coefficient[..., 1] < 0.0),
                    1.0,
                    jnp.nan,
                ),
            ),
        )
        origin = self.locator.physical_origin
        scale = self.locator.physical_scale
        physical = origin + jnp.stack((local_radial, local_vertical), axis=-1) * scale
        result = jnp.column_stack((physical, local_flux, kind))
        finite = supported & jnp.all(jnp.isfinite(result), axis=1)
        masks = jnp.stack((finite & (kind != 0.0), finite & (kind == 0.0)))
        return result, masks

    @jax.jit
    def candidate_table_status(self, psi):
        """Report whether fitted stationary points exceeded the retained table."""
        _candidates, masks = self._candidate_census(psi)
        counts = jnp.sum(masks, axis=1, dtype=jnp.int32)
        capacity = jnp.full(counts.shape, self.locator.maxsize, dtype=jnp.int32)
        return {
            "candidate_count": counts,
            "capacity": capacity,
            "truncated": counts > capacity,
        }

    @jax.jit
    def __call__(self, psi):
        """Return extrema and saddles from fixed quadratic fit matrices."""
        candidates, masks = self._candidate_census(psi)
        number = jnp.sum(masks, axis=1, dtype=jnp.int32)
        index = jnp.stack(
            [
                jnp.where(
                    mask,
                    size=self.locator.maxsize,
                    fill_value=0,
                )[0]
                for mask in masks
            ]
        )
        result = candidates[index]
        position = jnp.arange(1, self.locator.maxsize + 1)
        return jnp.where(
            position[None, :, None] <= number[:, None, None],
            result,
            jnp.full_like(result, jnp.nan),
        )

    def tree_flatten(self):
        """Return the locator and its fixed least-squares weights."""
        return (self.locator, self.fit_weight), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a fixed-design locator from JAX pytree leaves."""
        del aux_data
        return cls(*children)


@dataclass(frozen=True)
class PrescribedCurrentField:
    """Fixed conductor currents and their total-flux response matrix."""

    response: jnp.ndarray = field(repr=False)
    current: jnp.ndarray = field(repr=False)

    def __post_init__(self):
        """Validate one response column for every prescribed current."""
        response = jnp.asarray(self.response)
        current = jnp.asarray(self.current)
        if response.ndim != 2:
            raise ValueError("prescribed current response must be a matrix")
        if current.ndim != 1:
            raise ValueError("prescribed current must be a vector")
        if response.shape[1] != current.size:
            raise ValueError(
                "prescribed current response columns must match the current vector"
            )
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "current", current)

    @property
    def circuit_count(self) -> int:
        """Return the number of prescribed circuit currents."""
        return self.current.size

    def flux(self, current=None) -> jax.Array:
        """Return the prescribed conductor flux [Wb] at every target.

        An explicitly supplied vector replaces the stored circuit state.  The
        response remains immutable geometry while same-shaped current edits
        stay ordinary traced data.
        """
        conductor = self.current if current is None else jnp.asarray(current)
        if conductor.shape != self.current.shape:
            raise ValueError(
                "prescribed current must match the stored circuit vector shape"
            )
        return self.response @ conductor

    def flux_delta(self, current_delta) -> jax.Array:
        """Return the flux image of one same-shaped circuit-current edit [Wb]."""
        delta = jnp.asarray(current_delta, dtype=self.current.dtype)
        if delta.shape != self.current.shape:
            raise ValueError(
                "prescribed current delta must match the stored circuit vector shape"
            )
        return self.response @ delta


@dataclass
class ForwardFluxOperator:
    """Map a trial poloidal flux to the flux its plasma current generates."""

    grid: FluxTarget
    wall: FluxTarget
    source: ForwardSource
    external_current: jnp.ndarray = field(repr=False)
    area: jnp.ndarray = field(repr=False)
    cell_average_stencil: jnp.ndarray | None = field(repr=False, default=None)
    cell_average_weight: jnp.ndarray | None = field(repr=False, default=None)
    polarity: int = 1
    inside_material: jnp.ndarray | None = field(repr=False, default=None)
    moment_geometry: MomentGeometry | None = field(repr=False, default=None)
    sample: FluxTarget | None = field(repr=False, default=None)
    use_linear_moments: bool = field(repr=False, default=True)
    prescribed_current_field: InitVar[PrescribedCurrentField | None] = None
    prescribed_field: PrescribedCurrentField | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self, prescribed_current_field: PrescribedCurrentField | None):
        """Build the topology read and default the material mask."""
        self.prescribed_field = prescribed_current_field
        self.external_current = jnp.asarray(self.external_current)
        self.area = jnp.asarray(self.area)
        if self.cell_average_stencil is not None:
            self.cell_average_stencil = jnp.asarray(
                self.cell_average_stencil, dtype=jnp.int32
            )
            self.cell_average_weight = jnp.asarray(
                self.cell_average_weight, dtype=self.area.dtype
            )
        if self.inside_material is None:
            self.inside_material = jnp.ones(self.grid.node_number, dtype=bool)
        else:
            self.inside_material = jnp.asarray(self.inside_material, dtype=bool)
        try:
            raster_radius, raster_height = _structured_grid_axes(self.grid.coordinate)
        except ValueError:
            self._raster_radius = None
            self._raster_height = None
            self._raster_shape = None
        else:
            self._raster_radius = jnp.asarray(raster_radius, dtype=jnp.float64)
            self._raster_height = jnp.asarray(raster_height, dtype=jnp.float64)
            self._raster_shape = (raster_radius.size, raster_height.size)
        if self.area.shape != (self.grid.node_number,):
            raise ValueError("area must carry one control area per grid node")
        if (
            self.cell_average_stencil is not None
            and self.cell_average_stencil.shape != (self.grid.node_number, 5)
        ):
            raise ValueError("cell-average stencil must carry five nodes per grid cell")
        if self.cell_average_stencil is not None and self.cell_average_weight.shape != (
            5,
        ):
            raise ValueError("cell-average weights must carry five fixed entries")
        if self.inside_material.shape != (self.grid.node_number,):
            raise ValueError("inside_material must carry one flag per grid node")
        material_flag = np.asarray(self.inside_material, dtype=bool)
        grid_coordinate_host = np.asarray(self.grid.coordinate, dtype=np.float64)
        centroid = (
            grid_coordinate_host[material_flag].mean(axis=0)
            if material_flag.any()
            else grid_coordinate_host.mean(axis=0)
        )
        self._material_centroid = jnp.asarray(centroid, dtype=jnp.float64)
        polygons = (
            self.moment_geometry.polygons
            if self.moment_geometry is not None
            else tuple(np.zeros((3, 2)) for _ in range(self.grid.node_number))
        )
        partition_rings, partition_edges = cell_partition_geometry(
            self.grid.coordinate, self.grid.null.stencil, polygons
        )
        edge_parameter = np.asarray((0.0, 0.5, 1.0))
        edge_points = partition_edges[..., :1, :] + edge_parameter[
            None, None, :, None
        ] * (partition_edges[..., 1:, :] - partition_edges[..., :1, :])
        edge_mesh = StencilMesh(
            np.asarray(self.grid.coordinate),
            np.asarray(self.grid.null.stencil),
            np.asarray(self.area),
        )
        edge_stencil = edge_mesh.shared_node_flux_stencil(edge_points.reshape((-1, 2)))
        edge_gather = edge_stencil.gather_index.reshape((*edge_points.shape[:-1], -1))
        edge_weight = edge_stencil.weight.reshape((*edge_points.shape[:-1], -1))
        topology_geometry = {
            "connectivity_rings": jnp.asarray(partition_rings, dtype=jnp.int32),
            "connectivity_shared_edges": jnp.asarray(partition_edges),
            "connectivity_coordinate": jnp.asarray(self.grid.coordinate),
            "connectivity_edge_gather": jnp.asarray(edge_gather, dtype=jnp.int32),
            "connectivity_edge_weight": jnp.asarray(edge_weight),
        }
        self.topology = Topology(self.grid.null, self.wall.null, **topology_geometry)
        production_locator = self.grid.null.with_capacity(
            max(self.grid.null.maxsize, _PRODUCTION_STATIONARY_POINT_CAPACITY)
        )
        self._fixed_design_topology = Topology(
            _FixedDesignNull2D.from_locator(production_locator),
            self.wall.null,
            **topology_geometry,
        )
        wall_to_cell_distance = np.sum(
            (
                np.asarray(self.wall.coordinate)[:, None, :]
                - np.asarray(self.grid.coordinate)[None, :, :]
            )
            ** 2,
            axis=-1,
        )
        self._wall_carrier_index = jnp.asarray(
            np.argmin(wall_to_cell_distance, axis=1), dtype=jnp.int32
        )
        wall_heights = np.unique(np.asarray(self.wall.coordinate[:, 1]))
        wall_steps = np.diff(wall_heights)
        positive_steps = wall_steps[wall_steps > 0.0]
        self._wall_height_hysteresis = jnp.asarray(
            0.25 * np.min(positive_steps)
            if positive_steps.size
            else np.sqrt(np.finfo(np.float64).eps),
            dtype=self.area.dtype,
        )
        grid_coordinate = np.asarray(self.grid.coordinate)
        radial_steps = np.diff(np.unique(grid_coordinate[:, 0]))
        vertical_steps = np.diff(np.unique(grid_coordinate[:, 1]))
        grid_steps = np.concatenate(
            (radial_steps[radial_steps > 0.0], vertical_steps[vertical_steps > 0.0])
        )
        self._x_qualification_distance = jnp.asarray(
            1.5 * np.max(grid_steps) if grid_steps.size else 0.0,
            dtype=self.area.dtype,
        )
        if (
            self.prescribed_field is not None
            and self.prescribed_field.response.shape[0] != self.node_number
        ):
            raise ValueError(
                "prescribed current response rows must match every operator target"
            )
        if (
            self.moment_geometry is not None
            and len(self.moment_geometry.polygons) != self.grid.node_number
        ):
            raise ValueError("moment geometry must carry one polygon per grid node")
        if self.moment_geometry is not None and self.use_linear_moments:
            if self.sample is None:
                raise ValueError(
                    "linear current moments require direct pre-clip sample targets"
                )
            if self.sample.node_number != len(
                self.moment_geometry.sample_node_coordinates
            ):
                raise ValueError(
                    "direct sample target rows must match the moment sampling nodes"
                )
            self._build_support_moment_stencils()

    def _build_support_moment_stencils(self) -> None:
        """Build the fixed own-node projection for every support."""
        coordinate = np.asarray(self.grid.coordinate, dtype=np.float64)
        ring = np.asarray(self.grid.null.stencil, dtype=np.intp)
        area = np.asarray(self.area, dtype=np.float64)
        polygon_size = np.asarray(
            [len(polygon) for polygon in self.moment_geometry.polygons]
        )
        sample_size = np.asarray(self.moment_geometry.sample_vertex_count)
        stencils = []
        for vertex_count in (4, 6):
            support_centre = np.flatnonzero(sample_size == vertex_count)
            if len(support_centre) == 0:
                continue
            selected = (polygon_size[ring[:, 0]] == vertex_count) & (
                sample_size[ring[:, 0]] == vertex_count
            )
            mesh = StencilMesh(coordinate, ring[selected], area)
            stencil = mesh.current_moment_stencil(
                support_centre=support_centre,
                sampling_node_coordinate=self.moment_geometry.sample_node_coordinates,
                sampling_cell_node=self.moment_geometry.cell_sample_nodes[
                    :, :vertex_count
                ],
            )
            stencils.append(stencil)
        self._support_moment_stencils = tuple(stencils)

    @property
    def node_number(self) -> int:
        """Return the length of the physical and direct-sample flux vector."""
        direct = 0 if self.sample is None else self.sample.node_number
        return self.grid.node_number + self.wall.node_number + direct

    @property
    def physical_node_number(self) -> int:
        """Return the centre and wall prefix consumed by topology and receipts."""
        return self.grid.node_number + self.wall.node_number

    @property
    def radius(self) -> jax.Array:
        """Return the radius [m] of every plasma grid node."""
        return self.grid.coordinate[:, 0]

    def connectivity_grid_axes(
        self,
    ) -> tuple[jax.Array, jax.Array, tuple[int, int]]:
        """Return structured radius, height, and shape for connectivity reads."""
        radius, height = _structured_grid_axes(self.grid.coordinate)
        return (
            jnp.asarray(radius, dtype=jnp.float64),
            jnp.asarray(height, dtype=jnp.float64),
            (radius.size, height.size),
        )

    def raster_geometry(self) -> tuple[jax.Array, jax.Array, tuple[int, int]]:
        """Return the rectangular receipt geometry cached with the operator."""
        if self._raster_shape is None:
            raise ValueError("raster receipts require a tensor-product forward grid")
        return self._raster_radius, self._raster_height, self._raster_shape

    def raster_image(
        self,
        current_moments: CellCurrentMoments,
        current=None,
        prescribed_current=None,
    ) -> jax.Array:
        """Evaluate the grid from the exact conductor and plasma currents."""
        external = self.external(current, prescribed_current)[: self.grid.node_number]
        return external + self.grid.internal(current_moments)

    def connectivity_axis_seed(self, axis) -> tuple[jax.Array, jax.Array]:
        """Return the owning axis cell and the material mask used by its flood."""
        return axis_cell_seed(self.grid.coordinate, axis, self.inside_material)

    def _independent_rescue_axis(self, vmap_o) -> jax.Array:
        """Return the raw O extremum nearest the material centroid.

        Chosen from the un-ranked, un-qualified candidate table by proximity
        to a fixed geometric prior alone, so it carries no dependence on
        which candidate the first admission pass ranked highest. Widening
        material by this candidate's own cell can therefore only ever help
        the same candidate a correct first pass already selected, never
        substitute a different one in its place.
        """
        distance2 = jnp.sum((vmap_o[:, :2] - self._material_centroid) ** 2, axis=1)
        distance2 = jnp.where(jnp.isfinite(vmap_o[:, 0]), distance2, jnp.inf)
        return vmap_o[jnp.argmin(distance2), :2]

    def _fixed_design_read(self, physical, requested_class=None):
        """Read topology data after admitting a centroid-nearest rescue cell."""
        initial = self._fixed_design_topology.read_qualification(
            physical,
            self.polarity,
            self.inside_material,
            requested_class,
        )
        grid_flux, _wall_flux = self._fixed_design_topology.split_flux_map(physical)
        vmap_o, _vmap_x = self._fixed_design_topology.grid(grid_flux)
        rescue_axis = self._independent_rescue_axis(vmap_o)
        _seed, material = self.connectivity_axis_seed(rescue_axis)
        result = self._fixed_design_topology.read_qualification(
            physical,
            self.polarity,
            material,
            requested_class,
        )
        same_axis = jnp.all(jnp.equal(initial.state.axis, result.state.axis))
        admitted = result.axis_admitted & (~initial.axis_admitted | same_axis)
        return result.masks, result.state, result.connected, admitted

    def _current(self, current) -> jax.Array:
        """Return the conductor currents one evaluation should use."""
        return self.external_current if current is None else jnp.asarray(current)

    def external(self, current=None, prescribed_current=None) -> jax.Array:
        """Return the flux map [Wb] of every conductor but the plasma.

        ``current`` drives the ordinary conductor targets.  A separate
        ``prescribed_current`` replaces the complete vector held by the
        prescribed response policy; it is never added to ``current``.
        """
        conductor = self._current(current)
        physical = jnp.r_[self.grid.external(conductor), self.wall.external(conductor)]
        external = (
            physical
            if self.sample is None
            else jnp.r_[physical, self.sample.external(conductor)]
        )
        if self.prescribed_field is None:
            if prescribed_current is not None:
                raise ValueError(
                    "prescribed_current requires a prescribed current field"
                )
            return external
        return external + self.prescribed_field.flux(prescribed_current)

    def __getattribute__(self, name: str):
        """Retain the public prescribed-current accessor without storing a target."""
        if name == "prescribed_current_field":
            return object.__getattribute__(self, "prescribed_field")
        return object.__getattribute__(self, name)

    def _connectivity_class_margin(
        self, physical, topology: TopologyState
    ) -> jax.Array:
        """Read the signed reachable-wall minus X-point flux margin."""
        if (
            self.moment_geometry is not None
            and not self._fixed_design_topology.connectivity_radius.size
        ):
            emergent = self._fixed_design_read(physical)[1]
            return jnp.where(emergent.diverted, jnp.inf, -jnp.inf)
        return self._connectivity_read(physical, topology, classify=True)[
            "class_margin"
        ]

    def _connectivity_read(self, physical, topology: TopologyState, *, classify):
        """Return one saddle-aware boundary read for classification or masking."""
        connectivity_radius, connectivity_height, connectivity_shape = (
            self.connectivity_grid_axes()
        )
        grid_flux, wall_flux = self.topology.split_flux_map(physical)
        _vmap_o, vmap_x = self._fixed_design_topology.grid(grid_flux)
        classification_wall = jnp.concatenate(
            (topology.wall_point, topology.wall_point_flux[None])
        )
        radial_count, vertical_count = connectivity_shape
        _axis_seed, connectivity_material = self.connectivity_axis_seed(topology.axis)
        options = (
            {
                "classification_x": vmap_x,
                "classification_wall": classification_wall,
            }
            if classify
            else {}
        )
        return traced_boundary_read(
            grid_flux.reshape((radial_count, vertical_count)).T,
            connectivity_radius,
            connectivity_height,
            connectivity_material.reshape((radial_count, vertical_count)).T,
            topology.axis[0],
            topology.axis[1],
            96,
            18,
            2,
            jnp.empty((0,), dtype=connectivity_radius.dtype),
            jnp.asarray(1.0, dtype=grid_flux.dtype),
            self.wall.coordinate[:, 0],
            self.wall.coordinate[:, 1],
            wall_flux,
            **options,
        )

    def _carrier_shadow_read(self, physical, masks: DomainMasks):
        """Return wall-shadow operands from the carrier's own topology read."""
        if not hasattr(self, "_wall_carrier_index"):
            # Lightweight composition fixtures supply the operands directly
            # without constructing carrier geometry.
            return self._connectivity_read(physical, None, classify=False)
        grid_flux, _wall_flux = self.topology.split_flux_map(physical)
        _vmap_o, vmap_x = self._fixed_design_topology.grid(grid_flux)
        return {
            "xset": vmap_x[:, :2],
            "private_wall_node_mask": masks.private_flux[self._wall_carrier_index],
        }

    def topology_margin(self, psi) -> jax.Array:
        """Return the emergent continuous topology margin of one flux map.

        Positive values are diverted, negative values are limited, and zero is
        the marginal wall/X-point hand-off. A selected wall extremum outside
        the X-point height band is excluded by the private-flux shadow.
        """
        physical = jnp.asarray(psi)[: self.physical_node_number]
        _masks, topology, _connected, admitted = self._fixed_design_read(physical)
        require_qualified_axis(admitted)
        return self._connectivity_class_margin(physical, topology)

    def read(
        self, psi, requested_class=None
    ) -> tuple[DomainMasks, ForwardTopologyState]:
        """Return domain labels and an achieved saddle-aware topology read."""
        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, _connected, admitted = self._fixed_design_read(
            physical, requested_class
        )
        require_qualified_axis(admitted)
        return masks, ForwardTopologyState(
            axis=topology.axis,
            axis_flux=topology.axis_flux,
            boundary=topology.boundary,
            boundary_flux=topology.boundary_flux,
            x_point=topology.x_point,
            x_point_flux=topology.x_point_flux,
            wall_point=topology.wall_point,
            wall_point_flux=topology.wall_point_flux,
            # The legacy TopologyState.diverted leaf is intentionally not
            # forwarded: a pinned read may contain only the requested class.
            _class_margin_read=lambda: self._connectivity_class_margin(
                physical, topology
            ),
        )

    def shared_node_flux(self, psi) -> jax.Array:
        """Evaluate the plasma-grid flux on fixed atomic shared nodes."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for shared-node flux")
        grid_flux, _wall_flux = self.topology.split_flux_map(
            jnp.asarray(psi)[: self.physical_node_number]
        )
        return self.moment_geometry.shared_node_flux(grid_flux)

    def sample_node_flux(self, psi) -> jax.Array:
        """Return the direct pre-clip sample rows from one flux vector."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for sampling-node flux")
        if self.sample is None:
            raise ValueError("direct pre-clip sample targets are required")
        return jnp.asarray(psi)[self.physical_node_number :]

    def support_current_moments(
        self,
        profile,
        centroid_flux,
        sample_flux,
        support,
    ) -> CellCurrentMoments:
        """Evaluate every nonempty support through one moment callable."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        vectors = jnp.zeros(
            (3, self.grid.node_number), dtype=jnp.asarray(centroid_flux).dtype
        )
        for stencil in self._support_moment_stencils:
            vectors = vectors + jnp.stack(
                stencil.support_flux_moments(
                    profile,
                    centroid_flux,
                    sample_flux,
                    support,
                )
            )
        return CellCurrentMoments(*vectors)

    def sample_flux_field(self, centroid_flux, sample_flux, points):
        """Evaluate the own-node flux polynomial and gradient in every cell."""
        shape = points.shape[:2]
        values = jnp.zeros(shape, dtype=jnp.asarray(centroid_flux).dtype)
        radial = jnp.zeros_like(values)
        vertical = jnp.zeros_like(values)
        for stencil in self._support_moment_stencils:
            sampled = stencil.sample_flux_field(centroid_flux, sample_flux, points)
            values = values + sampled[0]
            radial = radial + sampled[1]
            vertical = vertical + sampled[2]
        return values, radial, vertical

    def coupling_current_moments(
        self, moments: CellCurrentMoments
    ) -> CellCurrentMoments:
        """Convert physical first moments to the fixed linear-basis vectors."""
        if self.moment_geometry is None:
            radial = np.asarray(moments.radial_moment)
            vertical = np.asarray(moments.vertical_moment)
            if np.any(radial != 0.0) or np.any(vertical != 0.0):
                raise ValueError(
                    "moment geometry is required for nonzero first current moments"
                )
            return CellCurrentMoments(
                moments.cell_current,
                jnp.zeros_like(moments.radial_moment),
                jnp.zeros_like(moments.vertical_moment),
            )
        second = jnp.asarray(
            self.moment_geometry.second_moment, dtype=moments.cell_current.dtype
        )
        radial_second = second[:, 0]
        vertical_second = second[:, 1]
        cross_second = second[:, 2]
        determinant = radial_second * vertical_second - cross_second**2
        radial = (
            vertical_second * moments.radial_moment
            - cross_second * moments.vertical_moment
        ) / determinant
        vertical = (
            radial_second * moments.vertical_moment
            - cross_second * moments.radial_moment
        ) / determinant
        return CellCurrentMoments(moments.cell_current, radial, vertical)

    def _support_partition(self, psi, requested_class=None):
        """Trace the profile-owned support and sampling state once."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, _connected, _admitted = self._fixed_design_read(
            physical, requested_class
        )
        if not self.use_linear_moments:
            raise ValueError("clipped support moments are required")
        sample_flux = self.sample_node_flux(psi)
        sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        profile_support = self._profile_support(masks, physical.dtype)
        return masks, topology, sample_psi_norm, profile_support

    def _profile_support(self, masks, dtype):
        """Return the fixed clipped support selected by domain labels."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        profile_support = self.moment_geometry.atomic_mesh.traced_clip(
            jnp.ones(
                len(self.moment_geometry.atomic_mesh.node_coordinates),
                dtype=dtype,
            )
        ).qualify(masks.profile_participation)
        return profile_support

    def _partition_for_state(self, psi, frozen):
        """Revalue one state on an already-decided discrete partition."""
        topology = frozen.topology
        physical = jnp.asarray(psi)[: self.physical_node_number]
        grid_flux, _wall_flux = self.topology.split_flux_map(physical)
        psi_norm = self.topology.normalize(
            topology.axis_flux, topology.boundary_flux, grid_flux
        )
        masks = DomainMasks(label=frozen.label, psi_norm=psi_norm)
        sample_flux = self.sample_node_flux(psi)
        sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        return masks, topology, sample_psi_norm, frozen.profile_support

    def _internal_on_partition(self, psi, frozen, target_current=None):
        """Return the plasma image while retaining one trip's partition."""
        partition = self._partition_for_state(psi, frozen)
        moments = self._partitioned_current_moments(partition)
        if target_current is not None:
            amplitude = self.current_normalisation_amplitude(
                target_current, jnp.sum(moments.cell_current)
            )
            moments = self.scaled_current_moments(moments, amplitude)
        return self.current_moment_image(moments)

    def _partitioned_current_moments(self, partition) -> CellCurrentMoments:
        masks, _topology, sample_psi_norm, profile_support = partition
        moments = self.source.current_moments(
            masks,
            self.support_current_moments,
            profile_support,
            sample_flux=sample_psi_norm,
        )
        return self.coupling_current_moments(moments)

    def _clipped_integral_measure(self, partition) -> ClippedIntegralMeasure:
        """Build the observation measure from one already-traced partition."""
        masks, topology, sample_psi_norm, profile_support = partition
        profile_moments = self.source.current_moments(
            masks,
            self.support_current_moments,
            profile_support,
            sample_flux=sample_psi_norm,
        )
        cell_current = jnp.where(
            masks.profile_participation, profile_moments.cell_current, 0.0
        )
        points, weights = clipped_support_quadrature(profile_support, masks.core)
        psi_norm, radial_gradient, vertical_gradient = self.sample_flux_field(
            masks.psi_norm, sample_psi_norm, points
        )
        radius = points[..., 0]
        pressure = self.source.core.pressure(
            radius,
            psi_norm,
            self.source.boundary_pressure,
            topology.flux_span,
        )
        total_flux_gradient_squared = topology.flux_span**2 * (
            radial_gradient**2 + vertical_gradient**2
        )
        field_squared = total_flux_gradient_squared / (2.0 * jnp.pi * radius) ** 2
        volume_weight = 2.0 * jnp.pi * radius * weights
        area = jnp.where(masks.core, profile_support.area, 0.0)
        centre_radius = profile_support.centroids[:, 0]
        radial_first = jnp.where(
            masks.core, profile_support.first_area_moment[:, 0], 0.0
        )
        radial_second = jnp.where(
            masks.core, profile_support.second_area_moment[:, 0, 0], 0.0
        )
        volume = 2.0 * jnp.pi * (centre_radius * area + radial_first)
        radial_volume = (
            2.0
            * jnp.pi
            * (
                centre_radius**2 * area
                + 2.0 * centre_radius * radial_first
                + radial_second
            )
        )
        return ClippedIntegralMeasure(
            area=area,
            volume=volume,
            radial_volume=radial_volume,
            cell_current=cell_current,
            pressure_volume=jnp.sum(pressure * volume_weight, axis=1),
            field_volume=jnp.sum(field_squared * volume_weight, axis=1),
            masks=masks,
        )

    def cell_current_moments(self, psi, requested_class=None) -> CellCurrentMoments:
        """Return the current and first moments driven by one trial flux."""
        if not self.use_linear_moments:
            physical = jnp.asarray(psi)[: self.physical_node_number]
            masks, _topology, _connected, _admitted = self._fixed_design_read(
                physical, requested_class
            )
            point_current = self.source.cell_current(self.radius, self.area, masks)
            density = point_current / self.area
            if self.cell_average_stencil is None:
                current = point_current
            else:
                gathered = density[self.cell_average_stencil]
                average = jnp.einsum("ni,i->n", gathered, self.cell_average_weight)
                current = jnp.where(
                    masks.profile_participation, average * self.area, 0.0
                )
            zero = jnp.zeros_like(current)
            return CellCurrentMoments(current, zero, zero)
        return self._partitioned_current_moments(
            self._support_partition(psi, requested_class)
        )

    @staticmethod
    def scaled_current_moments(
        moments: CellCurrentMoments, amplitude: jax.Array
    ) -> CellCurrentMoments:
        """Apply one common amplitude to every integrated current moment."""
        return CellCurrentMoments(*(amplitude * value for value in moments))

    @staticmethod
    def current_normalisation_amplitude(target_current, unscaled_current) -> jax.Array:
        """Return a guarded declared-current amplitude without clipping.

        Admissibility is established by products and magnitude comparisons
        before the division. Eager evaluations raise so an accepted state can
        never masquerade as normalised; traced evaluations return a non-finite
        amplitude which the solver rejects rather than clipping to a bound.
        """
        target = jnp.asarray(target_current)
        unscaled = jnp.asarray(unscaled_current, dtype=target.dtype)
        magnitude = jnp.abs(unscaled)
        lower, upper = SCALAR_CURRENT_AMPLITUDE_BAND
        admissible = (
            jnp.isfinite(target)
            & jnp.isfinite(unscaled)
            & (target * unscaled > 0.0)
            & (jnp.abs(target) >= lower * magnitude)
            & (jnp.abs(target) <= upper * magnitude)
        )
        safe_unscaled = jnp.where(admissible, unscaled, jnp.ones_like(unscaled))
        amplitude = jnp.where(admissible, target / safe_unscaled, jnp.nan)
        if not isinstance(admissible, jax.core.Tracer) and not bool(admissible):
            target_value = float(target)
            unscaled_value = float(unscaled)
            if not np.isfinite(target_value) or not np.isfinite(unscaled_value):
                attempted = float("nan")
            elif unscaled_value == 0.0:
                attempted = float("inf")
            else:
                attempted = target_value / unscaled_value
            raise CurrentNormalisationError(attempted)
        return amplitude

    def normalised_current_moments(
        self, psi, target_current, requested_class=None
    ) -> tuple[CellCurrentMoments, jax.Array]:
        """Return exact clipped moments and their declared-current amplitude."""
        moments = self.cell_current_moments(psi, requested_class)
        amplitude = self.current_normalisation_amplitude(
            target_current, jnp.sum(moments.cell_current)
        )
        return self.scaled_current_moments(moments, amplitude), amplitude

    def current_moments_and_observation(self, psi, requested_class=None):
        """Return current moments and observations from one traced partition."""
        partition = self._support_partition(psi, requested_class)
        return (
            self._partitioned_current_moments(partition),
            self._clipped_integral_measure(partition),
            partition[0],
            partition[1],
        )

    def normalised_current_moments_and_observation(
        self, psi, target_current, requested_class=None
    ):
        """Return one clipped partition, its scaled moments, and amplitude."""
        partition = self._support_partition(psi, requested_class)
        moments = self._partitioned_current_moments(partition)
        amplitude = self.current_normalisation_amplitude(
            target_current, jnp.sum(moments.cell_current)
        )
        return (
            self.scaled_current_moments(moments, amplitude),
            self._clipped_integral_measure(partition).with_current_amplitude(amplitude),
            partition[0],
            partition[1],
            amplitude,
        )

    def current_domain_masks(self, psi, requested_class=None) -> DomainMasks:
        """Return the achieved saddle-aware domain labels."""

        masks, _topology = self.read(psi, requested_class)
        return masks

    def residual_shadow_mask(
        self, psi, requested_class=None, previous_shadow=None
    ) -> jax.Array:
        """Return the composed flood and wall-height residual exclusion."""

        flood_shadow, wall_shadow = self.residual_shadow_components(
            psi, requested_class, previous_shadow
        )
        direct_sample_shadow = jnp.zeros(
            self.node_number - self.physical_node_number, dtype=bool
        )
        return jnp.concatenate((flood_shadow, wall_shadow, direct_sample_shadow))

    def residual_shadow_components(
        self, psi, requested_class=None, previous_shadow=None
    ) -> tuple[jax.Array, jax.Array]:
        """Return independent interior-flood and wall-height shadow components."""

        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, _connected, _admitted = self._fixed_design_read(
            physical, requested_class
        )
        return self._residual_shadow_components_from_read(
            physical, masks, topology, previous_shadow
        )

    def _residual_shadow_components_from_read(
        self, physical, masks, topology, previous_shadow=None
    ):
        """Build residual shadows from one already-completed topology read."""
        reading = self._carrier_shadow_read(physical, masks)
        if previous_shadow is None:
            previous_wall_shadow = jnp.zeros(self.wall.node_number, dtype=bool)
        else:
            previous_wall_shadow = jnp.asarray(previous_shadow, dtype=bool)[
                self.grid.node_number : self.physical_node_number
            ]
        wall_shadow = wall_height_shadow_mask(
            self.wall.coordinate[:, 1],
            topology.axis[1],
            topology.x_point,
            reading["xset"],
            reading["private_wall_node_mask"],
            previous_wall_shadow,
            self._wall_height_hysteresis,
            self._x_qualification_distance,
        )
        return masks.private_flux, wall_shadow

    def _frozen_topology_partition(
        self, psi, requested_class=None, previous_shadow=None
    ):
        """Read all discrete topology state once for an active-set boundary."""
        if self.moment_geometry is None or not self.use_linear_moments:
            raise ValueError("frozen topology requires clipped support moments")
        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, _connected, _admitted = self._fixed_design_read(
            physical, requested_class
        )
        flood_shadow, wall_shadow = self._residual_shadow_components_from_read(
            physical, masks, topology, previous_shadow
        )
        direct_sample_shadow = jnp.zeros(
            self.node_number - self.physical_node_number, dtype=bool
        )
        return _FrozenTopologyPartition(
            label=masks.label,
            topology=topology,
            profile_support=self._profile_support(masks, physical.dtype),
            residual_shadow=jnp.concatenate(
                (flood_shadow, wall_shadow, direct_sample_shadow)
            ),
        )

    def cell_current(self, psi, requested_class=None, target_current=None) -> jax.Array:
        """Return the per-cell plasma current [A] a trial flux drives."""
        if target_current is None:
            moments = self.cell_current_moments(psi, requested_class)
        else:
            moments, _amplitude = self.normalised_current_moments(
                psi, target_current, requested_class
            )
        return moments.cell_current

    def internal(self, psi, requested_class=None, target_current=None) -> jax.Array:
        """Return the flux map [Wb] generated by the plasma current."""
        if target_current is None:
            moments = self.cell_current_moments(psi, requested_class)
        else:
            moments, _amplitude = self.normalised_current_moments(
                psi, target_current, requested_class
            )
        return self.current_moment_image(moments)

    def current_moment_image(self, moments: CellCurrentMoments) -> jax.Array:
        """Return flux from an explicitly supplied cell-current moment image."""
        physical = jnp.r_[self.grid.internal(moments), self.wall.internal(moments)]
        if self.sample is None:
            return physical
        return jnp.r_[physical, self.sample.internal(moments)]

    def __call__(
        self,
        psi,
        current=None,
        requested_class=None,
        target_current=None,
        prescribed_current=None,
    ) -> jax.Array:
        """Return the total poloidal flux [Wb] one write-then-read cycle gives."""
        mapped = self.external(current, prescribed_current) + self.internal(
            psi, requested_class, target_current
        )
        return self._exclude_shadow_residual(psi, mapped, requested_class)

    def _exclude_shadow_residual(
        self, psi, mapped, requested_class=None, shadow=None
    ) -> jax.Array:
        """Copy trial flux through cells excluded from the residual domain."""

        if shadow is None:
            shadow = self.residual_shadow_mask(psi, requested_class)
        return jnp.where(shadow, psi, mapped)

    def residual(
        self,
        psi,
        current=None,
        requested_class=None,
        target_current=None,
        prescribed_current=None,
    ) -> jax.Array:
        """Return the free-boundary flux residual of a trial flux map."""
        return psi - self(
            psi,
            current,
            requested_class,
            target_current,
            prescribed_current,
        )

    def flux_map(
        self,
        current=None,
        requested_class=None,
        target_current=None,
        prescribed_current=None,
    ) -> Callable[[jax.Array], jax.Array]:
        """Return the fixed-point map ``psi -> g(psi)`` at one conductor state.

        The external contribution is evaluated once and captured, so a
        fixed-point ladder pays for the plasma coupling alone.
        """
        external = self.external(current, prescribed_current)

        def mapped(psi: jax.Array) -> jax.Array:
            """Return the free-boundary flux map of one trial flux."""
            image = external + self.internal(psi, requested_class, target_current)
            return self._exclude_shadow_residual(psi, image, requested_class)

        return mapped

    def flux_map_with_shadow(
        self,
        current=None,
        requested_class=None,
        target_current=None,
        prescribed_current=None,
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Return a fixed-point map evaluated with one promoted shadow mask."""
        external = self.external(current, prescribed_current)

        def mapped(psi: jax.Array, shadow: jax.Array) -> jax.Array:
            image = external + self.internal(psi, requested_class, target_current)
            return self._exclude_shadow_residual(
                psi, image, requested_class, shadow=shadow
            )

        if self.moment_geometry is not None and self.use_linear_moments:

            def read_partition(psi, previous_shadow=None):
                return self._frozen_topology_partition(
                    psi, requested_class, previous_shadow
                )

            def map_partition(psi, partition):
                image = external + self._internal_on_partition(
                    psi, partition, target_current
                )
                return self._exclude_shadow_residual(
                    psi,
                    image,
                    requested_class,
                    shadow=partition.residual_shadow,
                )

            mapped._read_frozen_partition = read_partition
            mapped._map_frozen_partition = map_partition
            mapped._frozen_partition_shadow = lambda partition: (
                partition.residual_shadow
            )

        return mapped
