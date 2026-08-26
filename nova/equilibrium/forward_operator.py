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

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.connectivity_boundary import traced_boundary_read
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
from nova.equilibrium.topology import Topology, TopologyState

__all__ = [
    "axis_cell_seed",
    "ForwardFluxOperator",
    "ForwardTopologyState",
    "PrescribedCurrentField",
]


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
    def __call__(self, psi):
        """Return extrema and saddles from fixed quadratic fit matrices."""
        sampled = jnp.asarray(psi, dtype=self.fit_dtype)[self.locator.stencil]
        sign = sampled[:, 1:] > sampled[:, :1]
        crossing_count = jnp.sum(sign != jnp.roll(sign, 1, axis=1), axis=1)
        number = jnp.asarray([jnp.sum(crossing_count == kind) for kind in (0, 4)])
        index = jnp.stack(
            [
                jnp.where(
                    crossing_count == kind,
                    size=self.locator.maxsize,
                    fill_value=0,
                )[0]
                for kind in (0, 4)
            ]
        )
        coefficient = jnp.einsum(
            "...ij,...j->...i", self.fit_weight[index], sampled[index]
        )
        determinant = (
            4.0 * coefficient[..., 0] * coefficient[..., 1] - coefficient[..., 4] ** 2
        )
        root_floor = jnp.asarray(1.0e-30, dtype=coefficient.dtype)
        safe_determinant = jnp.where(
            jnp.abs(determinant) < root_floor,
            jnp.where(determinant < 0.0, -root_floor, root_floor),
            determinant,
        )
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
        kind = jnp.where(
            jnp.abs(determinant) < jnp.asarray(1.0e-12, coefficient.dtype),
            jnp.nan,
            jnp.where(
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
            ),
        )
        origin = self.locator.physical_origin[index]
        scale = self.locator.physical_scale[index]
        physical = origin + jnp.stack((local_radial, local_vertical), axis=-1) * scale
        result = jnp.concatenate(
            (physical, local_flux[..., None], kind[..., None]), axis=-1
        )
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

    def flux(self) -> jax.Array:
        """Return the prescribed conductor flux [Wb] at every target."""
        return self.response @ self.current


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
        self.topology = Topology(self.grid.null, self.wall.null)
        self._fixed_design_topology = Topology(
            _FixedDesignNull2D.from_locator(self.grid.null), self.wall.null
        )
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

    def connectivity_axis_seed(self, axis) -> tuple[jax.Array, jax.Array]:
        """Return the owning axis cell and the material mask used by its flood."""
        return axis_cell_seed(self.grid.coordinate, axis, self.inside_material)

    def _fixed_design_read(self, physical, requested_class=None):
        """Read topology after admitting the continuous-axis owning cell."""
        _masks, initial, _connected = (
            self._fixed_design_topology.read_with_connectivity(
                physical,
                self.polarity,
                self.inside_material,
                requested_class,
            )
        )
        _seed, material = self.connectivity_axis_seed(initial.axis)
        return self._fixed_design_topology.read_with_connectivity(
            physical,
            self.polarity,
            material,
            requested_class,
        )

    def _current(self, current) -> jax.Array:
        """Return the conductor currents one evaluation should use."""
        return self.external_current if current is None else jnp.asarray(current)

    def external(self, current=None) -> jax.Array:
        """Return the flux map [Wb] of every conductor but the plasma."""
        conductor = self._current(current)
        physical = jnp.r_[self.grid.external(conductor), self.wall.external(conductor)]
        external = (
            physical
            if self.sample is None
            else jnp.r_[physical, self.sample.external(conductor)]
        )
        if self.prescribed_field is None:
            return external
        return external + self.prescribed_field.flux()

    def __getattribute__(self, name: str):
        """Retain the public prescribed-current accessor without storing a target."""
        if name == "prescribed_current_field":
            return object.__getattribute__(self, "prescribed_field")
        return object.__getattribute__(self, name)

    def _connectivity_class_margin(
        self, physical, topology: TopologyState
    ) -> jax.Array:
        """Read the signed reachable-wall minus X-point flux margin."""
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
        reading = traced_boundary_read(
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
            classification_x=vmap_x,
            classification_wall=classification_wall,
        )
        return reading["class_margin"]

    def topology_margin(self, psi) -> jax.Array:
        """Return the emergent continuous topology margin of one flux map.

        Positive values are diverted, negative values are limited, and zero is
        the marginal wall/X-point hand-off. A selected wall extremum outside
        the X-point height band is excluded by the private-flux shadow.
        """
        physical = jnp.asarray(psi)[: self.physical_node_number]
        _masks, topology, _connected = self._fixed_design_read(physical)
        return self._connectivity_class_margin(physical, topology)

    def read(
        self, psi, requested_class=None
    ) -> tuple[DomainMasks, ForwardTopologyState]:
        """Return domain labels and an achieved saddle-aware topology read."""
        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, _connected = self._fixed_design_read(physical, requested_class)
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

    def shared_domain_masks(
        self, masks: DomainMasks, topology: TopologyState, shared_flux
    ) -> DomainMasks:
        """Interpolate domain labels onto shared nodes without crossing the LCFS."""
        owner = self.moment_geometry.shared_flux_stencil.gather_index[:, 0]
        owner_label = masks.label[owner]
        psi_norm = (shared_flux - topology.axis_flux) / topology.flux_span
        closed = psi_norm <= 1.0
        label = jnp.where(
            owner_label == int(PlasmaDomain.EXCLUDED_MATERIAL),
            owner_label,
            jnp.where(
                owner_label == int(PlasmaDomain.PRIVATE_FLUX),
                owner_label,
                jnp.where(
                    closed,
                    jnp.asarray(int(PlasmaDomain.CORE), dtype=owner_label.dtype),
                    jnp.asarray(int(PlasmaDomain.COMMON_SOL), dtype=owner_label.dtype),
                ),
            ),
        )
        return DomainMasks(label=label, psi_norm=psi_norm)

    def _support_partition(self, psi, requested_class=None):
        """Trace the complementary supports and sampling state once."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        physical = jnp.asarray(psi)[: self.physical_node_number]
        masks, topology, connected = self._fixed_design_read(physical, requested_class)
        if not self.use_linear_moments:
            raise ValueError("clipped support moments are required")
        shared_flux = self.shared_node_flux(psi)
        sample_flux = self.sample_node_flux(psi)
        sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        signed_flux = self.polarity * (shared_flux - topology.boundary_flux)
        core_support = self.moment_geometry.atomic_mesh.traced_clip(signed_flux)
        core_support = core_support.qualify(connected)
        common_support = self.moment_geometry.atomic_mesh.traced_clip(-signed_flux)
        return masks, topology, sample_psi_norm, core_support, common_support

    def _partitioned_current_moments(self, partition) -> CellCurrentMoments:
        masks, _topology, sample_psi_norm, core_support, common_support = partition
        moments = self.source.current_moments(
            masks,
            self.support_current_moments,
            core_support,
            common_support,
            sample_flux=sample_psi_norm,
        )
        return self.coupling_current_moments(moments)

    def _clipped_integral_measure(self, partition) -> ClippedIntegralMeasure:
        """Build the observation measure from one already-traced partition."""
        masks, topology, sample_psi_norm, core_support, _common_support = partition
        closed_branch = masks.core | masks.common_sol
        core_moments = self.support_current_moments(
            self.source.core,
            masks.psi_norm,
            sample_psi_norm,
            core_support,
        )
        cell_current = jnp.where(closed_branch, core_moments.cell_current, 0.0)
        points, weights = clipped_support_quadrature(core_support, closed_branch)
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
        area = jnp.where(closed_branch, core_support.area, 0.0)
        centre_radius = core_support.centroids[:, 0]
        radial_first = jnp.where(
            closed_branch, core_support.first_area_moment[:, 0], 0.0
        )
        radial_second = jnp.where(
            closed_branch, core_support.second_area_moment[:, 0, 0], 0.0
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
        label = jnp.where(
            closed_branch & core_support.included,
            jnp.asarray(int(PlasmaDomain.CORE), dtype=masks.label.dtype),
            masks.label,
        )
        return ClippedIntegralMeasure(
            area=area,
            volume=volume,
            radial_volume=radial_volume,
            cell_current=cell_current,
            pressure_volume=jnp.sum(pressure * volume_weight, axis=1),
            field_volume=jnp.sum(field_squared * volume_weight, axis=1),
            masks=DomainMasks(label=label, psi_norm=masks.psi_norm),
        )

    def cell_current_moments(self, psi, requested_class=None) -> CellCurrentMoments:
        """Return the current and first moments driven by one trial flux."""
        if not self.use_linear_moments:
            masks, _topology = self.read(psi, requested_class)
            point_current = self.source.cell_current(self.radius, self.area, masks)
            density = point_current / self.area
            if self.cell_average_stencil is None:
                current = point_current
            else:
                gathered = density[self.cell_average_stencil]
                average = jnp.einsum("ni,i->n", gathered, self.cell_average_weight)
                current = jnp.where(
                    self.source.declared_support(masks), average * self.area, 0.0
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
        """Return domain labels following the shared-node clip partition."""
        masks, topology = self.read(psi, requested_class)
        if not self.use_linear_moments:
            return masks
        shared_flux = self.shared_node_flux(psi)
        support = self.moment_geometry.atomic_mesh.traced_clip(
            self.polarity * (shared_flux - topology.boundary_flux)
        )
        label = jnp.where(
            support.included,
            jnp.asarray(int(PlasmaDomain.CORE), dtype=masks.label.dtype),
            masks.label,
        )
        return DomainMasks(label=label, psi_norm=masks.psi_norm)

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
        self, psi, current=None, requested_class=None, target_current=None
    ) -> jax.Array:
        """Return the total poloidal flux [Wb] one write-then-read cycle gives."""
        return self.external(current) + self.internal(
            psi, requested_class, target_current
        )

    def residual(
        self, psi, current=None, requested_class=None, target_current=None
    ) -> jax.Array:
        """Return the free-boundary flux residual of a trial flux map."""
        return psi - self(psi, current, requested_class, target_current)

    def flux_map(
        self, current=None, requested_class=None, target_current=None
    ) -> Callable[[jax.Array], jax.Array]:
        """Return the fixed-point map ``psi -> g(psi)`` at one conductor state.

        The external contribution is evaluated once and captured, so a
        fixed-point ladder pays for the plasma coupling alone.
        """
        external = self.external(current)

        def mapped(psi: jax.Array) -> jax.Array:
            """Return the free-boundary flux map of one trial flux."""
            return external + self.internal(psi, requested_class, target_current)

        return mapped
