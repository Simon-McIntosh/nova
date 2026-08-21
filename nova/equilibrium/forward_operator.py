r"""Traced free-boundary flux map behind the forward equilibrium solve.

This is the implementation-level operator of
:class:`nova.equilibrium.forward.ForwardProfile`, not a second public
equilibrium problem. It carries one write-then-read cycle of the
free-boundary map: a trial total poloidal flux is read for its topology, the
prescribed flux functions are evaluated on the domains that read labels, and
the resulting cell currents are mapped back to flux through the precomputed
coupling operators. A root of :meth:`ForwardFluxOperator.residual` is a
free-boundary equilibrium.

The supplied source reaches the current image unchanged. There is no net
plasma current on this operator and no place to put one: the amplitude the
caller supplied is the amplitude the map uses, so a current image can never
be silently rescaled to meet a target.

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
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.target import FluxTarget
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.observation import (
    ClippedIntegralMeasure,
    clipped_support_quadrature,
)
from nova.equilibrium.source import ForwardSource
from nova.equilibrium.stencil_mesh import (
    CellCurrentMoments,
    MomentGeometry,
    StencilMesh,
)
from nova.equilibrium.topology import Topology, TopologyState

__all__ = ["ForwardFluxOperator"]


@dataclass
class ForwardFluxOperator:
    """Map a trial poloidal flux to the flux its plasma current generates."""

    grid: FluxTarget
    wall: FluxTarget
    source: ForwardSource
    external_current: jnp.ndarray = field(repr=False)
    area: jnp.ndarray = field(repr=False)
    polarity: int = 1
    inside_material: jnp.ndarray | None = field(repr=False, default=None)
    moment_geometry: MomentGeometry | None = field(repr=False, default=None)
    sample: FluxTarget | None = field(repr=False, default=None)
    use_linear_moments: bool = field(repr=False, default=True)

    def __post_init__(self):
        """Build the topology read and default the material mask."""
        self.topology = Topology(self.grid.null, self.wall.null)
        self.external_current = jnp.asarray(self.external_current)
        self.area = jnp.asarray(self.area)
        if self.inside_material is None:
            self.inside_material = jnp.ones(self.grid.node_number, dtype=bool)
        else:
            self.inside_material = jnp.asarray(self.inside_material, dtype=bool)
        if self.area.shape != (self.grid.node_number,):
            raise ValueError("area must carry one control area per grid node")
        if self.inside_material.shape != (self.grid.node_number,):
            raise ValueError("inside_material must carry one flag per grid node")
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

    def _current(self, current) -> jax.Array:
        """Return the conductor currents one evaluation should use."""
        return self.external_current if current is None else jnp.asarray(current)

    def external(self, current=None) -> jax.Array:
        """Return the flux map [Wb] of every conductor but the plasma."""
        conductor = self._current(current)
        physical = jnp.r_[self.grid.external(conductor), self.wall.external(conductor)]
        if self.sample is None:
            return physical
        return jnp.r_[physical, self.sample.external(conductor)]

    def read(self, psi, requested_class=None) -> tuple[DomainMasks, TopologyState]:
        """Return one shared domain/topology read, optionally class-pinned."""
        return self.topology.read(
            jnp.asarray(psi)[: self.physical_node_number],
            self.polarity,
            self.inside_material,
            requested_class,
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
        masks, topology, connected = self.topology.read_with_connectivity(
            physical,
            self.polarity,
            self.inside_material,
            requested_class,
        )
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
            current = self.source.cell_current(self.radius, self.area, masks)
            zero = jnp.zeros_like(current)
            return CellCurrentMoments(current, zero, zero)
        return self._partitioned_current_moments(
            self._support_partition(psi, requested_class)
        )

    def current_moments_and_observation(self, psi, requested_class=None):
        """Return current moments and observations from one traced partition."""
        partition = self._support_partition(psi, requested_class)
        return (
            self._partitioned_current_moments(partition),
            self._clipped_integral_measure(partition),
            partition[0],
            partition[1],
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

    def cell_current(self, psi, requested_class=None) -> jax.Array:
        """Return the per-cell plasma current [A] a trial flux drives."""
        return self.cell_current_moments(psi, requested_class).cell_current

    def internal(self, psi, requested_class=None) -> jax.Array:
        """Return the flux map [Wb] generated by the plasma current."""
        moments = self.cell_current_moments(psi, requested_class)
        return self.current_moment_image(moments)

    def current_moment_image(self, moments: CellCurrentMoments) -> jax.Array:
        """Return flux from an explicitly supplied cell-current moment image."""
        physical = jnp.r_[self.grid.internal(moments), self.wall.internal(moments)]
        if self.sample is None:
            return physical
        return jnp.r_[physical, self.sample.internal(moments)]

    def __call__(self, psi, current=None, requested_class=None) -> jax.Array:
        """Return the total poloidal flux [Wb] one write-then-read cycle gives."""
        return self.external(current) + self.internal(psi, requested_class)

    def residual(self, psi, current=None, requested_class=None) -> jax.Array:
        """Return the free-boundary flux residual of a trial flux map."""
        return psi - self(psi, current, requested_class)

    def flux_map(
        self, current=None, requested_class=None
    ) -> Callable[[jax.Array], jax.Array]:
        """Return the fixed-point map ``psi -> g(psi)`` at one conductor state.

        The external contribution is evaluated once and captured, so a
        fixed-point ladder pays for the plasma coupling alone.
        """
        external = self.external(current)

        def mapped(psi: jax.Array) -> jax.Array:
            """Return the free-boundary flux map of one trial flux."""
            return external + self.internal(psi, requested_class)

        return mapped
