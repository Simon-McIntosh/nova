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
concatenated over the plasma grid nodes followed by the wall nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.target import FluxTarget
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.source import ForwardSource
from nova.equilibrium.stencil_mesh import (
    CellCurrentMoments,
    InteriorCurrentMomentStencil,
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
    use_linear_moments: bool = field(repr=False, default=True)
    smoothing_epsilon: float = field(repr=False, default=0.0)

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
        if self.smoothing_epsilon < 0.0 or self.smoothing_epsilon > 1.0:
            raise ValueError("smoothing_epsilon must lie between zero and one")
        if (
            self.moment_geometry is not None
            and len(self.moment_geometry.polygons) != self.grid.node_number
        ):
            raise ValueError("moment geometry must carry one polygon per grid node")
        if self.moment_geometry is not None:
            self._build_moment_stencils()

    def _build_moment_stencils(self) -> None:
        """Build fixed full-cell contractions from the carried mesh geometry."""
        coordinate = np.asarray(self.grid.coordinate, dtype=np.float64)
        ring = np.asarray(self.grid.null.stencil, dtype=np.intp)
        area = np.asarray(self.area, dtype=np.float64)
        atomic = self.moment_geometry.atomic_mesh
        node = np.asarray(atomic.node_coordinates)
        polygon_size = np.asarray(
            [len(polygon) for polygon in self.moment_geometry.polygons]
        )
        stencils = []
        for vertex_count in (4, 6):
            selected = polygon_size[ring[:, 0]] == vertex_count
            if not np.any(selected):
                continue
            mesh = StencilMesh(coordinate, ring[selected], area)
            cell_node = np.zeros((len(coordinate), vertex_count), dtype=np.intp)
            for centre in mesh.centre:
                vertices = self.moment_geometry.polygons[int(centre)]
                distance = np.sum(
                    (vertices[:, np.newaxis, :] - node[np.newaxis, :, :]) ** 2,
                    axis=2,
                )
                nearest = np.argmin(distance, axis=1)
                if np.any(
                    np.sqrt(distance[np.arange(vertex_count), nearest])
                    > atomic.tolerance
                ):
                    raise ValueError("moment polygon vertex is absent from atomic mesh")
                cell_node[int(centre)] = nearest
            stencil = mesh.current_moment_stencil(
                cell_node, self.moment_geometry.second_moment[:, :2]
            )
            stencils.append(
                InteriorCurrentMomentStencil(
                    gather_index=stencil.gather_index,
                    contraction_weight=stencil.contraction_weight,
                    centre=stencil.centre,
                    cell_count=stencil.cell_count,
                    shared_node_count=len(node),
                )
            )
        self._moment_mesh = StencilMesh(coordinate, ring, area)
        self._interior_moment_stencils = tuple(stencils)

    @property
    def node_number(self) -> int:
        """Return the length of the concatenated grid and wall flux vector."""
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
        return jnp.r_[self.grid.external(conductor), self.wall.external(conductor)]

    def read(self, psi) -> tuple[DomainMasks, TopologyState]:
        """Return the domain labels and axis/separatrix state of a trial flux."""
        return self.topology.read(psi, self.polarity, self.inside_material)

    def shared_node_flux(self, psi) -> jax.Array:
        """Evaluate the plasma-grid flux on fixed atomic shared nodes."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for shared-node flux")
        grid_flux, _wall_flux = self.topology.split_flux_map(psi)
        return self.moment_geometry.shared_node_flux(grid_flux)

    def current_density_gradient(self, density) -> jax.Array:
        """Return fitted radial and vertical derivatives of a cell field."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current gradients")
        radial, vertical = self._moment_mesh.gradient(density)
        return jnp.stack([radial, vertical], axis=1)

    def interior_current_moments(
        self, centroid_density, shared_density
    ) -> CellCurrentMoments:
        """Apply every fixed interior moment contraction and combine its rows."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        vectors = jnp.zeros(
            (3, self.grid.node_number), dtype=jnp.asarray(centroid_density).dtype
        )
        for stencil in self._interior_moment_stencils:
            vectors = vectors + jnp.stack(stencil(centroid_density, shared_density))
        return CellCurrentMoments(*vectors)

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

    def cell_current_moments(self, psi) -> CellCurrentMoments:
        """Return the current and first moments driven by one trial flux."""
        if self.moment_geometry is None:
            raise ValueError("moment geometry is required for current moments")
        masks, topology = self.read(psi)
        if not self.use_linear_moments:
            current = self.source.cell_current(self.radius, self.area, masks)
            zero = jnp.zeros_like(current)
            return CellCurrentMoments(current, zero, zero)
        shared_flux = self.shared_node_flux(psi)
        shared_masks = self.shared_domain_masks(masks, topology, shared_flux)
        signed_flux = self.polarity * (shared_flux - topology.boundary_flux)
        if self.smoothing_epsilon == 0.0:
            core_support = self.moment_geometry.atomic_mesh.traced_clip(signed_flux)
            common_support = self.moment_geometry.atomic_mesh.traced_clip(-signed_flux)
        else:
            smoothing_width = self.smoothing_epsilon * jnp.abs(topology.flux_span)
            core_support = self.moment_geometry.atomic_mesh.traced_clip(
                signed_flux, smoothing_width=smoothing_width
            )
            common_support = self.moment_geometry.atomic_mesh.traced_clip(
                -signed_flux, smoothing_width=smoothing_width
            )
        moments = self.source.current_moments(
            self.radius,
            masks,
            self.moment_geometry.atomic_mesh.node_coordinates[:, 0],
            shared_masks,
            self.interior_current_moments,
            self.current_density_gradient,
            core_support,
            common_support,
            smoothing_epsilon=self.smoothing_epsilon,
        )
        return self.coupling_current_moments(moments)

    def current_domain_masks(self, psi) -> DomainMasks:
        """Return domain labels following the shared-node clip partition."""
        masks, topology = self.read(psi)
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

    def cell_current(self, psi) -> jax.Array:
        """Return the per-cell plasma current [A] a trial flux drives."""
        return self.cell_current_moments(psi).cell_current

    def internal(self, psi) -> jax.Array:
        """Return the flux map [Wb] generated by the plasma current."""
        moments = self.cell_current_moments(psi)
        return jnp.r_[self.grid.internal(moments), self.wall.internal(moments)]

    def __call__(self, psi, current=None) -> jax.Array:
        """Return the total poloidal flux [Wb] one write-then-read cycle gives."""
        return self.external(current) + self.internal(psi)

    def residual(self, psi, current=None) -> jax.Array:
        """Return the free-boundary flux residual of a trial flux map."""
        return psi - self(psi, current)

    def flux_map(self, current=None) -> Callable[[jax.Array], jax.Array]:
        """Return the fixed-point map ``psi -> g(psi)`` at one conductor state.

        The external contribution is evaluated once and captured, so a
        fixed-point ladder pays for the plasma coupling alone.
        """
        external = self.external(current)

        def mapped(psi: jax.Array) -> jax.Array:
            """Return the free-boundary flux map of one trial flux."""
            return external + self.internal(psi)

        return mapped
