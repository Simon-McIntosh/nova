"""Extract plasma topology from flux map."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.graphics.plot import Plot2D
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry
from nova.equilibrium.domain import axis_connected_component, classify_domains
from nova.equilibrium.flux_surface_connectivity import hex_edge_admissibility
from nova.jax.tree_util import Pytree


class TopologyState(NamedTuple):
    """Axis, boundary-selection and wall-limit state read from one flux map.

    ``diverted`` is the legacy boundary-selection predicate. On a pinned read
    it echoes the requested branch, so it is not an achieved topology class.
    Forward consumers obtain their achieved class from the saddle-aware
    connectivity comparator instead.
    """

    axis: jax.Array
    axis_flux: jax.Array
    boundary: jax.Array
    boundary_flux: jax.Array
    x_point: jax.Array
    x_point_flux: jax.Array
    wall_point: jax.Array
    wall_point_flux: jax.Array
    diverted: jax.Array

    @property
    def boundary_is_xpoint(self) -> jax.Array:
        """Return whether the selected boundary is the requested X-point."""

        return self.diverted

    @property
    def flux_span(self) -> jax.Array:
        """Return the total poloidal flux [Wb] from the axis to the boundary."""
        return self.boundary_flux - self.axis_flux


class BoundaryMode(StrEnum):
    """Physical obstruction that terminates the closed plasma boundary."""

    LIMITED = "limited"
    DIVERTED = "diverted"


class TopologyClass(IntEnum):
    """Device-compatible class requested from a topology-pinned read."""

    LIMITED = 0
    DIVERTED = 1


class NoQualifiedAxisError(ValueError):
    """No magnetic-axis candidate owns a resolved material component."""


@dataclass(frozen=True)
class TopologySolveReceipt:
    """Host-visible topology history for one forward solve.

    The nonlinear map keeps boolean topology on device.  This receipt is the
    explicit host boundary: it gives every completed solve a named final class,
    retains the class seen at each recorded iterate, and counts topology
    changes only as successfully traversed when the solve itself succeeded.
    A limited solve additionally publishes the wall-contact point that bound
    its last closed surface; a diverted solve leaves that field unset because
    its boundary is the X-point separatrix.
    """

    topology_class: BoundaryMode
    boundary_point_m: tuple[float, float]
    wall_contact_point_m: tuple[float, float] | None
    topology_history: tuple[BoundaryMode, ...]
    transition_count: int
    transitions_without_solver_failure: int
    solver_succeeded: bool

    def as_dict(self) -> dict[str, object]:
        """Return the strict-JSON representation of this receipt."""

        return {
            "topology_class": self.topology_class.value,
            "boundary_point_m": list(self.boundary_point_m),
            "wall_contact_point_m": (
                None
                if self.wall_contact_point_m is None
                else list(self.wall_contact_point_m)
            ),
            "topology_history": [mode.value for mode in self.topology_history],
            "transition_count": self.transition_count,
            "transitions_without_solver_failure": (
                self.transitions_without_solver_failure
            ),
            "solver_succeeded": self.solver_succeeded,
        }


def _host_point(point: jax.Array) -> tuple[float, float]:
    """Convert one device point to an immutable two-coordinate host value."""

    coordinates = jax.device_get(point)
    if coordinates.shape != (2,):
        raise ValueError("topology receipt points must have shape (2,)")
    return float(coordinates[0]), float(coordinates[1])


def boundary_mode(state: TopologyState) -> BoundaryMode:
    """Return the legacy selected-boundary mode of one topology read."""

    return (
        BoundaryMode.DIVERTED
        if bool(jax.device_get(state.boundary_is_xpoint))
        else BoundaryMode.LIMITED
    )


def topology_solve_receipt(
    states: Sequence[TopologyState], *, solver_succeeded: bool
) -> TopologySolveReceipt:
    """Summarise a non-empty topology history for one forward solve.

    ``states`` is ordered in solve-evaluation order and may contain repeated
    classes.  Only changes between adjacent recorded states are transitions.
    If the solve failed, the observed changes remain visible in
    ``transition_count`` but none are reported as traversed without failure.
    """

    if not states:
        raise ValueError("a topology solve receipt requires at least one state")
    history = tuple(boundary_mode(state) for state in states)
    transitions = sum(left is not right for left, right in zip(history, history[1:]))
    final_state = states[-1]
    final_mode = history[-1]
    wall_contact = (
        _host_point(final_state.wall_point)
        if final_mode is BoundaryMode.LIMITED
        else None
    )
    return TopologySolveReceipt(
        topology_class=final_mode,
        boundary_point_m=_host_point(final_state.boundary),
        wall_contact_point_m=wall_contact,
        topology_history=history,
        transition_count=transitions,
        transitions_without_solver_failure=transitions if solver_succeeded else 0,
        solver_succeeded=solver_succeeded,
    )


@dataclass
@jax.tree_util.register_pytree_node_class
class Topology(Pytree):
    """Manage plasma topology."""

    grid: Null2D
    wall: Null1D
    connectivity_radius: jax.Array | None = field(default=None, repr=False)
    connectivity_height: jax.Array | None = field(default=None, repr=False)

    def __post_init__(self):
        """Cache the tensor axes required by the saddle-aware component read."""
        if (
            self.connectivity_radius is not None
            and self.connectivity_height is not None
        ):
            return
        coordinate = np.asarray(self.grid.coordinate, dtype=np.float64)
        radius = np.unique(coordinate[:, 0])
        height = np.unique(coordinate[:, 1])
        expected = np.c_[
            np.repeat(radius, height.size),
            np.tile(height, radius.size),
        ]
        if coordinate.shape != expected.shape or not np.array_equal(
            coordinate, expected
        ):
            radius = np.empty(0, dtype=np.float64)
            height = np.empty(0, dtype=np.float64)
        self.connectivity_radius = jnp.asarray(radius, dtype=jnp.float64)
        self.connectivity_height = jnp.asarray(height, dtype=jnp.float64)

    @jax.jit
    def x_point_index(self, vmap_x, polarity, o_psi):
        """Return index of primary x-point."""
        x_psi = vmap_x[:, 2]
        score = jnp.asarray(polarity * (x_psi - o_psi), dtype=self.grid.fit_dtype)
        return jnp.nanargmax(score)

    @jax.jit
    def x_point_data(self, vmap_x, polarity, o_psi):
        """Return primary x-point data."""
        index = self.x_point_index(vmap_x, polarity, o_psi)
        return vmap_x[index]

    @jax.jit
    def x_point(self, psi_grid, polarity):
        """Return primary x-point position."""
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        return self.x_point_data(vmap_x, polarity, data_o[2])[:2]

    @jax.jit
    def x_psi(self, psi_grid, polarity):
        """Return primary x-point flux."""
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        return self.x_point_data(vmap_x, polarity, data_o[2])[2]

    @jax.jit
    def o_point_index(self, vmap_o, polarity, qualified=None):
        """Return primary o-point index."""
        o_psi = vmap_o[:, 2]
        score = jnp.asarray(polarity * o_psi, dtype=self.grid.fit_dtype)
        if qualified is None:
            qualified = jnp.isfinite(vmap_o[:, 0])
        return jnp.argmax(jnp.where(qualified, score, -jnp.inf))

    @jax.jit
    def o_point_data(self, vmap_o, polarity, qualified=None):
        """Return primary o-point data."""
        require_qualified = qualified is not None
        if qualified is None:
            qualified = jnp.isfinite(vmap_o[:, 0])
        if require_qualified:
            has_qualified = jnp.any(qualified)

            def validate(candidate_exists):
                if not candidate_exists:
                    raise NoQualifiedAxisError(
                        "no qualified magnetic-axis candidate has a resolved component"
                    )

            jax.debug.callback(validate, has_qualified)
        index = self.o_point_index(vmap_o, polarity, qualified)
        return vmap_o[index]

    @jax.jit
    def o_point(self, psi_grid, polarity):
        """Return primary o-point position."""
        vmap_o = self.grid(psi_grid)[0]
        return self.o_point_data(vmap_o, polarity)[:2]

    @jax.jit
    def o_psi(self, psi_grid, polarity):
        """Return primary o-point flux."""
        vmap_o = self.grid(psi_grid)[0]
        return self.o_point_data(vmap_o, polarity)[2]

    @jax.jit
    def w_point(self, psi_wall, polarity):
        """Return w_point position."""
        return self.wall(psi_wall, polarity)[:2]

    @jax.jit
    def w_psi(self, psi_wall, polarity):
        """Return wall-point flux."""
        return self.wall(psi_wall, polarity)[2]

    @jax.jit
    def boundary(self, data_o, vmap_x, data_w, polarity):
        """Return boundary data structure."""
        # x-point vertical bounds
        x_heights = vmap_x[:, 1]
        x_height_min = jnp.nanmin(x_heights)
        x_height_max = jnp.nanmax(x_heights)
        # select grid x-point
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        # o-point and w-point heights
        o_height = data_o[1]
        w_height = data_w[1]
        # A wall contact vertically beyond the x-point band lies in the
        # private-flux shadow of a null, so it cannot bind the plasma; a side
        # with no x-point beyond the axis casts no shadow (bound at infinity).
        x_height_min = jnp.where(x_height_min > o_height, -jnp.inf, x_height_min)
        x_height_max = jnp.where(x_height_max < o_height, jnp.inf, x_height_max)
        # asses plasma operational mode
        selection_flux = jnp.asarray(
            jnp.r_[data_x[2], data_w[2]], dtype=self.grid.fit_dtype
        )
        mode_index = jax.lax.cond(
            polarity < 0,
            jnp.nanargmin,
            jnp.nanargmax,
            selection_flux,
        )
        return jnp.where(
            (w_height < x_height_min) | (w_height > x_height_max),
            data_x,
            jnp.c_[data_x, data_w][:, mode_index],
        )

    @jax.jit
    def pinned_boundary(self, data_x, data_w, requested_class):
        """Return the saddle or wall anchor selected by a declared class."""

        return jnp.where(
            jnp.asarray(requested_class) == int(TopologyClass.DIVERTED),
            data_x,
            data_w,
        )

    @jax.jit
    def psi_mask(self, polarity, psi_grid, psi_boundary):
        """Return plasma filament psi-mask."""
        return jax.lax.cond(
            polarity > 0, jnp.greater_equal, jnp.less, psi_grid, psi_boundary
        )

    @jax.jit
    def x_mask(self, data_o, vmap_x):
        """Return plasma filament x-point mask.

        Each X-point cuts the grid at its own height and keeps the side the
        magnetic axis lies on, so a cell survives only where it is on the kept
        side of every one of them: the mask is a conjunction of one half-plane
        test per X-point. A cut can only ever remove cells, never restore one,
        which is what lets the tests be taken together rather than in sequence.
        The padded rows of the fixed-capacity table carry no null and cut
        nothing.
        """
        height = self.grid.coordinate[:, 1]
        x_height = vmap_x[:, 1]
        below = (x_height < data_o[1])[:, jnp.newaxis]
        test = jnp.where(
            below,
            height[jnp.newaxis, :] > x_height[:, jnp.newaxis],
            height[jnp.newaxis, :] < x_height[:, jnp.newaxis],
        )
        finite = jnp.isfinite(vmap_x[:, 0])[:, jnp.newaxis]
        return jnp.all(jnp.where(finite, test, True), axis=0)

    @partial(jax.jit, static_argnums=3)
    def psi_lcfs(self, psi_axis, psi_boundary, psi_norm=0.999):
        """Return poloidal flux at last closed flux surface."""
        return psi_norm * (psi_boundary - psi_axis) + psi_axis

    @jax.jit
    def normalize(self, psi_axis, psi_boundary, psi_grid):
        """Return normalized flux."""
        return (psi_grid - psi_axis) / (psi_boundary - psi_axis)

    @jax.jit
    def ionize(self, data_o, vmap_x, polarity, psi_grid, psi_lcfs):
        """Return ionization mask."""
        return self.x_mask(data_o, vmap_x) & self.psi_mask(polarity, psi_grid, psi_lcfs)

    @jax.jit
    def axis_component(self, psi_grid, boundary_flux, axis_flux, axis, closed, inside):
        """Return the closed, in-material hex component containing the axis."""
        if self.connectivity_radius.size == 0 or self.connectivity_height.size == 0:
            raise ValueError("topology connectivity requires a tensor-product grid")
        radial_count = self.connectivity_radius.shape[0]
        vertical_count = self.connectivity_height.shape[0]
        shape = (vertical_count, radial_count)
        flux = psi_grid.reshape((radial_count, vertical_count)).T
        confined = (closed & inside).reshape((radial_count, vertical_count)).T
        rings, shared_edges = _raster_hex_partition_geometry(
            self.connectivity_radius, self.connectivity_height
        )
        link_admissible = hex_edge_admissibility(
            flux,
            self.connectivity_radius,
            self.connectivity_height,
            boundary_flux,
            axis_flux,
            shared_edges,
        )
        coordinate = jnp.stack(
            jnp.meshgrid(self.connectivity_radius, self.connectivity_height), axis=-1
        )
        distance2 = jnp.sum((coordinate - axis) ** 2, axis=-1)
        seed_index = jnp.argmin(jnp.where(confined, distance2, jnp.inf))
        seed = jnp.zeros(shape, dtype=bool).reshape(-1).at[seed_index].set(True)
        seed = seed.reshape(shape) & jnp.any(confined)
        component = axis_connected_component(confined, rings, link_admissible, seed)
        return component.T.reshape(-1)

    @jax.jit
    def qualified_o_candidates(
        self, vmap_o, vmap_x, data_w, polarity, psi_grid, inside_material
    ):
        """Return O candidates that own a resolved material component."""
        coordinate = self.grid.coordinate

        def qualify(data_o):
            distance2 = jnp.sum((coordinate - data_o[:2]) ** 2, axis=1)
            owner_index = jnp.argmin(distance2)
            owner = jnp.arange(coordinate.shape[0]) == owner_index
            admitted_material = inside_material | owner
            data_b = self.boundary(data_o, vmap_x, data_w, polarity)
            closed = self.psi_mask(polarity, psi_grid, data_b[2])
            component = self.axis_component(
                psi_grid,
                data_b[2],
                data_o[2],
                data_o[:2],
                closed,
                admitted_material,
            )
            component_size = jnp.sum(component)
            governed_connection = jnp.any(component & inside_material)
            resolved = jnp.all(jnp.isfinite(data_b[:3]))
            return (
                jnp.all(jnp.isfinite(data_o[:3]))
                & resolved
                & governed_connection
                & (component_size > 1)
            )

        return jax.vmap(qualify)(vmap_o)

    @jax.jit
    def split_flux_map(self, psi):
        """Return poloidal flux maps split into grid and wall zones."""
        psi_grid = jax.lax.dynamic_slice_in_dim(psi, 0, self.grid.node_number)
        psi_wall = jax.lax.dynamic_slice_in_dim(
            psi, self.grid.node_number, self.wall.node_number
        )
        return psi_grid, psi_wall

    @jax.jit
    def update(self, psi, polarity):
        """Return normalized poloidal flux and ionization mask."""
        # split flux map into grid and wall zones
        psi_grid, psi_wall = self.split_flux_map(psi)
        # calculate flux map topology
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        data_w = self.wall(psi_wall, polarity)
        data_b = self.boundary(data_o, vmap_x, data_w, polarity)
        # normalize psi grid."""
        psi_norm = self.normalize(data_o[2], data_b[2], psi_grid)
        psi_lcfs = self.psi_lcfs(data_o[2], data_b[2])
        ionize = self.ionize(data_o, vmap_x, polarity, psi_grid, psi_lcfs)
        return psi_norm, ionize

    @jax.jit
    def read_with_connectivity(
        self, psi, polarity, inside_material, requested_class=None
    ):
        """Return domain labels, separatrix state, and axis connectivity.

        The same axis, X-point set and wall-limit read that :meth:`update`
        performs, published as a labelled domain partition instead of a single
        ionisation mask: the axis-connected cells inside the boundary become
        the core, the cells the X-point cut separates from the axis become the
        private-flux branch, and the remaining in-material cells become the
        common scrape-off layer.

        The closed test cuts at the BOUNDARY FLUX itself — the separatrix or
        the limiting surface the wall read returns — so a cell inside the
        boundary curve is plasma and the core mask reaches the boundary
        exactly. :meth:`update` keeps its own ionisation cut a declared
        fraction inside that surface, which is a guard on a fitted current
        image and not a statement about where the plasma ends; the two are
        different questions and no longer the same cells.
        """
        psi_grid, psi_wall = self.split_flux_map(psi)
        vmap_o, vmap_x = self.grid(psi_grid)
        data_w = self.wall(psi_wall, polarity)
        qualified_o = self.qualified_o_candidates(
            vmap_o,
            vmap_x,
            data_w,
            polarity,
            psi_grid,
            inside_material,
        )
        data_o = self.o_point_data(vmap_o, polarity, qualified_o)
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        emergent_boundary = self.boundary(data_o, vmap_x, data_w, polarity)
        if requested_class is None:
            data_b = emergent_boundary
            boundary_is_xpoint = jnp.equal(data_b[2], data_x[2])
        else:
            data_b = self.pinned_boundary(data_x, data_w, requested_class)
            boundary_is_xpoint = jnp.asarray(requested_class) == int(
                TopologyClass.DIVERTED
            )
        psi_norm = self.normalize(data_o[2], data_b[2], psi_grid)
        closed = self.psi_mask(polarity, psi_grid, data_b[2])
        connected = self.axis_component(
            psi_grid,
            data_b[2],
            data_o[2],
            data_o[:2],
            closed,
            inside_material,
        )
        masks = classify_domains(
            psi_norm,
            closed,
            connected,
            inside_material,
        )
        state = TopologyState(
            axis=data_o[:2],
            axis_flux=data_o[2],
            boundary=data_b[:2],
            boundary_flux=data_b[2],
            x_point=data_x[:2],
            x_point_flux=data_x[2],
            wall_point=data_w[:2],
            wall_point_flux=data_w[2],
            diverted=boundary_is_xpoint,
        )
        return masks, state, connected

    @jax.jit
    def read(self, psi, polarity, inside_material, requested_class=None):
        """Return the domain labels and axis/separatrix state of one flux map."""
        masks, state, _connected = self.read_with_connectivity(
            psi, polarity, inside_material, requested_class
        )
        return masks, state

    @jax.jit
    def read_batch(self, psi, polarity, inside_material):
        """Return :meth:`read` mapped over a leading batch axis."""
        return jax.vmap(self.read, in_axes=(0, None, None))(
            psi, polarity, inside_material
        )

    @jax.jit
    def update_batch(self, psi, polarity):
        """Return :meth:`update` mapped over a leading batch axis.

        The flux map gains a leading shot/time axis (psi has shape
        ``(batch, node)``); the fixed-size null bounds keep every slice the
        same shape so the categorisation vmaps cleanly. The returned
        ``(psi_norm, ionize)`` pair carries the same leading axis and is
        identical, slice for slice, to calling :meth:`update` per slice.
        """
        return jax.vmap(self.update, in_axes=(0, None))(psi, polarity)

    def plot(self, psi, polarity, axes=None):
        """Plot flux map including stationary points."""
        psi_grid, psi_wall = self.split_flux_map(psi)
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        data_w = self.wall(psi_wall, polarity)
        # plot stationary points
        axes = Plot2D().get_axes(axes=axes)
        axes.plot(*data_o[:2], "C0o")
        axes.plot(*data_x[:2], "C0x")
        axes.plot(*data_w[:2], "C0d")

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (
            self.grid,
            self.wall,
            self.connectivity_radius,
            self.connectivity_height,
        )
        aux_data = {}
        return (children, aux_data)
