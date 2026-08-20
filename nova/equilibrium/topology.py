"""Extract plasma topology from flux map."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.graphics.plot import Plot2D
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.domain import classify_domains
from nova.jax.tree_util import Pytree


class TopologyState(NamedTuple):
    """Axis, separatrix and wall-limit state read from one flux map."""

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
    def flux_span(self) -> jax.Array:
        """Return the total poloidal flux [Wb] from the axis to the boundary."""
        return self.boundary_flux - self.axis_flux


class BoundaryMode(StrEnum):
    """Physical obstruction that terminates the closed plasma boundary."""

    LIMITED = "limited"
    DIVERTED = "diverted"


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
    """Return the named boundary mode of one completed topology read."""

    return (
        BoundaryMode.DIVERTED
        if bool(jax.device_get(state.diverted))
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
    def o_point_index(self, vmap_o, polarity):
        """Return primary o-point index."""
        o_psi = vmap_o[:, 2]
        score = jnp.asarray(polarity * o_psi, dtype=self.grid.fit_dtype)
        return jnp.nanargmax(score)

    @jax.jit
    def o_point_data(self, vmap_o, polarity):
        """Return primary o-point data."""
        index = self.o_point_index(vmap_o, polarity)
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
    def read(self, psi, polarity, inside_material):
        """Return the domain labels and axis/separatrix state of one flux map.

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
        data_o = self.o_point_data(vmap_o, polarity)
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        data_w = self.wall(psi_wall, polarity)
        data_b = self.boundary(data_o, vmap_x, data_w, polarity)
        psi_norm = self.normalize(data_o[2], data_b[2], psi_grid)
        masks = classify_domains(
            psi_norm,
            self.psi_mask(polarity, psi_grid, data_b[2]),
            self.x_mask(data_o, vmap_x),
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
            diverted=jnp.equal(data_b[2], data_x[2]),
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
        children = (self.grid, self.wall)
        aux_data = {}
        return (children, aux_data)
