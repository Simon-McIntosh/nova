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
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _canonicalize_reciprocal_hex_edges,
    _points_inside_polygon,
    _raster_hex_partition_geometry,
)
from nova.equilibrium.domain import (
    DomainMasks,
    axis_connected_component,
    classify_domains,
)
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    polish_census_stationary_points,
)
from nova.geometry.hexstencil import HEX_RING
from nova.jax.tree_util import Pytree
from nova.linalg.tensor_spline import TensorBSpline, fit_tensor_spline

_MATERIAL_CONNECTED_FRACTION = 0.01
"""Minimum share of the material grid an O candidate's component must reach.

A candidate confined to a private well beside a coil or limiter can flood a
handful of cells regardless of grid resolution; a genuinely confined region
reaches a share of the material grid that scales with it. On production MAST
operands a spurious private-well candidate's component reaches a few tenths
of a percent of the material cell count while the true confined region
reaches several times ten percent, so one percent sits with wide margin on
both sides. Requiring the flood to reach this fraction of ``inside_material``
(rather than merely touching it once) is what keeps a private well from
qualifying as if it were the confined region.
"""


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


class AxisQualification(NamedTuple):
    """Non-throwing magnetic-axis selection result for trial admission."""

    data: jax.Array
    admitted: jax.Array


class TopologyQualification(NamedTuple):
    """Device topology read with an explicit magnetic-axis admission bit."""

    masks: DomainMasks
    state: TopologyState
    connected: jax.Array
    axis_admitted: jax.Array
    polish_receipt: dict[str, jax.Array]
    boundary_uncertain: jax.Array


def _carrier_polish_layout(coordinate, rings):
    """Embed one connected hex carrier in its half-offset axial lattice."""
    points = np.asarray(coordinate, dtype=np.float64)
    neighbours = np.asarray(rings, dtype=np.intp)
    unavailable = (
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.int32),
        np.zeros((2, 2), dtype=bool),
    )
    if points.ndim != 2 or points.shape[1] != 2 or neighbours.size == 0:
        return unavailable

    ring_by_centre = {int(row[0]): row for row in neighbours}
    first = int(neighbours[0, 0])
    axial = {first: (0, 0)}
    pending = [first]
    while pending:
        centre = pending.pop()
        row = ring_by_centre.get(centre)
        if row is None:
            continue
        origin = np.asarray(axial[centre])
        for slot, neighbour in enumerate(row[1:]):
            neighbour = int(neighbour)
            if neighbour == centre:
                continue
            position = tuple(origin + HEX_RING[slot])
            if neighbour in axial:
                if axial[neighbour] != position:
                    return unavailable
                continue
            axial[neighbour] = position
            pending.append(neighbour)

    if len(axial) < 50 or len(set(axial.values())) != len(axial):
        return unavailable
    indices = np.asarray(sorted(axial), dtype=np.int32)
    positions = np.asarray([axial[index] for index in indices])
    lower = positions.min(axis=0)
    positions -= lower
    radial_count, vertical_count = positions.max(axis=0) + 1
    if radial_count * vertical_count > 4 * len(points):
        return unavailable

    shape = (int(vertical_count), int(radial_count))
    radial = np.zeros(shape, dtype=np.float64)
    vertical = np.zeros(shape, dtype=np.float64)
    gather = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    for index, (radial_index, vertical_index) in zip(indices, positions, strict=True):
        radial[vertical_index, radial_index] = points[index, 0]
        vertical[vertical_index, radial_index] = points[index, 1]
        gather[vertical_index, radial_index] = index
        valid[vertical_index, radial_index] = True
    return radial, vertical, gather, valid


def require_qualified_axis(admitted: jax.Array) -> None:
    """Raise on the host when a completed topology read has no valid axis."""

    if isinstance(admitted, jax.core.Tracer):
        return
    if not bool(np.asarray(jax.device_get(admitted))):
        raise NoQualifiedAxisError(
            "no qualified magnetic-axis candidate has a resolved component"
        )


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
    connectivity_rings: jax.Array | None = field(default=None, repr=False)
    connectivity_shared_edges: jax.Array | None = field(default=None, repr=False)
    connectivity_coordinate: jax.Array | None = field(default=None, repr=False)
    connectivity_edge_gather: jax.Array | None = field(default=None, repr=False)
    connectivity_edge_weight: jax.Array | None = field(default=None, repr=False)
    polish_radial: jax.Array | None = field(default=None, repr=False)
    polish_vertical: jax.Array | None = field(default=None, repr=False)
    polish_gather: jax.Array | None = field(default=None, repr=False)
    polish_valid: jax.Array | None = field(default=None, repr=False)

    def __post_init__(self):
        """Cache the tensor axes required by the saddle-aware component read."""
        if all(
            value is not None
            for value in (
                self.connectivity_radius,
                self.connectivity_height,
                self.connectivity_rings,
                self.connectivity_shared_edges,
                self.connectivity_coordinate,
                self.connectivity_edge_gather,
                self.connectivity_edge_weight,
                self.polish_radial,
                self.polish_vertical,
                self.polish_gather,
                self.polish_valid,
            )
        ):
            return
        coordinate = np.asarray(self.grid.coordinate, dtype=np.float64)
        if self.connectivity_radius is None or self.connectivity_height is None:
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
        if self.connectivity_coordinate is None:
            self.connectivity_coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
        if self.connectivity_rings is None and self.connectivity_radius.size:
            rings, edges = _raster_hex_partition_geometry(
                self.connectivity_radius, self.connectivity_height
            )
            self.connectivity_rings = rings
            self.connectivity_shared_edges = edges
        if self.connectivity_edge_gather is None:
            self.connectivity_edge_gather = jnp.empty((0,), dtype=jnp.int32)
            self.connectivity_edge_weight = jnp.empty((0,), dtype=jnp.float64)
        if self.polish_radial is None:
            radial, vertical, gather, valid = _carrier_polish_layout(
                coordinate, self.connectivity_rings
            )
            self.polish_radial = jnp.asarray(radial, dtype=jnp.float64)
            self.polish_vertical = jnp.asarray(vertical, dtype=jnp.float64)
            self.polish_gather = jnp.asarray(gather, dtype=jnp.int32)
            self.polish_valid = jnp.asarray(valid, dtype=bool)

    @jax.jit
    def x_point_index(self, vmap_x, polarity, o_psi):
        """Return index of primary x-point.

        A candidate outside the wall polygon is excluded before the flux
        ranking runs, so a private-flux or coil-adjacent saddle that scores
        higher than the true separatrix on raw flux never wins by default.
        """
        x_psi = vmap_x[:, 2]
        inside_wall = _points_inside_polygon(
            vmap_x[:, 0],
            vmap_x[:, 1],
            self.wall.coordinate[:, 0],
            self.wall.coordinate[:, 1],
        )
        score = jnp.asarray(polarity * (x_psi - o_psi), dtype=self.grid.fit_dtype)
        return jnp.nanargmax(jnp.where(inside_wall, score, -jnp.inf))

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
        """Return the primary O-point index, or ``-1`` when none qualifies."""
        o_psi = vmap_o[:, 2]
        score = jnp.asarray(polarity * o_psi, dtype=self.grid.fit_dtype)
        if qualified is None:
            qualified = jnp.isfinite(vmap_o[:, 0])
        admitted = jnp.any(qualified)
        selected = jnp.argmax(jnp.where(qualified, score, -jnp.inf))
        return jnp.where(admitted, selected, -1)

    def o_point_data(self, vmap_o, polarity, qualified=None):
        """Return primary o-point data."""
        require_qualified = qualified is not None
        result = self.o_point_qualification(vmap_o, polarity, qualified)
        if require_qualified:
            require_qualified_axis(result.admitted)
        return result.data

    @jax.jit
    def o_point_qualification(self, vmap_o, polarity, qualified=None):
        """Return selected O data or an explicit all-NaN empty selection."""
        if qualified is None:
            qualified = jnp.isfinite(vmap_o[:, 0])
        admitted = jnp.any(qualified)
        index = self.o_point_index(vmap_o, polarity, qualified)
        data = jnp.where(admitted, vmap_o[index], jnp.full_like(vmap_o[0], jnp.nan))
        return AxisQualification(data, admitted)

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
    def psi_mask(self, polarity, psi_grid, psi_boundary, uncertainty=0.0):
        """Return the plasma-side mask outside an unresolved boundary band.

        ``uncertainty`` has flux units.  A grid value within that distance of
        the boundary is not resolved as closed: positive-polarity maps must
        exceed the upper edge of the band, while negative-polarity maps must
        fall below its lower edge.  The zero-width case retains the historic
        comparison, including its polarity-specific equality convention.
        """
        uncertainty = jnp.maximum(jnp.asarray(uncertainty, dtype=psi_grid.dtype), 0.0)
        threshold = jnp.where(
            polarity > 0,
            psi_boundary + uncertainty,
            psi_boundary - uncertainty,
        )
        return jax.lax.cond(
            polarity > 0, jnp.greater_equal, jnp.less, psi_grid, threshold
        )

    @jax.jit
    def boundary_interpolation_uncertainty(self, polish_receipt, boundary_is_xpoint):
        """Return the selected separatrix value's interpolation uncertainty.

        The tensor spline is the value authority.  The local census quadratic
        remains independent interpolation evidence on the same detected cell,
        so their absolute disagreement is the resolution-dependent uncertainty
        of comparing discrete grid values with the sub-cell separatrix value.
        Refining the grid contracts this band as those interpolants converge.
        A wall boundary or a carrier without tensor-spline authority has no
        stationary-value band and keeps the exact nodal comparison.
        """
        selected_x_value = polish_receipt["selected_value"][1]
        local_x_value = polish_receipt["local_value_evidence"][1]
        spline_authored = polish_receipt["spline_authored"][1]
        resolved = (
            boundary_is_xpoint
            & spline_authored
            & jnp.isfinite(selected_x_value)
            & jnp.isfinite(local_x_value)
        )
        return jnp.where(
            resolved,
            jnp.abs(selected_x_value - local_x_value),
            jnp.asarray(0.0, dtype=selected_x_value.dtype),
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
    def axis_component(
        self,
        psi_grid,
        boundary_flux,
        axis_flux,
        axis,
        closed,
        inside,
        saddle_cut=False,
        saddle=None,
        surface: TensorBSpline | None = None,
        boundary_uncertainty=0.0,
    ):
        """Return the closed, in-material hex component containing the axis.

        A selected saddle is read immediately on its axis side, using the same
        scale-relative inward offset as the raster connectivity boundary. That
        stronger cut is restricted to edges within three local cell pitches of
        the saddle, where the two separatrix branches are within one ring of
        each other. Elsewhere, exact-boundary links preserve the closed surface.
        """
        rings = self.connectivity_rings
        shared_edges = self.connectivity_shared_edges
        if saddle is None:
            saddle = jnp.full((2,), jnp.nan, dtype=psi_grid.dtype)
        inside_flux = jnp.where(closed & inside, psi_grid, jnp.nan)
        inward = _PRE_SADDLE_OFFSET_FRACTION * (
            jnp.nanmax(inside_flux) - jnp.nanmin(inside_flux)
        )
        direction = jnp.where(axis_flux >= boundary_flux, 1.0, -1.0)
        comparison_boundary_flux = boundary_flux + direction * boundary_uncertainty
        component_flux = comparison_boundary_flux + direction * inward
        component_flux = jnp.where(saddle_cut, component_flux, comparison_boundary_flux)
        component_closed = closed
        edge_midpoint = jnp.mean(shared_edges, axis=-2)
        structured = self.connectivity_radius.size and self.connectivity_height.size
        if structured:
            radial_count = self.connectivity_radius.shape[0]
            vertical_count = self.connectivity_height.shape[0]
            flux = psi_grid.reshape((radial_count, vertical_count)).T
            confined = (
                (component_closed & inside).reshape((radial_count, vertical_count)).T
            )
            exact_link = hex_edge_admissibility(
                flux,
                self.connectivity_radius,
                self.connectivity_height,
                comparison_boundary_flux,
                axis_flux,
                shared_edges,
                surface=surface,
            )
            inward_link = hex_edge_admissibility(
                flux,
                self.connectivity_radius,
                self.connectivity_height,
                component_flux,
                axis_flux,
                shared_edges,
                surface=surface,
            )
            coordinate = self.connectivity_coordinate.reshape(
                (radial_count, vertical_count, 2)
            ).transpose((1, 0, 2))
        else:
            confined = component_closed & inside
            edge_values = jnp.sum(
                self.connectivity_edge_weight * psi_grid[self.connectivity_edge_gather],
                axis=-1,
            )
            exact_link = hex_edge_admissibility(
                psi_grid,
                self.connectivity_coordinate[:, 0],
                self.connectivity_coordinate[:, 1],
                comparison_boundary_flux,
                axis_flux,
                shared_edges,
                edge_values=edge_values,
            )
            inward_link = hex_edge_admissibility(
                psi_grid,
                self.connectivity_coordinate[:, 0],
                self.connectivity_coordinate[:, 1],
                component_flux,
                axis_flux,
                shared_edges,
                edge_values=edge_values,
            )
            missing = (
                jnp.zeros(rings.shape, dtype=bool)
                .at[:, 1:]
                .set(rings[:, 1:] == rings[:, :1])
            )
            exact_link = exact_link & ~missing
            inward_link = inward_link & ~missing
            coordinate = self.connectivity_coordinate
        flat_coordinate = coordinate.reshape((-1, 2))
        centre = flat_coordinate[rings[:, :1]]
        neighbour = flat_coordinate[rings]
        edge_pitch = jnp.linalg.norm(neighbour - centre, axis=-1)
        saddle_distance = jnp.linalg.norm(edge_midpoint - saddle, axis=-1)
        saddle_neighbourhood = saddle_cut & (saddle_distance <= 3.0 * edge_pitch)
        link_admissible = exact_link & (inward_link | ~saddle_neighbourhood)
        link_admissible = _canonicalize_reciprocal_hex_edges(rings, link_admissible)
        distance2 = jnp.sum((coordinate - axis) ** 2, axis=-1)
        seed_index = jnp.argmin(jnp.where(confined, distance2, jnp.inf))
        seed = (
            jnp.zeros(confined.shape, dtype=bool).reshape(-1).at[seed_index].set(True)
        )
        seed = seed.reshape(confined.shape) & jnp.any(confined)
        component = axis_connected_component(confined, rings, link_admissible, seed)
        return component.T.reshape(-1) if structured else component.reshape(-1)

    @jax.jit
    def qualified_o_candidates(
        self,
        vmap_o,
        vmap_x,
        data_w,
        polarity,
        psi_grid,
        inside_material,
        surface: TensorBSpline | None = None,
    ):
        """Return O candidates whose flood reaches the confined material.

        Every finite candidate's owning cell is admitted into one shared seed
        mask up front, uniformly, before any candidate is tested — no
        candidate buys its own admission by being the one under test. The
        flood is grown through that shared mask (so a genuinely confined but
        wall-trimmed candidate can still seed), but the qualification test
        itself intersects the resulting component with the original,
        un-widened ``inside_material`` and requires that intersection to
        reach :data:`_MATERIAL_CONNECTED_FRACTION` of its cell count — a
        private well beside a coil cannot flood a comparable share of the
        material grid regardless of how deep its own flux extremum is.
        """
        coordinate = self.grid.coordinate
        finite_o = jnp.isfinite(vmap_o[:, 0])
        owner_index = jax.vmap(
            lambda position: jnp.argmin(jnp.sum((coordinate - position) ** 2, axis=1))
        )(vmap_o[:, :2])
        seedable = (
            jnp.zeros(coordinate.shape[0], dtype=bool).at[owner_index].max(finite_o)
        )
        seed_material = inside_material | seedable
        material_cell_count = jnp.sum(inside_material)
        connection_floor = jnp.maximum(
            jnp.floor(_MATERIAL_CONNECTED_FRACTION * material_cell_count), 1
        )

        def qualify(data_o):
            data_x = self.x_point_data(vmap_x, polarity, data_o[2])
            data_b = self.boundary(data_o, vmap_x, data_w, polarity)
            closed = self.psi_mask(polarity, psi_grid, data_b[2])
            component = self.axis_component(
                psi_grid,
                data_b[2],
                data_o[2],
                data_o[:2],
                closed,
                seed_material,
                jnp.equal(data_b[2], data_x[2]),
                data_x[:2],
                surface,
            )
            governed_size = jnp.sum(component & inside_material)
            governed_connection = governed_size >= connection_floor
            resolved = jnp.all(jnp.isfinite(data_b[:3]))
            return jnp.all(jnp.isfinite(data_o[:3])) & resolved & governed_connection

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
    def read_qualification(self, psi, polarity, inside_material, requested_class=None):
        """Return device topology data and magnetic-axis qualification.

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
        structured = bool(
            self.connectivity_radius.size and self.connectivity_height.size
        )
        if structured:
            radial_count = self.connectivity_radius.shape[0]
            vertical_count = self.connectivity_height.shape[0]
            flux = psi_grid.reshape((radial_count, vertical_count)).T
            surface = fit_tensor_spline(
                self.connectivity_radius,
                self.connectivity_height,
                flux,
            )
        else:
            flux = psi_grid[self.polish_gather]
            surface = None
        vmap_o, vmap_x = self.grid(psi_grid)
        data_w = self.wall(psi_wall, polarity)
        qualified_o = self.qualified_o_candidates(
            vmap_o,
            vmap_x,
            data_w,
            polarity,
            psi_grid,
            inside_material,
            surface,
        )
        selection = self.o_point_qualification(vmap_o, polarity, qualified_o)
        data_o = selection.data
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
        if structured:
            data_o, data_x, polish_receipt = polish_census_stationary_points(
                flux,
                self.connectivity_radius,
                self.connectivity_height,
                data_b[2],
                polarity,
                data_o,
                data_x,
                surface=surface,
            )
        else:
            data_o, data_x, polish_receipt = polish_census_stationary_points(
                flux,
                self.polish_radial,
                self.polish_vertical,
                data_b[2],
                polarity,
                data_o,
                data_x,
                self.polish_valid,
            )
        published_stationary = jnp.stack((data_o, data_x))
        published_stationary = published_stationary.at[:, :2].set(
            polish_receipt["selected_position_rz"]
        )
        published_stationary = published_stationary.at[:, 2].set(
            polish_receipt["selected_value"]
        )
        data_o, data_x = published_stationary
        data_b = jnp.where(boundary_is_xpoint, data_x, data_w)
        if structured:
            comparison_flux = surface(
                self.connectivity_coordinate[:, 0],
                self.connectivity_coordinate[:, 1],
            )
        else:
            comparison_flux = psi_grid
        boundary_uncertainty = self.boundary_interpolation_uncertainty(
            polish_receipt, boundary_is_xpoint
        )
        boundary_uncertain = (
            jnp.abs(comparison_flux - data_b[2]) <= boundary_uncertainty
        ) & (boundary_uncertainty > 0.0)
        polish_receipt = polish_receipt | {
            "boundary_interpolation_uncertainty": boundary_uncertainty,
            "boundary_comparison_spline_authored": jnp.asarray(structured),
        }
        psi_norm = self.normalize(data_o[2], data_b[2], comparison_flux)
        closed = self.psi_mask(
            polarity,
            comparison_flux,
            data_b[2],
            boundary_uncertainty,
        )
        connected = self.axis_component(
            comparison_flux,
            data_b[2],
            data_o[2],
            data_o[:2],
            closed,
            inside_material,
            boundary_is_xpoint,
            data_x[:2],
            surface,
            boundary_uncertainty,
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
        return TopologyQualification(
            masks,
            state,
            connected,
            selection.admitted,
            polish_receipt,
            boundary_uncertain,
        )

    def read_with_connectivity(
        self, psi, polarity, inside_material, requested_class=None
    ):
        """Return the host-qualified saddle-aware ``axis_component`` read."""

        result = self.read_qualification(
            psi, polarity, inside_material, requested_class
        )
        require_qualified_axis(result.axis_admitted)
        return result.masks, result.state, result.connected

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
            self.connectivity_rings,
            self.connectivity_shared_edges,
            self.connectivity_coordinate,
            self.connectivity_edge_gather,
            self.connectivity_edge_weight,
            self.polish_radial,
            self.polish_vertical,
            self.polish_gather,
            self.polish_valid,
        )
        aux_data = {}
        return (children, aux_data)
