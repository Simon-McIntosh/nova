"""Extract plasma topology from flux map."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from functools import partial
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from nova.graphics.plot import Plot2D
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _canonicalize_reciprocal_hex_edges,
    _points_inside_polygon,
    _points_inside_wall_units,
    _raster_hex_partition_geometry,
)
from nova.geometry import select
from nova.equilibrium.domain import (
    DomainMasks,
    axis_connected_component,
    classify_domains,
)
from nova.equilibrium.flux_surface_connectivity import (
    census_stationary_receipt,
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


def x_point_height_limits(axis_height, qualified_x_points):
    """Bound the axis-facing height interval with admitted saddle positions.

    A side with no saddle beyond the axis has an infinite limit and casts no
    private shadow. Finite wall heights outside this interval are in the
    private-shadow height bands; a height on either limit is not shadowed.
    """
    points = jnp.asarray(qualified_x_points)
    finite = jnp.all(jnp.isfinite(points[:, :2]), axis=1)
    lower = jnp.min(jnp.where(finite, points[:, 1], jnp.inf), initial=jnp.inf)
    upper = jnp.max(jnp.where(finite, points[:, 1], -jnp.inf), initial=-jnp.inf)
    return (
        jnp.where(lower > axis_height, -jnp.inf, lower),
        jnp.where(upper < axis_height, jnp.inf, upper),
    )


def private_wall_node_read(
    node_flux, node_height, axis_height, saddle_flux, polarity, qualified_x_points
):
    """Read private wall flux at each node, independently of cell ownership."""
    flux = jnp.asarray(node_flux)
    height = jnp.asarray(node_height)
    lower, upper = x_point_height_limits(axis_height, qualified_x_points)
    flux_side = jnp.isfinite(flux) & (polarity * (flux - saddle_flux) >= 0.0)
    height_band = jnp.isfinite(height) & ((height < lower) | (height > upper))
    qualified = jnp.isfinite(saddle_flux) & jnp.isfinite(axis_height)
    return {
        "private_wall_node_mask": qualified & flux_side & height_band,
        "wall_node_flux": flux,
        "admitted_saddle_flux": saddle_flux,
        "wall_node_private_flux_side": flux_side,
        "wall_node_height_band": height_band,
        "private_height_lower": lower,
        "private_height_upper": upper,
    }


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
    wall_unit_index: jax.Array = jnp.asarray(-1, dtype=jnp.int32)

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
    limiting_unit_index: int | None

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
            "limiting_unit_index": self.limiting_unit_index,
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
        limiting_unit_index=(
            int(jax.device_get(final_state.wall_unit_index))
            if final_mode is BoundaryMode.LIMITED
            else None
        ),
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
    wall_unit_offsets: jax.Array | None = field(default=None, repr=False)
    wall_unit_closed: jax.Array | None = field(default=None, repr=False)
    wall_unit_vessel: jax.Array | None = field(default=None, repr=False)

    def __post_init__(self):
        """Cache the tensor axes required by the saddle-aware component read."""
        if self.wall_unit_offsets is None:
            self.wall_unit_offsets = jnp.asarray(
                [0, self.wall.coordinate.shape[0]], dtype=jnp.int32
            )
        else:
            self.wall_unit_offsets = jnp.asarray(
                self.wall_unit_offsets, dtype=jnp.int32
            )
        unit_count = self.wall_unit_offsets.shape[0] - 1
        if self.wall_unit_closed is None:
            self.wall_unit_closed = jnp.ones((unit_count,), dtype=bool)
        else:
            self.wall_unit_closed = jnp.asarray(self.wall_unit_closed, dtype=bool)
        if self.wall_unit_vessel is None:
            self.wall_unit_vessel = jnp.ones((unit_count,), dtype=bool)
        else:
            self.wall_unit_vessel = jnp.asarray(self.wall_unit_vessel, dtype=bool)
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
    def contained_x_candidates(self, vmap_x):
        """Return finite saddle candidates within the first-wall polygon."""
        finite = jnp.all(jnp.isfinite(vmap_x[:, :3]), axis=1)
        if self.wall_unit_offsets is None or self.wall_unit_offsets.shape[0] == 2:
            return finite & _points_inside_polygon(
                vmap_x[:, 0],
                vmap_x[:, 1],
                self.wall.coordinate[:, 0],
                self.wall.coordinate[:, 1],
            )
        return finite & _points_inside_wall_units(
            vmap_x[:, 0],
            vmap_x[:, 1],
            self.wall.coordinate[:, 0],
            self.wall.coordinate[:, 1],
            self.wall_unit_offsets,
            self.wall_unit_closed,
            self.wall_unit_vessel,
        )

    @jax.jit
    def x_point_index(self, vmap_x, polarity, o_psi):
        """Return index of primary x-point.

        A candidate outside the wall polygon is excluded before the flux
        ranking runs, so a private-flux or coil-adjacent saddle that scores
        higher than the true separatrix on raw flux never wins by default.
        """
        x_psi = vmap_x[:, 2]
        inside_wall = self.contained_x_candidates(vmap_x)
        score = jnp.asarray(polarity * (x_psi - o_psi), dtype=self.grid.fit_dtype)
        return jnp.nanargmax(jnp.where(inside_wall, score, -jnp.inf))

    @jax.jit
    def x_point_data(self, vmap_x, polarity, o_psi):
        """Return primary x-point data."""
        index = self.x_point_index(vmap_x, polarity, o_psi)
        admitted = jnp.any(self.contained_x_candidates(vmap_x))
        return jnp.where(
            admitted,
            vmap_x[index],
            jnp.full_like(vmap_x[0], jnp.nan),
        )

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

    def _wall_anchor_candidates(self, wall_flux):
        """Return every fixed-shape wall bracket and its fitted extremum."""
        nodes = jnp.arange(wall_flux.size, dtype=jnp.int32)
        if self.wall_unit_offsets is None or self.wall_unit_offsets.shape[0] == 2:
            brackets = jnp.mod(
                nodes[:, jnp.newaxis] + jnp.asarray([-1, 0, 1]), wall_flux.size
            )
            units = jnp.zeros(wall_flux.size, dtype=jnp.int32)
            fitted = jnp.ones(wall_flux.size, dtype=bool)
        else:
            offsets = self.wall_unit_offsets
            units = jnp.searchsorted(offsets[1:], nodes, side="right")
            starts = offsets[units]
            ends = offsets[units + 1]
            closed = self.wall_unit_closed[units]
            previous = jnp.where(
                nodes > starts,
                nodes - 1,
                jnp.where(closed, ends - 1, starts),
            )
            following = jnp.where(
                nodes < ends - 1,
                nodes + 1,
                jnp.where(closed, starts, ends - 1),
            )
            open_starts = jnp.clip(nodes - 1, starts, jnp.maximum(starts, ends - 3))
            open_brackets = jnp.minimum(
                open_starts[:, jnp.newaxis] + jnp.arange(3),
                ends[:, jnp.newaxis] - 1,
            )
            closed_brackets = jnp.stack((previous, nodes, following), axis=1)
            brackets = jnp.where(closed[:, jnp.newaxis], closed_brackets, open_brackets)
            fitted = (ends - starts) >= 3

        coordinate = jnp.asarray(self.wall.coordinate)[brackets]
        values = jnp.asarray(wall_flux)[brackets]

        def fit_one(points, samples):
            length = select.length_2d(points[:, 0], points[:, 1], array_namespace=jnp)
            coefficients = select.traced_quadratic_wall(length, samples)
            position = select.wall_length(coefficients, array_namespace=jnp)
            interpolated_flux = (
                coefficients[0] * position**2
                + coefficients[1] * position
                + coefficients[2]
            )
            radius, height = select.wall_coordinate(
                position,
                points[:, 0],
                points[:, 1],
                length,
                array_namespace=jnp,
            )
            kind = jnp.where(
                coefficients[0] > 0,
                -1.0,
                jnp.where(coefficients[0] < 0, 1.0, jnp.nan),
            )
            return jnp.stack((radius, height, interpolated_flux, kind))

        candidates = jax.vmap(fit_one)(coordinate, values)
        sampled = jnp.column_stack(
            (
                self.wall.coordinate,
                wall_flux,
                jnp.full(wall_flux.shape, jnp.nan, dtype=wall_flux.dtype),
            )
        )
        return jnp.where(fitted[:, jnp.newaxis], candidates, sampled), units, brackets

    def _wall_anchor_selection(self, wall_flux, polarity, eligible=None):
        """Return the strongest eligible wall extremum and its bracket.

        Structured reads supply a containment mask whose true rows are fitted
        contacts reached by the axis-connected component at that candidate's
        own spline level. Ranking only those rows prevents a detached private
        lobe from publishing the limited plasma boundary.
        """
        candidates, units, brackets = self._wall_anchor_candidates(wall_flux)
        if eligible is None:
            eligible = jnp.isfinite(wall_flux)
        eligible = jnp.asarray(eligible, dtype=bool) & jnp.isfinite(wall_flux)
        signed = jnp.asarray(polarity, dtype=wall_flux.dtype) * wall_flux
        node = jnp.argmax(jnp.where(eligible, signed, -jnp.inf))
        data = jnp.where(
            jnp.any(eligible),
            candidates[node],
            jnp.full_like(candidates[0], jnp.nan),
        )
        return data, units[node], brackets[node]

    def _axis_connected_wall_candidates(
        self,
        candidates,
        polarity,
        comparison_flux,
        axis_data,
        inside_material,
        surface,
    ):
        """Return contacts reached by their axis-enclosing component.

        Each bracket supplies one fitted contact and the tensor spline supplies
        that contact's boundary level. The existing component flood is applied
        independently at every level, then the candidate is admitted only when
        that component reaches the fitted contact within one lattice pitch.
        Candidate count and raster shape determine every array, preserving the
        fixed shape under ``jit`` and ``vmap``.
        """
        radial_pitch = jnp.max(jnp.diff(self.connectivity_radius))
        vertical_pitch = jnp.max(jnp.diff(self.connectivity_height))
        pitch = jnp.maximum(radial_pitch, vertical_pitch)
        coordinate = self.connectivity_coordinate

        def reaches_axis_component(candidate):
            boundary_flux = candidate[2]
            closed = self.psi_mask(polarity, comparison_flux, boundary_flux)
            component = self.axis_component(
                comparison_flux,
                boundary_flux,
                axis_data[2],
                axis_data[:2],
                closed,
                inside_material,
                surface=surface,
                polarity=polarity,
            )
            distance = jnp.linalg.norm(coordinate - candidate[:2], axis=1)
            return (
                jnp.all(jnp.isfinite(candidate[:3]))
                & jnp.all(jnp.isfinite(axis_data[:3]))
                & jnp.any(component & (distance <= pitch))
            )

        return jax.vmap(reaches_axis_component)(candidates)

    def wall_anchor_bracket(self, psi_wall, polarity):
        """Return the three flat node indices supporting the selected unit."""
        return self._wall_anchor_selection(jnp.asarray(psi_wall), polarity)[2]

    def wall_anchor_data(
        self,
        psi_wall,
        polarity,
        requested_class=None,
        private_wall_node_mask=None,
        surface: TensorBSpline | None = None,
        comparison_flux=None,
        axis_data=None,
        inside_material=None,
        containment_required=True,
        x_point_flux=None,
    ):
        """Return the wall extremum on the surface that traces the boundary.

        A private-wall mask is meaningful only for a pinned limited read.  An
        emergent read retains its own saddle-height reachability rule, while a
        pinned diverted read selects its saddle and must not change when wall
        shadow evidence is supplied.

        Structured reads fit every wall bracket, evaluate every fitted contact
        on the tensor spline used for contours, and admit only contacts reached
        by the axis-connected component at their own level. A limited plasma
        therefore has one boundary authority: its last closed contour passes
        through the published wall contact. Unstructured reads have no tensor
        surface and retain their wall-zone samples.

        ``containment_required`` of ``None`` evaluates containment inside this
        pass rather than receiving it: the strongest fitted contact reached by
        the axis-enclosing component at its own level is screened against the
        pass' X-point level, and containment is required only when that
        contact outranks the saddle and would therefore bind the boundary. A
        diverted plasma keeps the raw wall extremum as its wall diagnostic.

        Masked samples receive a finite losing score before the wall extremum
        is selected.  Keeping the operand finite preserves the fixed-shape
        quadratic interpolation used by :class:`~nova.biot.null.Null1D` while
        preventing a masked node from winning the discrete wall bracket.
        """
        wall_flux = jnp.asarray(psi_wall)
        if surface is not None:
            wall_flux = surface(
                self.wall.coordinate[:, 0],
                self.wall.coordinate[:, 1],
            )
        masked_flux = wall_flux
        private_wall = jnp.zeros(wall_flux.shape, dtype=bool)
        highest = jnp.asarray(jnp.inf, dtype=wall_flux.dtype)
        if private_wall_node_mask is not None and requested_class is not None:
            private_wall = jnp.asarray(private_wall_node_mask, dtype=bool)
            if private_wall.shape != wall_flux.shape:
                raise ValueError("private wall mask must carry one flag per wall node")
            apply_mask = jnp.asarray(requested_class) == int(TopologyClass.LIMITED)
            private_wall = private_wall & apply_mask
            signed_flux = jnp.asarray(polarity, dtype=wall_flux.dtype) * wall_flux
            unmasked = ~private_wall & jnp.isfinite(signed_flux)
            lowest = jnp.min(jnp.where(unmasked, signed_flux, jnp.inf))
            highest = jnp.max(jnp.where(unmasked, signed_flux, -jnp.inf))
            span = highest - lowest
            scale = jnp.maximum(jnp.maximum(jnp.abs(lowest), jnp.abs(highest)), 1.0)
            losing_step = jnp.maximum(span, jnp.finfo(wall_flux.dtype).eps * scale)
            losing_score = jnp.where(jnp.any(unmasked), lowest - losing_step, jnp.nan)
            masked_flux = jnp.where(
                private_wall,
                jnp.asarray(polarity, dtype=wall_flux.dtype) * losing_score,
                wall_flux,
            )

        eligible = None
        if (
            surface is not None
            and comparison_flux is not None
            and axis_data is not None
            and inside_material is not None
        ):
            candidates = self._wall_anchor_candidates(masked_flux)[0]
            candidate_flux = surface(candidates[:, 0], candidates[:, 1])
            candidates = candidates.at[:, 2].set(candidate_flux)
            eligible = self._axis_connected_wall_candidates(
                candidates,
                polarity,
                comparison_flux,
                axis_data,
                inside_material,
                surface,
            )
            if containment_required is None:
                screened = self._wall_anchor_selection(masked_flux, polarity, eligible)[
                    0
                ]
                if x_point_flux is None:
                    containment_required = jnp.asarray(False)
                else:
                    ranked = jnp.asarray(polarity) * (
                        screened[2] - jnp.asarray(x_point_flux)
                    )
                    containment_required = (
                        jnp.isfinite(screened[2])
                        & jnp.isfinite(jnp.asarray(x_point_flux))
                        & (ranked > 0)
                    )
            eligible = jnp.where(
                jnp.asarray(containment_required),
                eligible,
                jnp.isfinite(masked_flux),
            )
        selected = self._wall_anchor_selection(masked_flux, polarity, eligible)[0]
        if private_wall_node_mask is not None and requested_class is not None:
            selected_score = jnp.asarray(polarity, dtype=wall_flux.dtype) * selected[2]
            bounded_flux = jnp.asarray(polarity, dtype=wall_flux.dtype) * jnp.minimum(
                selected_score, highest
            )
            bounded = selected.at[2].set(bounded_flux)
            selected = jnp.where(jnp.any(private_wall), bounded, selected)
        if surface is not None:
            selected = selected.at[2].set(surface(selected[0], selected[1]))
        return selected

    @jax.jit
    def boundary(self, data_o, vmap_x, data_w, polarity):
        """Return boundary data structure."""
        # x-point vertical bounds
        contained_x = self.contained_x_candidates(vmap_x)
        x_height_min, x_height_max = x_point_height_limits(
            data_o[1], jnp.where(contained_x[:, None], vmap_x[:, :2], jnp.nan)
        )
        # select grid x-point
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        # o-point and w-point heights
        w_height = data_w[1]
        # A wall contact vertically beyond the x-point band lies in the
        # private-flux shadow of a null, so it cannot bind the plasma; a side
        # with no x-point beyond the axis casts no shadow (bound at infinity).
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
        polarity=None,
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
        # The plasma-current polarity is the authoritative ordering of the
        # axis and boundary levels.  Using their sampled values here can
        # silently reverse the flood when a trial map has not yet preserved
        # the seed ordering; the same sign also governs psi_mask and saddle
        # admission below.
        direction = (
            jnp.where(jnp.asarray(polarity) >= 0, 1.0, -1.0)
            if polarity is not None
            else jnp.where(axis_flux >= boundary_flux, 1.0, -1.0)
        )
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
                polarity=polarity,
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
    def read_qualification(
        self,
        psi,
        polarity,
        inside_material,
        requested_class=None,
        private_wall_node_mask=None,
    ):
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
            comparison_flux = surface(
                self.connectivity_coordinate[:, 0],
                self.connectivity_coordinate[:, 1],
            )
        else:
            flux = psi_grid[self.polish_gather]
            surface = None
            comparison_flux = psi_grid
        census_authored = structured and hasattr(self.grid, "read_census")
        null_flux = psi_grid
        if getattr(self.grid, "direct_sample_count", 0):
            sample_flux = psi[self.grid.node_number + self.wall.node_number :]
            if sample_flux.shape[0] != self.grid.direct_sample_count:
                raise ValueError(
                    "own-node null census requires the direct sampling flux values"
                )
            null_flux = jnp.concatenate((psi_grid, sample_flux))
        if census_authored:
            (vmap_o, vmap_x), census = self.grid.read_census(null_flux)
        else:
            vmap_o, vmap_x = self.grid(null_flux)
            census = None
        data_w = self.wall_anchor_data(
            psi_wall,
            polarity,
            requested_class,
            private_wall_node_mask,
            surface,
        )
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
        if requested_class is None:
            # A class-free read has no branch to pin containment to, so each
            # containment pass evaluates it from the state that pass holds.
            containment_required = None
        else:
            containment_required = jnp.asarray(requested_class) == int(
                TopologyClass.LIMITED
            )
        if structured:
            data_w = self.wall_anchor_data(
                psi_wall,
                polarity,
                requested_class,
                private_wall_node_mask,
                surface,
                comparison_flux,
                data_o,
                inside_material,
                containment_required,
                self.x_point_data(vmap_x, polarity, data_o[2])[2],
            )
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
            data_w = self.wall_anchor_data(
                psi_wall,
                polarity,
                requested_class,
                private_wall_node_mask,
                surface,
                comparison_flux,
                data_o,
                inside_material,
                containment_required,
                self.x_point_data(vmap_x, polarity, data_o[2])[2],
            )
        wall_node = jnp.argmin(
            jnp.sum((self.wall.coordinate - data_w[:2]) ** 2, axis=1)
        )
        wall_unit_index = jnp.searchsorted(
            self.wall_unit_offsets[1:], wall_node, side="right"
        )
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
        if census_authored:
            selected_index = jnp.stack(
                (
                    self.o_point_index(vmap_o, polarity, qualified_o),
                    self.x_point_index(vmap_x, polarity, data_o[2]),
                )
            )
            polish_receipt = census_stationary_receipt(
                flux,
                self.connectivity_radius,
                self.connectivity_height,
                census,
                selected_index,
                jnp.stack((data_o, data_x)),
                jnp.stack(
                    (
                        selection.admitted,
                        jnp.all(jnp.isfinite(data_x[:3])),
                    )
                ),
            )
        elif structured:
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
            boundary_position = jax.lax.stop_gradient(data_b[:2])
            boundary_value = surface(boundary_position[0], boundary_position[1])
            data_b = data_b.at[2].set(boundary_value)
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
            polarity=polarity,
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
            wall_unit_index=wall_unit_index,
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
        self,
        psi,
        polarity,
        inside_material,
        requested_class=None,
        private_wall_node_mask=None,
    ):
        """Return the host-qualified saddle-aware ``axis_component`` read."""

        result = self.read_qualification(
            psi,
            polarity,
            inside_material,
            requested_class,
            private_wall_node_mask,
        )
        require_qualified_axis(result.axis_admitted)
        return result.masks, result.state, result.connected

    def read(
        self,
        psi,
        polarity,
        inside_material,
        requested_class=None,
        private_wall_node_mask=None,
    ):
        """Return the domain labels and axis/separatrix state of one flux map."""
        masks, state, _connected = self.read_with_connectivity(
            psi,
            polarity,
            inside_material,
            requested_class,
            private_wall_node_mask,
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
            self.wall_unit_offsets,
            self.wall_unit_closed,
            self.wall_unit_vessel,
        )
        aux_data = {}
        return (children, aux_data)


class TopologyReason(IntEnum):
    """Device-readable refusals; a finite position alone is never admission."""

    OK = 0
    CAPACITY = 1
    SINGULAR_REPRESENTATION = 2
    UNRESOLVED_TIE = 3
    NO_QUALIFIED_AXIS = 4
    SINGULAR_TANGENT = 5
    NONFINITE_FIELD = 6
    UNRESOLVED_COMPONENT = 7


class FieldJet(NamedTuple):
    """Physical total flux and its point derivatives in the declared gauge."""

    value: jax.Array
    gradient: jax.Array
    hessian: jax.Array


class ExteriorField(Protocol):
    """A pytree evaluator of prescribed physical flux at arbitrary points."""

    def evaluate(self, point: jax.Array) -> FieldJet:
        """Return point value, gradient and Hessian without mesh differencing."""
        ...


class StationaryRead(NamedTuple):
    """A polished point and separate primal and implicit-tangent verdicts."""

    position: jax.Array
    jet: FieldJet
    valid: jax.Array
    tangent_valid: jax.Array
    reason: jax.Array


class SaddleNormalForm(NamedTuple):
    """Four separatrix rays with regular quadratic curvature corrections.

    A ray is ``position + t*direction + t**2*curvature + t**3*cubic``.
    Directions are counterclockwise. Opposite rays remain distinct: their
    point contact cannot join the interiors of the two same-sign lobes.
    """

    position: jax.Array
    direction: jax.Array
    curvature: jax.Array
    cubic: jax.Array
    positive: jax.Array
    valid: jax.Array
    reason: jax.Array


def _conditioned_hessian(hessian, tolerance):
    """Test a dimensionless determinant before any inverse is evaluated."""
    scale = jnp.max(jnp.abs(hessian))
    scaled = hessian / jnp.where(scale > 0.0, scale, 1.0)
    determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
    valid = (
        jnp.all(jnp.isfinite(hessian))
        & (scale > 0.0)
        & (jnp.abs(determinant) > tolerance)
    )
    return valid, jnp.where(valid, hessian, jnp.eye(2, dtype=hessian.dtype))


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _implicit_stationary_position(field, seed, pitch, tolerance, iterations):
    """Polish a point using analytic jets, with an implicit position tangent."""

    def step(_index, position):
        jet = field.evaluate(position)
        valid, hessian = _conditioned_hessian(jet.hessian, tolerance)
        correction = jnp.linalg.solve(hessian, jet.gradient)
        length = jnp.linalg.norm(correction)
        correction = correction * jnp.minimum(
            1.0, pitch / jnp.where(length > 0.0, length, 1.0)
        )
        valid = valid & jnp.all(jnp.isfinite(correction))
        return position - jnp.where(valid, correction, 0.0)

    return jax.lax.fori_loop(0, iterations, step, seed)


@_implicit_stationary_position.defjvp
def _implicit_stationary_position_jvp(iterations, primals, tangents):
    field, seed, pitch, tolerance = primals
    field_tangent, _seed_tangent, _pitch_tangent, _tolerance_tangent = tangents
    point = _implicit_stationary_position(field, seed, pitch, tolerance, iterations)
    jet = field.evaluate(point)
    valid, hessian = _conditioned_hessian(jet.hessian, tolerance)
    gradient_tangent = jax.jvp(
        lambda operand: operand.evaluate(point).gradient,
        (field,),
        (field_tangent,),
    )[1]
    tangent = -jnp.linalg.solve(hessian, gradient_tangent)
    return point, jnp.where(valid, tangent, jnp.nan)


def stationary_read(field, seed, pitch, policy):
    """Read a null from a pytree point evaluator, never from sampled flux.

    ``field.evaluate(point)`` returns a :class:`FieldJet` of the total field.
    Its plasma term evaluates the booked current through the Biot kernels;
    its exterior term evaluates the prescribed field at the same point.
    The evaluator and all its numerical operands must be registered pytrees.
    """
    point = _implicit_stationary_position(
        field,
        jnp.asarray(seed),
        pitch,
        policy.hessian_tolerance,
        policy.polish_iterations,
    )
    jet = field.evaluate(point)
    tangent_valid, hessian = _conditioned_hessian(jet.hessian, policy.hessian_tolerance)
    finite = jnp.all(jnp.isfinite(point)) & jnp.isfinite(jet.value)
    finite = finite & jnp.all(jnp.isfinite(jet.gradient))
    error = jnp.linalg.norm(jnp.linalg.solve(hessian, jet.gradient))
    valid = finite & tangent_valid & (error <= policy.position_tolerance * pitch)
    reason = jnp.where(
        ~finite,
        int(TopologyReason.NONFINITE_FIELD),
        jnp.where(
            ~tangent_valid,
            int(TopologyReason.SINGULAR_TANGENT),
            jnp.where(
                valid, int(TopologyReason.OK), int(TopologyReason.UNRESOLVED_COMPONENT)
            ),
        ),
    ).astype(jnp.int32)
    return StationaryRead(point, jet, valid, tangent_valid, reason)


def saddle_normal_form(
    position, hessian, third_derivative, tolerance, fourth_derivative=None
):
    """Construct separatrix jets from kernel derivatives at the saddle.

    For unit tangent d with d.H.d = 0, write x(t) = x0 + t*d + t**2*c.
    Cancellation of the cubic flux term requires
    d.H.c = -D³psi[d,d,d]/6. Choosing c normal to d fixes the parameterization.
    No level-set root or small flux difference enters this construction.
    """
    hessian = jnp.asarray(hessian)
    third = jnp.asarray(third_derivative)
    conditioned, safe = _conditioned_hessian(hessian, tolerance)
    eigenvalue, eigenvector = jnp.linalg.eigh(safe)
    valid = conditioned & (eigenvalue[0] < 0.0) & (eigenvalue[1] > 0.0)
    valid = valid & jnp.all(jnp.isfinite(third))
    negative = jnp.where(valid, -eigenvalue[0], 1.0)
    positive = jnp.where(valid, eigenvalue[1], 1.0)
    first = jnp.sqrt(positive) * eigenvector[:, 0]
    second = jnp.sqrt(negative) * eigenvector[:, 1]
    direction = jnp.stack(
        (first + second, first - second, -first - second, -first + second)
    )
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction[jnp.argsort(jnp.arctan2(direction[:, 1], direction[:, 0]))]
    normal = direction @ safe
    cubic = jnp.einsum("ijk,ai,aj,ak->a", third, direction, direction, direction)
    norm_squared = jnp.sum(normal * normal, axis=1)
    curvature = -cubic[:, None] * normal / (6.0 * norm_squared[:, None])
    fourth = (
        jnp.zeros((2, 2, 2, 2), dtype=hessian.dtype)
        if fourth_derivative is None
        else jnp.asarray(fourth_derivative)
    )
    valid = valid & jnp.all(jnp.isfinite(fourth))
    remainder = (
        0.5 * jnp.einsum("ai,ij,aj->a", curvature, safe, curvature)
        + 0.5 * jnp.einsum("ijk,ai,aj,ak->a", third, direction, direction, curvature)
        + jnp.einsum(
            "ijkl,ai,aj,ak,al->a", fourth, direction, direction, direction, direction
        )
        / 24.0
    )
    correction = -remainder[:, None] * normal / norm_squared[:, None]
    sector_direction = direction + jnp.roll(direction, -1, axis=0)
    positive_sector = (
        jnp.einsum("ai,ij,aj->a", sector_direction, safe, sector_direction) > 0.0
    )
    return SaddleNormalForm(
        jnp.asarray(position),
        jnp.where(valid, direction, 0.0),
        jnp.where(valid, curvature, 0.0),
        jnp.where(valid, correction, 0.0),
        positive_sector & valid,
        valid,
        jnp.where(
            valid, int(TopologyReason.OK), int(TopologyReason.SINGULAR_REPRESENTATION)
        ).astype(jnp.int32),
    )


class SaddleFragments(NamedTuple):
    """Exact areas and edge intervals of four normal-form sectors in a cell."""

    area: jax.Array
    edge_interval: jax.Array
    ray_parameter: jax.Array
    valid: jax.Array
    reason: jax.Array


def _cross_plane(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _quadratic_parameters(quadratic, linear, constant):
    """Return both real roots, with absent roots padded by infinity."""
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(quadratic), jnp.abs(linear)), jnp.abs(constant)
    )
    threshold = 32.0 * jnp.finfo(jnp.asarray(quadratic).dtype).eps * scale
    curved = jnp.abs(quadratic) > threshold
    discriminant = linear * linear - 4.0 * quadratic * constant
    real = discriminant >= 0.0
    root = jnp.sqrt(jnp.where(real & curved, discriminant, 1.0))
    q = -0.5 * (linear + jnp.where(linear >= 0.0, root, -root))
    safe_q = jnp.where(q != 0.0, q, 1.0)
    safe_a = jnp.where(curved, quadratic, 1.0)
    first = q / safe_a
    second = constant / safe_q
    double = -linear / (2.0 * safe_a)
    first = jnp.where(q == 0.0, double, first)
    second = jnp.where(q == 0.0, double, second)
    straight = jnp.abs(linear) > threshold
    linear_root = -constant / jnp.where(straight, linear, 1.0)
    first = jnp.where(
        curved & real, first, jnp.where(~curved & straight, linear_root, jnp.inf)
    )
    second = jnp.where(curved & real, second, jnp.inf)
    return jnp.stack((first, second), axis=-1)


def saddle_cell_fragments(vertices, vertex_count, normal_form, tolerance):
    """Intersect regular saddle rays with a convex cell, without flux roots.

    The cell must contain the null and carry counterclockwise vertices. The
    curved edge integral is analytic for each cubic ray; intersections
    with the straight cell boundary solve only a geometric equation.
    Edge intervals distinguish same-sign sectors that touch only at the null.
    """
    vertices = jnp.asarray(vertices) - normal_form.position
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < vertex_count, slot + 1, 0)
    edge = vertices[following] - vertices
    live = slot < vertex_count
    direction = normal_form.direction
    curvature = normal_form.curvature
    cubic = normal_form.cubic
    parameters = _quadratic_parameters(
        _cross_plane(curvature[:, None, :], edge[None, :, :]),
        _cross_plane(direction[:, None, :], edge[None, :, :]),
        -_cross_plane(vertices[None, :, :], edge[None, :, :]),
    )
    finite_parameter = jnp.where(jnp.isfinite(parameters), parameters, 0.0)
    cubic_cross = _cross_plane(cubic[:, None, :], edge[None, :, :])[..., None]
    quadratic_cross = _cross_plane(curvature[:, None, :], edge[None, :, :])[..., None]
    linear_cross = _cross_plane(direction[:, None, :], edge[None, :, :])[..., None]
    constant_cross = -_cross_plane(vertices[None, :, :], edge[None, :, :])[..., None]

    def intersect(_index, parameter):
        residual = (
            (cubic_cross * parameter + quadratic_cross) * parameter + linear_cross
        ) * parameter + constant_cross
        derivative = (
            3.0 * cubic_cross * parameter + 2.0 * quadratic_cross
        ) * parameter + linear_cross
        return parameter - residual / jnp.where(derivative != 0.0, derivative, 1.0)

    finite_parameter = jax.lax.fori_loop(0, 12, intersect, finite_parameter)
    parameters = jnp.where(jnp.isfinite(parameters), finite_parameter, jnp.inf)
    points = (
        finite_parameter[..., None] * direction[:, None, None, :]
        + finite_parameter[..., None] ** 2 * curvature[:, None, None, :]
        + finite_parameter[..., None] ** 3 * cubic[:, None, None, :]
    )
    edge_length_squared = jnp.sum(edge * edge, axis=-1)
    fraction = (
        jnp.sum((points - vertices[None, :, None, :]) * edge[None, :, None, :], axis=-1)
        / jnp.where(edge_length_squared > 0.0, edge_length_squared, 1.0)[None, :, None]
    )
    hit = (
        live[None, :, None]
        & jnp.isfinite(parameters)
        & (parameters > tolerance)
        & (fraction >= -tolerance)
        & (fraction <= 1.0 + tolerance)
    )
    candidates = jnp.where(hit, parameters, jnp.inf).reshape(4, -1)
    selected = jnp.argmin(candidates, axis=1)
    ray_parameter = jnp.take_along_axis(candidates, selected[:, None], axis=1)[:, 0]
    ray_valid = jnp.isfinite(ray_parameter)
    parameter = jnp.where(ray_valid, ray_parameter, 0.0)
    edge_index = selected // 2
    boundary_fraction = jnp.take_along_axis(
        fraction.reshape(4, -1), selected[:, None], axis=1
    )[:, 0]
    boundary_parameter = edge_index + jnp.clip(boundary_fraction, 0.0, 1.0)
    end_parameter = jnp.roll(boundary_parameter, -1)
    end_parameter = jnp.where(
        end_parameter <= boundary_parameter, end_parameter + vertex_count, end_parameter
    )
    edge_parameter = jnp.where(
        slot[None, :] < edge_index[:, None], slot[None, :] + vertex_count, slot[None, :]
    )
    lower = jnp.clip(boundary_parameter[:, None] - edge_parameter, 0.0, 1.0)
    upper = jnp.clip(end_parameter[:, None] - edge_parameter, 0.0, 1.0)
    present = live[None, :] & (upper > lower)
    edge_interval = jnp.stack((lower, upper), axis=-1)
    edge_interval = jnp.where(present[..., None], edge_interval, 0.0)
    begin = vertices[None, :, :] + lower[..., None] * edge[None, :, :]
    end = vertices[None, :, :] + upper[..., None] * edge[None, :, :]
    boundary_integral = jnp.sum(
        jnp.where(present, _cross_plane(begin, end), 0.0), axis=1
    )
    curve_integral = (
        _cross_plane(direction, curvature) * parameter**3 / 3.0
        + _cross_plane(direction, cubic) * parameter**4 / 2.0
        + _cross_plane(curvature, cubic) * parameter**5 / 5.0
    )
    area = 0.5 * (boundary_integral + curve_integral - jnp.roll(curve_integral, -1))
    full_area = 0.5 * jnp.sum(
        jnp.where(live, _cross_plane(vertices, vertices[following]), 0.0)
    )
    contained = jnp.all(
        jnp.where(live, _cross_plane(edge, -vertices) >= -tolerance, True)
    )
    # First exits must wind once around the cell; a folded ray is unresolved.
    winding = jnp.sum(end_parameter - boundary_parameter)
    valid = (
        normal_form.valid
        & jnp.all(ray_valid)
        & contained
        & (full_area > 0.0)
        & jnp.all(area >= -tolerance * full_area)
        & (jnp.abs(winding - vertex_count) <= tolerance * width)
        & (jnp.abs(jnp.sum(area) - full_area) <= tolerance * full_area)
    )
    return SaddleFragments(
        jnp.where(valid, jnp.maximum(area, 0.0), jnp.nan),
        jnp.where(valid, edge_interval, 0.0),
        ray_parameter,
        valid,
        jnp.where(
            valid, int(TopologyReason.OK), int(TopologyReason.UNRESOLVED_COMPONENT)
        ).astype(jnp.int32),
    )


class CellFragments(NamedTuple):
    """Connected conic fragments and their open intervals on each cell edge."""

    area: jax.Array
    edge_interval: jax.Array
    valid: jax.Array
    required: jax.Array
    reason: jax.Array
    slice_breaks: jax.Array
    slice_labels: jax.Array
    slice_lower: jax.Array
    slice_upper: jax.Array


def _quadratic_value(coefficient, point):
    x, y = point[..., 0], point[..., 1]
    return (
        coefficient[0]
        + coefficient[1] * x
        + coefficient[2] * y
        + coefficient[3] * x * x
        + coefficient[4] * x * y
        + coefficient[5] * y * y
    )


def _vertical_polygon_interval(vertices, count, x):
    slot = jnp.arange(vertices.shape[0])
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edge = vertices[following] - vertices
    fraction = (x[..., None] - vertices[:, 0]) / jnp.where(
        edge[:, 0] != 0.0, edge[:, 0], 1.0
    )
    live = (slot < count) & (edge[:, 0] != 0.0)
    live = live & (fraction >= -1e-12) & (fraction <= 1.0 + 1e-12)
    y = vertices[:, 1] + fraction * edge[:, 1]
    lower = jnp.min(jnp.where(live, y, jnp.inf), axis=-1)
    upper = jnp.max(jnp.where(live, y, -jnp.inf), axis=-1)
    return lower, upper


def _positive_vertical_intervals(vertices, count, coefficient, x):
    lower, upper = _vertical_polygon_interval(vertices, count, x)
    quadratic = jnp.broadcast_to(coefficient[5], x.shape)
    linear = coefficient[2] + coefficient[4] * x
    constant = coefficient[0] + coefficient[1] * x + coefficient[3] * x * x
    roots = _quadratic_parameters(quadratic, linear, constant)
    roots = jnp.sort(jnp.clip(roots, lower[..., None], upper[..., None]), axis=-1)
    boundaries = jnp.concatenate((lower[..., None], roots, upper[..., None]), axis=-1)
    bottom, top = boundaries[..., :-1], boundaries[..., 1:]
    middle = 0.5 * (bottom + top)
    value = (
        quadratic[..., None] * middle**2
        + linear[..., None] * middle
        + constant[..., None]
    )
    present = (top > bottom) & (value > 0.0) & jnp.isfinite(value)
    return bottom, top, present


def quadratic_cell_fragments(vertices, vertex_count, coefficient, capacity=2):
    """Decompose a quadratic superlevel set by exact algebraic slice events.

    Vertical breaks include polygon corners, every conic/edge intersection,
    and vertical tangencies. Between breaks, root order is fixed. Fragments
    join across a break only if their limiting vertical intervals overlap in
    positive length. Area uses a cosine-transformed Gauss rule on each strip;
    the transform removes the square-root endpoint singularity of a conic.
    Coordinates are cell-local, with a pitch-sized unit.
    """
    vertices = jnp.asarray(vertices)
    coefficient = jnp.asarray(coefficient)
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < vertex_count, slot + 1, 0)
    end = vertices[following]
    edge = end - vertices
    live = slot < vertex_count
    first = _quadratic_value(coefficient, vertices)
    last = _quadratic_value(coefficient, end)
    middle = _quadratic_value(coefficient, 0.5 * (vertices + end))
    edge_quadratic = 2.0 * (first + last - 2.0 * middle)
    edge_linear = last - first - edge_quadratic
    roots = _quadratic_parameters(edge_quadratic, edge_linear, first)
    edge_root_valid = live[:, None] & (roots > 0.0) & (roots < 1.0)
    lower_x = jnp.min(jnp.where(live, vertices[:, 0], jnp.inf))
    upper_x = jnp.max(jnp.where(live, vertices[:, 0], -jnp.inf))
    edge_x = (
        vertices[:, 0, None] + jnp.where(edge_root_valid, roots, 0.0) * edge[:, 0, None]
    )
    vertical_tangent = _quadratic_parameters(
        coefficient[4] ** 2 - 4 * coefficient[5] * coefficient[3],
        2 * coefficient[2] * coefficient[4] - 4 * coefficient[5] * coefficient[1],
        coefficient[2] ** 2 - 4 * coefficient[5] * coefficient[0],
    )
    breaks = jnp.sort(
        jnp.concatenate(
            (
                jnp.where(live, vertices[:, 0], upper_x),
                jnp.where(edge_root_valid, edge_x, upper_x).reshape(-1),
                jnp.clip(vertical_tangent, lower_x, upper_x),
            )
        )
    )
    unique = jnp.concatenate((jnp.ones(1, dtype=bool), breaks[1:] > breaks[:-1]))
    breaks = breaks[
        jnp.nonzero(unique, size=breaks.size, fill_value=breaks.size - 1)[0]
    ]
    left, right = breaks[:-1], breaks[1:]
    middle_x = 0.5 * (left + right)
    bottom, top, present = _positive_vertical_intervals(
        vertices, vertex_count, coefficient, middle_x
    )
    present = present & (right > left)[:, None]
    strips = left.size
    # Evaluate limiting intervals at the event itself, never a pre-saddle offset.
    left_bottom, left_top, _ = _positive_vertical_intervals(
        vertices, vertex_count, coefficient, left
    )
    right_bottom, right_top, _ = _positive_vertical_intervals(
        vertices, vertex_count, coefficient, right
    )
    overlap = jnp.minimum(right_top[:-1, :, None], left_top[1:, None, :]) - jnp.maximum(
        right_bottom[:-1, :, None], left_bottom[1:, None, :]
    )
    join = (overlap > 0.0) & present[:-1, :, None] & present[1:, None, :]
    labels = jnp.where(present, jnp.arange(strips * 3).reshape(strips, 3), strips * 3)

    def propagate(_index, label):
        from_left = jnp.min(jnp.where(join, label[:-1, :, None], strips * 3), axis=1)
        from_right = jnp.min(jnp.where(join, label[1:, None, :], strips * 3), axis=2)
        changed = label.at[1:].min(from_left)
        return changed.at[:-1].min(from_right)

    labels = jax.lax.fori_loop(0, strips, propagate, labels)
    flat = labels.reshape(-1)
    is_root = (flat == jnp.arange(strips * 3)) & present.reshape(-1)
    required = jnp.sum(is_root, dtype=jnp.int32)
    owners = jnp.nonzero(is_root, size=capacity, fill_value=strips * 3)[0]
    fragment = jnp.argmax(labels[..., None] == owners, axis=-1)
    fragment = jnp.where(present, fragment, -1)
    gauss, weight = np.polynomial.legendre.leggauss(32)
    angle = 0.5 * jnp.pi * (jnp.asarray(gauss) + 1.0)
    parameter = 0.5 * (1.0 - jnp.cos(angle))
    transformed_weight = 0.25 * jnp.pi * jnp.sin(angle) * jnp.asarray(weight)
    x = left[:, None] + (right - left)[:, None] * parameter
    quad_lower, quad_upper, quad_present = _positive_vertical_intervals(
        vertices, vertex_count, coefficient, x
    )
    strip_area = (right - left)[:, None] * jnp.sum(
        jnp.where(quad_present, quad_upper - quad_lower, 0.0)
        * transformed_weight[None, :, None],
        axis=1,
    )
    area = jnp.sum(
        jnp.where(
            fragment[..., None] == jnp.arange(capacity), strip_area[..., None], 0.0
        ),
        axis=(0, 1),
    )
    # Split each authored edge at both exact conic roots.
    root_fraction = jnp.sort(jnp.clip(roots, 0.0, 1.0), axis=-1)
    boundary = jnp.concatenate(
        (jnp.zeros((width, 1)), root_fraction, jnp.ones((width, 1))), axis=1
    )
    start, stop = boundary[:, :-1], boundary[:, 1:]
    fraction = 0.5 * (start + stop)
    point = vertices[:, None, :] + fraction[..., None] * edge[:, None, :]
    edge_present = (
        live[:, None] & (stop > start) & (_quadratic_value(coefficient, point) > 0.0)
    )
    # A midpoint is strictly inside an open edge interval. Move only its lookup
    # coordinate inward to select the adjacent strip at a vertical cell edge.
    centre = jnp.sum(jnp.where(live[:, None], vertices, 0.0), axis=0) / vertex_count
    lookup = point + 1e-10 * (centre - point)
    strip = jnp.clip(
        jnp.searchsorted(breaks, lookup[..., 0], side="right") - 1, 0, strips - 1
    )
    lookup_lower, lookup_upper, lookup_present = _positive_vertical_intervals(
        vertices, vertex_count, coefficient, lookup[..., 0]
    )
    branch = jnp.argmax(
        lookup_present
        & (lookup[..., 1, None] >= lookup_lower)
        & (lookup[..., 1, None] <= lookup_upper),
        axis=-1,
    )
    edge_fragment = fragment[strip, branch]
    intervals = jnp.stack((start, stop), axis=-1)
    intervals = jnp.where(
        edge_present[None, ..., None]
        & (edge_fragment[None, ..., None] == jnp.arange(capacity)[:, None, None, None]),
        intervals[None, ...],
        0.0,
    )
    valid = (required <= capacity) & jnp.all(jnp.isfinite(area))
    return CellFragments(
        jnp.where(valid, area, jnp.nan),
        intervals,
        valid,
        required,
        jnp.where(valid, int(TopologyReason.OK), int(TopologyReason.CAPACITY)).astype(
            jnp.int32
        ),
        breaks,
        fragment,
        bottom,
        top,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BiotMomentCoupling:
    """Authored polygon kernels evaluated at arbitrary moving target points.

    Source moments are physical integrals about ``centre``. The inverse second
    area moment converts them to the linear source basis of the exact kernel.
    Neither sampled target flux nor derivatives of a mesh reconstruction enter.
    """

    edge: jax.Array
    weight: jax.Array
    norm: jax.Array
    centre: jax.Array
    inverse_second_moment: jax.Array
    reflection_axis: jax.Array
    reflection_partner: jax.Array

    @classmethod
    def from_polygons(cls, polygons):
        """Pack immutable source geometry outside any traced read."""
        from nova.biot.greens import section_centroid, second_moments
        from nova.biot.polygon import pad_batch
        from nova.biot.polygonanalytic import _horizontal_reflection

        polygons = tuple(np.asarray(polygon, dtype=np.float64) for polygon in polygons)
        edge, weight, norm = pad_batch(polygons)
        centre = np.stack([section_centroid(polygon) for polygon in polygons])
        second = np.asarray([second_moments(polygon) for polygon in polygons])
        matrix = np.stack((second[:, (0, 2)], second[:, (2, 1)]), axis=1)
        reflection_axis = np.full(len(polygons), np.nan)
        partner = np.broadcast_to(
            np.arange(edge.shape[0])[:, None], weight.shape
        ).copy()
        for index, polygon in enumerate(polygons):
            reflection = _horizontal_reflection(polygon)
            if reflection is not None:
                reflection_axis[index], vertices = reflection
                partner[: len(polygon), index] = vertices[
                    (np.arange(len(polygon)) + 1) % len(polygon)
                ]
        return cls(
            *map(
                jnp.asarray,
                (
                    edge,
                    weight,
                    norm,
                    centre,
                    np.linalg.inv(matrix),
                    reflection_axis,
                    partner,
                ),
            )
        )

    def coefficients(self, moments):
        values = jnp.stack(tuple(moments), axis=-1)
        first = jnp.einsum("nij,nj->ni", self.inverse_second_moment, values[:, 1:])
        return jnp.concatenate((values[:, :1], first), axis=1)

    def value_gradient(self, point, moments):
        from nova.biot.polygonanalytic import packed_analytic_moments

        rows = packed_analytic_moments(
            jnp,
            jnp.broadcast_to(point[0], self.norm.shape),
            jnp.broadcast_to(point[1], self.norm.shape),
            self.edge,
            self.weight,
            self.norm,
            self.centre.T,
            self.centre.T,
            self.reflection_axis,
            self.reflection_partner,
        )
        coefficient = self.coefficients(moments).T
        value = jnp.sum(jnp.stack(rows[:3]) * coefficient)
        radial_field = jnp.sum(jnp.stack(rows[3:6]) * coefficient)
        vertical_field = jnp.sum(jnp.stack(rows[6:]) * coefficient)
        gradient = 2 * jnp.pi * point[0] * jnp.stack((vertical_field, -radial_field))
        return value, gradient

    def evaluate(self, point, moments):
        value, gradient = self.value_gradient(point, moments)
        hessian = jax.jacfwd(lambda target: self.value_gradient(target, moments)[1])(
            point
        )
        return FieldJet(value, gradient, hessian)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TotalField:
    """One booked plasma image plus an explicit differentiable exterior.

    ``exterior.evaluate(point)`` supplies value, gradient and Hessian at the
    moving target. It is a pytree of conductor geometry/current in production.
    A closed-form fixture can supply analytic total minus the same kernel image
    of its fixed reference moments; its derivatives are point derivatives too.
    """

    moments: object
    coupling: BiotMomentCoupling
    exterior: ExteriorField

    def evaluate(self, point):
        plasma = self.coupling.evaluate(point, self.moments)
        exterior = self.exterior.evaluate(point)
        return jax.tree.map(jnp.add, plasma, exterior)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologyConvention:
    """Declared signed-flux orientation, independent of an observed ordering."""

    sigma_bp: jax.Array
    current_sign: jax.Array

    @classmethod
    def from_cocos(cls, identifier, plasma_current):
        from nova.io.cocos import convention

        declared = convention(identifier)
        if not np.isfinite(plasma_current) or plasma_current == 0:
            raise ValueError("plasma current must be finite and nonzero")
        return cls(jnp.asarray(declared.sigma_bp), jnp.asarray(np.sign(plasma_current)))

    @property
    def sigma(self):
        return -self.sigma_bp * self.current_sign


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TopologyGeometry:
    """Hex carrier geometry and reciprocal atomic edge incidence as operands."""

    vertices: jax.Array
    vertex_count: jax.Array
    centre: jax.Array
    pitch: jax.Array
    sample_points: jax.Array
    fit_inverse: jax.Array
    neighbour: jax.Array
    neighbour_edge: jax.Array
    wall_points: jax.Array
    wall_following: jax.Array
    wall_unit: jax.Array
    full_area: jax.Array

    @classmethod
    def from_cells(cls, cells, sampling_vertices, wall_units):
        """Atomise authored cells and preserve each wall unit's own closure."""
        from nova.equilibrium.separatrix_clip import AtomicCellMesh

        atomic = AtomicCellMesh.from_cells(cells)
        vertices = atomic.node_coordinates[atomic.cell_nodes]
        counts = atomic.cell_vertex_count
        centre = atomic.centroids
        neighbour = np.full(atomic.cell_nodes.shape, -1, dtype=np.int32)
        neighbour_edge = np.zeros_like(neighbour)
        owners = {}
        area = np.empty(len(cells))
        for index, count in enumerate(counts):
            polygon = vertices[index, :count]
            following_polygon = np.roll(polygon, -1, axis=0)
            signed_area = 0.5 * np.sum(
                polygon[:, 0] * following_polygon[:, 1]
                - polygon[:, 1] * following_polygon[:, 0]
            )
            if signed_area <= 0:
                raise ValueError("cells must have counterclockwise nonzero area")
            area[index] = signed_area
            for edge_index in range(count):
                pair = (
                    int(atomic.cell_nodes[index, edge_index]),
                    int(atomic.cell_nodes[index, (edge_index + 1) % count]),
                )
                key = tuple(sorted(pair))
                if key in owners:
                    peer, peer_edge = owners.pop(key)
                    neighbour[index, edge_index] = peer
                    neighbour_edge[index, edge_index] = peer_edge
                    neighbour[peer, peer_edge] = index
                    neighbour_edge[peer, peer_edge] = edge_index
                else:
                    owners[key] = (index, edge_index)
        samples = np.concatenate(
            (centre[:, None, :], np.asarray(sampling_vertices)), axis=1
        )
        # A wall sliver does not shrink the generator's sampling stencil.
        pitch = np.full(len(area), np.sqrt(np.median(area)))
        local = (samples - centre[:, None, :]) / pitch[:, None, None]
        x, y = local[..., 0], local[..., 1]
        design = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)
        if np.any(np.linalg.cond(design) > 1e6):
            raise ValueError("own-node field representation is singular")
        inverse = np.linalg.pinv(design)
        points, following, units = [], [], []
        for unit_index, unit in enumerate(wall_units):
            unit = np.asarray(unit, dtype=np.float64)
            if np.array_equal(unit[0], unit[-1]):
                unit = unit[:-1]
            offset = len(points)
            points.extend(unit)
            following.extend(offset + (np.arange(len(unit)) + 1) % len(unit))
            units.extend([unit_index] * len(unit))
        return cls(
            *map(
                jnp.asarray,
                (
                    vertices,
                    counts,
                    centre,
                    pitch,
                    samples,
                    inverse,
                    neighbour,
                    neighbour_edge,
                    np.asarray(points),
                    np.asarray(following, dtype=np.int32),
                    np.asarray(units, dtype=np.int32),
                    area,
                ),
            )
        )


class TopologyRead(NamedTuple):
    """One fixed-shape total-field topology and connected-support receipt."""

    axis: jax.Array
    axis_flux: jax.Array
    x_points: jax.Array
    x_point_flux: jax.Array
    x_point_valid: jax.Array
    boundary: jax.Array
    boundary_flux: jax.Array
    boundary_class: jax.Array
    membership: jax.Array
    fragment_area: jax.Array
    fragment_component: jax.Array
    fragment_selected: jax.Array
    edge_interval: jax.Array
    wall_labels: jax.Array
    domain_labels: jax.Array
    field_coefficients: jax.Array
    saddle_form: SaddleNormalForm
    qualified: jax.Array
    valid: jax.Array
    reason: jax.Array
    tangent_valid: jax.Array
    required_nulls: jax.Array
    required_fragments: jax.Array


def _inside_cells(geometry, point):
    slot = jnp.arange(geometry.vertices.shape[1])
    following = jnp.where(
        slot[None, :] + 1 < geometry.vertex_count[:, None], slot[None, :] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    cross = _cross_plane(end - geometry.vertices, point - geometry.vertices)
    return jnp.all(
        jnp.where(
            slot[None, :] < geometry.vertex_count[:, None],
            cross >= -1e-12 * geometry.pitch[:, None] ** 2,
            True,
        ),
        axis=1,
    )


def _point_values(field, points):
    return jax.vmap(lambda point: field.evaluate(point).value)(
        points.reshape(-1, 2)
    ).reshape(points.shape[:-1])


def _field_coefficients(field, geometry):
    values = _point_values(field, geometry.sample_points)
    return jnp.einsum("nij,nj->ni", geometry.fit_inverse, values)


def _null_census(field, geometry, coefficient, policy):
    hessian = jnp.stack(
        (
            jnp.stack((2 * coefficient[:, 3], coefficient[:, 4]), axis=-1),
            jnp.stack((coefficient[:, 4], 2 * coefficient[:, 5]), axis=-1),
        ),
        axis=1,
    )
    nonsingular, safe = jax.vmap(_conditioned_hessian, in_axes=(0, None))(
        hessian, policy.hessian_tolerance
    )
    offset = -jnp.linalg.solve(safe, coefficient[:, 1:3, None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * offset
    slot = jnp.arange(geometry.vertices.shape[1])
    following = jnp.where(
        slot[None, :] + 1 < geometry.vertex_count[:, None], slot[None, :] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    contained = jnp.all(
        jnp.where(
            slot[None, :] < geometry.vertex_count[:, None],
            _cross_plane(end - geometry.vertices, seeds[:, None, :] - geometry.vertices)
            >= 0.0,
            True,
        ),
        axis=1,
    )
    candidate = nonsingular & contained & jnp.all(jnp.isfinite(seeds), axis=1)
    required = jnp.sum(candidate, dtype=jnp.int32)
    gather = jnp.nonzero(candidate, size=policy.null_capacity, fill_value=0)[0]
    nulls = jax.vmap(stationary_read, in_axes=(None, 0, 0, None))(
        field,
        seeds[gather],
        geometry.pitch[gather],
        policy,
    )
    live = (jnp.arange(policy.null_capacity) < required) & nulls.valid
    inside = jax.vmap(lambda point: jnp.any(_inside_cells(geometry, point)))(
        nulls.position
    )
    live = live & inside
    separation = jnp.linalg.norm(
        nulls.position[:, None, :] - nulls.position[None, :, :], axis=-1
    )
    earlier = (
        jnp.arange(policy.null_capacity)[None, :]
        < jnp.arange(policy.null_capacity)[:, None]
    )
    duplicate = jnp.any(
        earlier
        & live[None, :]
        & (separation < policy.position_tolerance * jnp.min(geometry.pitch) * 8),
        axis=1,
    )
    return nulls, live & ~duplicate, required


def _wall_events(field, geometry, policy, sigma):
    start = geometry.wall_points
    edge = start[geometry.wall_following] - start

    def one(first, direction):
        def step(_index, parameter):
            jet = field.evaluate(first + parameter * direction)
            first_derivative = jet.gradient @ direction
            second_derivative = direction @ jet.hessian @ direction
            concave = sigma * second_derivative < 0.0
            update = first_derivative / jnp.where(concave, second_derivative, 1.0)
            return jnp.where(concave, jnp.clip(parameter - update, 0.0, 1.0), parameter)

        parameter = jax.lax.fori_loop(
            0, policy.polish_iterations, step, jnp.asarray(0.5)
        )
        points = first + jnp.asarray((0.0, 1.0, parameter))[:, None] * direction
        values = _point_values(field, points)
        selected = jnp.argmax(sigma * values)
        return points[selected], values[selected]

    return jax.vmap(one)(start, edge)


def _connected_fragment_labels(area, intervals, geometry, tolerance):
    cells, capacity = area.shape
    edge_interval = jnp.transpose(intervals, (0, 2, 1, 3, 4))
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_interval = edge_interval[peer, geometry.neighbour_edge]
    peer_interval = 1.0 - peer_interval[..., ::-1]
    overlap = jnp.minimum(
        edge_interval[:, :, :, None, :, None, 1],
        peer_interval[:, :, None, :, None, :, 1],
    ) - jnp.maximum(
        edge_interval[:, :, :, None, :, None, 0],
        peer_interval[:, :, None, :, None, :, 0],
    )
    links = (
        jnp.any(overlap > tolerance, axis=(-2, -1))
        & (geometry.neighbour >= 0)[..., None, None]
    )
    sentinel = cells * capacity
    initial = jnp.where(
        area > 0.0, jnp.arange(sentinel).reshape(cells, capacity), sentinel
    )

    def condition(state):
        iteration, _labels, changed = state
        return changed & (iteration < sentinel)

    def propagate(state):
        iteration, labels, _changed = state
        incoming = labels[peer]
        next_label = jnp.minimum(
            labels,
            jnp.min(jnp.where(links, incoming[:, :, None, :], sentinel), axis=(1, 3)),
        )
        return iteration + 1, next_label, jnp.any(next_label != labels)

    _, labels, _ = jax.lax.while_loop(
        condition, propagate, (jnp.asarray(0), initial, jnp.asarray(True))
    )
    wall = jnp.any(
        (geometry.neighbour < 0)[:, :, None, None]
        & (edge_interval[..., 1] - edge_interval[..., 0] > tolerance),
        axis=(1, 3),
    )
    return labels, wall


def _support_at_level(
    geometry, coefficient, sigma, level, axis, form, saddle_live, policy
):
    local_vertices = (geometry.vertices - geometry.centre[:, None, :]) / geometry.pitch[
        :, None, None
    ]
    signed = sigma * coefficient
    signed = signed.at[:, 0].add(-sigma * level)
    fragments = jax.vmap(quadratic_cell_fragments, in_axes=(0, 0, 0, None))(
        local_vertices,
        geometry.vertex_count,
        signed,
        policy.fragment_capacity,
    )
    area = fragments.area * geometry.pitch[:, None] ** 2
    intervals = fragments.edge_interval
    owner = _inside_cells(geometry, form.position) & saddle_live
    saddle = jax.vmap(saddle_cell_fragments, in_axes=(0, 0, None, None))(
        geometry.vertices,
        geometry.vertex_count,
        form,
        policy.edge_tolerance,
    )
    sectors = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    saddle_area = saddle.area[:, sectors]
    saddle_intervals = saddle.edge_interval[:, sectors, :, None, :]
    saddle_intervals = jnp.pad(
        saddle_intervals,
        ((0, 0), (0, policy.fragment_capacity - 2), (0, 0), (0, 2), (0, 0)),
    )
    saddle_area = jnp.pad(saddle_area, ((0, 0), (0, policy.fragment_capacity - 2)))
    area = jnp.where(owner[:, None], saddle_area, area)
    intervals = jnp.where(owner[:, None, None, None, None], saddle_intervals, intervals)
    labels, wall = _connected_fragment_labels(
        area, intervals, geometry, policy.edge_tolerance
    )
    axis_cells = _inside_cells(geometry, axis)
    axis_cell = jnp.argmax(axis_cells)
    axis_local = (axis - geometry.centre[axis_cell]) / geometry.pitch[axis_cell]
    axis_strip = jnp.clip(
        jnp.searchsorted(fragments.slice_breaks[axis_cell], axis_local[0], side="right")
        - 1,
        0,
        fragments.slice_labels.shape[1] - 1,
    )
    lower, upper, positive = _positive_vertical_intervals(
        local_vertices[axis_cell],
        geometry.vertex_count[axis_cell],
        signed[axis_cell],
        axis_local[0],
    )
    axis_interval = jnp.argmax(
        positive & (axis_local[1] >= lower) & (axis_local[1] <= upper)
    )
    axis_fragment = fragments.slice_labels[axis_cell, axis_strip, axis_interval]
    axis_label = labels[axis_cell, jnp.maximum(axis_fragment, 0)]
    selected = (labels == axis_label) & (area > 0.0)
    valid = jnp.all(jnp.where(owner, saddle.valid, fragments.valid))
    valid = valid & jnp.any(axis_cells) & (axis_fragment >= 0)
    return (
        area,
        intervals,
        labels,
        wall,
        selected,
        owner,
        valid,
        jnp.max(fragments.required),
    )


def read(field, geometry, convention, policy):
    """Read total-field nulls and the open axis-connected hex-cell interior.

    All numerical geometry, current and exterior data are pytree operands.
    The own-node quadratic describes smooth cells; the selected saddle cell
    uses regular point-derivative rays and their analytic area integrals.
    Refusals retain explicit validity and integer reasons under jit and vmap.
    """
    if policy.fragment_capacity < 2:
        raise ValueError("the topology read requires two fragment slots per cell")
    sigma = convention.sigma
    coefficient = _field_coefficients(field, geometry)
    nulls, live, required = _null_census(field, geometry, coefficient, policy)
    eigenvalues = jnp.linalg.eigvalsh(sigma * nulls.jet.hessian)
    axes = live & jnp.all(eigenvalues < 0.0, axis=1)
    saddles = live & (eigenvalues[:, 0] < 0.0) & (eigenvalues[:, 1] > 0.0)
    axis_index = jnp.argmax(axes)
    axis, axis_flux = nulls.position[axis_index], nulls.jet.value[axis_index]
    saddle_index = jnp.argmax(jnp.where(saddles, sigma * nulls.jet.value, -jnp.inf))
    saddle_point = nulls.position[saddle_index]
    third = jax.jacfwd(lambda point: field.evaluate(point).hessian)(saddle_point)
    fourth = jax.jacfwd(jax.jacfwd(lambda point: field.evaluate(point).hessian))(
        saddle_point
    )
    form = saddle_normal_form(
        saddle_point,
        sigma * nulls.jet.hessian[saddle_index],
        sigma * third,
        policy.hessian_tolerance,
        sigma * fourth,
    )
    saddle_live = jnp.any(saddles)
    wall_points, wall_flux = _wall_events(field, geometry, policy, sigma)
    wall_index = jnp.argmax(sigma * wall_flux)
    initial_level = jnp.where(
        saddle_live, nulls.jet.value[saddle_index], wall_flux[wall_index]
    )
    initial = _support_at_level(
        geometry, coefficient, sigma, initial_level, axis, form, saddle_live, policy
    )
    area, intervals, labels, wall, selected, owner, support_valid, fragment_count = (
        initial
    )
    owner_index = jnp.argmax(owner)
    private_index = jnp.where(selected[owner_index, 0], 1, 0)
    private_label = labels[owner_index, private_index]
    private_reaches_wall = jnp.any(wall & (labels == private_label))
    core_reaches_saddle = jnp.any(owner[:, None] & selected)
    saddle_admitted = saddle_live & core_reaches_saddle & private_reaches_wall
    # Wall contacts in a private component cannot compete with the confined
    # branch. Closed contact points use the component of their owning cell.
    wall_cell = jax.vmap(lambda point: jnp.argmax(_inside_cells(geometry, point)))(
        wall_points
    )
    on_core = jnp.any(selected[wall_cell], axis=1)
    wall_score = jnp.where(~saddle_live | on_core, sigma * wall_flux, -jnp.inf)
    wall_index = jnp.argmax(wall_score)
    diverted = saddle_admitted & (sigma * initial_level >= wall_score[wall_index])
    boundary = jnp.where(diverted, saddle_point, wall_points[wall_index])
    level = jnp.where(diverted, initial_level, wall_flux[wall_index])
    terminal = _support_at_level(
        geometry, coefficient, sigma, level, axis, form, diverted, policy
    )
    area, intervals, labels, wall, selected, owner, support_valid, fragment_count = (
        terminal
    )
    membership = jnp.sum(jnp.where(selected, area, 0.0), axis=1) / geometry.full_area
    finite = jnp.all(jnp.isfinite(coefficient)) & jnp.all(jnp.isfinite(membership))
    support_valid = support_valid & jnp.all(
        (membership >= -policy.edge_tolerance)
        & (membership <= 1.0 + policy.edge_tolerance)
    )
    membership = jnp.clip(membership, 0.0, 1.0)
    unique_axis = jnp.sum(axes) == 1
    # Multiple saddle branches require an event tree rather than a scalar
    # ranking; refuse that unresolved census explicitly.
    unambiguous = (jnp.sum(saddles) <= 1) & unique_axis
    valid = finite & support_valid & (required <= policy.null_capacity) & unambiguous
    reason = jnp.where(
        required > policy.null_capacity,
        int(TopologyReason.CAPACITY),
        jnp.where(
            ~jnp.any(axes),
            int(TopologyReason.NO_QUALIFIED_AXIS),
            jnp.where(
                ~unambiguous,
                int(TopologyReason.UNRESOLVED_TIE),
                jnp.where(
                    ~finite,
                    int(TopologyReason.NONFINITE_FIELD),
                    jnp.where(
                        ~support_valid,
                        int(TopologyReason.UNRESOLVED_COMPONENT),
                        int(TopologyReason.OK),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    qualified = valid & (~saddle_live | saddle_admitted)
    admitted = saddles & saddle_admitted
    return TopologyRead(
        axis,
        axis_flux,
        nulls.position,
        nulls.jet.value,
        admitted,
        boundary,
        level,
        jnp.where(
            diverted, int(TopologyClass.DIVERTED), int(TopologyClass.LIMITED)
        ).astype(jnp.int32),
        jnp.where(valid, membership, jnp.nan),
        area,
        labels,
        selected,
        intervals,
        wall,
        jnp.where(jnp.any(selected, axis=1), 1, 0),
        coefficient,
        form,
        qualified,
        valid,
        reason,
        jnp.all(jnp.where(live, nulls.tangent_valid, True)),
        required,
        fragment_count,
    )
