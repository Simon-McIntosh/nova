"""Evaluate a display flux map on a fine grid from a state's solved currents.

The flux a state produces is a matrix-vector product: the conductor currents and
the plasma-cell uniform, radial and vertical current moments contract against
the coupling kernels at whatever points are asked for.  A panel therefore does
not have to draw the lattice the solve was carried on; it can evaluate the same
field wherever it wants the contour to be read.

The coupling is supplied by the caller as :class:`DisplayCoupling`, a pair of
callables over target points.  Every producer already owns one -- the same
``hybrid_greens`` columns and circuit response rows it builds its lattice target
from -- so this module adds no kernel of its own and never interpolates the
lattice.  The moment product is carried by :class:`nova.biot.target.FluxTarget`,
built once per evaluation chunk, so the uniform, radial and vertical columns
contract exactly as the forward operator's own grid target contracts them
rather than through a second copy of that product rule.

The lattice spline stays the solve's authority.  The map equals the state at the
lattice nodes only when the state is a converged fixed point, so the builder
reports the largest node mismatch it sees.  The saddle of the map, and the
separatrix branches drawn from it, are read off the map's own tensor spline
through the same stationary-point polish and branch assembler the solve uses.

Where the solve is claimed to be a field of currents it is judged on the nodes
it claims: a reduced solve's closing trip keeps the trial state on the nodes it
promotes instead of the map's image, so those nodes carry a piece of the iterate
no field of the cell currents reproduces.  The builder therefore takes the
solve's own promoted mask and reports the node mismatch over the nodes it does
not promote, beside the unmasked figure it has always reported, the promoted
count and the distance from the nearest promoted node to the polished X-point.

The field a constrained route images is not only its conductor currents and cell
moments: a compensating conductor current folded into the prescribed vector
before imaging contributes flux of its own.  That term enters through
``compensation`` as a conductor-current flux and is added to the same sum the
external conductor flux is carried in, so the map is drawn from the currents the
solve actually imaged with.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D
from nova.biot.target import FluxTarget
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.equilibrium.separatrix_branches import (
    assemble_separatrix_branches,
)
from nova.equilibrium.topology import NoQualifiedAxisError
from nova.linalg.tensor_spline import fit_tensor_spline

#: Display spacing [m]: the coarsest a panel may be drawn at.
DEFAULT_SPACING = 0.010

#: Branch fields the map publishes, in the assembler's own vocabulary.
BRANCH_KEYS = (
    "closed_controls_rz",
    "closed_valid",
    "closed_segment_count",
    "open_controls_rz",
    "open_valid",
    "open_branch_valid",
    "open_branch_count",
    "open_segment_count",
    "closed_candidate_count",
    "well_formed",
)


@dataclass(frozen=True)
class DisplayCoupling:
    """Target-point coupling of one machine: flux blocks and external response.

    ``plasma`` returns the uniform, radial and vertical moment columns and
    ``external`` the flux the conductors and the prescribed-current response
    carry, both for the ``(n, 2)`` ``(radius, height)`` points they are handed.
    """

    plasma: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
    external: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DisplayFluxMap:
    """One state's field evaluated on its own rectangular display grid."""

    radius: np.ndarray
    height: np.ndarray
    flux: np.ndarray
    saddle_rz: np.ndarray
    saddle_flux: float
    lattice_saddle_rz: np.ndarray
    saddle_offset_mm: float
    branches: dict[str, np.ndarray] = field(default_factory=dict)
    saddle_converged: bool = False
    max_node_mismatch_wb: float = float("nan")
    max_solved_node_mismatch_wb: float = float("nan")
    promoted_node_count: int = 0
    nearest_promoted_to_xpoint_m: float = float("nan")
    reproduction_relative_error: float = float("nan")
    spacing_m: float = DEFAULT_SPACING
    build_seconds: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        """Return the map extent as ``(radial, vertical)``."""
        return self.flux.shape

    @property
    def well_formed(self) -> bool:
        """Return whether the map's separatrix assembly is well formed."""
        if "well_formed" not in self.branches:
            return False
        return bool(np.asarray(self.branches["well_formed"]))


def _target(coupling: DisplayCoupling, block: np.ndarray) -> FluxTarget:
    """Return the flux target of one block of display points.

    The target carries the plasma moment columns only: the external callable
    returns a flux sum rather than source columns, so there is no external
    product for the target to carry.
    """
    uniform, radial, vertical = (
        np.asarray(column, dtype=np.float64) for column in coupling.plasma(block)
    )
    return FluxTarget(
        source_target=jnp.zeros((block.shape[0], 0), dtype=jnp.float64),
        plasma_target=jnp.asarray(uniform),
        null=Null1D(jnp.asarray(block, dtype=jnp.float64)),
        plasma_target_r=jnp.asarray(radial),
        plasma_target_z=jnp.asarray(vertical),
    )


def _flux(
    coupling: DisplayCoupling,
    points: np.ndarray,
    moments,
    chunk: int,
    compensation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return the conductor-plus-plasma flux at ``points`` [Wb].

    Each chunk contracts its own moment columns through a
    :class:`FluxTarget`, so the uniform, radial and vertical moments are
    combined by the owner of that product rather than restated here.  The
    compensating conductor currents a constrained route folds into its
    prescribed vector are carried in the same sum as the external conductor
    flux, so they are one conductor-current term rather than a second
    coupling.
    """
    flux = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        total = np.asarray(
            _target(coupling, block).internal(moments), dtype=np.float64
        ) + np.asarray(coupling.external(block), dtype=np.float64)
        if compensation is not None:
            total = total + np.asarray(compensation(block), dtype=np.float64)
        flux[start : start + block.shape[0]] = total
    return flux


def _stationary_saddle(
    values: np.ndarray, radius: np.ndarray, height: np.ndarray, xpoint_rz
) -> tuple[np.ndarray, float, bool]:
    """Return ``values``' own polished saddle, its flux and whether it is one.

    The polish runs on the field's own tensor spline from a seed inside the
    saddle's cell, so the returned coordinate is a stationary point of this
    field rather than a coordinate carried over from another grid.
    ``xpoint_rz`` only selects which stationary cell is meant.  A point that
    the polish does not converge to, or whose Hessian is not of saddle type,
    is reported as absent: the position and flux are NaN and the caller draws
    no separatrix from it, so a seed that lands on an extremum cannot be read
    downstream as a saddle.
    """
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(values)
    )
    seed = np.asarray(xpoint_rz, dtype=np.float64).reshape(2)
    polished = polish_stationary_points(
        surface,
        jnp.asarray(seed)[None, :],
        jnp.ones((1,), dtype=bool),
    )
    position = np.asarray(polished["position_rz"], dtype=np.float64).reshape(2)
    value = np.asarray(polished["value"], dtype=np.float64).reshape(())
    converged = bool(np.asarray(polished["converged"]).reshape(()))
    hessian_type = int(np.asarray(polished["hessian_type"]).reshape(()))
    if converged and hessian_type < 0 and np.all(np.isfinite(position)):
        return position, float(value), True
    return np.full(2, np.nan), float("nan"), False


def _branches(
    values: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    level: float,
    axis_rz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the field's own separatrix at its admitted-saddle level."""
    # A fine contour can traverse both sides of each grid row and column.
    # Round the resulting display-grid allowance up for a stable JIT shape.
    branch_capacity = 1 << (2 * (radius.size + height.size) - 1).bit_length()
    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(values),
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(level),
            jnp.asarray(axis_rz),
            branch_capacity=branch_capacity,
        )
    )
    return {key: np.asarray(assembled[key]) for key in BRANCH_KEYS}


def _topology_state(operator, state, requested_class):
    """Return the operator's own axis and X-point read of ``state``."""
    try:
        _masks, topology = operator.read(state, requested_class)
    except NoQualifiedAxisError:
        return None
    return topology


def build_display_flux_map(
    profile,
    state,
    *,
    coupling: DisplayCoupling,
    moments=None,
    spacing: float = DEFAULT_SPACING,
    extent: tuple[float, float, float, float] | None = None,
    current=None,
    prescribed_current=None,
    target_current=None,
    requested_class=None,
    axis_rz=None,
    xpoint_rz=None,
    chunk: int = 4096,
    promoted=None,
    compensation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> DisplayFluxMap:
    """Return one state's field on a fine display grid.

    The map is evaluated from the state's own conductor and plasma currents,
    plus the compensating conductor currents ``compensation`` carries when a
    constrained route supplied them, at every display point.
    ``reproduction_relative_error`` is the largest deviation of that evaluation
    from the operator's own ``raster_image`` at the lattice nodes, and
    ``max_node_mismatch_wb`` the largest deviation from the state itself over
    every lattice node: zero for a converged fixed point, the fixed-point
    residual otherwise.  ``promoted`` is the solve's own mask of the nodes its
    closing trip kept at the trial state; those nodes carry a piece of the
    iterate no field of currents reproduces, so they are excluded from
    ``max_solved_node_mismatch_wb``, which is the mismatch over the nodes the
    solve does claim.  ``promoted_node_count`` and
    ``nearest_promoted_to_xpoint_m`` locate that set, the distance being taken
    to the polished X-point.  The saddles of the map and of the lattice spline are each
    polished on their own spline, and the map's separatrix branches are
    assembled on the map at the map's own saddle value.  A polish that does not
    land on a saddle -- either it fails to converge or the Hessian there is not
    of saddle type -- is reported as absent: the saddle position and value are
    NaN, ``saddle_converged`` is False, and no branches are assembled.
    """
    started = time.perf_counter()
    if spacing > DEFAULT_SPACING:
        raise ValueError(
            f"display spacing {spacing} m is coarser than the {DEFAULT_SPACING} m floor"
        )
    lattice = profile.lattice
    operator = profile.operator
    if moments is None:
        if target_current is None:
            moments = operator.current_moments_and_observation(state, requested_class)[
                0
            ]
        else:
            moments = operator.normalised_current_moments_and_observation(
                state, target_current, requested_class
            )[0]
    nodewise = (moments.cell_current, moments.radial_moment, moments.vertical_moment)
    node_count = int(lattice.node_count)
    if promoted is None:
        promoted_nodes = np.zeros(node_count, dtype=bool)
    else:
        promoted_nodes = np.asarray(promoted, dtype=bool).reshape(-1)
        state_size = np.asarray(state).size
        if promoted_nodes.size not in (node_count, state_size):
            raise ValueError(
                f"the promoted mask names {promoted_nodes.size} nodes, "
                f"the lattice carries {node_count} and the state {state_size}"
            )
        promoted_nodes = promoted_nodes[:node_count]

    radius_nodes = np.asarray(lattice.radius, dtype=np.float64)
    height_nodes = np.asarray(lattice.height, dtype=np.float64)
    bounds = extent or (
        float(radius_nodes[0]),
        float(radius_nodes[-1]),
        float(height_nodes[0]),
        float(height_nodes[-1]),
    )
    radius = np.arange(bounds[0], bounds[1] + 0.5 * spacing, spacing)
    height = np.arange(bounds[2], bounds[3] + 0.5 * spacing, spacing)
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[radius_grid.ravel(), height_grid.ravel()]
    flux = _flux(coupling, points, nodewise, chunk, compensation).reshape(
        radius.size, height.size
    )
    values = flux.T

    node_coordinate = np.asarray(lattice.coordinate)
    node_flux = _flux(coupling, node_coordinate, nodewise, chunk, compensation)
    compensation_nodes = (
        np.zeros(node_count, dtype=np.float64)
        if compensation is None
        else np.asarray(compensation(node_coordinate), dtype=np.float64)
    )
    raster = np.asarray(
        operator.raster_image(
            moments, current=current, prescribed_current=prescribed_current
        )[:node_count],
        dtype=np.float64,
    )
    state_nodes = np.asarray(state, dtype=np.float64)[:node_count]
    scale = max(float(np.max(np.abs(raster))), 1.0e-30)
    reproduction = (
        float(np.max(np.abs(node_flux - compensation_nodes - raster))) / scale
    )
    deviation = np.abs(node_flux - state_nodes)
    mismatch = float(np.max(deviation))
    solved_nodes = ~promoted_nodes
    solved_mismatch = (
        float(np.max(deviation[solved_nodes])) if solved_nodes.any() else float("nan")
    )

    topology = None
    if axis_rz is None or xpoint_rz is None:
        topology = _topology_state(operator, state, requested_class)
    if axis_rz is None and topology is not None:
        axis_rz = topology.axis
    if xpoint_rz is None and topology is not None:
        xpoint_rz = topology.x_point

    lattice_values = np.asarray(
        np.asarray(state, dtype=np.float64)[: lattice.node_count]
    ).reshape(tuple(int(value) for value in np.asarray(lattice.shape)))
    lattice_values = lattice_values.T

    saddle_converged = False
    if xpoint_rz is None:
        saddle_rz = np.full(2, np.nan)
        saddle_flux = float("nan")
        lattice_saddle_rz = np.full(2, np.nan)
        branches: dict[str, np.ndarray] = {}
    else:
        seed = np.asarray(xpoint_rz, dtype=np.float64).reshape(-1)[:2]
        saddle_rz, saddle_flux, saddle_converged = _stationary_saddle(
            values, radius, height, seed
        )
        lattice_saddle_rz, _lattice_flux, _lattice_converged = _stationary_saddle(
            lattice_values, radius_nodes, height_nodes, seed
        )
        if axis_rz is None or not saddle_converged:
            branches = {}
        else:
            branches = _branches(
                values, radius, height, saddle_flux, np.asarray(axis_rz).reshape(2)
            )
    offset = float(1.0e3 * np.linalg.norm(saddle_rz - lattice_saddle_rz))
    nearest_promoted = float("nan")
    if promoted_nodes.any():
        if saddle_converged and np.all(np.isfinite(saddle_rz)):
            offsets = node_coordinate[promoted_nodes] - saddle_rz
            nearest_promoted = float(np.min(np.linalg.norm(offsets, axis=1)))
    return DisplayFluxMap(
        radius=radius,
        height=height,
        flux=flux,
        saddle_rz=np.asarray(saddle_rz, dtype=np.float64),
        saddle_flux=saddle_flux,
        lattice_saddle_rz=np.asarray(lattice_saddle_rz, dtype=np.float64),
        saddle_offset_mm=offset,
        branches=branches,
        saddle_converged=saddle_converged,
        max_node_mismatch_wb=mismatch,
        max_solved_node_mismatch_wb=solved_mismatch,
        promoted_node_count=int(np.count_nonzero(promoted_nodes)),
        nearest_promoted_to_xpoint_m=nearest_promoted,
        reproduction_relative_error=reproduction,
        spacing_m=float(spacing),
        build_seconds=time.perf_counter() - started,
    )


__all__ = [
    "BRANCH_KEYS",
    "DEFAULT_SPACING",
    "DisplayCoupling",
    "DisplayFluxMap",
    "build_display_flux_map",
]
