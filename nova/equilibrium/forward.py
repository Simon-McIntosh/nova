r"""Forward free-boundary equilibrium solve from prescribed source functions.

``ForwardProfile`` is the public problem: prescribed sources and flux
functions in, equilibrium flux out. It is one cell of Nova's
``ProblemRepresentation`` matrix and has exactly one implementation-level
map, :class:`~nova.equilibrium.forward_operator.ForwardFluxOperator`, which
carries the same solve as a traced function so a fixed-point ladder can
differentiate it and a batch of slices can be solved under ``vmap``.

The solve reads no magnetic measurement, forms no whitened residual and
updates no profile coefficient. A caller that needs a state inferred from
diagnostics reconstructs it first with
:class:`~nova.equilibrium.profile.ReconstructProfile`, or supplies the
posterior source state an upstream estimator produced.

Host and accelerated routes drive the same immutable state and return the
same typed result. ``host`` runs the relaxed fixed-point step under host
control flow with an early exit, ``host_krylov`` hands the residual to a
Jacobian-free Newton-Krylov root find, and the three accelerated routes hand
the map to the shared fixed-point ladder in
:mod:`nova.equilibrium.fixed_point`. All of them report their residual
history through one
:class:`~nova.equilibrium.fixed_point.FixedPointResult`, so a convergence
plot does not need to know which route produced it.

Which fixed point a route reaches is a property of the seed, not of the
accelerator. An absolute source drives current only where the topology read
finds an axis-connected core, so the map always has a second, trivial fixed
point at the vacuum field; a globalised Newton step can cross to it. The
receipt reports the branch it landed on — an empty core and a zero ledger —
rather than hiding the outcome behind a converged residual.

Which route can reach a fixed point at all is a property of the map. A
relaxed iteration converges only where the map contracts, and the
free-boundary map of an elongated column held at fixed conductor currents
does not: displacing the column vertically moves it into shaping field whose
decay index is negative there, so the force acts along the displacement
rather than against it, and the Jacobian of the write-then-read cycle carries
an eigenvalue outside the unit circle on that mode. ``picard`` and
``anderson`` mix successive images of that map, so they walk away from such
an equilibrium however they are damped or seeded, until the column meets the
wall and the source switches itself off — and the vacuum branch they land on
converges to a BETTER residual than the equilibrium they left, which is why a
prescribed-source solve cannot be qualified by its residual alone.
``newton_krylov`` solves ``(I - J) s = f`` rather than iterating ``g``, and a
root find is indifferent to the sign of that eigenvalue, so it holds the
equilibrium — which is why it is the default route. A relaxed route stays the
cheaper choice wherever the map does contract, which is the limited,
low-elongation case, and the residual history says which case a caller is in
without any extra diagnostic: a relaxed residual that GROWS from a good seed
is the non-contractive mode, not a bad start.

No route resolves flux differences below the quantum of the axis-flux read
the normalised flux is formed against. That read is fitted in the precision
the null search carries, so the floor scales with the flux itself rather than
being a fixed number, and a tolerance carried over from another case can be
unreachable on this one. A Newton budget that has already reached the floor
spends its remaining steps rattling inside it rather than descending.

The result is a receipt, not just a flux map: it carries the residual
history, the axis and separatrix state, the domain-labelled current ledger,
the integral observations, the conservation residuals, the finite checks,
the force-balance closure the source declared with the conventions it is
unreadable without, the continuation each open domain was driven under, and
every normalisation action the solve took — which, for an absolute source, is
none.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium import fixed_point
from nova.equilibrium.conservation import (
    ConservationLedger,
    FluxLattice,
    FluxMesh,
    conservation_ledger,
    poloidal_field,
)
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.moment import (
    CurrentIntegralSupport,
    CurrentCells,
    MomentSeed,
    MomentConfig,
    MomentOrder,
    ReconstructMoment,
    limiter_radial_extent,
)
from nova.equilibrium.observation import (
    ClippedIntegralMeasure,
    ConstraintPinSet,
    ConstraintViolationError,
    CurrentLedger,
    CurrentMomentObservation,
    IntegralObservation,
    MomentIntegralSupport,
    MomentTargets,
    current_ledger,
    core_pressure,
    moment_residual,
    observe_current_moments,
    observe_moments,
    reject_unsupported_enforcement,
)
from nova.equilibrium.observation_kernels import ThomsonSignals, synthesize_thomson
from nova.equilibrium.map_extraction import sample_chord_psi_norm
from nova.equilibrium.source import (
    ContinuationLedger,
    CurrentNormalisationError,
    ForwardSource,
    NormalisationRecord,
    RotationRecord,
)
from nova.equilibrium.stencil_mesh import (
    CellCurrentMoments,
    MomentGeometry,
    StencilMesh,
)
from nova.equilibrium.topology import TopologyClass, TopologyState
from nova.geometry.hexstencil import hex_stencil

__all__ = [
    "FiniteCheck",
    "ColdSeedConstruction",
    "ForwardBranchReceipt",
    "ForwardColdSeedPortfolio",
    "ForwardColdSeedReceipt",
    "ForwardEquilibrium",
    "ForwardPerturbedSeedReceipt",
    "ForwardPortfolio",
    "ForwardProfile",
    "PerturbedSeedPolicy",
    "SaddleSeedGeometry",
]

#: Routes that drive the same map to the same fixed point.
SolveRoute = Literal["host", "host_krylov", "picard", "anderson", "newton_krylov"]

_HOST: tuple[str, ...] = ("host", "host_krylov")
_ACCELERATED: tuple[str, ...] = ("picard", "anderson", "newton_krylov")


def _lattice_cells(lattice: FluxLattice) -> tuple[np.ndarray, ...]:
    """Return rectangular control polygons centred on a structured lattice."""
    half_radial = 0.5 * lattice.radial_step
    half_vertical = 0.5 * lattice.vertical_step
    offset = np.asarray(
        [
            [-half_radial, -half_vertical],
            [half_radial, -half_vertical],
            [half_radial, half_vertical],
            [-half_radial, half_vertical],
        ]
    )
    return tuple(coordinate + offset for coordinate in lattice.coordinate)


def _cubic_cell_average_stencil(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the degree-three-exact five-node rectangular cell rule.

    The midpoint value carries weight ``5/6`` and its four axial neighbours
    each carry ``1/24``.  This is the cell-average Taylor correction through
    the discrete second derivatives, so it integrates every total-degree-three
    polynomial exactly and has fourth-order error for a smooth density.  At
    the outermost raster row the centre is repeated in all five slots, keeping
    the centroid rule where a complete cross does not exist.
    """
    radial_count, vertical_count = shape
    index = np.arange(radial_count * vertical_count, dtype=np.intp).reshape(shape)
    stencil = np.repeat(index[..., None], 5, axis=2)
    stencil[1:-1, 1:-1, 1] = index[:-2, 1:-1]
    stencil[1:-1, 1:-1, 2] = index[2:, 1:-1]
    stencil[1:-1, 1:-1, 3] = index[1:-1, :-2]
    stencil[1:-1, 1:-1, 4] = index[1:-1, 2:]
    weight = np.asarray([5.0 / 6.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0])
    return stencil.reshape(-1, 5), weight


class FiniteCheck(NamedTuple):
    """Finiteness of every array the result publishes."""

    flux: jax.Array
    cell_current: jax.Array
    moments: jax.Array
    conservation: jax.Array

    @property
    def passed(self) -> jax.Array:
        """Return whether every published array is finite."""
        return self.flux & self.cell_current & self.moments & self.conservation


class ForwardEquilibrium(NamedTuple):
    """Converged equilibrium and the receipts that qualify it."""

    flux: jax.Array
    cell_current: jax.Array
    domains: DomainMasks
    topology: TopologyState
    fixed_point: fixed_point.FixedPointResult
    moments: IntegralObservation
    ledger: CurrentLedger
    conservation: ConservationLedger
    normalisation: NormalisationRecord
    rotation: RotationRecord
    continuation: ContinuationLedger
    finite: FiniteCheck


class ForwardBranchReceipt(NamedTuple):
    """One topology-pinned branch and its terminal convergence qualification."""

    equilibrium: ForwardEquilibrium
    requested_class: jax.Array
    achieved_class: jax.Array
    converged: jax.Array
    residual: jax.Array
    iterations: jax.Array
    topology_consistent: jax.Array


class ForwardPortfolio(NamedTuple):
    """Limited and diverted receipts stacked on one fixed branch axis."""

    branches: ForwardBranchReceipt


class ColdSeedConstruction(IntEnum):
    """Independent geometry used to construct one cold branch seed."""

    CURRENT_CENTROID_DISC = 0
    AXIS_SADDLE_GEOMETRY = 1


@dataclass(frozen=True)
class SaddleSeedGeometry:
    """Declared magnetic-axis and saddle locations for a diverted cold seed."""

    axis: tuple[float, float]
    saddle: tuple[float, float]

    def __post_init__(self) -> None:
        """Require two distinct finite points inside physical coordinates."""

        axis = np.asarray(self.axis, dtype=np.float64)
        saddle = np.asarray(self.saddle, dtype=np.float64)
        if axis.shape != (2,) or saddle.shape != (2,):
            raise ValueError("axis and saddle must each be a coordinate pair")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(saddle)):
            raise ValueError("axis and saddle coordinates must be finite")
        if np.linalg.norm(saddle - axis) <= np.finfo(np.float64).eps:
            raise ValueError("axis and saddle coordinates must be distinct")
        object.__setattr__(self, "axis", tuple(float(value) for value in axis))
        object.__setattr__(self, "saddle", tuple(float(value) for value in saddle))


class ForwardColdSeedReceipt(NamedTuple):
    """One current-moment seed paired with its declared boundary anchor."""

    flux: jax.Array
    requested_class: jax.Array
    anchor: jax.Array
    anchor_flux: jax.Array
    anchor_available: jax.Array
    plasma_current: jax.Array
    centroid: jax.Array
    radius: jax.Array
    supported_cells: jax.Array
    construction: jax.Array
    declared_axis: jax.Array
    declared_boundary: jax.Array
    stored_flux_samples_used: jax.Array


class ForwardColdSeedPortfolio(NamedTuple):
    """Limited and diverted cold-seed receipts on one fixed branch axis."""

    branches: ForwardColdSeedReceipt


@dataclass(frozen=True)
class PerturbedSeedPolicy:
    """Declared near-basin amplitudes and bounded pinned-solve budget."""

    relative_amplitudes: tuple[float, ...] = (1.0e-3, 1.0e-2, 5.0e-2)
    newton_steps: int = 10
    gmres_iterations: int = 30
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        """Require a finite increasing ladder and positive solve controls."""
        amplitudes = np.asarray(self.relative_amplitudes, dtype=np.float64)
        if amplitudes.ndim != 1 or amplitudes.size < 1:
            raise ValueError("perturbation amplitudes need at least one rung")
        if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
            raise ValueError("perturbation amplitudes must be positive and finite")
        if np.any(np.diff(amplitudes) <= 0.0):
            raise ValueError("perturbation amplitudes must increase strictly")
        if self.newton_steps < 1 or self.gmres_iterations < 1:
            raise ValueError("perturbed-seed iteration budgets must be positive")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("perturbed-seed tolerance must be positive and finite")
        object.__setattr__(
            self,
            "relative_amplitudes",
            tuple(float(value) for value in amplitudes),
        )


class ForwardPerturbedSeedReceipt(NamedTuple):
    """Declared near-basin seeds and their pinned diverted solve outcomes."""

    relative_amplitude: jax.Array
    reference_flux_span: jax.Array
    seed_flux: jax.Array
    rungs: ForwardBranchReceipt
    root_relative_error: jax.Array
    passed: jax.Array
    largest_passing_amplitude: jax.Array


@dataclass
class ForwardProfile:
    """Solve the free-boundary equilibrium for prescribed sources and profiles.

    The flux functions supplied through
    :class:`~nova.equilibrium.source.ForwardSource` set the toroidal current
    density on the domains the topology read labels, and the equilibrium is
    the fixed point of the resulting write-then-read cycle. Fluxes are total
    poloidal fluxes, :math:`\\Phi = 2 \\pi R A_\\phi` in Wb. The physical grid
    and wall prefix may be followed by fixed direct-sampling nodes used only
    to construct the in-cell density on clipped supports.

    ``lattice`` is the mesh the plasma grid is carried on, meeting the
    :class:`~nova.equilibrium.conservation.FluxMesh` contract: a uniform
    :class:`~nova.equilibrium.conservation.FluxLattice` for a structured
    raster, or a :class:`~nova.equilibrium.stencil_mesh.StencilMesh` for the
    offset, wall-trimmed hexagonal tiling the package ships. It is required
    rather than optional because the conservation and integral receipts are
    differentiated on it; a solve that cannot produce its receipts is not the
    capability this class publishes. Nothing else in the solve reads it — the
    map, the ladder and the domain partition are already mesh-agnostic — so
    the mesh kind changes which stencil the receipts are formed on and
    nothing about the equilibrium that is found.
    """

    operator: ForwardFluxOperator
    lattice: FluxMesh
    evaluations: int = 60
    relaxation: float = 0.5
    newton_steps: int = 4

    def __post_init__(self):
        """Validate that the lattice indexes the operator's plasma grid."""
        if self.lattice.node_count != self.operator.grid.node_number:
            raise ValueError(
                "the flux lattice and the plasma grid must carry the same nodes"
            )
        if self.evaluations < 1:
            raise ValueError("evaluations must be at least one")

    @classmethod
    def from_lattice(
        cls,
        lattice: FluxLattice,
        source: ForwardSource,
        *,
        external_current,
        source_to_grid,
        plasma_to_grid,
        source_to_wall,
        plasma_to_wall,
        wall_coordinate,
        plasma_to_grid_r=None,
        plasma_to_grid_z=None,
        plasma_to_wall_r=None,
        plasma_to_wall_z=None,
        polarity: int = 1,
        inside_material=None,
        cubic_cell_average: bool = True,
        maxsize: int = 5,
        **kwargs,
    ) -> ForwardProfile:
        """Build the solve from a structured lattice and its coupling operators.

        The coupling operators carry total poloidal flux in Wb per ampere, the
        convention :func:`nova.biot.greens.hybrid_greens` assembles.
        """
        grid_null = Null2D.from_coordinates(
            lattice.coordinate, hex_stencil(lattice.shape), maxsize=maxsize
        )
        wall_null = Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64))
        moment_mesh = StencilMesh(
            coordinate=lattice.coordinate,
            stencil=hex_stencil(lattice.shape),
            area=lattice.cell_area,
        )
        cell_average_stencil, cell_average_weight = _cubic_cell_average_stencil(
            lattice.shape
        )
        operator = ForwardFluxOperator(
            grid=FluxTarget(
                source_target=jnp.asarray(source_to_grid),
                plasma_target=jnp.asarray(plasma_to_grid),
                null=grid_null,
                plasma_target_r=(
                    None if plasma_to_grid_r is None else jnp.asarray(plasma_to_grid_r)
                ),
                plasma_target_z=(
                    None if plasma_to_grid_z is None else jnp.asarray(plasma_to_grid_z)
                ),
            ),
            wall=FluxTarget(
                source_target=jnp.asarray(source_to_wall),
                plasma_target=jnp.asarray(plasma_to_wall),
                null=wall_null,
                plasma_target_r=(
                    None if plasma_to_wall_r is None else jnp.asarray(plasma_to_wall_r)
                ),
                plasma_target_z=(
                    None if plasma_to_wall_z is None else jnp.asarray(plasma_to_wall_z)
                ),
            ),
            source=source,
            external_current=jnp.asarray(external_current),
            area=jnp.asarray(lattice.cell_area),
            cell_average_stencil=(cell_average_stencil if cubic_cell_average else None),
            cell_average_weight=(cell_average_weight if cubic_cell_average else None),
            polarity=polarity,
            inside_material=inside_material,
            moment_geometry=MomentGeometry.from_cells(
                moment_mesh, _lattice_cells(lattice)
            ),
            use_linear_moments=all(
                block is not None
                for block in (
                    plasma_to_grid_r,
                    plasma_to_grid_z,
                    plasma_to_wall_r,
                    plasma_to_wall_z,
                )
            ),
        )
        return cls(operator=operator, lattice=lattice, **kwargs)

    @property
    def source(self) -> ForwardSource:
        """Return the immutable source state the solve consumes."""
        return self.operator.source

    def flux_map(
        self, current=None, requested_class=None, target_current=None
    ) -> Callable[[jax.Array], jax.Array]:
        """Return the traced map at one conductor state and optional constraints."""
        return self.operator.flux_map(current, requested_class, target_current)

    def cold_seed_portfolio(
        self,
        plasma_current: float,
        centroid,
        *,
        current=None,
        radius_fraction: float | None = None,
        diverted_geometry: SaddleSeedGeometry | None = None,
    ) -> ForwardColdSeedPortfolio:
        """Return production moment seeds paired with wall and saddle anchors.

        The limited state is the external field plus a uniform-disc zeroth
        current moment about the declared centroid. When diverted geometry is
        supplied, its seed is a cubic field whose only stationary points are
        the declared magnetic axis and saddle. Its gauge and axis-to-saddle
        span come from the independent moment seed, never from a reference
        flux map. The receipt retains every anchor input and that independence
        statement beside the production topology read of the resulting state.
        """

        centre = np.asarray(centroid, dtype=np.float64)
        if centre.shape != (2,) or not np.all(np.isfinite(centre)):
            raise ValueError("centroid must be a finite (radius, height) pair")
        config = MomentConfig(order=MomentOrder.CENTROID)
        fraction = (
            config.seed_radius_fraction
            if radius_fraction is None
            else float(radius_fraction)
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("radius_fraction must lie in (0, 1]")
        coordinate = np.asarray(self.operator.grid.coordinate, dtype=np.float64)
        wall = np.asarray(self.operator.wall.coordinate, dtype=np.float64)
        inboard, outboard = limiter_radial_extent(wall[:, 0], wall[:, 1], centre[1])
        supported_distance = min(centre[0] - inboard, outboard - centre[0])
        if supported_distance <= 0.0:
            raise ValueError("the current centroid lies outside the limiter")
        radius = fraction * supported_distance
        reconstruction = ReconstructMoment(
            CurrentCells(
                coordinate[:, 0],
                coordinate[:, 1],
                candidate=np.asarray(self.operator.inside_material, dtype=np.float64),
            ),
            major_radius=float(centre[0]),
            config=config,
        )
        cell_current = reconstruction.uniform_disc(
            float(centre[0]),
            float(centre[1]),
            radius,
            float(plasma_current),
        )
        uniform = jnp.asarray(cell_current, dtype=jnp.float64)
        zero = jnp.zeros_like(uniform)
        physical = CellCurrentMoments(uniform, zero, zero)
        coefficients = self.operator.coupling_current_moments(physical)
        neutral = self.operator.external(current) + self.operator.current_moment_image(
            coefficients
        )
        requested = jnp.asarray(
            (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)),
            dtype=jnp.int8,
        )

        limited_flux = neutral
        if diverted_geometry is None:
            diverted_flux = neutral
            diverted_axis = np.full(2, np.nan)
            diverted_boundary = np.full(2, np.nan)
            diverted_construction = ColdSeedConstruction.CURRENT_CENTROID_DISC
        else:
            diverted_flux = self._saddle_geometry_seed(neutral, diverted_geometry)
            diverted_axis = np.asarray(diverted_geometry.axis)
            diverted_boundary = np.asarray(diverted_geometry.saddle)
            diverted_construction = ColdSeedConstruction.AXIS_SADDLE_GEOMETRY
        flux = jnp.stack((limited_flux, diverted_flux))

        def anchor(branch_flux, branch_class):
            _masks, topology = self.operator.read(branch_flux, branch_class)
            available = jnp.all(jnp.isfinite(topology.boundary)) & jnp.isfinite(
                topology.boundary_flux
            )
            return topology.boundary, topology.boundary_flux, available

        anchors, anchor_flux, anchor_available = jax.vmap(anchor)(flux, requested)
        branch_count = requested.size
        declared_axis = jnp.stack((jnp.asarray(centre), jnp.asarray(diverted_axis)))
        declared_boundary = jnp.stack(
            (anchors[int(TopologyClass.LIMITED)], jnp.asarray(diverted_boundary))
        )
        return ForwardColdSeedPortfolio(
            branches=ForwardColdSeedReceipt(
                flux=flux,
                requested_class=requested,
                anchor=anchors,
                anchor_flux=anchor_flux,
                anchor_available=anchor_available,
                plasma_current=jnp.full(branch_count, float(plasma_current)),
                centroid=jnp.broadcast_to(jnp.asarray(centre), (branch_count, 2)),
                radius=jnp.full(branch_count, radius),
                supported_cells=jnp.full(
                    branch_count,
                    int(np.count_nonzero(cell_current)),
                    dtype=jnp.int32,
                ),
                construction=jnp.asarray(
                    (
                        int(ColdSeedConstruction.CURRENT_CENTROID_DISC),
                        int(diverted_construction),
                    ),
                    dtype=jnp.int8,
                ),
                declared_axis=declared_axis,
                declared_boundary=declared_boundary,
                stored_flux_samples_used=jnp.zeros(branch_count, dtype=bool),
            )
        )

    def moment_seed(
        self,
        boundary,
        plasma_current: float,
        *,
        current=None,
        radius_fraction: float | None = None,
    ) -> MomentSeed:
        """Predict amplitude/centroid and construct a constrained initial state.

        The flux-functions-only prediction is evaluated on the supplied
        boundary hypothesis by :class:`ReconstructMoment`. The compact seed
        then uses that class's disc support with a degree-one cell-centering
        correction, carrying the predicted net current and centroid exactly.
        Its radius remains a limiter-derived construction scale rather than a
        claimed width prediction.
        """

        coordinate = np.asarray(self.operator.grid.coordinate, dtype=np.float64)
        candidate = np.asarray(self.operator.inside_material, dtype=np.float64)
        reconstruction = ReconstructMoment(
            CurrentCells(
                coordinate[:, 0],
                coordinate[:, 1],
                candidate=candidate,
            ),
            major_radius=float(np.mean(coordinate[:, 0])),
            config=MomentConfig(order=MomentOrder.CENTROID),
        )
        prediction = reconstruction.predict_profile_moments(
            self.source.core,
            np.asarray(boundary, dtype=np.float64),
            float(plasma_current),
            cell_area=np.asarray(self.lattice.cell_area, dtype=np.float64),
        )
        centre = np.asarray(prediction.centroid, dtype=np.float64)
        wall = np.asarray(self.operator.wall.coordinate, dtype=np.float64)
        inboard, outboard = limiter_radial_extent(wall[:, 0], wall[:, 1], centre[1])
        supported_distance = min(centre[0] - inboard, outboard - centre[0])
        if supported_distance <= 0.0:
            raise ValueError("the predicted current centroid lies outside the limiter")
        fraction = (
            reconstruction.config.seed_radius_fraction
            if radius_fraction is None
            else float(radius_fraction)
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("radius_fraction must lie in (0, 1]")
        radius = fraction * supported_distance
        cell_current = reconstruction.centroid_disc(
            prediction.centroid_r,
            prediction.centroid_z,
            radius,
            prediction.plasma_current,
        )
        uniform = jnp.asarray(cell_current, dtype=jnp.float64)
        zero = jnp.zeros_like(uniform)
        physical = CellCurrentMoments(uniform, zero, zero)
        coefficients = self.operator.coupling_current_moments(physical)
        flux = self.operator.external(current) + self.operator.current_moment_image(
            coefficients
        )
        return MomentSeed(
            flux=flux,
            cell_current=uniform,
            moments=prediction,
            radius=float(radius),
            support=CurrentIntegralSupport.COMPACT_CENTROID_DISC,
            supported_cells=int(np.count_nonzero(cell_current)),
        )

    def _saddle_geometry_seed(
        self,
        moment_seed: jax.Array,
        geometry: SaddleSeedGeometry,
    ) -> jax.Array:
        """Return a cold field normalized at declared axis and saddle points."""

        axis = np.asarray(geometry.axis, dtype=np.float64)
        saddle = np.asarray(geometry.saddle, dtype=np.float64)
        displacement = saddle - axis
        distance = float(np.linalg.norm(displacement))
        along = displacement / distance
        across = np.array((-along[1], along[0]))
        coordinate_parts = [
            np.asarray(self.operator.grid.coordinate, dtype=np.float64),
            np.asarray(self.operator.wall.coordinate, dtype=np.float64),
        ]
        if self.operator.sample is not None:
            coordinate_parts.append(
                np.asarray(self.operator.sample.coordinate, dtype=np.float64)
            )
        coordinates = np.vstack(coordinate_parts)
        grid = coordinate_parts[0]
        axis_index = int(np.argmin(np.linalg.norm(grid - axis, axis=1)))
        saddle_index = int(np.argmin(np.linalg.norm(grid - saddle, axis=1)))
        neutral = np.asarray(moment_seed, dtype=np.float64)
        polarity = float(self.operator.polarity)
        flux_span = polarity * (neutral[axis_index] - neutral[saddle_index])
        resolution = np.finfo(np.float64).eps * max(np.max(np.abs(neutral)), 1.0)
        flux_span = max(float(flux_span), float(resolution))
        local = coordinates - axis
        axial = local @ along / distance
        transverse = local @ across / distance
        potential = -0.5 * axial**2 + axial**3 / 3.0 - 0.5 * transverse**2
        normalized = 6.0 * (potential + 1.0 / 6.0)
        saddle_flux = neutral[saddle_index]
        return jnp.asarray(saddle_flux + polarity * flux_span * normalized)

    def observe(self, flux, current=None, target_current=None) -> ForwardEquilibrium:
        """Return the full receipt of one flux map without iterating it.

        ``fixed_point`` reports the residual of the supplied map alone, so a
        caller can qualify an externally produced flux map through the same
        contract a solve returns.
        """
        residual = jnp.max(
            jnp.abs(
                self.operator.residual(flux, current, target_current=target_current)
            )
        )
        scale = jnp.maximum(jnp.max(jnp.abs(flux)), 1.0e-30)
        history = fixed_point.FixedPointResult(
            state=jnp.asarray(flux),
            residual=residual / scale,
            trace=jnp.atleast_1d(residual / scale),
        )
        return self._receipt(jnp.asarray(flux), history, target_current=target_current)

    def integral_observation(self, flux, target_current=None) -> IntegralObservation:
        """Return the integral observations of one flux map.

        This is the differentiable moment map: it reads the topology, applies
        the declared source and integrates, with no conservation differencing
        in the way, so ``jacfwd`` through it costs one observation.
        """
        _current, support_integrals, _masks, topology, _amplitude = (
            self._integral_state(flux, target_current=target_current)
        )
        return observe_moments(support_integrals, topology.flux_span)

    def thomson_observation(
        self,
        flux,
        profile_psi_norm,
        electron_temperature,
        electron_density,
        chord_coordinates,
        **options,
    ) -> ThomsonSignals:
        """Compose a solved flux map with the deterministic Thomson kernel."""
        if not isinstance(self.lattice, FluxLattice):
            raise TypeError("Thomson observations require a structured FluxLattice")
        state = jnp.asarray(flux)
        grid_flux = state[: self.lattice.node_count].reshape(self.lattice.shape)
        _masks, topology = self.operator.read(state)
        return synthesize_thomson(
            self.lattice.radius,
            self.lattice.height,
            grid_flux,
            profile_psi_norm,
            electron_temperature,
            electron_density,
            chord_coordinates,
            axis_flux=topology.axis_flux,
            boundary_flux=topology.boundary_flux,
            **options,
        )

    def thomson_observation_map(
        self,
        flux,
        profile_psi_norm,
        electron_temperature,
        electron_density,
        chord_coordinates,
        **options,
    ) -> jax.Array:
        """Return Thomson temperature and density on one fixed output axis."""
        signals = self.thomson_observation(
            flux,
            profile_psi_norm,
            electron_temperature,
            electron_density,
            chord_coordinates,
            **options,
        )
        return jnp.concatenate(
            (
                signals.electron_temperature.reshape(-1),
                signals.electron_density.reshape(-1),
            )
        )

    def thomson_observation_jacobian(
        self,
        flux,
        profile_psi_norm,
        electron_temperature,
        electron_density,
        chord_coordinates,
        **options,
    ) -> jax.Array:
        """Differentiate the public Thomson map with respect to solved flux."""
        return jax.jacfwd(
            lambda state: self.thomson_observation_map(
                state,
                profile_psi_norm,
                electron_temperature,
                electron_density,
                chord_coordinates,
                **options,
            )
        )(jnp.asarray(flux))

    def current_moment_observation(
        self,
        flux,
        *,
        support: MomentIntegralSupport,
        target_current=None,
    ) -> CurrentMomentObservation:
        """Return net current and centroid on one explicitly declared support."""
        current, _integrals, masks, _topology, _amplitude = self._integral_state(
            flux, target_current=target_current
        )
        return observe_current_moments(
            current.cell_current,
            self.operator.grid.coordinate,
            core_mask=masks.core,
            support=support,
        )

    def current_moment_map(
        self,
        flux,
        *,
        support: MomentIntegralSupport,
        target_current=None,
    ) -> jax.Array:
        """Return current amplitude and centroid on one fixed output axis."""
        return self.current_moment_observation(
            flux, support=support, target_current=target_current
        ).stack()

    def current_moment_jacobian(
        self,
        flux,
        *,
        support: MomentIntegralSupport,
        target_current=None,
    ) -> jax.Array:
        """Differentiate the support-declared current-moment map over flux."""
        return jax.jacfwd(
            lambda state: self.current_moment_map(
                state, support=support, target_current=target_current
            )
        )(jnp.asarray(flux))

    def constraint_residual(
        self, flux, pins: ConstraintPinSet, target_current=None
    ) -> jax.Array:
        """Return interval-scaled deterministic residuals for trusted pins.

        Each isoflux pair contributes one residual per endpoint against the
        shared target. Each current moment contributes one residual on its
        declared support. The uncertainty intervals are acceptance scales,
        not statistical weights.
        """
        if not isinstance(pins, ConstraintPinSet):
            raise TypeError("pins must be a ConstraintPinSet")
        if pins.isoflux and not isinstance(self.lattice, FluxLattice):
            raise TypeError("isoflux constraints require a structured FluxLattice")
        state = jnp.asarray(flux)
        residuals = []
        if pins.isoflux:
            grid_flux = state[: self.lattice.node_count].reshape(self.lattice.shape)
            _masks, topology = self.operator.read(state)
            for pin in pins.isoflux:
                coordinates = jnp.asarray(
                    (pin.first_coordinate, pin.second_coordinate), dtype=state.dtype
                )
                sampled = sample_chord_psi_norm(
                    self.lattice.radius,
                    self.lattice.height,
                    grid_flux,
                    coordinates,
                    axis_flux=topology.axis_flux,
                    boundary_flux=topology.boundary_flux,
                )
                residuals.extend(
                    (sampled.psi_norm - pin.psi_norm) / pin.uncertainty.absolute
                )
        observations: dict[MomentIntegralSupport, CurrentMomentObservation] = {}
        for pin in pins.moments:
            if pin.support not in observations:
                observations[pin.support] = self.current_moment_observation(
                    state, support=pin.support, target_current=target_current
                )
            observed = observations[pin.support].value(pin.name)
            residuals.append((observed - pin.target) / pin.uncertainty.absolute)
        return jnp.stack(residuals)

    def constraint_jacobian(
        self, flux, pins: ConstraintPinSet, target_current=None
    ) -> jax.Array:
        """Differentiate trusted-pin residuals with respect to solved flux."""
        return jax.jacfwd(
            lambda state: self.constraint_residual(state, pins, target_current)
        )(jnp.asarray(flux))

    def constraints_satisfied(
        self, flux, pins: ConstraintPinSet, target_current=None
    ) -> jax.Array:
        """Return whether every trusted pin lies inside its stated interval."""
        residual = self.constraint_residual(flux, pins, target_current)
        return jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.abs(residual) <= 1.0)

    def _require_constraints(
        self, flux, pins: ConstraintPinSet | None, target_current=None
    ) -> None:
        """Refuse a candidate root outside any supplied trusted interval."""
        if pins is None:
            return
        residual = np.asarray(self.constraint_residual(flux, pins, target_current))
        if not np.all(np.isfinite(residual)) or np.any(np.abs(residual) > 1.0):
            worst = float(np.nanmax(np.abs(residual)))
            raise ConstraintViolationError(
                "solved state violates a trusted constraint interval; "
                f"maximum scaled residual {worst:.6g}"
            )

    def _integral_state(self, flux, requested_class=None, target_current=None):
        """Return current and integral state for this construction's geometry."""
        if self.operator.use_linear_moments:
            if target_current is None:
                return (
                    *self.operator.current_moments_and_observation(
                        flux, requested_class
                    ),
                    None,
                )
            return self.operator.normalised_current_moments_and_observation(
                flux, target_current, requested_class
            )
        masks, topology = self.operator.read(flux, requested_class)
        current_moments = self.operator.cell_current_moments(flux, requested_class)
        cell_current = current_moments.cell_current
        radius = jnp.asarray(self.lattice.node_radius)
        area = jnp.where(masks.core, self.operator.area, 0.0)
        volume = 2.0 * jnp.pi * radius * area
        radial, vertical = poloidal_field(
            self.lattice, jnp.asarray(flux)[: self.lattice.node_count]
        )
        pressure = jnp.where(
            masks.core,
            core_pressure(self.operator.source, masks, radius, topology.flux_span),
            0.0,
        )
        support_integrals = ClippedIntegralMeasure(
            area=area,
            volume=volume,
            radial_volume=radius * volume,
            cell_current=jnp.where(masks.core, cell_current, 0.0),
            pressure_volume=pressure * volume,
            field_volume=(radial**2 + vertical**2) * volume,
            masks=masks,
        )
        amplitude = None
        if target_current is not None:
            amplitude = self.operator.current_normalisation_amplitude(
                target_current, jnp.sum(current_moments.cell_current)
            )
            current_moments = self.operator.scaled_current_moments(
                current_moments, amplitude
            )
            support_integrals = support_integrals.with_current_amplitude(amplitude)
        return current_moments, support_integrals, masks, topology, amplitude

    def _receipt(
        self,
        flux: jax.Array,
        history: fixed_point.FixedPointResult,
        requested_class=None,
        target_current=None,
    ) -> ForwardEquilibrium:
        """Return the typed result of one converged or supplied flux map."""
        current_moments, support_integrals, masks, topology, amplitude = (
            self._integral_state(flux, requested_class, target_current)
        )
        cell_current = current_moments.cell_current
        grid_flux = flux[: self.lattice.node_count]
        radius = jnp.asarray(self.lattice.node_radius)
        moments = observe_moments(support_integrals, topology.flux_span)
        conservation = conservation_ledger(
            self.lattice,
            grid_flux,
            self.operator.source,
            masks,
            topology.flux_span,
        )
        return ForwardEquilibrium(
            flux=flux,
            cell_current=cell_current,
            domains=masks,
            topology=topology,
            fixed_point=history,
            moments=moments,
            ledger=current_ledger(cell_current, support_integrals.masks),
            conservation=conservation,
            normalisation=self.source.normalisation_record(
                flux.dtype, amplitude=amplitude
            ),
            rotation=self.operator.source.rotation_record(radius, masks),
            continuation=self.operator.source.continuation_ledger(flux.dtype),
            finite=FiniteCheck(
                flux=jnp.all(jnp.isfinite(flux)),
                cell_current=jnp.all(jnp.isfinite(cell_current)),
                moments=jnp.all(jnp.isfinite(moments.stack())),
                conservation=jnp.all(jnp.isfinite(jnp.stack([*conservation[:-1]]))),
            ),
        )

    def _host_history(
        self, trace, flux, current, target_current=None
    ) -> fixed_point.FixedPointResult:
        """Return the shared fixed-point result of a host solve."""
        mapped = self.operator(flux, current, target_current=target_current)
        scale = jnp.maximum(jnp.max(jnp.abs(mapped)), 1.0e-30)
        return fixed_point.FixedPointResult(
            state=flux,
            residual=jnp.max(jnp.abs(mapped - flux)) / scale,
            trace=jnp.asarray(trace),
        )

    def _solve_host(
        self,
        initial_flux,
        current,
        *,
        evaluations: int | None = None,
        relaxation: float | None = None,
        tolerance: float = 1.0e-10,
        target_current=None,
        **options,
    ) -> ForwardEquilibrium:
        """Drive the map with a host relaxed fixed-point iteration.

        The eager counterpart of :func:`nova.equilibrium.fixed_point.picard`:
        the same relaxed step under host control flow, with an early exit
        once the relative residual passes ``tolerance``. Run to the same
        evaluation count with an unreachable tolerance it reproduces the
        traced ladder step for step while the domain labels hold; once the
        residual reaches the flux one boundary cell's current carries, the
        relaxed step cycles between labellings instead of contracting and
        the two routes separate at that floor.
        """
        if options:
            raise TypeError(
                f"unexpected host solve argument(s) {', '.join(sorted(options))}"
            )
        budget = self.evaluations if evaluations is None else int(evaluations)
        step = self.relaxation if relaxation is None else float(relaxation)
        mapped = self.flux_map(current, target_current=target_current)
        trace = np.full(budget, np.nan)
        state = np.asarray(initial_flux, dtype=np.float64)
        for index in range(budget):
            image = np.asarray(mapped(jnp.asarray(state)))
            residual = np.max(np.abs(image - state)) / max(
                np.max(np.abs(image)), 1.0e-30
            )
            trace[index] = residual
            state = state + step * (image - state)
            if residual < tolerance:
                break
        flux = jnp.asarray(state)
        return self._receipt(
            flux,
            self._host_history(trace, flux, current, target_current),
            target_current=target_current,
        )

    def _solve_host_krylov(
        self, initial_flux, current, target_current=None, **options
    ) -> ForwardEquilibrium:
        """Drive the map with a host Jacobian-free Newton-Krylov root find.

        The Krylov step is globalised only by its own line search, so it
        moves freely between the branches of the free-boundary map and needs
        a seed already on the intended one.
        """
        mapped = self.flux_map(current, target_current=target_current)
        trace = np.full(self.evaluations, np.nan)
        recorded = 0
        initial = np.asarray(initial_flux, dtype=np.float64)
        initial_image = np.asarray(mapped(jnp.asarray(initial)))
        rejection_scale = 1.0e6 * max(float(np.max(np.abs(initial_image))), 1.0)

        def residual(psi):
            """Return the host free-boundary residual of a trial flux."""
            try:
                return np.asarray(mapped(jnp.asarray(psi))) - psi
            except CurrentNormalisationError:
                return np.full_like(np.asarray(psi, dtype=np.float64), rejection_scale)

        def record(psi, value):
            """Record the relative residual of one accepted host step."""
            nonlocal recorded
            if recorded < trace.size:
                total = np.max(np.abs(psi + value))
                trace[recorded] = np.max(np.abs(value)) / max(total, 1.0e-30)
                recorded += 1

        solution = scipy.optimize.newton_krylov(
            residual,
            initial,
            callback=record,
            **options,
        )
        flux = jnp.asarray(solution)
        return self._receipt(
            flux,
            self._host_history(trace, flux, current, target_current),
            target_current=target_current,
        )

    def _solve_accelerated(
        self,
        route: str,
        initial_flux,
        current,
        requested_class=None,
        target_current=None,
        **options,
    ) -> ForwardEquilibrium:
        """Drive the map with the shared fixed-point ladder."""
        mapped = self.flux_map(current, requested_class, target_current)
        if route == "newton_krylov":
            history = fixed_point.newton_krylov(
                mapped,
                initial_flux,
                **{"newton_steps": self.newton_steps, **options},
            )
        else:
            scheme = fixed_point.picard if route == "picard" else fixed_point.anderson
            history = scheme(
                mapped,
                initial_flux,
                **{
                    "evaluations": self.evaluations,
                    "relaxation": self.relaxation,
                    **options,
                },
            )
        return self._receipt(history.state, history, requested_class, target_current)

    def solve(
        self,
        initial_flux,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        target_current=None,
        enforce: Sequence[str] = (),
        pins: ConstraintPinSet | None = None,
        **options,
    ) -> ForwardEquilibrium:
        """Return the equilibrium the prescribed source supports.

        The default route is the root find because the map does not contract
        at the states this class is usually asked for. An elongated column
        held at fixed conductor currents is axisymmetrically unstable, and the
        write-then-read cycle carries that as a Jacobian eigenvalue outside
        the unit circle — 1.25 to 1.40 measured on a diverted reference case.
        A step relaxed by ``beta`` scales that mode by
        ``(1 - beta) + beta * lambda``, which exceeds one for every ``beta``
        in ``(0, 1]`` once ``lambda`` does, so no damping rescues a relaxed
        route; ``newton_krylov`` solves ``(I - J) s = f`` instead and is
        indifferent to the sign of that eigenvalue. ``picard`` and
        ``anderson`` stay available by name — they are the cheaper choice on a
        map that does contract, which is the limited, low-elongation case, and
        they are the only routes that expose the map iteration itself to a
        caller measuring how it behaves.

        ``enforce`` names integral observations the caller wants closed by
        the solve. The declared source has to carry one scalar degree of
        freedom per enforced moment; an absolute source carries none, so any
        request fails here, before a single profile value is read.

        ``target_current`` is a declared plasma current [A]. When present, one
        common source amplitude is eliminated from the exact clipped current
        moments inside every map evaluation and published in the result's
        normalisation record. Omitting it retains the absolute map.
        """
        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route == "host":
            equilibrium = self._solve_host(
                initial_flux, current, target_current=target_current, **options
            )
        elif route == "host_krylov":
            equilibrium = self._solve_host_krylov(
                initial_flux, current, target_current=target_current, **options
            )
        elif route not in _ACCELERATED:
            raise ValueError(
                f"unknown solve route {route!r}; available: "
                f"{', '.join((*_HOST, *_ACCELERATED))}"
            )
        else:
            equilibrium = self._solve_accelerated(
                route,
                initial_flux,
                current,
                target_current=target_current,
                **options,
            )
        self._require_constraints(equilibrium.flux, pins, target_current)
        return equilibrium

    def _iteration_count(self, route: str, options: dict[str, object]) -> int:
        """Return the fixed number of nonlinear state updates a route performs."""

        if route == "newton_krylov":
            return int(options.get("warmup", 8)) + int(
                options.get("newton_steps", self.newton_steps)
            )
        return int(options.get("evaluations", self.evaluations))

    def _branch_receipt(
        self,
        initial_flux,
        requested_class,
        current,
        target_current=None,
        *,
        route: str,
        tolerance: float,
        iterations: int,
        pins: ConstraintPinSet | None = None,
        **options,
    ) -> ForwardBranchReceipt:
        """Solve one pinned branch and qualify it against an emergent terminal read."""

        equilibrium = self._solve_accelerated(
            route,
            initial_flux,
            current,
            requested_class=requested_class,
            target_current=target_current,
            **options,
        )
        _masks, achieved = self.operator.read(equilibrium.flux)
        requested = jnp.asarray(requested_class, dtype=jnp.int8)
        achieved_class = jnp.asarray(achieved.diverted, dtype=jnp.int8)
        consistent = achieved_class == requested
        residual = equilibrium.fixed_point.residual
        converged = (
            jnp.isfinite(residual)
            & (residual <= tolerance)
            & consistent
            & equilibrium.finite.passed
        )
        if pins is not None:
            converged = converged & self.constraints_satisfied(
                equilibrium.flux, pins, target_current
            )
        return ForwardBranchReceipt(
            equilibrium=equilibrium,
            requested_class=requested,
            achieved_class=achieved_class,
            converged=converged,
            residual=residual,
            iterations=jnp.asarray(iterations, dtype=jnp.int32),
            topology_consistent=consistent,
        )

    def solve_branch(
        self,
        initial_flux,
        requested_class,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        target_current=None,
        enforce: Sequence[str] = (),
        pins: ConstraintPinSet | None = None,
        tolerance: float = 1.0e-10,
        **options,
    ) -> ForwardBranchReceipt:
        """Return one topology-pinned solve with an honest terminal receipt."""

        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route not in _ACCELERATED:
            raise ValueError(
                "a pinned branch needs a fixed-shape route; "
                f"available: {', '.join(_ACCELERATED)}"
            )
        return self._branch_receipt(
            initial_flux,
            requested_class,
            current,
            target_current,
            route=route,
            tolerance=tolerance,
            iterations=self._iteration_count(route, options),
            pins=pins,
            **options,
        )

    def solve_diverted_perturbations(
        self,
        reference_flux,
        perturbation_direction,
        policy: PerturbedSeedPolicy = PerturbedSeedPolicy(),
        *,
        current=None,
        target_current=None,
    ) -> ForwardPerturbedSeedReceipt:
        """Recover declared near-basin seeds through the pinned diverted branch.

        Amplitudes scale the production-read axis-to-saddle flux span. The
        supplied direction fixes the perturbation shape; normalising it only
        sets the largest pointwise displacement to the declared amplitude.
        Every rung is solved by the same pinned branch path used by portfolios.
        """
        reference = jnp.asarray(reference_flux)
        direction = jnp.asarray(perturbation_direction)
        if reference.shape != direction.shape:
            raise ValueError("reference flux and perturbation direction must align")
        requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        _masks, topology = self.operator.read(reference, requested)
        flux_span = jnp.abs(topology.flux_span)
        direction_scale = jnp.max(jnp.abs(direction))
        amplitudes = jnp.asarray(policy.relative_amplitudes, dtype=jnp.float64)
        normalized = direction / direction_scale
        seeds = reference + amplitudes[:, None] * flux_span * normalized
        iterations = policy.newton_steps
        rungs = jax.vmap(
            lambda seed: self._branch_receipt(
                seed,
                requested,
                current,
                target_current,
                route="newton_krylov",
                tolerance=policy.tolerance,
                iterations=iterations,
                warmup=0,
                newton_steps=policy.newton_steps,
                gmres_iterations=policy.gmres_iterations,
            )
        )(seeds)
        scale = jnp.maximum(
            jnp.max(jnp.abs(reference)), jnp.finfo(reference.dtype).tiny
        )
        root_error = (
            jnp.max(jnp.abs(rungs.equilibrium.flux - reference), axis=1) / scale
        )
        passed = rungs.converged & (root_error <= policy.tolerance)
        largest = jnp.max(jnp.where(passed, amplitudes, -jnp.inf))
        largest = jnp.where(jnp.any(passed), largest, jnp.nan)
        return ForwardPerturbedSeedReceipt(
            relative_amplitude=amplitudes,
            reference_flux_span=flux_span,
            seed_flux=seeds,
            rungs=rungs,
            root_relative_error=root_error,
            passed=passed,
            largest_passing_amplitude=largest,
        )

    def solve_portfolio(
        self,
        initial_flux,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        target_current=None,
        enforce: Sequence[str] = (),
        pins: ConstraintPinSet | None = None,
        tolerance: float = 1.0e-10,
        **options,
    ) -> ForwardPortfolio:
        """Solve limited and diverted branches together on one fixed branch axis.

        ``initial_flux`` has shape ``(2, node)`` in limited, diverted order.
        The implementation maps the same pinned branch function over that
        leading axis, so an outer ``vmap`` can add shot, time or ensemble axes
        without introducing a second physics path.
        """

        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route not in _ACCELERATED:
            raise ValueError(
                "a topology portfolio needs a fixed-shape route; "
                f"available: {', '.join(_ACCELERATED)}"
            )
        initial_flux = jnp.asarray(initial_flux)
        if initial_flux.ndim != 2 or initial_flux.shape[0] != 2:
            raise ValueError(
                "portfolio initial_flux must have shape (2, node) in "
                "limited, diverted order"
            )
        requested = jnp.asarray(
            (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)),
            dtype=jnp.int8,
        )
        current_axis = None if current is None or jnp.ndim(current) == 1 else 0
        target_axis = (
            None if target_current is None or jnp.ndim(target_current) == 0 else 0
        )
        iterations = self._iteration_count(route, options)
        branches = jax.vmap(
            lambda flux, branch_class, conductor, target: self._branch_receipt(
                flux,
                branch_class,
                conductor,
                target,
                route=route,
                tolerance=tolerance,
                iterations=iterations,
                pins=pins,
                **options,
            ),
            in_axes=(0, 0, current_axis, target_axis),
        )(initial_flux, requested, current, target_current)
        return ForwardPortfolio(branches=branches)

    def solve_batch(
        self,
        initial_flux,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        target_current=None,
        enforce: Sequence[str] = (),
        **options,
    ) -> ForwardEquilibrium:
        """Return :meth:`solve` mapped over a leading ensemble axis.

        The trial flux always carries the ensemble axis; the conductor
        currents carry it only when a batched array is supplied, so one
        machine state can be shared across an ensemble of seeds.
        """
        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route not in _ACCELERATED:
            raise ValueError(
                "a batched ensemble solve needs a fixed-shape route; "
                f"available: {', '.join(_ACCELERATED)}"
            )
        current_axis = None if current is None or jnp.ndim(current) == 1 else 0
        target_axis = (
            None if target_current is None or jnp.ndim(target_current) == 0 else 0
        )
        return jax.vmap(
            lambda flux, conductor, target: self._solve_accelerated(
                route, flux, conductor, target_current=target, **options
            ),
            in_axes=(0, current_axis, target_axis),
        )(initial_flux, current, target_current)

    def moment_residual(self, flux, targets: MomentTargets) -> jax.Array:
        """Return the scale-normalised integral-observation residuals."""
        return moment_residual(self.integral_observation(flux), targets)

    def moment_jacobian(self, flux, targets: MomentTargets) -> jax.Array:
        """Return the derivative of the moment residuals with respect to flux.

        The map is differentiated at a converged flux, so the Jacobian is the
        observation operator a conditioning or reconstruction caller needs;
        this class never applies it to a profile itself.
        """
        return jax.jacfwd(lambda state: self.moment_residual(state, targets))(
            jnp.asarray(flux)
        )
