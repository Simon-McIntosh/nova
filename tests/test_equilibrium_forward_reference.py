r"""Forward equilibrium on the production hex plasma mesh against a reference case.

The solve contract is pinned elsewhere on a synthetic rectangular lattice whose
coupling is assembled from point Green functions. This module drives the same
absolute-source free-boundary map on the mesh the package actually ships: the
hexagonal plasma grid a first-wall contour is tiled with, every cell coupled
through its own polygon by the closed-form section kernel, on real machine
geometry with the poloidal field coils a reference scenario equilibrium was
produced with.

Nothing here is fitted. The pressure and diamagnetic gradients are the
reference case's own tabulated profiles, read as absolute flux functions; the
conductor currents are the reference case's own, and every conductor the entry
declares carries one — the twelve wound poloidal field packs, the two skewed
in-vessel stabilisation plates, and the hundred and three passive loops of the
vessel and the structure inside it. The solve is then asked to reproduce the
reference flux map, boundary, plasma current, poloidal beta and internal
inductance from those inputs alone, and the deviations are the measurement.

Two conventions have to be crossed to do that, and both are pinned by
construction rather than by a fitted factor:

flux sense
    The stored equilibrium carries poloidal flux with the opposite sense to the
    package's total poloidal flux, so ``Phi = -psi`` node for node. That single
    negation also fixes the source: the package writes its gradients against
    the NEGATED total flux, so the tabulated ``dp/dpsi`` and ``F dF/dpsi`` reach
    :class:`~nova.equilibrium.source.DomainProfile` unchanged and unscaled, and
    the axis sits at a flux MINIMUM, which is the negative polarity the
    topology read is given.

total flux
    Both sides carry the total poloidal flux in Wb rather than flux per radian,
    so no factor of :math:`2 \pi` is applied anywhere. The check that this is
    the right reading is arithmetic, not assumed: integrating the current
    density the two tabulated gradients drive over the cells inside the stored
    boundary returns the stored plasma current, and that identity is asserted
    below before any solve runs.

The mesh is not a raster. Cells are hexagons on a half-offset lattice, trimmed
where the first wall cuts them, so a wall-clipped cell carries a smaller area
and a displaced centroid; the six-neighbour rings the grid solve tessellates
are what the null search reads. Nothing is adapted to that here.
:class:`~nova.equilibrium.forward.ForwardProfile` is constructed on the
production mesh through
:class:`~nova.equilibrium.stencil_mesh.StencilMesh` and its ``solve`` is
called unmodified, so the source state, the free-boundary map, the shared
fixed-point ladder, the domain partition, the integral observations AND the
published conservation, continuation, rotation and normalisation receipts are
all the public path on the shipped mesh.

The same solve is also driven one level down, through
:class:`~nova.equilibrium.forward_operator.ForwardFluxOperator` and the
observation operators directly, and the two are required to return the same
flux map. That cross-check is what separates a receipt layer that reads the
solve from one that changes it.

The internal inductance is integrated twice over, from two poloidal fields
that share no arithmetic. The operator path takes the field straight from the
grid solve that produced the flux coupling, so it is the analytic field of the
cell polygons; the published receipt differences the solved map on the ring
fit instead. Neither is calibrated against the other and they agree to one
percent, which is what turns the internal inductance into a measurement rather
than a restatement of the source.

A few results shaped the rest of the module and are worth stating up front.

The passive structure is in the machine, and is not the deviation floor.
    The entry declares a hundred and three passive loops carrying 14.9 kA
    between them at this slice, and every one of them reaches the coupling.
    Reaching it takes reading a section the machine descriptions here do not
    otherwise use: all but one of the loops declare a SKEWED cross-section, as
    does the in-vessel stabilisation pair among the driven coils, so a reader
    that resolves only rectangles finds no conductor at all in either family
    and silently models a different machine. Both are read through the
    parallelogram the entry itself declares.

    What that is worth is measured twice over, on the reference-native suite
    mesh. Before any
    solve, the flux the passive currents put on the plasma cells is a
    matrix-vector product, and it comes to 0.093 percent of the axis to
    boundary span. Through the coupled solve that direct input moves the
    reproduction score by 0.638 percentage points, four times the registered
    0.15-point cross-source ceiling. The direct field still meets its own
    budget; noncontractive free-boundary feedback supplies the excess. The
    closure assertion therefore remains red under the map-gain chain rather
    than treating the amplified output as a larger passive-current budget. The
    stabilisation pair, at 65 A against the structure's 14.9 kA, is worth 0.002
    percent of the span — a fiftieth of the structure and four decimal places
    below the deviation.

    The pair is worth stating exactly, because 65 A is a circuit current and
    not an ampere-turn count. The entry declares it as ONE coil carrying one
    current, with two elements at turn counts +1 and -1: the magnitude travels
    with the coil and the opposite signs are the whole of the entry's statement
    that the two branches are wired in anti-series, since it carries no circuit
    or supply description to say so a second time. One turn per branch makes
    the ampere-turns numerically the same 65 as the current. Read instead as
    the four turns per branch a wound in-vessel pair would carry, the drive and
    everything linear in it — including the flux the pair puts on the plasma
    cells — scales by four, to 0.008 percent of the span. The pair is
    negligible under either reading, which is why the module bounds it at four
    turns rather than settling which the entry meant.

    The gain defect does not disappear under refinement. The same closure moves
    the score by 0.610, 0.639 and 0.638 points at 566, 1587 and 2214 cells,
    respectively. That mesh-independent response rules out coarse resolution
    as its cause and assigns the discrepancy to the coupled-map feedback. It
    does not change the physical reading of the entry: every declared passive
    current remains an input at its authored value.

The route is a root find because the map does not contract.
    An elongated diverted column held at fixed conductor currents is
    axisymmetrically unstable, and that shows up here as an eigenvalue of the
    free-boundary map outside the unit circle: power iteration on the exact
    tangent at the converged equilibrium measures 1.25 and 1.40 on the banked
    coarse and intermediate carriers. A step relaxed by :math:`\beta` scales
    that mode by :math:`(1-\beta) + \beta \lambda`, so no damping rescues a
    relaxed Picard route. Seeded ON the converged state Picard drives the
    residual back up,
    while history-dependent Anderson mixing is required only to remain finite
    over the same fixed budget. Seeded on the stored map the Picard trace may
    contract and burst before the column reaches the wall and the source,
    which drives current only on an axis-connected core, switches itself off.
    The Newton step solves ``(I - J) s = f`` and is indifferent to the sign of
    that eigenvalue. Both branches are pinned below, because the collapsed one
    converges to a BETTER residual than the physical one, and a solve of a real
    equilibrium therefore cannot be qualified by its residual alone.

    What the large-amplitude drift does is vertical, but the dominant
    eigenvector of the LINEARISED map is not simply a rigid vertical shift:
    displacing the state along it moves the magnetic axis about twice as far in
    major radius as in height. The linear statement pinned below is therefore
    non-contraction, which is what selects the route, and not the
    identification of a vertical eigenmode.

    The eager Krylov route agrees where it can run. Driven through
    ``host_krylov`` on the same seed, :func:`scipy.optimize.newton_krylov`
    reaches the same equilibrium — the same axis to four decimals, and a flux
    map within 3.9e-07 Wb of the ladder route, five parts in a billion of the
    span. Under a single-precision axis fit it instead raises
    ``NoConvergence``, at a tolerance derived from the flux ladder and at one
    two decades looser alike: what defeated it there was the noise the fit put
    under its residual, not its globalisation.

No solve resolves flux below what its own arithmetic can express.
    Two floors compete and the coarser one binds. One is a step of the ladder
    the fitted axis flux lands on, since every profile is evaluated on a
    normalised flux formed against that value; the other is the round-off the
    plasma coupling accumulates over a dot product across the cells. Which one
    binds follows the precision the null search fits at, so this module reads
    the dtype off the fit rather than naming one — see :func:`flux_resolution`
    — and the difference is not academic. On this case the axis flux is about
    83 Wb: fitted in single precision its ladder step is 7.6e-06 Wb, decades
    above the coupling sum and binding alone, and the root find stalls against
    it four decades down. Fitted in double, one step falls to a single unit in
    the last place, the coupling sum binds instead at about 2e-11 Wb, and the
    same budget runs six decades further. This floor is a lower resolvability
    limit, not a promise that a fixed-budget Krylov route terminates within a
    fixed multiple of it. Production acceptance is therefore the registered
    normalised residual plus finite, diverted physical qualification; the
    residual-to-resolution ratio is logged as a diagnostic. A tolerance
    carried across from another case or precision would be unreachable or
    meaningless, which is why nothing here names an absolute one.

The published shape moments are not the reference's own.
    Recomputing poloidal beta and internal inductance from the stored flux map
    — the volume it encloses, the stored pressure over that volume, the field
    its own gradient carries — does not return the published ``beta_pol`` and
    ``li_3``, though the enclosed volume agrees to a fraction of a percent.
    That quadrature runs on a raster and touches no package code, so the
    disagreement is internal to the entry. The comparison below is therefore
    made against the recomputed pair; comparing against the published scalars
    would charge the solve for a definition it never used.

Running this file as ``python -m tests.test_equilibrium_forward_reference
figures`` from the repository root writes the evidence figures instead of
running the suite.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.constants import mu_0
import xarray

from nova.database.zarrstore import ZarrStore

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium import fixed_point
    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.observation import current_ledger, observe_moments
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.stencil_mesh import (
        MomentGeometry,
        RING_CONDITION_LIMIT,
        StencilMesh,
        ring_condition,
    )
    from nova.frame.coilset import CoilSet
    from nova.jax.config import configure_dtypes

with skip_import("imas"):
    import imas

    from nova.imas.machine import Oblique

#: Stored equilibrium the demonstration reproduces. The scenario pulse carries
#: its own machine description — the conductor sections and turn counts of both
#: the driven coils and the passive structure, the first wall contour, and every
#: current waveform on one time base — so geometry, drive and reference come
#: from one entry and cannot disagree about the machine. The slice is a flat-top
#: diverted burn point.
PULSE, RUN, MACHINE = 135011, 7, "iter"
DD_VERSION = "3.39.0"
TIME_SLICE = 353

#: Users the pulse store is searched under, in order. The package resolves
#: ``public`` against the shared store and any other name against that user's
#: own, so trying both covers a site store and a personal one.
CANDIDATE_USERS = ("public", getpass.getuser())

#: Target cell counts. The plasma grid is generated from a negative cell-count
#: request, so the realised count follows the wall area and the hexagon pitch:
#: -500 gives 566 cells at a 0.237 m pitch, -1500 gives 1587 at 0.137 m,
#: and -2100 gives 2214 at 0.115 m. The suite and its default evidence figure
#: use the reference-native carrier; the other two remain banked intervention
#: rungs rather than competing defaults.
#: Refining does not reduce the deviation against the reference, it RAISES it —
#: the solved flux map misses the stored one by 1.50 % of the span at 566 cells
#: and 6.86 % at 1587 — which is what identifies the deviation as a
#: machine-model difference rather than a discretisation error. A
#: discretisation error would fall. What rises with it is the stored map's own
#: free-boundary residual under this machine model, 0.96 % to 1.35 %: the
#: finer mesh resolves more of a disagreement that was always there, and the
#: solve then slides further along the soft vertical and radial force balance
#: before it comes to rest.
SUITE_CELLS = -2100
EVIDENCE_CELLS = SUITE_CELLS
#: Poloidal field coil filaments per conductor, and wall nodes PER FIRST-WALL
#: SEGMENT the limiter flux is searched on — the boundary sampler distributes
#: the request per segment, not over the whole loop, so this multiplies the
#: contour's own vertex count.
COIL_FILAMENTS = -25
WALL_NODES = 3
#: Elements each passive loop is decomposed into. The request is the coarsest
#: the inserter takes, which tiles most plates into three or four pieces and a
#: quarter of them into one. What justifies the coarseness is not that a plate
#: is thin against the mesh — it is not, at 3 to 128 mm against a 137 to
#: 237 mm cell pitch — but that the whole passive family contributes a tenth
#: of a percent of the flux span at the plasma cells, so how its current is
#: distributed WITHIN a plate is a correction to a term that is already an
#: order below the deviation being chased. What the decomposition may not do
#: is change the footprint, and that is asserted: every element's tiles sum to
#: the area its own section declares.
PASSIVE_ELEMENTS = -1

#: Poloidal cross-section codes the entry's conductor elements declare, from
#: the data dictionary's ``geometry_type`` enumeration. This pulse uses only
#: these two and anything else is refused rather than approximated, because a
#: conductor dropped for want of a section is a silent change of machine.
RECTANGLE_SECTION, OBLIQUE_SECTION = 2, 3

#: Root-find budget. Each Newton step linearises the map once and solves
#: ``(I - J) s = f`` in a fixed-shape Krylov space, which is what lets it hold
#: an equilibrium the relaxed iteration cannot — see the vertical drift below.
#: The budget is generous on purpose. How far it gets is set by the arithmetic
#: rather than by the step count: the residual falls four decades over the
#: first four steps and then descends as far as the solve can express, which
#: is the axis-flux ladder under a single-precision fit and the coupling
#: round-off six decades below it under a double one.
NEWTON_STEPS = 10
KRYLOV_ITERATIONS = 30
#: Relaxed evaluations the vertical drift is measured over, and its step.
RELAXED_EVALUATIONS = 60
RELAXATION = 0.5

#: Raster the reference's own flux map is integrated on to recover the moments
#: its published scalars do not reproduce. The integrals are converged to five
#: digits by 200 radial nodes, so this is comfortable rather than marginal.
RASTER_NODES = 401

#: Pre-registered tolerances, all measured on the suite mesh.
#:
#: The plasma current is an area-weighted sum of a source the stored profiles
#: fix exactly, so its error is the error in WHICH cells the topology read
#: labels core — one boundary shell of a coarse hexagonal mesh.
RESIDUAL_TOLERANCE = 1.0e-6
PLASMA_CURRENT_TOLERANCE = 0.03
#: The axis is interpolated inside one cell by the same biquadratic fit the
#: null search uses, so it resolves well below the cell pitch.
AXIS_TOLERANCE = 0.06
#: Sup-norm flux deviation over the labelled core, read against the axis to
#: boundary flux span. The floor is not the mesh: the stored map is only a
#: fixed point of this machine model to about one percent of the span, and the
#: solve moves along the soft vertical/radial force balance until it is one.
FLUX_TOLERANCE = 0.10
#: The two shape moments, against the moments recomputed from the reference's
#: own flux map rather than against its published scalars.
MOMENT_TOLERANCE = 0.08
#: Midplane boundary radii, in cell pitches — the core label is a whole-cell
#: selection, so a boundary that lands one cell in or out is exact at this
#: resolution.
BOUNDARY_PITCHES = 1.5

#: Agreement between the area a conductor's tiles carry and the area its own
#: section declares. The floor is the tiler's arithmetic, not the model: the
#: worst element of the hundred and seventeen reproduces its declared area to
#: 3.6e-7 relative, and a conductor tiled into the wrong footprint would miss
#: by a percent or more.
AREA_TOLERANCE = 1.0e-5
#: Ceiling on the flux the passive structure alone puts on the plasma cells,
#: as a fraction of the axis to boundary span. The banked 566/1587/2214-cell
#: ladder measures 9.3e-4/9.6e-4/9.7e-4 — a tenth of a percent, and insensitive
#: to the mesh, because it is a property of the drive rather than of the solve.
PASSIVE_FLUX_CEILING = 2.0e-3
#: How far the stored map's own free-boundary residual has to exceed that
#: contribution for the passive structure to be excluded as the deviation
#: floor. The banked coarse and intermediate carriers measure 10.3 and 14.1,
#: so the structure is an order too small and grows no closer under refinement;
#: this pin keeps the claim at "an order" with room to spare.
PASSIVE_SHORTFALL = 5.0
#: Ceiling on the passive closure's change in flux-map reproduction deviation,
#: in percentage points of the reference flux span. The reference-native
#: carrier measures a 0.097-point direct response against a 0.638-point coupled
#: move, retaining the ceiling as a deliberately visible physics-owned row.
PASSIVE_REPRODUCTION_MOVE_CEILING = 0.15
#: Internal-inductance closure response measured on the evidence mesh, in
#: percentage points. Unlike flux-map reproduction, l_i responds strongly to
#: current redistribution: passives move it by +0.904734074 points and improve
#: its deviation to -0.135265926 percent. The suite-mesh response must retain
#: that improvement and remain between the flux-scale move and this banked one.
PASSIVE_INTERNAL_INDUCTANCE_MOVE = 0.904734074

#: The two elements the in-vessel stabilisation pair is declared as, labelled
#: the way a two-element coil is labelled below. They are the only driven
#: elements with a skewed section, so a rectangle-only reader loses the pair
#: along with the whole passive family.
STABILISATION_ELEMENTS = ("VS3U", "VS3L")
#: Ceiling on the flux the stabilisation pair alone puts on the plasma cells, as
#: a fraction of the axis to boundary span. The banked coarse and intermediate
#: carriers measure 2.0e-5 and 2.1e-5, a fiftieth of what the passive structure
#: carries.
STABILISATION_FLUX_CEILING = 1.0e-4
#: Turns per branch the pair is bounded at. The entry declares ONE, and the
#: bound is written for four so that it also covers a reading in which each
#: branch carries the four turns of a wound in-vessel pair rather than the
#: single turn declared. The flux a conductor puts on the cells is linear in
#: its ampere-turns, so the factor carries straight through and the pin holds
#: without having to settle which reading the entry meant.
STABILISATION_TURN_BOUND = 4.0

#: Agreement between the current the tabulated gradients imply on the stored
#: boundary and the stored plasma current. This is the convention pin, not a
#: solve result: it fails if the flux sense, the total-flux reading or the
#: gradient normalisation is wrong, and it is insensitive to the mesh only
#: because the integrand is smooth.
CONVENTION_TOLERANCE = 0.02
#: Agreement between the package's integral observation operator and an
#: independent quadrature of the same field on a raster.
QUADRATURE_TOLERANCE = 0.05

#: Power iterations used to read the dominant eigenvalue of the free-boundary
#: map at the converged equilibrium. The Rayleigh quotient is within a percent
#: of its limit by twenty; the count is fixed rather than run to a tolerance,
#: so a map whose spectrum does not separate costs the same and cannot hang.
POWER_ITERATIONS = 40
#: How far the dominant eigenvalue has to clear one for non-contraction to be
#: a measurement rather than a rounding artefact. The banked coarse and
#: intermediate carriers measure 1.249 and 1.401.
CONTRACTION_MARGIN = 1.05
#: Residual growth the raw Picard map must show over the bounded budget when
#: seeded on the equilibrium itself. History-dependent mixing is not required
#: to follow the same expanding direction.
DRIFT_GROWTH = 10.0

#: Agreement between the published solve and the same map driven one level
#: down through the operator. Both hand the same seed to the same ladder on
#: the same couplings, so the measured difference is exactly zero and this
#: tolerance would catch any receipt layer that moved the solution at all.
ROUTE_AGREEMENT = 1.0e-12
#: Grad-Shafranov residual the receipt reports on the suite mesh, read against
#: its own drive. It is a discretisation measure, not an equilibrium one: the
#: solved map is the coupling image of piecewise-uniform cell currents, so the
#: elliptic operator of a one-pitch ring recovers the drive only to the
#: truncation of that ring.
GRAD_SHAFRANOV_TOLERANCE = 0.08
#: Agreement between the internal inductance the receipt integrates from
#: the DIFFERENCED flux map and the one the analytic field of the cell
#: polygons gives. Measured at 1.0 % on the suite mesh.
FIELD_INTEGRAL_TOLERANCE = 0.03
#: The two identically-vanishing residuals, as a fraction of the
#: Grad-Shafranov one. On a ring fit they sit at the truncation floor of the
#: second derivative rather than at round-off, so what qualifies them is this
#: margin below the residual that carries the physics.
DIVERGENCE_MARGIN = 0.2

FIGURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "figures"
    / "flux-function-forward-equilibrium"
)


# --------------------------------------------------------------------------
# the stored reference
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Conductor:
    """One conductor element of the reference machine and its slice current.

    An axis-aligned element is carried as the four scalars the entry declares
    and a skewed one as the vertex loop its parallelogram resolves to, because
    the two take different insert paths and a parallelogram has no faithful
    ``(width, height)`` reading. Exactly one of the two is set.
    """

    name: str
    current: float
    turns: float
    rectangle: tuple[float, float, float, float] | None = None
    polygon: np.ndarray | None = None

    @property
    def placement(self) -> tuple:
        """Return the positional arguments the element's insert path takes."""
        if self.rectangle is not None:
            return self.rectangle
        return (self.polygon,)

    @property
    def declared_area(self) -> float:
        """Return the cross-section area [m^2] the entry declares for it."""
        if self.rectangle is not None:
            return self.rectangle[2] * self.rectangle[3]
        radius, height = self.polygon[:, 0], self.polygon[:, 1]
        return 0.5 * abs(
            np.dot(radius, np.roll(height, -1)) - np.dot(height, np.roll(radius, -1))
        )


def _oblique_polygon(geometry) -> np.ndarray:
    """Return the parallelogram vertices one oblique element declares.

    The reading is the package's own, :class:`nova.imas.machine.Oblique`, and
    two of its choices are worth stating because neither is guessable from the
    field names. ``(r, z)`` is a CORNER of the parallelogram rather than its
    centre, and the two angles are referred to DIFFERENT axes: ``alpha`` turns
    the first side off the major radius axis while ``beta`` turns the second
    off the height axis, so an axis-aligned plate appears as ``alpha = pi``
    with ``beta = 0`` rather than as a right angle between the two.

    Neither choice can be pinned from this entry, and the obvious geometric
    tests do not separate them: reading ``(r, z)`` as the centre instead
    displaces every plate by half its own diagonal — a median 0.28 m, which is
    of order the cell pitch — yet leaves the inventory equally clear of the
    first wall (no plate meets it either way) and no more self-overlapped
    (0.006 of 3.40 m^2 shared, both ways). The vessel plates are packed close
    enough that a shift of that size lands them back among their neighbours.

    What makes the ambiguity tolerable is therefore not that the two readings
    are distinguishable but that the answer does not turn on it. The passive
    structure's whole contribution to the flux at the plasma cells is a tenth
    of a percent of the flux span, an order below the deviation this module
    measures, so no rearrangement of it within its own footprint reaches the
    quantity being chased.
    """
    return np.asarray(
        Oblique(
            None,
            {
                name: float(getattr(geometry.oblique, name))
                for name in ("r", "z", "length_alpha", "length_beta", "alpha", "beta")
            },
        ).poly.exterior.coords
    )[:-1]


def _conductor(name, geometry, current, turns) -> Conductor | None:
    """Return one placed conductor element, or ``None`` if the entry places none."""
    section = int(geometry.geometry_type)
    if section == RECTANGLE_SECTION:
        rectangle = geometry.rectangle
        if float(rectangle.width) <= 0.0:
            return None
        return Conductor(
            name=name,
            current=current,
            turns=turns,
            rectangle=(
                float(rectangle.r),
                float(rectangle.z),
                float(rectangle.width),
                float(rectangle.height),
            ),
        )
    if section == OBLIQUE_SECTION:
        return Conductor(
            name=name,
            current=current,
            turns=turns,
            polygon=_oblique_polygon(geometry),
        )
    raise ValueError(f"conductor {name} declares unsupported section {section}")


#: ``eq=False`` keeps the identity hash: the fields are arrays, so a generated
#: ``__eq__``/``__hash__`` pair could not be evaluated, and the memoised
#: quadrature below needs the instance to be hashable.
@dataclass(frozen=True, eq=False)
class ReferenceCase:
    """One stored equilibrium slice and the machine it was produced on."""

    user: str
    time: float
    plasma_current: float
    poloidal_beta: float
    internal_inductance: float
    axis: np.ndarray
    flux_axis: float
    flux_boundary: float
    reference_radius: float
    psi_norm: np.ndarray
    p_prime: np.ndarray
    ff_prime: np.ndarray
    pressure: np.ndarray
    field_function: np.ndarray
    safety_factor: np.ndarray
    boundary: np.ndarray
    separatrix: np.ndarray
    x_point: np.ndarray
    wall: np.ndarray
    active: tuple[Conductor, ...]
    passive: tuple[Conductor, ...]
    unplaced: tuple[tuple[str, float], ...]
    grid_radius: np.ndarray
    grid_height: np.ndarray
    grid_flux: np.ndarray

    def drive(self, passive: bool = True) -> tuple[Conductor, ...]:
        """Return the conductor elements one machine model carries.

        The driven coils are always present. The passive structure is a model
        choice, kept switchable because whether it belongs in the machine is
        the question the deviation table below answers.
        """
        return self.active + (self.passive if passive else ())

    @property
    def flux_span(self) -> float:
        """Return the axis to boundary total poloidal flux [Wb]."""
        return self.flux_boundary - self.flux_axis

    @property
    def spline(self):
        """Return the interpolant of the stored two-dimensional flux map."""
        from scipy.interpolate import RectBivariateSpline

        return RectBivariateSpline(self.grid_radius, self.grid_height, self.grid_flux)

    def flux(self, radius, height) -> np.ndarray:
        """Return the stored total poloidal flux [Wb] at any point."""
        return -self.spline.ev(np.asarray(radius), np.asarray(height))

    @lru_cache(maxsize=1)  # noqa: B019 - one frozen case per session
    def map_moments(self) -> dict[str, float]:
        """Return the shape moments the stored FLUX MAP implies.

        The published ``beta_pol`` and ``li_3`` scalars are not reproducible
        from the rest of the entry, so they cannot be the yardstick. This
        integrates the stored two-dimensional map itself — the enclosed volume,
        the stored pressure profile over that volume, and the poloidal field
        the map's own gradient carries — inside the stored boundary contour,
        with no package code involved. It is the same definition
        :func:`~nova.equilibrium.observation.observe_moments` publishes, so the
        two are directly comparable.
        """
        from matplotlib.path import Path

        radius = np.linspace(
            self.boundary[:, 0].min() - 0.05,
            self.boundary[:, 0].max() + 0.05,
            RASTER_NODES,
        )
        height = np.linspace(
            self.boundary[:, 1].min() - 0.05,
            self.boundary[:, 1].max() + 0.05,
            int(1.8 * RASTER_NODES),
        )
        grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
        cell = (radius[1] - radius[0]) * (height[1] - height[0])
        inside = (
            Path(self.boundary)
            .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
            .reshape(grid_r.shape)
        )
        volume_element = np.where(inside, 2.0 * np.pi * grid_r * cell, 0.0)
        psi_norm = (-self.spline.ev(grid_r, grid_z) - self.flux_axis) / self.flux_span
        pressure = np.interp(np.clip(psi_norm, 0.0, 1.0), self.psi_norm, self.pressure)
        field_squared = (
            self.spline.ev(grid_r, grid_z, dx=1) ** 2
            + self.spline.ev(grid_r, grid_z, dy=1) ** 2
        ) / (2.0 * np.pi * grid_r) ** 2
        pressure_integral = float(np.sum(pressure * volume_element))
        field_integral = float(np.sum(field_squared * volume_element))
        reference = mu_0 * self.reference_radius * self.plasma_current**2
        return {
            "volume": float(volume_element.sum()),
            "pressure_integral": pressure_integral,
            "field_integral": field_integral,
            "poloidal_beta": 4.0 * pressure_integral / reference,
            "internal_inductance": 2.0 * field_integral / (mu_0 * reference),
        }


def _uri(user: str) -> str:
    """Return the entry locator one candidate user resolves to."""
    return (
        f"imas:hdf5?user={user};pulse={PULSE};run={RUN};"
        f"database={MACHINE};version={DD_VERSION.split('.')[0]}"
    )


def _read_active(coils) -> tuple[list[Conductor], list[tuple[str, float]]]:
    """Return the driven conductor elements and any the entry does not place.

    A coil wound in two packs declares one current and two elements, and the
    element carries the turn count that current flows through. The sign of that
    count is the wiring: it travels onto the element's current here, while the
    magnitude stays a turn count, because the insert takes a signed current and
    a positive number of turns. The central solenoid's split pack declares both
    of its elements at +554, the same sense; the stabilisation pair declares +1
    and -1, opposed.

    That sign is the only place the wiring appears. This entry populates neither
    ``circuit`` nor ``supply``, so there is no connection matrix to read the
    anti-series pair from and no second declaration to check the turn signs
    against.
    """
    placed, unplaced = [], []
    for coil in coils.coil:
        current = float(np.asarray(coil.current.data)[TIME_SLICE])
        found = False
        for position, element in enumerate(coil.element):
            turns = float(element.turns_with_sign)
            label = str(coil.name)
            if len(coil.element) > 1:
                label = f"{label}{'UL'[position]}"
            conductor = _conductor(
                label, element.geometry, current * np.sign(turns), abs(turns)
            )
            if conductor is None:
                continue
            placed.append(conductor)
            found = True
        if not found:
            unplaced.append((str(coil.name), current))
    return placed, unplaced


def _read_passive(loops) -> list[Conductor]:
    """Return the passive structure elements and the currents they carry.

    Every loop in this entry carries exactly one element, so the loop current
    is the element current and no distribution over a pack is implied.
    """
    placed = []
    for index in range(len(loops.loop)):
        loop = loops.loop[index]
        current = float(np.asarray(loop.current)[TIME_SLICE])
        for element in loop.element:
            turns = float(element.turns_with_sign)
            conductor = _conductor(
                str(loop.name), element.geometry, current * turns, 1.0
            )
            if conductor is not None:
                placed.append(conductor)
    return placed


def _read_reference(user: str) -> ReferenceCase:
    """Return the stored slice, its profiles and its machine description."""
    entry = imas.DBEntry(_uri(user), "r", dd_version=DD_VERSION)
    try:
        equilibrium = entry.get("equilibrium")
        coils = entry.get("pf_active")
        loops = entry.get("pf_passive")
        wall = entry.get("wall")
    finally:
        entry.close()

    slice_ = equilibrium.time_slice[TIME_SLICE]
    globals_ = slice_.global_quantities
    profiles = slice_.profiles_1d
    flux_1d = np.asarray(profiles.psi)
    separatrix = slice_.boundary_separatrix
    surface = slice_.profiles_2d[0]
    active, unplaced = _read_active(coils)

    return ReferenceCase(
        user=user,
        time=float(np.asarray(equilibrium.time)[TIME_SLICE]),
        plasma_current=float(globals_.ip),
        poloidal_beta=float(globals_.beta_pol),
        internal_inductance=float(globals_.li_3),
        axis=np.array(
            [float(globals_.magnetic_axis.r), float(globals_.magnetic_axis.z)]
        ),
        # the stored flux runs opposite to the package's total poloidal flux
        flux_axis=-float(globals_.psi_axis),
        flux_boundary=-float(globals_.psi_boundary),
        reference_radius=float(equilibrium.vacuum_toroidal_field.r0),
        psi_norm=(flux_1d - flux_1d[0]) / (flux_1d[-1] - flux_1d[0]),
        p_prime=np.asarray(profiles.dpressure_dpsi),
        ff_prime=np.asarray(profiles.f_df_dpsi),
        pressure=np.asarray(profiles.pressure),
        field_function=np.asarray(profiles.f),
        safety_factor=np.asarray(profiles.q),
        boundary=np.c_[
            np.asarray(slice_.boundary.outline.r), np.asarray(slice_.boundary.outline.z)
        ],
        separatrix=np.c_[
            np.asarray(separatrix.outline.r), np.asarray(separatrix.outline.z)
        ],
        x_point=np.array(
            [[float(point.r), float(point.z)] for point in separatrix.x_point]
        ),
        wall=np.c_[
            np.asarray(wall.description_2d[0].limiter.unit[0].outline.r),
            np.asarray(wall.description_2d[0].limiter.unit[0].outline.z),
        ],
        active=tuple(active),
        passive=tuple(_read_passive(loops)),
        unplaced=tuple(unplaced),
        grid_radius=np.asarray(surface.grid.dim1),
        grid_height=np.asarray(surface.grid.dim2),
        grid_flux=np.asarray(surface.psi),
    )


@lru_cache(maxsize=1)
def reference_case() -> ReferenceCase | str:
    """Return the stored reference, or why it could not be reached."""
    reasons = []
    for user in CANDIDATE_USERS:
        try:
            return _read_reference(user)
        except Exception as error:  # noqa: BLE001 - any failure is unavailability
            reasons.append(f"{user}: {type(error).__name__}")
    return f"pulse {PULSE}/{RUN} unreachable ({'; '.join(reasons)})"


def require_reference() -> ReferenceCase:
    """Return the stored reference or skip the calling test."""
    case = reference_case()
    if isinstance(case, str):
        pytest.skip(case)
    return case


# --------------------------------------------------------------------------
# the machine and its hexagonal plasma mesh
# --------------------------------------------------------------------------
@dataclass
class HexMachine:
    """Machine geometry, its hexagonal plasma mesh and their couplings."""

    coilset: CoilSet | None = field(repr=False)
    source_current: np.ndarray
    passive_columns: int
    node: np.ndarray
    area: np.ndarray
    cell_polygons: tuple[np.ndarray, ...]
    hexagon: np.ndarray
    stencil: np.ndarray
    wall_node: np.ndarray
    source_to_grid: np.ndarray
    plasma_to_grid: np.ndarray
    plasma_to_grid_r: np.ndarray
    plasma_to_grid_z: np.ndarray
    sampling_vertices: np.ndarray
    sample_coordinates: np.ndarray
    source_to_sample: np.ndarray
    plasma_to_sample: np.ndarray
    plasma_to_sample_r: np.ndarray
    plasma_to_sample_z: np.ndarray
    source_to_wall: np.ndarray
    plasma_to_wall: np.ndarray
    plasma_to_wall_r: np.ndarray
    plasma_to_wall_z: np.ndarray
    radial_field: tuple[np.ndarray, ...]
    vertical_field: tuple[np.ndarray, ...]
    cache_receipt: MachineCacheReceipt | None = field(default=None, repr=False)

    @cached_property
    def moment_geometry(self) -> MomentGeometry:
        """Build the fixed current-moment geometry once for this machine."""
        return MomentGeometry.from_cells(
            receipt_mesh(self),
            self.cell_polygons,
            sampling_vertices=self.sampling_vertices,
        )

    @property
    def passive_flux(self) -> np.ndarray:
        """Return the flux [Wb] the passive structure alone puts on the cells."""
        if self.passive_columns == 0:
            return np.zeros(len(self.node))
        return (
            self.source_to_grid[:, -self.passive_columns :]
            @ self.source_current[-self.passive_columns :]
        )

    @property
    def radius(self) -> np.ndarray:
        """Return the major radius [m] of every plasma cell centroid."""
        return self.node[:, 0]

    @property
    def interior_stencil(self) -> np.ndarray:
        """Return the neighbour rings that are whole hexagonal neighbourhoods.

        The tessellation recovers six-neighbour rings from a triangulation of
        the cell centroids, which also produces a ring for a cell the first
        wall has clipped. That ring is not a hexagonal neighbourhood — the
        centroid has been pulled inside the cut and its neighbours are no
        longer at equal spacing — so a quadratic fitted on it can report a
        stationary point that the flux map does not have. Keeping only the
        rings whose centre and all six neighbours are uncut leaves the null
        search on the regular part of the lattice.
        """
        return self.stencil[self.hexagon[self.stencil].all(axis=1)]

    def poloidal_field_squared(self, source_current, current_moments) -> jnp.ndarray:
        """Return the squared poloidal field [T^2] the two sources produce."""
        radial = jnp.asarray(self.radial_field[0]) @ source_current + sum(
            jnp.asarray(block) @ moment
            for block, moment in zip(
                self.radial_field[1:], current_moments, strict=True
            )
        )
        vertical = jnp.asarray(self.vertical_field[0]) @ source_current + sum(
            jnp.asarray(block) @ moment
            for block, moment in zip(
                self.vertical_field[1:], current_moments, strict=True
            )
        )
        return radial**2 + vertical**2


@dataclass(frozen=True)
class MachineCacheReceipt:
    """Describe one persistent fixture-machine cache request."""

    store: str
    key: str
    hit: bool
    lock_wait_seconds: float
    load_seconds: float
    build_seconds: float
    store_seconds: float
    validation_seconds: float
    arrays_verified: int
    bytes_verified: int
    bitwise_stored_precision: bool


MACHINE_CACHE_SCHEMA = "hex-machine-native-array-carrier"
MACHINE_CACHE_FILENAME = "solovev_hex_machine"
_MACHINE_ARRAY_FIELDS = (
    "source_current",
    "node",
    "area",
    "hexagon",
    "stencil",
    "wall_node",
    "source_to_grid",
    "plasma_to_grid",
    "plasma_to_grid_r",
    "plasma_to_grid_z",
    "sampling_vertices",
    "sample_coordinates",
    "source_to_sample",
    "plasma_to_sample",
    "plasma_to_sample_r",
    "plasma_to_sample_z",
    "source_to_wall",
    "plasma_to_wall",
    "plasma_to_wall_r",
    "plasma_to_wall_z",
)
_REFERENCE_ARRAY_FIELDS = (
    "axis",
    "psi_norm",
    "p_prime",
    "ff_prime",
    "pressure",
    "field_function",
    "safety_factor",
    "boundary",
    "separatrix",
    "x_point",
    "wall",
    "grid_radius",
    "grid_height",
    "grid_flux",
)
_REFERENCE_SCALAR_FIELDS = (
    "time",
    "plasma_current",
    "poloidal_beta",
    "internal_inductance",
    "flux_axis",
    "flux_boundary",
    "reference_radius",
)


def _array_identity(value) -> dict[str, object]:
    """Return shape, dtype and semantic content hash for one numerical array."""
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _conductor_identity(conductor: Conductor) -> dict[str, object]:
    """Return the placed conductor content that determines one coupling column."""
    return {
        "name": conductor.name,
        "current": float(conductor.current),
        "turns": float(conductor.turns),
        "rectangle": conductor.rectangle,
        "polygon": (
            None if conductor.polygon is None else _array_identity(conductor.polygon)
        ),
    }


def machine_cache_identity(
    case: ReferenceCase, cells: int, *, passive: bool = True
) -> dict[str, object]:
    """Return the complete semantic identity of one fixture operator carrier."""
    configuration = CoilSet(
        dcoil=COIL_FILAMENTS,
        dplasma=cells,
        tplasma="hex",
        nwall=WALL_NODES,
    )
    precision = getattr(configuration.precision, "value", configuration.precision)
    reference = {name: float(getattr(case, name)) for name in _REFERENCE_SCALAR_FIELDS}
    reference["arrays"] = {
        name: _array_identity(getattr(case, name)) for name in _REFERENCE_ARRAY_FIELDS
    }
    reference["active"] = [_conductor_identity(conductor) for conductor in case.active]
    reference["passive"] = [
        _conductor_identity(conductor) for conductor in case.passive
    ]
    reference["unplaced"] = [list(item) for item in case.unplaced]
    return {
        "schema": MACHINE_CACHE_SCHEMA,
        "reference_locator": {
            "pulse": PULSE,
            "run": RUN,
            "dd_version": DD_VERSION,
            "time_slice": TIME_SLICE,
        },
        "reference_content": reference,
        "discretisation": {
            "cells": int(cells),
            "coil_filaments": COIL_FILAMENTS,
            "wall_nodes": WALL_NODES,
            "passive_elements": PASSIVE_ELEMENTS,
            "passive": bool(passive),
            "plasma_shape": configuration.tplasma,
        },
        "precision": str(precision),
        "routes": configuration.route_attrs,
    }


def _packed_machine_arrays(machine: HexMachine) -> dict[str, np.ndarray]:
    """Return every numerical input needed to reconstruct a fixture machine."""
    arrays = {
        name: np.array(getattr(machine, name), copy=True)
        for name in _MACHINE_ARRAY_FIELDS
    }
    for position, array in enumerate(machine.radial_field):
        arrays[f"radial_field_{position}"] = np.array(array, copy=True)
    for position, array in enumerate(machine.vertical_field):
        arrays[f"vertical_field_{position}"] = np.array(array, copy=True)
    offsets = np.zeros(len(machine.cell_polygons) + 1, dtype=np.int64)
    for position, polygon in enumerate(machine.cell_polygons):
        offsets[position + 1] = offsets[position] + len(polygon)
    arrays["cell_polygon_offsets"] = offsets
    arrays["cell_polygon_vertices"] = np.concatenate(machine.cell_polygons, axis=0)
    arrays["passive_columns"] = np.asarray(machine.passive_columns, dtype=np.int64)
    for name, array in arrays.items():
        if array.dtype.kind == "O":
            raise TypeError(f"fixture cache array {name} has object dtype")
    return arrays


def _payload_digest(arrays: dict[str, np.ndarray]) -> str:
    """Return an ordered digest of every stored array including dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _machine_dataset(
    machine: HexMachine, identity: dict[str, object], key: str
) -> xarray.Dataset:
    """Return a native-dtype Zarr payload for one fixture machine."""
    arrays = _packed_machine_arrays(machine)
    variables = {
        name: (tuple(f"{name}_axis_{axis}" for axis in range(array.ndim)), array)
        for name, array in arrays.items()
    }
    return xarray.Dataset(
        variables,
        attrs={
            "cache_schema": MACHINE_CACHE_SCHEMA,
            "cache_key": key,
            "semantic_identity": json.dumps(
                identity, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            "payload_digest": _payload_digest(arrays),
        },
    )


def _machine_from_dataset(
    data: xarray.Dataset, identity: dict[str, object], key: str
) -> HexMachine:
    """Validate and reconstruct a fixture machine from one loaded Zarr group."""
    expected_identity = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if data.attrs.get("cache_schema") != MACHINE_CACHE_SCHEMA:
        raise ValueError("fixture cache schema differs from the requested carrier")
    if data.attrs.get("cache_key") != key:
        raise ValueError("fixture cache group does not carry its requested key")
    if data.attrs.get("semantic_identity") != expected_identity:
        raise ValueError("fixture cache semantic descriptor differs from the request")
    arrays = {name: np.array(data[name].values, copy=True) for name in data.data_vars}
    if data.attrs.get("payload_digest") != _payload_digest(arrays):
        raise ValueError("fixture cache payload digest does not match its arrays")
    required = set(_MACHINE_ARRAY_FIELDS) | {
        "cell_polygon_offsets",
        "cell_polygon_vertices",
        "passive_columns",
        *(f"radial_field_{position}" for position in range(4)),
        *(f"vertical_field_{position}" for position in range(4)),
    }
    if arrays.keys() != required:
        missing = sorted(required - arrays.keys())
        extra = sorted(arrays.keys() - required)
        raise ValueError(
            f"fixture cache arrays differ: missing={missing}, extra={extra}"
        )
    offsets = arrays.pop("cell_polygon_offsets")
    vertices = arrays.pop("cell_polygon_vertices")
    polygons = tuple(
        np.array(vertices[offsets[position] : offsets[position + 1]], copy=True)
        for position in range(len(offsets) - 1)
    )
    passive_columns = int(arrays.pop("passive_columns"))
    radial_field = tuple(
        arrays.pop(f"radial_field_{position}") for position in range(4)
    )
    vertical_field = tuple(
        arrays.pop(f"vertical_field_{position}") for position in range(4)
    )
    return HexMachine(
        coilset=None,
        passive_columns=passive_columns,
        cell_polygons=polygons,
        radial_field=radial_field,
        vertical_field=vertical_field,
        **arrays,
    )


def assert_machine_arrays_bitwise_identical(
    first: HexMachine, second: HexMachine
) -> tuple[int, int]:
    """Assert native dtype, shape and bytes agree for every cached input array."""
    first_arrays = _packed_machine_arrays(first)
    second_arrays = _packed_machine_arrays(second)
    if first_arrays.keys() != second_arrays.keys():
        raise AssertionError("fixture machines expose different cached array names")
    bytes_verified = 0
    for name in first_arrays:
        left = np.ascontiguousarray(first_arrays[name])
        right = np.ascontiguousarray(second_arrays[name])
        if left.dtype != right.dtype:
            raise AssertionError(
                f"fixture cache dtype changed for {name}: {left.dtype} != {right.dtype}"
            )
        if left.shape != right.shape:
            raise AssertionError(
                f"fixture cache shape changed for {name}: {left.shape} != {right.shape}"
            )
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise AssertionError(f"fixture cache bytes changed for {name}")
        bytes_verified += left.nbytes
    return len(first_arrays), bytes_verified


def _machine_cache_store(
    case: ReferenceCase, cells: int, *, passive: bool = True
) -> tuple[ZarrStore, dict[str, object]]:
    """Return the shared user-data store and semantic identity for one request."""
    identity = machine_cache_identity(case, cells, passive=passive)
    store = ZarrStore(filename=MACHINE_CACHE_FILENAME, dirname=".nova")
    store.group = store.hash_attrs(identity)
    return store, identity


@contextmanager
def _machine_cache_lock(store: ZarrStore):
    """Serialize readers and the miss builder without a GPFS rename."""
    lock_path = store.filepath.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        before = perf_counter()
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        waited = perf_counter() - before
        try:
            yield waited
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_cached_machine(store: ZarrStore, identity: dict[str, object]) -> HexMachine:
    """Load and validate one fixture machine group."""
    reader = ZarrStore(
        filename=store.filename,
        dirname=store.dirname,
        group=store.group,
    )
    reader.load()
    return _machine_from_dataset(reader.data, identity, store.group)


def cached_machine(
    case: ReferenceCase, cells: int, *, passive: bool = True
) -> HexMachine:
    """Load one shared fixture machine, building and publishing only on a miss."""
    store, identity = _machine_cache_store(case, cells, passive=passive)
    with _machine_cache_lock(store) as lock_wait_seconds:
        load_started = perf_counter()
        try:
            machine = _load_cached_machine(store, identity)
        except FileNotFoundError, KeyError, OSError, ValueError, AssertionError:
            load_seconds = perf_counter() - load_started
        else:
            load_seconds = perf_counter() - load_started
            arrays_verified, bytes_verified = assert_machine_arrays_bitwise_identical(
                machine, machine
            )
            machine.cache_receipt = MachineCacheReceipt(
                store=str(store.filepath),
                key=store.group,
                hit=True,
                lock_wait_seconds=lock_wait_seconds,
                load_seconds=load_seconds,
                build_seconds=0.0,
                store_seconds=0.0,
                validation_seconds=0.0,
                arrays_verified=arrays_verified,
                bytes_verified=bytes_verified,
                bitwise_stored_precision=True,
            )
            return machine

        store.delete_group()
        build_started = perf_counter()
        machine = build_machine(case, cells, passive=passive)
        build_seconds = perf_counter() - build_started
        store.data = _machine_dataset(machine, identity, store.group)
        store_started = perf_counter()
        store.store(mode=store.get_mode())
        store_seconds = perf_counter() - store_started
        validation_started = perf_counter()
        stored = _load_cached_machine(store, identity)
        arrays_verified, bytes_verified = assert_machine_arrays_bitwise_identical(
            machine, stored
        )
        validation_seconds = perf_counter() - validation_started
        machine.cache_receipt = MachineCacheReceipt(
            store=str(store.filepath),
            key=store.group,
            hit=False,
            lock_wait_seconds=lock_wait_seconds,
            load_seconds=load_seconds,
            build_seconds=build_seconds,
            store_seconds=store_seconds,
            validation_seconds=validation_seconds,
            arrays_verified=arrays_verified,
            bytes_verified=bytes_verified,
            bitwise_stored_precision=True,
        )
        return machine


def machine_cache_summary(name: str, machine: HexMachine) -> str:
    """Return one compact cache receipt line for a run log."""
    receipt = machine.cache_receipt
    if receipt is None:
        raise ValueError("machine was not requested through the persistent cache")
    status = "warm" if receipt.hit else "cold"
    return (
        f"CACHE fixture={name} status={status} key={receipt.key} "
        f"load_s={receipt.load_seconds:.9g} build_s={receipt.build_seconds:.9g} "
        f"store_s={receipt.store_seconds:.9g} "
        f"validation_s={receipt.validation_seconds:.9g} "
        f"arrays={receipt.arrays_verified} bytes={receipt.bytes_verified} "
        f"bitwise={receipt.bitwise_stored_precision} store={receipt.store}"
    )


def build_machine(
    case: ReferenceCase, cells: int, *, passive: bool = True
) -> HexMachine:
    """Return the coilset, its hexagonal plasma mesh and their couplings.

    Every plasma cell is coupled through its own polygon: an interior cell is a
    hexagon and a cell the first wall cuts is the clipped polygon, both routed
    to the closed-form section kernel by the plasma grid constructor.

    Both conductor families the entry declares are built the same way, through
    the direct coilset route rather than through a machine description: a
    rectangular element by its four scalars and a skewed one by its vertex
    loop. The passive structure is decomposed coarsely — see
    ``PASSIVE_ELEMENTS`` — because its plates are thin against the mesh, while
    the driven coils keep the filament count a wound pack needs.

    The limiter contour is handed to the wall solve explicitly. Left to default
    it reads the plasma polygon off the SUBFRAME, which is one hexagonal cell
    rather than the first wall, and the limiter flux would then be searched
    around a single cell.
    """
    coilset = CoilSet(
        dcoil=COIL_FILAMENTS, dplasma=cells, tplasma="hex", nwall=WALL_NODES
    )
    drive = case.drive(passive)
    for conductor in case.active:
        coilset.coil.insert(
            *conductor.placement,
            nturn=conductor.turns,
            part="pf",
            name=conductor.name,
        )
    for conductor in case.passive if passive else ():
        coilset.coil.insert(
            *conductor.placement,
            nturn=conductor.turns,
            part="passive",
            name=conductor.name,
            delta=PASSIVE_ELEMENTS,
        )
    coilset.firstwall.insert(case.wall, turn="hex")
    coilset.plasmagrid.solve()
    coilset.plasmawall.solve(boundary=case.wall)

    grid = coilset.plasmagrid.data
    sample = coilset.plasmagrid.sample_data
    limiter = coilset.plasmawall.data
    order = [str(label) for label in np.asarray(grid.coords["source"])]
    expected = [conductor.name for conductor in drive]
    if order[:-1] != expected:
        raise ValueError(f"coupling column order {order} is not the conductor order")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    section = np.asarray(coilset.subframe.loc[:, "section"], dtype=object)[plasma]
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    return HexMachine(
        coilset=coilset,
        source_current=np.array([conductor.current for conductor in drive]),
        passive_columns=len(case.passive) if passive else 0,
        node=np.c_[np.asarray(grid.x), np.asarray(grid.z)].astype(float),
        area=np.asarray(coilset.aloc["plasma", "area"], dtype=float),
        cell_polygons=tuple(
            np.asarray(polygon.poly.exterior.coords, dtype=float)[:-1, :2]
            for polygon in material
        ),
        hexagon=np.asarray([name == "hexagon" for name in section]),
        stencil=np.asarray(grid["stencil"]),
        wall_node=np.c_[np.asarray(limiter.x), np.asarray(limiter.z)].astype(float),
        source_to_grid=np.asarray(grid["Psi"])[:, :-1],
        plasma_to_grid=np.asarray(grid["Psi_"]),
        plasma_to_grid_r=np.asarray(grid["PsiR_"]),
        plasma_to_grid_z=np.asarray(grid["PsiZ_"]),
        sampling_vertices=np.asarray(coilset.plasmagrid.sampling_vertices),
        sample_coordinates=np.asarray(coilset.plasmagrid.sample_coordinates),
        source_to_sample=np.asarray(sample["Psi"])[:, :-1],
        plasma_to_sample=np.asarray(sample["Psi_"]),
        plasma_to_sample_r=np.asarray(sample["PsiR_"]),
        plasma_to_sample_z=np.asarray(sample["PsiZ_"]),
        source_to_wall=np.asarray(limiter["Psi"])[:, :-1],
        plasma_to_wall=np.asarray(limiter["Psi_"]),
        plasma_to_wall_r=np.asarray(limiter["PsiR_"]),
        plasma_to_wall_z=np.asarray(limiter["PsiZ_"]),
        radial_field=(
            np.asarray(grid["Br"])[:, :-1],
            np.asarray(grid["Br_"]),
            np.asarray(grid["BrR_"]),
            np.asarray(grid["BrZ_"]),
        ),
        vertical_field=(
            np.asarray(grid["Bz"])[:, :-1],
            np.asarray(grid["Bz_"]),
            np.asarray(grid["BzR_"]),
            np.asarray(grid["BzZ_"]),
        ),
    )


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------
def _flux_function(psi_norm: np.ndarray, sample: np.ndarray):
    """Return a traceable absolute flux function of normalised flux."""
    grid = jnp.asarray(psi_norm)
    value = jnp.asarray(sample)

    def gradient(argument):
        """Return the tabulated gradient at one normalised flux."""
        return jnp.interp(jnp.asarray(argument), grid, value)

    return gradient


def forward_source(case: ReferenceCase) -> ForwardSource:
    """Return the absolute source state the stored profiles declare."""
    return ForwardSource(
        core=DomainProfile(
            p_prime=_flux_function(case.psi_norm, case.p_prime),
            ff_prime=_flux_function(case.psi_norm, case.ff_prime),
        ),
        boundary_pressure=float(case.pressure[-1]),
        boundary_field_function=float(case.field_function[-1]),
    )


def forward_operator(case: ReferenceCase, machine: HexMachine) -> ForwardFluxOperator:
    """Return the free-boundary map on the hexagonal mesh.

    The polarity is negative because the stored flux sense puts the magnetic
    axis at a MINIMUM of the total poloidal flux; every plasma cell lies inside
    the first wall by construction, so no cell is excluded material.
    """
    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.asarray(machine.source_to_grid),
            plasma_target=jnp.asarray(machine.plasma_to_grid),
            null=Null2D.from_coordinates(
                machine.node, machine.interior_stencil, maxsize=5
            ),
            plasma_target_r=jnp.asarray(machine.plasma_to_grid_r),
            plasma_target_z=jnp.asarray(machine.plasma_to_grid_z),
        ),
        wall=FluxTarget(
            source_target=jnp.asarray(machine.source_to_wall),
            plasma_target=jnp.asarray(machine.plasma_to_wall),
            null=Null1D(jnp.asarray(machine.wall_node, dtype=jnp.float64)),
            plasma_target_r=jnp.asarray(machine.plasma_to_wall_r),
            plasma_target_z=jnp.asarray(machine.plasma_to_wall_z),
        ),
        source=forward_source(case),
        external_current=jnp.asarray(machine.source_current),
        area=jnp.asarray(machine.area),
        polarity=-1,
        moment_geometry=machine.moment_geometry,
        sample=FluxTarget(
            source_target=jnp.asarray(machine.source_to_sample),
            plasma_target=jnp.asarray(machine.plasma_to_sample),
            null=Null1D(jnp.asarray(machine.sample_coordinates, dtype=jnp.float64)),
            plasma_target_r=jnp.asarray(machine.plasma_to_sample_r),
            plasma_target_z=jnp.asarray(machine.plasma_to_sample_z),
        ),
    )


def receipt_mesh(machine: HexMachine) -> StencilMesh:
    """Return the mesh the published receipts are differentiated on.

    Every ring the fit can carry is handed over, which is a different and
    weaker selection than the null search uses. The null search is given the
    regular rings — centre and all six neighbours whole hexagons — because a
    quadratic fitted on a clipped cell's displaced centroid can report a
    stationary point the flux map does not have. A DERIVATIVE has no such
    requirement: the centroid and the flux at it are both exact, so a
    least-squares quadratic through them is a valid local approximation
    whatever the neighbourhood looks like, and the only real requirement is
    that the cluster determine a quadratic at all.

    The difference is not cosmetic. Restricted to the regular rings, 37 of the
    383 core cells on the banked coarse mesh carry no derivative, all of them at the
    plasma edge where the poloidal field is largest, and the internal
    inductance the receipt integrates falls 16 % short — 15 % of that from the
    missing cells and 1 % from the fit. Selecting on conditioning instead
    leaves every core cell covered. The limit used is the one the mesh itself
    enforces, so nothing is excluded here that would not be refused there; the
    receipts below are unchanged by tightening it to twenty, which is the
    evidence that this choice is not doing any work of its own.
    """
    stencil = np.asarray(machine.stencil)
    condition = ring_condition(machine.node, stencil)
    return StencilMesh(
        coordinate=machine.node,
        stencil=stencil[condition < RING_CONDITION_LIMIT],
        area=machine.area,
    )


def flux_resolution(profile: ForwardProfile, equilibrium) -> float:
    """Return the finest flux difference [Wb] the whole solve can express.

    Two floors compete and the coarser one binds. One is a step of the ladder
    the axis flux lands on: every profile is evaluated on a normalised flux
    formed against that fitted value, so its spacing in the precision the null
    search fits at floors everything downstream. The other is the round-off the
    plasma coupling accumulates — each residual entry is a dot product over the
    grid's cells, so it carries up to that many units in the last place of the
    flux scale.

    Which one binds follows the fit precision, so neither is assumed: a single
    precision fit puts its ladder decades above the coupling sum and binds
    alone, while a double one puts a step at a single unit in the last place,
    underneath the sum, and the sum binds instead. The dtype is read off the
    fit, and the magnitude is taken because the spacing of a negative value is
    negative and this case's axis flux is negative.
    """
    dtype = np.dtype(profile.operator.grid.null.fit_dtype)
    axis_flux = abs(float(equilibrium.topology.axis_flux))
    ladder = float(np.spacing(np.asarray(axis_flux, dtype=dtype)))
    scale = float(jnp.max(jnp.abs(equilibrium.flux)))
    accumulation = profile.operator.grid.node_number * float(np.spacing(scale))
    return max(ladder, accumulation)


def forward_profile(case: ReferenceCase, machine: HexMachine) -> ForwardProfile:
    """Return the published solve carried on the production hexagonal mesh."""
    return ForwardProfile(
        operator=forward_operator(case, machine),
        lattice=receipt_mesh(machine),
        newton_steps=NEWTON_STEPS,
    )


@dataclass
class SolvedEquilibrium:
    """One converged solve and the observations that qualify it."""

    case: ReferenceCase = field(repr=False)
    machine: HexMachine = field(repr=False)
    flux: jnp.ndarray = field(repr=False)
    cell_current: jnp.ndarray = field(repr=False)
    masks: object = field(repr=False)
    topology: object = field(repr=False)
    moments: object = field(repr=False)
    ledger: object = field(repr=False)
    fixed_point: object = field(repr=False)

    @property
    def grid_flux(self) -> np.ndarray:
        """Return the solved total poloidal flux [Wb] on the plasma cells."""
        return np.asarray(self.flux)[: len(self.machine.node)]

    @property
    def reference_flux(self) -> np.ndarray:
        """Return the stored total poloidal flux [Wb] on the plasma cells."""
        return self.case.flux(self.machine.radius, self.machine.node[:, 1])

    @property
    def reference_scale(self) -> float:
        """Return the reference radius the stored shape moments are read on.

        The published poloidal beta and internal inductance are normalised on
        the volume-averaged major radius of the labelled core; the stored pair
        is normalised on the machine's vacuum-field reference radius. The two
        differ by exactly this ratio and nothing else.
        """
        return float(self.moments.major_radius) / self.case.reference_radius

    @property
    def pitch(self) -> float:
        """Return the hexagonal cell pitch [m] of the mesh."""
        return float(np.sqrt(np.median(self.machine.area[self.machine.hexagon])))

    def midplane_radii(self, flux_map, height: float) -> tuple[float, float]:
        """Return the inboard and outboard core radii on one horizontal cut."""
        radius, vertical = self.machine.radius, self.machine.node[:, 1]
        band = np.abs(vertical - height) < self.pitch
        selected = radius[band & np.asarray(flux_map)]
        return float(selected.min()), float(selected.max())

    def deviations(self) -> dict[str, float]:
        """Return every row of the reproduction against the stored slice.

        One definition, read by the closure comparison and by the evidence
        figures alike, so a number quoted in either is the number the other
        asserts on. The four fractional rows are percentages of their own
        reference and the four geometric rows are millimetres; the two shape
        moments are referred first, because the published pair and the
        observation operator's normalise on different major radii.
        """
        moments = self.case.map_moments()
        scale = self.reference_scale
        axis = np.asarray(self.topology.axis)
        core = np.asarray(self.masks.core)
        height = float(self.case.axis[1])
        inboard, outboard = self.midplane_radii(core, height)
        band = np.abs(self.case.boundary[:, 1] - height) < 0.5
        stored = self.case.boundary[band, 0]
        deviation = np.max(np.abs(self.grid_flux - self.reference_flux)[core])
        return {
            "plasma current": 100.0
            * (float(self.moments.plasma_current) / self.case.plasma_current - 1.0),
            "poloidal beta": 100.0
            * (
                float(self.moments.poloidal_beta) * scale / moments["poloidal_beta"]
                - 1.0
            ),
            "internal inductance": 100.0
            * (
                float(self.moments.internal_inductance)
                * scale
                / moments["internal_inductance"]
                - 1.0
            ),
            "flux sup-norm": 100.0 * deviation / abs(self.case.flux_span),
            "axis radius": 1e3 * (axis[0] - self.case.axis[0]),
            "axis height": 1e3 * (axis[1] - self.case.axis[1]),
            "inboard edge": 1e3 * (inboard - stored.min()),
            "outboard edge": 1e3 * (outboard - stored.max()),
        }


#: Deviation rows carried as a percentage of their own reference; the rest of
#: :meth:`SolvedEquilibrium.deviations` is in millimetres.
FRACTIONAL_ROWS = (
    "plasma current",
    "poloidal beta",
    "internal inductance",
    "flux sup-norm",
)


def seed_flux(case: ReferenceCase, machine: HexMachine) -> jnp.ndarray:
    """Return the stored map on the plasma cells and the wall nodes."""
    return jnp.asarray(
        np.r_[
            case.flux(machine.radius, machine.node[:, 1]),
            case.flux(machine.wall_node[:, 0], machine.wall_node[:, 1]),
            case.flux(
                machine.sample_coordinates[:, 0], machine.sample_coordinates[:, 1]
            ),
        ]
    )


def stored_map_residual(case: ReferenceCase, machine: HexMachine, core) -> float:
    """Return how far the stored map is from a fixed point of one machine.

    The stored flux map is pushed once through the free-boundary map and the
    sup-norm of what it moves over the labelled core is read against the axis
    to boundary span. Nothing is solved, so this is the machine model's own
    disagreement with the reference before any solve can spread it around —
    the yardstick any candidate missing conductor has to reach.
    """
    seed = seed_flux(case, machine)
    image = forward_operator(case, machine).flux_map()(seed)
    moved = np.asarray(image - seed)[: len(machine.node)]
    return float(np.max(np.abs(moved)[np.asarray(core)]) / abs(case.flux_span))


def solve(case: ReferenceCase, machine: HexMachine) -> SolvedEquilibrium:
    """Return the equilibrium the stored sources and coil currents support.

    The route is the root find, not the relaxed ladder, and that is physics
    rather than taste. An elongated diverted plasma held by fixed coil currents
    is axisymmetrically unstable to a vertical displacement, so the
    free-boundary map has an eigenvalue outside the unit circle and EVERY
    relaxed iteration walks down it until the plasma meets the wall and the
    solve falls onto the vacuum branch. The Newton step linearises the map and
    solves ``(I - J) s = f``, which is indifferent to the sign of that
    eigenvalue, so it holds the equilibrium the seed was placed on.

    The trial flux is the stored map, which puts the root find on the confined
    branch of a problem whose other fixed point is the vacuum field.
    """
    operator = forward_operator(case, machine)
    history = fixed_point.newton_krylov(
        operator.flux_map(),
        seed_flux(case, machine),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=KRYLOV_ITERATIONS,
        warmup=0,
    )
    current_moments, measure, masks, topology = (
        operator.current_moments_and_observation(history.state)
    )
    cell_current = current_moments.cell_current
    return SolvedEquilibrium(
        case=case,
        machine=machine,
        flux=history.state,
        cell_current=cell_current,
        masks=masks,
        topology=topology,
        moments=observe_moments(measure, topology.flux_span),
        ledger=current_ledger(cell_current, measure.masks),
        fixed_point=history,
    )


@lru_cache(maxsize=4)
def _machine(cells: int, passive: bool) -> tuple[ReferenceCase, HexMachine]:
    """Return the reference and its production mesh at one resolution.

    The coupling assembly dominates the cost of this module, so the operator
    and the published solve are driven on ONE machine rather than on two
    identical ones. Neither argument carries a default, because a default
    would let the same machine be requested by two different call signatures
    and the memo would then miss and build it twice.
    """
    configure_dtypes()
    case = require_reference()
    return case, build_machine(case, cells, passive=passive)


def test_persistent_fixture_machine_cache_builds_once(monkeypatch, tmp_path):
    """A semantic miss publishes once and the next request restores native bytes."""

    def values(shape, dtype=np.float64):
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    machine = HexMachine(
        coilset=None,
        source_current=values((2,), np.float32),
        passive_columns=1,
        node=values((3, 2)),
        area=values((3,), np.float32),
        cell_polygons=(values((4, 2)), values((5, 2)), values((6, 2))),
        hexagon=np.array([True, False, True]),
        stencil=values((2, 3), np.int32),
        wall_node=values((2, 2)),
        source_to_grid=values((3, 2)),
        plasma_to_grid=values((3, 3)),
        plasma_to_grid_r=values((3, 3)),
        plasma_to_grid_z=values((3, 3)),
        sampling_vertices=values((3, 6), np.int16),
        sample_coordinates=values((4, 2)),
        source_to_sample=values((4, 2)),
        plasma_to_sample=values((4, 3)),
        plasma_to_sample_r=values((4, 3)),
        plasma_to_sample_z=values((4, 3)),
        source_to_wall=values((2, 2)),
        plasma_to_wall=values((2, 3)),
        plasma_to_wall_r=values((2, 3)),
        plasma_to_wall_z=values((2, 3)),
        radial_field=(
            values((3, 2)),
            values((3, 3)),
            values((3, 3)),
            values((3, 3)),
        ),
        vertical_field=(
            values((3, 2)),
            values((3, 3)),
            values((3, 3)),
            values((3, 3)),
        ),
    )
    identity = {"schema": MACHINE_CACHE_SCHEMA, "content": "synthetic"}

    def synthetic_store(_case, _cells, *, passive=True):
        assert passive
        store = ZarrStore(filename="fixture_machine", dirname=tmp_path)
        store.group = store.hash_attrs(identity)
        return store, identity

    builds = []

    def synthetic_build(_case, _cells, *, passive=True):
        assert passive
        builds.append(True)
        return machine

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_machine_cache_store", synthetic_store)
    monkeypatch.setattr(module, "build_machine", synthetic_build)
    cold = cached_machine(None, -3, passive=True)
    warm = cached_machine(None, -3, passive=True)
    arrays_verified, bytes_verified = assert_machine_arrays_bitwise_identical(
        cold, warm
    )
    assert builds == [True]
    assert not cold.cache_receipt.hit
    assert warm.cache_receipt.hit
    assert arrays_verified == 31
    assert bytes_verified > 0
    assert cold.cache_receipt.bitwise_stored_precision
    assert warm.cache_receipt.bitwise_stored_precision


@lru_cache(maxsize=4)
def _solved(cells: int, passive: bool) -> SolvedEquilibrium:
    """Return the converged solve on one mesh resolution."""
    return solve(*_machine(cells, passive))


@lru_cache(maxsize=2)
def _published(cells: int):
    """Return the published solve and its mesh at one resolution.

    The route is left at its default, so this is also where the shipped
    default is exercised: the demonstration would not converge on a relaxed
    one and says so through the eigenvalue measured below.
    """
    case, machine = _machine(cells, True)
    profile = forward_profile(case, machine)
    equilibrium = profile.solve(
        seed_flux(case, machine),
        gmres_iterations=KRYLOV_ITERATIONS,
        warmup=0,
    )
    return profile, equilibrium


@pytest.fixture(scope="module")
def solved() -> SolvedEquilibrium:
    """Return the converged solve on the suite mesh."""
    return _solved(SUITE_CELLS, True)


@pytest.fixture(scope="module")
def published():
    """Return the ForwardProfile solve and its receipts on the suite mesh."""
    return _published(SUITE_CELLS)


def test_hex_operator_carries_the_authored_cell_geometry(solved):
    """The production constructor keeps every plasma polygon and shared node."""
    operator = forward_operator(solved.case, solved.machine)
    geometry = operator.moment_geometry
    coordinate = solved.machine.node
    flux = (
        0.7
        + 0.2 * coordinate[:, 0]
        - 0.3 * coordinate[:, 1]
        + 0.1 * coordinate[:, 0] ** 2
        + 0.05 * coordinate[:, 0] * coordinate[:, 1]
    )
    shared = geometry.atomic_mesh.node_coordinates
    expected = (
        0.7
        + 0.2 * shared[:, 0]
        - 0.3 * shared[:, 1]
        + 0.1 * shared[:, 0] ** 2
        + 0.05 * shared[:, 0] * shared[:, 1]
    )

    np.testing.assert_allclose(
        geometry.shared_node_flux(jnp.asarray(flux)), expected, rtol=0.0, atol=2.0e-13
    )
    assert len(geometry.polygons) == len(solved.machine.node)
    assert geometry.atomic_mesh.cell_nodes.shape[0] == len(solved.machine.node)


def test_couplings_and_current_vectors_share_geometric_expansion_centres(published):
    """Matrix translation responses recover the vector-side geometric centres."""
    from nova.biot.greens import section_centroid
    from nova.biot.polysection import PolySection
    from nova.biot.polygonanalytic import (
        polygon_analytic_field_moments,
        polygon_analytic_flux_moments,
    )
    from shapely.geometry import Polygon

    profile, equilibrium = published
    operator = profile.operator
    machine = _machine(SUITE_CELLS, True)[1]
    geometry = operator.moment_geometry
    vector_centre = np.asarray(geometry.atomic_mesh.centroids)
    geometric_centre = np.asarray(
        [section_centroid(vertices) for vertices in geometry.polygons]
    )
    matrix_centre = np.asarray(
        [
            PolySection._material_area_centroid(Polygon(vertices))
            for vertices in geometry.polygons
        ]
    )
    cell_width = np.sqrt(np.asarray(machine.area))
    sample = np.unique(np.linspace(0, len(machine.node) - 1, 12, dtype=np.intp))
    target_r = machine.radius[sample]
    target_z = machine.node[sample, 1]
    recovered = np.empty_like(vector_centre)
    translation_spread = np.empty_like(vector_centre)
    field_translation_error = np.empty((len(machine.node), 2))
    uniform_absolute_error = []
    uniform_relative_error = []

    for column, (vertices, centre, authored_centre) in enumerate(
        zip(geometry.polygons, vector_centre, matrix_centre, strict=True)
    ):
        flux = polygon_analytic_flux_moments(
            target_r, target_z, vertices, expansion_point=authored_centre
        )
        probe_offset = cell_width[column] * np.array([0.125, -0.125])
        probe_centre = authored_centre + probe_offset
        translated_flux = polygon_analytic_flux_moments(
            target_r, target_z, vertices, expansion_point=probe_centre
        )
        translation_scale = float(np.dot(flux[0], flux[0]))
        recovered_offset = np.array(
            [
                -np.dot(flux[0], translated_flux[1] - flux[1]) / translation_scale,
                -np.dot(flux[0], translated_flux[2] - flux[2]) / translation_scale,
            ]
        )
        recovered[column] = probe_centre - recovered_offset
        translation_spread[column] = [
            np.max(np.abs(translated_flux[1] - flux[1] + probe_offset[0] * flux[0]))
            / np.max(np.abs(flux[0])),
            np.max(np.abs(translated_flux[2] - flux[2] + probe_offset[1] * flux[0]))
            / np.max(np.abs(flux[0])),
        ]
        assembled_uniform = machine.plasma_to_grid[sample, column]
        uniform_difference = np.abs(flux[0] - assembled_uniform)
        uniform_scale = np.maximum(
            np.maximum(np.abs(flux[0]), np.abs(assembled_uniform)), 1.0e-300
        )
        uniform_absolute_error.extend(uniform_difference.tolist())
        uniform_relative_error.extend((uniform_difference / uniform_scale).tolist())
        radial, vertical = polygon_analytic_field_moments(
            target_r, target_z, vertices, expansion_point=authored_centre
        )
        for index, (direct, assembled) in enumerate(
            (
                (
                    radial,
                    (
                        machine.radial_field[1][sample, column],
                        machine.radial_field[2][sample, column],
                        machine.radial_field[3][sample, column],
                    ),
                ),
                (
                    vertical,
                    (
                        machine.vertical_field[1][sample, column],
                        machine.vertical_field[2][sample, column],
                        machine.vertical_field[3][sample, column],
                    ),
                ),
            )
        ):
            matrix_offset = centre - authored_centre
            expected_radial = direct[1] + matrix_offset[0] * direct[0]
            expected_vertical = direct[2] + matrix_offset[1] * direct[0]
            scale = max(
                float(np.max(np.abs(assembled[1]))),
                float(np.max(np.abs(assembled[2]))),
                1.0e-30,
            )
            field_translation_error[column, index] = (
                max(
                    float(np.max(np.abs(assembled[1] - expected_radial))),
                    float(np.max(np.abs(assembled[2] - expected_vertical))),
                )
                / scale
            )

    matrix_vector_delta = recovered - vector_centre
    geometric_vector_delta = geometric_centre - vector_centre
    matrix_geometric_delta = recovered - geometric_centre
    matrix_vector_widths = matrix_vector_delta / cell_width[:, None]
    geometric_vector_widths = geometric_vector_delta / cell_width[:, None]
    matrix_geometric_widths = matrix_geometric_delta / cell_width[:, None]
    current_moments = operator.cell_current_moments(equilibrium.flux)
    inconsistent_current = -matrix_vector_delta[:, 0] * np.asarray(
        current_moments.radial_moment
    ) - matrix_vector_delta[:, 1] * np.asarray(current_moments.vertical_moment)
    grid_impact = machine.plasma_to_grid @ inconsistent_current
    wall_impact = machine.plasma_to_wall @ inconsistent_current
    total_impact = np.r_[grid_impact, wall_impact]
    per_cell_impact = np.max(
        np.abs(
            np.vstack([machine.plasma_to_grid, machine.plasma_to_wall])
            * inconsistent_current[None, :]
        ),
        axis=0,
    )

    for index in range(len(machine.node)):
        print(
            "expansion_centre_cell="
            f"{index} whole_hexagon={bool(machine.hexagon[index])} "
            f"recovered_matrix={recovered[index].tolist()} "
            f"atomic_mesh={vector_centre[index].tolist()} "
            f"geometric={geometric_centre[index].tolist()} "
            f"matrix_minus_vector={matrix_vector_delta[index].tolist()} "
            f"cell_width_m={cell_width[index]:.17g} "
            f"matrix_minus_vector_cell_widths="
            f"{matrix_vector_widths[index].tolist()} "
            f"flux_impact_wb={per_cell_impact[index]:.17g}"
        )

    clipped = ~machine.hexagon
    score = {
        "cell_count": len(machine.node),
        "clipped_cell_count": int(clipped.sum()),
        "matrix_vector_max_m": float(np.max(np.abs(matrix_vector_delta))),
        "matrix_vector_clipped_max_m": float(
            np.max(np.abs(matrix_vector_delta[clipped]))
        ),
        "matrix_vector_max_cell_widths": float(np.max(np.abs(matrix_vector_widths))),
        "matrix_vector_clipped_max_cell_widths": float(
            np.max(np.abs(matrix_vector_widths[clipped]))
        ),
        "geometric_vector_max_m": float(np.max(np.abs(geometric_vector_delta))),
        "geometric_vector_max_cell_widths": float(
            np.max(np.abs(geometric_vector_widths))
        ),
        "matrix_geometric_max_m": float(np.max(np.abs(matrix_geometric_delta))),
        "matrix_geometric_max_cell_widths": float(
            np.max(np.abs(matrix_geometric_widths))
        ),
        "uniform_g0_worst_absolute_wb_per_a": float(np.max(uniform_absolute_error)),
        "uniform_g0_worst_relative": float(np.max(uniform_relative_error)),
        "translation_identity_spread_m": float(np.max(np.abs(translation_spread))),
        "field_translation_relative_error": float(np.max(field_translation_error)),
        "flux_map_impact_sup_wb": float(np.max(np.abs(total_impact))),
        "flux_map_impact_rms_wb": float(np.sqrt(np.mean(total_impact**2))),
        "flux_map_impact_fraction_of_span": float(
            np.max(np.abs(total_impact)) / abs(float(equilibrium.topology.flux_span))
        ),
    }
    print(f"expansion_centre_score={score!r}")
    assert score["matrix_vector_max_cell_widths"] < 1.0e-12, score
    assert score["geometric_vector_max_cell_widths"] < 1.0e-12, score
    assert score["matrix_geometric_max_cell_widths"] < 1.0e-12, score


# --------------------------------------------------------------------------
# the conventions, pinned before any solve
# --------------------------------------------------------------------------
def test_the_stored_gradients_integrate_to_the_stored_plasma_current():
    """The two tabulated gradients drive the stored current on the stored map.

    This is the whole convention chain in one number. It reads the stored flux
    map to label which cells lie inside the stored boundary, evaluates the
    package's own current density from the tabulated gradients there, and sums
    it over the cell areas. A wrong flux sense flips it, a flux-per-radian
    reading scales it by two pi, and a mis-scaled gradient changes its
    magnitude — none of which any later fit could absorb.
    """
    case = require_reference()
    radius, height = np.meshgrid(
        np.linspace(case.wall[:, 0].min(), case.wall[:, 0].max(), 401),
        np.linspace(case.wall[:, 1].min(), case.wall[:, 1].max(), 601),
        indexing="ij",
    )
    area = (radius[1, 0] - radius[0, 0]) * (height[0, 1] - height[0, 0])
    psi_norm = (case.flux(radius, height) - case.flux_axis) / case.flux_span
    inside = psi_norm <= 1.0
    clipped = np.clip(psi_norm, 0.0, 1.0)
    p_prime = np.interp(clipped, case.psi_norm, case.p_prime)
    ff_prime = np.interp(clipped, case.psi_norm, case.ff_prime)
    density = -2.0 * np.pi * (radius * p_prime + ff_prime / (mu_0 * radius))
    current = np.sum(np.where(inside, density, 0.0)) * area
    assert abs(current / case.plasma_current - 1.0) < CONVENTION_TOLERANCE, (
        f"{current:.6e} against {case.plasma_current:.6e}"
    )


def test_the_stored_boundary_encloses_the_stored_axis():
    """The reference slice is the diverted flat-top point the module claims."""
    case = require_reference()
    assert len(case.x_point) == 1
    assert case.x_point[0, 1] < case.axis[1]
    assert case.boundary[:, 0].min() < case.axis[0] < case.boundary[:, 0].max()
    assert abs(case.plasma_current) > 1.0e7
    assert case.flux_span > 0.0


# --------------------------------------------------------------------------
# the machine model, against the machine the entry declares
# --------------------------------------------------------------------------
def test_every_conductor_the_entry_declares_is_placed(solved):
    """Nothing the entry carries current on is left out of the machine.

    Both conductor families are read and both reach the coupling: fourteen
    driven elements from ``pf_active`` — twelve rectangular central solenoid
    and poloidal field packs plus the two skewed in-vessel stabilisation
    plates — and one hundred and three passive loops from ``pf_passive``. The
    skewed section is what makes this worth asserting rather than assuming: a
    reader that resolves only rectangles finds no conductor in the passive
    family and none in the stabilisation pair, and drops both without raising,
    so the count is checked against the entry rather than against itself.

    Placement is checked by area rather than by count, because a conductor
    tiled into the wrong footprint is as wrong as one omitted, and it is the
    area the coupling integrates over. The tile counts themselves are not
    asserted — they are the inserter's business, and it takes most plates to
    three or four pieces and a quarter of them to one — but the footprint is:
    every element's tiles sum to the area its own section declares, the worst
    of the hundred and seventeen to 3.6e-7 relative.
    """
    case, machine = solved.case, solved.machine
    assert case.unplaced == (), case.unplaced
    assert len(case.active) == 14
    assert sum(1 for conductor in case.active if conductor.polygon is not None) == 2
    assert len(case.passive) == 103
    subframe = machine.coilset.subframe
    plasma = np.asarray(subframe.loc[:, "plasma"], dtype=bool)
    owner = np.asarray(subframe.frame, dtype=object)[~plasma]
    area = np.asarray(subframe.loc[:, "area"], dtype=float)[~plasma]
    for conductor in case.drive():
        carried = float(area[owner == conductor.name].sum())
        assert abs(carried / conductor.declared_area - 1.0) < AREA_TOLERANCE, (
            conductor.name,
            carried,
        )


def test_the_stabilisation_pair_carries_the_turns_the_entry_declares(solved):
    """The pair's turn count, wiring and worth are read off the entry.

    The two in-vessel stabilisation plates are the one place in this entry where
    a current has to be told apart from an ampere-turn count, and the only
    driven elements a rectangle-only reader would drop. The coil declares a
    single current of 65.4 A and two elements at turn counts +1 and -1, so what
    the machine carries is two branches of one turn at equal magnitude and
    opposite sign. The opposite signs are the entry's whole statement of the
    anti-series wiring — it populates no ``circuit`` and no ``supply`` — and the
    single turn is what makes 65.4 A simultaneously the circuit current and the
    ampere-turns of either branch, a coincidence that would not survive the four
    turns per branch a wound pair carries.

    Which reading the entry meant is left open, because nothing here turns on
    it. The flux the pair puts on the plasma cells is 0.002 % of the axis to
    boundary span, and what is asserted is that it stays below the passive
    structure's own contribution even when multiplied by the four turns it is
    not declared with. The passive structure is in turn an order below the
    deviation this module measures, which puts the pair four decimal places
    beneath it under either reading.
    """
    case, machine = solved.case, solved.machine
    pair = [
        conductor
        for conductor in case.active
        if conductor.name in STABILISATION_ELEMENTS
    ]
    assert [conductor.name for conductor in pair] == list(STABILISATION_ELEMENTS)
    assert all(conductor.polygon is not None for conductor in pair)
    assert {conductor.turns for conductor in pair} == {1.0}, pair
    assert pair[0].current == -pair[1].current != 0.0, pair

    names = [conductor.name for conductor in case.drive()]
    columns = np.array([name in STABILISATION_ELEMENTS for name in names])
    assert columns.sum() == len(STABILISATION_ELEMENTS)
    flux = machine.source_to_grid[:, columns] @ machine.source_current[columns]
    peak = float(np.max(np.abs(flux)) / abs(case.flux_span))
    assert peak < STABILISATION_FLUX_CEILING, peak
    passive = float(np.max(np.abs(machine.passive_flux)) / abs(case.flux_span))
    assert STABILISATION_TURN_BOUND * peak < passive, (peak, passive)


def test_the_passive_structure_cannot_carry_the_reproduction_gap(solved):
    """The leading candidate for the deviation floor is measured, and is too small.

    The passive structure is the leading candidate for the deviation that
    survives mesh refinement: a hundred and three loops carrying 14.9 kA
    between them, close enough to the plasma to matter and easy to leave out.
    What they are worth can be bounded without solving anything, because the
    flux they put on the plasma cells is a direct matrix-vector product of the
    coupling with their own currents, and it comes to 0.093 % of the axis to
    boundary span against the 0.958 % by which the stored map misses being a
    fixed point of this machine — short by a factor of ten.

    That is a falsification, not a closure. The passive currents are real and
    now carried, but they cannot be the floor. What remains is attributed
    below.
    """
    case, machine = solved.case, solved.machine
    assert machine.passive_columns == len(case.passive)
    peak = float(np.max(np.abs(machine.passive_flux)) / abs(case.flux_span))
    assert peak < PASSIVE_FLUX_CEILING, peak
    residual = stored_map_residual(case, machine, solved.masks.core)
    assert residual > PASSIVE_SHORTFALL * peak, (residual, peak)


def test_the_passive_closure_moves_the_reproduction_by_a_tenth_of_a_percent(solved):
    """Solving both machine models is what the closure is finally worth.

    The bound above is on the flux the passive currents put in. This is on
    what comes out: the same source, the same profiles and the same mesh
    solved twice, once with the passive structure in the machine and once
    without.

    The direct passive field is independently bounded at a tenth of a percent
    of the reference span. The coupled-map response must remain commensurate
    with that registered cross-source budget; a larger response is a map-gain
    regression, not evidence for relaxing the passive-current ceiling. The
    passive structure still belongs in the machine because its authored
    current is physical input, while repair of the amplified response belongs
    to the coupled-map feedback path.
    """
    without_current = solved.machine.source_current.copy()
    without_current[-solved.machine.passive_columns :] = 0.0
    without_machine = replace(
        solved.machine, source_current=without_current, passive_columns=0
    )
    without = solve(solved.case, without_machine).deviations()
    structure = solved.deviations()
    closed = without["flux sup-norm"] - structure["flux sup-norm"]
    # The direct passive field meets its own input budget, but the coupled map
    # has noncontractive free-boundary resolvent gain. This closure ceiling
    # therefore remains a deliberately red regression owned by the map-gain
    # chain rather than being relaxed to admit the amplified response.
    assert 0.0 < closed < PASSIVE_REPRODUCTION_MOVE_CEILING, closed
    assert closed / without["flux sup-norm"] < 0.15, closed
    assert structure["flux sup-norm"] > 1.0, structure["flux sup-norm"]
    for name in ("axis radius", "axis height"):
        assert abs(structure[name]) < abs(without[name]), name
        assert abs(structure[name] - without[name]) < 5.0, name
    for name in ("plasma current", "poloidal beta"):
        assert (
            abs(structure[name] - without[name]) < PASSIVE_REPRODUCTION_MOVE_CEILING
        ), name
    inductance_move = structure["internal inductance"] - without["internal inductance"]
    assert abs(structure["internal inductance"]) < abs(
        without["internal inductance"]
    ), (without["internal inductance"], structure["internal inductance"])
    assert (
        PASSIVE_REPRODUCTION_MOVE_CEILING
        < abs(inductance_move)
        < PASSIVE_INTERNAL_INDUCTANCE_MOVE
    ), inductance_move


# --------------------------------------------------------------------------
# why the route is a root find
# --------------------------------------------------------------------------
def test_the_free_boundary_map_does_not_contract_at_the_equilibrium(published):
    """The route choice is forced by a measured eigenvalue, not by preference.

    A relaxed iteration converges only where the map contracts. Power iteration
    on the exact tangent at the converged equilibrium returns a dominant
    eigenvalue ABOVE one, so no amount of damping rescues a relaxed route here:
    a step relaxed by ``beta`` scales that mode by
    ``(1 - beta) + beta * lambda``, which exceeds one for every ``beta`` in
    ``(0, 1]`` once ``lambda`` does. That single number is why the
    demonstration is a root find, and it is measured rather than assumed.

    The iteration count is fixed rather than run to a tolerance, so a map whose
    spectrum does not separate costs the same and cannot hang.
    """
    profile, equilibrium = published
    tangent = jax.linearize(profile.flux_map(), equilibrium.flux)[1]
    generator = np.random.default_rng(11)
    vector = jnp.asarray(generator.normal(size=equilibrium.flux.shape))
    vector = vector / jnp.linalg.norm(vector)
    for _ in range(POWER_ITERATIONS):
        vector = tangent(vector)
        vector = vector / jnp.linalg.norm(vector)
    eigenvalue = float(jnp.dot(vector, tangent(vector)))
    print(f"dominant_map_eigenvalue={eigenvalue:.12g}")
    assert eigenvalue > CONTRACTION_MARGIN, eigenvalue
    assert (1.0 - RELAXATION) + RELAXATION * eigenvalue > 1.0


def test_the_published_equilibrium_is_the_scored_post_map_state(published):
    """The published state is the map image whose unchanged gates are scored."""
    profile, equilibrium = published
    case, machine = _machine(SUITE_CELLS, True)
    operator = profile.operator
    candidate = profile.flux_map()(equilibrium.flux)
    receipt = profile.observe(candidate)

    current_moments, measure, masks, topology = (
        operator.current_moments_and_observation(candidate)
    )
    cell_current = current_moments.cell_current
    moments = observe_moments(measure, topology.flux_span)
    direct = SolvedEquilibrium(
        case=case,
        machine=machine,
        flux=candidate,
        cell_current=cell_current,
        masks=masks,
        topology=topology,
        moments=moments,
        ledger=current_ledger(cell_current, measure.masks),
        fixed_point=receipt.fixed_point,
    )

    reference = case.map_moments()
    core = np.asarray(masks.core)
    flux_deviation = float(
        np.max(np.abs(direct.grid_flux - direct.reference_flux)[core])
        / abs(case.flux_span)
    )
    axis_deviation = float(np.max(np.abs(np.asarray(topology.axis) - case.axis)))
    scale = direct.reference_scale
    plasma_current_deviation = abs(
        float(moments.plasma_current) / case.plasma_current - 1.0
    )
    poloidal_beta_deviation = abs(
        float(moments.poloidal_beta) * scale / reference["poloidal_beta"] - 1.0
    )
    internal_inductance_deviation = abs(
        float(moments.internal_inductance) * scale / reference["internal_inductance"]
        - 1.0
    )
    quadrature_deviation = abs(
        float(moments.poloidal_field_integral) / reference["field_integral"] - 1.0
    )
    field_deviation = abs(
        float(receipt.moments.poloidal_field_integral)
        / float(moments.poloidal_field_integral)
        - 1.0
    )
    grad_shafranov = float(receipt.conservation.relative_grad_shafranov)
    publication_gap = float(
        jnp.max(jnp.abs(candidate - equilibrium.flux))
        / jnp.maximum(jnp.max(jnp.abs(candidate)), 1.0e-30)
    )
    score = {
        "publication_gap": publication_gap,
        "post_map_residual": float(receipt.fixed_point.residual),
        "axis_max_deviation_m": axis_deviation,
        "flux_deviation_fraction": flux_deviation,
        "plasma_current_deviation_fraction": plasma_current_deviation,
        "poloidal_beta_deviation_fraction": poloidal_beta_deviation,
        "internal_inductance_deviation_fraction": internal_inductance_deviation,
        "analytic_field_quadrature_deviation_fraction": quadrature_deviation,
        "receipt_field_deviation_fraction": field_deviation,
        "grad_shafranov_residual": grad_shafranov,
        "relative_divergence_b": float(receipt.conservation.relative_divergence_b),
        "relative_divergence_j": float(receipt.conservation.relative_divergence_j),
    }
    print(f"post_map_score={score!r}")

    assert float(receipt.fixed_point.residual) < RESIDUAL_TOLERANCE, score
    assert axis_deviation < AXIS_TOLERANCE, score
    assert flux_deviation < FLUX_TOLERANCE, score
    assert plasma_current_deviation < PLASMA_CURRENT_TOLERANCE, score
    assert poloidal_beta_deviation < MOMENT_TOLERANCE, score
    assert internal_inductance_deviation < MOMENT_TOLERANCE, score
    assert quadrature_deviation < QUADRATURE_TOLERANCE, score
    assert field_deviation < FIELD_INTEGRAL_TOLERANCE, score
    assert grad_shafranov < GRAD_SHAFRANOV_TOLERANCE, score
    for name in ("relative_divergence_b", "relative_divergence_j"):
        assert score[name] < DIVERGENCE_MARGIN * grad_shafranov, score
    assert publication_gap < RESIDUAL_TOLERANCE, score


def test_the_relaxed_routes_leave_the_equilibrium_on_a_bounded_budget(published):
    """Fixed-budget routes remain finite while Picard exposes non-contraction.

    The expanding direction belongs to the raw fixed-point map, so the Picard
    route must expose it. Anderson applies history-dependent mixing and is not
    required to amplify the same direction. Both routes are instead required
    to consume exactly their fixed evaluation budget and keep a finite trace;
    neither is given a tolerance whose pursuit could make the test hang.
    """
    profile, equilibrium = published
    mapped = profile.flux_map()
    for scheme in (fixed_point.picard, fixed_point.anderson):
        history = scheme(
            mapped,
            equilibrium.flux,
            evaluations=RELAXED_EVALUATIONS,
            relaxation=RELAXATION,
        )
        trace = np.asarray(history.trace)
        assert trace.size == RELAXED_EVALUATIONS
        assert np.all(np.isfinite(trace)), scheme.__name__
        if scheme is fixed_point.picard:
            assert trace[-1] / trace[0] > DRIFT_GROWTH, trace[-1]


# --------------------------------------------------------------------------
# the mesh
# --------------------------------------------------------------------------
def test_the_plasma_mesh_is_hexagonal_and_wall_trimmed(solved):
    """Interior cells are whole hexagons and the wall cuts the rest."""
    machine = solved.machine
    assert machine.hexagon.sum() > 0.5 * machine.hexagon.size
    assert not machine.hexagon.all(), "no cell was trimmed by the first wall"
    whole = machine.area[machine.hexagon]
    np.testing.assert_allclose(whole, whole[0], rtol=1e-9)
    assert machine.area[~machine.hexagon].min() < whole[0]
    assert machine.stencil.shape[1] == 7
    assert machine.plasma_to_grid.shape == (len(machine.node),) * 2


def test_the_solve_reaches_its_fixed_point(solved):
    """The root find converges on the confined, diverted branch."""
    assert float(solved.fixed_point.residual) < RESIDUAL_TOLERANCE
    trace = np.asarray(solved.fixed_point.trace)
    assert np.nanmin(trace) < np.nanmax(trace)
    assert np.all(np.isfinite(np.asarray(solved.flux)))
    assert bool(solved.topology.diverted)


def test_the_relaxed_route_walks_down_the_vertical_instability(solved):
    """Relaxation loses the plasma the root find holds.

    The equilibrium is vertically unstable at fixed conductor currents, so the
    free-boundary map carries an eigenvalue outside the unit circle. Started on
    the stored map, the relaxed iteration eventually leaves the confined root
    and reaches the wall, where the source that drives current only on an
    axis-connected core switches off. The receipt records that honestly: an
    empty core and a zero ledger rather than a converged wrong answer.

    This is why the demonstration is a root find. It is also the reason a
    prescribed-source solve of a real elongated equilibrium cannot be qualified
    by a residual alone: the run leaves an equilibrium whose residual sits at
    the solve floor and then CONTRACTS onto the vacuum branch. The coupled map
    is non-normal, so contractions and bursts may occur before the branch is
    selected; the semantic statement is the nontrivial transient followed by
    contraction onto the zero-current branch, not the ordering of an arbitrary
    early window. Both are read within the trace rather than against the root
    find's residual, whose floor moves with the arithmetic the host emits.
    """
    operator = forward_operator(solved.case, solved.machine)
    history = fixed_point.picard(
        operator.flux_map(),
        seed_flux(solved.case, solved.machine),
        evaluations=RELAXED_EVALUATIONS,
        relaxation=RELAXATION,
    )
    trace = np.asarray(history.trace)
    assert trace.size == RELAXED_EVALUATIONS
    assert np.all(np.isfinite(trace))
    masks, topology = operator.read(history.state)
    assert int(np.asarray(masks.core).sum()) == 0
    assert not bool(topology.diverted)
    current = operator.source.cell_current(
        jnp.asarray(solved.machine.radius), operator.area, masks
    )
    assert float(jnp.sum(current)) == 0.0
    # the vacuum branch attracts where the equilibrium repelled: the tail of
    # the same trace contracts, at the geometric rate a relaxed step gives on
    # a fixed point it is converging to
    tail = trace[-(RELAXED_EVALUATIONS // 4) :]
    assert tail[-1] < tail[0] / 10.0, tail[[0, -1]]
    assert trace.max() > 10.0 * tail[-1]


def test_no_current_appears_outside_the_declared_support(solved):
    """Only the labelled core carries source current."""
    ledger = solved.ledger
    assert abs(float(ledger.core)) > 1.0e7
    assert float(ledger.common_sol) == 0.0
    assert float(ledger.private_flux) == 0.0
    assert float(ledger.excluded_material) == 0.0
    np.testing.assert_allclose(solved.moments.plasma_current, ledger.core, rtol=1e-12)
    counts = np.asarray(solved.masks.cell_count())
    assert counts[PlasmaDomain.CORE] > 0
    assert counts[PlasmaDomain.COMMON_SOL] > 0
    assert counts[PlasmaDomain.EXCLUDED_MATERIAL] == 0
    assert counts.sum() == solved.masks.label.size


# --------------------------------------------------------------------------
# the reproduction
# --------------------------------------------------------------------------
def test_the_solved_axis_matches_the_stored_axis(solved):
    """The topology read recovers the stored magnetic axis."""
    axis = np.asarray(solved.topology.axis)
    assert np.max(np.abs(axis - solved.case.axis)) < AXIS_TOLERANCE, axis


def test_the_solved_flux_map_matches_the_stored_map(solved):
    """The converged map reproduces the stored flux over the labelled core."""
    core = np.asarray(solved.masks.core)
    deviation = np.max(np.abs(solved.grid_flux - solved.reference_flux)[core])
    assert deviation / abs(solved.case.flux_span) < FLUX_TOLERANCE, deviation


def test_the_solved_plasma_current_matches_the_stored_current(solved):
    """The absolute source predicts the stored plasma current unaided."""
    observed = float(solved.moments.plasma_current)
    assert (
        abs(observed / solved.case.plasma_current - 1.0) < PLASMA_CURRENT_TOLERANCE
    ), observed


def test_the_published_shape_moments_disagree_with_the_stored_flux_map():
    """The reference's published scalars are not what its own map integrates.

    Recomputing the two shape moments from the stored flux map — the volume it
    encloses, the stored pressure profile over that volume, and the poloidal
    field its own gradient carries — does not return the published
    ``beta_pol`` and ``li_3``. Nothing from the package takes part in that
    quadrature, so the disagreement is internal to the reference entry, and it
    is why the solve is compared against the recomputed pair below rather than
    against the published one. The enclosed VOLUME agrees to a fraction of a
    percent, so this is a moment definition rather than a geometry error.
    """
    case = require_reference()
    moments = case.map_moments()
    assert abs(moments["volume"] / 808.1 - 1.0) < 0.01
    assert moments["poloidal_beta"] / case.poloidal_beta < 0.8
    assert moments["internal_inductance"] / case.internal_inductance > 1.1


def test_the_solved_shape_moments_match_the_stored_flux_map(solved):
    """Poloidal beta and internal inductance agree once both are referred.

    The published pair is normalised on the machine's vacuum-field reference
    radius and the observation operator's on the volume-averaged major radius
    of the labelled core, so the single ratio between them is applied before
    the comparison. The yardstick is the pair recomputed from the stored map.
    """
    scale = solved.reference_scale
    reference = solved.case.map_moments()
    for name in ("poloidal_beta", "internal_inductance"):
        observed = float(getattr(solved.moments, name)) * scale
        expected = reference[name]
        assert abs(observed / expected - 1.0) < MOMENT_TOLERANCE, (name, observed)


def test_the_observation_operator_matches_an_independent_quadrature(solved):
    """The field integral the moments use is the field the map carries.

    The internal inductance integrates the poloidal field the coupling
    operators produce at the cell centroids — an analytic field of the cell
    polygons, never a difference of the solved map. Integrating the gradient of
    the reference map on a raster instead reaches the same number by a route
    that shares no code, which is what makes the internal inductance a
    measurement rather than a restatement of the source.
    """
    reference = solved.case.map_moments()
    observed = float(solved.moments.poloidal_field_integral)
    assert abs(observed / reference["field_integral"] - 1.0) < QUADRATURE_TOLERANCE, (
        observed
    )


def test_the_solved_boundary_matches_the_stored_boundary(solved):
    """The core reaches the stored boundary on the midplane cut."""
    height = float(solved.case.axis[1])
    inboard, outboard = solved.midplane_radii(solved.masks.core, height)
    band = np.abs(solved.case.boundary[:, 1] - height) < 0.5
    stored = solved.case.boundary[band, 0]
    tolerance = BOUNDARY_PITCHES * solved.pitch
    assert abs(inboard - stored.min()) < tolerance, inboard
    assert abs(outboard - stored.max()) < tolerance, outboard


# --------------------------------------------------------------------------
# the published solve, unmodified, on the production mesh
# --------------------------------------------------------------------------
def test_the_published_solve_runs_on_the_production_mesh(published):
    """``ForwardProfile.solve`` returns a full receipt on the shipped mesh.

    Nothing about the class is adapted here: it is constructed with the
    hexagonal mesh in place of a uniform lattice and asked for the same solve.
    What that buys over driving the operator directly is the receipt layer —
    the conservation residuals, the continuation and rotation records and the
    finite check — none of which a raster-bound receipt could produce on an
    offset, wall-trimmed, variable-area mesh.
    """
    profile, equilibrium = published
    assert isinstance(profile.lattice, StencilMesh)
    assert profile.lattice.node_count == profile.operator.grid.node_number
    assert float(equilibrium.fixed_point.residual) < RESIDUAL_TOLERANCE
    scale = float(jnp.max(jnp.abs(equilibrium.flux)))
    residual = float(equilibrium.fixed_point.residual) * scale
    resolution = flux_resolution(profile, equilibrium)
    print(
        "published_solve_resolution_diagnostic="
        f"{{'residual_amplitude_wb': {residual!r}, "
        f"'arithmetic_resolution_wb': {resolution!r}, "
        f"'residual_over_resolution': {residual / resolution!r}}}"
    )
    assert bool(equilibrium.finite.passed)
    assert bool(equilibrium.topology.diverted)
    assert abs(float(equilibrium.moments.plasma_current)) > 1.0e7


def test_the_receipt_layer_reads_the_solve_without_changing_it(published, solved):
    """The published solve and the operator drive reach one flux map.

    Both hand the same seed to the same ladder, so the two flux maps agree to
    the last bit; the comparison is stated in flux units against the step the
    single-precision axis fit puts under the normalised flux, which is the
    finest difference anything downstream of the topology read can express.
    """
    profile, equilibrium = published
    resolution = flux_resolution(profile, equilibrium)
    deviation = float(jnp.max(jnp.abs(equilibrium.flux - solved.flux)))
    assert deviation < resolution, (deviation, resolution)
    for name in ("plasma_current", "volume", "major_radius"):
        published_value = float(getattr(equilibrium.moments, name))
        operator_value = float(getattr(solved.moments, name))
        assert abs(published_value / operator_value - 1.0) < ROUTE_AGREEMENT, name


def test_the_receipt_is_read_where_the_source_is_declared(published):
    """Residuals come from complete rings inside the declared support.

    The absolute source declares the core alone, so the checked set is the
    core eroded by the ring width. Every core cell has to carry a complete
    ring for that erosion to mean what it says — a core cell the mesh could
    not differentiate would silently leave the support instead of being
    trimmed from its edge.
    """
    profile, equilibrium = published
    mesh = profile.lattice
    core = np.asarray(equilibrium.domains.core)
    carries_ring = np.zeros(mesh.node_count, dtype=bool)
    carries_ring[mesh.centre] = True
    assert not (core & ~carries_ring).any(), int((core & ~carries_ring).sum())
    checked = int(equilibrium.conservation.checked_cells)
    assert 0 < checked < int(core.sum())


def test_the_conservation_receipt_qualifies_the_equilibrium(published):
    """The physical residual is small and the identical ones are far smaller.

    The Grad-Shafranov residual is the only one of the four a converged but
    wrong solve can fail, and it is read against its own drive. The two
    identically-vanishing residuals sit at the truncation floor of the ring
    fit rather than at round-off — the fitted operators do not commute — so
    what qualifies them is the margin between the two, not an absolute floor.
    """
    _profile, equilibrium = published
    ledger = equilibrium.conservation
    grad_shafranov = float(ledger.relative_grad_shafranov)
    assert grad_shafranov < GRAD_SHAFRANOV_TOLERANCE, grad_shafranov
    for name in ("relative_divergence_b", "relative_divergence_j"):
        identical = float(getattr(ledger, name))
        assert identical < DIVERGENCE_MARGIN * grad_shafranov, (name, identical)


def test_the_receipt_records_an_absolute_static_unrotated_source(published):
    """The records say the solve took no liberty with the supplied profiles.

    An absolute source carries no scalar degree of freedom, so the
    normalisation record has to report no action taken; the closure is static,
    so the rotation record carries no angular frequency; and the profiles are
    declared on the core alone, so no domain is driven under a continuation.
    Those three together are the statement that the reproduction below is of
    the supplied profiles and not of a rescaled version of them.
    """
    _profile, equilibrium = published
    assert equilibrium.normalisation.policy_name == "absolute"
    assert not bool(equilibrium.normalisation.rescaled)
    assert float(equilibrium.normalisation.amplitude) == 1.0
    assert equilibrium.rotation.closure_name == "static"
    assert not bool(equilibrium.rotation.active)
    assert not bool(equilibrium.continuation.active)


def test_the_differenced_field_agrees_with_the_analytic_cell_field(published, solved):
    """Two independent poloidal fields give the same internal inductance.

    The receipt differentiates the solved flux map on the ring fit. The
    operator path instead reads the analytic field of the cell polygons
    straight from the coupling operators — never a difference of the map. The
    two share no arithmetic, so their agreement measures the ring fit against
    a Green-function field on the mesh the field was generated on.
    """
    _profile, equilibrium = published
    receipt = float(equilibrium.moments.poloidal_field_integral)
    analytic = float(solved.moments.poloidal_field_integral)
    assert abs(receipt / analytic - 1.0) < FIELD_INTEGRAL_TOLERANCE, receipt


# --------------------------------------------------------------------------
# evidence figures
# --------------------------------------------------------------------------
#: Ink for the machine cross-section panels, following the convention the IMAS
#: plotting stack draws an equilibrium over a machine with: the computed
#: surfaces in blue, the stored boundary as a dashed sienna reference against
#: them, the wall in black, and the conductors as unfilled grey outlines so the
#: filaments a wound pack is decomposed into stay countable instead of reading
#: as one solid block. The two families that convention has no colour for get
#: their own. The vessel is violet, deliberately clear of the red the reference
#: is drawn in, because a vessel indistinguishable from the stored boundary is
#: worse than no vessel. The stabilisation pair is a green nothing else uses,
#: and is ringed as well as filled: two 13 cm plates lying against the wall are
#: a few pixels at machine scale, and a panel that claims to show every
#: conductor has to let the reader find them.
SURFACE_INK = "#3366cc"
REFERENCE_INK = "#a02c00"
WALL_INK = "#000000"
CONDUCTOR_INK = "#888888"
PASSIVE_INK = "#7b5aa6"
STABILISATION_INK = "#1b7837"


def _mesh_panel(axes, solved):
    """Draw the labelled hexagonal mesh, the solved surfaces and the boundary."""
    case, machine = solved.case, solved.machine
    core = np.asarray(solved.masks.core)
    axes.set_aspect("equal")
    marker = (72.0 * 2.2 * solved.pitch / 9.0) ** 2
    axes.scatter(
        machine.radius[~core],
        machine.node[~core, 1],
        s=marker,
        marker="h",
        c="0.90",
        linewidths=0,
    )
    axes.scatter(
        machine.radius[core],
        machine.node[core, 1],
        s=marker,
        marker="h",
        c="#b9d3ef",
        linewidths=0,
    )
    axes.tricontour(
        machine.radius,
        machine.node[:, 1],
        solved.grid_flux,
        levels=np.linspace(
            float(solved.topology.axis_flux), float(solved.topology.boundary_flux), 11
        ),
        colors="C0",
        linewidths=0.7,
    )
    axes.plot(case.wall[:, 0], case.wall[:, 1], "-", color="0.35", lw=1.1)
    closed = np.r_[case.boundary, case.boundary[:1]]
    axes.plot(closed[:, 0], closed[:, 1], "--", color="C3", lw=1.5)
    axes.plot(*np.asarray(solved.topology.axis), "o", color="C0", ms=5)
    axes.plot(*case.axis, "x", color="C3", ms=7, mew=1.6)
    axes.plot(*np.asarray(solved.topology.x_point), "o", color="C0", ms=5)
    axes.plot(*case.x_point[0], "x", color="C3", ms=7, mew=1.6)
    axes.text(3.95, 4.55, "stored boundary", color="C3", fontsize="small")
    axes.text(3.95, 4.05, "solved surfaces", color="C0", fontsize="small")
    axes.text(
        6.7,
        -4.0,
        "%d core cells\n%d open cells\n%.3f m pitch"
        % (int(core.sum()), int((~core).sum()), solved.pitch),
        fontsize="x-small",
        color="0.3",
    )
    axes.set_xlabel("$R$ [m]")
    axes.set_ylabel("$Z$ [m]")


def _reproduction_figure(figure, solved):
    """Draw the solved flux map, the stored boundary and the profile pair."""
    import matplotlib.pyplot as plt

    case, machine = solved.case, solved.machine
    grid = figure.add_gridspec(3, 3, width_ratios=(1.15, 1.0, 1.0))
    _mesh_panel(figure.add_subplot(grid[:, 0]), solved)

    for column, (values, label) in enumerate(
        (
            (case.p_prime, r"supplied $p'$ [Pa/Wb]"),
            (case.ff_prime, r"supplied $FF'$ [T$^2$m$^2$/Wb]"),
        )
    ):
        panel = figure.add_subplot(grid[0, 1 + column])
        panel.plot(case.psi_norm, values, "-", color="C3", lw=1.4)
        panel.axhline(0.0, color="0.8", lw=0.6)
        panel.set_xlabel(r"$\psi_N$")
        panel.set_title(label, fontsize="small")
        plt.setp(panel.get_yticklabels(), fontsize="x-small")

    height = float(case.axis[1])
    band = np.abs(machine.node[:, 1] - height) < solved.pitch
    order = np.argsort(machine.radius[band])
    cut = machine.radius[band][order]

    panel = figure.add_subplot(grid[1, 1:])
    panel.plot(cut, solved.grid_flux[band][order], "-", color="C0", lw=1.6)
    panel.plot(cut, solved.reference_flux[band][order], "--", color="C3", lw=1.2)
    panel.set_ylabel(r"$\Phi$ [Wb]")
    panel.set_title("midplane flux, solved against stored", fontsize="small")
    panel.tick_params(labelbottom=False)
    plt.setp(panel.get_yticklabels(), fontsize="x-small")

    panel = figure.add_subplot(grid[2, 1:])
    deviation = (solved.grid_flux - solved.reference_flux)[band][order]
    panel.plot(cut, 100.0 * deviation / solved.case.flux_span, "-", color="C0", lw=1.4)
    panel.axhline(0.0, color="0.8", lw=0.6)
    panel.set_xlabel("$R$ [m]")
    panel.set_ylabel(r"$\Delta\Phi$ [% of span]")
    plt.setp(panel.get_yticklabels(), fontsize="x-small")
    reference = case.map_moments()
    scale = solved.reference_scale
    panel.set_title(
        "$I_p$ %+.2f%%   $\\beta_p$ %+.1f%%   $l_i$ %+.1f%%   "
        "axis %.0f mm   residual %.0e"
        % (
            100.0 * (float(solved.moments.plasma_current) / case.plasma_current - 1.0),
            100.0
            * (
                float(solved.moments.poloidal_beta) * scale / reference["poloidal_beta"]
                - 1.0
            ),
            100.0
            * (
                float(solved.moments.internal_inductance)
                * scale
                / reference["internal_inductance"]
                - 1.0
            ),
            1e3 * np.max(np.abs(np.asarray(solved.topology.axis) - case.axis)),
            float(solved.fixed_point.residual),
        ),
        fontsize="small",
    )


def _instability_figure(figure, solved):
    """Draw the vertical drift the relaxed route takes off the equilibrium."""
    operator = forward_operator(solved.case, solved.machine)
    mapped = operator.flux_map()
    state = seed_flux(solved.case, solved.machine)
    radius = jnp.asarray(solved.machine.radius)
    residual, height, current = [], [], []
    for _ in range(RELAXED_EVALUATIONS):
        image = mapped(state)
        residual.append(
            float(
                jnp.max(jnp.abs(image - state))
                / jnp.maximum(jnp.max(jnp.abs(image)), 1.0e-30)
            )
        )
        masks, topology = operator.read(state)
        height.append(float(topology.axis[1]))
        current.append(
            float(jnp.sum(operator.source.cell_current(radius, operator.area, masks)))
        )
        state = state + RELAXATION * (image - state)

    step = np.arange(RELAXED_EVALUATIONS)
    upper, lower = figure.subplots(2, 1, sharex=True)
    upper.semilogy(step, residual, "-", color="C3", lw=1.5)
    upper.axhline(
        float(solved.fixed_point.residual), color="C0", lw=1.2, linestyle="--"
    )
    upper.text(
        RELAXED_EVALUATIONS * 0.55,
        float(solved.fixed_point.residual) * 2.0,
        "root find",
        color="C0",
        fontsize="small",
    )
    upper.text(
        RELAXED_EVALUATIONS * 0.1,
        max(residual) * 0.3,
        "relaxed iteration",
        color="C3",
        fontsize="small",
    )
    upper.set_ylabel("fixed-point residual")
    upper.set_title(
        "seeded on the equilibrium, the relaxed route leaves it", fontsize="small"
    )

    lower.plot(step, height, "-", color="C3", lw=1.5)
    lower.axhline(float(solved.topology.axis[1]), color="C0", lw=1.2, linestyle="--")
    lost = np.flatnonzero(np.asarray(current) == 0.0)
    if lost.size:
        lower.axvline(lost[0], color="0.6", lw=1.0)
        lower.text(
            lost[0] + 1,
            float(solved.topology.axis[1]),
            "core empty",
            color="0.4",
            fontsize="small",
            va="top",
        )
    lower.set_xlabel("map evaluation")
    lower.set_ylabel("magnetic axis $Z$ [m]")


def _machine_panel(axes, solved, title):
    """Draw one machine model and the equilibrium it supports.

    The machine and its mesh are drawn by the package's own frame plot, which
    renders every subframe element as its own polygon patch: the hexagonal
    plasma cells, the filaments each wound pack is decomposed into and the
    plates each passive loop is tiled with all appear exactly as the coupling
    integrates them, rather than as markers standing in for them.

    It is called once per conductor family, because patch keywords in that call
    apply to every polygon it draws and the four families want different ink —
    the wound packs, the vessel, the stabilisation pair and the plasma cells.
    The plasma rows are drawn FIRST and need an edge: an opaque mesh drawn last
    hides any conductor standing inside the first wall, and without an edge a
    few hundred hexagons merge into a single blob so the mesh the coupling
    integrates over stops being visible either way. The frame plot also leaves
    the axes switched off, its house style for a machine schematic; this module
    labels R and Z, so they go back on.

    The annotation sits in the upper RIGHT because the upper left is where the
    top poloidal field coil is, and a text box there whites out a conductor the
    panel exists to show.
    """
    case, machine = solved.case, solved.machine
    subframe = machine.coilset.subframe
    plasma = np.asarray(subframe.loc[:, "plasma"], dtype=bool)
    owner = np.asarray(subframe.frame, dtype=object)
    passive = ~plasma & (np.asarray(subframe.part, dtype=object) == "passive")
    pair = ~plasma & np.array([name in STABILISATION_ELEMENTS for name in owner])
    machine.coilset.plot(
        index=plasma,
        axes=axes,
        facecolor="#eaf0f8",
        edgecolor="0.72",
        linewidth=0.25,
    )
    machine.coilset.plot(
        index=~plasma & ~passive & ~pair,
        axes=axes,
        facecolor="none",
        edgecolor=CONDUCTOR_INK,
        linewidth=0.4,
    )
    if passive.any():
        machine.coilset.plot(
            index=passive, axes=axes, facecolor=PASSIVE_INK, edgecolor=PASSIVE_INK
        )
    machine.coilset.plot(
        index=pair, axes=axes, facecolor=STABILISATION_INK, edgecolor=STABILISATION_INK
    )
    axes.axis("on")
    axes.set_aspect("equal")
    for conductor in case.active:
        if conductor.name in STABILISATION_ELEMENTS:
            axes.plot(
                *conductor.polygon.mean(axis=0),
                "o",
                color=STABILISATION_INK,
                mfc="none",
                ms=9,
                mew=0.9,
            )
    axes.plot(case.wall[:, 0], case.wall[:, 1], "-", color=WALL_INK, lw=1.0)
    axes.tricontour(
        machine.radius,
        machine.node[:, 1],
        solved.grid_flux,
        levels=np.linspace(
            float(solved.topology.axis_flux), float(solved.topology.boundary_flux), 9
        ),
        colors=SURFACE_INK,
        linewidths=0.7,
    )
    closed = np.r_[case.boundary, case.boundary[:1]]
    axes.plot(closed[:, 0], closed[:, 1], "--", color=REFERENCE_INK, lw=1.6)
    axes.plot(*np.asarray(solved.topology.axis), "o", color=SURFACE_INK, ms=5)
    axes.plot(*case.axis, "x", color=REFERENCE_INK, ms=7, mew=1.6)
    axes.plot(*np.asarray(solved.topology.x_point), "o", color=SURFACE_INK, ms=5)
    axes.plot(*case.x_point[0], "x", color=REFERENCE_INK, ms=7, mew=1.6)
    rows = solved.deviations()
    axes.text(
        0.97,
        0.97,
        "%d source columns\n%d core cells\nflux %.2f %% of span\naxis %+.0f, %+.0f mm"
        % (
            machine.source_to_grid.shape[1],
            int(np.asarray(solved.masks.core).sum()),
            rows["flux sup-norm"],
            rows["axis radius"],
            rows["axis height"],
        ),
        transform=axes.transAxes,
        fontsize="x-small",
        color="0.25",
        va="top",
        ha="right",
        multialignment="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
    )
    axes.set_title(title, fontsize="small")
    axes.set_xlabel("$R$ [m]")
    axes.set_ylabel("$Z$ [m]")
    axes.tick_params(labelsize="x-small")


def _closure_figure(figure, without, structure):
    """Draw the two machine models the passive closure compares.

    Both panels are held on the SAME limits, taken from the model that reaches
    further — the one carrying the vessel. Letting each autoscale to its own
    conductors would draw the two machines at two magnifications, which is the
    one thing a before-and-after pair must not do.

    The key goes in the lower right of the left panel, the one corner of the
    machine both models leave empty.
    """
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    _machine_panel(axes[0], without, "driven coils alone")
    _machine_panel(axes[1], structure, "with the passive structure")
    key = (
        ("stored boundary", REFERENCE_INK),
        ("solved surfaces", SURFACE_INK),
        ("passive structure", PASSIVE_INK),
        ("stabilisation pair", STABILISATION_INK),
    )
    for position, (label, colour) in enumerate(reversed(key)):
        axes[0].text(
            0.98,
            0.015 + 0.04 * position,
            label,
            transform=axes[0].transAxes,
            color=colour,
            fontsize="x-small",
            ha="right",
        )
    axes[1].set_ylabel("")
    margin = 0.3
    axes[0].set_xlim(
        min(axes[1].get_xlim()[0], axes[0].get_xlim()[0]) - margin,
        max(axes[1].get_xlim()[1], axes[0].get_xlim()[1]) + margin,
    )
    axes[0].set_ylim(
        min(axes[1].get_ylim()[0], axes[0].get_ylim()[0]) - margin,
        max(axes[1].get_ylim()[1], axes[0].get_ylim()[1]) + margin,
    )


def _profile_figure(figure, without, structure):
    """Draw every deviation row of the two models, and what the closure is worth.

    The two upper panels are the reproduction itself, row by row and model by
    model. The lower one is the attribution: the flux the passive structure
    puts on the plasma cells, set against the deviation it was supposed to
    account for. It is a decade short, which is why the closure moves the
    upper panels so little.
    """
    from matplotlib.patches import Patch

    grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.85))
    model = (
        (without, "0.62", "driven coils alone"),
        (structure, "#2a6099", "with the passive structure"),
    )
    table = {label: solved.deviations() for solved, _, label in model}
    geometric = [name for name in table[model[0][2]] if name not in FRACTIONAL_ROWS]
    offset = 0.2
    for position, (names, unit) in enumerate(
        ((FRACTIONAL_ROWS, "% of reference"), (geometric, "mm"))
    ):
        panel = figure.add_subplot(grid[position, 0])
        centre = np.arange(len(names))
        for shift, (_solved, colour, label) in zip((-offset, offset), model):
            values = [table[label][name] for name in names]
            panel.barh(centre + shift, values, height=0.34, color=colour)
            for row, value in zip(centre, values):
                panel.text(
                    value,
                    row + shift,
                    " %+.2f " % value,
                    fontsize="xx-small",
                    color=colour,
                    va="center",
                    ha="left" if value >= 0.0 else "right",
                )
        if position == 0:
            # the flux deviation is the widest row here, so the key needs an
            # opaque box the legend can place clear of the bars
            panel.legend(
                handles=[
                    Patch(facecolor=colour, label=label) for _, colour, label in model
                ],
                loc="lower left",
                fontsize="x-small",
                framealpha=0.9,
                edgecolor="none",
            )
        panel.axvline(0.0, color="0.4", lw=0.8)
        panel.set_yticks(centre, names, fontsize="x-small")
        panel.invert_yaxis()
        panel.set_xlabel(unit, fontsize="x-small")
        panel.tick_params(labelsize="x-small")
        panel.margins(x=0.22)

    panel = figure.add_subplot(grid[2, 0])
    case, machine = structure.case, structure.machine
    scale = [
        (
            "passive structure at the cells",
            100.0 * float(np.max(np.abs(machine.passive_flux))) / abs(case.flux_span),
            "#2a6099",
        ),
        (
            "stored map's own residual",
            100.0 * stored_map_residual(case, machine, structure.masks.core),
            "0.45",
        ),
        (
            "reproduction deviation",
            table["with the passive structure"]["flux sup-norm"],
            "C3",
        ),
    ]
    centre = np.arange(len(scale))
    panel.barh(
        centre,
        [value for _, value, _ in scale],
        height=0.5,
        color=[colour for _, _, colour in scale],
    )
    for row, (_label, value, colour) in zip(centre, scale):
        panel.text(
            value * 1.15,
            row,
            "%.3f %%" % value,
            fontsize="x-small",
            color=colour,
            va="center",
        )
    panel.set_xscale("log")
    panel.set_yticks(centre, [label for label, _, _ in scale], fontsize="x-small")
    panel.invert_yaxis()
    panel.set_xlabel("% of the axis to boundary flux span", fontsize="x-small")
    panel.tick_params(labelsize="x-small")
    panel.margins(x=0.35)


def _receipt_figure(figure, solved, equilibrium):
    """Draw where the receipts are read and the two fields they are read with.

    The left panel is the mesh the published receipt is differentiated on: a
    cell is shaded by whether it carries a fitted derivative at all and, of
    those, whether it survives the erosion into the declared support the
    residuals are reported over. The right panels are the evidence the fit is
    a derivative operator: the conditioning of every ring against the limit
    the mesh enforces, and the poloidal field the receipt differences out of
    the solved map against the analytic field of the cell polygons on the
    midplane cut, which share no arithmetic.
    """
    import matplotlib.pyplot as plt

    from nova.equilibrium.conservation import poloidal_field

    case, machine = solved.case, solved.machine
    mesh = receipt_mesh(machine)
    core = np.asarray(equilibrium.domains.core)
    carries = np.zeros(mesh.node_count, dtype=bool)
    carries[mesh.centre] = True
    checked = np.asarray(mesh.erode(jnp.asarray(core), 2)) & np.asarray(mesh.interior())

    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
    axes = figure.add_subplot(grid[:, 0])
    axes.set_aspect("equal")
    marker = (72.0 * 2.2 * solved.pitch / 9.0) ** 2
    for mask, colour, label in (
        (~carries, "#e8b0a8", "no fitted derivative"),
        (carries & ~checked, "0.86", "differentiated"),
        (checked, "#5588bb", "residuals reported here"),
    ):
        axes.scatter(
            machine.radius[mask],
            machine.node[mask, 1],
            s=marker,
            marker="h",
            c=colour,
            linewidths=0,
            label="%s (%d)" % (label, int(mask.sum())),
        )
    axes.plot(case.wall[:, 0], case.wall[:, 1], "-", color="0.35", lw=1.1)
    closed = np.r_[case.boundary, case.boundary[:1]]
    axes.plot(closed[:, 0], closed[:, 1], "--", color="C3", lw=1.3)
    axes.legend(loc="lower left", fontsize="xx-small", frameon=False)
    axes.set_xlabel("$R$ [m]")
    axes.set_ylabel("$Z$ [m]")
    axes.set_title("where the receipt is read", fontsize="small")

    panel = figure.add_subplot(grid[0, 1])
    panel.hist(mesh.ring_condition, bins=np.logspace(0.6, 3.0, 40), color="0.55")
    panel.axvline(RING_CONDITION_LIMIT, color="C3", lw=1.2, linestyle="--")
    panel.text(
        RING_CONDITION_LIMIT * 0.85,
        panel.get_ylim()[1] * 0.6,
        "refused above",
        color="C3",
        fontsize="x-small",
        ha="right",
    )
    panel.set_xscale("log")
    panel.set_yscale("log")
    panel.set_xlabel("ring fit condition number")
    panel.set_title(
        "%d rings, worst %.0f" % (len(mesh.stencil), mesh.ring_condition.max()),
        fontsize="small",
    )
    plt.setp(panel.get_yticklabels(), fontsize="x-small")

    radial, vertical = poloidal_field(mesh, jnp.asarray(solved.grid_flux))
    fitted = np.sqrt(np.asarray(radial) ** 2 + np.asarray(vertical) ** 2)
    current_moments = forward_operator(case, machine).cell_current_moments(solved.flux)
    analytic = np.sqrt(
        np.asarray(
            machine.poloidal_field_squared(
                jnp.asarray(machine.source_current), current_moments
            )
        )
    )
    height = float(case.axis[1])
    band = (np.abs(machine.node[:, 1] - height) < solved.pitch) & carries
    order = np.argsort(machine.radius[band])
    cut = machine.radius[band][order]
    panel = figure.add_subplot(grid[1, 1])
    panel.plot(cut, analytic[band][order], "-", color="C3", lw=1.5)
    panel.plot(cut, fitted[band][order], "--", color="C0", lw=1.3)
    panel.text(
        cut[0],
        analytic[band][order].max() * 0.92,
        "cell polygons",
        color="C3",
        fontsize="x-small",
    )
    panel.text(
        cut[0],
        analytic[band][order].max() * 0.78,
        "differenced map",
        color="C0",
        fontsize="x-small",
    )
    panel.set_xlabel("$R$ [m]")
    panel.set_ylabel(r"$|B_p|$ [T]")
    receipt = float(equilibrium.moments.poloidal_field_integral)
    panel.set_title(
        r"$\int B_p^2\,\mathrm{d}V$ agrees to %+.2f %%"
        % (100.0 * (receipt / float(solved.moments.poloidal_field_integral) - 1.0)),
        fontsize="small",
    )
    plt.setp(panel.get_yticklabels(), fontsize="x-small")


def render_figures(directory: Path = FIGURE_DIRECTORY, cells: int = EVIDENCE_CELLS):
    """Write the evidence figures and return the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    solved = _solved(cells, True)
    _profile, equilibrium = _published(cells)
    written = []

    without = _solved(cells, False)
    figure = plt.figure(figsize=(7.4, 7.4), constrained_layout=True)
    _closure_figure(figure, without, solved)
    path = directory / "dina-closure-reproduction.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure = plt.figure(figsize=(7.0, 7.6), constrained_layout=True)
    _profile_figure(figure, without, solved)
    path = directory / "dina-closure-profiles.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure = plt.figure(figsize=(11.0, 7.4), constrained_layout=True)
    _reproduction_figure(figure, solved)
    path = directory / "reference-reproduction.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    _instability_figure(figure, solved)
    path = directory / "vertical-drift.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure = plt.figure(figsize=(10.4, 6.6), constrained_layout=True)
    _receipt_figure(figure, solved, equilibrium)
    path = directory / "receipt-mesh.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


if __name__ == "__main__":
    if "figures" in sys.argv[1:]:
        for written in render_figures():
            print(written)
    else:
        pytest.main([__file__])
