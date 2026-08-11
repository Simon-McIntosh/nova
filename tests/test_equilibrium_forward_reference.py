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
conductor currents are the reference case's own coil currents. The solve is
then asked to reproduce the reference flux map, boundary, plasma current,
poloidal beta and internal inductance from those inputs alone, and the
deviations are the measurement.

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

Two results shaped the rest of the module and are worth stating up front.

The route is a root find because the map does not contract.
    An elongated diverted column held at fixed conductor currents is
    axisymmetrically unstable, and that shows up here as an eigenvalue of the
    free-boundary map outside the unit circle: power iteration on the exact
    tangent at the converged equilibrium measures 1.25 on the suite mesh and
    1.40 at 1587 cells. A step relaxed by :math:`\beta` scales that mode by
    :math:`(1-\beta) + \beta \lambda`, so no damping rescues a relaxed route —
    seeded ON the converged state both Picard and Anderson drive the residual
    back up by three decades, and seeded on the stored map they walk the axis
    steadily downward until the column reaches the wall and the source, which
    drives current only on an axis-connected core, switches itself off. The
    Newton step solves ``(I - J) s = f`` and is indifferent to the sign of that
    eigenvalue. Both branches are pinned below, because the collapsed one
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
    same budget runs six decades further. A tolerance carried across from
    another case, or from another precision, is therefore either unreachable
    or meaningless, which is why nothing here names an absolute one.

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

import getpass
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0

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
        RING_CONDITION_LIMIT,
        StencilMesh,
        ring_condition,
    )
    from nova.frame.coilset import CoilSet
    from nova.jax.config import configure_dtypes

with skip_import("imas"):
    import imas

#: Stored equilibrium the demonstration reproduces. The scenario pulse carries
#: its own machine description — the poloidal field coil rectangles and turn
#: counts, the first wall contour, and the coil current waveforms — so geometry,
#: drive and reference come from one entry and cannot disagree about the
#: machine. The slice is a flat-top diverted burn point.
PULSE, RUN, MACHINE = 135011, 7, "iter"
DD_VERSION = "3.39.0"
TIME_SLICE = 353

#: Users the pulse store is searched under, in order. The package resolves
#: ``public`` against the shared store and any other name against that user's
#: own, so trying both covers a site store and a personal one.
CANDIDATE_USERS = ("public", getpass.getuser())

#: Target cell counts. The plasma grid is generated from a negative cell-count
#: request, so the realised count follows the wall area and the hexagon pitch:
#: -500 gives 566 cells at a 0.237 m pitch and -1500 gives 1587 at 0.137 m.
#: The suite runs the coarser of the two and the evidence figure the finer.
#: Refining does NOT reduce the deviation against the reference, which is what
#: identifies the deviation as a machine-model difference rather than a
#: discretisation error — the free-boundary residual of the STORED map under
#: this machine model is about one percent of the flux span at both.
SUITE_CELLS = -500
EVIDENCE_CELLS = -1500
#: Poloidal field coil filaments per conductor, and wall nodes PER FIRST-WALL
#: SEGMENT the limiter flux is searched on — the boundary sampler distributes
#: the request per segment, not over the whole loop, so this multiplies the
#: contour's own vertex count.
COIL_FILAMENTS = -25
WALL_NODES = 3

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

#: Agreement between the current the tabulated gradients imply on the stored
#: boundary and the stored plasma current. This is the convention pin, not a
#: solve result: it fails if the flux sense, the total-flux reading or the
#: gradient normalisation is wrong, and it is insensitive to the mesh only
#: because the integrand is smooth.
CONVENTION_TOLERANCE = 0.02
#: Agreement between the package's integral observation operator and an
#: independent quadrature of the same field on a raster.
QUADRATURE_TOLERANCE = 0.05

#: How far above the flux resolution the converged residual may sit. Reading
#: convergence against the resolution rather than against a fixed number is
#: what makes the pin follow the precision the null search fits at: the same
#: assertion holds at 0.18 of the resolution under a single-precision fit and
#: 0.011 under a double one, where the absolute residuals differ by seven
#: decades.
RESOLUTION_MARGIN = 10.0

#: Power iterations used to read the dominant eigenvalue of the free-boundary
#: map at the converged equilibrium. The Rayleigh quotient is within a percent
#: of its limit by twenty; the count is fixed rather than run to a tolerance,
#: so a map whose spectrum does not separate costs the same and cannot hang.
POWER_ITERATIONS = 40
#: How far the dominant eigenvalue has to clear one for non-contraction to be
#: a measurement rather than a rounding artefact. Measured 1.249 on the suite
#: mesh and 1.401 at 1587 cells.
CONTRACTION_MARGIN = 1.05
#: Residual growth the relaxed and mixed routes show over the bounded budget
#: when seeded on the equilibrium itself. Measured ~6.7e3 and ~8.8e3.
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
    coil_name: tuple[str, ...]
    coil_current: np.ndarray
    coil_turn: np.ndarray
    coil_geometry: np.ndarray
    unplaced: tuple[tuple[str, float], ...]
    grid_radius: np.ndarray
    grid_height: np.ndarray
    grid_flux: np.ndarray

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


def _read_reference(user: str) -> ReferenceCase:
    """Return the stored slice, its profiles and its machine description."""
    entry = imas.DBEntry(_uri(user), "r", dd_version=DD_VERSION)
    try:
        equilibrium = entry.get("equilibrium")
        coils = entry.get("pf_active")
        wall = entry.get("wall")
    finally:
        entry.close()

    slice_ = equilibrium.time_slice[TIME_SLICE]
    globals_ = slice_.global_quantities
    profiles = slice_.profiles_1d
    flux_1d = np.asarray(profiles.psi)
    separatrix = slice_.boundary_separatrix
    surface = slice_.profiles_2d[0]

    name, current, turn, geometry, unplaced = [], [], [], [], []
    for coil in coils.coil:
        conductor = float(np.asarray(coil.current.data)[TIME_SLICE])
        placed = False
        for position, element in enumerate(coil.element):
            rectangle = element.geometry.rectangle
            if float(rectangle.width) <= 0.0:
                continue
            label = str(coil.name)
            if len(coil.element) > 1:
                label = f"{label}{'UL'[position]}"
            name.append(label)
            current.append(conductor * np.sign(float(element.turns_with_sign)))
            turn.append(abs(float(element.turns_with_sign)))
            geometry.append(
                (
                    float(rectangle.r),
                    float(rectangle.z),
                    float(rectangle.width),
                    float(rectangle.height),
                )
            )
            placed = True
        if not placed:
            unplaced.append((str(coil.name), conductor))

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
        coil_name=tuple(name),
        coil_current=np.asarray(current),
        coil_turn=np.asarray(turn),
        coil_geometry=np.asarray(geometry),
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

    node: np.ndarray
    area: np.ndarray
    hexagon: np.ndarray
    stencil: np.ndarray
    wall_node: np.ndarray
    source_to_grid: np.ndarray
    plasma_to_grid: np.ndarray
    source_to_wall: np.ndarray
    plasma_to_wall: np.ndarray
    radial_field: tuple[np.ndarray, np.ndarray]
    vertical_field: tuple[np.ndarray, np.ndarray]

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

    def poloidal_field_squared(self, coil_current, cell_current) -> jnp.ndarray:
        """Return the squared poloidal field [T^2] the two sources produce."""
        radial = (
            jnp.asarray(self.radial_field[0]) @ coil_current
            + jnp.asarray(self.radial_field[1]) @ cell_current
        )
        vertical = (
            jnp.asarray(self.vertical_field[0]) @ coil_current
            + jnp.asarray(self.vertical_field[1]) @ cell_current
        )
        return radial**2 + vertical**2


def build_machine(case: ReferenceCase, cells: int) -> HexMachine:
    """Return the coilset, its hexagonal plasma mesh and their couplings.

    Every plasma cell is coupled through its own polygon: an interior cell is a
    hexagon and a cell the first wall cuts is the clipped polygon, both routed
    to the closed-form section kernel by the plasma grid constructor.

    The limiter contour is handed to the wall solve explicitly. Left to default
    it reads the plasma polygon off the SUBFRAME, which is one hexagonal cell
    rather than the first wall, and the limiter flux would then be searched
    around a single cell.
    """
    coilset = CoilSet(
        dcoil=COIL_FILAMENTS, dplasma=cells, tplasma="hex", nwall=WALL_NODES
    )
    for label, turn, (radius, height, width, thickness) in zip(
        case.coil_name, case.coil_turn, case.coil_geometry
    ):
        coilset.coil.insert(
            radius, height, width, thickness, nturn=turn, part="pf", name=label
        )
    coilset.firstwall.insert(case.wall, turn="hex")
    coilset.plasmagrid.solve()
    coilset.plasmawall.solve(boundary=case.wall)

    grid = coilset.plasmagrid.data
    limiter = coilset.plasmawall.data
    order = [str(label) for label in np.asarray(grid.coords["source"])]
    if order[:-1] != list(case.coil_name):
        raise ValueError(f"coupling column order {order} is not the coil order")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    section = np.asarray(coilset.subframe.loc[:, "section"], dtype=object)[plasma]
    return HexMachine(
        node=np.c_[np.asarray(grid.x), np.asarray(grid.z)].astype(float),
        area=np.asarray(coilset.aloc["plasma", "area"], dtype=float),
        hexagon=np.asarray([name == "hexagon" for name in section]),
        stencil=np.asarray(grid["stencil"]),
        wall_node=np.c_[np.asarray(limiter.x), np.asarray(limiter.z)].astype(float),
        source_to_grid=np.asarray(grid["Psi"])[:, :-1],
        plasma_to_grid=np.asarray(grid["Psi_"]),
        source_to_wall=np.asarray(limiter["Psi"])[:, :-1],
        plasma_to_wall=np.asarray(limiter["Psi_"]),
        radial_field=(np.asarray(grid["Br"])[:, :-1], np.asarray(grid["Br_"])),
        vertical_field=(np.asarray(grid["Bz"])[:, :-1], np.asarray(grid["Bz_"])),
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
            jnp.asarray(machine.source_to_grid),
            jnp.asarray(machine.plasma_to_grid),
            Null2D.from_coordinates(machine.node, machine.interior_stencil, maxsize=5),
        ),
        wall=FluxTarget(
            jnp.asarray(machine.source_to_wall),
            jnp.asarray(machine.plasma_to_wall),
            Null1D(jnp.asarray(machine.wall_node, dtype=jnp.float64)),
        ),
        source=forward_source(case),
        external_current=jnp.asarray(case.coil_current),
        area=jnp.asarray(machine.area),
        polarity=-1,
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
    383 core cells on the suite mesh carry no derivative, all of them at the
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


def seed_flux(case: ReferenceCase, machine: HexMachine) -> jnp.ndarray:
    """Return the stored map on the plasma cells and the wall nodes."""
    return jnp.asarray(
        np.r_[
            case.flux(machine.radius, machine.node[:, 1]),
            case.flux(machine.wall_node[:, 0], machine.wall_node[:, 1]),
        ]
    )


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
    masks, topology = operator.read(history.state)
    radius = jnp.asarray(machine.radius)
    cell_current = operator.source.cell_current(radius, operator.area, masks)
    return SolvedEquilibrium(
        case=case,
        machine=machine,
        flux=history.state,
        cell_current=cell_current,
        masks=masks,
        topology=topology,
        moments=observe_moments(
            operator.source,
            masks,
            radius,
            operator.area,
            cell_current,
            machine.poloidal_field_squared(operator.external_current, cell_current),
            topology.flux_span,
        ),
        ledger=current_ledger(cell_current, masks),
        fixed_point=history,
    )


@lru_cache(maxsize=2)
def _machine(cells: int) -> tuple[ReferenceCase, HexMachine]:
    """Return the reference and its production mesh at one resolution.

    The coupling assembly dominates the cost of this module, so the operator
    and the published solve are driven on ONE machine rather than on two
    identical ones.
    """
    configure_dtypes()
    case = require_reference()
    return case, build_machine(case, cells)


@lru_cache(maxsize=2)
def _solved(cells: int) -> SolvedEquilibrium:
    """Return the converged solve on one mesh resolution."""
    return solve(*_machine(cells))


@lru_cache(maxsize=2)
def _published(cells: int):
    """Return the published solve and its mesh at one resolution."""
    case, machine = _machine(cells)
    profile = forward_profile(case, machine)
    equilibrium = profile.solve(
        seed_flux(case, machine),
        route="newton_krylov",
        gmres_iterations=KRYLOV_ITERATIONS,
        warmup=0,
    )
    return profile, equilibrium


@pytest.fixture(scope="module")
def solved() -> SolvedEquilibrium:
    """Return the converged solve on the suite mesh."""
    return _solved(SUITE_CELLS)


@pytest.fixture(scope="module")
def published():
    """Return the ForwardProfile solve and its receipts on the suite mesh."""
    return _published(SUITE_CELLS)


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
    assert eigenvalue > CONTRACTION_MARGIN, eigenvalue
    assert (1.0 - RELAXATION) + RELAXATION * eigenvalue > 1.0


def test_the_relaxed_routes_leave_the_equilibrium_on_a_bounded_budget(published):
    """Seeded on the equilibrium itself, both mixed routes walk away from it.

    This is the non-contraction above expressed as a run rather than as an
    eigenvalue: started ON the converged state, where the residual already sits
    at the solve floor, the relaxed step and the Anderson mixing of it both
    drive it back UP by more than three decades. Neither is given a tolerance
    to chase — the budget is a fixed evaluation count — so the pin cannot hang
    on a route that was never going to converge.
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
        assert trace[-1] / trace[0] > DRIFT_GROWTH, (scheme.__name__, trace[-1])


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
    the very equilibrium the root find converged to, the relaxed iteration
    drives the axis steadily downward and the residual UPWARD until the plasma
    reaches the wall and the source, which drives current only on an
    axis-connected core, switches off. The receipt records that honestly: an
    empty core and a zero ledger rather than a converged wrong answer.

    This is why the demonstration is a root find. It is also the reason a
    prescribed-source solve of a real elongated equilibrium cannot be qualified
    by a residual alone: the run leaves an equilibrium whose residual sits at
    the solve floor and then CONTRACTS onto the vacuum branch, so the two
    directions of this one trace — growth away from the physical fixed point,
    decay onto the trivial one — are the whole statement. Both are read within
    the trace rather than against the root find's residual, which is a floor
    set by the arithmetic and moves decades with the precision the axis is
    fitted at.
    """
    operator = forward_operator(solved.case, solved.machine)
    history = fixed_point.picard(
        operator.flux_map(),
        seed_flux(solved.case, solved.machine),
        evaluations=RELAXED_EVALUATIONS,
        relaxation=RELAXATION,
    )
    trace = np.asarray(history.trace)
    early = trace[: RELAXED_EVALUATIONS // 4]
    assert early[-1] > early[0], "the relaxed residual did not grow"
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
    # the sharper statement: the root find reaches the floor its own
    # arithmetic sets, whatever precision the null search is fitting at
    scale = float(jnp.max(jnp.abs(equilibrium.flux)))
    residual = float(equilibrium.fixed_point.residual) * scale
    resolution = flux_resolution(profile, equilibrium)
    assert residual < RESOLUTION_MARGIN * resolution, (residual, resolution)
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
    analytic = np.sqrt(
        np.asarray(
            machine.poloidal_field_squared(
                jnp.asarray(case.coil_current), solved.cell_current
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
    solved = _solved(cells)
    _profile, equilibrium = _published(cells)
    written = []

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
