"""Current-moment reconstruction of the plasma boundary from external magnetics.

The plasma boundary is **externally well-posed**: outside the plasma
:math:`j_\\phi = 0`, so the poloidal flux solves the homogeneous
Grad-Shafranov operator :math:`\\Delta^* \\psi = 0`, and the external magnetics
fix a finite set of low-order current moments — the total current :math:`I_p`,
the current centroid :math:`(R, Z)` and the low-order shape moments — that set
the boundary shape. Reading the boundary off a *free* interior cell current
instead lets null-space slack (current arrangements that fit the magnetics
equally well) leak small-scale lumpiness into the near-boundary field, which
surfaces as spurious off-axis saddles and an under-sized last closed surface.

This module carries the plasma current on one nested ladder of monomial current
moments in the normalised in-plasma coordinates, indexed by
:class:`MomentOrder`:

* ``CENTROID`` — :math:`I_p` and the current centroid. Its self-sized member is
  the uniform current disc: :math:`I_p` spread evenly over a disc about the
  fitted centroid whose radius is a fixed point of its own push-out boundary
  minor radius. No shape freedom at all — the elongation comes from the real
  coil field shaping the :math:`\\psi_N = 1` contour around the disc, which is
  why the seed is robust where a free whole-vessel moment fit is not (its
  centroid swings ~0.5 m across fit order).
* ``QUADRUPOLE`` — the three degree-2 zero-sum moments (elongation and tilt).
  On top of the self-sized seed this rung is fitted to the *residual* sensor
  signature and accepted only through the over-fit gate below; degree 1 is
  skipped there because the position freedom is already spent on the centroid
  fit, so dipole terms only absorb noise and re-shift the centroid.
* ``OCTUPOLE`` — the degree-3 moments that carry triangularity.

Amplitudes are fitted to the plasma sensor signature (``measured - vacuum``) by
whitened least squares with the total current pinned hard to the Rogowski
:math:`I_p`: the monopole column is the candidate-cell mask and every higher
moment is built zero-sum, so the absolute current measurement lands on exactly
one coefficient and the remaining shape coefficients are fitted to the
:math:`I_p`-subtracted signature. Because the current carries only a handful of
moments, the external field is a well-conditioned function of precisely the
quantities the magnetics constrain.

The over-fit gate on the residual stages is machine-agnostic and referee-free:
a stage is accepted only if it moves the push-out boundary by less than
``gate_shift_fraction`` of the seed radius. Sensor misfit does not separate a
physical shape correction from an over-fit swing — the boundary shift does.

Green's columns come from the canonical axisymmetric kernels in
:mod:`nova.biot.greens` (finite-area cylinder near the section, point filament
far), so the reconstructed flux is directly comparable with every other read on
the spine. Conventions: total poloidal flux :math:`\\Phi = 2 \\pi R A_\\phi`
[Wb], explicit :math:`\\mu_0`, raw SI throughout.
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.optimize import minimize

from nova.biot.greens import hybrid_greens
from nova.equilibrium.measurement import (
    Magnetics,
    SliceMeasurement,
    whitened_solve,
)


class MomentOrder(IntEnum):
    """Maximum monomial degree of the current-moment ladder.

    The value is the degree, so ``int(order)`` indexes the monomial family and
    the members are ordered from the coarsest rung upward.
    """

    CENTROID = 1
    """Total current and current centroid — the self-sized uniform-disc seed."""

    QUADRUPOLE = 2
    """Adds elongation and tilt (the three degree-2 zero-sum moments)."""

    OCTUPOLE = 3
    """Adds triangularity (the degree-3 moments)."""


def moment_terms(order: MomentOrder | int) -> list[tuple[int, int]]:
    """Return the ``(p, q)`` monomial powers :math:`u^p v^q` with ``p + q <= order``.

    Raw monomials span the same low-order moment space as the symmetrised
    :math:`u^2 \\pm v^2` combinations but are simpler and orthogonalise cleanly
    under the whitened fit. The monopole comes first and each degree follows
    intact, so a lower rung is always a prefix of a higher one.
    """
    if int(order) < 1:
        raise ValueError(f"order must be at least MomentOrder.CENTROID, got {order}")
    return [
        (p, q)
        for degree in range(int(order) + 1)
        for p in range(degree + 1)
        for q in [degree - p]
    ]


def _term_label(p: int, q: int) -> str:
    """Return the readable ``u^p v^q`` label of one monomial."""
    if p == 0 and q == 0:
        return "1"
    parts = []
    if p:
        parts.append("u" if p == 1 else f"u^{p}")
    if q:
        parts.append("v" if q == 1 else f"v^{q}")
    return "".join(parts)


def build_moment_basis(
    r_cells: np.ndarray,
    z_cells: np.ndarray,
    candidate: np.ndarray,
    r0: float,
    *,
    order: MomentOrder | int = MomentOrder.OCTUPOLE,
    z0: float = 0.0,
    scale: float | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Return the current-moment basis on the candidate cells.

    Each column is a monomial :math:`u^p v^q` of the normalised in-plasma
    coordinates :math:`u = (R - r_0) / a`, :math:`v = (Z - z_0) / a`, restricted
    to the conductor-clear in-limiter candidate cells and zero elsewhere. A
    current ``cell_current = basis @ coefficients`` is therefore a smooth
    low-order distribution whose external field is fixed by its moments.

    Every higher moment is made zero-sum over the candidate cells by
    subtracting its mean. That leaves the span unchanged (the subtracted piece
    is a multiple of the monopole) but decouples the total-current constraint
    onto the single monopole coefficient.

    Parameters
    ----------
    r_cells, z_cells
        Cell centre coordinates [m].
    candidate
        Nonzero on the conductor-clear in-limiter cells.
    r0
        Radial centre of the normalised coordinate [m].
    order
        Maximum monomial degree.
    z0
        Vertical centre of the normalised coordinate [m].
    scale
        Normalising length :math:`a` [m]; defaults to the RMS candidate-cell
        radius about ``(r0, z0)`` — a geometry-derived scale needing no
        per-shot tuning.

    Returns
    -------
    basis
        ``(n_cells, n_terms)`` candidate-masked monomial columns.
    labels
        Readable column labels.
    scale
        The length scale actually used [m].
    """
    r = np.asarray(r_cells, dtype=np.float64)
    z = np.asarray(z_cells, dtype=np.float64)
    keep = np.asarray(candidate, dtype=np.float64) > 0.0
    if scale is None:
        if keep.any():
            scale = float(np.sqrt(np.mean((r[keep] - r0) ** 2 + (z[keep] - z0) ** 2)))
        else:  # degenerate geometry
            scale = 1.0
    scale = max(float(scale), 1e-9)

    u = (r - r0) / scale
    v = (z - z0) / scale
    columns, labels = [], []
    for p, q in moment_terms(order):
        if p == 0 and q == 0:
            column = keep.astype(np.float64)  # the monopole is the candidate mask
        else:
            column = (u**p) * (v**q)
            if keep.any():
                column = column - column[keep].mean()
            column = np.where(keep, column, 0.0)
        columns.append(column)
        labels.append(_term_label(p, q))
    return np.stack(columns, axis=1), labels, scale


@dataclass(frozen=True)
class CurrentCells:
    """Plasma current cells: centres, finite section and candidate selection.

    The finite section ``(dr, dz)`` is what keeps the Green's columns smooth
    where a sensor or grid node approaches a cell; a point filament would be
    log-singular there.
    """

    r: np.ndarray
    z: np.ndarray
    dr: float | np.ndarray = 0.05
    dz: float | np.ndarray = 0.05
    candidate: np.ndarray | None = None

    def __post_init__(self):
        """Default the candidate selection to every cell."""
        if self.candidate is None:
            object.__setattr__(self, "candidate", np.ones(self.number, float))

    @property
    def number(self) -> int:
        """Return the cell count."""
        return int(np.asarray(self.r).size)

    def section(self, index: int) -> tuple[float, float]:
        """Return the ``(dr, dz)`` section extents of one cell [m]."""
        return (
            float(np.broadcast_to(self.dr, (self.number,))[index]),
            float(np.broadcast_to(self.dz, (self.number,))[index]),
        )


def sensor_coupling(cells: CurrentCells, magnetics: Magnetics) -> np.ndarray:
    """Return the ``(n_sensor, n_cells)`` per-ampere cell-to-sensor coupling."""
    columns = []
    for index in range(cells.number):
        dr, dz = cells.section(index)
        psi, br, bz = hybrid_greens(
            np.asarray(magnetics.r, dtype=np.float64),
            np.asarray(magnetics.z, dtype=np.float64),
            float(np.asarray(cells.r)[index]),
            float(np.asarray(cells.z)[index]),
            dr,
            dz,
        )
        columns.append(magnetics.project(psi, br, bz))
    return np.column_stack(columns)


def grid_coupling(
    cells: CurrentCells, grid_r: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    """Return the ``(n_node, n_cells)`` per-ampere cell-to-grid flux coupling."""
    target_r = np.asarray(grid_r, dtype=np.float64).ravel()
    target_z = np.asarray(grid_z, dtype=np.float64).ravel()
    columns = []
    for index in range(cells.number):
        dr, dz = cells.section(index)
        psi, _br, _bz = hybrid_greens(
            target_r,
            target_z,
            float(np.asarray(cells.r)[index]),
            float(np.asarray(cells.z)[index]),
            dr,
            dz,
        )
        columns.append(psi)
    return np.column_stack(columns)


def limiter_radial_extent(
    limiter_r: np.ndarray, limiter_z: np.ndarray, height: float
) -> tuple[float, float]:
    """Return the inboard and outboard limiter radii at a given height [m].

    Falls back to the polygon bounding radii when the height clears the
    polygon, which keeps the seed sizing defined for a centroid that has left
    the vessel.
    """
    lr = np.asarray(limiter_r, dtype=np.float64)
    lz = np.asarray(limiter_z, dtype=np.float64)
    crossing = []
    for index in range(len(lr)):
        za, zb = lz[index], lz[(index + 1) % len(lr)]
        ra, rb = lr[index], lr[(index + 1) % len(lr)]
        if (za - height) * (zb - height) <= 0.0 and za != zb:
            crossing.append(ra + (height - za) / (zb - za) * (rb - ra))
    if not crossing:
        return float(lr.min()), float(lr.max())
    return float(min(crossing)), float(max(crossing))


def ring_shift_rms(
    ring: np.ndarray | None,
    other: np.ndarray | None,
    centre: tuple[float, float],
) -> float:
    """Return the RMS radial distance between two boundary rings [m].

    This is the over-fit gate metric: how far a candidate stage moved the
    boundary. A missing ring is infinitely far, so a stage that loses the
    boundary altogether can never be accepted.
    """
    if ring is None or other is None:
        return float("inf")
    angle = np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0])
    radius = np.hypot(ring[:, 0] - centre[0], ring[:, 1] - centre[1])
    other_angle = np.arctan2(other[:, 1] - centre[1], other[:, 0] - centre[0])
    other_radius = np.hypot(other[:, 0] - centre[0], other[:, 1] - centre[1])
    order = np.argsort(other_angle)
    interpolated = np.interp(
        angle, other_angle[order], other_radius[order], period=2.0 * np.pi
    )
    return float(np.sqrt(np.mean((interpolated - radius) ** 2)))


@dataclass(frozen=True)
class MomentConfig:
    """Reconstruction knobs; the defaults are the validated configuration."""

    order: MomentOrder = MomentOrder.OCTUPOLE
    """Highest rung of the ladder the free fit climbs."""

    ip_anchor: bool = True
    """Pin the total current hard to the Rogowski measurement."""

    ridge: float = 1e-8
    """Tikhonov floor on the column-normalised normal equations."""

    z0: float = 0.0
    """Vertical centre of the normalised coordinate [m]."""

    scale: float | None = None
    """Override for the geometry-derived normalising length [m]."""

    filament_dr: float = 0.05
    """Radial section of the centroid-fit trial filament [m]."""

    filament_dz: float = 0.05
    """Vertical section of the centroid-fit trial filament [m]."""

    seed_radius_fraction: float = 0.9
    """Initial seed radius as a fraction of the limiter-bounded minor distance."""

    radius_tolerance: float = 5e-3
    """Convergence tolerance of the self-sized radius fixed point [m]."""

    max_radius_iterations: int = 8
    """Iteration cap on the radius fixed point."""

    min_cells: int = 5
    """Smallest seed disc worth evaluating."""

    quadrupole_ridge: float = 1e-3
    """Column-normalised ridge on the three-moment residual stage."""

    gate_shift_fraction: float = 0.15
    """Accept a residual stage only below this boundary shift / seed radius."""


@dataclass
class MomentInversion:
    """One slice's current-moment fit: currents plus moment diagnostics."""

    cell_current: np.ndarray
    """``(n_cells,)`` fitted cell current [A]."""

    coefficients: np.ndarray
    """Moment amplitudes [A] per normalised monomial."""

    labels: list[str]
    misfit: float
    """Whitened mean-square sensor residual over the trusted rows."""

    plasma_current_fit: float
    """Total fitted current [A]."""

    plasma_current_error: float
    """Relative departure from the Rogowski current."""

    centroid_r: float
    centroid_z: float
    scale: float
    """Normalising length used [m]."""

    order: MomentOrder = MomentOrder.OCTUPOLE
    covariance: np.ndarray | None = field(default=None, repr=False)


@dataclass
class MomentReconstruction:
    """One slice's boundary reconstruction on the current-moment ladder."""

    ring: np.ndarray | None
    """``(n, 2)`` push-out boundary polygon [m], ``None`` if none was found."""

    flux: np.ndarray = field(repr=False)
    """Gauged total flux on the grid [Wb] — plasma plus coil."""

    plasma_flux: np.ndarray = field(repr=False)
    """Plasma-only flux on the grid [Wb]."""

    centroid_r: float
    centroid_z: float
    radius: float
    """Converged self-sized seed radius [m]."""

    cell_current: np.ndarray = field(repr=False)
    misfit: float
    """Whitened mean-square sensor residual of the accepted stage."""

    order: MomentOrder
    quadrupole_applied: bool
    """Whether the residual quadrupole stage passed the over-fit gate."""

    quadrupole_shift_fraction: float
    """Boundary shift of the quadrupole stage / seed radius."""

    axis_flux: float
    """Flux maximum inside the boundary [Wb] — the confined-side reference."""

    boundary_flux: float
    """Separatrix or wall-tangency flux [Wb] from the push-out."""


@dataclass
class ReconstructMoment:
    """Reconstruct the plasma boundary on a current-moment ladder.

    Holds the machine-fixed geometry — the current cells and their cached
    Green's couplings — and reconstructs one slice at a time. Build it once per
    machine description and reuse it across slices; the couplings are pure
    geometry.

    ``sensor_coupling`` and ``grid_flux_coupling`` may be supplied directly
    (the fast path when they are already cached, and how the analytic tests
    inject a conditioned coupling); otherwise they are built from
    :mod:`nova.biot.greens` on first use, which needs ``magnetics`` for the
    sensor rows and ``grid`` for the flux map.
    """

    cells: CurrentCells
    magnetics: Magnetics | None = None
    grid: object | None = None
    """Reconstruction grid supplying ``rg``, ``zg``, ``inside_limiter`` and the
    limiter polygon — the substrate the push-out boundary read runs on."""
    major_radius: float = 0.0
    """Radial centre of the normalised moment coordinate [m]."""
    config: MomentConfig = field(default_factory=MomentConfig)
    sensor_coupling: np.ndarray | None = field(default=None, repr=False)
    grid_flux_coupling: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Build whichever couplings the geometry supports and the caller omitted."""
        if self.sensor_coupling is None and self.magnetics is not None:
            self.sensor_coupling = sensor_coupling(self.cells, self.magnetics)
        if self.grid_flux_coupling is None and self.grid is not None:
            radius, height = np.meshgrid(self.grid.rg, self.grid.zg)
            self.grid_flux_coupling = grid_coupling(self.cells, radius, height)

    # --- the whitened moment fit -------------------------------------------

    def basis(self, config: MomentConfig, centre=(None, None)) -> tuple:
        """Return the moment basis about a centre, defaulting to the machine axis."""
        r0 = self.major_radius if centre[0] is None else float(centre[0])
        z0 = config.z0 if centre[1] is None else float(centre[1])
        return build_moment_basis(
            self.cells.r,
            self.cells.z,
            self.cells.candidate,
            r0,
            order=config.order,
            z0=z0,
            scale=config.scale,
        )

    def fit(
        self,
        measurement: SliceMeasurement,
        config: MomentConfig | None = None,
        centre=(None, None),
    ) -> MomentInversion:
        """Fit the moment amplitudes of one slice by whitened least squares.

        With the anchor on, the monopole is pinned to
        ``plasma_current / n_candidate`` — a trusted absolute measurement that
        also removes the trivial zero-current solution — and only the shape
        coefficients are fitted, to the :math:`I_p`-subtracted signature. The
        ridge is a numerical floor applied in a column-normalised frame so it
        does not bias the fit.
        """
        config = config or self.config
        basis, labels, scale = self.basis(config, centre)
        response = self.sensor_coupling @ basis

        weight = measurement.weight
        target = measurement.signature
        if config.ip_anchor:
            n_candidate = max(float(basis[:, 0].sum()), 1.0)
            monopole = float(measurement.plasma_current) / n_candidate
            target = target - response[:, 0] * monopole
            shape_response = response[:, 1:]
        else:
            monopole = None
            shape_response = response

        coefficients, covariance = whitened_solve(
            shape_response, target, weight, config.ridge
        )
        if config.ip_anchor:
            coefficients = np.concatenate([[monopole], coefficients])

        cell_current = basis @ coefficients
        keep = np.asarray(measurement.mask, dtype=bool)
        residual = (response @ coefficients - measurement.signature) * weight
        misfit = float((residual[keep] ** 2).sum() / max(int(keep.sum()), 1))

        plasma_current_fit = float(cell_current.sum())
        reference = float(measurement.plasma_current)
        denominator = plasma_current_fit if abs(plasma_current_fit) > 1e-12 else 1.0
        return MomentInversion(
            cell_current=cell_current,
            coefficients=coefficients,
            labels=labels,
            misfit=misfit,
            plasma_current_fit=plasma_current_fit,
            plasma_current_error=float(
                abs(plasma_current_fit - reference) / max(abs(reference), 1.0)
            ),
            centroid_r=float((cell_current * self.cells.r).sum() / denominator),
            centroid_z=float((cell_current * self.cells.z).sum() / denominator),
            scale=scale,
            order=MomentOrder(int(config.order)),
            covariance=covariance,
        )

    # --- the self-sized centroid seed --------------------------------------

    def filament_signature(self, r0: float, z0: float) -> np.ndarray:
        """Return the per-ampere sensor signature of one trial filament."""
        if self.magnetics is None:
            raise AttributeError("supply magnetics to fit a current centroid")
        psi, br, bz = hybrid_greens(
            np.asarray(self.magnetics.r, dtype=np.float64),
            np.asarray(self.magnetics.z, dtype=np.float64),
            float(r0),
            float(z0),
            self.config.filament_dr,
            self.config.filament_dz,
        )
        return self.magnetics.project(psi, br, bz)

    def fit_centroid(
        self, measurement: SliceMeasurement, config: MomentConfig | None = None
    ) -> tuple[float, float]:
        """Return the current centroid from a two-freedom filament-position fit.

        Minimises the whitened residual of a single :math:`I_p`-carrying
        filament against the coil-subtracted signature, seeded from the coarse
        ``CENTROID``-rung moment fit. Direct and robust where a free
        whole-vessel least-squares centroid is not.
        """
        config = config or self.config
        weight = measurement.weight
        target = measurement.signature
        plasma_current = float(measurement.plasma_current)

        def objective(position: np.ndarray) -> float:
            signature = self.filament_signature(position[0], position[1])
            return float(np.sum((weight * (plasma_current * signature - target)) ** 2))

        seed = self.fit(
            measurement,
            MomentConfig(
                order=MomentOrder.CENTROID,
                ip_anchor=config.ip_anchor,
                ridge=config.ridge,
                z0=config.z0,
                scale=config.scale,
            ),
        )
        solution = minimize(
            objective,
            [seed.centroid_r, seed.centroid_z],
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-6},
        )
        return float(solution.x[0]), float(solution.x[1])

    def uniform_disc(
        self, r0: float, z0: float, radius: float, plasma_current: float
    ) -> np.ndarray:
        """Return the plasma current spread evenly over a disc about a centre.

        The ``CENTROID``-rung member with no shape freedom: the boundary
        elongation comes from the coil field shaping the flux contour around
        the disc, not from the current distribution.
        """
        inside = (np.hypot(self.cells.r - r0, self.cells.z - z0) < radius) & (
            np.asarray(self.cells.candidate) > 0.0
        )
        count = int(inside.sum())
        if count < self.config.min_cells:
            raise ValueError(
                f"seed disc holds {count} cells, below min_cells="
                f"{self.config.min_cells}"
            )
        cell_current = np.zeros(self.cells.number)
        cell_current[inside] = float(plasma_current) / count
        return cell_current

    def flux_map(
        self, cell_current: np.ndarray, vacuum_flux: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the gauged total flux map [Wb] of a cell current.

        Absolutely gauged: the coil contribution is added, so the boundary flux
        is recovered by the push-out itself rather than assumed.
        """
        flux = self.grid_flux_coupling @ cell_current
        if vacuum_flux is not None:
            flux = flux + np.asarray(vacuum_flux, dtype=np.float64).ravel()
        return flux.reshape(len(self.grid.zg), len(self.grid.rg))

    def push_out(self, flux: np.ndarray, centre: tuple[float, float]):
        """Return the push-out boundary read, or ``None`` if none was found.

        One monotone flux-offset push on the gauged total flux: the level is
        swept outward from the axis and the outermost closed axis-enclosing
        contour that stays inside the limiter is the separatrix, so the boundary
        flux is recovered by the push itself. The read is nova's vmap-safe
        connectivity boundary, taken at the separatrix rather than a hair
        inside it.
        """
        from nova.equilibrium.connectivity_boundary import host_boundary_read

        read = host_boundary_read(flux, self.grid, centre, lcfs_norm=1.0)
        return read if read.found else None

    @staticmethod
    def boundary_ring(read, centre: tuple[float, float]) -> np.ndarray | None:
        """Return the boundary polygon of a push-out read about a centre."""
        if read is None:
            return None
        from nova.equilibrium.labels import LCFS_ANGLES

        radius = np.asarray(read.radii, dtype=np.float64)
        angle = np.asarray(LCFS_ANGLES, dtype=np.float64)
        return np.column_stack(
            [centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)]
        )

    def _boundary_of_disc(
        self,
        measurement: SliceMeasurement,
        centre: tuple[float, float],
        radius: float,
    ) -> tuple[float, object]:
        """Return the largest trial radius at or below ``radius`` that pushes out.

        A disc pressed against the wall leaves no closed axis-enclosing contour
        inside the limiter, so the trial radius is halved until a boundary
        closes. Backtracking here rather than tuning the initial fraction per
        machine keeps the sizing geometry-derived: the iteration finds the
        vessel-supported scale on its own.
        """
        for _ in range(self.config.max_radius_iterations):
            cell_current = self.uniform_disc(
                *centre, radius, measurement.plasma_current
            )
            read = self.push_out(
                self.flux_map(cell_current, measurement.vacuum_flux), centre
            )
            if read is not None:
                return radius, read
            radius *= 0.5
        raise ValueError("no push-out boundary found at any seed radius")

    def self_sized_seed(
        self, measurement: SliceMeasurement, centre: tuple[float, float]
    ) -> tuple[float, object]:
        """Return the self-consistent seed radius and its boundary ring.

        The seed radius is a fixed point: a disc of radius ``a`` pushes out to a
        boundary of mean minor radius ``b(a)``, and the seed is sized so
        ``a = b(a)``. Damped by averaging successive iterates, which keeps the
        iteration stable when the push-out over-shoots.
        """
        config = self.config
        limiter_inboard, limiter_outboard = limiter_radial_extent(
            self.grid.limiter_r, self.grid.limiter_z, centre[1]
        )
        minor_distance = min(centre[0] - limiter_inboard, limiter_outboard - centre[0])
        if minor_distance <= 0.0:
            raise ValueError("current centroid lies outside the limiter")

        radius = config.seed_radius_fraction * minor_distance
        read = None
        for _ in range(config.max_radius_iterations):
            radius, read = self._boundary_of_disc(measurement, centre, radius)
            minor_radius = float(np.mean(np.asarray(read.radii, dtype=np.float64)))
            updated = 0.5 * radius + 0.5 * minor_radius
            converged = abs(updated - radius) < config.radius_tolerance
            radius = updated
            if converged:
                break
        return radius, read

    def quadrupole_stage(
        self,
        measurement: SliceMeasurement,
        centre: tuple[float, float],
        radius: float,
        seed_current: np.ndarray,
    ) -> tuple[np.ndarray, object, float]:
        """Fit the quadrupole rung on the seed residual, before the gate.

        The three degree-2 zero-sum moments over the seed disc are fitted to
        the residual signature the seed leaves. Degree 1 is skipped: the
        position freedom is already spent on the centroid fit, so dipole terms
        would only absorb noise and re-shift the centroid.
        """
        basis, _labels, _scale = build_moment_basis(
            self.cells.r,
            self.cells.z,
            np.where(
                np.hypot(self.cells.r - centre[0], self.cells.z - centre[1]) < radius,
                1.0,
                0.0,
            ),
            centre[0],
            order=MomentOrder.QUADRUPOLE,
            z0=centre[1],
        )
        quadrupole = basis[:, 3:6]
        residual = measurement.signature - self.sensor_coupling @ seed_current
        coefficients, _covariance = whitened_solve(
            self.sensor_coupling @ quadrupole,
            residual,
            measurement.weight,
            self.config.quadrupole_ridge,
        )
        cell_current = seed_current + quadrupole @ coefficients
        flux = self.flux_map(cell_current, measurement.vacuum_flux)
        return cell_current, self.push_out(flux, centre), float(np.sum(coefficients**2))

    def misfit(self, measurement: SliceMeasurement, cell_current: np.ndarray) -> float:
        """Return the whitened mean-square sensor residual of a cell current."""
        keep = np.asarray(measurement.mask, dtype=bool)
        residual = (
            self.sensor_coupling @ cell_current - measurement.signature
        ) * measurement.weight
        return float((residual[keep] ** 2).sum() / max(int(keep.sum()), 1))

    def solve(
        self, measurement: SliceMeasurement, order: MomentOrder | None = None
    ) -> MomentReconstruction:
        """Reconstruct one slice's boundary by climbing the moment ladder.

        The ``CENTROID`` rung is the self-sized uniform disc about the fitted
        current centroid. At ``QUADRUPOLE`` and above the residual shape stage
        is fitted on top of it and accepted only if it moves the push-out
        boundary by less than ``gate_shift_fraction`` of the seed radius — the
        over-fit gate. The reported flux is absolutely gauged (plasma plus
        coil), so the boundary flux comes from the push-out itself.
        """
        order = MomentOrder(int(order if order is not None else self.config.order))
        centre = self.fit_centroid(measurement)
        radius, seed_read = self.self_sized_seed(measurement, centre)
        seed_current = self.uniform_disc(*centre, radius, measurement.plasma_current)
        seed_ring = self.boundary_ring(seed_read, centre)

        cell_current, read, ring = seed_current, seed_read, seed_ring
        shift_fraction = 0.0
        quadrupole_applied = False
        if order >= MomentOrder.QUADRUPOLE:
            trial_current, trial_read, _amplitude = self.quadrupole_stage(
                measurement, centre, radius, seed_current
            )
            trial_ring = self.boundary_ring(trial_read, centre)
            shift_fraction = ring_shift_rms(seed_ring, trial_ring, centre) / radius
            quadrupole_applied = shift_fraction < self.config.gate_shift_fraction
            if quadrupole_applied:
                cell_current, read, ring = trial_current, trial_read, trial_ring

        return MomentReconstruction(
            ring=ring,
            flux=self.flux_map(cell_current, measurement.vacuum_flux),
            plasma_flux=self.flux_map(cell_current),
            centroid_r=centre[0],
            centroid_z=centre[1],
            radius=radius,
            cell_current=cell_current,
            misfit=self.misfit(measurement, cell_current),
            order=order,
            quadrupole_applied=quadrupole_applied,
            quadrupole_shift_fraction=float(shift_fraction),
            axis_flux=float(read.axis_psi),
            boundary_flux=float(read.psi_bnd),
        )
