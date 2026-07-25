"""Plasma screening circuit -- the plasma as a dynamic filament system.

The passive-structure treatment applied to the plasma itself.  The core region is
tiled into finite-area patches (groups of in-limiter grid cells), the patch mutual
matrix comes from the same axisymmetric finite-area kernel as everything else (the
grid's cached cell Green's matrix -- no new kernels), patch resistances come from
the bounded :class:`~nova.transport.current_diffusion.EtaProfile` family at each
patch's normalised flux, and the L/R eigenmodes form the dynamic screening state.

Physics carried here and nowhere else in the classical spine:

* during a fast flux swing the INDUCTANCE structure alone concentrates the
  incremental current in the outermost patches -- the outer plasma shields the
  core.  The skin effect IS circuit screening: applying a loop voltage to a
  nested system puts ``di/dt`` proportional to ``L^-1 1``, which is almost
  entirely on the outermost conductor;
* the current then penetrates inward on the local resistive time, so the filament
  resistances -- the resistivity closure -- set the decay of every screening mode.
  Ramp transients constrain the resistivity exactly the way coil-only transients
  constrain the passive-structure resistances.

In the flux-surface-averaged nested limit this circuit IS the 1D flux-diffusion
operator (:func:`~nova.transport.current_diffusion.diffuse_psi`), pinned by test
on a shared analytic nested-circle case, so the current dynamics and the flux
diffusion are one system with one unknown.

Screening modes are built in the ZERO-NET-CURRENT subspace: a measured plasma
current already pins the total, so the free dynamics live in the redistribution
directions and a mode added to a per-slice fit can never fight the current anchor.

Every construction rule is machine-agnostic: patches are binned in the
dimensionless ``(sqrt(psi_n), poloidal angle)`` coordinates, and every coupling is
the analytic kernel -- a uniform geometric rescale leaves the mode shapes
invariant and maps ``tau -> tau s^2`` exactly (``L`` scales with ``s``, ``R`` with
``1/s``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nova.biot.greens import MU0
from nova.circuit.linkage import guard_positive_definite
from nova.circuit.propagate import integrate_eddy_ode


@dataclass(frozen=True)
class CoreCells:
    """In-limiter grid cells carrying plasma current, in Green's-matrix order.

    ``index`` are the cells' positions in the equilibrium grid's own flat order --
    the row/column order every Green's matrix uses -- and ``r``/``z`` their
    centres [m].  ``area`` is the (uniform) cell area [m^2]; a cell's current is
    its current density times that area.
    """

    index: np.ndarray
    r: np.ndarray
    z: np.ndarray
    area: float

    @property
    def n_cells(self) -> int:
        """Return the number of in-limiter cells."""
        return int(self.index.size)


@dataclass(frozen=True)
class PatchTiling:
    """Grouping of core grid cells into plasma patches.

    ``cell_position`` indexes into :attr:`CoreCells.index`; ``owner`` maps each
    tiled cell to its patch; ``share`` is the cell's current share within its
    patch (uniform current density, so the shares sum to one per patch).
    ``psi_n``, ``r`` and ``z`` are share-weighted patch centroids.
    """

    cell_position: np.ndarray
    owner: np.ndarray
    share: np.ndarray
    psi_n: np.ndarray
    r: np.ndarray
    z: np.ndarray

    @property
    def n_patches(self) -> int:
        """Return the number of patches."""
        return int(self.psi_n.size)

    def cell_matrix(self, n_cells: int) -> np.ndarray:
        """Share matrix mapping patch currents to cell currents [A]."""
        matrix = np.zeros((n_cells, self.n_patches))
        matrix[self.cell_position, self.owner] = self.share
        return matrix

    def bin_cell_currents(self, cell_currents: np.ndarray) -> np.ndarray:
        """Patch currents [A] from in-limiter cell currents."""
        selected = np.asarray(cell_currents, dtype=np.float64)[self.cell_position]
        return np.bincount(self.owner, weights=selected, minlength=self.n_patches)


def tile_core_patches(
    cells: CoreCells,
    psi_n_cell: np.ndarray,
    in_core: np.ndarray,
    axis: tuple[float, float],
    *,
    n_rad: int = 10,
    n_pol: int = 8,
) -> PatchTiling:
    """Tile the axis-connected core into radial-by-poloidal patches.

    Radial bins are uniform in ``sqrt(psi_n)``, which is approximately the minor
    radius, so the edge screening layer is resolved in physical depth; poloidal
    bins are uniform in the angle about the magnetic axis.  The innermost radial
    bin stays a single patch: the axis region has no meaningful angle.  Both
    coordinates are dimensionless, so the rule transfers across machines and grids
    unchanged.  Empty bins are dropped.
    """
    in_core = np.asarray(in_core, dtype=bool)
    psi_n_cell = np.asarray(psi_n_cell, dtype=np.float64)
    positions = np.flatnonzero(in_core)
    if positions.size == 0:
        raise ValueError("the core selection holds no in-limiter cells")
    psi_n = np.clip(psi_n_cell[positions], 0.0, 1.0)
    radius = cells.r[positions]
    height = cells.z[positions]

    i_radial = np.minimum((np.sqrt(psi_n) * n_rad).astype(int), n_rad - 1)
    angle = np.arctan2(height - axis[1], radius - axis[0])
    i_poloidal = np.minimum(
        ((angle + np.pi) / (2.0 * np.pi) * n_pol).astype(int), n_pol - 1
    )
    bin_id = np.where(i_radial == 0, 0, 1 + (i_radial - 1) * n_pol + i_poloidal)

    unique, owner = np.unique(bin_id, return_inverse=True)
    n_patches = unique.size
    counts = np.bincount(owner, minlength=n_patches).astype(np.float64)
    share = 1.0 / counts[owner]
    return PatchTiling(
        cell_position=positions,
        owner=owner,
        share=share,
        psi_n=np.bincount(owner, weights=share * psi_n, minlength=n_patches),
        r=np.bincount(owner, weights=share * radius, minlength=n_patches),
        z=np.bincount(owner, weights=share * height, minlength=n_patches),
    )


@dataclass(frozen=True)
class PlasmaCircuit:
    """Patch-space L/R system of the tiled plasma region.

    ``lmat`` is the two-section flux linkage [Wb/A] -- the grid's own cached cell
    Green's matrix contracted with the patch share matrix, symmetrised and
    positive-definite-guarded exactly as the passive circuit build.  Resistance is
    per-cell exact: ``sum(share^2 2 pi r eta(psi_n) / area)`` over a patch's cells,
    parallel toroidal paths at fixed shares.  ``m_channel`` are the drive-channel
    flux linkages [Wb/A].
    """

    tiling: PatchTiling
    lmat: np.ndarray
    m_channel: np.ndarray
    cell_r: np.ndarray
    cell_psi_n: np.ndarray
    cell_area: float
    n_cells: int

    @property
    def n_patches(self) -> int:
        """Return the number of patches."""
        return self.tiling.n_patches

    def cell_matrix(self) -> np.ndarray:
        """Share matrix over the FULL in-limiter cell set (Green's-matrix order)."""
        return self.tiling.cell_matrix(self.n_cells)

    def r_diag(self, eta) -> np.ndarray:
        """Diagonal patch resistances [Ohm] at the given resistivity closure."""
        resistivity = np.asarray(eta(self.cell_psi_n), dtype=np.float64)
        per_cell = 2.0 * np.pi * self.cell_r * resistivity / self.cell_area
        return np.bincount(
            self.tiling.owner,
            weights=self.tiling.share**2 * per_cell,
            minlength=self.n_patches,
        )


def build_plasma_circuit(
    cells: CoreCells,
    tiling: PatchTiling,
    psi_n_cell: np.ndarray,
    cell_greens_psi: np.ndarray,
    *,
    channel_psi_columns: np.ndarray | None = None,
) -> PlasmaCircuit:
    """Assemble the patch L/R system from the grid's own analytic kernels.

    ``cell_greens_psi`` ``(n_cells, n_cells)`` is the per-ampere total flux at
    every in-limiter cell from a unit current in each in-limiter cell -- the
    finite-area kernel throughout, in :attr:`CoreCells.index` order.
    ``channel_psi_columns`` ``(n_cells, n_channels)`` are the drive channels' flux
    at those cells, share-averaged onto the patches.  ``psi_n_cell`` is where the
    resistivity closure is evaluated.
    """
    share_matrix = tiling.cell_matrix(cells.n_cells)
    lmat = share_matrix.T @ np.asarray(cell_greens_psi) @ share_matrix
    lmat = guard_positive_definite(0.5 * (lmat + lmat.T), floor_frac=1e-6)

    positions = tiling.cell_position
    if channel_psi_columns is None or np.shape(channel_psi_columns)[1] == 0:
        m_channel = np.zeros((tiling.n_patches, 0))
    else:
        columns = np.asarray(channel_psi_columns, dtype=np.float64)[positions, :]
        m_channel = np.column_stack(
            [
                np.bincount(
                    tiling.owner,
                    weights=tiling.share * columns[:, channel],
                    minlength=tiling.n_patches,
                )
                for channel in range(columns.shape[1])
            ]
        )
    return PlasmaCircuit(
        tiling=tiling,
        lmat=lmat,
        m_channel=m_channel,
        cell_r=cells.r[positions],
        cell_psi_n=np.clip(
            np.asarray(psi_n_cell, dtype=np.float64)[positions], 0.0, 1.0
        ),
        cell_area=float(cells.area),
        n_cells=cells.n_cells,
    )


def build_plasma_circuit_from_state(
    cells: CoreCells,
    psi_n_cell: np.ndarray,
    in_core: np.ndarray,
    axis: tuple[float, float],
    cell_greens_psi: np.ndarray,
    *,
    n_rad: int = 10,
    n_pol: int = 8,
    channel_psi_columns: np.ndarray | None = None,
) -> PlasmaCircuit:
    """Tile and assemble in one step from an equilibrium's core read."""
    tiling = tile_core_patches(
        cells, psi_n_cell, in_core, axis, n_rad=n_rad, n_pol=n_pol
    )
    return build_plasma_circuit(
        cells,
        tiling,
        psi_n_cell,
        cell_greens_psi,
        channel_psi_columns=channel_psi_columns,
    )


# ---------------------------------------------------------------------------
# exact-ZOH evolution
# ---------------------------------------------------------------------------
def circuit_eigensystem(circuit: PlasmaCircuit, eta) -> tuple[np.ndarray, np.ndarray]:
    """Unconstrained eigensystem ``(tau, v)`` of ``R v = (1/tau) L v``.

    ``v`` is L-orthonormal, so patch currents map to mode amplitudes by
    ``a = v' L i`` and back by ``i = v a``.
    """
    from scipy.linalg import eigh

    rate, vectors = eigh(np.diag(circuit.r_diag(eta)), circuit.lmat)
    return 1.0 / np.clip(rate, 1e-12, None), vectors


def _evolve_lr_system(
    lmat: np.ndarray,
    r_vec: np.ndarray,
    m_channel: np.ndarray,
    volt_pattern: np.ndarray,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_channel_of_t: np.ndarray | None,
) -> np.ndarray:
    """Exact-ZOH evolution of a diagonal-resistance L/R system.

    Dynamics ``L di/dt + R i = u(t) p - M dI_c/dt`` with ``p`` the voltage-drive
    pattern -- which conductors the loop voltage acts on.  In the L-orthonormal
    eigenbasis both drive terms are exactly integrable per step (piecewise-constant
    voltage, piecewise-linear flux: the same zero-order-hold contract as the
    passive structure), so the integration is EXACT for those drive classes at any
    step size.  Returns the current state ``(n_t, n_circuits)``.
    """
    from scipy.linalg import eigh

    times = np.asarray(times, dtype=np.float64)
    voltage = np.asarray(loop_voltage, dtype=np.float64)
    rate, vectors = eigh(np.diag(np.asarray(r_vec, dtype=np.float64)), lmat)
    tau = 1.0 / np.clip(rate, 1e-12, None)
    pattern = vectors.T @ np.asarray(volt_pattern, dtype=np.float64)
    volt_mode = voltage[:, np.newaxis] * pattern[np.newaxis, :]
    if i_channel_of_t is not None and m_channel.shape[1]:
        psi_mode = (
            np.asarray(i_channel_of_t, dtype=np.float64) @ (vectors.T @ m_channel).T
        )
    else:
        psi_mode = np.zeros((times.size, tau.size))
    initial = vectors.T @ (lmat @ np.asarray(i0, dtype=np.float64))
    state, _swing = integrate_eddy_ode(
        tau, times, psi_mode, initial=initial, voltage_mode=volt_mode
    )
    return state @ vectors.T


def evolve_patch_currents(
    circuit: PlasmaCircuit,
    eta,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_channel_of_t: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH evolution of the full patch-current state ``(n_t, n_patches)``.

    The loop voltage is identical for every toroidal loop enclosing the solenoid,
    so a central-solenoid drive is exactly the uniform pattern; ``i_channel_of_t``
    is the drive-current history whose swing drives the mode flux.
    """
    return _evolve_lr_system(
        circuit.lmat,
        circuit.r_diag(eta),
        circuit.m_channel,
        np.ones(circuit.n_patches),
        times,
        i0=i0,
        loop_voltage=loop_voltage,
        i_channel_of_t=i_channel_of_t,
    )


def steady_state_currents(circuit: PlasmaCircuit, eta, ip_amperes: float) -> np.ndarray:
    """Fully-penetrated (resistive steady-state) patch currents at total current.

    Under a constant loop voltage the steady state is current proportional to
    CONDUCTANCE -- the equilibrated profile the plasma reaches once the ramp is
    slow against the resistive time.
    """
    conductance = 1.0 / circuit.r_diag(eta)
    return conductance * (float(ip_amperes) / conductance.sum())


def loop_voltage_for_ip(
    circuit: PlasmaCircuit,
    eta,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    ip_target: float,
    i_channel_of_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Constant loop voltage over the interval that lands the total current.

    The system is linear in the applied voltage, so two exact integrations -- the
    free response and the unit-voltage response -- determine it in closed form.
    Returns ``(voltage, i_end)``.
    """
    times = np.asarray(times, dtype=np.float64)
    free = evolve_patch_currents(
        circuit,
        eta,
        times,
        i0=i0,
        loop_voltage=np.zeros(times.size),
        i_channel_of_t=i_channel_of_t,
    )[-1]
    unit = evolve_patch_currents(
        circuit,
        eta,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
    )[-1]
    denominator = float(unit.sum())
    if abs(denominator) < 1e-30:
        raise ValueError("degenerate voltage response -- check the circuit")
    voltage = (float(ip_target) - float(free.sum())) / denominator
    return voltage, free + voltage * unit


# ---------------------------------------------------------------------------
# coupled system: plasma patches + fixed external conductors, ONE circuit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledCircuit:
    """Plasma patches plus fixed external conductors as ONE L/R system.

    The full dynamic content of the machine in a single block circuit: the plasma
    patch block, the external-conductor block, and the exact cross-linkage between
    them, so the flux diffusion, the passive structures and the plasma
    redistribution evolve together and every ``dPhi/dt`` term -- drive swing,
    structure eddies, plasma back-reaction -- is carried by one integration.  The
    loop voltage acts on the plasma rows only: the solenoid electromotive force
    drives the loops the plasma closes, while the structure is passive.
    """

    plasma: PlasmaCircuit
    lmat: np.ndarray
    m_channel: np.ndarray
    r_external: np.ndarray

    @property
    def n_plasma(self) -> int:
        """Return the number of plasma patch rows."""
        return self.plasma.n_patches

    @property
    def n_external(self) -> int:
        """Return the number of external conductor rows."""
        return int(self.r_external.size)

    @property
    def n_total(self) -> int:
        """Return the total number of circuit rows."""
        return int(self.lmat.shape[0])

    def r_vec(self, eta) -> np.ndarray:
        """Return the full diagonal resistance vector [Ohm]."""
        return np.concatenate([self.plasma.r_diag(eta), self.r_external])

    def volt_pattern(self) -> np.ndarray:
        """Return the loop-voltage drive pattern (plasma rows only)."""
        return np.concatenate([np.ones(self.n_plasma), np.zeros(self.n_external)])


def patch_external_linkage(tiling: PatchTiling, psi_columns: np.ndarray) -> np.ndarray:
    """Flux each patch links per ampere of each external circuit [Wb/A].

    ``psi_columns`` ``(n_cells, n_external)`` are the external circuits' flux at
    the in-limiter cells (the finite-area kernel -- for instance a passive set's
    grid columns restricted to those cells); the patch linkage is the share-weighted
    average over the patch's cells, exactly the drive-linkage construction.
    """
    columns = np.asarray(psi_columns, dtype=np.float64)[tiling.cell_position, :]
    return np.column_stack(
        [
            np.bincount(
                tiling.owner,
                weights=tiling.share * columns[:, external],
                minlength=tiling.n_patches,
            )
            for external in range(columns.shape[1])
        ]
    )


def build_coupled_circuit(
    circuit: PlasmaCircuit,
    *,
    l_external: np.ndarray,
    r_external: np.ndarray,
    m_external_channel: np.ndarray,
    m_patch_external: np.ndarray,
) -> CoupledCircuit:
    """Assemble the plasma-plus-external block system.

    ``m_external_channel`` must be in the SAME drive-channel order as
    ``circuit.m_channel``; reciprocity supplies the transposed cross block.  The
    block linkage is symmetrised and positive-definite-guarded as each sub-block
    build is.
    """
    n_plasma = circuit.n_patches
    n_external = int(np.asarray(r_external).size)
    total = n_plasma + n_external
    lmat = np.zeros((total, total))
    lmat[:n_plasma, :n_plasma] = circuit.lmat
    lmat[:n_plasma, n_plasma:] = m_patch_external
    lmat[n_plasma:, :n_plasma] = np.asarray(m_patch_external, dtype=np.float64).T
    lmat[n_plasma:, n_plasma:] = l_external
    return CoupledCircuit(
        plasma=circuit,
        lmat=guard_positive_definite(0.5 * (lmat + lmat.T), floor_frac=1e-8),
        m_channel=np.vstack(
            [circuit.m_channel, np.asarray(m_external_channel, dtype=np.float64)]
        ),
        r_external=np.asarray(r_external, dtype=np.float64),
    )


def evolve_coupled(
    coupled: CoupledCircuit,
    eta,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    loop_voltage: np.ndarray,
    i_channel_of_t: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH evolution of the coupled plasma-plus-external state."""
    return _evolve_lr_system(
        coupled.lmat,
        coupled.r_vec(eta),
        coupled.m_channel,
        coupled.volt_pattern(),
        times,
        i0=i0,
        loop_voltage=loop_voltage,
        i_channel_of_t=i_channel_of_t,
    )


def coupled_loop_voltage_for_ip(
    coupled: CoupledCircuit,
    eta,
    times: np.ndarray,
    *,
    i0: np.ndarray,
    ip_target: float,
    i_channel_of_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Constant loop voltage landing the total over the PLASMA rows.

    The same closed-form two-integration linearity as :func:`loop_voltage_for_ip`,
    with the total taken over the plasma rows only: the external rows carry eddies,
    not plasma current.  Returns ``(voltage, i_end)`` with the full end state.
    """
    times = np.asarray(times, dtype=np.float64)
    n_plasma = coupled.n_plasma
    free = evolve_coupled(
        coupled,
        eta,
        times,
        i0=i0,
        loop_voltage=np.zeros(times.size),
        i_channel_of_t=i_channel_of_t,
    )[-1]
    unit = evolve_coupled(
        coupled,
        eta,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.ones(times.size),
    )[-1]
    denominator = float(unit[:n_plasma].sum())
    if abs(denominator) < 1e-30:
        raise ValueError("degenerate voltage response -- check the coupled circuit")
    voltage = (float(ip_target) - float(free[:n_plasma].sum())) / denominator
    return voltage, free + voltage * unit


def mean_plasma_linked_flux(
    coupled: CoupledCircuit,
    i_state: np.ndarray,
    i_channel: np.ndarray | None = None,
) -> float:
    """Patch-mean linked flux of the plasma rows [Wb].

    The scalar whose balance ``dPsi/dt + mean(R i) = u`` integrates the per-patch
    circuit equation.  With the state integrated from zero (a vacuum start) the
    flux ledger closes with no free constant, the remaining terms being the
    frozen-geometry chain's carry of the shape and inductance change.
    """
    n_plasma = coupled.n_plasma
    flux = coupled.lmat[:n_plasma, :] @ np.asarray(i_state, dtype=np.float64)
    if i_channel is not None and coupled.m_channel.shape[1]:
        flux = flux + coupled.m_channel[:n_plasma, :] @ np.asarray(
            i_channel, dtype=np.float64
        )
    return float(flux.mean())


def plasma_self_inductance(
    r_axis: np.ndarray,
    minor_radius: np.ndarray,
    elongation: np.ndarray | float = 1.0,
    internal_inductance: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Large-aspect self-inductance of the plasma ring [H].

    ``L = mu0 R (ln(8R / (a sqrt(kappa))) - 2 + li/2)`` -- the external term at the
    elongation-corrected effective minor radius plus the internal-inductance share.
    All arguments broadcast, so a measured ``(R, a, kappa, li)`` TRACE yields
    ``L(t)`` directly: the ``dL/dt`` of a growing, shifting, shaping column is as
    much a flux-balance term as the moving centroid, and both must be carried for
    the applied loop voltage to come out right.  Returns NaN where the large-aspect
    form does not apply.
    """
    r_axis = np.asarray(r_axis, dtype=np.float64)
    minor_radius = np.asarray(minor_radius, dtype=np.float64)
    elongation = np.asarray(elongation, dtype=np.float64)
    internal_inductance = np.asarray(internal_inductance, dtype=np.float64)
    effective = minor_radius * np.sqrt(elongation)
    with np.errstate(divide="ignore", invalid="ignore"):
        inductance = (
            MU0
            * r_axis
            * (np.log(8.0 * r_axis / effective) - 2.0 + 0.5 * internal_inductance)
        )
    return np.where(
        (r_axis > 0.0) & (effective > 0.0) & (effective < r_axis * 8.0),
        inductance,
        np.nan,
    )


# ---------------------------------------------------------------------------
# screening eigenbasis: zero-net-current modes for the per-slice fit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScreeningBasis:
    """Zero-net-current L/R eigenmodes of the plasma circuit.

    ``tau`` are the resistive decay times [s] -- the only place the resistivity
    closure enters; ``v`` ``(n_patches, n_modes)`` the L-orthonormal patch-current
    patterns, each with exactly zero net current, since the measured plasma current
    pins the total and the free screening dynamics live in the redistribution
    subspace.  ``i_cell`` expands a unit mode amplitude to in-limiter cell currents
    [A]; ``a_sensor`` and ``psi_grid`` are the sensor and full-grid flux columns per
    unit amplitude (``psi_grid`` is empty unless a grid flux solver is supplied).
    ``lmat`` and ``r_diag`` are retained for the backbone drive terms of
    :func:`screening_trajectory`.
    """

    tau: np.ndarray
    v: np.ndarray
    i_cell: np.ndarray
    a_sensor: np.ndarray
    psi_grid: np.ndarray
    m_channel: np.ndarray
    lmat: np.ndarray
    r_diag: np.ndarray
    tiling: PatchTiling

    @property
    def n_modes(self) -> int:
        """Return the number of modes retained."""
        return int(self.tau.size)


def zero_net_current_subspace(n_patches: int) -> np.ndarray:
    """Orthonormal basis of the zero-net-current directions of a patch set."""
    projector = np.eye(n_patches) - np.full((n_patches, n_patches), 1.0 / n_patches)
    basis, _upper = np.linalg.qr(projector)
    return basis[:, : n_patches - 1]


def screening_eigenbasis(
    circuit: PlasmaCircuit,
    eta,
    g_sensor: np.ndarray,
    *,
    n_modes: int = 2,
    sensor_scale: np.ndarray | None = None,
    grid_flux=None,
) -> ScreeningBasis:
    """Reduce the plasma circuit to its slowest zero-net-current modes.

    The net-current direction is deflated BEFORE the eigensolve -- the generalised
    problem is restricted to the subspace of zero-sum current vectors -- so every
    kept mode is a pure radial or poloidal redistribution at fixed total current.
    Modes are ranked by decay time, slowest first, because a slow mode the sensors
    can see is exactly the history a per-slice fit cannot absorb; with
    ``sensor_scale`` the ranking becomes the whitened relevance
    ``tau ||a_sensor / scale||`` the passive reduction uses.

    ``g_sensor`` ``(n_sensors, n_cells)`` maps in-limiter cell currents to sensor
    readings.  ``grid_flux``, when given, is called with the mode's cell currents
    and must return that mode's flux on the full grid -- the equilibrium solver's
    own Dirichlet solve, which is the identical path plasma current takes into the
    field.  Without it ``psi_grid`` is empty and the sensor columns still stand.
    """
    from scipy.linalg import eigh

    subspace = zero_net_current_subspace(circuit.n_patches)
    r_diag = circuit.r_diag(eta)
    rate, v_reduced = eigh(
        subspace.T @ np.diag(r_diag) @ subspace,
        subspace.T @ circuit.lmat @ subspace,
    )
    tau = 1.0 / np.clip(rate, 1e-12, None)
    v_full = subspace @ v_reduced  # L-orthonormal, zero net current

    i_cell_all = circuit.cell_matrix() @ v_full
    a_sensor_all = np.asarray(g_sensor, dtype=np.float64) @ i_cell_all

    if sensor_scale is None:
        keep = np.argsort(tau)[::-1][: int(n_modes)]
    else:
        from nova.circuit.passive import select_modes

        keep = select_modes(tau, a_sensor_all, sensor_scale, n_modes)

    i_cell = i_cell_all[:, keep]
    psi_grid = (
        np.zeros((0, keep.size))
        if grid_flux is None
        else np.column_stack([grid_flux(i_cell[:, mode]) for mode in range(keep.size)])
    )
    return ScreeningBasis(
        tau=tau[keep],
        v=v_full[:, keep],
        i_cell=i_cell,
        a_sensor=a_sensor_all[:, keep],
        psi_grid=psi_grid,
        m_channel=v_full[:, keep].T @ circuit.m_channel,
        lmat=circuit.lmat,
        r_diag=r_diag,
        tiling=circuit.tiling,
    )


def screening_trajectory(
    basis: ScreeningBasis,
    times: np.ndarray,
    *,
    i_channel_of_t: np.ndarray | None = None,
    i_backbone_patch: np.ndarray | None = None,
    psi_extra_mode: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH screening-mode amplitudes along a drive history.

    Decomposing the patch state into a backbone plus a screening deviation, the
    mode dynamics are ``da/dt + a/tau = -dPsi_m/dt - v' R i_backbone`` with the
    linked flux ``Psi_m = v'(M i_channel + L i_backbone)`` -- the drive swing plus
    the backbone's own flux history, which avoids the fixed-mutual approximation
    exactly as the passive trajectory does, and the resistive term is the
    electromotive force a non-solution backbone leaves behind.

    ``i_backbone_patch`` ``(n_t, n_patches)`` is the backbone patch-current history
    (a first-pass fit binned onto the tiling); ``None`` drops the backbone terms.
    ``psi_extra_mode`` ``(n_t, n_modes)`` adds linked flux from conductors the drive
    channels do not carry -- a predicted structure-eddy history, for instance -- so
    all of the machine's ``dPhi/dt`` terms enter the mode dynamics.
    """
    times = np.asarray(times, dtype=np.float64)
    psi_mode = np.zeros((times.size, basis.n_modes))
    volt_mode = np.zeros((times.size, basis.n_modes))
    if i_channel_of_t is not None and basis.m_channel.shape[1]:
        psi_mode += np.asarray(i_channel_of_t, dtype=np.float64) @ basis.m_channel.T
    if i_backbone_patch is not None:
        backbone = np.asarray(i_backbone_patch, dtype=np.float64)
        psi_mode += backbone @ (basis.v.T @ basis.lmat).T
        volt_mode -= backbone @ (basis.v.T @ np.diag(basis.r_diag)).T
    if psi_extra_mode is not None:
        psi_mode += np.asarray(psi_extra_mode, dtype=np.float64)
    state, _swing = integrate_eddy_ode(
        basis.tau, times, psi_mode, voltage_mode=volt_mode
    )
    return state


__all__ = [
    "CoreCells",
    "CoupledCircuit",
    "PatchTiling",
    "PlasmaCircuit",
    "ScreeningBasis",
    "build_coupled_circuit",
    "build_plasma_circuit",
    "build_plasma_circuit_from_state",
    "circuit_eigensystem",
    "coupled_loop_voltage_for_ip",
    "evolve_coupled",
    "evolve_patch_currents",
    "loop_voltage_for_ip",
    "mean_plasma_linked_flux",
    "patch_external_linkage",
    "plasma_self_inductance",
    "screening_eigenbasis",
    "screening_trajectory",
    "steady_state_currents",
    "tile_core_patches",
    "zero_net_current_subspace",
]
