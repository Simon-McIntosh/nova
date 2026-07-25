"""Plasma screening circuit -- the plasma as a dynamic filament system.

Pinned on an analytic nested-circle case at large aspect ratio, so every check is
against a closed form or against the 1D flux-diffusion operator, with no data
dependency:

* **the 1D-limit identity** -- in the flux-surface-averaged nested limit the patch
  circuit IS ``diffuse_psi``: driving both with the same loop voltage and the same
  resistivity gives the same enclosed-current profile, and the agreement is
  non-trivial because a genuine skin has formed;
* **the analytic thin-shell time** -- a single uniform shell patch decays at
  ``mu0 L_shape A / (2 pi eta)``;
* **screening is the skin effect** -- early current sits on the outermost patches;
  late current relaxes to the conductance-weighted steady state;
* **exact zero-order hold** -- a coarse and a dense integration of a
  piecewise-linear drive agree to round-off, while a smooth drive converges at
  second order;
* **zero-net-current modes** -- deflation makes every mode a pure redistribution,
  and the slowest one is edge-weighted;
* **one coupled circuit** -- an external ring screens the plasma swing (Lenz), the
  current-target solve lands the PLASMA total, and the flux ledger closes from a
  quiescent start with no free constant;
* **machine-agnostic construction** -- a uniform geometric rescale maps
  ``tau -> tau s^2`` exactly and leaves the mode shapes invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.biot.greens import MU0, hybrid_greens
from nova.circuit import screening
from nova.circuit.screening import CoreCells
from nova.transport.current_diffusion import EtaProfile, FluxSurfaceGeometry

#: large-aspect nested-circle machine: toroidal corrections are O(a/R0)^2, ~0.4%
R0 = 8.0
A_MINOR = 0.5
B0 = 1.0
ETA_FLAT = EtaProfile(eta0=3.0e-7, contrast=0.0, shape=1.0)


def _nested_circle_cells(r0=R0, a=A_MINOR, n=45):
    """In-limiter cells of a circular-limiter grid, with their Green's matrix.

    The limiter is a circle of radius ``1.24 a`` about ``(r0, 0)``; the cell
    Green's matrix is the finite-area axisymmetric kernel between every pair of
    cell sections, exactly as an equilibrium grid caches it.
    """
    grid_r, grid_z = np.meshgrid(
        np.linspace(r0 - 1.4 * a, r0 + 1.4 * a, n),
        np.linspace(-1.4 * a, 1.4 * a, n),
    )
    flat_r, flat_z = grid_r.ravel(), grid_z.ravel()
    dr = float(grid_r[0, 1] - grid_r[0, 0])
    dz = float(grid_z[1, 0] - grid_z[0, 0])
    inside = np.hypot(flat_r - r0, flat_z) <= 1.24 * a
    index = np.flatnonzero(inside)
    cells = CoreCells(index=index, r=flat_r[index], z=flat_z[index], area=dr * dz)
    greens = np.column_stack(
        [
            hybrid_greens(
                cells.r, cells.z, float(cells.r[j]), float(cells.z[j]), dr, dz
            )[0]
            for j in range(cells.n_cells)
        ]
    )
    return cells, greens, (flat_r, flat_z, dr, dz)


def _core_state(cells: CoreCells, r0=R0, a=A_MINOR):
    """Normalised flux and core selection of the nested-circle equilibrium."""
    rho = np.hypot(cells.r - r0, cells.z) / a
    return np.clip(rho**2, 0.0, 1.0), rho <= 1.0


def _fixture_circuit(cells, greens, r0=R0, a=A_MINOR, n_rad=12, n_pol=8):
    psi_n, in_core = _core_state(cells, r0, a)
    return screening.build_plasma_circuit_from_state(
        cells, psi_n, in_core, (r0, 0.0), greens, n_rad=n_rad, n_pol=n_pol
    )


def _analytic_geometry(n_rho: int = 24) -> FluxSurfaceGeometry:
    """Exact 1D metrics of concentric circular surfaces at large aspect ratio."""
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=np.zeros(n_rho + 1),  # zero initial current: uniform flux
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=4.0 * np.pi**2 * R0 * A_MINOR**2 * rho_face,
        vpr_cell=4.0 * np.pi**2 * R0 * A_MINOR**2 * rho_cell,
        g2_face=16.0 * np.pi**4 * A_MINOR**2 * rho_face**2,
        g3_face=np.full(n_rho + 1, 1.0 / R0**2),
        g3_cell=np.full(n_rho, 1.0 / R0**2),
        f_face=np.full(n_rho + 1, R0 * B0),
        f_cell=np.full(n_rho, R0 * B0),
        b2_cell=np.full(n_rho, B0**2),
        inv_r_cell=np.full(n_rho, 1.0 / R0),
        phi_b=B0 * np.pi * A_MINOR**2,
        r0=R0,
        ip_amperes=0.0,
        axis_psi=0.0,
        boundary_psi=0.0,
        volume=2.0 * np.pi**2 * R0 * A_MINOR**2,
        q_face=np.ones(n_rho + 1),
        # the kernel's physical flux DECREASES outward for a positive current, so
        # the fixture must match that orientation or the raw flux comparison sees a
        # mirror image (the current profile is blind to it: flat-resistivity
        # diffusion is odd-symmetric in the flux)
        flux_sign=-1.0,
    )


@pytest.fixture(scope="module")
def nested():
    """The nested-circle cells, their Green's matrix, and the patch circuit."""
    cells, greens, grid = _nested_circle_cells()
    return cells, greens, grid, _fixture_circuit(cells, greens)


@pytest.mark.slow
def test_flux_surface_limit_matches_the_flux_diffusion_operator(nested):
    """The patch circuit and the 1D diffusion operator are one system: driven with
    the same loop voltage and resistivity they give the same enclosed current."""
    from nova.transport.current_diffusion import diffuse_psi

    _cells, _greens, _grid, circuit = nested
    times = np.linspace(0.0, 0.06, 121)  # fast against mu0 a^2 / eta, about 1 s
    i_patch = screening.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),  # constant: exactly ZOH-representable
    )
    ip_of_t = i_patch.sum(axis=1)
    assert ip_of_t[-1] > 0.0

    geometry = _analytic_geometry(n_rho=24)
    step = diffuse_psi(geometry, ETA_FLAT, t_grid=times, ip_of_t=ip_of_t)
    i_1d = geometry.enclosed_current(step["psi_face"][-1])

    tiling = circuit.tiling
    cell_current = tiling.share * i_patch[-1][tiling.owner]
    rho_cell = np.sqrt(np.clip(circuit.cell_psi_n, 0.0, 1.0))
    order = np.argsort(rho_cell)
    cumulative = np.cumsum(cell_current[order])
    i_circuit = np.interp(geometry.rho_face, rho_cell[order], cumulative, left=0.0)

    ip_end = float(ip_of_t[-1])
    error = (i_circuit[2:-1] - i_1d[2:-1]) / ip_end
    assert float(np.sqrt(np.mean(error**2))) < 0.05

    # the pin must be non-trivial: a genuine skin formed, so the enclosed-current
    # profile is hollow relative to the fully-penetrated steady state
    i_steady = screening.steady_state_currents(circuit, ETA_FLAT, ip_end)
    cumulative_steady = np.cumsum((tiling.share * i_steady[tiling.owner])[order])
    at_mid = float(np.interp(0.7, rho_cell[order], cumulative))
    at_mid_steady = float(np.interp(0.7, rho_cell[order], cumulative_steady))
    assert at_mid < 0.6 * at_mid_steady


@pytest.mark.slow
def test_thin_shell_decay_time_is_analytic(nested):
    cells, greens, _grid, _circuit = nested
    shell_radius = 0.4
    rho = np.hypot(cells.r - R0, cells.z) / A_MINOR
    half_width = 0.55 * np.sqrt(cells.area)
    shell = np.abs(rho * A_MINOR - shell_radius) <= half_width
    psi_n = np.clip(rho**2, 0.0, 1.0)
    # a single radial and poloidal bin makes the whole selection ONE patch
    circuit = screening.build_plasma_circuit_from_state(
        cells, psi_n, shell, (R0, 0.0), greens, n_rad=1, n_pol=1
    )
    assert circuit.n_patches == 1
    tau, _v = screening.circuit_eigensystem(circuit, ETA_FLAT)
    cross_section = circuit.tiling.cell_position.size * circuit.cell_area
    analytic = (
        MU0
        * (np.log(8.0 * R0 / shell_radius) - 2.0)
        * cross_section
        / (2.0 * np.pi * ETA_FLAT.eta0)
    )
    assert abs(float(tau[0]) - analytic) / analytic < 0.08


@pytest.mark.slow
def test_a_fast_drive_puts_the_current_on_the_surface(nested):
    """The skin effect IS circuit screening: early current lives at the edge and
    late current relaxes to the conductance-weighted steady state."""
    _cells, _greens, _grid, circuit = nested
    resistive_time = MU0 * A_MINOR**2 / ETA_FLAT.eta0
    rho_patch = np.sqrt(np.clip(circuit.tiling.psi_n, 0.0, 1.0))
    outer = rho_patch > 0.8

    early = screening.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        np.linspace(0.0, 2e-3, 21),
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(21),
    )[-1]
    outer_fraction_early = float(early[outer].sum() / early.sum())
    assert outer_fraction_early > 0.8

    times = np.linspace(0.0, 12.0 * resistive_time, 400)
    late = screening.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=np.ones(times.size),
    )[-1]
    steady = screening.steady_state_currents(circuit, ETA_FLAT, float(late.sum()))
    assert float(np.abs(late - steady).max() / np.abs(steady).max()) < 0.05
    outer_fraction_late = float(late[outer].sum() / late.sum())
    assert outer_fraction_late < outer_fraction_early - 0.3  # penetration happened


@pytest.mark.slow
def test_the_current_target_solve_lands_the_total(nested):
    _cells, _greens, _grid, circuit = nested
    times = np.linspace(0.0, 5e-3, 41)
    voltage, i_end = screening.loop_voltage_for_ip(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        ip_target=2.0e5,
    )
    assert voltage > 0.0
    assert abs(float(i_end.sum()) - 2.0e5) < 1e-6 * 2.0e5


@pytest.mark.slow
def test_zero_order_hold_trajectory_is_exact_for_a_linear_drive(nested):
    from nova.circuit.propagate import integrate_eddy_ode

    _cells, _greens, _grid, circuit = nested
    basis = screening.screening_eigenbasis(
        circuit, ETA_FLAT, np.zeros((1, circuit.n_cells)), n_modes=3
    )
    coarse = np.linspace(0.0, 0.08, 9)
    dense = np.linspace(0.0, 0.08, 401)
    slope = np.linspace(1.0, 3.0, basis.n_modes)

    assert np.allclose(
        screening.screening_trajectory(basis, coarse), 0.0
    )  # no drive, no state

    state_coarse, _ = integrate_eddy_ode(basis.tau, coarse, np.outer(coarse, slope))
    state_dense, _ = integrate_eddy_ode(basis.tau, dense, np.outer(dense, slope))
    at_coarse = np.column_stack(
        [
            np.interp(coarse, dense, state_dense[:, mode])
            for mode in range(basis.n_modes)
        ]
    )
    assert np.abs(state_coarse - at_coarse).max() < 1e-10 * max(
        np.abs(state_dense).max(), 1e-30
    )

    # a smooth (quadratic) drive is not exactly ZOH-representable, so halving the
    # step must shrink the error by about four
    def end_state(n_steps):
        times = np.linspace(0.0, 0.08, n_steps)
        state, _ = integrate_eddy_ode(basis.tau, times, np.outer(times**2, slope))
        return float(state[-1, 0])

    exact = end_state(4001)
    coarse_error = abs(end_state(11) - exact)
    fine_error = abs(end_state(21) - exact)
    assert coarse_error / max(fine_error, 1e-30) > 3.0


@pytest.mark.slow
def test_screening_modes_are_zero_net_and_edge_weighted(nested):
    _cells, _greens, _grid, circuit = nested
    basis = screening.screening_eigenbasis(
        circuit, ETA_FLAT, np.zeros((1, circuit.n_cells)), n_modes=3
    )
    assert basis.n_modes == 3
    # zero net current: exact by construction through the subspace deflation
    net = np.abs(basis.i_cell.sum(axis=0))
    assert net.max() < 1e-9 * np.abs(basis.i_cell).sum(axis=0).max()
    assert np.all(np.diff(basis.tau) <= 1e-12)  # slowest first

    # the leading (slowest) mode is edge-weighted beyond the uniform area share
    rho_cell = np.sqrt(np.clip(circuit.cell_psi_n, 0.0, 1.0))
    weight = np.abs(basis.i_cell[circuit.tiling.cell_position, 0])
    mode_mean = float((weight * rho_cell).sum() / weight.sum())
    assert mode_mean > float(rho_cell.mean())
    assert basis.psi_grid.shape == (0, 3)  # no grid solver supplied


@pytest.mark.slow
def test_a_supplied_grid_solver_fills_the_flux_columns(nested):
    """The grid flux per mode comes from the equilibrium solver's own path; the
    basis only needs the callable, so the seam is a one-line injection."""
    cells, _greens, grid, circuit = nested
    flat_r, flat_z, dr, dz = grid

    def grid_flux(i_cell):
        # a direct kernel superposition stands in for a Dirichlet grid solve
        return sum(
            current
            * hybrid_greens(
                flat_r, flat_z, float(cells.r[j]), float(cells.z[j]), dr, dz
            )[0]
            for j, current in enumerate(i_cell)
            if current != 0.0
        )

    basis = screening.screening_eigenbasis(
        circuit,
        ETA_FLAT,
        np.zeros((1, circuit.n_cells)),
        n_modes=1,
        grid_flux=grid_flux,
    )
    assert basis.psi_grid.shape == (flat_r.size, 1)
    assert np.isfinite(basis.psi_grid).all()


def _external_ring(flat_r, flat_z, r_ring, z_ring, width=0.04):
    """One structure-like external ring: its grid flux column and its self-L."""
    column = hybrid_greens(flat_r, flat_z, r_ring, z_ring, width, width)[0]
    self_linkage = hybrid_greens(
        np.array([r_ring]), np.array([z_ring]), r_ring, z_ring, width, width
    )[0]
    return column, float(self_linkage[0])


@pytest.mark.slow
def test_the_coupled_circuit_reduces_to_the_uncoupled_one(nested):
    cells, _greens, grid, circuit = nested
    _column, self_linkage = _external_ring(grid[0], grid[1], R0, 0.68)
    coupled = screening.build_coupled_circuit(
        circuit,
        l_external=np.array([[self_linkage]]),
        r_external=np.array([1.0e-4]),
        m_external_channel=np.zeros((1, 0)),
        m_patch_external=np.zeros((circuit.n_patches, 1)),
    )
    times = np.linspace(0.0, 0.05, 41)
    voltage = np.ones(times.size)
    plasma_only = screening.evolve_patch_currents(
        circuit,
        ETA_FLAT,
        times,
        i0=np.zeros(circuit.n_patches),
        loop_voltage=voltage,
    )
    together = screening.evolve_coupled(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=voltage,
    )
    n_plasma = circuit.n_patches
    reference = np.abs(plasma_only[-1]).max()
    assert np.abs(together[:, :n_plasma] - plasma_only).max() < 1e-6 * reference
    # an undriven, uncoupled external ring stays exactly quiescent
    assert np.abs(together[:, n_plasma:]).max() < 1e-12 * reference


@pytest.mark.slow
def test_an_external_ring_screens_the_plasma_swing(nested):
    """Lenz: a fast plasma current swing induces an OPPOSING external eddy."""
    cells, _greens, grid, circuit = nested
    column, self_linkage = _external_ring(grid[0], grid[1], R0, 0.68)
    patch_external = screening.patch_external_linkage(
        circuit.tiling, column[cells.index][:, np.newaxis]
    )
    assert patch_external.shape == (circuit.n_patches, 1)
    assert (patch_external > 0).all()  # coaxial rings link positively
    coupled = screening.build_coupled_circuit(
        circuit,
        l_external=np.array([[self_linkage]]),
        r_external=np.array([2.0e-5]),
        m_external_channel=np.zeros((1, 0)),
        m_patch_external=patch_external,
    )
    n_plasma = circuit.n_patches
    times = np.linspace(0.0, 2e-3, 41)  # fast against both time scales
    state = screening.evolve_coupled(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.ones(times.size),
    )
    assert float(state[-1, :n_plasma].sum()) > 0.0
    assert float(state[-1, n_plasma]) < 0.0  # opposing (screening) eddy

    # the current-target solve lands the PLASMA total, not the grand total
    _voltage, i_end = screening.coupled_loop_voltage_for_ip(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        ip_target=1.0e5,
    )
    assert abs(float(i_end[:n_plasma].sum()) - 1.0e5) < 1e-3 * 1.0e5
    assert float(i_end[n_plasma]) < 0.0


@pytest.mark.slow
def test_the_flux_ledger_closes_from_a_quiescent_start(nested):
    """The applied voltage integral equals the linked-flux change plus the
    resistive dissipation -- exact when the state integrates from zero, which is
    the pre-breakdown contract: no free integration constant."""
    cells, _greens, grid, circuit = nested
    column, self_linkage = _external_ring(grid[0], grid[1], R0, 0.68)
    coupled = screening.build_coupled_circuit(
        circuit,
        l_external=np.array([[self_linkage]]),
        r_external=np.array([2.0e-5]),
        m_external_channel=np.zeros((1, 0)),
        m_patch_external=screening.patch_external_linkage(
            circuit.tiling, column[cells.index][:, np.newaxis]
        ),
    )
    times = np.linspace(0.0, 0.08, 401)
    applied = 0.7
    state = screening.evolve_coupled(
        coupled,
        ETA_FLAT,
        times,
        i0=np.zeros(coupled.n_total),
        loop_voltage=np.full(times.size, applied),
    )
    flux_start = screening.mean_plasma_linked_flux(coupled, state[0])
    flux_end = screening.mean_plasma_linked_flux(coupled, state[-1])
    assert abs(flux_start) < 1e-30  # a zero state links zero flux, no constant
    n_plasma = circuit.n_patches
    resistance = circuit.r_diag(ETA_FLAT)
    dissipated = float(
        np.trapezoid(
            (state[:, :n_plasma] * resistance[np.newaxis, :]).mean(axis=1), times
        )
    )
    driven = applied * float(times[-1] - times[0])
    closure = abs(driven - ((flux_end - flux_start) + dissipated))
    assert closure < 5e-4 * abs(driven)


@pytest.mark.slow
def test_a_uniform_rescale_maps_the_decay_times_by_the_square():
    """Machine-agnostic construction: the linkage scales with the size and the
    resistance inversely, so a uniform rescale maps every decay time by ``s^2``
    exactly and leaves the dimensionless mode shapes untouched."""
    factor = 2.0
    small_cells, small_greens, _grid = _nested_circle_cells(n=29)
    large_cells, large_greens, _grid = _nested_circle_cells(
        r0=factor * R0, a=factor * A_MINOR, n=29
    )
    small = _fixture_circuit(small_cells, small_greens, n_rad=8, n_pol=6)
    large = _fixture_circuit(
        large_cells,
        large_greens,
        r0=factor * R0,
        a=factor * A_MINOR,
        n_rad=8,
        n_pol=6,
    )
    assert small.n_patches == large.n_patches
    tau_small, v_small = screening.circuit_eigensystem(small, ETA_FLAT)
    tau_large, v_large = screening.circuit_eigensystem(large, ETA_FLAT)
    ratio = tau_large / tau_small
    assert np.abs(ratio / factor**2 - 1.0).max() < 1e-6
    # the mode shapes are dimensionless: identical up to sign
    for mode in range(3):
        overlap = abs(
            float(v_small[:, mode] @ small.lmat @ v_large[:, mode])
            / np.sqrt(float(v_small[:, mode] @ small.lmat @ v_small[:, mode]))
            / np.sqrt(float(v_large[:, mode] @ small.lmat @ v_large[:, mode]))
        )
        assert overlap > 0.999


def test_the_tiling_rules_are_dimensionless():
    """Binning in ``(sqrt(psi_n), angle)`` is scale-free: the innermost bin is one
    patch, shares sum to one per patch, and cell currents bin back exactly."""
    radius = np.array([1.0, 1.02, 1.2, 1.2, 1.4, 1.4])
    height = np.array([0.0, 0.01, 0.3, -0.3, 0.5, -0.5])
    cells = CoreCells(index=np.arange(6), r=radius, z=height, area=0.01)
    psi_n = np.array([0.0, 0.01, 0.3, 0.3, 0.9, 0.9])
    tiling = screening.tile_core_patches(
        cells, psi_n, np.ones(6, dtype=bool), (1.0, 0.0), n_rad=3, n_pol=2
    )
    # the two axis cells share the single innermost patch, whatever their angle
    assert tiling.owner[0] == tiling.owner[1]
    # one axis patch, plus an upper and a lower patch in each of the two outer
    # radial bins (sqrt(psi_n) = 0.55 and 0.95 at n_rad = 3)
    assert tiling.n_patches == 5
    for patch in range(tiling.n_patches):
        np.testing.assert_allclose(tiling.share[tiling.owner == patch].sum(), 1.0)
    # binning a cell-current vector recovers the patch totals exactly
    cell_current = np.arange(6, dtype=float)
    np.testing.assert_allclose(
        tiling.bin_cell_currents(cell_current).sum(), cell_current.sum()
    )


def test_the_zero_net_current_subspace_is_orthonormal_and_sums_to_zero():
    basis = screening.zero_net_current_subspace(6)
    assert basis.shape == (6, 5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-12)


def test_plasma_self_inductance_is_the_large_aspect_form():
    inductance = screening.plasma_self_inductance(0.9, 0.5, 1.8, 0.9)
    expected = MU0 * 0.9 * (np.log(8.0 * 0.9 / (0.5 * np.sqrt(1.8))) - 2.0 + 0.45)
    np.testing.assert_allclose(inductance, expected, rtol=1e-12)
    # a trace broadcasts, and an inapplicable geometry returns NaN not nonsense
    traced = screening.plasma_self_inductance(
        np.array([0.9, 0.9]), np.array([0.1, 0.0])
    )
    assert np.isfinite(traced[0]) and np.isnan(traced[1])
