"""Shot-level current-diffusion flux-ledger scoring."""

from dataclasses import replace

import numpy as np
import pytest

from nova.biot.greens import MU0
from nova.transport.current_diffusion import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
)
from nova.transport.parity_flux_ledger import (
    FROZEN_GATE_SHOTS,
    ReconstructedEquilibrium,
    flux_ledger_tolerance,
    score_gated_flux_ledgers,
    score_shot_flux_ledger,
)


def circular_geometry(*, current: float = 5.0e5, radial_cells: int = 24):
    """Return analytic large-aspect geometry with a uniform-current flux shape."""

    minor_radius, major_radius, field = 0.5, 3.0, 2.0
    rho_face = np.linspace(0.0, 1.0, radial_cells + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    radius = minor_radius * rho_face
    boundary_toroidal_flux = np.pi * minor_radius**2 * field
    f_face = np.full_like(rho_face, major_radius * field)
    g2_face = 16.0 * np.pi**4 * radius**2
    g3_face = np.full_like(rho_face, 1.0 / major_radius**2)
    volume_derivative = 4.0 * np.pi**2 * major_radius * minor_radius**2 * rho_face
    diffusion = np.zeros_like(rho_face)
    diffusion[1:] = g2_face[1:] * g3_face[1:] / rho_face[1:]
    gradient = np.zeros_like(rho_face)
    gradient[1:] = (
        current
        * rho_face[1:] ** 2
        * 16.0
        * np.pi**3
        * MU0
        * boundary_toroidal_flux
        / (diffusion[1:] * f_face[1:])
    )
    psi_face = np.concatenate(
        [[0.0], np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * np.diff(rho_face))]
    )
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=psi_face,
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=volume_derivative,
        vpr_cell=0.5 * (volume_derivative[:-1] + volume_derivative[1:]),
        g2_face=g2_face,
        g3_face=g3_face,
        g3_cell=np.full(radial_cells, 1.0 / major_radius**2),
        f_face=f_face,
        f_cell=np.full(radial_cells, major_radius * field),
        b2_cell=np.full(radial_cells, field**2),
        inv_r_cell=np.full(radial_cells, 1.0 / major_radius),
        phi_b=boundary_toroidal_flux,
        r0=major_radius,
        ip_amperes=current,
        axis_psi=0.0,
        boundary_psi=float(psi_face[-1]),
        volume=2.0 * np.pi**2 * major_radius * minor_radius**2,
        q_face=np.ones_like(rho_face),
    )


def reconstructed_shot(*, discrepancy_fraction: float):
    """Build three reconstructed states around an exact diffusion trajectory."""

    eta = EtaProfile(eta0=1.0e-6, contrast=0.0, shape=1.0)
    times = (0.0, 0.002, 0.004)
    geometries = [circular_geometry()]
    pattern = np.linspace(-1.0, 1.0, geometries[0].rho_face.size)
    for start, end in zip(times[:-1], times[1:], strict=True):
        initial = geometries[-1]
        step = CurrentDiffusion(initial, eta).evolve(
            np.array([start, end]),
            np.array([initial.ip_amperes, initial.ip_amperes]),
        )
        predicted = np.asarray(step["psi_face"][-1])
        swing = predicted - initial.psi_face
        error = discrepancy_fraction * np.sqrt(np.mean(swing**2)) * pattern
        observed = predicted + error
        geometries.append(
            replace(
                initial,
                psi_face=observed,
                axis_psi=float(observed[0]),
                boundary_psi=float(observed[-1]),
            )
        )
    return tuple(
        ReconstructedEquilibrium(time, geometry, 0.7 + 0.02 * index)
        for index, (time, geometry) in enumerate(zip(times, geometries, strict=True))
    )


def test_shot_report_carries_verdict_flux_consumption_and_profile_separation():
    """The temporal scalar never erases the physical terms that produced it."""

    eta = EtaProfile(eta0=1.0e-6, contrast=0.0, shape=1.0)
    report = score_shot_flux_ledger(
        FROZEN_GATE_SHOTS[0], reconstructed_shot(discrepancy_fraction=0.001), eta=eta
    )

    assert report.bound == pytest.approx(0.004)
    assert report.rms_fraction is not None and report.rms_fraction < report.bound
    assert report.passed is True
    assert len(report.slice_rms_fractions) == 3
    assert report.observed_surface_flux_consumption_wb != 0.0
    assert report.predicted_surface_flux_consumption_wb != 0.0
    assert report.predicted_surface_flux_consumption_wb == pytest.approx(
        report.resistive_flux_consumption_wb + report.inductive_flux_consumption_wb
    )
    assert len(report.internal_inductance) == len(report.poloidal_beta) == 3
    np.testing.assert_allclose(
        report.li_minus_beta_p,
        np.asarray(report.internal_inductance) - np.asarray(report.poloidal_beta),
    )


def test_every_frozen_shot_is_reported_with_pass_fail_or_reason():
    """A missing ledger stays named rather than shrinking the scored cohort."""

    eta = EtaProfile(eta0=1.0e-6, contrast=0.0, shape=1.0)
    reconstructions = {
        shot: reconstructed_shot(
            discrepancy_fraction=0.001 if shot != FROZEN_GATE_SHOTS[-2] else 0.02
        )
        for shot in FROZEN_GATE_SHOTS[:-1]
    }
    reports = score_gated_flux_ledgers(reconstructions, eta=eta)

    assert tuple(report.shot for report in reports) == FROZEN_GATE_SHOTS
    assert all(report.bound == pytest.approx(0.004) for report in reports)
    assert all(report.passed is True for report in reports[:4])
    assert reports[4].passed is False
    assert reports[4].rms_fraction is not None
    assert reports[4].rms_fraction > reports[4].bound
    assert reports[5].passed is None
    assert reports[5].rms_fraction is None
    assert reports[5].reason == "reconstructed equilibria are unavailable"


def test_invalid_shot_names_the_ledger_failure():
    """A non-evolving reconstructed sequence becomes evidence, not an omission."""

    geometry = circular_geometry()
    stalled = (
        ReconstructedEquilibrium(0.0, geometry, 0.7),
        ReconstructedEquilibrium(0.1, geometry, 0.7),
    )
    reports = score_gated_flux_ledgers({FROZEN_GATE_SHOTS[0]: stalled})

    assert reports[0].passed is None
    assert "zero observed flux swing" in reports[0].reason


def test_bound_is_read_from_the_registered_temporal_tolerance():
    tolerance = flux_ledger_tolerance()

    assert tolerance.field.value == "current_diffusion_flux_ledger_rms_fraction"
    assert tolerance.bound == pytest.approx(0.004)
