"""Evolved transport states crossing the physical flux-function source seam."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import electron_volt, mu_0

from nova.equilibrium import convention
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.transport.evolved_state import forward_source_from_receipt
from nova.transport.forward import (
    AchievedBoundaryValues,
    FluxConsumptionLedger,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportGeometry,
    TransportProvenance,
    TransportRung,
    TransportState,
)
from tests.test_equilibrium_forward_solve import (
    DRIVE,
    EVALUATIONS,
    FF_PRIME,
    P_PRIME,
)

pytest_plugins = ("tests.test_equilibrium_forward_solve",)

jax.config.update("jax_platforms", "cpu")

FACES = 65
FLUX_SPAN = -0.35
BOUNDARY_FIELD_FUNCTION = 5.0
ELECTRON_DENSITY = 1.0e20
FIXED_POINT_TOLERANCE = 1.0e-6
DIVERGENCE_TOLERANCE = 1.0e-12
FORCE_BALANCE_TOLERANCE = 0.1
CURRENT_LEDGER_TOLERANCE = 1.0e-12
RETURN_CURRENT_TOLERANCE = 1.0e-10


def _evolved_receipt_and_geometry(target_current):
    """Return transport data encoding the source of the shared solve fixture."""
    rho = np.linspace(0.0, 1.0, FACES)
    psi_norm = rho**2
    physical_flux = FLUX_SPAN * psi_norm
    flux_sign = float(np.sign(FLUX_SPAN))
    pressure_amplitude = 2.0 * DRIVE * P_PRIME
    diamagnetic_amplitude = 2.0 * DRIVE * FF_PRIME
    tail_shape = 0.5 - psi_norm + 0.5 * psi_norm**2
    pressure = FLUX_SPAN * pressure_amplitude * tail_shape
    temperature_sum = pressure / (ELECTRON_DENSITY * 1.0e3 * electron_volt)

    g3_face = np.ones_like(rho)
    current_per_volume_amplitude = -(pressure_amplitude + diamagnetic_amplitude / mu_0)
    enclosed_volume = 2.0 * target_current / current_per_volume_amplitude
    volume_face = enclosed_volume * psi_norm
    ip_profile_face = (
        enclosed_volume * current_per_volume_amplitude * (psi_norm - 0.5 * psi_norm**2)
    )
    field_function_squared = (
        BOUNDARY_FIELD_FUNCTION**2
        + 2.0 * FLUX_SPAN * diamagnetic_amplitude * tail_shape
    )
    geometry = TransportGeometry(
        {
            "rho_face": rho,
            "ip_profile_face": ip_profile_face,
            "volume_face": volume_face,
            "g3_face": g3_face,
            "f_face": np.sqrt(field_function_squared),
            "flux_sign": flux_sign,
            "valid": True,
        }
    )
    state = TransportState(
        rho=rho,
        psi=flux_sign * physical_flux,
        ion_temperature=0.5 * temperature_sum,
        electron_temperature=0.5 * temperature_sum,
        electron_density=np.full_like(rho, ELECTRON_DENSITY),
    )
    current = float(ip_profile_face[-1])
    receipt = ForwardTransportReceipt(
        state=state,
        flux_consumption=FluxConsumptionLedger(
            boundary=0.0,
            resistive=0.0,
            internal=0.0,
            mean_axis_voltage=0.0,
            mean_boundary_voltage=0.0,
        ),
        plasma_current=PlasmaCurrentLedger(
            requested_initial=current,
            requested_final=current,
            achieved_initial=current,
            achieved_final=current,
        ),
        boundary=AchievedBoundaryValues(
            psi=float(state.psi[-1]),
            plasma_current=current,
            ion_temperature=float(state.ion_temperature[-1]),
            electron_temperature=float(state.electron_temperature[-1]),
            electron_density=float(state.electron_density[-1]),
        ),
        diagnostics=SolverDiagnostics(
            engine_status="NO_ERROR",
            steps=1,
            outer_iterations=1,
            inner_iterations=1,
        ),
        provenance=TransportProvenance(
            rung=TransportRung.TORAX_MULTI_CHANNEL,
            engine="torax",
            engine_version="1.4.3",
        ),
    )
    return receipt, geometry, psi_norm, pressure_amplitude, diamagnetic_amplitude


@pytest.fixture(scope="module")
def mapped_case(machine, converged):
    """Attach the mapped source to the smallest existing free-boundary machine."""
    receipt, geometry, psi_norm, pressure_amplitude, diamagnetic_amplitude = (
        _evolved_receipt_and_geometry(float(converged.moments.plasma_current))
    )
    source = forward_source_from_receipt(
        receipt, geometry, ion_density_per_electron=1.0
    )
    profile, seed, _vacuum = machine
    profile = replace(profile, operator=replace(profile.operator, source=source))
    return (
        receipt,
        source,
        profile,
        seed,
        psi_norm,
        pressure_amplitude,
        diamagnetic_amplitude,
    )


def test_mapping_restores_the_negated_total_flux_gradient_sense(mapped_case):
    """Pressure and current recover Nova's signed total-flux derivatives."""
    (
        receipt,
        source,
        _profile,
        _seed,
        psi_norm,
        pressure_amplitude,
        diamagnetic_amplitude,
    ) = mapped_case
    expected_pressure = pressure_amplitude * (1.0 - psi_norm)
    expected_diamagnetic = diamagnetic_amplitude * (1.0 - psi_norm)
    observed_pressure = source.core.p_prime(psi_norm)
    observed_diamagnetic = source.core.ff_prime(psi_norm)

    assert receipt.state.psi[-1] > receipt.state.psi[0]
    assert FLUX_SPAN < 0.0
    np.testing.assert_allclose(
        observed_pressure, expected_pressure, rtol=2.0e-11, atol=5.0e-11
    )
    np.testing.assert_allclose(
        observed_diamagnetic, expected_diamagnetic, rtol=2.0e-11, atol=2.0e-12
    )
    radius = jnp.asarray(1.0)
    expected_current_density = -convention.TOTAL_FLUX_FACTOR * (
        radius * observed_pressure + observed_diamagnetic / (mu_0 * radius)
    )
    np.testing.assert_allclose(
        source.core.current_density(radius, psi_norm), expected_current_density
    )
    assert np.all(np.asarray(expected_current_density[:-1]) > 0.0)


def test_mapping_crosses_the_callable_source_boundary_with_si_primitives(mapped_case):
    """The evolved profiles are callables, not sampled source images."""
    receipt, source, *_rest = mapped_case
    assert isinstance(source, ForwardSource)
    assert isinstance(source.core, DomainProfile)
    assert callable(source.core.p_prime)
    assert callable(source.core.ff_prime)
    assert float(source.boundary_pressure) == pytest.approx(0.0, abs=1.0e-10)
    assert float(source.boundary_field_function) == pytest.approx(
        BOUNDARY_FIELD_FUNCTION
    )
    with pytest.raises(TypeError, match="callable flux function"):
        DomainProfile(
            p_prime=np.asarray(source.core.p_prime(np.linspace(0.0, 1.0, 5))),
            ff_prime=source.core.ff_prime,
        )
    assert receipt.plasma_current.achieved_final > 1.0e5


def test_mapped_source_converges_with_force_and_current_receipts(mapped_case):
    """The evolved source drives the shared machine to a qualified equilibrium."""
    receipt, _source, profile, seed, *_rest = mapped_case
    equilibrium = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    conservation = equilibrium.conservation
    current_scale = max(abs(float(equilibrium.moments.plasma_current)), 1.0)
    current_residual = (
        abs(float(equilibrium.ledger.total - equilibrium.moments.plasma_current))
        / current_scale
    )
    return_current_residual = (
        abs(
            float(
                equilibrium.moments.plasma_current
                - receipt.plasma_current.achieved_final
            )
        )
        / current_scale
    )

    assert float(equilibrium.fixed_point.residual) < FIXED_POINT_TOLERANCE
    assert bool(equilibrium.finite.passed)
    assert float(conservation.relative_divergence_b) < DIVERGENCE_TOLERANCE
    assert float(conservation.relative_divergence_j) < DIVERGENCE_TOLERANCE
    assert float(conservation.relative_grad_shafranov) < FORCE_BALANCE_TOLERANCE
    assert float(conservation.relative_force) < FORCE_BALANCE_TOLERANCE
    assert current_residual < CURRENT_LEDGER_TOLERANCE
    assert return_current_residual < RETURN_CURRENT_TOLERANCE
