"""Single-sweep exchange primitives over evolving radial coordinates."""

from __future__ import annotations

import copy
import dataclasses

import jax
import numpy as np

from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.transport.coupled_window import (
    Waveform,
    equilibrium_sweep,
    transport_sweep,
)
from nova.transport.evolved_state import EvolvedFluxFunction
from nova.transport.forward import ForwardTransport, TransportGeometry, TransportRung
from tests.test_equilibrium_forward_solve import DRIVE, EVALUATIONS, FF_PRIME, P_PRIME
from tests.test_forward_transport import _request

pytest_plugins = ("tests.test_equilibrium_forward_solve",)

jax.config.update("jax_platforms", "cpu")

PROFILE_INTERPOLATION_TOLERANCE = 2.0e-14
TRANSPORT_SWEEP_TOLERANCE = 2.0e-13
EQUILIBRIUM_RESIDUAL_TOLERANCE = 1.0e-6


def test_profile_time_interpolation_stays_on_normalised_radial_coordinates():
    """Changing boundary flux cannot make temporal interpolation mix radii."""
    time = np.array([0.0, 1.0])
    radial_grid = np.array(
        [
            [0.0, 0.18, 0.52, 1.0],
            [0.0, 0.37, 0.81, 1.0],
        ]
    )
    phi_boundary = np.array([1.5, 4.5])
    values = np.stack(
        [
            2.0 + 3.0 * sample_time + 4.0 * sample_grid
            for sample_time, sample_grid in zip(time, radial_grid, strict=True)
        ]
    )
    waveform = Waveform(
        time=time,
        radial_grid=radial_grid,
        phi_boundary=phi_boundary,
        axis_reference=np.array([-0.2, 0.1]),
        boundary_reference=np.array([0.6, 1.4]),
        values={"profile": values},
    )
    query_time = 0.35
    query_grid = np.linspace(0.0, 1.0, 17)
    sample = waveform.sample(query_time, radial_grid=query_grid)

    expected = 2.0 + 3.0 * query_time + 4.0 * query_grid
    np.testing.assert_allclose(
        sample.values["profile"],
        expected,
        rtol=0.0,
        atol=PROFILE_INTERPOLATION_TOLERANCE,
    )
    np.testing.assert_allclose(sample.phi_boundary, 2.55, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(sample.axis_reference, -0.095, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(sample.boundary_reference, 0.88, rtol=0.0, atol=1.0e-15)
    assert not waveform.time.flags.writeable
    assert not waveform.radial_grid.flags.writeable
    assert not waveform.values["profile"].flags.writeable
    assert not sample.values["profile"].flags.writeable


def test_transport_sweep_matches_the_equivalent_interpolated_facade_interval():
    """A one-interval sweep is the facade solve at its interpolated geometry."""
    request = _request(TransportRung.NATIVE_PSI_DIFFUSION)
    base_record = {
        name: np.array(value, copy=True)
        for name, value in request.geometry.record.items()
    }
    early_record = copy.deepcopy(base_record)
    late_record = copy.deepcopy(base_record)
    early_record["phi_b"] *= 0.94
    late_record["phi_b"] *= 1.06
    early_record["r0"] *= 0.98
    late_record["r0"] *= 1.02
    geometry_waveform = Waveform.from_geometries(
        [-0.01, 0.02],
        [TransportGeometry(early_record), TransportGeometry(late_record)],
    )
    time = np.asarray(request.waveforms.time)
    current = np.asarray(request.waveforms.plasma_current)
    state_snapshot = {
        name: np.array(getattr(request.initial_state, name), copy=True)
        for name in (
            "rho",
            "psi",
            "ion_temperature",
            "electron_temperature",
            "electron_density",
        )
    }
    geometry_snapshot = np.array(geometry_waveform.phi_boundary, copy=True)

    sweep = transport_sweep(
        geometry_waveform,
        request.initial_state,
        time,
        current,
        request.model,
    )
    midpoint = geometry_waveform.sample(float(np.mean(time))).geometry()
    direct = ForwardTransport().solve(dataclasses.replace(request, geometry=midpoint))

    assert sweep.geometry_time.tolist() == [float(np.mean(time))]
    assert float(midpoint.record["phi_b"]) != float(early_record["phi_b"])
    assert float(midpoint.record["phi_b"]) != float(late_record["phi_b"])
    for name in (
        "rho",
        "psi",
        "ion_temperature",
        "electron_temperature",
        "electron_density",
    ):
        np.testing.assert_allclose(
            getattr(sweep.state, name),
            getattr(direct.state, name),
            rtol=TRANSPORT_SWEEP_TOLERANCE,
            atol=TRANSPORT_SWEEP_TOLERANCE,
        )
        np.testing.assert_array_equal(
            getattr(request.initial_state, name), state_snapshot[name]
        )
    np.testing.assert_allclose(
        np.asarray(dataclasses.astuple(sweep.receipts[0].flux_consumption)),
        np.asarray(dataclasses.astuple(direct.flux_consumption)),
        rtol=TRANSPORT_SWEEP_TOLERANCE,
        atol=TRANSPORT_SWEEP_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(dataclasses.astuple(sweep.receipts[0].plasma_current)),
        np.asarray(dataclasses.astuple(direct.plasma_current)),
        rtol=TRANSPORT_SWEEP_TOLERANCE,
        atol=TRANSPORT_SWEEP_TOLERANCE,
    )
    np.testing.assert_array_equal(geometry_waveform.phi_boundary, geometry_snapshot)


def test_equilibrium_sweep_consumes_interpolated_sources_and_returns_receipts(
    machine, converged
):
    """Coarse equilibrium samples retain every solve's conservation ledger."""
    profile, _seed, _vacuum = machine
    source_before = profile.source
    flux_before = np.array(converged.flux, copy=True)
    time = np.array([0.0, 1.0])
    common_grid = np.linspace(0.0, 1.0, 25)
    scale = np.array([0.999, 1.001])
    p_prime = np.stack(
        [2.0 * DRIVE * P_PRIME * factor * (1.0 - common_grid) for factor in scale]
    )
    ff_prime = np.stack(
        [2.0 * DRIVE * FF_PRIME * factor * (1.0 - common_grid) for factor in scale]
    )
    source_waveform = Waveform(
        time=time,
        radial_grid=np.stack([common_grid, common_grid]),
        phi_boundary=np.array([2.0, 2.2]),
        axis_reference=np.array([-0.34, -0.35]),
        boundary_reference=np.array([0.01, 0.02]),
        values={"p_prime": p_prime, "ff_prime": ff_prime},
    )

    def source_from_sample(sample):
        return ForwardSource(
            core=DomainProfile(
                p_prime=EvolvedFluxFunction(
                    sample.radial_grid, sample.values["p_prime"]
                ),
                ff_prime=EvolvedFluxFunction(
                    sample.radial_grid, sample.values["ff_prime"]
                ),
            ),
            boundary_pressure=source_before.boundary_pressure,
            boundary_field_function=source_before.boundary_field_function,
        )

    coarse_time = np.array([0.25, 0.75])
    sweep = equilibrium_sweep(
        profile,
        converged.flux,
        source_waveform,
        coarse_time,
        source_from_sample,
        route="anderson",
        solve_options={"evaluations": EVALUATIONS},
    )

    assert len(sweep.equilibria) == coarse_time.size
    assert sweep.conservation == tuple(
        equilibrium.conservation for equilibrium in sweep.equilibria
    )
    for sample_time, sample, equilibrium in zip(
        coarse_time, sweep.source_samples, sweep.equilibria, strict=True
    ):
        expected_scale = 0.999 + 0.002 * sample_time
        expected_axis_source = 2.0 * DRIVE * P_PRIME * expected_scale
        np.testing.assert_allclose(
            sample.values["p_prime"][0],
            expected_axis_source,
            rtol=0.0,
            atol=PROFILE_INTERPOLATION_TOLERANCE * abs(expected_axis_source),
        )
        assert float(equilibrium.fixed_point.residual) < EQUILIBRIUM_RESIDUAL_TOLERANCE
        assert bool(equilibrium.finite.passed)
        assert np.isfinite(float(equilibrium.conservation.relative_divergence_b))
        assert np.isfinite(float(equilibrium.conservation.relative_divergence_j))
    assert profile.source is source_before
    np.testing.assert_array_equal(converged.flux, flux_before)
