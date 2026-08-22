"""Single-sweep exchange primitives over evolving radial coordinates."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from nova.equilibrium import SelectionHistory, SelectionPolicy, SelectionReason
from nova.equilibrium.forward import ForwardPortfolio
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.transport.coupled_window import (
    ConvergedNonConfinedError,
    ExchangeSweepResult,
    TransportSweepReceipt,
    Waveform,
    WindowConfig,
    WindowConvergenceError,
    equilibrium_sweep,
    solve_window,
    transport_sweep,
)
from nova.transport.evolved_state import EvolvedFluxFunction
from nova.transport.forward import (
    AchievedBoundaryValues,
    FluxConsumptionLedger,
    ForwardTransport,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportGeometry,
    TransportProvenance,
    TransportRung,
    TransportState,
)
from tests.test_equilibrium_forward_solve import DRIVE, EVALUATIONS, FF_PRIME, P_PRIME
from tests.test_forward_transport import _request

pytest_plugins = ("tests.test_equilibrium_forward_solve",)

jax.config.update("jax_platforms", "cpu")

PROFILE_INTERPOLATION_TOLERANCE = 2.0e-14
TRANSPORT_SWEEP_TOLERANCE = 2.0e-13
EQUILIBRIUM_RESIDUAL_TOLERANCE = 1.0e-6
WINDOW_CONVERGENCE_TOLERANCE = 1.0e-10
WEAK_COUPLING_AGREEMENT = 3.0e-3
STRONG_COUPLING_DIVERGENCE = 0.75
LEDGER_CLOSURE_TOLERANCE = 3.0e-12


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
    assert len(sweep.branch_receipts) == coarse_time.size
    assert tuple(
        receipt.selection.selected_class for receipt in sweep.branch_receipts
    ) == (TopologyClass.LIMITED, TopologyClass.LIMITED)
    assert sweep.branch_receipts[1].selection.previous_class is TopologyClass.LIMITED
    assert (
        sweep.branch_receipts[1].selection.reason is SelectionReason.HISTORY_CONTINUITY
    )
    assert tuple(
        receipt.selection.next_history.sequence_index
        for receipt in sweep.branch_receipts
    ) == (1, 2)
    assert all(
        receipt.core_cell_counts[int(receipt.selection.selected_class)] > 0
        for receipt in sweep.branch_receipts
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


def _exchange_waveform(time, radial_points, channel, value):
    """Return one fixed-shape exchanged field on its side's sample grid."""
    time = np.asarray(time, dtype=np.float64)
    radial_grid = np.broadcast_to(
        np.linspace(0.0, 1.0, radial_points), (time.size, radial_points)
    )
    return Waveform(
        time=time,
        radial_grid=radial_grid,
        phi_boundary=np.full(time.shape, 2.5),
        axis_reference=np.full(time.shape, -0.35),
        boundary_reference=np.full(time.shape, 0.02),
        values={channel: np.full(radial_grid.shape, value)},
    )


def _transport_window_receipt(time):
    """Return exactly closing interval ledgers with a measured round-off floor."""
    time = np.asarray(time, dtype=np.float64)
    rho = np.linspace(0.0, 1.0, 5)
    state = TransportState(
        rho=rho,
        psi=0.2 * rho**2,
        ion_temperature=8.0 - 7.0 * rho,
        electron_temperature=7.0 - 6.0 * rho,
        electron_density=1.2e20 - 2.0e19 * rho,
    )
    current = 1.0e6
    achieved_current = current + 2.0e-6
    receipts = []
    for index in range(time.size - 1):
        boundary_flux = 0.03 * (index + 1)
        resistive_flux = 0.4 * boundary_flux
        internal_flux = boundary_flux - resistive_flux + 5.0e-14
        receipts.append(
            ForwardTransportReceipt(
                state=state,
                flux_consumption=FluxConsumptionLedger(
                    boundary=boundary_flux,
                    resistive=resistive_flux,
                    internal=internal_flux,
                    mean_axis_voltage=resistive_flux
                    / float(time[index + 1] - time[index]),
                    mean_boundary_voltage=boundary_flux
                    / float(time[index + 1] - time[index]),
                ),
                plasma_current=PlasmaCurrentLedger(
                    requested_initial=current,
                    requested_final=current,
                    achieved_initial=achieved_current,
                    achieved_final=achieved_current,
                ),
                boundary=AchievedBoundaryValues(
                    psi=float(state.psi[-1]),
                    plasma_current=achieved_current,
                    ion_temperature=float(state.ion_temperature[-1]),
                    electron_temperature=float(state.electron_temperature[-1]),
                    electron_density=float(state.electron_density[-1]),
                ),
                diagnostics=SolverDiagnostics(
                    engine_status="converged",
                    steps=1,
                    outer_iterations=1,
                    inner_iterations=1,
                ),
                provenance=TransportProvenance(
                    rung=TransportRung.NATIVE_PSI_DIFFUSION,
                    engine="coupled-fixture",
                    engine_version="analytic",
                ),
            )
        )
    return TransportSweepReceipt(
        time=time,
        geometry_time=0.5 * (time[:-1] + time[1:]),
        receipts=tuple(receipts),
    )


class _AffineWindow:
    """Analytic two-side map with a declared combined coupling strength."""

    def __init__(self, config, coupling):
        self.config = config
        self.coupling = coupling
        self.geometry_template = _exchange_waveform(
            config.equilibrium_grid, 5, "geometry", 0.0
        )
        self.source_template = _exchange_waveform(
            config.transport_grid, 7, "source", 0.0
        )

    @staticmethod
    def _map(input_waveform, template, input_channel, output_channel, gain, offset):
        values = np.stack(
            [
                gain
                * input_waveform.sample(
                    float(time), radial_grid=template.radial_grid[index]
                ).values[input_channel]
                + offset
                for index, time in enumerate(template.time)
            ]
        )
        return Waveform(
            time=template.time,
            radial_grid=template.radial_grid,
            phi_boundary=template.phi_boundary,
            axis_reference=template.axis_reference,
            boundary_reference=template.boundary_reference,
            values={output_channel: values},
        )

    def transport(self, geometry, sample_grid):
        source = self._map(
            geometry,
            self.source_template,
            "geometry",
            "source",
            gain=1.0,
            offset=1.0,
        )
        return ExchangeSweepResult(
            waveform=source,
            receipt=_transport_window_receipt(sample_grid),
        )

    def equilibrium(self, source, _sample_grid):
        geometry = self._map(
            source,
            self.geometry_template,
            "source",
            "geometry",
            gain=self.coupling,
            offset=0.0,
        )
        return ExchangeSweepResult(
            waveform=geometry,
            receipt={"finite_conservation_receipts": True},
        )


def _field_difference(left, right, name):
    """Return the scale-normalised separation of two waveform fields."""
    left_value = np.asarray(left.values[name])
    right_value = np.asarray(right.values[name])
    scale = max(float(np.max(np.abs(left_value))), 1.0e-30)
    return float(np.max(np.abs(left_value - right_value))) / scale


def _solve_affine_window(coupling, *, damping=1.0, iteration_cap=180):
    """Return one analytic converged window and its cap-one diagnostic."""
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=iteration_cap,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    exchange = _AffineWindow(config, coupling)
    converged = solve_window(
        exchange.geometry_template,
        exchange.source_template,
        config,
        exchange.equilibrium,
        exchange.transport,
        damping=damping,
    )
    one_iteration = dataclasses.replace(config, iteration_cap=1)
    with pytest.raises(WindowConvergenceError) as raised:
        solve_window(
            exchange.geometry_template,
            exchange.source_template,
            one_iteration,
            exchange.equilibrium,
            exchange.transport,
            damping=damping,
        )
    return converged, raised.value, exchange, config


def test_converged_and_cap_one_windows_share_one_measured_coupling_axis():
    """One staggered pass is close when coupling is weak and far when strong."""
    weak, weak_single, weak_exchange, weak_config = _solve_affine_window(0.002)
    strong, strong_single, strong_exchange, strong_config = _solve_affine_window(
        0.8, damping=0.8
    )

    weak_difference = _field_difference(
        weak.geometry_waveform, weak_single.geometry_waveform, "geometry"
    )
    strong_difference = _field_difference(
        strong.geometry_waveform, strong_single.geometry_waveform, "geometry"
    )
    assert weak_difference < WEAK_COUPLING_AGREEMENT
    assert strong_difference > STRONG_COUPLING_DIVERGENCE
    assert weak.convergence.maximum_residual <= weak_config.tolerance
    assert strong.convergence.maximum_residual <= strong_config.tolerance
    assert weak.convergence.iterations_used > 1
    assert strong.convergence.iterations_used > weak.convergence.iterations_used
    assert weak.convergence.contraction_estimate is not None
    assert strong.convergence.contraction_estimate is not None
    assert weak.convergence.contraction_estimate < 1.0
    assert strong.convergence.contraction_estimate < 1.0
    assert weak.convergence.damping_applied == 1.0
    assert strong.convergence.damping_applied == 0.8
    assert set(weak.convergence.exit_residual) == {
        "geometry.axis_reference",
        "geometry.boundary_reference",
        "geometry.geometry",
        "geometry.phi_boundary",
        "geometry.radial_grid",
        "source.axis_reference",
        "source.boundary_reference",
        "source.phi_boundary",
        "source.radial_grid",
        "source.source",
    }
    np.testing.assert_array_equal(
        weak_exchange.geometry_template.values["geometry"], 0.0
    )
    np.testing.assert_array_equal(weak_exchange.source_template.values["source"], 0.0)
    np.testing.assert_array_equal(
        strong_exchange.geometry_template.values["geometry"], 0.0
    )
    np.testing.assert_array_equal(strong_exchange.source_template.values["source"], 0.0)


def test_window_receipt_closes_flux_and_boundary_current_ledgers():
    """The window aggregates interval conservation without hiding residuals."""
    window, _single, _exchange, config = _solve_affine_window(0.002)
    conservation = window.conservation
    flux = conservation.flux_consumption
    current = conservation.plasma_current

    np.testing.assert_allclose(
        flux.boundary,
        flux.resistive + flux.internal,
        rtol=LEDGER_CLOSURE_TOLERANCE,
        atol=0.0,
    )
    assert conservation.flux_closure_residual < LEDGER_CLOSURE_TOLERANCE
    assert conservation.current_continuity_residual < LEDGER_CLOSURE_TOLERANCE
    assert conservation.flux_closure_residual < config.tolerance
    assert conservation.current_continuity_residual < config.tolerance
    assert conservation.current_continuity_error > 0.0
    assert current.requested_initial == current.requested_final
    assert current.achieved_initial == current.achieved_final


def test_coreless_branch_outcome_names_its_sample_and_window_exchange(
    machine, converged, monkeypatch
):
    """A converged vacuum portfolio cannot cross the equilibrium boundary."""
    profile, _seed, _vacuum = machine

    def coreless_portfolio(_profile, _initial_flux, **_options):
        branches = SimpleNamespace(
            equilibrium=SimpleNamespace(
                domains=SimpleNamespace(core=np.zeros((2, 5), dtype=bool))
            ),
            converged=np.ones(2, dtype=bool),
            topology_consistent=np.ones(2, dtype=bool),
            residual=np.asarray((2.0e-16, 3.0e-16)),
        )
        return ForwardPortfolio(branches=branches)

    monkeypatch.setattr(type(profile), "solve_portfolio", coreless_portfolio)
    source_waveform = _exchange_waveform((0.0, 1.0), 5, "source", 1.0)
    with pytest.raises(ConvergedNonConfinedError) as unqualified:
        equilibrium_sweep(
            profile,
            converged.flux,
            source_waveform,
            (0.5,),
            lambda _sample: profile.source,
            route="anderson",
            selection_history=SelectionHistory(selected_class=TopologyClass.LIMITED),
            selection_policy=SelectionPolicy(
                cold_start_class=TopologyClass.LIMITED,
                persistence_threshold=1,
            ),
        )

    assert unqualified.value.exchange_index is None
    assert unqualified.value.branch_receipt.sample_index == 0
    assert unqualified.value.branch_receipt.sample_time == 0.5
    assert unqualified.value.branch_receipt.core_cell_counts == (0, 0)

    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=2,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    exchange = _AffineWindow(config, coupling=0.002)

    def coreless_update(_source, _sample_grid):
        raise unqualified.value

    with pytest.raises(ConvergedNonConfinedError, match="exchange 1") as qualified:
        solve_window(
            exchange.geometry_template,
            exchange.source_template,
            config,
            coreless_update,
            exchange.transport,
        )
    assert qualified.value.exchange_index == 1
    assert qualified.value.branch_receipt is unqualified.value.branch_receipt


def test_nonconverging_window_serializes_its_exhaustion_receipt_before_raising(
    tmp_path,
):
    """A divergent exchange cannot escape as a degraded window result."""
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=4,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    exchange = _AffineWindow(config, coupling=1.05)
    serialized = tmp_path / "exhausted-window.tsv"

    def serialize_failure(error):
        serialized.write_text(
            "iterations\tmaximum_residual\ttrace_rows\n"
            f"{error.convergence.iterations_used}\t"
            f"{error.convergence.maximum_residual:.17g}\t"
            f"{len(error.convergence.residual_trace)}\n",
            encoding="utf-8",
        )

    with pytest.raises(WindowConvergenceError, match="did not converge") as raised:
        solve_window(
            exchange.geometry_template,
            exchange.source_template,
            config,
            exchange.equilibrium,
            exchange.transport,
            failure_serializer=serialize_failure,
        )

    assert serialized.is_file()
    assert serialized.read_text(encoding="utf-8").splitlines()[1].split("\t")[0] == "4"
    assert raised.value.convergence.iterations_used == config.iteration_cap
    assert len(raised.value.convergence.residual_trace) == config.iteration_cap
    assert raised.value.convergence.maximum_residual > config.tolerance
    assert raised.value.convergence.contraction_estimate is not None
    assert np.isfinite(raised.value.convergence.contraction_estimate)
