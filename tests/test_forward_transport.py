"""Contract checks for the public deterministic transport facade."""

from __future__ import annotations

import copy

import jax
import numpy as np
import pytest

import nova.transport.forward as forward_module
from nova.biot.greens import MU0
from nova.transport import (
    ForwardTransport,
    ForwardTransportInput,
    TransportEngineError,
    TransportGeometry,
    TransportModel,
    TransportRung,
    TransportState,
    TransportWaveforms,
)
from nova.transport.current_diffusion import EtaProfile


def _circular_record(n_rho=4):
    major_radius = 3.0
    minor_radius = 0.5
    vacuum_field = 2.0
    plasma_current = 5.0e5
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    radius_face = minor_radius * rho_face
    phi_b = np.pi * minor_radius**2 * vacuum_field
    f_face = np.full_like(rho_face, major_radius * vacuum_field)
    g2_face = 16.0 * np.pi**4 * radius_face**2
    g3_face = np.full_like(rho_face, 1.0 / major_radius**2)
    d_face = np.zeros_like(rho_face)
    d_face[1:] = g2_face[1:] * g3_face[1:] / rho_face[1:]
    gradient = np.zeros_like(rho_face)
    gradient[1:] = (
        plasma_current
        * rho_face[1:] ** 2
        * 16.0
        * np.pi**3
        * MU0
        * phi_b
        / (d_face[1:] * f_face[1:])
    )
    psi_face = np.concatenate(
        [[0.0], np.cumsum(0.5 * (gradient[:-1] + gradient[1:]) * np.diff(rho_face))]
    )
    vpr_face = 4.0 * np.pi**2 * major_radius * minor_radius**2 * rho_face
    volume_face = 2.0 * np.pi**2 * major_radius * (minor_radius * rho_face) ** 2
    poloidal_field = np.maximum(
        MU0 * plasma_current * rho_face / (2.0 * np.pi * minor_radius), 1.0e-6
    )
    int_dl_over_bp = 2.0 * np.pi * np.maximum(radius_face, 1.0e-6) / poloidal_field
    grad_psi = np.maximum(np.abs(gradient) / (2.0 * np.pi * minor_radius), 1.0e-6)
    b2_face = vacuum_field**2 + poloidal_field**2
    return {
        "rho_face": rho_face,
        "rho_cell": rho_cell,
        "psi_face": psi_face,
        "psi_n_face": rho_face**2,
        "psi_n_cell": rho_cell**2,
        "vpr_face": vpr_face,
        "vpr_cell": 0.5 * (vpr_face[:-1] + vpr_face[1:]),
        "g2_face": g2_face,
        "g3_face": g3_face,
        "g3_cell": np.full_like(rho_cell, 1.0 / major_radius**2),
        "f_face": f_face,
        "f_cell": np.full_like(rho_cell, major_radius * vacuum_field),
        "b2_cell": np.full_like(rho_cell, vacuum_field**2),
        "inv_r_cell": np.full_like(rho_cell, 1.0 / major_radius),
        "inv_r_face": np.full_like(rho_face, 1.0 / major_radius),
        "phi_b": phi_b,
        "r0": major_radius,
        "ip_amperes": plasma_current,
        "axis_psi": float(psi_face[0]),
        "boundary_psi": float(psi_face[-1]),
        "volume": float(volume_face[-1]),
        "q_face": np.ones_like(rho_face),
        "volume_face": volume_face,
        "ip_profile_face": plasma_current * rho_face**2,
        "int_dl_over_bp_face": int_dl_over_bp,
        "grad_psi_face": grad_psi,
        "grad_psi2_face": grad_psi**2,
        "grad_psi2_over_r2_face": grad_psi**2 / major_radius**2,
        "b2_face": b2_face,
        "inv_b2_face": 1.0 / b2_face,
        "r_in_face": major_radius - radius_face,
        "r_out_face": major_radius + radius_face,
        "elongation_face": np.ones_like(rho_face),
        "delta_upper_face": np.zeros_like(rho_face),
        "delta_lower_face": np.zeros_like(rho_face),
        "flux_sign": 1.0,
        "valid": True,
    }


def _initial_state(record):
    rho = np.asarray(record["rho_face"])
    return TransportState(
        rho=rho,
        psi=np.asarray(record["psi_face"]),
        ion_temperature=8.0 - 7.0 * rho**2,
        electron_temperature=7.0 - 6.0 * rho**2,
        electron_density=1.2e20 - 4.0e19 * rho**2,
    )


def _waveforms(record):
    current = float(record["ip_amperes"])
    return TransportWaveforms(
        time=np.array([0.0, 0.01]),
        plasma_current=np.array([current, current * 1.01]),
    )


def _torax_model():
    from torax._src.test_utils.default_configs import get_default_config_dict

    config = get_default_config_dict()
    config["numerics"].update(
        {
            "fixed_dt": 0.01,
            "max_dt": 0.01,
            "min_dt": 1.0e-8,
            "adaptive_dt": False,
        }
    )
    config["time_step_calculator"] = {"calculator_type": "fixed"}
    return TransportModel(TransportRung.TORAX_MULTI_CHANNEL, torax_config=config)


def _request(rung):
    record = _circular_record()
    model = (
        TransportModel(
            rung,
            eta=EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0),
        )
        if rung is TransportRung.NATIVE_PSI_DIFFUSION
        else _torax_model()
    )
    return ForwardTransportInput(
        geometry=TransportGeometry(record),
        initial_state=_initial_state(record),
        waveforms=_waveforms(record),
        model=model,
    )


def _assert_common_receipt(receipt, request):
    assert receipt.provenance.rung is request.model.rung
    assert receipt.diagnostics.steps >= 1
    assert receipt.diagnostics.outer_iterations >= 1
    assert (
        receipt.plasma_current.requested_initial == request.waveforms.plasma_current[0]
    )
    assert (
        receipt.plasma_current.requested_final == request.waveforms.plasma_current[-1]
    )
    assert receipt.boundary.psi == receipt.state.psi[-1]
    assert receipt.boundary.plasma_current == receipt.plasma_current.achieved_final
    assert np.isclose(
        receipt.flux_consumption.boundary,
        receipt.flux_consumption.resistive + receipt.flux_consumption.internal,
    )


def test_native_and_torax_rungs_share_the_typed_call_and_receipt(monkeypatch):
    """Both physics engines cross the same typed facade without mutating inputs."""
    jax.config.update("jax_platforms", "cpu")
    native_request = _request(TransportRung.NATIVE_PSI_DIFFUSION)
    native_snapshot = np.array(native_request.initial_state.psi)
    native_receipt = ForwardTransport().solve(native_request)
    _assert_common_receipt(native_receipt, native_request)
    np.testing.assert_array_equal(native_request.initial_state.psi, native_snapshot)
    assert native_receipt.provenance.engine == "nova.current_diffusion"
    assert np.isfinite(native_receipt.flux_consumption.mean_axis_voltage)
    assert native_receipt.plasma_current.achieved_final > 0.0

    torax_request = _request(TransportRung.TORAX_MULTI_CHANNEL)
    channel_snapshots = {
        name: np.array(getattr(torax_request.initial_state, name))
        for name in (
            "psi",
            "ion_temperature",
            "electron_temperature",
            "electron_density",
        )
    }
    real_run = forward_module._run_torax_simulation
    engine_calls = []

    def recorded_run(config):
        result = real_run(config)
        engine_calls.append((config, result))
        return result

    monkeypatch.setattr(forward_module, "_run_torax_simulation", recorded_run)
    torax_receipt = ForwardTransport().solve(torax_request)
    _assert_common_receipt(torax_receipt, torax_request)
    assert torax_receipt.provenance.engine == "torax"
    assert torax_receipt.diagnostics.engine_status == "NO_ERROR"
    assert torax_receipt.state.rho.size == torax_receipt.state.psi.size
    assert not np.array_equal(
        torax_receipt.state.ion_temperature,
        np.interp(
            torax_receipt.state.rho,
            torax_request.initial_state.rho,
            torax_request.initial_state.ion_temperature,
        ),
    )
    for name, snapshot in channel_snapshots.items():
        np.testing.assert_array_equal(
            getattr(torax_request.initial_state, name), snapshot
        )

    assert len(engine_calls) == 1
    direct_output, direct_history = real_run(engine_calls[0][0])
    assert direct_history.sim_error.name == "NO_ERROR"
    direct_profiles = direct_output.children["profiles"].dataset
    np.testing.assert_allclose(
        torax_receipt.state.psi,
        np.asarray(direct_profiles["psi"][-1]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torax_receipt.state.ion_temperature,
        np.asarray(direct_profiles["T_i"][-1]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torax_receipt.state.electron_temperature,
        np.asarray(direct_profiles["T_e"][-1]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torax_receipt.state.electron_density,
        np.asarray(direct_profiles["n_e"][-1]),
        rtol=1.0e-10,
        atol=1.0e8,
    )


def test_engine_failure_raises_without_a_fabricated_state(monkeypatch):
    request = _request(TransportRung.TORAX_MULTI_CHANNEL)

    class ErrorStatus:
        name = "NAN_DETECTED"

    class FailedHistory:
        sim_error = ErrorStatus()

    monkeypatch.setattr(
        forward_module,
        "_run_torax_simulation",
        lambda _config: (None, FailedHistory()),
    )
    with pytest.raises(TransportEngineError, match="NAN_DETECTED"):
        ForwardTransport().solve(request)


def test_input_containers_own_immutable_copies():
    record = _circular_record()
    record_snapshot = copy.deepcopy(record)
    geometry = TransportGeometry(record)
    state = _initial_state(record)
    waveform = _waveforms(record)

    record["psi_face"][0] = 99.0
    assert geometry.record["psi_face"][0] == record_snapshot["psi_face"][0]
    assert not geometry.record["psi_face"].flags.writeable
    assert not state.psi.flags.writeable
    assert not waveform.time.flags.writeable
    with pytest.raises(ValueError):
        state.psi[0] = 99.0
