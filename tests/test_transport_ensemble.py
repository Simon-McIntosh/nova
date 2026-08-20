"""Batch contract checks for deterministic forward transport."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from nova.biot.greens import MU0
from nova.transport.current_diffusion import EtaProfile
from nova.transport.ensemble import (
    EnsembleForwardTransport,
    EnsembleTransportInput,
    EnsembleTransportState,
)
from nova.transport.forward import (
    ForwardTransport,
    ForwardTransportInput,
    TransportGeometry,
    TransportModel,
    TransportRung,
    TransportState,
    TransportWaveforms,
)


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


def _state(record, offset):
    rho = np.asarray(record["rho_face"])
    return TransportState(
        rho=rho,
        psi=np.asarray(record["psi_face"]) + offset * (1.0 - rho**2) * 1.0e-3,
        ion_temperature=8.0 - 7.0 * rho**2 + offset * 0.2,
        electron_temperature=7.0 - 6.0 * rho**2 + offset * 0.1,
        electron_density=1.2e20 - 4.0e19 * rho**2 + offset * 1.0e18,
    )


def _model(rung):
    if rung is TransportRung.NATIVE_PSI_DIFFUSION:
        return TransportModel(
            rung,
            eta=EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0),
        )

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
    return TransportModel(rung, torax_config=config)


def _ensemble_request(rung):
    record = _circular_record()
    member_states = tuple(
        (member_id, _state(record, offset))
        for member_id, offset in (
            ("draw-blue", -1.0),
            ("draw-amber", 0.0),
            ("draw-violet", 1.0),
        )
    )
    current = float(record["ip_amperes"])
    return EnsembleTransportInput(
        geometry=TransportGeometry(record),
        initial_states=EnsembleTransportState.from_members(member_states),
        waveforms=TransportWaveforms(
            time=np.array([0.0, 0.01]),
            plasma_current=np.array([current, current * 1.01]),
        ),
        model=_model(rung),
    )


def _scalar_receipts(request):
    forward = ForwardTransport()
    return tuple(
        forward.solve(
            ForwardTransportInput(
                geometry=request.geometry,
                initial_state=request.initial_states.member(index),
                waveforms=request.waveforms,
                model=request.model,
            )
        )
        for index in range(len(request.initial_states.member_ids))
    )


def _numeric_receipt(receipt):
    return np.concatenate(
        [
            *(
                np.ravel(getattr(receipt.state, name))
                for name in (
                    "rho",
                    "psi",
                    "ion_temperature",
                    "electron_temperature",
                    "electron_density",
                )
            ),
            np.asarray(dataclasses.astuple(receipt.flux_consumption)),
            np.asarray(dataclasses.astuple(receipt.plasma_current)),
            np.asarray(dataclasses.astuple(receipt.boundary)),
            np.asarray(
                (
                    receipt.diagnostics.steps,
                    receipt.diagnostics.outer_iterations,
                    receipt.diagnostics.inner_iterations,
                )
            ),
        ]
    )


@pytest.mark.parametrize("rung", tuple(TransportRung))
def test_vmapped_members_match_scalar_loop_and_keep_identity(rung):
    jax.config.update("jax_platforms", "cpu")
    request = _ensemble_request(rung)
    expected = _scalar_receipts(request)
    actual = EnsembleForwardTransport().solve(request)

    assert len(actual.members) == 3
    assert actual.member_ids == request.initial_states.member_ids
    signatures = []
    for member_id, scalar in zip(actual.member_ids, expected, strict=True):
        recovered = actual.for_member(member_id)
        np.testing.assert_allclose(
            _numeric_receipt(recovered),
            _numeric_receipt(scalar),
            rtol=1.0e-10,
            atol=1.0e-8,
        )
        signatures.append(float(recovered.state.psi[0]))
        assert recovered.provenance == scalar.provenance
        assert recovered.diagnostics.engine_status == scalar.diagnostics.engine_status
    assert len(set(signatures)) == 3


@pytest.mark.parametrize("rung", tuple(TransportRung))
def test_jitted_and_eager_batches_agree(rung):
    jax.config.update("jax_platforms", "cpu")
    request = _ensemble_request(rung)
    eager = EnsembleForwardTransport().solve(request)
    compiled = EnsembleForwardTransport().solve(request, jit=True)

    assert compiled.member_ids == eager.member_ids
    for member_id in eager.member_ids:
        np.testing.assert_allclose(
            _numeric_receipt(compiled.for_member(member_id)),
            _numeric_receipt(eager.for_member(member_id)),
            rtol=1.0e-10,
            atol=1.0e-8,
        )


def test_member_state_batch_is_immutable_and_rejects_ambiguous_identity():
    record = _circular_record()
    state = _state(record, 0.0)
    batch = EnsembleTransportState.from_members((("first", state), ("second", state)))

    assert not batch.psi.flags.writeable
    with pytest.raises(ValueError):
        batch.psi[0, 0] = 1.0
    with pytest.raises(ValueError, match="unique"):
        EnsembleTransportState.from_members((("same", state), ("same", state)))
