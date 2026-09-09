"""Transport request and receipt coverage for the coupled equilibrium seam."""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from nova.equilibrium import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes
from nova.transport.coupled_window import (
    TransportSweepReceipt,
    Waveform,
    transport_sweep,
)
from nova.transport.forward import (
    ForwardTransport,
    ForwardTransportReceipt,
    TransportRung,
)
from tests.test_forward_transport import _request


configure_dtypes()


def _coupled_input():
    transport_input = _request(TransportRung.NATIVE_PSI_DIFFUSION)
    equilibrium_request = ForwardSolveRequest.from_defaults(
        carrier_identity="transport-forward-fixture",
        source_profile=object(),
        seed_policy=ExplicitSolveSeed(np.zeros(3, dtype=np.float64)),
    )
    return dataclasses.replace(
        transport_input,
        equilibrium_request=equilibrium_request,
    )


def test_native_coupled_receipt_round_trips_route_and_equilibrium_defaults():
    """A native interval records both sides of its coupled provenance."""

    transport_input = _coupled_input()
    receipt = ForwardTransport().solve(transport_input)
    payload = json.loads(json.dumps(receipt.to_dict()))
    restored = ForwardTransportReceipt.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.provenance.rung is TransportRung.NATIVE_PSI_DIFFUSION
    assert restored.equilibrium_resolved_defaults is not None
    assert restored.equilibrium_resolved_defaults.to_dict()["policy"]


def test_interval_ensemble_round_trips_every_transport_receipt():
    """Every interval in a coupled window retains the embedded defaults."""

    transport_input = _coupled_input()
    time = np.linspace(
        float(transport_input.waveforms.time[0]),
        float(transport_input.waveforms.time[-1]),
        3,
    )
    current = np.interp(
        time,
        transport_input.waveforms.time,
        transport_input.waveforms.plasma_current,
    )
    geometry = Waveform.from_geometries(
        transport_input.waveforms.time,
        (transport_input.geometry, transport_input.geometry),
    )
    sweep = transport_sweep(
        geometry,
        transport_input.initial_state,
        time,
        current,
        transport_input.model,
        equilibrium_request=transport_input.equilibrium_request,
    )
    payload = json.loads(json.dumps(sweep.to_dict()))
    restored = TransportSweepReceipt.from_dict(payload)

    assert len(restored.receipts) == 2
    assert restored.to_dict() == payload
    assert all(
        receipt.equilibrium_resolved_defaults is not None
        and receipt.equilibrium_resolved_defaults.to_dict()["policy"]
        and receipt.provenance.rung is TransportRung.NATIVE_PSI_DIFFUSION
        for receipt in restored.receipts
    )
