"""Identity-preserving ensemble execution for deterministic transport.

The scalar :class:`~nova.transport.forward.ForwardTransport` contract owns the
physics and its engine-specific failure handling.  This module adds only the
batching boundary: member states are stacked on a leading axis, mapped through
the scalar forward with :func:`jax.vmap`, and restored as individually
addressable scalar receipts.  Geometry, waveforms, and model selection remain
shared immutable inputs for the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from nova.transport.forward import (
    AchievedBoundaryValues,
    FluxConsumptionLedger,
    ForwardTransport,
    ForwardTransportInput,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportGeometry,
    TransportModel,
    TransportProvenance,
    TransportRung,
    TransportState,
    TransportWaveforms,
)


_STATE_CHANNELS = (
    "rho",
    "psi",
    "ion_temperature",
    "electron_temperature",
    "electron_density",
)


def _readonly_array(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EnsembleTransportState:
    """Frozen transport states stacked as ``(member, radial_sample)`` arrays."""

    member_ids: tuple[str, ...]
    rho: np.ndarray
    psi: np.ndarray
    ion_temperature: np.ndarray
    electron_temperature: np.ndarray
    electron_density: np.ndarray

    def __post_init__(self) -> None:
        member_ids = tuple(str(member_id) for member_id in self.member_ids)
        if not member_ids:
            raise ValueError("an ensemble must contain at least one member")
        if any(not member_id for member_id in member_ids):
            raise ValueError("ensemble member identities must be non-empty")
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("ensemble member identities must be unique")
        object.__setattr__(self, "member_ids", member_ids)

        expected_shape = None
        for name in _STATE_CHANNELS:
            array = _readonly_array(getattr(self, name))
            object.__setattr__(self, name, array)
            if array.ndim != 2:
                raise ValueError(
                    f"ensemble state channel {name} must have member and radial axes"
                )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError("all ensemble state channels must share one shape")

        if expected_shape is None or expected_shape[0] != len(member_ids):
            raise ValueError("the leading state axis must match the member identities")
        if expected_shape[1] < 3:
            raise ValueError("transport states need at least three radial samples")
        if np.any(np.diff(self.rho, axis=1) <= 0.0):
            raise ValueError("each member rho grid must be strictly increasing")
        if np.any(self.rho[:, 0] < 0.0) or np.any(self.rho[:, -1] > 1.0):
            raise ValueError("each member rho grid must lie in [0, 1]")

    @classmethod
    def from_members(
        cls, members: tuple[tuple[str, TransportState], ...]
    ) -> EnsembleTransportState:
        """Stack identified scalar states without retaining caller-owned arrays."""
        if not members:
            raise ValueError("an ensemble must contain at least one member")
        return cls(
            member_ids=tuple(member_id for member_id, _state in members),
            **{
                name: np.stack(
                    [np.asarray(getattr(state, name)) for _member_id, state in members]
                )
                for name in _STATE_CHANNELS
            },
        )

    def member(self, index: int) -> TransportState:
        """Return one scalar state at its original batch position."""
        return TransportState(
            **{name: getattr(self, name)[index] for name in _STATE_CHANNELS}
        )


@dataclass(frozen=True)
class EnsembleTransportInput:
    """Shared forward inputs plus the identified member-state batch."""

    geometry: TransportGeometry
    initial_states: EnsembleTransportState
    waveforms: TransportWaveforms
    model: TransportModel

    def member(self, index: int) -> ForwardTransportInput:
        """Recover the scalar request associated with one batch position."""
        return ForwardTransportInput(
            geometry=self.geometry,
            initial_state=self.initial_states.member(index),
            waveforms=self.waveforms,
            model=self.model,
        )


@dataclass(frozen=True)
class EnsembleMemberReceipt:
    """One member identity paired inseparably with its scalar receipt."""

    member_id: str
    transport: ForwardTransportReceipt


@dataclass(frozen=True)
class EnsembleTransportReceipt:
    """Ordered member receipts with identity-based lookup."""

    members: tuple[EnsembleMemberReceipt, ...]

    @property
    def member_ids(self) -> tuple[str, ...]:
        """Return identities in the exact input order."""
        return tuple(member.member_id for member in self.members)

    def for_member(self, member_id: str) -> ForwardTransportReceipt:
        """Recover one member's receipt by identity, independent of position."""
        for member in self.members:
            if member.member_id == member_id:
                return member.transport
        raise KeyError(member_id)


class EnsembleForwardTransport:
    """Apply a scalar :class:`ForwardTransport` over an identified state batch."""

    def __init__(self, forward: ForwardTransport | None = None) -> None:
        self._forward = forward or ForwardTransport()

    def solve(
        self, inputs: EnsembleTransportInput, *, jit: bool = False
    ) -> EnsembleTransportReceipt:
        """Advance all members with ``vmap`` and preserve scalar receipts."""
        callback = self._member_callback(inputs)
        radial_samples = inputs.initial_states.rho.shape[1]
        if inputs.model.rung is TransportRung.TORAX_MULTI_CHANNEL:
            radial_samples = np.asarray(inputs.geometry.record["rho_face"]).size + 1
        output_spec = (
            jax.ShapeDtypeStruct((len(_STATE_CHANNELS), radial_samples), jnp.float64),
            jax.ShapeDtypeStruct((14,), jnp.float64),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )

        def solve_member(rho, psi, ion_temperature, electron_temperature, density):
            return jax.pure_callback(
                callback,
                output_spec,
                rho,
                psi,
                ion_temperature,
                electron_temperature,
                density,
                vmap_method="sequential",
            )

        mapped: Callable[..., tuple[jax.Array, jax.Array, jax.Array]] = jax.vmap(
            solve_member
        )
        if jit:
            mapped = jax.jit(mapped)
        state_values, receipt_values, diagnostic_values = mapped(
            *(
                jnp.asarray(getattr(inputs.initial_states, name), dtype=jnp.float64)
                for name in _STATE_CHANNELS
            )
        )
        return self._restore_receipts(
            inputs,
            np.asarray(state_values),
            np.asarray(receipt_values),
            np.asarray(diagnostic_values),
        )

    def _member_callback(self, inputs: EnsembleTransportInput):
        def callback(*channels):
            request = ForwardTransportInput(
                geometry=inputs.geometry,
                initial_state=TransportState(
                    **dict(zip(_STATE_CHANNELS, channels, strict=True))
                ),
                waveforms=inputs.waveforms,
                model=inputs.model,
            )
            receipt = self._forward.solve(request)
            return _pack_receipt(receipt)

        return callback

    @staticmethod
    def _restore_receipts(
        inputs: EnsembleTransportInput,
        state_values: np.ndarray,
        receipt_values: np.ndarray,
        diagnostic_values: np.ndarray,
    ) -> EnsembleTransportReceipt:
        provenance = _provenance(inputs.model.rung)
        members = []
        for index, member_id in enumerate(inputs.initial_states.member_ids):
            state = TransportState(
                **{
                    name: state_values[index, channel]
                    for channel, name in enumerate(_STATE_CHANNELS)
                }
            )
            values = receipt_values[index]
            diagnostics = diagnostic_values[index]
            members.append(
                EnsembleMemberReceipt(
                    member_id=member_id,
                    transport=ForwardTransportReceipt(
                        state=state,
                        flux_consumption=FluxConsumptionLedger(*values[0:5]),
                        plasma_current=PlasmaCurrentLedger(*values[5:9]),
                        boundary=AchievedBoundaryValues(*values[9:14]),
                        diagnostics=SolverDiagnostics(
                            engine_status=(
                                "converged"
                                if inputs.model.rung
                                is TransportRung.NATIVE_PSI_DIFFUSION
                                else "NO_ERROR"
                            ),
                            steps=int(diagnostics[0]),
                            outer_iterations=int(diagnostics[1]),
                            inner_iterations=int(diagnostics[2]),
                        ),
                        provenance=provenance,
                    ),
                )
            )
        return EnsembleTransportReceipt(tuple(members))


def _pack_receipt(receipt: ForwardTransportReceipt):
    state = np.stack(
        [
            np.asarray(getattr(receipt.state, name), dtype=np.float64)
            for name in _STATE_CHANNELS
        ]
    )
    values = np.asarray(
        (
            receipt.flux_consumption.boundary,
            receipt.flux_consumption.resistive,
            receipt.flux_consumption.internal,
            receipt.flux_consumption.mean_axis_voltage,
            receipt.flux_consumption.mean_boundary_voltage,
            receipt.plasma_current.requested_initial,
            receipt.plasma_current.requested_final,
            receipt.plasma_current.achieved_initial,
            receipt.plasma_current.achieved_final,
            receipt.boundary.psi,
            receipt.boundary.plasma_current,
            receipt.boundary.ion_temperature,
            receipt.boundary.electron_temperature,
            receipt.boundary.electron_density,
        ),
        dtype=np.float64,
    )
    diagnostics = np.asarray(
        (
            receipt.diagnostics.steps,
            receipt.diagnostics.outer_iterations,
            receipt.diagnostics.inner_iterations,
        ),
        dtype=np.int32,
    )
    return state, values, diagnostics


def _provenance(rung: TransportRung) -> TransportProvenance:
    if rung is TransportRung.NATIVE_PSI_DIFFUSION:
        return TransportProvenance(
            rung=rung,
            engine="nova.current_diffusion",
            engine_version="1",
        )
    return TransportProvenance(
        rung=rung,
        engine="torax",
        engine_version=version("torax"),
    )


__all__ = [
    "EnsembleForwardTransport",
    "EnsembleMemberReceipt",
    "EnsembleTransportInput",
    "EnsembleTransportReceipt",
    "EnsembleTransportState",
]
