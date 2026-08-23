"""Member-preserving batching for coupled transport--equilibrium windows.

The batch boundary is deliberately expressed in arrays and Nova-owned window
types.  An upstream consumer can project its state contract into
``MemberArrayBatch`` without Nova importing or recreating that contract.  Both
side operators receive the entire identified batch once per exchange, so a
device-vectorised equilibrium or transport implementation stays batched while
the host assembles the typed per-member receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from nova.equilibrium.topology import TopologyClass
from nova.transport.coupled_window import (
    CouplingState,
    EquilibriumSweepReceipt,
    TransportSweepReceipt,
    Waveform,
    WindowConfig,
    WindowConservationError,
    WindowConvergenceError,
    WindowConvergenceReceipt,
    WindowDampingBackoffReceipt,
    WindowReceipt,
    _contraction_estimate,
    _window_conservation,
)


def _member_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if not resolved or any(not value for value in resolved):
        raise ValueError("a window batch needs non-empty member identities")
    if len(set(resolved)) != len(resolved):
        raise ValueError("window batch member identities must be unique")
    return resolved


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MemberArrayBatch:
    """Explicit typed arrays whose leading axis is the identified member axis."""

    member_ids: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        member_ids = _member_ids(self.member_ids)
        arrays: dict[str, np.ndarray] = {}
        if not self.arrays:
            raise ValueError("a member array batch needs at least one named array")
        for name, value in self.arrays.items():
            array = _readonly(value)
            if array.ndim < 1 or array.shape[0] != len(member_ids):
                raise ValueError(
                    f"member array {name} must carry the member axis first"
                )
            if array.dtype.kind not in "biufc":
                raise TypeError(f"member array {name} must have a numeric dtype")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"member array {name} must be finite")
            arrays[str(name)] = array
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))

    def member(self, member_id: str) -> Mapping[str, np.ndarray]:
        """Return one identity-selected array view without positional guessing."""
        try:
            index = self.member_ids.index(member_id)
        except ValueError as error:
            raise KeyError(member_id) from error
        return MappingProxyType(
            {name: value[index] for name, value in self.arrays.items()}
        )


@dataclass(frozen=True, slots=True)
class BatchedWaveform:
    """Compatible waveforms stacked on an explicit leading member axis."""

    member_ids: tuple[str, ...]
    time: np.ndarray
    radial_grid: np.ndarray
    phi_boundary: np.ndarray
    axis_reference: np.ndarray
    boundary_reference: np.ndarray
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        member_ids = _member_ids(self.member_ids)
        time = _readonly(self.time)
        radial_grid = _readonly(self.radial_grid)
        phi_boundary = _readonly(self.phi_boundary)
        axis_reference = _readonly(self.axis_reference)
        boundary_reference = _readonly(self.boundary_reference)
        if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0.0):
            raise ValueError("a batched waveform needs one increasing shared time grid")
        member_count = len(member_ids)
        if radial_grid.ndim != 3 or radial_grid.shape[:2] != (
            member_count,
            time.size,
        ):
            raise ValueError(
                "batched radial grids must have shape (member, time, radius)"
            )
        expected_coordinates = (member_count, time.size)
        for name, coordinate in (
            ("phi_boundary", phi_boundary),
            ("axis_reference", axis_reference),
            ("boundary_reference", boundary_reference),
        ):
            if coordinate.shape != expected_coordinates:
                raise ValueError(f"batched {name} must have shape (member, time)")
        frozen_values: dict[str, np.ndarray] = {}
        for name, value in self.values.items():
            array = _readonly(value)
            if array.ndim < 2 or array.shape[:2] != expected_coordinates:
                raise ValueError(
                    f"batched waveform channel {name} must begin with member and time"
                )
            frozen_values[str(name)] = array
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "radial_grid", radial_grid)
        object.__setattr__(self, "phi_boundary", phi_boundary)
        object.__setattr__(self, "axis_reference", axis_reference)
        object.__setattr__(self, "boundary_reference", boundary_reference)
        object.__setattr__(self, "values", MappingProxyType(frozen_values))

    @classmethod
    def from_members(cls, members: tuple[tuple[str, Waveform], ...]) -> BatchedWaveform:
        """Stack compatible identified scalar waveforms."""
        if not members:
            raise ValueError("a batched waveform needs at least one member")
        member_ids = tuple(member_id for member_id, _waveform in members)
        waveforms = tuple(waveform for _member_id, waveform in members)
        first = waveforms[0]
        if any(not np.array_equal(waveform.time, first.time) for waveform in waveforms):
            raise ValueError("all member waveforms must share one time grid")
        if any(set(waveform.values) != set(first.values) for waveform in waveforms):
            raise ValueError("all member waveforms must carry the same channels")
        try:
            return cls(
                member_ids=member_ids,
                time=first.time,
                radial_grid=np.stack([waveform.radial_grid for waveform in waveforms]),
                phi_boundary=np.stack(
                    [waveform.phi_boundary for waveform in waveforms]
                ),
                axis_reference=np.stack(
                    [waveform.axis_reference for waveform in waveforms]
                ),
                boundary_reference=np.stack(
                    [waveform.boundary_reference for waveform in waveforms]
                ),
                values={
                    name: np.stack([waveform.values[name] for waveform in waveforms])
                    for name in first.values
                },
            )
        except ValueError as error:
            raise ValueError(
                "all member waveform fields must have compatible shapes"
            ) from error

    def member(self, member_id: str) -> Waveform:
        """Recover one scalar waveform by identity."""
        try:
            index = self.member_ids.index(member_id)
        except ValueError as error:
            raise KeyError(member_id) from error
        return Waveform(
            time=self.time,
            radial_grid=self.radial_grid[index],
            phi_boundary=self.phi_boundary[index],
            axis_reference=self.axis_reference[index],
            boundary_reference=self.boundary_reference[index],
            values={name: value[index] for name, value in self.values.items()},
        )


@dataclass(frozen=True, slots=True)
class BatchedCouplingState:
    """Identified leading-axis representation of the coupled boundary state."""

    geometry: BatchedWaveform
    source: BatchedWaveform

    def __post_init__(self) -> None:
        if self.geometry.member_ids != self.source.member_ids:
            raise ValueError(
                "coupling-state sides must use identical member identities"
            )

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.geometry.member_ids

    @classmethod
    def from_members(
        cls, members: tuple[tuple[str, CouplingState], ...]
    ) -> BatchedCouplingState:
        """Stack scalar coupling states without changing their field contract."""
        return cls(
            geometry=BatchedWaveform.from_members(
                tuple((member_id, state.geometry) for member_id, state in members)
            ),
            source=BatchedWaveform.from_members(
                tuple((member_id, state.source) for member_id, state in members)
            ),
        )

    def member(self, member_id: str) -> CouplingState:
        return CouplingState(
            geometry=self.geometry.member(member_id),
            source=self.source.member(member_id),
        )


@dataclass(frozen=True, slots=True)
class WindowBatchInput:
    """All explicit deterministic inputs for one identified member batch."""

    seam_state: MemberArrayBatch
    actuator_waveforms: MemberArrayBatch
    geometry: MemberArrayBatch
    coupling_state: BatchedCouplingState

    def __post_init__(self) -> None:
        member_ids = self.coupling_state.member_ids
        for name, values in (
            ("seam state", self.seam_state),
            ("actuator waveforms", self.actuator_waveforms),
            ("geometry", self.geometry),
        ):
            if values.member_ids != member_ids:
                raise ValueError(
                    f"{name} member identities differ from the coupling state"
                )

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.coupling_state.member_ids


@dataclass(frozen=True, slots=True)
class BatchedExchangeSweepResult:
    """One batched side update and its identity-aligned scalar receipts."""

    waveform: BatchedWaveform
    receipts: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.receipts) != len(self.waveform.member_ids):
            raise ValueError("a batched side receipt is required for every member")


@dataclass(frozen=True, slots=True)
class WindowMemberReceipt:
    """One admitted member identity paired with its full scalar window receipt."""

    member_id: str
    window: WindowReceipt

    @property
    def fields(self) -> CouplingState:
        return self.window.coupling_state

    @property
    def transport_state(self):
        return self.window.transport_receipt.state

    def _final_equilibrium(self):
        receipt = self.window.equilibrium_receipt
        if not isinstance(receipt, EquilibriumSweepReceipt) or not receipt.equilibria:
            raise TypeError(
                "member topology and moments require an EquilibriumSweepReceipt"
            )
        return receipt.equilibria[-1]

    @property
    def topology_class(self) -> TopologyClass:
        equilibrium = self._final_equilibrium()
        return (
            TopologyClass.DIVERTED
            if bool(equilibrium.topology.diverted)
            else TopologyClass.LIMITED
        )

    @property
    def moments(self):
        return self._final_equilibrium().moments

    @property
    def equilibrium_conservation(self):
        return tuple(self.window.equilibrium_receipt.conservation)

    @property
    def conservation(self):
        return self.window.conservation

    @property
    def convergence(self) -> WindowConvergenceReceipt:
        return self.window.convergence


@dataclass(frozen=True, slots=True)
class WindowBatchReceipt:
    """Ordered admitted members returned only after every member converges."""

    members: tuple[WindowMemberReceipt, ...]

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.member_id for member in self.members)

    def for_member(self, member_id: str) -> WindowMemberReceipt:
        for member in self.members:
            if member.member_id == member_id:
                return member
        raise KeyError(member_id)


class WindowRefusalReason(StrEnum):
    """Typed reason an individual member was not admitted."""

    CONVERGENCE = "convergence"
    CONSERVATION = "conservation"


@dataclass(frozen=True, slots=True)
class WindowMemberRefusal:
    """One refused identity and the scalar typed error proving why."""

    member_id: str
    reason: WindowRefusalReason
    error: WindowConvergenceError | WindowConservationError


class WindowBatchError(RuntimeError):
    """A batch containing refused members; no degraded batch receipt is returned."""

    def __init__(
        self,
        admitted: tuple[WindowMemberReceipt, ...],
        refusals: tuple[WindowMemberRefusal, ...],
    ) -> None:
        identities = ", ".join(refusal.member_id for refusal in refusals)
        super().__init__(
            f"window batch refused {len(refusals)} member(s): {identities}"
        )
        self.admitted = admitted
        self.refusals = refusals


BatchUpdate = Callable[
    [WindowBatchInput, BatchedWaveform, np.ndarray], BatchedExchangeSweepResult
]


def solve_window_batch(
    inputs: WindowBatchInput,
    config: WindowConfig,
    equilibrium_update: BatchUpdate,
    transport_update: BatchUpdate,
    *,
    damping: float = 1.0,
    failure_serializer: Callable[[WindowBatchError], None] | None = None,
) -> WindowBatchReceipt:
    """Advance all members through one batched fixed-point exchange.

    Each side operator is called once per batch exchange, never once per
    member.  Receipt assembly remains on the host and preserves the scalar
    ``solve_window`` trajectory for a batch of one.  A member that misses
    convergence or conservation is represented by ``WindowMemberRefusal``;
    any refusal raises ``WindowBatchError`` and prevents a degraded batch from
    being mistaken for an admitted ensemble.
    """
    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("window damping must lie in (0, 1]")
    if damping < config.damping_floor:
        raise ValueError("initial window damping cannot be below its declared floor")

    member_ids = inputs.member_ids
    states = [inputs.coupling_state.member(member_id) for member_id in member_ids]
    residual_traces: list[list[Mapping[str, float]]] = [[] for _ in member_ids]
    gating_traces: list[list[float]] = [[] for _ in member_ids]
    all_field_traces: list[list[float]] = [[] for _ in member_ids]
    continuation_traces: list[list[float]] = [[] for _ in member_ids]
    backoff_traces: list[list[WindowDampingBackoffReceipt]] = [[] for _ in member_ids]
    dampings = [float(damping) for _ in member_ids]
    active = np.ones(len(member_ids), dtype=bool)
    admitted: list[WindowMemberReceipt] = []
    refusals: list[WindowMemberRefusal] = []

    for iteration in range(1, config.effective_hard_iteration_ceiling + 1):
        batched_state = BatchedCouplingState.from_members(
            tuple(zip(member_ids, states, strict=True))
        )
        transported = transport_update(
            inputs, batched_state.geometry, config.transport_grid
        )
        if not isinstance(transported, BatchedExchangeSweepResult):
            raise TypeError(
                "batch transport update must return BatchedExchangeSweepResult"
            )
        equilibrated = equilibrium_update(
            inputs, transported.waveform, config.equilibrium_grid
        )
        if not isinstance(equilibrated, BatchedExchangeSweepResult):
            raise TypeError(
                "batch equilibrium update must return BatchedExchangeSweepResult"
            )
        if transported.waveform.member_ids != member_ids:
            raise ValueError("batch transport update changed member identities")
        if equilibrated.waveform.member_ids != member_ids:
            raise ValueError("batch equilibrium update changed member identities")

        for index, member_id in enumerate(member_ids):
            if not active[index]:
                continue
            transport_receipt = transported.receipts[index]
            if not isinstance(transport_receipt, TransportSweepReceipt):
                raise TypeError(
                    "each batch transport member needs a TransportSweepReceipt"
                )
            candidate = CouplingState(
                geometry=equilibrated.waveform.member(member_id),
                source=transported.waveform.member(member_id),
            )
            residual = states[index].residual(candidate)
            residual_traces[index].append(residual)
            gating_traces[index].append(
                states[index].residual_norm(residual, include_excluded=False)
            )
            all_field_traces[index].append(
                states[index].residual_norm(residual, include_excluded=True)
            )
            convergence = WindowConvergenceReceipt(
                iterations_used=iteration,
                contraction_estimate=_contraction_estimate(gating_traces[index]),
                exit_residual=residual,
                damping_applied=dampings[index],
                residual_trace=tuple(residual_traces[index]),
                gating_norm_trace=tuple(gating_traces[index]),
                all_field_norm_trace=tuple(all_field_traces[index]),
                iterations_past_cap=max(0, iteration - config.iteration_cap),
                continuation_contractions=tuple(continuation_traces[index]),
                damping_backoffs=tuple(backoff_traces[index]),
            )
            if convergence.gating_norm <= config.tolerance:
                conservation = _window_conservation(transport_receipt)
                if (
                    conservation.flux_closure_residual > config.tolerance
                    or conservation.current_continuity_residual > config.tolerance
                ):
                    error = WindowConservationError(conservation)
                    refusals.append(
                        WindowMemberRefusal(
                            member_id,
                            WindowRefusalReason.CONSERVATION,
                            error,
                        )
                    )
                else:
                    admitted.append(
                        WindowMemberReceipt(
                            member_id,
                            WindowReceipt(
                                geometry_waveform=candidate.geometry,
                                source_waveform=candidate.source,
                                equilibrium_receipt=equilibrated.receipts[index],
                                transport_receipt=transport_receipt,
                                convergence=convergence,
                                conservation=conservation,
                            ),
                        )
                    )
                active[index] = False
                continue

            contraction = convergence.contraction_estimate
            hard_ceiling_reached = iteration == config.effective_hard_iteration_ceiling
            contraction_unavailable = iteration > 1 and (
                contraction is None or not np.isfinite(contraction)
            )
            stalled_at_floor = (
                contraction is not None
                and np.isfinite(contraction)
                and contraction >= config.contraction_threshold
                and dampings[index] <= config.damping_floor
            )
            cap_needs_measurement = (
                iteration >= config.iteration_cap and contraction is None
            )
            if (
                hard_ceiling_reached
                or contraction_unavailable
                or stalled_at_floor
                or cap_needs_measurement
            ):
                error = WindowConvergenceError(
                    convergence,
                    candidate.geometry,
                    candidate.source,
                    equilibrated.receipts[index],
                    transport_receipt,
                )
                refusals.append(
                    WindowMemberRefusal(
                        member_id,
                        WindowRefusalReason.CONVERGENCE,
                        error,
                    )
                )
                active[index] = False
                continue
            if contraction is not None and contraction >= config.contraction_threshold:
                damping_before = dampings[index]
                dampings[index] = max(config.damping_floor, 0.5 * dampings[index])
                backoff_traces[index].append(
                    WindowDampingBackoffReceipt(
                        iteration=iteration,
                        trigger_contraction=float(contraction),
                        damping_before=damping_before,
                        damping_after=dampings[index],
                    )
                )
            if iteration >= config.iteration_cap:
                if contraction is None:
                    raise AssertionError("a continuation needs a measured contraction")
                continuation_traces[index].append(float(contraction))
            states[index] = states[index].blend(candidate, dampings[index])

        if not np.any(active):
            break

    admitted_by_id = {member.member_id: member for member in admitted}
    admitted_ordered = tuple(
        admitted_by_id[member_id]
        for member_id in member_ids
        if member_id in admitted_by_id
    )
    if refusals:
        error = WindowBatchError(admitted_ordered, tuple(refusals))
        if failure_serializer is not None:
            failure_serializer(error)
        raise error
    return WindowBatchReceipt(admitted_ordered)


__all__ = [
    "BatchedCouplingState",
    "BatchedExchangeSweepResult",
    "BatchedWaveform",
    "MemberArrayBatch",
    "WindowBatchError",
    "WindowBatchInput",
    "WindowBatchReceipt",
    "WindowMemberReceipt",
    "WindowMemberRefusal",
    "WindowRefusalReason",
    "solve_window_batch",
]
