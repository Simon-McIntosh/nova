"""Member-preserving coupled-window batch contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest

from nova.equilibrium.topology import TopologyClass
from nova.transport import (
    BatchedCouplingState,
    BatchedExchangeSweepResult,
    BatchedWaveform,
    CouplingState,
    EquilibriumSweepReceipt,
    MemberArrayBatch,
    WindowBatchError,
    WindowBatchInput,
    WindowConfig,
    WindowRefusalReason,
    solve_window,
    solve_window_batch,
)
from tests.test_transport_coupled_window import (
    WINDOW_CONVERGENCE_TOLERANCE,
    _AffineWindow,
)


def _inputs(exchanges: tuple[tuple[str, _AffineWindow], ...]) -> WindowBatchInput:
    member_ids = tuple(member_id for member_id, _exchange in exchanges)
    states = tuple(
        (
            member_id,
            CouplingState(exchange.geometry_template, exchange.source_template),
        )
        for member_id, exchange in exchanges
    )
    count = len(member_ids)
    return WindowBatchInput(
        seam_state=MemberArrayBatch(
            member_ids,
            {
                "rho": np.broadcast_to(np.linspace(0.0, 1.0, 7), (count, 7)),
                "p_prime": np.arange(count * 7, dtype=np.float64).reshape(count, 7),
            },
        ),
        actuator_waveforms=MemberArrayBatch(
            member_ids,
            {"plasma_current": np.full((count, 4), 1.0e6)},
        ),
        geometry=MemberArrayBatch(
            member_ids,
            {"wall_outline": np.broadcast_to(np.eye(2), (count, 2, 2))},
        ),
        coupling_state=BatchedCouplingState.from_members(states),
    )


class _BatchAffine:
    def __init__(self, exchanges: tuple[tuple[str, _AffineWindow], ...]):
        self.exchanges = exchanges
        self.transport_calls = 0
        self.equilibrium_calls = 0

    def transport(self, _inputs, geometry, sample_grid):
        self.transport_calls += 1
        results = tuple(
            exchange.transport(geometry.member(member_id), sample_grid)
            for member_id, exchange in self.exchanges
        )
        return BatchedExchangeSweepResult(
            BatchedWaveform.from_members(
                tuple(
                    (member_id, result.waveform)
                    for (member_id, _exchange), result in zip(
                        self.exchanges, results, strict=True
                    )
                )
            ),
            tuple(result.receipt for result in results),
        )

    def equilibrium(self, _inputs, source, sample_grid):
        self.equilibrium_calls += 1
        results = tuple(
            exchange.equilibrium(source.member(member_id), sample_grid)
            for member_id, exchange in self.exchanges
        )
        return BatchedExchangeSweepResult(
            BatchedWaveform.from_members(
                tuple(
                    (member_id, result.waveform)
                    for (member_id, _exchange), result in zip(
                        self.exchanges, results, strict=True
                    )
                )
            ),
            tuple(result.receipt for result in results),
        )


def _assert_tree_equal(left, right):
    if dataclasses.is_dataclass(left):
        assert type(left) is type(right)
        for field in dataclasses.fields(left):
            _assert_tree_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        assert set(left) == set(right)
        for name in left:
            _assert_tree_equal(left[name], right[name])
    elif isinstance(left, tuple):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_tree_equal(left_value, right_value)
    else:
        assert left == right


def test_batch_of_one_matches_scalar_full_window_receipt_trajectory():
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=180,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    scalar_exchange = _AffineWindow(config, coupling=0.2)
    scalar = solve_window(
        scalar_exchange.geometry_template,
        scalar_exchange.source_template,
        config,
        scalar_exchange.equilibrium,
        scalar_exchange.transport,
    )
    batch_exchange = _AffineWindow(config, coupling=0.2)
    operators = _BatchAffine((("member-blue", batch_exchange),))
    batch = solve_window_batch(
        _inputs(operators.exchanges),
        config,
        operators.equilibrium,
        operators.transport,
    )

    assert batch.member_ids == ("member-blue",)
    _assert_tree_equal(batch.for_member("member-blue").window, scalar)
    assert operators.transport_calls == scalar.convergence.iterations_used
    assert operators.equilibrium_calls == scalar.convergence.iterations_used


def test_batch_preserves_member_identity_and_calls_each_side_once_per_exchange():
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=180,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    exchanges = tuple(
        (member_id, _AffineWindow(config, coupling=coupling, source_offset=offset))
        for member_id, coupling, offset in (
            ("draw-violet", 0.2, 1.1),
            ("draw-blue", 0.002, 0.9),
            ("draw-amber", 0.08, 1.0),
        )
    )
    operators = _BatchAffine(exchanges)
    batch = solve_window_batch(
        _inputs(exchanges), config, operators.equilibrium, operators.transport
    )

    assert batch.member_ids == tuple(member_id for member_id, _exchange in exchanges)
    maximum_iterations = max(
        member.convergence.iterations_used for member in batch.members
    )
    assert operators.transport_calls == maximum_iterations
    assert operators.equilibrium_calls == maximum_iterations
    for member_id, exchange in exchanges:
        member = batch.for_member(member_id)
        expected = solve_window(
            exchange.geometry_template,
            exchange.source_template,
            config,
            exchange.equilibrium,
            exchange.transport,
        )
        _assert_tree_equal(member.window, expected)
        assert member.transport_state is member.window.transport_receipt.state
        np.testing.assert_array_equal(
            member.fields.geometry.values["geometry"],
            expected.geometry_waveform.values["geometry"],
        )


def test_unconverged_member_is_a_typed_refusal_not_a_degraded_receipt():
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=10,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
        contraction_threshold=0.1,
        hard_iteration_ceiling=20,
        damping_floor=0.5,
    )
    exchanges = (
        ("accepted", _AffineWindow(config, coupling=0.002)),
        ("refused", _AffineWindow(config, coupling=-2.0)),
    )
    operators = _BatchAffine(exchanges)

    with pytest.raises(WindowBatchError, match="refused") as raised:
        solve_window_batch(
            _inputs(exchanges), config, operators.equilibrium, operators.transport
        )

    assert tuple(member.member_id for member in raised.value.admitted) == ("accepted",)
    assert len(raised.value.refusals) == 1
    refusal = raised.value.refusals[0]
    assert refusal.member_id == "refused"
    assert refusal.reason is WindowRefusalReason.CONVERGENCE
    assert refusal.error.convergence.gating_norm > config.tolerance


def test_member_receipt_exposes_topology_moments_and_equilibrium_ledgers():
    config = WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=180,
        tolerance=WINDOW_CONVERGENCE_TOLERANCE,
    )
    exchanges = (("tagged", _AffineWindow(config, coupling=0.002)),)
    operators = _BatchAffine(exchanges)
    original_equilibrium = operators.equilibrium

    def typed_equilibrium(inputs, source, sample_grid):
        result = original_equilibrium(inputs, source, sample_grid)
        equilibrium = SimpleNamespace(
            topology=SimpleNamespace(diverted=np.asarray(True)),
            moments=SimpleNamespace(plasma_current=1.0e6),
            conservation=SimpleNamespace(relative_divergence_b=0.0),
        )
        receipt = EquilibriumSweepReceipt(
            time=sample_grid,
            source_samples=(),
            equilibria=(equilibrium,),
            branch_receipts=(),
        )
        return BatchedExchangeSweepResult(result.waveform, (receipt,))

    batch = solve_window_batch(
        _inputs(exchanges), config, typed_equilibrium, operators.transport
    )
    member = batch.for_member("tagged")

    assert member.topology_class is TopologyClass.DIVERTED
    assert member.moments.plasma_current == 1.0e6
    assert member.equilibrium_conservation[0].relative_divergence_b == 0.0
    assert member.conservation is member.window.conservation


def test_explicit_array_payloads_reject_member_axis_or_identity_drift():
    with pytest.raises(ValueError, match="member axis first"):
        MemberArrayBatch(("one", "two"), {"pressure": np.ones((1, 5))})
    with pytest.raises(ValueError, match="unique"):
        MemberArrayBatch(("same", "same"), {"pressure": np.ones((2, 5))})
