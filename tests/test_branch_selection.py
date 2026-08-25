"""Case matrix for pure post-solve topology branch selection."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import (
    BranchAdmissibility,
    SelectionHistory,
    SelectionPolicy,
    SelectionReason,
    select_forward_branch,
)
from nova.equilibrium.forward import ForwardPortfolio
from nova.equilibrium.branch_selection import (
    TracedSelectionInput,
    forward_branch_selection_input,
    initial_traced_selection_state,
    scan_forward_branch_selection,
)
from nova.equilibrium.topology import TopologyClass


def _portfolio(
    limited_converged: bool,
    diverted_converged: bool,
    *,
    limited_consistent: bool = True,
    diverted_consistent: bool = True,
    residuals: tuple[float, float] = (1.0e-9, 1.0e-9),
) -> ForwardPortfolio:
    """Return the minimal real portfolio envelope the selector consumes."""

    branches = SimpleNamespace(
        converged=(limited_converged, diverted_converged),
        topology_consistent=(limited_consistent, diverted_consistent),
        residual=residuals,
    )
    return ForwardPortfolio(branches=branches)


def _traced_sequence(
    availability: list[tuple[bool, bool]],
    admissibility: list[tuple[bool, bool]],
    *,
    probabilities: list[float] | None = None,
    margins: list[float] | None = None,
) -> TracedSelectionInput:
    """Build time-major array evidence with distinct branch flux maps."""

    count = len(availability)
    if probabilities is None:
        probabilities = [0.25] * count
    if margins is None:
        margins = [-0.01] * count
    flux = jnp.broadcast_to(
        jnp.asarray([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        (count, 2, 3),
    )
    branches = SimpleNamespace(
        equilibrium=SimpleNamespace(flux=flux),
        converged=jnp.asarray(availability),
        topology_consistent=jnp.ones((count, 2), dtype=bool),
        residual=jnp.ones((count, 2)) * 1.0e-9,
    )
    return forward_branch_selection_input(
        branches,
        jnp.asarray(probabilities),
        jnp.asarray(margins),
        jnp.asarray(admissibility),
    )


@pytest.fixture
def policy() -> SelectionPolicy:
    """Declare a limited cold start and three-slice transition persistence."""

    return SelectionPolicy(
        cold_start_class=TopologyClass.LIMITED,
        persistence_threshold=3,
    )


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        ((True, False), TopologyClass.LIMITED),
        ((False, True), TopologyClass.DIVERTED),
    ),
)
def test_sole_valid_branch_wins_without_history(policy, available, expected):
    receipt = select_forward_branch(_portfolio(*available), SelectionHistory(), policy)
    assert receipt.selected_class is expected
    assert receipt.reason is SelectionReason.SOLE_VALID
    assert not receipt.switched


def test_both_valid_on_cold_start_use_the_declared_scenario_class():
    policy = SelectionPolicy(
        cold_start_class=TopologyClass.DIVERTED,
        persistence_threshold=2,
    )
    receipt = select_forward_branch(_portfolio(True, True), SelectionHistory(), policy)
    assert receipt.selected_class is TopologyClass.DIVERTED
    assert receipt.reason is SelectionReason.COLD_START
    assert not receipt.switched


def test_neither_valid_returns_no_selected_branch(policy):
    receipt = select_forward_branch(
        _portfolio(False, False),
        SelectionHistory(selected_class=TopologyClass.LIMITED),
        policy,
    )
    assert receipt.selected_class is None
    assert receipt.reason is SelectionReason.NO_VALID_BRANCH
    assert not receipt.switched
    assert receipt.next_history.selected_class is TopologyClass.LIMITED


@pytest.mark.parametrize("prior", (TopologyClass.LIMITED, TopologyClass.DIVERTED))
def test_tied_valid_branches_preserve_history(policy, prior):
    receipt = select_forward_branch(
        _portfolio(True, True, residuals=(4.0e-10, 4.0e-10)),
        SelectionHistory(selected_class=prior),
        policy,
    )
    assert receipt.selected_class is prior
    assert receipt.reason is SelectionReason.HISTORY_CONTINUITY
    assert not receipt.switched


@pytest.mark.parametrize(
    ("admissibility", "expected"),
    (
        (BranchAdmissibility(limited=False), TopologyClass.DIVERTED),
        (BranchAdmissibility(diverted=False), TopologyClass.LIMITED),
    ),
)
def test_admissibility_disqualifies_an_otherwise_converged_cold_candidate(
    policy, admissibility, expected
):
    receipt = select_forward_branch(
        _portfolio(True, True),
        SelectionHistory(),
        policy,
        admissibility,
    )
    assert receipt.selected_class is expected
    assert receipt.reason is SelectionReason.SOLE_VALID


def test_topology_inconsistency_disqualifies_a_converged_candidate(policy):
    receipt = select_forward_branch(
        _portfolio(True, True, limited_consistent=False),
        SelectionHistory(),
        policy,
    )
    assert receipt.selected_class is TopologyClass.DIVERTED
    assert receipt.reason is SelectionReason.SOLE_VALID
    assert not receipt.availability.limited


def test_branch_disappearance_switches_immediately(policy):
    receipt = select_forward_branch(
        _portfolio(False, True),
        SelectionHistory(selected_class=TopologyClass.LIMITED),
        policy,
    )
    assert receipt.selected_class is TopologyClass.DIVERTED
    assert receipt.reason is SelectionReason.BRANCH_DISAPPEARED
    assert receipt.switched
    assert receipt.next_history.pending_count == 0


def test_persistent_admissibility_transition_switches_once_at_declared_slice(policy):
    history = SelectionHistory()
    admissibility_sequence = (
        BranchAdmissibility(),
        BranchAdmissibility(),
        BranchAdmissibility(limited=False),
        BranchAdmissibility(limited=False),
        BranchAdmissibility(limited=False),
        BranchAdmissibility(),
    )
    expected = (
        TopologyClass.LIMITED,
        TopologyClass.LIMITED,
        TopologyClass.LIMITED,
        TopologyClass.LIMITED,
        TopologyClass.DIVERTED,
        TopologyClass.DIVERTED,
    )
    reasons = []
    selections = []
    switches = []
    for admissibility in admissibility_sequence:
        receipt = select_forward_branch(
            _portfolio(True, True), history, policy, admissibility
        )
        selections.append(receipt.selected_class)
        switches.append(receipt.switched)
        reasons.append(receipt.reason)
        history = receipt.next_history

    assert tuple(selections) == expected
    assert switches == [False, False, False, False, True, False]
    assert sum(switches) == 1
    assert reasons == [
        SelectionReason.COLD_START,
        SelectionReason.HISTORY_CONTINUITY,
        SelectionReason.ADMISSIBILITY_PENDING,
        SelectionReason.ADMISSIBILITY_PENDING,
        SelectionReason.ADMISSIBILITY_PERSISTED,
        SelectionReason.HISTORY_CONTINUITY,
    ]
    assert history.sequence_index == len(expected)


def test_selection_receipt_records_policy_criteria_and_switch_reason(policy):
    receipt = select_forward_branch(
        _portfolio(False, True),
        SelectionHistory(selected_class=TopologyClass.LIMITED, sequence_index=8),
        policy,
    )
    data = receipt.as_dict()
    assert data["sequence_index"] == 8
    assert data["selected_class"] == "diverted"
    assert data["previous_class"] == "limited"
    assert data["switched"] is True
    assert data["reason"] == "branch_disappeared"
    assert data["policy"] == {
        "cold_start_rule": "declared_class",
        "cold_start_class": "limited",
        "persistence_threshold": 3,
        "disappearance_criterion": "immediate_alternate",
        "admissibility_criterion": "persistent_alternate",
    }


def test_selection_is_referentially_transparent(policy):
    history = SelectionHistory(selected_class=TopologyClass.DIVERTED)
    portfolio = _portfolio(True, True)
    first = select_forward_branch(portfolio, history, policy)
    second = select_forward_branch(portfolio, history, policy)
    assert first == second
    assert history == SelectionHistory(selected_class=TopologyClass.DIVERTED)


def test_policy_rejects_a_zero_persistence_threshold():
    with pytest.raises(ValueError, match="at least one"):
        SelectionPolicy(
            cold_start_class=TopologyClass.LIMITED,
            persistence_threshold=0,
        )


def test_traced_scan_reproduces_the_host_selector_sequence(policy):
    admissibility_pairs = [
        (True, True),
        (True, True),
        (False, True),
        (False, True),
        (False, True),
        (True, True),
    ]
    evidence = _traced_sequence(
        [(True, True)] * len(admissibility_pairs), admissibility_pairs
    )
    final_state, traced = scan_forward_branch_selection(
        evidence,
        initial_traced_selection_state(),
        jnp.asarray(int(policy.cold_start_class), dtype=jnp.int8),
        jnp.asarray(policy.persistence_threshold, dtype=jnp.int32),
    )

    history = SelectionHistory()
    host_classes = []
    host_reasons = []
    for limited, diverted in admissibility_pairs:
        receipt = select_forward_branch(
            _portfolio(True, True),
            history,
            policy,
            BranchAdmissibility(limited=limited, diverted=diverted),
        )
        host_classes.append(int(receipt.selected_class))
        host_reasons.append(list(SelectionReason).index(receipt.reason))
        history = receipt.next_history

    np.testing.assert_array_equal(traced.selected_class, host_classes)
    np.testing.assert_array_equal(traced.reason_code, host_reasons)
    assert int(final_state.selected_class) == int(history.selected_class)
    assert int(final_state.pending_count) == history.pending_count
    assert int(final_state.sequence_index) == history.sequence_index


def test_traced_scan_jits_and_vmaps_array_selector_state():
    evidence = _traced_sequence(
        [(True, True), (True, False), (False, True), (False, False)],
        [(True, True)] * 4,
        probabilities=[0.2, 0.4, 0.6, 0.8],
        margins=[-0.02, -0.01, 0.01, 0.02],
    )
    state = initial_traced_selection_state()
    cold_start = jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8)
    threshold = jnp.asarray(2, dtype=jnp.int32)
    traced_jaxpr = jax.make_jaxpr(scan_forward_branch_selection)(
        evidence, state, cold_start, threshold
    )
    assert any(equation.primitive.name == "scan" for equation in traced_jaxpr.eqns)

    batch_size = 3
    batched_evidence = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), evidence
    )
    batched_state = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), state
    )
    compiled = jax.jit(
        jax.vmap(scan_forward_branch_selection, in_axes=(0, 0, None, None))
    )
    final_state, traced = compiled(
        batched_evidence, batched_state, cold_start, threshold
    )
    jax.block_until_ready((final_state, traced))

    assert traced.selected_class.shape == (batch_size, 4)
    assert final_state.availability.shape == (batch_size, 2)
    assert final_state.admissibility.shape == (batch_size, 2)
    np.testing.assert_array_equal(final_state.degrade_path_firings, [2, 2, 2])
    np.testing.assert_array_equal(final_state.two_qualified_selections, [1, 1, 1])


def test_smooth_weight_and_exact_margin_retain_separate_roles():
    evidence = _traced_sequence(
        [(True, True), (True, True)],
        [(True, True), (True, True)],
        probabilities=[0.25, 0.75],
        margins=[float("inf"), -1.0],
    )
    state = initial_traced_selection_state()
    cold_start = jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8)
    threshold = jnp.asarray(2, dtype=jnp.int32)
    _final_state, traced = jax.jit(scan_forward_branch_selection)(
        evidence, state, cold_start, threshold
    )

    np.testing.assert_allclose(traced.diverted_weight, [0.25, 0.75])
    np.testing.assert_array_equal(traced.comparator_class, [-1, 0])
    gradient = jax.grad(
        lambda probabilities: jnp.sum(
            scan_forward_branch_selection(
                evidence._replace(p_diverted=probabilities),
                state,
                cold_start,
                threshold,
            )[1].flux
        )
    )(evidence.p_diverted)
    assert np.all(np.asarray(gradient) != 0.0)


def test_host_receipt_counts_degrades_against_two_qualified_selections(policy):
    history = SelectionHistory()
    for availability in ((True, True), (True, False), (False, True)):
        receipt = select_forward_branch(_portfolio(*availability), history, policy)
        history = receipt.next_history

    data = receipt.as_dict()
    assert data["selection_cohort"] == {
        "degrade_path_firings": 2,
        "two_qualified_selections": 1,
    }
