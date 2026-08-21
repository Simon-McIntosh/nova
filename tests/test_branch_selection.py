"""Case matrix for pure post-solve topology branch selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nova.equilibrium import (
    BranchAdmissibility,
    SelectionHistory,
    SelectionPolicy,
    SelectionReason,
    select_forward_branch,
)
from nova.equilibrium.forward import ForwardPortfolio
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
