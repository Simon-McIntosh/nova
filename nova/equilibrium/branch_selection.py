"""Pure post-solve selection of a topology branch from a forward portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium.forward import ForwardBranchReceipt, ForwardPortfolio
from nova.equilibrium.topology import TopologyClass

__all__ = [
    "AdmissibilityCriterion",
    "BranchAdmissibility",
    "BranchAvailability",
    "ColdStartRule",
    "DisappearanceCriterion",
    "SelectionHistory",
    "SelectionPolicy",
    "SelectionReason",
    "SelectionReceipt",
    "TracedSelectionInput",
    "TracedSelectionState",
    "TracedSelectionStep",
    "forward_branch_selection_input",
    "initial_traced_selection_state",
    "scan_forward_branch_selection",
    "select_forward_branch",
    "traced_select_forward_branch",
]


_NO_CLASS = -1


class ColdStartRule(StrEnum):
    """Rule used when no branch has previously been selected."""

    DECLARED_CLASS = "declared_class"


class DisappearanceCriterion(StrEnum):
    """Transition rule when the selected branch stops being available."""

    IMMEDIATE_ALTERNATE = "immediate_alternate"


class AdmissibilityCriterion(StrEnum):
    """Transition rule when a converged branch becomes inadmissible."""

    PERSISTENT_ALTERNATE = "persistent_alternate"


class SelectionReason(StrEnum):
    """Mutually exclusive reason the selector returned its verdict."""

    COLD_START = "cold_start"
    SOLE_VALID = "sole_valid"
    HISTORY_CONTINUITY = "history_continuity"
    BRANCH_DISAPPEARED = "branch_disappeared"
    ADMISSIBILITY_PENDING = "admissibility_pending"
    ADMISSIBILITY_PERSISTED = "admissibility_persisted"
    NO_VALID_BRANCH = "no_valid_branch"
    NO_ADMISSIBLE_ALTERNATIVE = "no_admissible_alternative"


_REASON_VALUES = tuple(SelectionReason)
_REASON_CODES = {reason: index for index, reason in enumerate(_REASON_VALUES)}


class TracedSelectionInput(NamedTuple):
    """Array-only evidence for one post-solve branch selection.

    The branch axis is ordered limited, diverted. ``p_diverted`` is the smooth
    blend weight. ``class_margin`` remains the exact signed comparator for
    reporting and gates; it is never substituted for the smooth weight.
    """

    flux: jax.Array
    availability: jax.Array
    admissibility: jax.Array
    residuals: jax.Array
    p_diverted: jax.Array
    class_margin: jax.Array


class TracedSelectionState(NamedTuple):
    """Device-visible selector history and current qualification masks."""

    selected_class: jax.Array
    pending_class: jax.Array
    pending_count: jax.Array
    sequence_index: jax.Array
    availability: jax.Array
    admissibility: jax.Array
    degrade_path_firings: jax.Array
    two_qualified_selections: jax.Array


class TracedSelectionStep(NamedTuple):
    """Array-only selection result emitted by one traced transition."""

    flux: jax.Array
    diverted_weight: jax.Array
    class_margin: jax.Array
    comparator_class: jax.Array
    selected_class: jax.Array
    previous_class: jax.Array
    switched: jax.Array
    reason_code: jax.Array
    qualified: jax.Array
    degraded: jax.Array
    both_qualified: jax.Array
    availability: jax.Array
    admissibility: jax.Array
    residuals: jax.Array


def forward_branch_selection_input(
    branches: ForwardBranchReceipt,
    p_diverted: jax.Array,
    class_margin: jax.Array,
    admissibility: jax.Array | None = None,
) -> TracedSelectionInput:
    """Build array evidence directly from a paired forward-branch receipt."""

    availability = jnp.asarray(branches.converged, dtype=bool) & jnp.asarray(
        branches.topology_consistent, dtype=bool
    )
    if admissibility is None:
        admissibility = jnp.ones_like(availability, dtype=bool)
    return TracedSelectionInput(
        flux=jnp.asarray(branches.equilibrium.flux),
        availability=availability,
        admissibility=jnp.asarray(admissibility, dtype=bool),
        residuals=jnp.asarray(branches.residual),
        p_diverted=jnp.asarray(p_diverted),
        class_margin=jnp.asarray(class_margin),
    )


def initial_traced_selection_state() -> TracedSelectionState:
    """Return an array-only state with no previously selected branch."""

    return TracedSelectionState(
        selected_class=jnp.asarray(_NO_CLASS, dtype=jnp.int8),
        pending_class=jnp.asarray(_NO_CLASS, dtype=jnp.int8),
        pending_count=jnp.asarray(0, dtype=jnp.int32),
        sequence_index=jnp.asarray(0, dtype=jnp.int32),
        availability=jnp.zeros((2,), dtype=bool),
        admissibility=jnp.ones((2,), dtype=bool),
        degrade_path_firings=jnp.asarray(0, dtype=jnp.int32),
        two_qualified_selections=jnp.asarray(0, dtype=jnp.int32),
    )


def _class_value(value: TopologyClass | None) -> str | None:
    """Return the stable provenance spelling of an optional topology class."""

    return None if value is None else value.name.lower()


@dataclass(frozen=True)
class BranchAvailability:
    """Post-solve availability derived from branch receipt qualification."""

    limited: bool
    diverted: bool

    def for_class(self, topology_class: TopologyClass) -> bool:
        """Return availability for one topology class."""

        return (
            self.diverted if topology_class is TopologyClass.DIVERTED else self.limited
        )

    def as_dict(self) -> dict[str, bool]:
        """Return stable limited/diverted provenance data."""

        return {"limited": self.limited, "diverted": self.diverted}


@dataclass(frozen=True)
class BranchAdmissibility:
    """Declared physical admissibility of each converged candidate."""

    limited: bool = True
    diverted: bool = True

    def for_class(self, topology_class: TopologyClass) -> bool:
        """Return admissibility for one topology class."""

        return (
            self.diverted if topology_class is TopologyClass.DIVERTED else self.limited
        )

    def as_dict(self) -> dict[str, bool]:
        """Return stable limited/diverted provenance data."""

        return {"limited": self.limited, "diverted": self.diverted}


@dataclass(frozen=True)
class SelectionPolicy:
    """Declared cold-start and transition criteria for branch selection."""

    cold_start_class: TopologyClass
    persistence_threshold: int
    cold_start_rule: ColdStartRule = ColdStartRule.DECLARED_CLASS
    disappearance_criterion: DisappearanceCriterion = (
        DisappearanceCriterion.IMMEDIATE_ALTERNATE
    )
    admissibility_criterion: AdmissibilityCriterion = (
        AdmissibilityCriterion.PERSISTENT_ALTERNATE
    )

    def __post_init__(self) -> None:
        """Normalize enum values and reject a vacuous persistence rule."""

        object.__setattr__(
            self, "cold_start_class", TopologyClass(self.cold_start_class)
        )
        object.__setattr__(self, "cold_start_rule", ColdStartRule(self.cold_start_rule))
        object.__setattr__(
            self,
            "disappearance_criterion",
            DisappearanceCriterion(self.disappearance_criterion),
        )
        object.__setattr__(
            self,
            "admissibility_criterion",
            AdmissibilityCriterion(self.admissibility_criterion),
        )
        if self.persistence_threshold < 1:
            raise ValueError("persistence_threshold must be at least one")

    def as_dict(self) -> dict[str, str | int]:
        """Return every selection criterion as provenance data."""

        return {
            "cold_start_rule": self.cold_start_rule.value,
            "cold_start_class": _class_value(self.cold_start_class),
            "persistence_threshold": self.persistence_threshold,
            "disappearance_criterion": self.disappearance_criterion.value,
            "admissibility_criterion": self.admissibility_criterion.value,
        }


@dataclass(frozen=True)
class SelectionHistory:
    """Immutable history threaded between independently completed portfolios."""

    selected_class: TopologyClass | None = None
    pending_class: TopologyClass | None = None
    pending_count: int = 0
    sequence_index: int = 0
    degrade_path_firings: int = 0
    two_qualified_selections: int = 0

    def __post_init__(self) -> None:
        """Normalize class values and validate pending-transition state."""

        if self.selected_class is not None:
            object.__setattr__(
                self, "selected_class", TopologyClass(self.selected_class)
            )
        if self.pending_class is not None:
            object.__setattr__(self, "pending_class", TopologyClass(self.pending_class))
        if self.pending_count < 0:
            raise ValueError("pending_count cannot be negative")
        if self.sequence_index < 0:
            raise ValueError("sequence_index cannot be negative")
        if self.degrade_path_firings < 0:
            raise ValueError("degrade_path_firings cannot be negative")
        if self.two_qualified_selections < 0:
            raise ValueError("two_qualified_selections cannot be negative")
        if (self.pending_class is None) != (self.pending_count == 0):
            raise ValueError("pending_class and pending_count must be set together")
        if self.pending_class is not None and self.pending_class is self.selected_class:
            raise ValueError("a pending transition must target the alternate class")


@dataclass(frozen=True)
class SelectionReceipt:
    """Selected class, fired rule, evidence, policy, and next history state."""

    selected_class: TopologyClass | None
    previous_class: TopologyClass | None
    switched: bool
    reason: SelectionReason
    availability: BranchAvailability
    admissibility: BranchAdmissibility
    residuals: tuple[float, float]
    policy: SelectionPolicy
    next_history: SelectionHistory

    def as_dict(self) -> dict[str, Any]:
        """Return a strict provenance representation of this selection."""

        return {
            "sequence_index": self.next_history.sequence_index - 1,
            "selected_class": _class_value(self.selected_class),
            "previous_class": _class_value(self.previous_class),
            "switched": self.switched,
            "reason": self.reason.value,
            "availability": self.availability.as_dict(),
            "admissibility": self.admissibility.as_dict(),
            "residuals": {
                "limited": self.residuals[int(TopologyClass.LIMITED)],
                "diverted": self.residuals[int(TopologyClass.DIVERTED)],
            },
            "policy": self.policy.as_dict(),
            "persistence": {
                "pending_class": _class_value(self.next_history.pending_class),
                "pending_count": self.next_history.pending_count,
            },
            "selection_cohort": {
                "degrade_path_firings": self.next_history.degrade_path_firings,
                "two_qualified_selections": (
                    self.next_history.two_qualified_selections
                ),
            },
        }


def _history_to_traced(
    history: SelectionHistory,
    availability: jax.Array,
    admissibility: jax.Array,
) -> TracedSelectionState:
    """Convert host history to the array boundary consumed by the core."""

    selected = (
        _NO_CLASS if history.selected_class is None else int(history.selected_class)
    )
    pending = _NO_CLASS if history.pending_class is None else int(history.pending_class)
    return TracedSelectionState(
        selected_class=jnp.asarray(selected, dtype=jnp.int8),
        pending_class=jnp.asarray(pending, dtype=jnp.int8),
        pending_count=jnp.asarray(history.pending_count, dtype=jnp.int32),
        sequence_index=jnp.asarray(history.sequence_index, dtype=jnp.int32),
        availability=jnp.asarray(availability, dtype=bool),
        admissibility=jnp.asarray(admissibility, dtype=bool),
        degrade_path_firings=jnp.asarray(history.degrade_path_firings, dtype=jnp.int32),
        two_qualified_selections=jnp.asarray(
            history.two_qualified_selections, dtype=jnp.int32
        ),
    )


def _advance_traced_selection(
    state: TracedSelectionState,
    evidence: TracedSelectionInput,
    cold_start_class: jax.Array,
    persistence_threshold: jax.Array,
) -> tuple[TracedSelectionState, TracedSelectionStep]:
    """Advance one selection using only integer, boolean, and floating arrays."""

    limited = jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8)
    diverted = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    no_class = jnp.asarray(_NO_CLASS, dtype=jnp.int8)
    availability = jnp.asarray(evidence.availability, dtype=bool)
    admissibility = jnp.asarray(evidence.admissibility, dtype=bool)
    selectable = availability & admissibility
    limited_selectable, diverted_selectable = selectable[0], selectable[1]
    both_qualified = limited_selectable & diverted_selectable
    exactly_one_qualified = limited_selectable ^ diverted_selectable
    any_qualified = limited_selectable | diverted_selectable

    previous = state.selected_class
    has_previous = previous != no_class
    alternate = jnp.where(previous == limited, diverted, limited)
    previous_available = jnp.where(
        previous == diverted, availability[1], availability[0]
    )
    previous_admissible = jnp.where(
        previous == diverted, admissibility[1], admissibility[0]
    )
    alternate_selectable = jnp.where(
        alternate == diverted, diverted_selectable, limited_selectable
    )
    sole_class = jnp.where(limited_selectable, limited, diverted)

    cold_selected = jnp.where(
        ~any_qualified,
        no_class,
        jnp.where(exactly_one_qualified, sole_class, cold_start_class),
    )
    cold_reason = jnp.where(
        ~any_qualified,
        _REASON_CODES[SelectionReason.NO_VALID_BRANCH],
        jnp.where(
            exactly_one_qualified,
            _REASON_CODES[SelectionReason.SOLE_VALID],
            _REASON_CODES[SelectionReason.COLD_START],
        ),
    )

    pending_count = jnp.where(
        state.pending_class == alternate, state.pending_count + 1, 1
    )
    persistence_reached = pending_count >= persistence_threshold
    history_selected = jnp.where(
        ~previous_available,
        jnp.where(alternate_selectable, alternate, no_class),
        jnp.where(
            previous_admissible,
            previous,
            jnp.where(
                ~alternate_selectable,
                previous,
                jnp.where(persistence_reached, alternate, previous),
            ),
        ),
    )
    history_reason = jnp.where(
        ~previous_available,
        jnp.where(
            alternate_selectable,
            _REASON_CODES[SelectionReason.BRANCH_DISAPPEARED],
            _REASON_CODES[SelectionReason.NO_VALID_BRANCH],
        ),
        jnp.where(
            previous_admissible,
            _REASON_CODES[SelectionReason.HISTORY_CONTINUITY],
            jnp.where(
                ~alternate_selectable,
                _REASON_CODES[SelectionReason.NO_ADMISSIBLE_ALTERNATIVE],
                jnp.where(
                    persistence_reached,
                    _REASON_CODES[SelectionReason.ADMISSIBILITY_PERSISTED],
                    _REASON_CODES[SelectionReason.ADMISSIBILITY_PENDING],
                ),
            ),
        ),
    )
    selected = jnp.where(has_previous, history_selected, cold_selected).astype(jnp.int8)
    reason_code = jnp.where(has_previous, history_reason, cold_reason).astype(jnp.int8)
    pending_active = (
        has_previous
        & previous_available
        & ~previous_admissible
        & alternate_selectable
        & ~persistence_reached
    )
    next_pending_class = jnp.where(pending_active, alternate, no_class).astype(jnp.int8)
    next_pending_count = jnp.where(pending_active, pending_count, 0).astype(jnp.int32)
    anchored_selected = jnp.where(selected == no_class, previous, selected).astype(
        jnp.int8
    )
    switched = has_previous & (selected != no_class) & (selected != previous)
    degraded = exactly_one_qualified

    smooth_weight = jnp.clip(jnp.asarray(evidence.p_diverted), 0.0, 1.0)
    diverted_weight = jnp.where(
        both_qualified,
        smooth_weight,
        jnp.where(
            limited_selectable,
            0.0,
            jnp.where(diverted_selectable, 1.0, jnp.nan),
        ),
    )
    limited_flux = evidence.flux[0]
    diverted_flux = evidence.flux[1]
    selected_flux = limited_flux + diverted_weight * (diverted_flux - limited_flux)
    class_margin = jnp.asarray(evidence.class_margin)
    comparator_class = jnp.where(
        jnp.isfinite(class_margin), class_margin >= 0, no_class
    ).astype(jnp.int8)

    next_state = TracedSelectionState(
        selected_class=anchored_selected,
        pending_class=next_pending_class,
        pending_count=next_pending_count,
        sequence_index=state.sequence_index + 1,
        availability=availability,
        admissibility=admissibility,
        degrade_path_firings=state.degrade_path_firings + degraded.astype(jnp.int32),
        two_qualified_selections=(
            state.two_qualified_selections + both_qualified.astype(jnp.int32)
        ),
    )
    step = TracedSelectionStep(
        flux=selected_flux,
        diverted_weight=diverted_weight,
        class_margin=class_margin,
        comparator_class=comparator_class,
        selected_class=selected,
        previous_class=previous,
        switched=switched,
        reason_code=reason_code,
        qualified=any_qualified,
        degraded=degraded,
        both_qualified=both_qualified,
        availability=availability,
        admissibility=admissibility,
        residuals=jnp.asarray(evidence.residuals),
    )
    return next_state, step


def traced_select_forward_branch(
    evidence: TracedSelectionInput,
    state: TracedSelectionState,
    cold_start_class: jax.Array,
    persistence_threshold: jax.Array,
) -> tuple[TracedSelectionState, TracedSelectionStep]:
    """Select one branch through the production array-only core."""

    return _advance_traced_selection(
        state, evidence, cold_start_class, persistence_threshold
    )


def scan_forward_branch_selection(
    evidence: TracedSelectionInput,
    initial_state: TracedSelectionState,
    cold_start_class: jax.Array,
    persistence_threshold: jax.Array,
) -> tuple[TracedSelectionState, TracedSelectionStep]:
    """Advance a time-ordered selector state with :func:`jax.lax.scan`."""

    return jax.lax.scan(
        lambda state, item: _advance_traced_selection(
            state, item, cold_start_class, persistence_threshold
        ),
        initial_state,
        evidence,
    )


def _pair(values, name: str) -> tuple[Any, Any]:
    """Return a fixed limited/diverted pair from one portfolio field."""

    pair = tuple(values)
    if len(pair) != 2:
        raise ValueError(f"portfolio {name} must have exactly two branch entries")
    return pair[0], pair[1]


def select_forward_branch(
    portfolio: ForwardPortfolio,
    history: SelectionHistory,
    policy: SelectionPolicy,
    admissibility: BranchAdmissibility | None = None,
) -> SelectionReceipt:
    """Select one completed branch without mutating history or solve state.

    Branch availability is derived only from the portfolio's converged and
    topology-consistent receipt flags. Physical admissibility is a declared
    post-solve input. A disappeared selected branch switches immediately when
    the alternate is selectable; an admissibility-only transition switches
    after the declared consecutive-slice threshold. When both branches remain
    selectable, history wins. A cold start uses the policy's declared class.
    """

    if admissibility is None:
        admissibility = BranchAdmissibility()
    converged = _pair(portfolio.branches.converged, "converged")
    consistent = _pair(portfolio.branches.topology_consistent, "topology_consistent")
    residual_pair = _pair(portfolio.branches.residual, "residual")
    available_pair = tuple(
        bool(branch_converged) and bool(branch_consistent)
        for branch_converged, branch_consistent in zip(
            converged, consistent, strict=True
        )
    )
    availability = BranchAvailability(*available_pair)
    residuals = float(residual_pair[0]), float(residual_pair[1])
    evidence = TracedSelectionInput(
        flux=jnp.zeros((2,), dtype=jnp.asarray(residuals).dtype),
        availability=jnp.asarray(available_pair, dtype=bool),
        admissibility=jnp.asarray(
            (admissibility.limited, admissibility.diverted), dtype=bool
        ),
        residuals=jnp.asarray(residuals),
        p_diverted=jnp.asarray(0.5),
        class_margin=jnp.asarray(jnp.nan),
    )
    state, step = traced_select_forward_branch(
        evidence,
        _history_to_traced(history, evidence.availability, evidence.admissibility),
        jnp.asarray(int(policy.cold_start_class), dtype=jnp.int8),
        jnp.asarray(policy.persistence_threshold, dtype=jnp.int32),
    )
    selected_code = int(step.selected_class)
    previous_code = int(step.previous_class)
    pending_code = int(state.pending_class)
    selected = None if selected_code == _NO_CLASS else TopologyClass(selected_code)
    previous = None if previous_code == _NO_CLASS else TopologyClass(previous_code)
    pending = None if pending_code == _NO_CLASS else TopologyClass(pending_code)
    state_selected_code = int(state.selected_class)
    next_history = SelectionHistory(
        selected_class=(
            None
            if state_selected_code == _NO_CLASS
            else TopologyClass(state_selected_code)
        ),
        pending_class=pending,
        pending_count=int(state.pending_count),
        sequence_index=int(state.sequence_index),
        degrade_path_firings=int(state.degrade_path_firings),
        two_qualified_selections=int(state.two_qualified_selections),
    )
    return SelectionReceipt(
        selected_class=selected,
        previous_class=previous,
        switched=bool(step.switched),
        reason=_REASON_VALUES[int(step.reason_code)],
        availability=availability,
        admissibility=admissibility,
        residuals=residuals,
        policy=policy,
        next_history=next_history,
    )
