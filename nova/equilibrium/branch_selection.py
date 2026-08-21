"""Pure post-solve selection of a topology branch from a forward portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from nova.equilibrium.forward import ForwardPortfolio
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
    "select_forward_branch",
]


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
        }


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
    previous = history.selected_class

    def selectable(topology_class: TopologyClass) -> bool:
        return availability.for_class(topology_class) and admissibility.for_class(
            topology_class
        )

    def finish(
        selected: TopologyClass | None,
        reason: SelectionReason,
        *,
        pending_class: TopologyClass | None = None,
        pending_count: int = 0,
    ) -> SelectionReceipt:
        anchor = previous if selected is None else selected
        next_history = SelectionHistory(
            selected_class=anchor,
            pending_class=pending_class,
            pending_count=pending_count,
            sequence_index=history.sequence_index + 1,
        )
        return SelectionReceipt(
            selected_class=selected,
            previous_class=previous,
            switched=(
                previous is not None
                and selected is not None
                and selected is not previous
            ),
            reason=reason,
            availability=availability,
            admissibility=admissibility,
            residuals=residuals,
            policy=policy,
            next_history=next_history,
        )

    limited = TopologyClass.LIMITED
    diverted = TopologyClass.DIVERTED
    if previous is None:
        limited_selectable = selectable(limited)
        diverted_selectable = selectable(diverted)
        if not limited_selectable and not diverted_selectable:
            return finish(None, SelectionReason.NO_VALID_BRANCH)
        if limited_selectable != diverted_selectable:
            selected = limited if limited_selectable else diverted
            return finish(selected, SelectionReason.SOLE_VALID)
        return finish(policy.cold_start_class, SelectionReason.COLD_START)

    alternate = diverted if previous is limited else limited
    if not availability.for_class(previous):
        if selectable(alternate):
            return finish(alternate, SelectionReason.BRANCH_DISAPPEARED)
        return finish(None, SelectionReason.NO_VALID_BRANCH)

    if admissibility.for_class(previous):
        return finish(previous, SelectionReason.HISTORY_CONTINUITY)

    if not selectable(alternate):
        return finish(previous, SelectionReason.NO_ADMISSIBLE_ALTERNATIVE)

    pending_count = (
        history.pending_count + 1 if history.pending_class is alternate else 1
    )
    if pending_count >= policy.persistence_threshold:
        return finish(alternate, SelectionReason.ADMISSIBILITY_PERSISTED)
    return finish(
        previous,
        SelectionReason.ADMISSIBILITY_PENDING,
        pending_class=alternate,
        pending_count=pending_count,
    )
