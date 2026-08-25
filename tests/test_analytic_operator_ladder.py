import copy
import json
from pathlib import Path

import pytest

from benchmarks import analytic_operator_ladder as ladder


RECEIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/discrete-operator-analytic-error/operator-refinement-ladder.json"
)


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_clean_posed_term_audit_reports_unresolved_solver_selection() -> None:
    receipt = _receipt()

    verdict = ladder._verdict(
        receipt["refinement_ladder"],
        receipt["convergence_order_fit"],
        receipt["headline_metrics"]["worst_full_state_one_application_relative_sup"],
    )

    assert verdict["posed_terms_match"] is True
    assert verdict["cause"] == "solver_or_root_selection_unresolved"
    assert (
        verdict["reason"] == "analytic_fixed_point_admitted_and_all_posed_terms_match"
    )


def test_clean_posed_term_audit_rejects_posing_cause() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["verdict"] = ladder._verdict(
        receipt["refinement_ladder"],
        receipt["convergence_order_fit"],
        receipt["headline_metrics"]["worst_full_state_one_application_relative_sup"],
    )
    receipt["verdict"]["cause"] = "posing"

    with pytest.raises(RuntimeError, match="clean posed-term audit"):
        ladder._validate(receipt)
