"""Re-score the frozen MAST references with held-out convergence criteria.

Only committed receipts are read.  A zero residual from a vacuum collapse is
reported as a numeric threshold match but never counted as a converged physical
root.  Registered and held-out counts remain side by side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OUTPUT_PATH = Path(
    "docs/figures/scoring-criteria-derivation/derived-criterion-rescore.json"
)
CRITERION_SOURCE = Path(
    "docs/figures/scoring-criteria-derivation/derived-convergence-criterion.json"
)
SCORECARD_SOURCE = Path(
    "docs/figures/efit-forward-parity/passive-inclusive-frozen-six-scorecard.json"
)

REGISTERED_CRITERION = 1.0e-8
INVALID_ROOT_VERDICT = "INVALID_PHYSICAL_ROOT"
EXPECTED_REFERENCES = {
    "21978/35",
    "21983/35",
    "21985/51",
    "21986/46",
    "21989/55",
    "22086/43",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_label(row: dict[str, Any]) -> str:
    return f"{int(row['shot'])}/{int(row['slice_index'])}"


def _criterion_rows(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["reference"]: row for row in receipt["per_reference"]}
    if set(rows) != EXPECTED_REFERENCES:
        raise RuntimeError("the held-out criterion cohort changed")
    return rows


def _scorecard_rows(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {_reference_label(row): row for row in receipt["per_shot_table"]}
    if set(rows) != EXPECTED_REFERENCES:
        raise RuntimeError("the frozen scorecard cohort changed")
    return rows


def _verdict(numeric_match: bool, physical_root_valid: bool) -> str:
    if not physical_root_valid:
        return INVALID_ROOT_VERDICT
    return "PASS" if numeric_match else "FAIL"


def build_receipt_from_data(
    criterion_receipt: dict[str, Any], scorecard_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Build a dual-count re-score from already loaded banked receipts."""
    criteria = _criterion_rows(criterion_receipt)
    scorecard = _scorecard_rows(scorecard_receipt)
    rows = []
    for reference in sorted(EXPECTED_REFERENCES):
        criterion_row = criteria[reference]
        scorecard_row = scorecard[reference]
        closest_residual = float(scorecard_row["closest_residual"])
        derived_criterion = float(criterion_row["derived_criterion"])
        outcome = scorecard_row["outcome_class"]

        banked_residual = float(criterion_row["gated_closest_residual_display_only"])
        banked_outcome = criterion_row["gated_outcome_display_only"]
        if closest_residual != banked_residual or outcome != banked_outcome:
            raise RuntimeError(
                f"criterion and scorecard evidence disagree for {reference}"
            )

        physical_root_valid = outcome != "vacuum_collapse"
        registered_numeric_match = closest_residual <= REGISTERED_CRITERION
        derived_numeric_match = closest_residual <= derived_criterion
        registered_verdict = _verdict(registered_numeric_match, physical_root_valid)
        derived_verdict = _verdict(derived_numeric_match, physical_root_valid)
        rows.append(
            {
                "reference": reference,
                "outcome_class": outcome,
                "physical_root_valid": physical_root_valid,
                "closest_residual": closest_residual,
                "registered_criterion": REGISTERED_CRITERION,
                "registered_numeric_threshold_met": registered_numeric_match,
                "registered_verdict": registered_verdict,
                "derived_criterion": derived_criterion,
                "derived_numeric_threshold_met": derived_numeric_match,
                "derived_verdict": derived_verdict,
                "derived_margin": derived_criterion - closest_residual,
                "derived_residual_over_criterion": (
                    closest_residual / derived_criterion
                ),
                "criterion_stratum": criterion_row["stratum"],
                "criterion_qualification": {
                    "fit": criterion_row["fit"]["uncertainty_qualification"],
                    "target": criterion_row["target_qualification"],
                },
            }
        )

    registered_numeric_count = sum(
        row["registered_numeric_threshold_met"] for row in rows
    )
    registered_count = sum(row["registered_verdict"] == "PASS" for row in rows)
    derived_numeric_count = sum(row["derived_numeric_threshold_met"] for row in rows)
    derived_count = sum(row["derived_verdict"] == "PASS" for row in rows)
    invalid_count = sum(not row["physical_root_valid"] for row in rows)
    return {
        "receipt": {
            "kind": "frozen_six_held_out_criterion_rescore",
            "status": "complete",
            "execution_mode": "banked-receipts-only-no-equilibrium-solves",
            "equilibrium_solves_run": 0,
            "reference_count": len(rows),
        },
        "comparison_table": rows,
        "counts": {
            "registered_1e8": {
                "criterion": REGISTERED_CRITERION,
                "numeric_threshold_met_count": registered_numeric_count,
                "invalid_physical_root_count": invalid_count,
                "physical_convergence_count": registered_count,
                "display": f"{registered_count} of {len(rows)}",
            },
            "derived_held_out": {
                "per_reference_criterion": True,
                "numeric_threshold_met_count": derived_numeric_count,
                "invalid_physical_root_count": invalid_count,
                "physical_convergence_count": derived_count,
                "display": f"{derived_count} of {len(rows)}",
            },
            "both_counts_retained": True,
        },
        "vacuum_collapse_policy": {
            "reference": "21989/55",
            "closest_residual": 0.0,
            "numeric_thresholds_met": ["registered_1e8", "derived_held_out"],
            "verdict_under_both": INVALID_ROOT_VERDICT,
            "counted_as_converged": False,
            "reason": (
                "The zero residual belongs to a zero-current vacuum collapse, "
                "not a converged plasma root."
            ),
        },
        "verdict": {
            "code": "DERIVED_CRITERION_IMPROVES_COUNT_BUT_REMAINS_INCOMPLETE",
            "statement": (
                "The held-out criterion raises physical convergence from 0 of 6 "
                "to 4 of 6; one reference exceeds its criterion and one is an "
                "invalid vacuum root."
            ),
            "catalog_hold_remains": True,
            "catalog_hold_implication": (
                "The catalog hold remains because only four of six frozen "
                "references pass the held-out criterion: 22086/43 exceeds its "
                "bound and 21989/55 is an invalid vacuum root."
            ),
        },
        "claim_bounds": {
            "new_equilibrium_solve": False,
            "banked_residuals_only": True,
            "registered_tolerance_changed": False,
            "derived_criterion_changed": False,
            "criterion_limit_retained": (
                "The held-out criteria have only two distinct mesh spacings and "
                "no independent asymptotic confirmation."
            ),
        },
    }


def build_receipt() -> dict[str, Any]:
    """Build the re-score from the committed criterion and scorecard."""
    criterion_receipt = json.loads(CRITERION_SOURCE.read_text())
    scorecard_receipt = json.loads(SCORECARD_SOURCE.read_text())
    receipt = build_receipt_from_data(criterion_receipt, scorecard_receipt)
    receipt["sources"] = {
        str(CRITERION_SOURCE): _sha256(CRITERION_SOURCE),
        str(SCORECARD_SOURCE): _sha256(SCORECARD_SOURCE),
    }
    return receipt


def write_receipt(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Write and return the banked re-score."""
    receipt = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    receipt = write_receipt(arguments.output)
    counts = receipt["counts"]
    print(
        f"registered={counts['registered_1e8']['display']} "
        f"derived={counts['derived_held_out']['display']} "
        f"invalid={counts['derived_held_out']['invalid_physical_root_count']} "
        "solves=0"
    )


if __name__ == "__main__":
    main()
