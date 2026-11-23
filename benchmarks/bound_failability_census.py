"""Census the observed headroom of terminal acceptance criteria.

Numeric envelope slack is the registered bound divided by the largest banked
observation of the quantity that component scores.  A dual envelope's effective
slack is its smaller component slack because either component can reject a
candidate.  Exact-equality criteria are recorded but have no numeric ratio:
their zero bound and zero observed mismatch form an indeterminate ratio, while
any nonzero mismatch still demonstrates that they are failable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parents[1]
REGISTRATION_SOURCE = (
    HERE / "docs/figures/forward-operator-refinement/criterion-family.json"
)
ACCEPTANCE_SOURCE = (
    HERE / "docs/figures/derived-observable-parity/integrated-acceptance.json"
)
CORRECTION_SOURCE = (
    HERE / "docs/figures/roundoff-scale-acceptance-bounds/corrected-criteria.json"
)
SCALING_SOURCE = (
    HERE / "docs/figures/roundoff-scale-acceptance-bounds/divergence-floor-scaling.json"
)
DEFAULT_OUTPUT = (
    HERE / "docs/figures/roundoff-scale-acceptance-bounds/bound-failability-census.json"
)

FLAG_THRESHOLD = 20.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = source["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]["bounds"]
    if len(rows) != 69 or len({row["observable"] for row in rows}) != 69:
        raise RuntimeError("the acceptance bank is not 69 unique registrations")
    return rows


def _cohort_observations(acceptance: dict[str, Any]) -> dict[str, dict[str, float]]:
    observations: dict[str, dict[str, float]] = {}
    for batch in acceptance["batch_results"]:
        for row in batch["per_observable"]:
            current = observations.setdefault(
                row["observable"], {"absolute": 0.0, "relative": 0.0}
            )
            current["absolute"] = max(
                current["absolute"], float(row["maximum_absolute_difference"])
            )
            current["relative"] = max(
                current["relative"], float(row["maximum_relative_difference"])
            )
    return observations


def _corrected_registration(corrections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["observable"]: {
            "criterion_kind": row["criterion_kind"],
            "absolute_bound": float(row["absolute_bound"]),
        }
        for row in corrections["corrected_criteria"]
    }


def _divergence_b_magnitude(corrections: dict[str, Any]) -> float:
    row = next(
        row
        for row in corrections["corrected_criteria"]
        if row["observable"] == "conservation.divergence_b"
    )
    return max(float(value) for value in row["derivation"]["banked_exemplar"]["values"])


def _component(
    *, name: str, bound: float, observed: float, evidence: str
) -> dict[str, Any]:
    if observed <= 0.0:
        raise RuntimeError(f"{name} has no positive banked observation")
    return {
        "component": name,
        "bound": bound,
        "largest_observed_scored_value": observed,
        "slack": bound / observed,
        "observation_evidence": evidence,
    }


def _exact_row(bound: dict[str, Any], cohort: dict[str, float]) -> dict[str, Any]:
    maximum = max(cohort.values())
    if maximum != 0.0:
        raise RuntimeError(
            f"exact-equality registration {bound['observable']} has a nonzero mismatch"
        )
    return {
        "observable": bound["observable"],
        "criterion_kind": "exact_equality",
        "assessment": "numeric_slack_not_assessable",
        "effective_slack": None,
        "components": [
            {
                "component": "exact_mismatch",
                "bound": 0.0,
                "largest_observed_scored_value": 0.0,
                "slack": None,
                "reason": "zero divided by zero has no numeric slack ratio",
            }
        ],
        "failability": {
            "demonstrated": True,
            "plausible_perturbation": (
                "change any one emitted value, label, predicate, or policy code "
                "by one representable unit"
            ),
            "reason": "any nonzero mismatch fails an exact-equality criterion",
        },
        "candidate_category_error": False,
    }


def _numeric_row(
    bound: dict[str, Any],
    cohort: dict[str, float],
    divergence_b_magnitude: float,
    divergence_j_magnitude: float,
    divergence_j_failability: dict[str, Any],
) -> dict[str, Any]:
    name = bound["observable"]
    components = []
    if name == "conservation.divergence_b":
        components.append(
            _component(
                name="absolute",
                bound=float(bound["absolute_bound"]),
                observed=divergence_b_magnitude,
                evidence=str(CORRECTION_SOURCE.relative_to(HERE)),
            )
        )
    elif name == "conservation.divergence_j":
        components.append(
            _component(
                name="absolute",
                bound=float(bound["absolute_bound"]),
                observed=divergence_j_magnitude,
                evidence=str(SCALING_SOURCE.relative_to(HERE)),
            )
        )
    else:
        for component in ("absolute", "relative"):
            bound_key = f"{component}_bound"
            calibration_key = f"calibration_maximum_{component}_difference"
            if bound_key not in bound:
                continue
            observed = max(float(bound[calibration_key]), cohort[component])
            evidence = (
                str(REGISTRATION_SOURCE.relative_to(HERE))
                if float(bound[calibration_key]) >= cohort[component]
                else str(ACCEPTANCE_SOURCE.relative_to(HERE))
            )
            components.append(
                _component(
                    name=component,
                    bound=float(bound[bound_key]),
                    observed=observed,
                    evidence=evidence,
                )
            )
    effective = min(component["slack"] for component in components)
    flagged = effective > FLAG_THRESHOLD
    if name == "conservation.divergence_j":
        failability = {
            "demonstrated": bool(divergence_j_failability["exceeds_bound"]),
            "plausible_perturbation": divergence_j_failability["perturbation"],
            "perturbed_scored_value": float(
                divergence_j_failability["acceptance_absolute_difference"]
            ),
            "perturbed_bound_ratio": float(divergence_j_failability["bound_ratio"]),
            "reason": (
                "the banked perturbation exceeds the envelope, so the criterion is "
                "failable even though it cannot catch a twentyfold degradation"
            ),
        }
    else:
        limiting = min(components, key=lambda component: component["slack"])
        failability = {
            "demonstrated": True,
            "plausible_perturbation": (
                f"increase the banked {limiting['component']} compiled-versus-eager "
                f"difference for {name} above its calibration maximum"
            ),
            "reason": (
                "a banked calibration observation already reaches the limiting envelope"
                if effective <= 1.0 + 1e-12
                else (
                    "the observed quantity is close enough that a finite degradation "
                    "fails"
                )
            ),
        }
    return {
        "observable": name,
        "criterion_kind": bound["criterion_kind"],
        "assessment": "numeric_slack_assessed",
        "effective_slack": effective,
        "components": components,
        "failability": failability,
        "candidate_category_error": flagged,
        "flag_reason": (
            "effective slack exceeds the stated twentyfold discrimination threshold"
            if flagged
            else None
        ),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_receipt() -> dict[str, Any]:
    registration_source = _read_json(REGISTRATION_SOURCE)
    acceptance = _read_json(ACCEPTANCE_SOURCE)
    corrections = _read_json(CORRECTION_SOURCE)
    scaling = _read_json(SCALING_SOURCE)
    registration = _registration(registration_source)
    corrected = _corrected_registration(corrections)
    cohort = _cohort_observations(acceptance)
    if set(cohort) != {row["observable"] for row in registration}:
        raise RuntimeError("the frozen cohort does not cover every registration")

    rows = []
    for source_bound in registration:
        bound = source_bound | corrected.get(source_bound["observable"], {})
        if bound["criterion_kind"] == "exact_equality":
            rows.append(_exact_row(bound, cohort[bound["observable"]]))
        else:
            rows.append(
                _numeric_row(
                    bound,
                    cohort[bound["observable"]],
                    _divergence_b_magnitude(corrections),
                    float(
                        scaling["replacement_registration"][
                            "largest_observed_observable_magnitude"
                        ]
                    ),
                    scaling["failability"],
                )
            )

    numeric_slacks = [
        float(row["effective_slack"])
        for row in rows
        if row["effective_slack"] is not None
    ]
    flagged = [row for row in rows if row["candidate_category_error"]]
    by_name = {row["observable"]: row for row in rows}
    if not math.isclose(
        by_name["conservation.divergence_b"]["effective_slack"],
        1.6108655800448077,
    ):
        raise RuntimeError("divergence_b calibration anchor was not reproduced")
    if not math.isclose(
        by_name["conservation.divergence_j"]["effective_slack"],
        22.627416997969522,
    ):
        raise RuntimeError("divergence_j calibration anchor was not reproduced")
    if len(rows) != 69 or len(numeric_slacks) != 34:
        raise RuntimeError("the census does not cover all 69 registrations")

    sources = [
        REGISTRATION_SOURCE,
        ACCEPTANCE_SOURCE,
        CORRECTION_SOURCE,
        SCALING_SOURCE,
    ]
    return {
        "artifact": "bound_failability_census",
        "status": "complete",
        "scope": "measurement_only_no_bound_changes",
        "registered_bound_count": len(rows),
        "numeric_slack_assessed_count": len(numeric_slacks),
        "numeric_slack_not_assessable_count": len(rows) - len(numeric_slacks),
        "not_assessable_reason": (
            "Thirty-five exact-equality criteria have zero bound and zero observed "
            "mismatch, so their ratio is indeterminate; any mismatch nevertheless "
            "fails them. No registration is omitted from the table."
        ),
        "flagging_rule": {
            "candidate_category_error_threshold": FLAG_THRESHOLD,
            "comparison": "effective_slack > threshold",
            "justification": (
                "The calibrated divergence_j envelope at 22.63x is failable but "
                "cannot catch a twentyfold degradation; slack beyond 20x is "
                "therefore investigated as a possible mismatch of criterion kind, "
                "not described as comfortable margin."
            ),
            "flagged_count": len(flagged),
            "flagged_observables": [row["observable"] for row in flagged],
        },
        "distribution": {
            "population": "numeric effective slack for 34 envelope criteria",
            "minimum": min(numeric_slacks),
            "percentile_25": _percentile(numeric_slacks, 0.25),
            "median": _percentile(numeric_slacks, 0.5),
            "percentile_75": _percentile(numeric_slacks, 0.75),
            "percentile_90": _percentile(numeric_slacks, 0.9),
            "percentile_95": _percentile(numeric_slacks, 0.95),
            "maximum": max(numeric_slacks),
            "at_or_below_2x_count": sum(value <= 2.0 for value in numeric_slacks),
            "between_2x_and_20x_count": sum(
                2.0 < value <= FLAG_THRESHOLD for value in numeric_slacks
            ),
            "above_20x_count": sum(value > FLAG_THRESHOLD for value in numeric_slacks),
        },
        "calibration_anchors": {
            "conservation.divergence_b": {
                "slack": by_name["conservation.divergence_b"]["effective_slack"],
                "interpretation": "discriminating",
            },
            "conservation.divergence_j": {
                "slack": by_name["conservation.divergence_j"]["effective_slack"],
                "interpretation": "failable but not tight",
            },
        },
        "full_per_bound_table": sorted(rows, key=lambda row: row["observable"]),
        "bound_changes_authored": 0,
        "evidence_sources": [
            {"path": str(path.relative_to(HERE)), "sha256": _sha256(path)}
            for path in sources
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
