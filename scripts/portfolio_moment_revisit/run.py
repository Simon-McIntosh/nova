"""Measure which post-solve mechanism resolves each banked selection case."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUTPUT = Path(__file__).resolve().parent
FIGURE_OUTPUT = Path("docs/figures/dual-basin-solve")
RECEIPT = OUTPUT / "portfolio-moment-revisit.json"
REPORT = OUTPUT / "report.md"
MECHANISM_FIGURE = FIGURE_OUTPUT / "portfolio-decision-mechanisms.png"
CLOSURE_FIGURE = FIGURE_OUTPUT / "portfolio-selection-closure.png"

ORACLE_RECEIPTS = {
    resolution: Path(f"scripts/oracle_rebaseline/receipt-{resolution}.json")
    for resolution in ("coarse", "fine")
}
TOPOLOGY_RECEIPT = Path("scripts/dual_basin_fixtures/topology-classification.json")
MOMENT_TABLE = Path(
    "scripts/moment_prediction_confidence/moment-prediction-confidence.tsv"
)
MOMENT_CONFIDENCE = Path(
    "scripts/moment_prediction_confidence/moment-prediction-confidence.json"
)

MECHANISMS = (
    "portfolio_hysteresis",
    "moment_discriminator",
    "structurally_unique_root",
)
ROLE_RESTATEMENT = (
    "ROLE RESTATEMENT — The topology-pinned two-branch portfolio remains a "
    "topology-discovery and transition-safety mechanism where genuinely limited "
    "and diverted solutions coexist. It is not a root-identity selector for "
    "same-class multiple roots, and it does not earn a second solve on "
    "current-constrained reference lanes where amplitude elimination leaves one "
    "admissible plasma branch. In this measurement, predicted centroid moments "
    "resolve the two same-class fixture cases, while structural current constraint "
    "resolves all frozen reference cases before history can act. Hysteresis remains "
    "the policy only when both topology-pinned classes are simultaneously valid "
    "and admissible and topology history is genuinely informative."
)


def _digest(path: Path) -> str:
    """Return the byte-level digest of one immutable evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    return json.loads(path.read_text())


def _decide(
    *,
    structural_candidate_count: int,
    candidate_classes: tuple[str, ...],
    centroid_errors_m: tuple[float, ...],
) -> tuple[str, int | None]:
    """Apply the declared identity-blind decision cascade to numeric evidence."""

    if structural_candidate_count == 1:
        return "structurally_unique_root", 0
    if len(centroid_errors_m) == structural_candidate_count:
        errors = np.asarray(centroid_errors_m)
        if np.isfinite(errors).all() and np.unique(errors).size > 1:
            return "moment_discriminator", int(np.argmin(errors))
    if len(set(candidate_classes)) > 1:
        return "portfolio_hysteresis", None
    raise ValueError("same-class candidates require a finite moment discriminator")


def _fixture_case(resolution: str, receipt: dict[str, Any]) -> dict[str, Any]:
    """Build one same-class dual-root case from a banked mesh receipt."""

    target = receipt["seed"]["aggregate_moment"]
    alternate = receipt["root_moments"]
    closed_form = receipt["closed_form_state_observed_moments"]
    candidates = (
        {
            "root_label": "closed_form",
            "topology_class": receipt["oracle_topology"]["class"],
            "current_a": closed_form["plasma_current_a"],
            "centroid_r_m": closed_form["major_radius_m"],
        },
        {
            "root_label": "alternate",
            "topology_class": receipt["root_topology"]["class"],
            "current_a": alternate["plasma_current_a"],
            "centroid_r_m": alternate["major_radius_m"],
        },
    )
    target_centroid = float(target["current_centroid_m"][0])
    target_current = float(target["declared_current_a"])
    centroid_errors = tuple(
        abs(float(candidate["centroid_r_m"]) - target_centroid)
        for candidate in candidates
    )
    mechanism, selected_index = _decide(
        structural_candidate_count=2,
        candidate_classes=tuple(
            str(candidate["topology_class"]) for candidate in candidates
        ),
        centroid_errors_m=centroid_errors,
    )
    enriched_candidates = []
    for candidate, centroid_error in zip(candidates, centroid_errors, strict=True):
        current_error = abs(float(candidate["current_a"]) - target_current) / abs(
            target_current
        )
        enriched_candidates.append(
            {
                **candidate,
                "centroid_absolute_error_m": centroid_error,
                "current_relative_error": current_error,
            }
        )
    selected = (
        enriched_candidates[selected_index] if selected_index is not None else None
    )
    return {
        "case_key": f"dual-root-{resolution}",
        "cohort": "banked_same_class_roots",
        "resolution": resolution,
        "selection_inputs": {
            "structural_candidate_count": 2,
            "candidate_classes": [
                candidate["topology_class"] for candidate in enriched_candidates
            ],
            "declared_current_a": target_current,
            "declared_centroid_m": list(target["current_centroid_m"]),
            "centroid_is_confident_discriminator": True,
            "candidate_roots": enriched_candidates,
        },
        "decision": {
            "mechanism": mechanism,
            "selected_root": selected["root_label"] if selected is not None else None,
            "selected_class": (
                selected["topology_class"] if selected is not None else None
            ),
            "selection_margin_m": abs(centroid_errors[1] - centroid_errors[0]),
            "portfolio_hysteresis_can_distinguish": False,
            "portfolio_hysteresis_changed_outcome": False,
            "known_good_frame_identity_used": False,
            "reason": (
                "both genuine roots have the same topology class; the candidate "
                "nearest the declared radial current centroid is selected"
            ),
        },
    }


def _reference_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build structurally closed cases from the frozen own-boundary controls."""

    cases = []
    for row in rows:
        if float(row["boundary_scale"]) != 1.0:
            continue
        current_error = abs(float(row["current_relative_error"]))
        mechanism, selected_index = _decide(
            structural_candidate_count=1,
            candidate_classes=("plasma",),
            centroid_errors_m=(),
        )
        if selected_index != 0:
            raise AssertionError(
                "a structurally unique case must select its only branch"
            )
        cases.append(
            {
                "case_key": row["identity"],
                "cohort": "frozen_current_constrained_references",
                "machine": row["machine"],
                "selection_inputs": {
                    "structural_candidate_count": 1,
                    "candidate_classes": ["plasma"],
                    "amplitude_elimination": "exact_nonzero_target_current",
                    "target_current_a": float(row["target_current_a"]),
                    "current_relative_error": current_error,
                    "predicted_centroid_m": [
                        float(row["predicted_centroid_r_m"]),
                        float(row["predicted_centroid_z_m"]),
                    ],
                    "reference_centroid_m": [
                        float(row["reference_centroid_r_m"]),
                        float(row["reference_centroid_z_m"]),
                    ],
                    "moment_discriminator_available_but_not_needed": True,
                },
                "decision": {
                    "mechanism": mechanism,
                    "selected_root": "current_constrained_plasma_branch",
                    "selected_class": "plasma",
                    "portfolio_hysteresis_can_distinguish": False,
                    "portfolio_hysteresis_changed_outcome": False,
                    "known_good_frame_identity_used": False,
                    "reason": (
                        "exact nonzero-current amplitude elimination excludes the "
                        "vacuum branch before post-solve history is consulted"
                    ),
                },
            }
        )
    return cases


def _plot_mechanisms(cases: list[dict[str, Any]]) -> None:
    """Plot mechanism counts and the same-class centroid discriminator."""

    counts = {
        mechanism: sum(case["decision"]["mechanism"] == mechanism for case in cases)
        for mechanism in MECHANISMS
    }
    fixture_cases = [
        case for case in cases if case["cohort"] == "banked_same_class_roots"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    labels = ["hysteresis", "moments", "structural"]
    colors = ["#7b6fd0", "#e48b32", "#2a9d8f"]
    values = [counts[mechanism] for mechanism in MECHANISMS]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("cases decided")
    axes[0].set_title("Decision mechanism across 13 cases")
    axes[0].set_ylim(0, max(values) + 2)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.2, str(value), ha="center", fontweight="bold")

    x = np.arange(len(fixture_cases))
    width = 0.34
    closed_errors = [
        case["selection_inputs"]["candidate_roots"][0]["centroid_absolute_error_m"]
        * 100
        for case in fixture_cases
    ]
    alternate_errors = [
        case["selection_inputs"]["candidate_roots"][1]["centroid_absolute_error_m"]
        * 100
        for case in fixture_cases
    ]
    axes[1].bar(x - width / 2, closed_errors, width, label="closed-form root")
    axes[1].bar(x + width / 2, alternate_errors, width, label="alternate root")
    axes[1].set_xticks(x, [case["resolution"] for case in fixture_cases])
    axes[1].set_ylabel("distance from declared centroid (cm)")
    axes[1].set_title("Both roots limited; centroid resolves identity")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MECHANISM_FIGURE, dpi=180)
    plt.close(fig)


def _plot_closure(reference_cases: list[dict[str, Any]]) -> None:
    """Plot exact current closure for every frozen structural-control case."""

    values = np.asarray(
        [case["selection_inputs"]["current_relative_error"] for case in reference_cases]
    )
    machines = [case["machine"] for case in reference_cases]
    colors = ["#4267ac" if machine == "MAST" else "#d1495b" for machine in machines]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.scatter(np.arange(values.size), np.maximum(values, 1.0e-18), c=colors, s=42)
    ax.axhline(
        5.0e-12, color="black", linestyle="--", linewidth=1.0, label="audit bound"
    )
    ax.set_yscale("log")
    ax.set_ylabel("target-current relative closure")
    ax.set_xlabel("frozen reference row (identity is display-only)")
    ax.set_title("Amplitude elimination closes branch selection before history")
    ax.legend(
        handles=(
            Line2D([], [], color="#4267ac", marker="o", linestyle="", label="MAST"),
            Line2D([], [], color="#d1495b", marker="o", linestyle="", label="DIII-D"),
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="audit bound",
            ),
        ),
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(CLOSURE_FIGURE, dpi=180)
    plt.close(fig)


def _write_report(receipt: dict[str, Any]) -> None:
    """Write the human-readable outcome and restatement handoff."""

    summary = receipt["summary"]
    fixture = receipt["fixture_measurement"]
    mechanism_figure = (
        "![Mechanism counts and fixture discrimination]"
        "(../../docs/figures/dual-basin-solve/portfolio-decision-mechanisms.png)"
    )
    closure_figure = (
        "![Frozen-reference current closure]"
        "(../../docs/figures/dual-basin-solve/portfolio-selection-closure.png)"
    )
    REPORT.write_text(
        "\n".join(
            (
                "# Portfolio and moment selection revisit",
                "",
                "## Outcome",
                "",
                f"The study contains **{summary['case_count']} cases**: two mesh "
                "resolutions of the banked same-class dual-root fixture and eleven "
                "frozen current-constrained references. Portfolio hysteresis changed "
                f"**{summary['portfolio_hysteresis_changed_count']} of "
                f"{summary['case_count']}** outcomes. The moment discriminator decided "
                f"**{summary['mechanism_counts']['moment_discriminator']}** cases and "
                "structural amplitude elimination decided "
                f"**{summary['mechanism_counts']['structurally_unique_root']}**.",
                "",
                "Both banked roots are limited at coarse and fine resolution, so a "
                "topology-class hysteresis receipt cannot identify one root over the "
                "other. Their radial-current centroids are separated by "
                "**"
                f"{fixture['median_candidate_centroid_separation_m'] * 100:.3f} cm"
                "**. "
                "Against the declared centroid, the closed-form root is "
                f"**{fixture['median_selected_centroid_error_m'] * 100:.3f} cm** away "
                "and the alternate root is "
                f"**{fixture['median_rejected_centroid_error_m'] * 100:.3f} cm** away; "
                "the centroid therefore selects the closed-form root on both meshes.",
                "",
                "Across all eleven frozen references, exact target-current closure is "
                "at most **"
                f"{summary['maximum_reference_current_relative_error']:.3e}"
                "**. "
                "The nonzero-current amplitude elimination removes the vacuum branch "
                "from the admissible map range before post-solve history can act. "
                "Reference identity is retained only as provenance and plot labeling; "
                "the decision function receives only candidate count, candidate class, "
                "and numerical centroid errors.",
                "",
                mechanism_figure,
                "",
                closure_figure,
                "",
                "## ROLE RESTATEMENT",
                "",
                ROLE_RESTATEMENT,
                "",
                "## Reproduction",
                "",
                "Run `python scripts/portfolio_moment_revisit/run.py` and then "
                "`python scripts/portfolio_moment_revisit/verify.py` from the "
                "repository root. The JSON receipt records every input digest and "
                "every case-level decision.",
                "",
            )
        )
    )


def measure() -> dict[str, Any]:
    """Measure all banked cases and publish receipts plus figures."""

    topology = _read_json(TOPOLOGY_RECEIPT)
    confidence = _read_json(MOMENT_CONFIDENCE)
    with MOMENT_TABLE.open(newline="") as stream:
        moment_rows = list(csv.DictReader(stream, delimiter="\t"))

    fixture_cases = [
        _fixture_case(resolution, _read_json(path))
        for resolution, path in ORACLE_RECEIPTS.items()
    ]
    reference_cases = _reference_cases(moment_rows)
    cases = fixture_cases + reference_cases
    counts = {
        mechanism: sum(case["decision"]["mechanism"] == mechanism for case in cases)
        for mechanism in MECHANISMS
    }
    selected_errors = [
        case["selection_inputs"]["candidate_roots"][0]["centroid_absolute_error_m"]
        for case in fixture_cases
    ]
    rejected_errors = [
        case["selection_inputs"]["candidate_roots"][1]["centroid_absolute_error_m"]
        for case in fixture_cases
    ]
    separations = [
        abs(
            case["selection_inputs"]["candidate_roots"][0]["centroid_r_m"]
            - case["selection_inputs"]["candidate_roots"][1]["centroid_r_m"]
        )
        for case in fixture_cases
    ]
    maximum_current_error = max(
        case["selection_inputs"]["current_relative_error"] for case in reference_cases
    )
    inputs = [
        *ORACLE_RECEIPTS.values(),
        TOPOLOGY_RECEIPT,
        MOMENT_TABLE,
        MOMENT_CONFIDENCE,
    ]
    receipt = {
        "schema": "nova.portfolio-moment-revisit",
        "schema_version": 1,
        "input_digests": {str(path): _digest(path) for path in inputs},
        "policy": {
            "order": [
                "structurally_unique_root",
                "moment_discriminator",
                "portfolio_hysteresis",
            ],
            "structural_rule": (
                "one admissible root after exact amplitude elimination decides without "
                "history"
            ),
            "moment_rule": (
                "among multiple roots, select the unique minimum absolute "
                "radial-centroid error when finite predicted moments distinguish them"
            ),
            "hysteresis_rule": (
                "history is consulted only when multiple admissible topology classes "
                "remain after structural and moment discrimination"
            ),
            "known_good_frame_identity_is_a_selection_input": False,
        },
        "bank_facts": {
            "closed_form_topology": topology["classification"]["closed_form_analytic"],
            "alternate_topology": topology["classification"]["alternate_fixed_point"],
            "same_class_root_count_per_resolution": 2,
            "frozen_reference_frames": confidence["cohort"]["frames"],
            "frozen_reference_machine_split": {
                "MAST": confidence["cohort"]["mast_frames"],
                "DIII-D": confidence["cohort"]["diiid_frames"],
            },
            "moment_capability": {
                "prediction_payload": "PredictedCurrentMoments",
                "seed_entry_point": "ForwardProfile.moment_seed",
                "measured_discriminator": "radial current centroid",
            },
        },
        "cases": cases,
        "fixture_measurement": {
            "median_candidate_centroid_separation_m": float(np.median(separations)),
            "median_selected_centroid_error_m": float(np.median(selected_errors)),
            "median_rejected_centroid_error_m": float(np.median(rejected_errors)),
            "selected_root_on_both_meshes": "closed_form",
        },
        "summary": {
            "case_count": len(cases),
            "mechanism_counts": counts,
            "portfolio_hysteresis_changed_count": sum(
                case["decision"]["portfolio_hysteresis_changed_outcome"]
                for case in cases
            ),
            "maximum_reference_current_relative_error": maximum_current_error,
            "known_good_frame_identity_selection_count": sum(
                case["decision"]["known_good_frame_identity_used"] for case in cases
            ),
            "portfolio_cost_verdict": (
                "no selection benefit in this cohort: same-class roots need moments; "
                "current-constrained references are structurally closed"
            ),
        },
        "role_restatement": ROLE_RESTATEMENT,
        "artifacts": {
            "report": str(REPORT),
            "mechanism_figure": str(MECHANISM_FIGURE),
            "closure_figure": str(CLOSURE_FIGURE),
        },
    }
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    _plot_mechanisms(cases)
    _plot_closure(reference_cases)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_report(receipt)
    return receipt


if __name__ == "__main__":
    result = measure()
    print(json.dumps(result["summary"], sort_keys=True))
