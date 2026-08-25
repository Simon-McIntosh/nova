"""Contrast one machine-precision MAST frame with the plateau cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_RESIDUAL_RECEIPT = Path(
    "docs/figures/plateau-input-attribution/label-seed-residual-field.json"
)
DEFAULT_MACHINE_PRECISION_RECEIPT = Path(
    "docs/figures/current-constrained-forward-solve/mast-constrained/"
    "current-constrained-frozen-six-scorecard.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/plateau-input-attribution/converged-frame-contrast.json"
)
CONTROL_REFERENCE = "22086/43"
MACHINE_PRECISION_RECEIPT_COMMIT = "a0aee18c330c35e8009bbe0b96f7fd3d078b9a00"
EXPECTED_MACHINE_PRECISION_RESIDUAL = 2.911868346631881e-16
EXPECTED_CURRENT_ROUTE_RESIDUAL = 0.005049061244966342


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_name(row: dict[str, Any]) -> str:
    reference = row["reference"]
    return f"{int(reference['shot'])}/{int(reference['slice_index'])}"


def _nested(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        value = value[key]
    return value


def _numeric_contrast(
    rows: dict[str, dict[str, Any]],
    path: tuple[str, ...],
) -> dict[str, Any]:
    control_value = float(_nested(rows[CONTROL_REFERENCE], path))
    plateau_values = {
        reference: float(_nested(row, path))
        for reference, row in rows.items()
        if reference != CONTROL_REFERENCE
    }
    plateau_minimum = min(plateau_values.values())
    plateau_maximum = max(plateau_values.values())
    return {
        "control_value": control_value,
        "plateau_values_by_reference": plateau_values,
        "plateau_sample_count": len(plateau_values),
        "plateau_range": {
            "minimum": plateau_minimum,
            "maximum": plateau_maximum,
        },
        "control_outside_plateau_range": bool(
            control_value < plateau_minimum or control_value > plateau_maximum
        ),
    }


def _categorical_contrast(
    rows: dict[str, dict[str, Any]],
    path: tuple[str, ...],
) -> dict[str, Any]:
    control_value = _nested(rows[CONTROL_REFERENCE], path)
    plateau_values = {
        reference: _nested(row, path)
        for reference, row in rows.items()
        if reference != CONTROL_REFERENCE
    }
    plateau_unique = sorted(set(plateau_values.values()))
    return {
        "control_value": control_value,
        "plateau_values_by_reference": plateau_values,
        "plateau_sample_count": len(plateau_values),
        "plateau_unique_values": plateau_unique,
        "control_absent_from_plateau_values": control_value not in plateau_unique,
    }


def _production_paths_between(first: str, second: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", first, second, "--", "nova"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        path
        for path in result.stdout.splitlines()
        if path.endswith(".py") and path.startswith("nova/")
    )


def _machine_precision_row(receipt: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in receipt["per_shot"] if _reference_name(row) == CONTROL_REFERENCE
    ]
    if len(matches) != 1:
        raise ValueError(
            f"machine-precision receipt must contain one {CONTROL_REFERENCE} row"
        )
    return matches[0]


def _measured_rows(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {_reference_name(row): row for row in receipt["per_reference"]}
    if len(rows) != 6 or CONTROL_REFERENCE not in rows:
        raise ValueError(
            "residual receipt must contain the frozen six including control"
        )
    return rows


def build_receipt(
    residual_receipt_path: Path = DEFAULT_RESIDUAL_RECEIPT,
    machine_precision_receipt_path: Path = DEFAULT_MACHINE_PRECISION_RECEIPT,
) -> dict[str, Any]:
    """Build the exact contrast from the two immutable source receipts."""
    residual_receipt = json.loads(residual_receipt_path.read_text(encoding="utf-8"))
    machine_precision_receipt = json.loads(
        machine_precision_receipt_path.read_text(encoding="utf-8")
    )
    rows = _measured_rows(residual_receipt)
    control = rows[CONTROL_REFERENCE]
    control_solve = control["label_seeded_newton_solve"]
    machine_precision_row = _machine_precision_row(machine_precision_receipt)
    machine_precision_solve = machine_precision_row["constrained_solve"]

    if machine_precision_solve["terminal_residual"] != (
        EXPECTED_MACHINE_PRECISION_RESIDUAL
    ):
        raise ValueError("machine-precision control residual changed")
    if control_solve["terminal_relative_residual"] != EXPECTED_CURRENT_ROUTE_RESIDUAL:
        raise ValueError("current label-seeded control residual changed")
    if control_solve["terminal_plasma_current_fraction_of_label"] != 1.0:
        raise ValueError("current label-seeded control no longer pins plasma current")

    substitution_definitions = {
        "wall": (
            "boundary-band plus wall-state fraction of full residual squared magnitude",
            (
                "one_application_residual",
                "candidate_pattern_scores",
                "scores",
                "wall",
            ),
        ),
        "conductor_current_wiring": (
            "largest single-circuit response projection fraction of full residual "
            "squared magnitude",
            (
                "one_application_residual",
                "candidate_pattern_scores",
                "scores",
                "conductor_current_wiring",
            ),
        ),
        "profiles": (
            "closed-core residual fraction explained by twelve labelled-flux bins",
            (
                "one_application_residual",
                "candidate_pattern_scores",
                "scores",
                "profiles",
            ),
        ),
        "discretisation": (
            "normalised neighbour-difference energy on one-stencil grid edges",
            (
                "one_application_residual",
                "candidate_pattern_scores",
                "scores",
                "discretisation",
            ),
        ),
    }
    substitution_contrasts = {
        candidate: {
            "measured_property": definition,
            **_numeric_contrast(rows, path),
        }
        for candidate, (definition, path) in substitution_definitions.items()
    }

    supporting_properties = {
        "full_state_residual_sup_wb": _numeric_contrast(
            rows,
            ("one_application_residual", "norms", "full_state", "sup_wb"),
        ),
        "full_state_residual_l2_wb": _numeric_contrast(
            rows,
            ("one_application_residual", "norms", "full_state", "l2_wb"),
        ),
        "closed_flux_squared_magnitude_fraction": _numeric_contrast(
            rows,
            (
                "one_application_residual",
                "regional_squared_magnitude",
                "closed_flux_region",
                "fraction_of_grid_squared_magnitude",
            ),
        ),
        "scrape_off_squared_magnitude_fraction": _numeric_contrast(
            rows,
            (
                "one_application_residual",
                "regional_squared_magnitude",
                "scrape_off_layer",
                "fraction_of_grid_squared_magnitude",
            ),
        ),
        "boundary_band_squared_magnitude_fraction": _numeric_contrast(
            rows,
            (
                "one_application_residual",
                "regional_squared_magnitude",
                "within_one_stencil_width_of_boundary",
                "fraction_of_grid_squared_magnitude",
            ),
        ),
        "best_stored_circuit_one_based": _categorical_contrast(
            rows,
            (
                "one_application_residual",
                "single_circuit_green_pattern",
                "best_stored_circuit_one_based",
            ),
        ),
        "equivalent_current_correction_a": _numeric_contrast(
            rows,
            (
                "one_application_residual",
                "single_circuit_green_pattern",
                "equivalent_current_correction_a",
            ),
        ),
        "terminal_relative_residual": _numeric_contrast(
            rows, ("label_seeded_newton_solve", "terminal_relative_residual")
        ),
        "terminal_plasma_current_fraction_of_label": _numeric_contrast(
            rows,
            (
                "label_seeded_newton_solve",
                "terminal_plasma_current_fraction_of_label",
            ),
        ),
        "stored_source_support_nodes": _numeric_contrast(
            rows, ("reference", "source_coordinate", "support_nodes")
        ),
        "stored_lcfs_contour_discrepancy_fraction": _numeric_contrast(
            rows,
            (
                "reference",
                "qualification_before_attribution",
                "stored_lcfs_contour_sup_discrepancy_fraction_of_declared_span",
            ),
        ),
    }

    earlier_tree = MACHINE_PRECISION_RECEIPT_COMMIT
    current_tree = str(residual_receipt["source_revision"])
    production_paths = _production_paths_between(earlier_tree, current_tree)
    if len(production_paths) <= 1:
        raise ValueError(
            "tree comparison must retain its multi-mechanism qualification"
        )

    response_identity = control["prescribed_current_policy"]["response_input_digests"][
        "combined_sha256"
    ]
    earlier_response_identity = machine_precision_row["prescribed_current_policy"][
        "response_input_digests"
    ]["combined_sha256"]
    if response_identity != earlier_response_identity:
        raise ValueError("the compared routes do not share the response identity")

    separates = {
        name: item["control_outside_plateau_range"]
        for name, item in substitution_contrasts.items()
    }
    separating_candidates = [name for name, value in separates.items() if value]
    return {
        "receipt": "MAST machine-precision frame contrast",
        "source_revision": current_tree,
        "inputs": {
            "residual_field_receipt": {
                "path": str(residual_receipt_path),
                "sha256": _sha256(residual_receipt_path),
                "source_revision": current_tree,
            },
            "machine_precision_receipt": {
                "path": str(machine_precision_receipt_path),
                "sha256": _sha256(machine_precision_receipt_path),
                "artifact_banking_commit": earlier_tree,
                "execution_source_revision_embedded": False,
            },
        },
        "comparison_contract": {
            "control_reference": CONTROL_REFERENCE,
            "plateau_references": sorted(
                reference for reference in rows if reference != CONTROL_REFERENCE
            ),
            "control_sample_count": 1,
            "plateau_sample_count": 5,
            "total_sample_count": 6,
            "separation_rule": (
                "descriptive non-overlap: the control value must lie outside the "
                "inclusive range of the five plateau values"
            ),
            "power_qualification": (
                "A one-control, five-comparator range contrast is descriptive; it "
                "does not establish a population effect or causal mechanism."
            ),
        },
        "route_context": {
            "machine_precision_route": {
                "reference_seeded": True,
                "target_current_a": machine_precision_solve[
                    "terminal_plasma_current_a"
                ],
                "terminal_relative_residual": machine_precision_solve[
                    "terminal_residual"
                ],
                "terminal_branch_classification": machine_precision_solve[
                    "outcome_class"
                ],
                "terminal_plasma_current_fraction_of_label": 1.0,
                "response_identity": earlier_response_identity,
            },
            "current_pinned_label_seeded_route": {
                "source_revision": current_tree,
                "reference_seeded": True,
                "target_current_a": control_solve["target_current_a"],
                "terminal_relative_residual": control_solve[
                    "terminal_relative_residual"
                ],
                "terminal_branch_classification": control_solve[
                    "terminal_branch_classification"
                ],
                "terminal_plasma_current_fraction_of_label": control_solve[
                    "terminal_plasma_current_fraction_of_label"
                ],
                "response_identity": response_identity,
            },
            "shared_route_properties": {
                "initial_state": "stored EFIT label",
                "requested_branch": "diverted",
                "target_current_policy": "absolute stored scalar plasma current",
                "stored_circuit_count": 101,
                "response_identity_equal": True,
            },
            "tree_difference": {
                "machine_precision_artifact_banking_tree": earlier_tree,
                "current_measurement_tree": current_tree,
                "changed_production_python_path_count": len(production_paths),
                "changed_production_python_paths": production_paths,
                "machine_precision_execution_source_revision_was_not_embedded": True,
                "attribution": "DECLINED",
                "reason": (
                    "More than one production mechanism changed, and the older "
                    "receipt did not embed its execution source revision. The "
                    "residual change is context, not a causal estimate."
                ),
            },
        },
        "substitution_contrasts": substitution_contrasts,
        "supporting_property_contrasts": supporting_properties,
        "verdict": {
            "separating_candidates": separating_candidates,
            "separating_candidate_count": len(separating_candidates),
            "properties_compared": sorted(substitution_contrasts),
            "sample_count": 6,
            "statement": (
                "None of the four measured substitution-pattern properties "
                "separates 22086/43 from the five plateau frames: every control "
                "value lies inside the corresponding five-frame range."
            ),
            "causal_attribution": "DECLINED",
        },
    }


def write_receipt(receipt: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def check_receipt(
    output: Path = DEFAULT_OUTPUT,
    residual_receipt: Path = DEFAULT_RESIDUAL_RECEIPT,
    machine_precision_receipt: Path = DEFAULT_MACHINE_PRECISION_RECEIPT,
) -> dict[str, Any]:
    """Fail closed unless the banked receipt exactly reproduces its inputs."""
    banked = json.loads(output.read_text(encoding="utf-8"))
    expected = build_receipt(residual_receipt, machine_precision_receipt)
    if banked != expected:
        raise ValueError("banked contrast receipt does not reproduce its inputs")
    if banked["verdict"]["separating_candidate_count"] != 0:
        raise ValueError("banked contrast no longer records the measured null")
    if banked["route_context"]["tree_difference"]["attribution"] != "DECLINED":
        raise ValueError("multi-mechanism historical attribution was not declined")
    return banked


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument(
        "--residual-receipt", type=Path, default=DEFAULT_RESIDUAL_RECEIPT
    )
    measure.add_argument(
        "--machine-precision-receipt",
        type=Path,
        default=DEFAULT_MACHINE_PRECISION_RECEIPT,
    )
    measure.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check = commands.add_parser("check")
    check.add_argument(
        "--residual-receipt", type=Path, default=DEFAULT_RESIDUAL_RECEIPT
    )
    check.add_argument(
        "--machine-precision-receipt",
        type=Path,
        default=DEFAULT_MACHINE_PRECISION_RECEIPT,
    )
    check.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    if arguments.command == "measure":
        receipt = build_receipt(
            arguments.residual_receipt, arguments.machine_precision_receipt
        )
        write_receipt(receipt, arguments.output)
    else:
        receipt = check_receipt(
            arguments.output,
            arguments.residual_receipt,
            arguments.machine_precision_receipt,
        )
    print(
        "CONVERGED_FRAME_CONTRAST "
        f"references={receipt['comparison_contract']['total_sample_count']} "
        f"substitutions={len(receipt['substitution_contrasts'])} "
        f"separating={receipt['verdict']['separating_candidate_count']} "
        f"attribution={receipt['verdict']['causal_attribution']} verdict=PASS"
    )


if __name__ == "__main__":
    main()
