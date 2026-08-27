"""Score the complete MAST and DIII-D geometry banks without cohort shrinkage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MAST_BANK = Path(
    "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
DIIID_BANK = Path(
    "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/efit-baseline-demonstration/demonstration-score/receipt.json"
)

EXPECTED_MAST_ARMS = tuple(
    (shot, slice_index, arm)
    for shot, slice_index in (
        (21978, 35),
        (21983, 35),
        (21985, 51),
        (21986, 46),
        (21989, 55),
        (22086, 43),
    )
    for arm in ("pure", "mixed")
)
EXPECTED_DIIID_ROWS = (
    ("d3d_shot_00000c4a7b.parquet", 179),
    ("d3d_shot_0003ff34e7.parquet", 44),
    ("d3d_shot_001554e054.parquet", 144),
    ("d3d_shot_002495e835.parquet", 146),
    ("d3d_shot_0040ca9bdc.parquet", 137),
)

MAST_CLASS_AGREEMENT_MINIMUM = 7
CLOSED_BOUNDARY_RMS_MAXIMUM_M = 0.55
SADDLE_DISTANCE_MAXIMUM_M = 0.04


class ScoreRefusal(ValueError):
    """Raised when a bank cannot enter the fixed-cohort demonstration score."""


def _identity_record(
    identity: tuple[Any, ...], names: tuple[str, ...]
) -> dict[str, Any]:
    return dict(zip(names, identity, strict=True))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _explicit_boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _gate_row(
    machine: str,
    identity: tuple[Any, ...],
    names: tuple[str, ...],
    *,
    value: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "identity": {"machine": machine, **_identity_record(identity, names)},
        "value": value,
        "reason": reason,
    }


def _validate_identities(
    actual: list[tuple[Any, ...]],
    expected: tuple[tuple[Any, ...], ...],
    cohort: str,
) -> None:
    if not actual:
        raise ScoreRefusal(f"{cohort} bank is empty")
    duplicates = sorted(
        identity for identity, count in Counter(actual).items() if count > 1
    )
    if duplicates:
        raise ScoreRefusal(
            f"{cohort} bank contains duplicate identities: {duplicates!r}"
        )
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    substituted = sorted(actual_set - expected_set)
    if missing or substituted or len(actual) != len(expected):
        mismatch = (
            f"missing={missing!r}, substituted={substituted!r}, "
            f"rows={len(actual)}, expected={len(expected)}"
        )
        raise ScoreRefusal(f"{cohort} cohort mismatch: {mismatch}")


def _mast_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ScoreRefusal("MAST bank rows are missing")
    if any(not isinstance(row, dict) for row in rows):
        raise ScoreRefusal("MAST bank contains a non-object row")
    identities = [
        (row.get("shot"), row.get("slice_index"), row.get("arm")) for row in rows
    ]
    _validate_identities(identities, EXPECTED_MAST_ARMS, "MAST")
    ordered = {identity: row for identity, row in zip(identities, rows, strict=True)}
    return [ordered[identity] for identity in EXPECTED_MAST_ARMS]


def _diiid_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    rows = result.get("frame_records") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise ScoreRefusal("DIII-D bank frame records are missing")
    if any(not isinstance(row, dict) for row in rows):
        raise ScoreRefusal("DIII-D bank contains a non-object row")
    identities = [(row.get("shot"), row.get("frame")) for row in rows]
    _validate_identities(identities, EXPECTED_DIIID_ROWS, "DIII-D")
    ordered = {identity: row for identity, row in zip(identities, rows, strict=True)}
    return [ordered[identity] for identity in EXPECTED_DIIID_ROWS]


def _mast_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_receipts = []
    for identity, row in zip(EXPECTED_MAST_ARMS, rows, strict=True):
        declared_identity = f"{identity[0]}/{identity[1]}"
        if row.get("identity") != declared_identity:
            raise ScoreRefusal(
                f"MAST row {identity!r} substituted its declared identity field"
            )
        converged = _explicit_boolean(row.get("converged"))
        reference_class = row.get("efit_label")
        achieved_class = row.get("nova_achieved_class")
        agrees = _explicit_boolean(row.get("label_agreement"))
        computed_agreement = None
        if reference_class in {"limited", "diverted"} and achieved_class in {
            "limited",
            "diverted",
        }:
            computed_agreement = achieved_class == reference_class
        if computed_agreement is None or agrees is not computed_agreement:
            agrees = None
        failures = row.get("comparison_failures")
        comparison_failures = failures if isinstance(failures, list) else None
        row_receipts.append(
            {
                "identity": _identity_record(identity, ("shot", "slice_index", "arm")),
                "converged": converged,
                "class_agreement": agrees,
                "comparison_failures": comparison_failures,
                "closed_boundary_symmetric_rms_m": _finite_number(
                    row.get("binding_to_efit_lcfs_rms_m")
                ),
                "saddle_distance_m": _finite_number(
                    row.get("selected_saddle_to_efit_x_point_m")
                ),
            }
        )
    return {
        "declared_row_count": len(EXPECTED_MAST_ARMS),
        "rows": row_receipts,
    }


def _diiid_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_receipts = []
    for identity, row in zip(EXPECTED_DIIID_ROWS, rows, strict=True):
        metrics = row.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        failures = metrics.get("boundary_comparison_failures")
        row_receipts.append(
            {
                "identity": _identity_record(identity, ("shot", "frame")),
                "finite": _explicit_boolean(row.get("finite")),
                "converged": _explicit_boolean(row.get("converged")),
                "class_agreement": _explicit_boolean(
                    metrics.get("topology_class_agreement")
                ),
                "comparison_failures": failures if isinstance(failures, list) else None,
                "closed_boundary_symmetric_rms_m": _finite_number(
                    metrics.get("closed_boundary_symmetric_rms_distance_m")
                ),
                "saddle_distance_m": _finite_number(
                    metrics.get("polished_saddle_to_nearest_efit_x_m")
                ),
            }
        )
    return {
        "declared_row_count": len(rows),
        "rows": row_receipts,
    }


def _class_agreement_gate(mast_score: dict[str, Any]) -> dict[str, Any]:
    contributing_rows = []
    agreement_count = 0
    for row in mast_score["rows"]:
        agreement = row["class_agreement"]
        if agreement is True:
            reason = "class_agreement"
            agreement_count += 1
        elif agreement is False:
            reason = "class_disagreement"
        else:
            reason = "class_agreement_unavailable"
        identity = tuple(row["identity"][key] for key in ("shot", "slice_index", "arm"))
        contributing_rows.append(
            _gate_row(
                "MAST",
                identity,
                ("shot", "slice_index", "arm"),
                value=agreement,
                reason=reason,
            )
        )
    return {
        "measured": {
            "agreement_count": agreement_count,
            "declared_denominator": len(EXPECTED_MAST_ARMS),
        },
        "threshold": {
            "minimum_inclusive": MAST_CLASS_AGREEMENT_MINIMUM,
            "declared_denominator": len(EXPECTED_MAST_ARMS),
        },
        "contributing_rows": contributing_rows,
        "excluded_rows": [],
        "passes": agreement_count >= MAST_CLASS_AGREEMENT_MINIMUM,
    }


def _rms_gate(
    mast_score: dict[str, Any], diiid_score: dict[str, Any]
) -> dict[str, Any]:
    contributing_rows = []
    excluded_rows = []
    disqualifying_exclusion = False
    sources = (
        ("MAST", mast_score["rows"], ("shot", "slice_index", "arm")),
        ("DIII-D", diiid_score["rows"], ("shot", "frame")),
    )
    for machine, rows, names in sources:
        for row in rows:
            identity = tuple(row["identity"][name] for name in names)
            agreement = row["class_agreement"]
            if agreement is False:
                excluded_rows.append(
                    _gate_row(
                        machine,
                        identity,
                        names,
                        value=None,
                        reason="class_disagreement",
                    )
                )
                continue
            reasons = []
            if agreement is None:
                reasons.append("class_agreement_unavailable")
            if row["converged"] is not True:
                reasons.append(
                    "non_converged"
                    if row["converged"] is False
                    else "convergence_unavailable"
                )
            if machine == "DIII-D" and row["finite"] is not True:
                reasons.append(
                    "non_finite_receipt"
                    if row["finite"] is False
                    else "finite_receipt_unavailable"
                )
            failures = row["comparison_failures"]
            if failures is None:
                reasons.append("comparison_failures_unavailable")
            else:
                reasons.extend(
                    f"comparison_operand_failure:{failure}" for failure in failures
                )
            rms = row["closed_boundary_symmetric_rms_m"]
            if rms is None:
                reasons.append("closed_boundary_rms_unavailable")
            if reasons:
                disqualifying_exclusion = True
                excluded_rows.append(
                    _gate_row(
                        machine,
                        identity,
                        names,
                        value=None,
                        reason=";".join(reasons),
                    )
                )
            else:
                contributing_rows.append(
                    _gate_row(
                        machine,
                        identity,
                        names,
                        value=rms,
                        reason="eligible_converged_class_agreement",
                    )
                )
    values = [row["value"] for row in contributing_rows]
    worst = max(values) if values else None
    return {
        "measured": {
            "worst_closed_boundary_symmetric_rms_m": worst,
            "eligible_row_count": len(contributing_rows),
            "declared_denominator": len(EXPECTED_MAST_ARMS) + len(EXPECTED_DIIID_ROWS),
        },
        "threshold": {"maximum_inclusive_m": CLOSED_BOUNDARY_RMS_MAXIMUM_M},
        "contributing_rows": contributing_rows,
        "excluded_rows": excluded_rows,
        "passes": bool(values)
        and not disqualifying_exclusion
        and worst <= CLOSED_BOUNDARY_RMS_MAXIMUM_M,
    }


def _saddle_gate(
    mast_score: dict[str, Any], diiid_score: dict[str, Any]
) -> dict[str, Any]:
    contributing_rows = []
    excluded_rows = []
    sources = (
        ("MAST", mast_score["rows"], ("shot", "slice_index", "arm")),
        ("DIII-D", diiid_score["rows"], ("shot", "frame")),
    )
    for machine, rows, names in sources:
        for row in rows:
            identity = tuple(row["identity"][name] for name in names)
            saddle = row["saddle_distance_m"]
            if saddle is None:
                excluded_rows.append(
                    _gate_row(
                        machine,
                        identity,
                        names,
                        value=None,
                        reason="saddle_distance_unavailable",
                    )
                )
            else:
                contributing_rows.append(
                    _gate_row(
                        machine,
                        identity,
                        names,
                        value=saddle,
                        reason="finite_saddle_distance",
                    )
                )
    values = [row["value"] for row in contributing_rows]
    worst = max(values) if values else None
    return {
        "measured": {
            "worst_saddle_distance_m": worst,
            "declared_denominator": len(EXPECTED_MAST_ARMS) + len(EXPECTED_DIIID_ROWS),
        },
        "threshold": {"maximum_inclusive_m": SADDLE_DISTANCE_MAXIMUM_M},
        "contributing_rows": contributing_rows,
        "excluded_rows": excluded_rows,
        "passes": bool(values)
        and not excluded_rows
        and worst <= SADDLE_DISTANCE_MAXIMUM_M,
    }


def score_payloads(mast: dict[str, Any], diiid: dict[str, Any]) -> dict[str, Any]:
    """Return the three fixed-cohort gate results or refuse an incomplete bank."""

    mast_score = _mast_score(_mast_rows(mast))
    diiid_score = _diiid_score(_diiid_rows(diiid))
    gates = {
        "mast_class_agreement": _class_agreement_gate(mast_score),
        "closed_boundary_symmetric_rms": _rms_gate(mast_score, diiid_score),
        "declared_row_saddle_distance": _saddle_gate(mast_score, diiid_score),
    }
    return {
        "mast": mast_score,
        "diiid": diiid_score,
        "gates": gates,
        "verdict": "PASS" if all(gate["passes"] for gate in gates.values()) else "FAIL",
    }


def _read_bank(path: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise ScoreRefusal(f"cannot read strict JSON bank {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ScoreRefusal(f"bank {path} is not a JSON object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _input_receipt(
    path: Path,
    digest: str,
    identities: tuple[tuple[Any, ...], ...],
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": digest,
        "declared_identities": [
            _identity_record(identity, names) for identity in identities
        ],
    }


def run(mast_path: Path, diiid_path: Path, output: Path) -> dict[str, Any]:
    """Write a strict endpoint receipt, retaining any fail-closed refusal."""

    mast_payload, mast_digest = _read_bank(mast_path)
    diiid_payload, diiid_digest = _read_bank(diiid_path)
    receipt: dict[str, Any] = {
        "artifact": "fixed-cohort EFIT baseline demonstration score",
        "contract": "banked-inclusive-three-gate-contract",
        "inputs": {
            "mast": _input_receipt(
                mast_path,
                mast_digest,
                EXPECTED_MAST_ARMS,
                ("shot", "slice_index", "arm"),
            ),
            "diiid": _input_receipt(
                diiid_path, diiid_digest, EXPECTED_DIIID_ROWS, ("shot", "frame")
            ),
        },
        "thresholds": {
            "mast_class_agreement_minimum_inclusive": MAST_CLASS_AGREEMENT_MINIMUM,
            "mast_class_agreement_denominator": len(EXPECTED_MAST_ARMS),
            "closed_boundary_symmetric_rms_maximum_inclusive_m": (
                CLOSED_BOUNDARY_RMS_MAXIMUM_M
            ),
            "declared_row_saddle_distance_maximum_inclusive_m": (
                SADDLE_DISTANCE_MAXIMUM_M
            ),
            "declared_row_saddle_distance_denominator": len(EXPECTED_MAST_ARMS)
            + len(EXPECTED_DIIID_ROWS),
        },
    }
    try:
        receipt.update(score_payloads(mast_payload, diiid_payload))
    except ScoreRefusal as error:
        receipt.update(
            {
                "refusal": {"type": type(error).__name__, "reason": str(error)},
                "verdict": "FAIL",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast-bank", type=Path, default=MAST_BANK)
    parser.add_argument("--diiid-bank", type=Path, default=DIIID_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.mast_bank, arguments.diiid_bank, arguments.output)
    print(json.dumps({"verdict": receipt["verdict"], "output": str(arguments.output)}))
    if receipt["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
