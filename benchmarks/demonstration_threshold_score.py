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
MAST_CLOSED_BOUNDARY_RMS_MAXIMUM_M = 0.55
SADDLE_DISTANCE_MAXIMUM_M = 0.04


class ScoreRefusal(ValueError):
    """Raised when a bank cannot enter the fixed-cohort demonstration score."""


def _identity_record(
    identity: tuple[Any, ...], names: tuple[str, ...]
) -> dict[str, Any]:
    return dict(zip(names, identity, strict=True))


def _strict_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScoreRefusal(f"{description} is missing or not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScoreRefusal(f"{description} is non-finite")
    return result


def _required_boolean(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        raise ScoreRefusal(f"{description} must be an explicit Boolean")
    return value


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
        raise ScoreRefusal(
            f"{cohort} cohort mismatch: {mismatch}"
        )


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
    agreement_count = 0
    rms_values: list[tuple[tuple[Any, ...], float]] = []
    saddle_values: list[tuple[tuple[Any, ...], float]] = []
    row_receipts = []
    for identity, row in zip(EXPECTED_MAST_ARMS, rows, strict=True):
        declared_identity = f"{identity[0]}/{identity[1]}"
        if row.get("identity") != declared_identity:
            raise ScoreRefusal(
                f"MAST row {identity!r} substituted its declared identity field"
            )
        converged = _required_boolean(
            row.get("converged"), f"MAST row {identity!r} convergence"
        )
        reference_class = row.get("efit_label")
        achieved_class = row.get("nova_achieved_class")
        if reference_class not in {"limited", "diverted"} or achieved_class not in {
            "limited",
            "diverted",
        }:
            raise ScoreRefusal(f"MAST row {identity!r} has an empty topology class")
        agrees = achieved_class == reference_class
        if "label_agreement" in row and row["label_agreement"] is not agrees:
            raise ScoreRefusal(
                f"MAST row {identity!r} carries an inconsistent agreement flag"
            )
        failures = row.get("comparison_failures")
        if not isinstance(failures, list) or failures:
            raise ScoreRefusal(
                f"MAST row {identity!r} has incomplete comparison operands"
            )
        saddle = _strict_number(
            row.get("selected_saddle_to_efit_x_point_m"),
            f"MAST row {identity!r} saddle distance",
        )
        saddle_values.append((identity, saddle))
        rms = None
        if agrees:
            if not converged:
                raise ScoreRefusal(
                    f"MAST class-agreeing row {identity!r} is non-converged and cannot "
                    "be removed from the RMS gate"
                )
            rms = _strict_number(
                row.get("binding_to_efit_lcfs_rms_m"),
                f"MAST row {identity!r} closed-boundary RMS",
            )
            rms_values.append((identity, rms))
            agreement_count += 1
        row_receipts.append(
            {
                "identity": _identity_record(identity, ("shot", "slice_index", "arm")),
                "converged": converged,
                "class_agreement": agrees,
                "closed_boundary_symmetric_rms_m": rms,
                "saddle_distance_m": saddle,
            }
        )
    if not rms_values:
        raise ScoreRefusal("MAST eligible closed-boundary RMS cohort is empty")
    worst_rms_identity, worst_rms = max(rms_values, key=lambda item: item[1])
    worst_saddle_identity, worst_saddle = max(saddle_values, key=lambda item: item[1])
    return {
        "agreement_count": agreement_count,
        "agreement_denominator": len(EXPECTED_MAST_ARMS),
        "worst_closed_boundary_symmetric_rms_m": worst_rms,
        "worst_closed_boundary_symmetric_rms_identity": _identity_record(
            worst_rms_identity, ("shot", "slice_index", "arm")
        ),
        "worst_saddle_distance_m": worst_saddle,
        "worst_saddle_identity": _identity_record(
            worst_saddle_identity, ("shot", "slice_index", "arm")
        ),
        "rows": row_receipts,
    }


def _diiid_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    saddle_values: list[tuple[tuple[Any, ...], float]] = []
    row_receipts = []
    for identity, row in zip(EXPECTED_DIIID_ROWS, rows, strict=True):
        if not _required_boolean(
            row.get("finite"), f"DIII-D row {identity!r} finite receipt"
        ):
            raise ScoreRefusal(f"DIII-D row {identity!r} is non-finite")
        if not _required_boolean(
            row.get("converged"), f"DIII-D row {identity!r} convergence"
        ):
            raise ScoreRefusal(
                f"DIII-D declared row {identity!r} is non-converged and cannot be "
                "removed from the saddle gate"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ScoreRefusal(f"DIII-D row {identity!r} metrics are empty")
        saddle = _strict_number(
            metrics.get("polished_saddle_to_nearest_efit_x_m"),
            f"DIII-D row {identity!r} saddle distance",
        )
        saddle_values.append((identity, saddle))
        row_receipts.append(
            {
                "identity": _identity_record(identity, ("shot", "frame")),
                "converged": True,
                "saddle_distance_m": saddle,
            }
        )
    worst_identity, worst_saddle = max(saddle_values, key=lambda item: item[1])
    return {
        "declared_row_count": len(rows),
        "worst_saddle_distance_m": worst_saddle,
        "worst_saddle_identity": _identity_record(worst_identity, ("shot", "frame")),
        "rows": row_receipts,
    }


def score_payloads(mast: dict[str, Any], diiid: dict[str, Any]) -> dict[str, Any]:
    """Return the three fixed-cohort gate results or refuse an incomplete bank."""

    mast_score = _mast_score(_mast_rows(mast))
    diiid_score = _diiid_score(_diiid_rows(diiid))
    if mast_score["worst_saddle_distance_m"] >= diiid_score["worst_saddle_distance_m"]:
        worst_saddle = mast_score["worst_saddle_distance_m"]
        worst_saddle_identity = {
            "machine": "MAST",
            **mast_score["worst_saddle_identity"],
        }
    else:
        worst_saddle = diiid_score["worst_saddle_distance_m"]
        worst_saddle_identity = {
            "machine": "DIII-D",
            **diiid_score["worst_saddle_identity"],
        }
    gates = {
        "mast_class_agreement": {
            "value": mast_score["agreement_count"],
            "denominator": len(EXPECTED_MAST_ARMS),
            "minimum_inclusive": MAST_CLASS_AGREEMENT_MINIMUM,
            "passes": mast_score["agreement_count"] >= MAST_CLASS_AGREEMENT_MINIMUM,
        },
        "mast_closed_boundary_symmetric_rms": {
            "value_m": mast_score["worst_closed_boundary_symmetric_rms_m"],
            "maximum_inclusive_m": MAST_CLOSED_BOUNDARY_RMS_MAXIMUM_M,
            "worst_identity": mast_score[
                "worst_closed_boundary_symmetric_rms_identity"
            ],
            "passes": mast_score["worst_closed_boundary_symmetric_rms_m"]
            <= MAST_CLOSED_BOUNDARY_RMS_MAXIMUM_M,
        },
        "declared_row_saddle_distance": {
            "value_m": worst_saddle,
            "maximum_inclusive_m": SADDLE_DISTANCE_MAXIMUM_M,
            "declared_row_denominator": len(EXPECTED_MAST_ARMS)
            + len(EXPECTED_DIIID_ROWS),
            "worst_identity": worst_saddle_identity,
            "passes": worst_saddle <= SADDLE_DISTANCE_MAXIMUM_M,
        },
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
            "mast_closed_boundary_symmetric_rms_maximum_inclusive_m": (
                MAST_CLOSED_BOUNDARY_RMS_MAXIMUM_M
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
