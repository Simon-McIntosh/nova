#!/usr/bin/env python3
"""Aggregate forward-labeller shot manifests into a completion receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


QUALIFICATION_FIELD = "ForwardSolveReceipt.qualified"
QUALIFICATION_COMPUTATION = (
    "result.converged and equilibrium.finite.passed and "
    "terminal_residual <= policy.qualification_tolerance"
)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finite_value(value: Any) -> bool:
    """Return whether an optional scalar is finite."""
    return value is not None and math.isfinite(float(value))


def _unqualified_reason(row: dict[str, Any], tolerance: float) -> str:
    """Return one mutually exclusive explanation for an unqualified slice."""
    if row.get("exception"):
        return "slice_exception"
    if not row.get("converged"):
        return "not_converged"
    residual = float(row.get("terminal_residual", math.nan))
    if not math.isfinite(residual):
        return "non_finite_terminal_residual"
    if residual > tolerance:
        return "residual_above_qualification_tolerance"
    return "receipt_finite_check_failed"


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return the requested robust summary of a finite sample."""
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if not finite.size:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
    }


def _shot_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """Return complete per-shot quality, branch and throughput evidence."""
    rows = [row for row in item.get("slices", ()) if row.get("written")]
    excluded = [row for row in item.get("slices", ()) if row.get("excluded")]
    tolerance = float(item.get("policy", {}).get("qualification_tolerance", math.nan))
    reasons: dict[str, int] = {}
    for row in rows:
        if row.get("qualified"):
            continue
        reason = _unqualified_reason(row, tolerance)
        reasons[reason] = reasons.get(reason, 0) + 1
    differences = [
        1000.0
        * (
            float(row.get("achieved_current_centroid_z", math.nan))
            - float(row.get("target_current_centroid_z", math.nan))
        )
        for row in rows
        if row.get("achieved_current_centroid_z") is not None
        and row.get("target_current_centroid_z") is not None
    ]
    slice_wall_seconds = sum(float(row.get("wall_seconds", 0.0)) for row in rows)
    return {
        "shot": int(item["shot"]),
        "admitted_slices": len(rows),
        "excluded_slices": len(excluded),
        "converged_slices": sum(bool(row.get("converged")) for row in rows),
        "unconverged_slices": sum(not bool(row.get("converged")) for row in rows),
        "qualified_slices": sum(bool(row.get("qualified")) for row in rows),
        "unqualified_slices": sum(not bool(row.get("qualified")) for row in rows),
        "qualification": {
            "field": QUALIFICATION_FIELD,
            "computation": QUALIFICATION_COMPUTATION,
            "tolerance": tolerance,
        },
        "unqualified_by_reason": dict(sorted(reasons.items())),
        "branch_guard_agreement_slices": sum(
            bool(row.get("branch_guard_ok")) for row in rows
        ),
        "vertical_centroid_difference_mm": {
            "definition": (
                "1000 * (achieved_current_centroid_z - efm/current_centrd_z)"
            ),
            "signed": _distribution(differences),
            "absolute": _distribution([abs(value) for value in differences]),
        },
        "slice_wall_seconds": slice_wall_seconds,
        "mean_slice_wall_seconds": _fraction(slice_wall_seconds, len(rows)),
        "slices_per_second_per_card": _fraction(len(rows), slice_wall_seconds),
        "warm_rate_eligible": float(item.get("setup_wall_seconds", 0.0)) == 0.0,
    }


def aggregate(root: Path, *, expected_shots: int | None = None) -> dict[str, Any]:
    """Return completion and throughput metrics for one sessions directory."""
    manifests = []
    for path in sorted(root.glob("*.manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path.resolve())
        manifests.append(record)
    complete = [item for item in manifests if item.get("status") == "complete"]
    failed = [item for item in manifests if item.get("status") != "complete"]
    slices = [
        row for item in complete for row in item.get("slices", ()) if row.get("written")
    ]
    excluded = [
        row
        for item in complete
        for row in item.get("slices", ())
        if row.get("excluded")
    ]
    converged = sum(bool(row.get("converged")) for row in slices)
    unconverged = len(slices) - converged
    qualified = sum(bool(row.get("qualified")) for row in slices)
    guarded = [
        row for row in slices if row.get("target_source") == "efm/current_centrd_z"
    ]
    guard_agreement = sum(bool(row.get("branch_guard_ok")) for row in guarded)
    centroid_slices = sum(
        _finite_value(row.get("achieved_current_centroid_r"))
        and _finite_value(row.get("achieved_current_centroid_z"))
        for row in slices
    )
    companion_records = [
        item
        for item in complete
        if item.get("companion") and Path(item["companion"]).is_file()
    ]
    companion_slices = sum(
        int(item.get("companion_slice_count", 0)) for item in companion_records
    )
    shot_metrics = [_shot_metrics(item) for item in complete]
    warm_shots = [item for item in shot_metrics if item["warm_rate_eligible"]]
    warm_slices = sum(int(item["admitted_slices"]) for item in warm_shots)
    warm_seconds = sum(float(item["slice_wall_seconds"]) for item in warm_shots)
    gpu_seconds = sum(
        float(item.get("setup_wall_seconds", 0.0))
        + float(item.get("shot_wall_seconds", 0.0))
        for item in complete
    )
    expected = len(manifests) if expected_shots is None else int(expected_shots)
    return {
        "schema": "nova-forward-labeller-completion",
        "root": str(root.resolve()),
        "expected_shots": expected,
        "manifest_count": len(manifests),
        "completed_shots": len(complete),
        "failed_shots": len(failed),
        "completion_fraction": _fraction(len(complete), expected),
        "slices": len(slices),
        "admitted_slices": len(slices),
        "excluded_slices": len(excluded),
        "converged_slices": converged,
        "unconverged_slices": unconverged,
        "qualified_slices": qualified,
        "converged_fraction": _fraction(converged, len(slices)),
        "qualified_fraction": _fraction(qualified, len(slices)),
        "branch_guard_evaluated_slices": len(guarded),
        "branch_guard_agreement_slices": guard_agreement,
        "branch_guard_agreement_fraction": _fraction(guard_agreement, len(guarded)),
        "current_centroid_slices": centroid_slices,
        "companion_shots": len(companion_records),
        "companion_slices": companion_slices,
        "companion_completion_fraction": _fraction(
            len(companion_records), len(complete)
        ),
        "flux_function_grid_points": sorted(
            {
                int(item["flux_function_grid_points"])
                for item in companion_records
                if item.get("flux_function_grid_points") is not None
            }
        ),
        "slices_per_second_per_card": _fraction(len(slices), gpu_seconds),
        "warm_rate_definition": (
            "admitted slices in shots after the setup and compilation-bearing "
            "first shot of each shard, divided by their summed per-slice wall time"
        ),
        "warm_shots": len(warm_shots),
        "warm_slices": warm_slices,
        "warm_wall_seconds": warm_seconds,
        "warm_slices_per_second_per_card": _fraction(warm_slices, warm_seconds),
        "gpu_hours": gpu_seconds / 3600.0,
        "nova_revisions": sorted({str(item.get("nova_revision")) for item in complete}),
        "carrier_identities": sorted(
            {str(item.get("carrier_identity")) for item in complete}
        ),
        "policy_digests": sorted({str(item.get("policy_digest")) for item in complete}),
        "shots": shot_metrics,
        "failed": [
            {
                "shot": item.get("shot"),
                "failure": item.get("failure"),
                "traceback": item.get("traceback"),
                "manifest": item["_path"],
            }
            for item in failed
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shots", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = aggregate(arguments.root, expected_shots=arguments.expected_shots)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["failed_shots"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
