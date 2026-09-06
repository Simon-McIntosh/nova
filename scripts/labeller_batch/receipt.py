#!/usr/bin/env python3
"""Aggregate forward-labeller shot manifests into a completion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


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
    converged = sum(bool(row.get("converged")) for row in slices)
    qualified = sum(bool(row.get("qualified")) for row in slices)
    guarded = [
        row for row in slices if row.get("target_source") == "efm/current_centrd_z"
    ]
    guard_agreement = sum(bool(row.get("branch_guard_ok")) for row in guarded)
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
        "converged_slices": converged,
        "qualified_slices": qualified,
        "converged_fraction": _fraction(converged, len(slices)),
        "qualified_fraction": _fraction(qualified, len(slices)),
        "branch_guard_evaluated_slices": len(guarded),
        "branch_guard_agreement_slices": guard_agreement,
        "branch_guard_agreement_fraction": _fraction(guard_agreement, len(guarded)),
        "slices_per_second_per_card": _fraction(len(slices), gpu_seconds),
        "gpu_hours": gpu_seconds / 3600.0,
        "nova_revisions": sorted({str(item.get("nova_revision")) for item in complete}),
        "carrier_identities": sorted(
            {str(item.get("carrier_identity")) for item in complete}
        ),
        "policy_digests": sorted({str(item.get("policy_digest")) for item in complete}),
        "failed": [
            {
                "shot": item.get("shot"),
                "failure": item.get("failure"),
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
