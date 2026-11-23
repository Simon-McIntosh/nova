"""Read recovery-ladder activity from persisted certificate parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASE_NAMES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-single-null",
)


def _summary(telemetry):
    promotions = telemetry.get("promotion_globalisation", [])
    rebuild = [p for p in promotions if p.get("model_rebuild_activated")]
    descent = [p for p in promotions if p.get("steepest_descent_activated")]
    continuation = [p for p in promotions if p.get("continuation_activated")]
    backtrack = sum(int(p.get("backtrack_count") or 0) for p in promotions)
    return {
        "trips": telemetry.get("trip_count"),
        "promotions_attempted": telemetry.get("attempted_newton_promotions"),
        "promotions_accepted": telemetry.get("accepted_newton_promotions"),
        "promotions_recorded": len(promotions),
        "rebuild_rung_promotions": len(rebuild),
        "descent_rung_promotions": len(descent),
        "continuation_promotions": len(continuation),
        "backtrack_trials": backtrack,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--cells", type=int, default=-300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.parts)
    rows = []
    for case in CASE_NAMES:
        name = "%s-production-route-cells-%d.json" % (case, abs(args.cells))
        path = root / name
        if not path.exists():
            rows.append({"case": case, "status": "missing"})
            continue
        row = json.loads(path.read_text())
        solver = row.get("solver", {})
        telemetry = solver.get("production_telemetry") or {}
        entry = {
            "case": case,
            "status": "recorded",
            "converged": solver.get("converged"),
            "termination": solver.get("termination"),
            "terminal_fixed_point_residual": solver.get(
                "terminal_fixed_point_residual"
            ),
        }
        entry.update(_summary(telemetry))
        rows.append(entry)
    payload = {"requested_cells": args.cells, "rows": rows}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "telemetry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# Recovery-ladder activity in the persisted certificate parts", ""]
    lines.append(
        "| case | trips | promotions | rebuild | descent | backtrack | converged |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["status"] != "recorded":
            lines.append("| `%s` | missing | | | | | |" % row["case"])
            continue
        lines.append(
            "| `%s` | %s | %s/%s | %s | %s | %s | %s |"
            % (
                row["case"],
                row["trips"],
                row["promotions_recorded"],
                row["promotions_attempted"],
                row["rebuild_rung_promotions"],
                row["descent_rung_promotions"],
                row["backtrack_trials"],
                row["converged"],
            )
        )
    lines.append("")
    (out / "telemetry.md").write_text("\n".join(lines) + "\n")
    for row in rows:
        print("TELEMETRY %s %s" % (row["case"], json.dumps(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
