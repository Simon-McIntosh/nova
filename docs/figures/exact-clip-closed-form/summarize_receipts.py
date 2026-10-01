"""Assemble completed CPU receipts without hiding missing or refused rows."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-single-null",
)
MOMENTS = (
    "current",
    "radial",
    "vertical",
    "radial_squared",
    "radial_vertical",
    "vertical_squared",
)


def main():
    programs = {}
    for arm in ("baseline", "candidate", "negative"):
        path = ROOT / f"{arm}-program.json"
        programs[arm] = json.loads(path.read_text()) if path.exists() else None
    rows = []
    missing = []
    for case in CASES:
        for cells in (110, 300):
            path = ROOT / f"{case}-{cells}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
            else:
                missing.append(path.name)
    summary = {"programs": programs, "rows": rows, "missing_rows": missing}
    baseline = programs["baseline"] or {}
    candidate = programs["candidate"] or {}
    negative = programs["negative"] or {}
    if all(p.get("completed") for p in (baseline, candidate, negative)):
        summary["byte_reduction_fraction"] = (
            1 - candidate["serialized_bytes"] / baseline["serialized_bytes"]
        )
        summary["negative_minus_baseline_bytes"] = (
            negative["serialized_bytes"] - baseline["serialized_bytes"]
        )
        summary["negative_top_five_match_baseline"] = (
            negative["top_five_hlo_ops"] == baseline["top_five_hlo_ops"]
        )
    summary["complete_row_census"] = not missing
    negative_log = ROOT / "negative-program.log"
    negative_text = negative_log.read_text() if negative_log.exists() else ""
    refusal_tests = (
        "test_arc_route_carries_the_fixed_edge_and_point_bound",
        "test_production_boundary_reduction_has_no_gauss_evaluations",
        "test_production_boundary_lowering_uses_endpoint_recurrence",
    )
    failure_lines = [
        line for line in negative_text.splitlines() if line.startswith("FAILED ")
    ]
    summary["negative_control_guards_fired"] = bool(
        "restored_gauss_test_exit=1" in negative_text
        and len(failure_lines) == len(refusal_tests)
        and all(any(name in line for line in failure_lines) for name in refusal_tests)
    )
    summary["full_gate_passed"] = bool(
        not missing
        and len(rows) == 8
        and all(row["passed"] for row in rows)
        and candidate.get("completed")
        and candidate.get("below_byte_ceiling")
        and candidate.get("hlo_bernstein_binom_occurrences") == 0
        and negative.get("completed")
        and negative["serialized_bytes"] == baseline.get("serialized_bytes")
        and summary["negative_control_guards_fired"]
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    parts = [
        "<h3>Measured executable and row comparison</h3>",
        "<table><thead><tr><th>Arm</th><th>Serialized bytes</th>"
        "<th>Compile seconds</th><th>Optimized instructions</th>"
        "<th>Top five HLO operations</th></tr></thead><tbody>",
    ]
    for arm, result in programs.items():
        if result is None or not result.get("completed"):
            parts.append(f'<tr><td>{arm}</td><td colspan="4">not completed</td></tr>')
            continue
        operations = "; ".join(
            f"{name}: {count:,}" for name, count in result["top_five_hlo_ops"]
        )
        parts.append(
            f"<tr><td>{arm}</td><td>{result['serialized_bytes']:,}</td>"
            f"<td>{result['compile_seconds']:.3f}</td>"
            f"<td>{result['hlo_instructions']:,}</td><td>{operations}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        "<p>Moment columns are relative L₂ differences against the four-node Gauss "
        "reference. Image differences are supremum differences divided by flux span. "
        "The moment fence is 1e-12; the image floor is 1.52e-15. A refused cell "
        "remains a refusal even when admitted cells agree.</p>"
    )
    parts.append(
        "<table><thead><tr><th>Case / requested cells</th><th>Order 0: current</th>"
        "<th>Order 1: radial</th><th>Order 1: vertical</th>"
        "<th>Order 2: radial squared</th><th>Order 2: radial vertical</th>"
        "<th>Order 2: vertical squared</th><th>Image / span</th>"
        "<th>Refused cells</th><th>Pass</th></tr></thead><tbody>"
    )
    for result in rows:
        parts.append(
            f"<tr><td>{html.escape(result['case'])} / {result['requested_cells']}</td>"
            + "".join(
                f"<td>{result['moment_relative_l2'][name]:.3e}</td>" for name in MOMENTS
            )
            + f"<td>{result['frozen_image_sup_over_span']:.3e}</td>"
            + f"<td>{result['refused_cell_count']}</td><td>{result['passed']}</td></tr>"
        )
    for name in missing:
        parts.append(
            f'<tr><td>{html.escape(name)}</td><td colspan="9">'
            "unmeasured; see row log</td></tr>"
        )
    parts.append("</tbody></table>")
    if "negative_minus_baseline_bytes" in summary:
        parts.append(
            "<p>Restoring the Gauss path changes the byte count by "
            f"{summary['negative_minus_baseline_bytes']:+,} bytes against baseline; "
            "the top-five operation counts match: "
            f"{summary['negative_top_five_match_baseline']}. "
            "The byte reduction of the candidate is "
            f"{summary['byte_reduction_fraction']:.2%}.</p>"
        )
    parts.append(
        f"<p><strong>Full node gate passed: {summary['full_gate_passed']}.</strong> "
        "The program-size target is 100,000,000 bytes. Full acceptance also requires "
        "removing Bernstein binomial lowering and completing every requested row "
        "and the declared negative control.</p>"
    )
    (ROOT / "measurement-tables.html").write_text("\n".join(parts) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("programs", "rows")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
