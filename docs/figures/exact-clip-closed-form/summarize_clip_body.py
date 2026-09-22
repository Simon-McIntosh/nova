"""Read completed clip-body receipts without treating missing rows as passes."""

from __future__ import annotations
import json
from pathlib import Path
import re
import measure_clip_body as measure

ROOT, OUTPUT = measure.ROOT, measure.OUTPUT


def suite(arm):
    modules = {}
    for filename in measure.TESTS:
        name = Path(filename).stem
        log = OUTPUT / (arm + "-" + name + ".log")
        if arm == "baseline" and name == "test_equilibrium_separatrix_clip":
            log = OUTPUT / "baseline-refusal-corrected.log"
        text = log.read_text()
        matched = re.findall(r"(\d+) passed(?:, .*?)? in [\d.]+s", text)
        failures = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
        assert matched or failures, f"no pytest completion summary in {log}"
        modules[filename] = dict(
            passed=int(matched[-1]) if matched else 0,
            failure_ids=failures,
            log=str(log),
        )
    failures = sorted({name for m in modules.values() for name in m["failure_ids"]})
    return dict(
        modules=modules,
        passed=sum(m["passed"] for m in modules.values()),
        failure_ids=failures,
        failure_count=len(failures),
        completed=True,
        exit_status=int(bool(failures)),
    )


def main():
    programs = {
        arm: json.loads((OUTPUT / (arm + "-program.json")).read_text())
        for arm in ("baseline", "candidate", "negative")
    }
    assert all(p["completed"] and p["serialized_bytes"] > 0 for p in programs.values())
    assert all(dict(p["top_five_hlo_ops"])["parameter"] > 0 for p in programs.values())
    cases = (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
        "strong-rotation-compact-static",
        "diverted-single-null",
    )
    rows = [
        json.loads((OUTPUT / f"{case}-{cells}.json").read_text())
        for case in cases
        for cells in (110, 300)
    ]
    suites = {arm: suite(arm) for arm in ("baseline", "candidate")}
    jobs = {p["job"] for p in programs.values()} | {r["job"] for r in rows}
    assert len(jobs) == 1
    added = sorted(
        set(suites["candidate"]["failure_ids"]) - set(suites["baseline"]["failure_ids"])
    )
    gates = dict(
        eight_rows_agree=all(r["passed"] for r in rows),
        zero_added_failures=not added,
        candidate_below_ceiling=programs["candidate"]["serialized_bytes"] < 100_000_000,
        negative_bytes_rise=programs["negative"]["serialized_bytes"]
        > programs["candidate"]["serialized_bytes"],
        cell_body_marker=programs["candidate"].get("shared_cell_body_present", False),
    )
    result = dict(
        programs=programs,
        rows=rows,
        suites=suites,
        jobs=sorted(jobs),
        added_failure_ids=added,
        gates=gates,
        measurements_completed=True,
        passed=all(gates.values()),
        candidate_source="23a8522b6b0ab27216c46db7d150e2b96433f347",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                gates=gates,
                test_counts={a: v["passed"] for a, v in suites.items()},
                bytes={a: v["serialized_bytes"] for a, v in programs.items()},
            ),
            indent=2,
        )
    )
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
