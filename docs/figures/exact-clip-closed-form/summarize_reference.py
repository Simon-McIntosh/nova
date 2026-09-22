"""Combine measurements, source hashes and independent numerical errors."""

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "bernstein-reference-audit"
BASELINE_TEST_OUTPUT = Path(__file__).resolve().parent / "bernstein-reference"
SOURCE = "067ea754ce138a8b93a704f42877ab501957c9ad"
BASE = "b3abce70bd92fa143dbfbfd9d449f7de06de7dc2"
CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-single-null",
)
TESTS = ("test_interpolant", "test_exact_clip_moments", "test_flux_surface_extraction")


def main():
    programs = {
        arm: json.loads((OUTPUT / f"{arm}-program.json").read_text())
        for arm in ("baseline", "candidate")
    }
    reference = json.loads((OUTPUT / "reference-summary.json").read_text())
    rows = [
        json.loads((OUTPUT / f"{case}-{cells}.json").read_text())
        for case in CASES
        for cells in (110, 300)
    ]
    suites = {}
    for arm in ("baseline", "candidate"):
        suites[arm] = []
        for test in TESTS:
            test_output = BASELINE_TEST_OUTPUT if arm == "baseline" else OUTPUT
            log = test_output / f"{arm}-{test}.log"
            text = log.read_text()
            passed = re.findall(r"(\d+) passed", text)
            assert passed, f"passing-test positive control absent: {log}"
            suites[arm].append(
                {
                    "module": test + ".py",
                    "passed": int(passed[-1]),
                    "exit_status": int(
                        (test_output / f"{arm}-{test}.exit").read_text()
                    ),
                    "failure_ids": re.findall(r"^FAILED ([^\n]+)", text, re.MULTILINE),
                    "log": str(log),
                }
            )
    jobs = (
        {p["job"] for p in programs.values()}
        | {r["job"] for r in rows}
        | {reference["job"]}
    )
    assert len(jobs) == 1
    for path in (
        "nova/linalg/interpolant.py",
        "nova/equilibrium/flux_surface_extraction.py",
        "tests/test_interpolant.py",
        "tests/test_flux_surface_extraction.py",
        "tests/test_exact_clip_moments.py",
    ):
        assert (
            subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)
            == (ROOT / path).read_bytes()
        )
    assert all(r["source_sha256"] == reference["source_sha256"] for r in rows)
    assert all(p["completed"] for p in programs.values())
    assert all(
        r["previous_route_equivalence_is_acceptance_gate"] is False for r in rows
    )
    baseline, candidate = programs["baseline"], programs["candidate"]
    assert baseline["bernstein_binom_present"]
    assert (
        not candidate["bernstein_binom_present"] and candidate["static_basis_present"]
    )
    assert (
        sum(count for _, count in candidate["instruction_ownership"])
        + candidate["unowned_instructions"]
        == candidate["hlo_instructions"]
    )
    result = {
        "measurements_completed": True,
        "source_checkpoint": SOURCE,
        "base_revision": BASE,
        "job": jobs.pop(),
        "programs": programs,
        "suites": suites,
        "analytic_rows": rows,
        "independent_reference": reference,
        "numerical_no_worse_passed": reference["passed"],
        "baseline_suite_context_job": "1275884",
        "zero_added_module_failures": all(
            x["exit_status"] == 0 and not x["failure_ids"]
            for records in suites.values()
            for x in records
        ),
        "serialized_byte_reduction_fraction": 1
        - candidate["serialized_bytes"] / baseline["serialized_bytes"],
        "coordinator_ceiling": {
            "bytes": 100000000,
            "measured_candidate_bytes": candidate["serialized_bytes"],
            "under_ceiling": candidate["serialized_bytes"] < 100000000,
            "verdict_owner": "coordinator",
        },
        "numerical_follow_on": (
            "Strict no-worse inequalities remain failed at roundoff scale; "
            "coefficient controls did not remove them. Shared root and moment "
            "routines require separate source scope."
        ),
        "negative_control_receipt": str(
            OUTPUT.parent / "bernstein/negative-program.json"
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "measurements_completed",
                    "numerical_no_worse_passed",
                    "zero_added_module_failures",
                    "coordinator_ceiling",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["numerical_no_worse_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
