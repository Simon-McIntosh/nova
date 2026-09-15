"""Gate whole-solve executable size and traced operator-path replication.

The input receipts are emitted by :mod:`benchmarks.program_scope_census`.
The gate deliberately checks that the map control contains one copy of each
sentinel before accepting the solve's absence or singleton count.  This keeps
an empty or stale instrument from reporting a structural improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CELLS = (300, 1000)
MAX_EXECUTABLE_BYTES = 50 * (1 << 20)
REQUIRED_OPERATOR_COPIES = 1
REPLICATION_PATHS = ("current-moment path", "topology read")


def _load_rungs(directory: Path) -> dict[int, dict[str, Any]]:
    """Load complete census receipts for the required cell counts."""
    rows: dict[int, dict[str, Any]] = {}
    for cells in REQUIRED_CELLS:
        path = directory / f"{cells}.json"
        if not path.exists():
            raise ValueError(f"missing census receipt {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("requested_cells", -1)) != cells:
            raise ValueError(f"receipt {path} does not describe {cells} cells")
        rows[cells] = row
    return rows


def _copy_count(program: dict[str, Any], path: str) -> int:
    replication = program.get("replication", {})
    if path not in replication:
        raise ValueError(f"missing replication sentinel {path!r}")
    return int(replication[path]["copy_count"])


def _effective_executable_bytes(program: dict[str, Any]) -> tuple[int, str]:
    """Prefer serialized bytes and retain generated code as a conservative fallback."""
    executable = program.get("executable")
    if not isinstance(executable, dict):
        raise ValueError("missing executable-size measurement")
    serialized = executable.get("serialized_bytes")
    if serialized is not None:
        return int(serialized), "serialized executable"
    generated = executable.get("generated_code_bytes")
    if generated is not None:
        return int(generated), "generated code fallback"
    error = executable.get("serialization_error")
    suffix = f": {error}" if error else ""
    raise ValueError(f"executable size is absent{suffix}")


def evaluate_gate(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compare before and after receipts and return a quantitative verdict."""
    rows = []
    failures: list[str] = []
    for cells in REQUIRED_CELLS:
        before = baseline[cells]
        after = candidate[cells]
        map_counts = {
            path: _copy_count(after["map"], path) for path in REPLICATION_PATHS
        }
        for path, count in map_counts.items():
            if count != REQUIRED_OPERATOR_COPIES:
                failures.append(
                    f"{cells}-cell map sentinel {path!r} saw {count}, expected one"
                )
        solve_counts = {
            path: _copy_count(after["solve"], path) for path in REPLICATION_PATHS
        }
        for path, count in solve_counts.items():
            if count != REQUIRED_OPERATOR_COPIES:
                failures.append(
                    f"{cells}-cell solve {path!r} has {count} traced copies"
                )
        executable_bytes, executable_measure = _effective_executable_bytes(
            after["solve"]
        )
        if cells == 1000 and executable_bytes > MAX_EXECUTABLE_BYTES:
            failures.append(
                f"1000-cell solve executable is {executable_bytes} bytes, "
                f"limit {MAX_EXECUTABLE_BYTES}"
            )
        rows.append(
            {
                "requested_cells": cells,
                "before": {
                    "solve_instructions": int(before["solve"]["total_instructions"]),
                    "map_instructions": int(before["map"]["total_instructions"]),
                    "operator_copies": {
                        path: _copy_count(before["solve"], path)
                        for path in REPLICATION_PATHS
                    },
                },
                "after": {
                    "solve_instructions": int(after["solve"]["total_instructions"]),
                    "map_instructions": int(after["map"]["total_instructions"]),
                    "instruction_ratio": (
                        float(after["solve"]["total_instructions"])
                        / float(after["map"]["total_instructions"])
                    ),
                    "operator_copies": solve_counts,
                    "map_operator_copies": map_counts,
                    "executable_bytes": executable_bytes,
                    "executable_measure": executable_measure,
                },
            }
        )
    return {"passed": not failures, "failures": failures, "rows": rows}


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Solve program size gate",
        "",
        "| cells | before solve/map instructions | after solve/map instructions | "
        "after ratio | executable bytes | moment copies | topology copies |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        before = row["before"]
        after = row["after"]
        lines.append(
            f"| {row['requested_cells']} | "
            f"{before['solve_instructions']:,}/{before['map_instructions']:,} | "
            f"{after['solve_instructions']:,}/{after['map_instructions']:,} | "
            f"{after['instruction_ratio']:.2f} | {after['executable_bytes']:,} | "
            f"{after['operator_copies']['current-moment path']} | "
            f"{after['operator_copies']['topology read']} |"
        )
    lines.extend(["", f"Verdict: **{'PASS' if result['passed'] else 'FAIL'}**."])
    if result["failures"]:
        lines.extend(["", "Refusals:"])
        lines.extend(f"- {failure}" for failure in result["failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gate(
        _load_rungs(args.baseline_dir), _load_rungs(args.candidate_dir)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    print(f"SOLVE_PROGRAM_SIZE_GATE={'PASS' if result['passed'] else 'FAIL'}")
    for failure in result["failures"]:
        print(f"REFUSAL {failure}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
