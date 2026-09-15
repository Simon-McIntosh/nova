"""Gate whole-solve executable size, instructions, and compilation time.

The input receipts are emitted by :mod:`benchmarks.program_scope_census`.
Sentinel copy counts remain diagnostic metrics because optimized HLO inlines
called programs.  The gate still requires each map control to see the known
operator paths, which keeps an empty or stale instrument from reporting an
improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CELLS = (300, 1000)
BASELINE_300_EXECUTABLE_BYTES = 461_724_765
MAX_300_EXECUTABLE_BYTES = 440_000_000
MAX_300_SOLVE_INSTRUCTIONS = 210_000
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
    if generated is not None and int(generated) > 0:
        return int(generated), "generated code fallback"
    error = executable.get("serialization_error")
    suffix = f": {error}" if error else ""
    raise ValueError(f"executable size is absent{suffix}")


def _compile_seconds(row: dict[str, Any], cells: int, label: str) -> float:
    value = float(row.get("compile_seconds", 0.0))
    if value <= 0.0:
        raise ValueError(f"{cells}-cell {label} compile time is absent")
    return value


def evaluate_gate(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    *,
    baseline_300_executable_bytes: int = BASELINE_300_EXECUTABLE_BYTES,
) -> dict[str, Any]:
    """Compare before and after receipts and return a quantitative verdict."""
    rows = []
    failures: list[str] = []
    findings: list[str] = []
    for cells in REQUIRED_CELLS:
        before = baseline[cells]
        after = candidate[cells]
        map_counts = {
            path: _copy_count(after["map"], path) for path in REPLICATION_PATHS
        }
        for path, count in map_counts.items():
            if count <= 0:
                failures.append(
                    f"{cells}-cell map sentinel {path!r} saw no known-present path"
                )
        solve_counts = {
            path: _copy_count(after["solve"], path) for path in REPLICATION_PATHS
        }
        before_instructions = int(before["solve"]["total_instructions"])
        after_instructions = int(after["solve"]["total_instructions"])
        if after_instructions >= before_instructions:
            failures.append(
                f"{cells}-cell solve instructions did not shrink: "
                f"{before_instructions} before, {after_instructions} after"
            )
        try:
            before_compile_seconds = _compile_seconds(before, cells, "baseline")
            after_compile_seconds = _compile_seconds(after, cells, "candidate")
        except ValueError as error:
            before_compile_seconds = before.get("compile_seconds")
            after_compile_seconds = after.get("compile_seconds")
            failures.append(str(error))
        else:
            if after_compile_seconds >= before_compile_seconds:
                failures.append(
                    f"{cells}-cell compile time did not shrink: "
                    f"{before_compile_seconds:.3f}s before, "
                    f"{after_compile_seconds:.3f}s after"
                )
        try:
            executable_bytes, executable_measure = _effective_executable_bytes(
                after["solve"]
            )
        except ValueError as error:
            executable_bytes = None
            executable_measure = "unavailable"
            if cells == 300:
                failures.append(f"{cells}-cell solve {error}")
            else:
                findings.append(f"{cells}-cell solve {error}")
        before_executable_bytes = (
            baseline_300_executable_bytes if cells == 300 else None
        )
        if cells == 300 and executable_bytes is not None:
            if executable_bytes >= baseline_300_executable_bytes:
                failures.append(
                    "300-cell solve executable did not shrink: "
                    f"{baseline_300_executable_bytes} bytes before, "
                    f"{executable_bytes} after"
                )
            if executable_bytes > MAX_300_EXECUTABLE_BYTES:
                failures.append(
                    f"300-cell solve executable is {executable_bytes} bytes, "
                    f"limit {MAX_300_EXECUTABLE_BYTES}"
                )
            if after_instructions > MAX_300_SOLVE_INSTRUCTIONS:
                failures.append(
                    f"300-cell solve has {after_instructions} instructions, "
                    f"limit {MAX_300_SOLVE_INSTRUCTIONS}"
                )
        rows.append(
            {
                "requested_cells": cells,
                "before": {
                    "solve_instructions": before_instructions,
                    "map_instructions": int(before["map"]["total_instructions"]),
                    "compile_seconds": before_compile_seconds,
                    "executable_bytes": before_executable_bytes,
                    "operator_copies": {
                        path: _copy_count(before["solve"], path)
                        for path in REPLICATION_PATHS
                    },
                },
                "after": {
                    "solve_instructions": after_instructions,
                    "map_instructions": int(after["map"]["total_instructions"]),
                    "compile_seconds": after_compile_seconds,
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
    return {
        "passed": not failures,
        "failures": failures,
        "findings": findings,
        "limits": {
            "300_cell_solve_instructions": MAX_300_SOLVE_INSTRUCTIONS,
            "300_cell_executable_bytes": MAX_300_EXECUTABLE_BYTES,
        },
        "rows": rows,
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Solve program size gate",
        "",
        "| cells | solve instructions before / after | map floor before / after | "
        "compile seconds before / after | executable bytes before / after | "
        "solve copies moment / topology | map copies moment / topology |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        before = row["before"]
        after = row["after"]
        executable_bytes = after["executable_bytes"]
        executable_display = (
            f"{executable_bytes:,}" if executable_bytes is not None else "unavailable"
        )
        before_executable = before["executable_bytes"]
        before_executable_display = (
            f"{before_executable:,}" if before_executable is not None else "unavailable"
        )
        before_compile = before["compile_seconds"]
        after_compile = after["compile_seconds"]
        compile_display = (
            f"{float(before_compile):.3f} / {float(after_compile):.3f}"
            if before_compile is not None and after_compile is not None
            else "unavailable"
        )
        lines.append(
            f"| {row['requested_cells']} | "
            f"{before['solve_instructions']:,} / {after['solve_instructions']:,} | "
            f"{before['map_instructions']:,} / {after['map_instructions']:,} | "
            f"{compile_display} | "
            f"{before_executable_display} / {executable_display} | "
            f"{after['operator_copies']['current-moment path']} / "
            f"{after['operator_copies']['topology read']} | "
            f"{after['map_operator_copies']['current-moment path']} / "
            f"{after['map_operator_copies']['topology read']} |"
        )
    lines.extend(["", f"Verdict: **{'PASS' if result['passed'] else 'FAIL'}**."])
    if result["failures"]:
        lines.extend(["", "Refusals:"])
        lines.extend(f"- {failure}" for failure in result["failures"])
    if result["findings"]:
        lines.extend(["", "Measured findings:"])
        lines.extend(f"- {finding}" for finding in result["findings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-300-executable-bytes",
        type=int,
        default=BASELINE_300_EXECUTABLE_BYTES,
        help="recorded baseline executable bytes for the 300-cell comparison",
    )
    args = parser.parse_args()
    result = evaluate_gate(
        _load_rungs(args.baseline_dir),
        _load_rungs(args.candidate_dir),
        baseline_300_executable_bytes=args.baseline_300_executable_bytes,
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
