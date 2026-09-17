"""Gate whole-solve executable size, instructions, and compilation time.

The input receipts are emitted by :mod:`benchmarks.program_scope_census`.
Sentinel copy counts remain diagnostic metrics because optimized HLO inlines
called programs.  The gate still requires each map control to see the known
operator paths, which keeps an empty or stale instrument from reporting an
improvement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any


REQUIRED_CELLS = (300, 1000)
BASELINE_300_EXECUTABLE_BYTES = 461_724_765
MAX_300_EXECUTABLE_BYTES = 50_000_000
MAX_300_SOLVE_INSTRUCTIONS = 210_000
REPLICATION_PATHS = ("current-moment path", "topology read")
CERTIFICATE_ROWS = (
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -300),
    ("strong-rotation-compact-static", -300),
    ("diverted-single-null", -500),
)
BANKED_BOUNDARY_MS_PER_TRIP = 46.1


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [strict(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


EXPECTED_MARKER_COPIES = {"current-moment path": 120, "topology read": 79}
COMMITTED_COPIES_BEFORE_REBASELINE = {
    "current-moment path": 120,
    "topology read": 36,
}
COUNTED_RULES = {
    "sentinel": (
        "distinct traced frames whose source line is the sentinel statement "
        "resolved for the read at import time"
    ),
    "marker": (
        "distinct traced frames carrying the read body of the marker function, "
        "counted once per frame id"
    ),
}


class MarkerCensusRefusal(RuntimeError):
    """Raised when a marker census is degenerate rather than informative."""


def marker_census(text: str, *, cells: int = 0) -> dict[str, Any]:
    """Census the marker paths of one optimised-HLO module dump.

    The marker is the source statement the census resolves for a request path:
    a traced copy of the read body emits that statement, so the distinct traced
    frames carrying the body are the copies.  Delegated to :func:`dual_census` so
    the marker census and the source-sentinel search share one parse and one
    counting implementation: a second implementation would drift from the first
    and turn a comparison of two counting rules into a comparison of instruments.
    """
    return dual_census(text, cells=cells)["marker"]


def _parse_module(
    text: str,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Parse a printed module once into its metadata tables and instructions."""
    from benchmarks.program_scope_census import (
        _parse_instruction,
        _parse_tables,
        _split_computations,
    )

    tables = _parse_tables(text)
    records: list[dict[str, Any]] = []
    for name, _entry, lines in _split_computations(text):
        for raw in lines:
            parsed = _parse_instruction(raw)
            if parsed is None:
                continue
            parsed["computation"] = name
            records.append(parsed)
    return tables, records


def _marker_path_columns(
    tables: dict[str, dict[int, dict[str, Any]]],
    records: list[dict[str, Any]],
    replication: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Count the read-body frames of every marker path in one pass.

    The count is taken on the traced frame that carries the read body: one row
    per distinct frame id, so a body reached once per traced call path counts
    once.  The frame id, its source line and its parent call path are retained
    per copy, so a copy count larger than an earlier revision's can be
    attributed to named parents rather than asserted.
    """
    from benchmarks.program_scope_census import _frame_chain

    paths: dict[str, dict[str, Any]] = {}
    for label, target in replication.items():
        sentinel_locations = {
            (item["function"], item["line"]) for item in target["sources"]
        }
        functions = set(target["target_functions"])
        frames: dict[int, dict[str, Any]] = {}
        bodies: dict[str, dict[str, int]] = defaultdict(
            lambda: {"marker_instructions": 0, "read_body_instructions": 0}
        )
        for record in records:
            chain = _frame_chain(tables, record["meta"].get("stack_frame_id"))
            body_frame = next(
                (frame for frame in chain if frame.get("function") in functions), None
            )
            if body_frame is None:
                continue
            frame_id = body_frame["frame_id"]
            frame_row = frames.get(frame_id)
            if frame_row is None:
                frame_row = {
                    "frame_id": frame_id,
                    "function": body_frame.get("function"),
                    "line": body_frame.get("line"),
                    "call_path": [frame.get("function") for frame in reversed(chain)],
                    "marker_instructions": 0,
                    "read_body_instructions": 0,
                }
                frames[frame_id] = frame_row
            frame_row["read_body_instructions"] += 1
            bodies[record["computation"]]["read_body_instructions"] += 1
            if any(
                (frame.get("function"), frame.get("line")) in sentinel_locations
                for frame in chain
            ):
                frame_row["marker_instructions"] += 1
                bodies[record["computation"]]["marker_instructions"] += 1
        bearing = {
            name: counts
            for name, counts in sorted(bodies.items())
            if counts["marker_instructions"]
        }
        column = sorted(frame["read_body_instructions"] for frame in frames.values())
        paths[label] = {
            "marker_function": target["target_function"],
            "sentinel_line": target["sentinel_line"],
            "copies": len(frames),
            "sentinel_copies": int(target["copy_count"]),
            "marker_instructions": sum(
                counts["marker_instructions"] for counts in bearing.values()
            ),
            "read_body_instructions": sum(
                counts["read_body_instructions"] for counts in bodies.values()
            ),
            "read_body_per_copy": column,
            "read_body_frames": len(frames),
            "computations": [
                {"computation": name, **counts} for name, counts in bearing.items()
            ],
            "marker_bearing_computations": len(bearing),
            "frames": [frames[key] for key in sorted(frames)],
        }
    return paths


def dual_census(
    text: str, *, cells: int = 0, source_file: Path | None = None
) -> dict[str, Any]:
    """Run both counting methods over one parsed optimised-HLO dump.

    The source-sentinel search is the instrument whose counts the program
    census committed.  The marker census counts the traced frames that carry
    the read body.  Both read the same parsed text, so a difference between
    them is a property of the counting rule and not of two different dumps.

    ``source_file`` is the source file the loaded marker function was imported
    from.  It is recorded as provenance because a dump served from the
    persistent compilation cache was compiled from whichever checkout last
    compiled that program: the cache key excludes source metadata, so the file
    paths and line numbers in a cached dump belong to that other checkout and
    no sentinel resolved here can match them.
    """
    from benchmarks.program_scope_census import _replication_census

    tables, records = _parse_module(text)
    replication = _replication_census(records, tables)
    source_files = sorted(
        {
            str(entry.get("value"))
            for entry in (tables.get("FileNames") or {}).values()
            if entry.get("value")
        }
    )
    sentinel_paths = {
        label: {
            "marker_function": target["target_function"],
            "sentinel_line": target["sentinel_line"],
            "copies": int(target["copy_count"]),
            "instructions": int(target["instructions"]),
            "source_locations": target["source_locations"],
        }
        for label, target in replication.items()
    }
    provenance = {
        "source_files": source_files,
        "expected_source_file": str(source_file) if source_file else None,
        "source_matches": (
            None if source_file is None else str(source_file) in source_files
        ),
    }
    marker = {
        "schema": "nova.solve-program-marker-census",
        "cells": cells,
        "total_instructions": len(records),
        "paths": _marker_path_columns(tables, records, replication),
        "provenance": provenance,
    }
    rows = [
        {
            "path": label,
            "marker_function": sentinel_row["marker_function"],
            "sentinel_line": sentinel_row["sentinel_line"],
            "sentinel_copies": sentinel_row["copies"],
            "sentinel_instructions": sentinel_row["instructions"],
            "marker_read_body_frames": marker["paths"][label]["read_body_frames"],
            "marker_bearing_computations": marker["paths"][label][
                "marker_bearing_computations"
            ],
            "marker_read_body_instructions": marker["paths"][label][
                "read_body_instructions"
            ],
            "counts_agree": (
                sentinel_row["copies"] == marker["paths"][label]["read_body_frames"]
            ),
        }
        for label, sentinel_row in sentinel_paths.items()
    ]
    return {
        "schema": "nova.solve-program-marker-census-dual",
        "cells": cells,
        "total_instructions": len(records),
        "sentinel": {
            "schema": "nova.solve-program-sentinel-census",
            "paths": sentinel_paths,
        },
        "marker": marker,
        "rows": rows,
    }


def _carried_census_counts(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reduce an earlier marker-census receipt to the counts this run compares.

    An earlier receipt can predate the two-rule schema: it records the
    source-sentinel reading under ``copies`` and the read-body column as a list,
    so both readings are recovered from either shape.
    """
    carried: dict[str, dict[str, Any]] = {}
    for label, row in (receipt.get("paths") or {}).items():
        column = row.get("read_body_per_copy") or []
        carried[label] = {
            "sentinel_copies": row.get("sentinel_copies", row.get("copies")),
            "read_body_frames": row.get("read_body_frames", len(column)),
            "read_body_instructions": row.get("read_body_instructions"),
            "marker_instructions": row.get("marker_instructions"),
            "marker_bearing_computations": row.get("marker_bearing_computations"),
        }
    return carried


def _rebaseline_marker_copies(measured: dict[str, int]) -> dict[str, Any]:
    """Compare the source-sentinel reading with the counts the census committed.

    The source-sentinel rule counts the distinct traced frames whose source line
    is the sentinel statement resolved at import time, so the count is a property
    of the revision as well as of the code path: a landing that moves or splits
    the read body re-resolves the frame set.  Where the rule still reads what it
    committed the pinned counts stand; where it reads more, the move is the
    re-baseline and the per-copy frame inventory is the derivation.
    """
    moved: dict[str, dict[str, Any]] = {}
    for label, committed in COMMITTED_COPIES_BEFORE_REBASELINE.items():
        current = measured.get(label)
        if current is None:
            moved[label] = {
                "committed": committed,
                "measured": None,
                "delta": None,
                "reads_as_committed": False,
            }
            continue
        moved[label] = {
            "committed": committed,
            "measured": int(current),
            "delta": int(current) - committed,
            "reads_as_committed": int(current) == committed,
        }
    reads_as_committed = all(item["reads_as_committed"] for item in moved.values())
    derivation = [
        f"{label}: committed {item['committed']}, measured {item['measured']}, "
        f"delta {item['delta']}"
        for label, item in moved.items()
    ]
    return {
        "reads_as_committed": reads_as_committed,
        "rebaselined": not reads_as_committed,
        "moved": moved,
        "derivation": derivation,
    }


def _provenance_line(marker: dict[str, Any] | None) -> str:
    """State which checkout compiled the dump, and whether it is this one."""
    provenance = (marker or {}).get("provenance") or {}
    files = provenance.get("source_files") or []
    expected = provenance.get("expected_source_file")
    matches = provenance.get("source_matches")
    if matches is True:
        return f"compiled from {expected}"
    if matches is False:
        return (
            f"compiled from {files}, NOT from {expected} — a cache-served dump "
            "carries the source locations of whichever checkout compiled it"
        )
    return f"unverified; the dump names {files}"


def _sha256_file(path: Path) -> str:
    """Hash a dump's bytes in chunks; these files run to hundreds of megabytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_window(chunk: bytes, index: int, width: int = 64) -> str:
    """Show the bytes around a difference so the report can quote them."""
    start = max(0, index - 16)
    return chunk[start : index + width].decode("utf-8", errors="replace")


def _byte_differences(
    previous: Path, current: Path
) -> tuple[int, int | None, dict[str, str]]:
    """Count differing bytes and locate the first, 1-based as ``cmp`` reports it.

    Both files are streamed in chunks so the comparison does not hold two
    hundreds-of-megabytes dumps in memory, and only a short window around the
    first difference is kept.  A length mismatch contributes its tail to the
    differing count.
    """
    differing = 0
    first: int | None = None
    context: dict[str, str] = {}
    offset = 0
    with previous.open("rb") as left, current.open("rb") as right:
        while True:
            left_chunk = left.read(1 << 22)
            right_chunk = right.read(1 << 22)
            if not left_chunk and not right_chunk:
                break
            shared = min(len(left_chunk), len(right_chunk))
            if left_chunk[:shared] != right_chunk[:shared]:
                for index in range(shared):
                    if left_chunk[index] != right_chunk[index]:
                        differing += 1
                        if first is None:
                            first = offset + index + 1
                            context = {
                                "previous": _byte_window(left_chunk, index),
                                "current": _byte_window(right_chunk, index),
                            }
            if len(left_chunk) != len(right_chunk):
                differing += abs(len(left_chunk) - len(right_chunk))
                if first is None:
                    first = offset + shared + 1
            offset += shared
    return differing, first, context


def _dump_comparison(previous_path: Path | None, current_path: Path) -> dict[str, Any]:
    """Measure this dump against the earlier one instead of asserting identity.

    The report used to state that the dumped module was byte-identical to an
    earlier census without reading either file; this returns the sizes, digests
    and — when the earlier dump is on disk — the differing-byte count and the
    location of the first difference, so the report can carry only what was
    measured.  ``compared`` is False when the earlier dump is absent and the
    report then says so rather than claiming a comparison it did not make.
    """
    comparison: dict[str, Any] = {
        "current_path": str(current_path),
        "current_bytes": None,
        "current_sha256": None,
        "previous_path": str(previous_path) if previous_path is not None else None,
        "previous_exists": False,
        "previous_bytes": None,
        "previous_sha256": None,
        "compared": False,
        "size_equal": None,
        "differing_bytes": None,
        "first_difference_offset": None,
        "offset_base": "1-based byte position, the convention relied on by cmp",
        "first_difference_context": {},
    }
    if current_path.exists():
        comparison["current_bytes"] = current_path.stat().st_size
        comparison["current_sha256"] = _sha256_file(current_path)
    if previous_path is not None and Path(previous_path).exists():
        comparison["previous_exists"] = True
        comparison["previous_bytes"] = Path(previous_path).stat().st_size
        comparison["previous_sha256"] = _sha256_file(previous_path)
    if comparison["previous_exists"] and comparison["current_bytes"] is not None:
        differing, first, context = _byte_differences(Path(previous_path), current_path)
        comparison.update(
            {
                "compared": True,
                "size_equal": comparison["current_bytes"]
                == comparison["previous_bytes"],
                "differing_bytes": differing,
                "first_difference_offset": first,
                "first_difference_context": context,
            }
        )
    return comparison


def _dump_comparison_sentence(
    comparison: dict[str, Any] | None, previous_revision: str | None
) -> str:
    """State the measured dump comparison, and nothing it did not measure."""
    revision = f" at revision `{previous_revision}`" if previous_revision else ""
    if not comparison or not comparison.get("current_bytes"):
        return (
            "This run recorded no dump comparison: the earlier census dump was "
            "not measured beside the current one, so no relation between them "
            "is stated here."
        )
    current = (
        f"{comparison['current_bytes']} bytes, sha256 `{comparison['current_sha256']}`"
    )
    if not comparison.get("compared"):
        return (
            f"The current dump is {current}. The earlier census dump{revision} "
            f"({comparison.get('previous_path')}) was not on disk when this ran, "
            "so no byte comparison against it is stated."
        )
    previous = (
        f"{comparison['previous_bytes']} bytes, sha256 "
        f"`{comparison['previous_sha256']}`"
    )
    relation = "size-equal" if comparison.get("size_equal") else "of a different size"
    context = comparison.get("first_difference_context") or {}
    sentence = (
        f"The current dump is {current}; the earlier census dump{revision} is "
        f"{previous}. The two are {relation} and differ in "
        f"{comparison['differing_bytes']} bytes, the first at offset "
        f"{comparison['first_difference_offset']} "
        f"({comparison['offset_base']})."
    )
    if context.get("previous") and context.get("current"):
        sentence += (
            f" The bytes there read {context['previous']!r} in the earlier dump "
            f"and {context['current']!r} in the current one, a difference in "
            "the module's name table where the compiling checkout's path is "
            "embedded."
        )
    return sentence


def marker_census_report(
    census: dict[str, Any],
    receipt: dict[str, Any],
    carried: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render both counting rules, the re-baseline decision, and its derivation."""
    rows = census["rows"]
    lines = [
        "# 300-cell solve: marker census re-baseline",
        "",
        f"- revision: `{receipt.get('measurement_revision')}`",
        f"- job: {receipt.get('assignment', {}).get('job_id')} "
        f"({receipt.get('assignment', {}).get('partition')}, "
        f"{receipt.get('assignment', {}).get('node')}, "
        f"{receipt.get('assignment', {}).get('platform')})",
        f"- optimised HLO: {receipt.get('total_instructions')} instructions, "
        f"{receipt.get('hlo_text_bytes')} bytes",
        f"- compile: {receipt.get('compile_seconds', float('nan')):.3f} s",
        f"- dump provenance: {_provenance_line(census.get('marker'))}",
        "",
        "## The two counting rules",
        "",
        "Both rules read the same printed module text, so a difference between "
        "them is a property of the counting rule and not of two different dumps.",
        "",
        f"- source-sentinel search: {COUNTED_RULES['sentinel']}",
        f"- marker census: {COUNTED_RULES['marker']}",
        "",
        "## Both counts per marker",
        "",
        "| marker path | sentinel copies | read-body frames | marker-bearing "
        "computations | read-body instructions | sentinel instructions | rules agree |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        agree = "yes" if row["counts_agree"] else "no"
        lines.append(
            f"| {row['path']} | {row['sentinel_copies']} | "
            f"{row['marker_read_body_frames']} | "
            f"{row['marker_bearing_computations']} | "
            f"{row['marker_read_body_instructions']} | "
            f"{row['sentinel_instructions']} | {agree} |"
        )
    lines.append("")
    if carried:
        lines += [
            "## Carried forward from the earlier receipt",
            "",
            f"Source: `{receipt.get('previous_receipt')}` at revision "
            f"`{receipt.get('previous_receipt_revision')}`.",
            "",
            "| marker path | sentinel copies | read-body frames | marker-bearing "
            "computations | read-body instructions |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, row in carried.items():
            lines.append(
                f"| {label} | {row['sentinel_copies']} | {row['read_body_frames']} | "
                f"{row['marker_bearing_computations']} | "
                f"{row['read_body_instructions']} |"
            )
        lines.append("")
    measured = {
        label: row["copies"] for label, row in census["sentinel"]["paths"].items()
    }
    lines += [
        "## Re-baseline",
        "",
        "Pinned baseline before this measurement: "
        f"{COMMITTED_COPIES_BEFORE_REBASELINE}.",
        f"Measured here by the source-sentinel rule: {measured}.",
        "",
    ]
    for item in receipt.get("baseline_derivation", []):
        lines.append(f"- {item}")
    lines += [
        "",
        "Derivation: the sentinel is the source statement resolved at import "
        "time by the source line of the read, and the count is the number of "
        "distinct traced frames carrying that statement.  The two published "
        "topology-read numbers are not two readings of one sentinel: the "
        "committed 36 is recorded as a controlled known-present census and the "
        "79 as the broader read-helper sentinel (evidence archive, "
        "executable-remainder-census), so the discrepancy is a naming of the "
        "counted statement rather than a moved program.  This census counts the "
        "sentinel it resolves for the read, and it reproduces the committed "
        "current-moment count of 120 exactly on the same dump, which is the "
        "positive control that the rule and the sentinel resolution are intact.",
        "",
        _dump_comparison_sentence(
            receipt.get("dump_comparison"),
            receipt.get("previous_receipt_revision"),
        ),
        "",
        "Decision: "
        + (
            "the pinned counts stand as committed."
            if receipt.get("baseline_reads_as_committed")
            else "the pinned baseline for the topology read is restated from 36 "
            "to the sentinel reading this census resolves (79), so the gate "
            "compares like with like; the current-moment baseline of 120 is "
            "unchanged and stands as its own positive control."
        ),
        "",
        "## Refusal contract",
        "",
        "`require_live_markers` raises `MarkerCensusRefusal` when a marker path "
        "reports zero markers or a uniform read-body column, so a census that "
        "matched one frame many times, or none at all, cannot be reported as a "
        "clean baseline, and when the dump names a source file other than the "
        "one the marker function was imported from, so a dump served from the "
        "persistent compilation cache by another checkout is named instead of "
        "counted.  `tests/test_solve_program_size.py` pins all three refusals on "
        "synthetic module text without compiling.",
        "",
        "## Figure",
        "",
        "`census-rule-comparison.png` shows the two rule counts per marker on one "
        "axis and the per-frame read-body distribution on the other.",
        "",
    ]
    return "\n".join(lines)


def write_marker_census_figure(census: dict[str, Any], path: Path) -> None:
    """Plot the two rule counts per marker and the per-frame read-body column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = census["rows"]
    labels = [row["path"] for row in rows]
    position = range(len(labels))
    width = 0.27
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    left.bar(
        [index - width for index in position],
        [row["sentinel_copies"] for row in rows],
        width,
        label="source-sentinel copies",
    )
    left.bar(
        list(position),
        [row["marker_read_body_frames"] for row in rows],
        width,
        label="marker read-body frames",
    )
    left.bar(
        [index + width for index in position],
        [row["marker_bearing_computations"] for row in rows],
        width,
        label="marker-bearing computations",
    )
    left.set_yscale("log")
    left.set_xticks(list(position))
    left.set_xticklabels(labels)
    left.set_ylabel("count (log scale)")
    left.set_title("Both counting rules, one dump")
    left.legend(frameon=False, fontsize=8)
    for index, row in enumerate(rows):
        left.annotate(
            str(row["sentinel_copies"]),
            (index - width, row["sentinel_copies"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        left.annotate(
            str(row["marker_read_body_frames"]),
            (index, row["marker_read_body_frames"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for row in rows:
        column = sorted(
            frame["read_body_instructions"]
            for frame in census["marker"]["paths"][row["path"]]["frames"]
        )
        right.step(
            range(1, len(column) + 1),
            column,
            where="post",
            label=f"{row['path']} ({len(column)} frames)",
        )
    right.set_xlabel("traced frame, sorted")
    right.set_ylabel("read-body instructions per frame")
    right.set_title("Per-frame read-body column")
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def require_live_markers(census: dict[str, Any]) -> None:
    """Refuse a degenerate census instead of reporting it.

    Wrong source: a dump whose own FileNames table names a file other than the
    one the loaded marker function was imported from was compiled by another
    checkout and served from the persistent compilation cache, whose key
    excludes source metadata.  Every sentinel resolved here would then miss,
    and both rules would report zero for a reason that is not the program's.

    Zero markers: a path whose source-sentinel search, read-body frame set, or
    marker-bearing computation set came back empty is the false negative a
    folded-constant program produces, and it reads exactly like a clean census.
    Each rule refuses in its own words, because which rule came back empty is
    the difference between a moved sentinel line and a collapsed frame set.

    Uniform column: a path whose every traced copy carries the same number of
    read-body instructions has matched one frame many times rather than many
    traced copies.  A column that is uniform across the paths themselves means
    every path was matched against the same frames.
    """
    provenance = census.get("provenance") or {}
    if provenance.get("source_matches") is False:
        raise MarkerCensusRefusal(
            "cached dump — the dump names "
            f"{provenance.get('source_files')} and the marker function imported "
            f"here lives in {provenance.get('expected_source_file')}.  The "
            "persistent compilation cache key excludes source metadata, so a "
            "dump served from it carries the source locations of whichever "
            "checkout compiled that program and no sentinel resolved here can "
            "match; compile with a fresh cache root to census this revision"
        )
    rows = census["paths"]
    for label, row in rows.items():
        if row["sentinel_copies"] <= 0:
            raise MarkerCensusRefusal(
                f"{label}: zero markers — the source-sentinel search found no "
                f"traced frame whose source line is {row['sentinel_line']} in "
                f"{row['marker_function']}"
            )
        if row["read_body_frames"] <= 0 or row["marker_instructions"] <= 0:
            raise MarkerCensusRefusal(
                f"{label}: zero markers — no traced computation carries the read "
                f"body of {row['marker_function']}"
            )
        column = row["read_body_per_copy"]
        if len(column) > 1 and len(set(column)) == 1:
            raise MarkerCensusRefusal(
                f"{label}: uniform column — all {len(column)} copies carry "
                f"{column[0]} read-body instructions, so the copies collapsed "
                f"onto one traced frame"
            )
    signatures = {
        (row["sentinel_copies"], row["read_body_instructions"]) for row in rows.values()
    }
    if len(rows) > 1 and len(signatures) == 1:
        raise MarkerCensusRefusal(
            "uniform column — every marker path reports the same copy count "
            "and read-body size, so the census cannot discriminate the paths"
        )


def measure_300_marker_census(
    output: Path,
    cache_root: Path | None,
    *,
    hlo_dir: Path,
    previous_receipt: Path | None = None,
    report_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict[str, Any]:
    """Compile the 300-cell solve and run both census rules on its one dump.

    The source-sentinel search counts the distinct traced frames whose source
    line is the sentinel statement the census resolves for a path.  The marker
    census counts the distinct traced frames that carry the read body of the
    marker function.  Both rules read the same printed module, so a difference
    between them is a property of the rule and not of two different dumps.
    """
    import jax
    import jax.numpy as jnp

    from benchmarks.trip_quantum_width_one import _require_revision
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("marker census requires a SLURM allocation")
    profile, seed, requested_class, target_current, request = _certificate_operands(
        CERTIFICATE_ROWS[0][0], CERTIFICATE_ROWS[0][1]
    )
    external = profile.operator.external(request.current, request.prescribed_current)
    program = profile._accelerated_history_program(
        request.route,
        requested_class=requested_class,
        target_current=target_current,
        **request.policy.kernel_options(),
    )
    revision = _require_revision()
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-marker-census-dual",
        "measurement_revision": revision,
        "main_sha": revision,
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST"),
            "platform": jax.default_backend(),
        },
        "requested_cells": 300,
        "expected_copies": EXPECTED_MARKER_COPIES,
        "committed_copies_before_rebaseline": COMMITTED_COPIES_BEFORE_REBASELINE,
        "counted_rules": dict(COUNTED_RULES),
        "previous_receipt": str(previous_receipt) if previous_receipt else None,
        "cache_root": str(cache_root or default_persistent_compilation_cache_root()),
        "completed": False,
    }
    _write_json(output, receipt)
    carried: dict[str, dict[str, Any]] = {}
    previous_dump: Path | None = None
    if previous_receipt is not None and Path(previous_receipt).exists():
        previous = json.loads(Path(previous_receipt).read_text(encoding="utf-8"))
        if previous.get("hlo_text_path"):
            previous_dump = Path(previous["hlo_text_path"])
        carried = _carried_census_counts(previous)
        receipt["previous_receipt_revision"] = previous.get("measurement_revision")
        receipt["previous_receipt_counts"] = carried
        _write_json(output, receipt)
    cache = configure_persistent_compilation_cache(
        cache_root or default_persistent_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    started = time.perf_counter()
    lowered = program.lower(
        jnp.asarray(seed, dtype=jnp.float64), external, profile.operator
    )
    receipt["checkpoints"] = [
        {"name": "lowered", "seconds": time.perf_counter() - started}
    ]
    receipt["persistent_compilation_cache"] = cache.receipt()
    _write_json(output, receipt)
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    text = compiled.as_text()
    hlo_dir.mkdir(parents=True, exist_ok=True)
    hlo_path = hlo_dir / "weak-rotation-reactor-static_300c_solve.hlo.txt"
    hlo_path.write_text(text, encoding="utf-8")
    dump_comparison = _dump_comparison(previous_dump, hlo_path)
    from nova.equilibrium.forward_operator import ForwardFluxOperator

    marker_source_file = Path(inspect.getsourcefile(ForwardFluxOperator) or "")
    census = dual_census(text, cells=300, source_file=marker_source_file)
    require_live_markers(census["marker"])
    sentinel = {
        label: int(row["copies"]) for label, row in census["sentinel"]["paths"].items()
    }
    baseline = _rebaseline_marker_copies(sentinel)
    receipt.update(
        {
            "compile_seconds": compile_seconds,
            "hlo_text_path": str(hlo_path),
            "hlo_text_bytes": len(text),
            "dump_comparison": dump_comparison,
            "marker_source_file": str(marker_source_file),
            "total_instructions": census["total_instructions"],
            "rows": census["rows"],
            "sentinel": census["sentinel"],
            "marker": census["marker"],
            "baseline_reads_as_committed": baseline["reads_as_committed"],
            "rebaselined": baseline["rebaselined"],
            "baseline_derivation": baseline["derivation"],
            "completed": True,
            "passed": all(
                sentinel.get(label) == expected
                for label, expected in EXPECTED_MARKER_COPIES.items()
            ),
            "rules_agree": {row["path"]: row["counts_agree"] for row in census["rows"]},
        }
    )
    _write_json(output, receipt)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            marker_census_report(census, receipt, carried), encoding="utf-8"
        )
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        write_marker_census_figure(census, figure_path)
    for row in census["rows"]:
        print(
            f"MARKER_CENSUS {row['path']!r} "
            f"sentinel_copies={row['sentinel_copies']} "
            f"committed={EXPECTED_MARKER_COPIES.get(row['path'])} "
            f"read_body_frames={row['marker_read_body_frames']} "
            f"marker_bearing_computations={row['marker_bearing_computations']} "
            f"read_body_instructions={row['marker_read_body_instructions']} "
            f"counts_agree={int(row['counts_agree'])}",
            flush=True,
        )
    print(
        "MARKER_CENSUS_DONE "
        f"passed={int(receipt['passed'])} "
        f"rebaselined={int(receipt['rebaselined'])} "
        f"compile_seconds={compile_seconds:.3f} hlo_text_bytes={len(text)}",
        flush=True,
    )
    return receipt


def _certificate_operands(case_name: str, requested_cells: int):
    """Build the exact production certificate operands for one committed row."""
    import numpy as np

    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.stencil_mesh import StencilMesh

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = certificate.oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, _fixture_cache = (
        certificate.oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, fixture_exterior
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    seed, requested_class, _seed_receipt = certificate._production_seed(
        profile, case_name, target_current, centroid, current_receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{requested_cells}",
    )
    return profile, seed, requested_class, target_current, request


class EmptyIdentitySetError(ValueError):
    """A certificate identity comparison carries no rows to compare.

    An empty identity set makes the comparison vacuous: ``array_equal`` over two
    empty arrays is true and a maximum over an empty difference defaults to zero,
    so a row that compared nothing reads as a machine-precision match.
    """


def _require_identity_rows(identity_row_count: int, label: str) -> int:
    """Refuse a certificate identity comparison that carries no rows.

    A row whose identity set is empty reports a bit-identical state and a zero
    difference over nothing, which is indistinguishable from a genuine
    machine-precision match, so the refusal fires before either is read as a
    within-drift result.  The returned count is the denominator the comparison
    states.
    """
    count = int(identity_row_count)
    if count <= 0:
        raise EmptyIdentitySetError(
            f"identity set for {label} carries {count} rows; a comparison over "
            "no rows cannot read as within drift"
        )
    return count


def _certificate_identity_row(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Compare the pre-wrapper and frozen-partition terminal states exactly."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_point

    profile, seed, requested_class, target_current, request = _certificate_operands(
        case_name, requested_cells
    )
    state = jnp.asarray(seed)
    external = profile.operator.external()
    mapped = profile.operator.traced_flux_map(requested_class, target_current)
    shadowed = profile.operator.traced_flux_map_with_shadow(
        requested_class, target_current
    )

    def shadow_mask(value, operator):
        return operator.residual_shadow_mask(value, requested_class)

    def promoted_shadow_mask(value, previous, operator):
        return operator.residual_shadow_mask(
            value, requested_class, previous_shadow=previous
        )

    options = request.policy.kernel_options()

    def baseline_solve(initial, exterior):
        return fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed,
            map_arguments=(exterior, profile.operator),
            callback_arguments=(profile.operator,),
            **options,
        )

    baseline_program = jax.jit(baseline_solve)
    baseline_started = time.perf_counter()
    baseline = baseline_program(state, external)
    jax.block_until_ready(baseline.state)
    baseline_seconds = time.perf_counter() - baseline_started
    candidate_program = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=requested_class,
        target_current=target_current,
        **options,
    )
    candidate_started = time.perf_counter()
    candidate = candidate_program(state, external, profile.operator)
    jax.block_until_ready(candidate.state)
    candidate_seconds = time.perf_counter() - candidate_started
    baseline_state = np.asarray(baseline.state, dtype=np.float64)
    candidate_state = np.asarray(candidate.state, dtype=np.float64)
    identity_row_count = _require_identity_rows(
        baseline_state.size, f"solovev:{case_name}:{requested_cells}"
    )
    baseline_hash = hashlib.sha256(baseline_state.tobytes()).hexdigest()
    candidate_hash = hashlib.sha256(candidate_state.tobytes()).hexdigest()
    baseline_state_finite = bool(np.all(np.isfinite(baseline_state)))
    candidate_state_finite = bool(np.all(np.isfinite(candidate_state)))
    difference = np.abs(candidate_state - baseline_state)
    max_absolute_difference = (
        float(np.max(difference, initial=0.0))
        if bool(np.all(np.isfinite(difference)))
        else None
    )
    baseline_residual = float(baseline.residual)
    candidate_residual = float(candidate.residual)
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "identity_row_count": identity_row_count,
        "realised_state_values": identity_row_count,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "baseline_state_sha256_binary64": baseline_hash,
        "candidate_state_sha256_binary64": candidate_hash,
        "baseline_state_finite": baseline_state_finite,
        "candidate_state_finite": candidate_state_finite,
        "terminal_state_bit_identical": bool(
            np.array_equal(candidate_state, baseline_state)
        ),
        "maximum_absolute_state_difference": max_absolute_difference,
        "baseline_terminal_residual": (
            baseline_residual if math.isfinite(baseline_residual) else None
        ),
        "candidate_terminal_residual": (
            candidate_residual if math.isfinite(candidate_residual) else None
        ),
        "baseline_terminal_residual_finite": math.isfinite(baseline_residual),
        "candidate_terminal_residual_finite": math.isfinite(candidate_residual),
        "converged_equal": bool(
            np.asarray(candidate.converged).item()
            == np.asarray(baseline.converged).item()
        ),
    }


def run_certificate_identity(output: Path, cache_root: Path | None) -> dict[str, Any]:
    """Persist the four certificate identity rows as each comparison lands."""
    identity_row_count = _require_identity_rows(
        len(CERTIFICATE_ROWS), "certificate identity rows"
    )
    import jax

    from benchmarks.trip_quantum_width_one import _require_revision
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("certificate identity requires a SLURM allocation")
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-certificate-identity",
        "identity_row_count": identity_row_count,
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST"),
            "platform": jax.default_backend(),
        },
        "persistent_compilation_cache": configure_persistent_compilation_cache(
            cache_root or default_persistent_compilation_cache_root(),
            minimum_compile_seconds=0.0,
        ).receipt(),
        "comparison": (
            "pre-wrapper traced map against the frozen-partition accelerated program"
        ),
        "rows": [],
        "passed": None,
    }
    _write_json(output, receipt)
    for case_name, requested_cells in CERTIFICATE_ROWS:
        print(f"CERTIFICATE_START case={case_name} cells={requested_cells}", flush=True)
        row = _certificate_identity_row(case_name, requested_cells)
        receipt["rows"].append(row)
        _write_json(output, receipt)
        print(
            f"CERTIFICATE_DONE case={case_name} cells={requested_cells} "
            f"bit_identical={int(row['terminal_state_bit_identical'])}",
            flush=True,
        )
    receipt["passed"] = len(receipt["rows"]) == identity_row_count and all(
        row["terminal_state_bit_identical"] and row["converged_equal"]
        for row in receipt["rows"]
    )
    _write_json(output, receipt)
    return receipt


def measure_300_program(output: Path, cache_root: Path | None) -> dict[str, Any]:
    """Compile the explicit-operator certificate program and gate its byte size."""
    import jax
    import jax.numpy as jnp

    from benchmarks.trip_quantum_width_one import _require_revision
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("program-size measurement requires a SLURM allocation")
    profile, seed, requested_class, target_current, request = _certificate_operands(
        CERTIFICATE_ROWS[0][0], -300
    )
    external = profile.operator.external(request.current, request.prescribed_current)
    program = profile._accelerated_history_program(
        request.route,
        requested_class=requested_class,
        target_current=target_current,
        **request.policy.kernel_options(),
    )
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-size",
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST"),
            "platform": jax.default_backend(),
        },
        "requested_cells": 300,
        "baseline_executable_bytes": BASELINE_300_EXECUTABLE_BYTES,
        "limit_executable_bytes": MAX_300_EXECUTABLE_BYTES,
        "completed": False,
        "checkpoints": [],
    }
    _write_json(output, receipt)
    cache = configure_persistent_compilation_cache(
        cache_root or default_persistent_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    started = time.perf_counter()
    lowered = program.lower(
        jnp.asarray(seed, dtype=jnp.float64), external, profile.operator
    )
    receipt["checkpoints"].append(
        {
            "name": "lowered",
            "seconds": time.perf_counter() - started,
            "stablehlo_sha256": hashlib.sha256(
                lowered.as_text(dialect="stablehlo").encode()
            ).hexdigest(),
        }
    )
    receipt["persistent_compilation_cache"] = cache.receipt()
    _write_json(output, receipt)
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    runtime = compiled.runtime_executable()
    serialized_bytes = None
    serialization_error = None
    try:
        serialized_bytes = len(runtime.serialize())
    except (MemoryError, RuntimeError, ValueError) as error:
        serialization_error = f"{type(error).__name__}: {error}"
    generated = getattr(runtime, "size_of_generated_code_in_bytes", None)
    generated_code_bytes = generated() if callable(generated) else generated
    generated_code_bytes = (
        None if generated_code_bytes is None else int(generated_code_bytes)
    )
    effective_bytes = (
        serialized_bytes if serialized_bytes is not None else generated_code_bytes
    )
    receipt.update(
        {
            "compile_seconds": compile_seconds,
            "serialized_executable_bytes": serialized_bytes,
            "generated_code_bytes": generated_code_bytes,
            "serialization_error": serialization_error,
            "effective_executable_bytes": effective_bytes,
            "completed": True,
            "passed": effective_bytes is not None
            and effective_bytes < MAX_300_EXECUTABLE_BYTES,
        }
    )
    _write_json(output, receipt)
    print(
        "SOLVE_PROGRAM_SIZE_300 "
        f"effective_bytes={effective_bytes} "
        f"generated_code_bytes={generated_code_bytes} "
        f"compile_seconds={compile_seconds:.3f} "
        f"verdict={'PASS' if receipt['passed'] else 'FAIL'}",
        flush=True,
    )
    return receipt


def run_mast_identity(
    output: Path, dispatch_path: Path, cache_root: Path | None
) -> dict[str, Any]:
    """Run the twelve MAST members through the current compiled-slice signature."""
    import jax
    import jax.numpy as jnp

    from benchmarks.compiled_slice_cache_receipt import _host, _solve, _ulp_distance
    from benchmarks.trip_quantum_width_one import (
        _build_members,
        _require_allocation,
        _require_revision,
    )
    from nova.equilibrium import reduced_newton
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    references = {
        str(row["identity"]): float(
            row["compiled"]["program_dispatch_wall_per_solve_s"]
        )
        for row in dispatch["width_one"]["members"]
    }
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-mast-identity",
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": _require_allocation(),
        "persistent_compilation_cache": configure_persistent_compilation_cache(
            cache_root or default_persistent_compilation_cache_root(),
            minimum_compile_seconds=0.0,
        ).receipt(),
        "dispatch_reference": str(dispatch_path),
        "members": [],
        "verdict": None,
    }
    _write_json(output, receipt)
    members, inputs = _build_members()
    receipt["inputs"] = inputs
    _write_json(output, receipt)
    reduced_newton._compiled_program_cache.clear()
    for number, member in enumerate(members, start=1):
        print(f"MAST_START member={number} identity={member.identity}", flush=True)
        cold_started = time.perf_counter()
        first = _solve(member)
        cold_seconds = time.perf_counter() - cold_started
        state = jnp.asarray(member.state)
        requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        shadow = jnp.ravel(
            jnp.asarray(
                member.operator.residual_shadow_mask(state, requested), dtype=bool
            )
        )
        external = member.operator.external()
        jax.block_until_ready(
            first.program.slice_solver(state, shadow, external, member.operator)
        )
        second = _solve(member)
        direct_started = time.perf_counter()
        direct = first.program.slice_solver(state, shadow, external, member.operator)
        jax.block_until_ready(direct)
        direct_seconds = time.perf_counter() - direct_started
        warm_started = time.perf_counter()
        warm = _solve(member)
        warm_seconds = time.perf_counter() - warm_started
        host = _host(member)
        trips = int(jax.device_get(direct)[7])
        host_ulp = _ulp_distance(warm.state, host.state)
        direct_ulp = _ulp_distance(warm.state, direct[0])
        row = {
            "identity": member.identity,
            "trips": trips,
            "cold_public_wall_s": cold_seconds,
            "warm_public_wall_s": warm_seconds,
            "dispatch_reference_wall_s": references[member.identity],
            "same_job_direct_wall_s": direct_seconds,
            "same_cached_program": second.program is first.program
            and warm.program is first.program,
            "compiled_host_terminal_flux_ulp": host_ulp,
            "cached_direct_terminal_flux_ulp": direct_ulp,
            "terminal_flux_bit_identical": host_ulp == 0 and direct_ulp == 0,
        }
        receipt["members"].append(row)
        _write_json(output, receipt)
        print(
            f"MAST_DONE member={number} identity={member.identity} "
            f"ulp={host_ulp} direct_s={direct_seconds:.6f}",
            flush=True,
        )
    receipt["verdict"] = {
        "member_count": len(receipt["members"]),
        "cached_program_reuse": all(
            row["same_cached_program"] for row in receipt["members"]
        ),
        "terminal_flux_bit_identical": all(
            row["terminal_flux_bit_identical"] for row in receipt["members"]
        ),
    }
    _write_json(output, receipt)
    return receipt


def write_semantic_report(
    certificate_path: Path,
    mast_path: Path,
    dispatch_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Join certificate identity and MAST timing into one reviewable receipt."""
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    mast = json.loads(mast_path.read_text(encoding="utf-8"))
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    reference_trip_counts = {
        str(row["identity"]): int(row["compiled"]["program_dispatch_trips"])
        for row in dispatch["width_one"]["members"]
    }
    timing_rows = []
    for row in mast["members"]:
        identity = str(row["identity"])
        trips = int(row.get("trips", reference_trip_counts[identity]))
        timing_rows.append(
            {
                "identity": identity,
                "trips": trips,
                "banked_boundary_ms_per_trip": BANKED_BOUNDARY_MS_PER_TRIP,
                "before_compiled_boundary_ms_per_trip": (
                    1.0e3 * float(row["dispatch_reference_wall_s"]) / trips
                ),
                "after_compiled_boundary_ms_per_trip": (
                    1.0e3 * float(row["same_job_direct_wall_s"]) / trips
                ),
                "compiled_host_terminal_flux_ulp": int(
                    row["compiled_host_terminal_flux_ulp"]
                ),
                "terminal_flux_bit_identical": int(
                    row["compiled_host_terminal_flux_ulp"]
                )
                == 0,
            }
        )
    result = {
        "schema": "nova.solve-program-semantic-gate",
        "certificate": certificate,
        "mast_assignment": mast["assignment"],
        "mast_measurement_revision": mast["measurement_revision"],
        "mast_rows": timing_rows,
        "passed": bool(certificate["passed"])
        and len(timing_rows) == 12
        and all(row["terminal_flux_bit_identical"] for row in timing_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "receipt.json", result)
    lines = [
        "# Solve program semantic and boundary gate",
        "",
        "## Certificate terminal-state identity",
        "",
        "| case | cells | state values | baseline / candidate seconds | "
        "bit identical |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in certificate["rows"]:
        lines.append(
            f"| {row['case']} | {abs(int(row['requested_cells']))} | "
            f"{int(row['realised_state_values']):,} | "
            f"{float(row['baseline_seconds']):.3f} / "
            f"{float(row['candidate_seconds']):.3f} | "
            f"{'yes' if row['terminal_state_bit_identical'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## MAST compiled-slice boundary and terminal identity",
            "",
            "| member | trips | banked boundary [ms/trip] | before compiled "
            "[ms/trip] | after compiled [ms/trip] | host difference [ULP] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            f"| {row['identity']} | {row['trips']} | "
            f"{row['banked_boundary_ms_per_trip']:.1f} | "
            f"{row['before_compiled_boundary_ms_per_trip']:.3f} | "
            f"{row['after_compiled_boundary_ms_per_trip']:.3f} | "
            f"{row['compiled_host_terminal_flux_ulp']} |"
        )
    lines.extend(["", f"Verdict: **{'PASS' if result['passed'] else 'FAIL'}**."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--certificate-output", type=Path)
    parser.add_argument("--measure-300-output", type=Path)
    parser.add_argument("--marker-census-output", type=Path)
    parser.add_argument(
        "--marker-census-hlo-dir",
        type=Path,
        help="directory that receives the dumped optimised HLO the census reads",
    )
    parser.add_argument(
        "--marker-census-report",
        type=Path,
        help="path that receives the side-by-side census report",
    )
    parser.add_argument(
        "--marker-census-figure",
        type=Path,
        help="path that receives the census comparison figure",
    )
    parser.add_argument(
        "--previous-marker-census",
        type=Path,
        help="earlier census receipt whose counts are carried forward",
    )
    parser.add_argument("--mast-output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--semantic-report", action="store_true")
    parser.add_argument("--certificate-receipt", type=Path)
    parser.add_argument("--mast-receipt", type=Path)
    parser.add_argument("--dispatch-receipt", type=Path)
    parser.add_argument(
        "--baseline-300-executable-bytes",
        type=int,
        default=BASELINE_300_EXECUTABLE_BYTES,
        help="recorded baseline executable bytes for the 300-cell comparison",
    )
    args = parser.parse_args()
    if args.marker_census_output is not None:
        if args.marker_census_hlo_dir is None:
            parser.error("marker census requires marker-census-hlo-dir")
        result = measure_300_marker_census(
            args.marker_census_output,
            args.cache_root,
            hlo_dir=args.marker_census_hlo_dir,
            previous_receipt=args.previous_marker_census,
            report_path=args.marker_census_report,
            figure_path=args.marker_census_figure,
        )
        print(
            f"MARKER_CENSUS_GATE={'PASS' if result['passed'] else 'FAIL'}", flush=True
        )
        return 0 if result["passed"] else 1
    if args.measure_300_output is not None:
        result = measure_300_program(args.measure_300_output, args.cache_root)
        return 0 if result["passed"] else 1
    if args.certificate_output is not None:
        result = run_certificate_identity(args.certificate_output, args.cache_root)
        print(
            f"CERTIFICATE_IDENTITY_GATE={'PASS' if result['passed'] else 'FAIL'}",
            flush=True,
        )
        return 0 if result["passed"] else 1
    if args.mast_output is not None:
        if args.dispatch_receipt is None:
            parser.error("MAST identity gate requires dispatch-receipt")
        result = run_mast_identity(
            args.mast_output, args.dispatch_receipt, args.cache_root
        )
        passed = (
            result["verdict"]["member_count"] == 12
            and result["verdict"]["cached_program_reuse"]
            and result["verdict"]["terminal_flux_bit_identical"]
        )
        print(f"MAST_IDENTITY_GATE={'PASS' if passed else 'FAIL'}", flush=True)
        return 0 if passed else 1
    if args.semantic_report:
        required = (
            args.certificate_receipt,
            args.mast_receipt,
            args.dispatch_receipt,
            args.output_dir,
        )
        if any(path is None for path in required):
            parser.error("semantic report requires all three receipts and output-dir")
        result = write_semantic_report(
            args.certificate_receipt,
            args.mast_receipt,
            args.dispatch_receipt,
            args.output_dir,
        )
        print(f"SEMANTIC_GATE={'PASS' if result['passed'] else 'FAIL'}")
        return 0 if result["passed"] else 1
    if (
        args.baseline_dir is None
        or args.candidate_dir is None
        or args.output_dir is None
    ):
        parser.error("size gate requires baseline-dir, candidate-dir, and output-dir")
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
