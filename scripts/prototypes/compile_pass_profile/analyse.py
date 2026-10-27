"""Attribute an XLA compile wall to its per-pass HLO dumps.

XLA writes one dump per pass and restarts its dump index inside each pipeline,
so the order that carries the wall is the order the files were written. The gap
between one dump and the next is the cost of the pass that ran between them.
The chain is bounded by the epoch times the profiling process took around
``Lowered.compile()``, which is what lets the profile say how much of the wall
it did not see.
"""

import argparse
import json
from pathlib import Path
import re

MODULE = re.compile(r"^module_\d+$")
STAMP = re.compile(r"^\d{15,}$")
INDEX = re.compile(r"^\d+$")
INSTRUCTION = re.compile(r"^\s+(?:ROOT\s+)?[\w.%/-]+\s*=")
SKIP = (
    "metadata=",
    "frontend_attributes=",
    "sharding=",
    "replica_groups=",
    "is_scheduled=",
    "backend_config=",
)
TOLERANCE = 1.02


def describe(path):
    """Return (module, pass label, dump index) for one dump file."""
    parts = path.name.split(".")
    if parts and STAMP.match(parts[0]):
        parts = parts[1:]
    module = None
    if parts and MODULE.match(parts[0]):
        module = parts[0]
        parts = parts[1:]
    index = next((int(part) for part in parts if INDEX.match(part)), None)
    body = [part for part in parts if not INDEX.match(part)]
    if body:
        body[-1] = Path(body[-1]).stem
    marker = next(
        (position for position, part in enumerate(body) if part.startswith("before_")),
        None,
    )
    if marker is not None and marker > 0:
        label = body[marker - 1] + "/" + body[marker][len("before_") :]
    elif marker == 0:
        label = body[marker][len("before_") :]
    else:
        label = ".".join(body)
    return module, label or path.name, index


def count_instructions(path):
    total = 0
    fused = 0
    try:
        with path.open("r", errors="replace") as stream:
            for line in stream:
                stripped = line.rstrip()
                if stripped.endswith(") {"):
                    continue
                if any(token in line for token in SKIP):
                    continue
                if INSTRUCTION.match(line):
                    total += 1
                    if "fusion(" in line:
                        fused += 1
    except OSError:
        return None, None
    return total, fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())

    rows = []
    for path in args.dumps.iterdir():
        if not path.is_file() or not path.name.endswith((".txt", ".ll")):
            continue
        module, label, index = describe(path)
        rows.append(
            {
                "module": module,
                "pass": label,
                "index": index,
                "stamp": path.stat().st_mtime_ns,
                "bytes": path.stat().st_size,
                "path": path,
            }
        )
    if not rows:
        raise SystemExit("no dumps found")

    modules = {}
    for row in rows:
        modules.setdefault(row["module"], []).append(row)
    _, chosen = max(modules.items(), key=lambda item: len(item[1]))
    chosen.sort(key=lambda row: (row["stamp"], row["path"].name))
    base = chosen[0]["stamp"]
    for position, row in enumerate(chosen):
        row["start"] = (row["stamp"] - base) / 1e9
        following = (
            (chosen[position + 1]["stamp"] - base) / 1e9
            if position + 1 < len(chosen)
            else row["start"]
        )
        row["seconds"] = max(0.0, following - row["start"])
        row["instructions"], row["fusion_instructions"] = count_instructions(
            row["path"]
        )

    attributed = sum(row["seconds"] for row in chosen)
    grouped = {}
    for row in chosen:
        entry = grouped.setdefault(
            row["pass"], {"seconds": 0.0, "dumps": 0, "instructions": []}
        )
        entry["seconds"] += row["seconds"]
        entry["dumps"] += 1
        if row["instructions"] is not None:
            entry["instructions"].append(row["instructions"])
    table = [
        {
            "pass": label,
            "seconds": round(entry["seconds"], 4),
            "share_of_attributed": round(entry["seconds"] / attributed, 4)
            if attributed
            else 0.0,
            "dumps": entry["dumps"],
            "instructions_entry": entry["instructions"][0]
            if entry["instructions"]
            else None,
            "instructions_exit": entry["instructions"][-1]
            if entry["instructions"]
            else None,
        }
        for label, entry in sorted(
            grouped.items(), key=lambda item: -item[1]["seconds"]
        )
    ]

    compile_seconds = receipt.get("compile_seconds")
    started = receipt.get("compile_started_epoch")
    finished = receipt.get("compile_finished_epoch")
    before_first = chosen[0]["stamp"] / 1e9 - started if started else None
    after_last = finished - chosen[-1]["stamp"] / 1e9 if finished else None
    coverage = attributed / compile_seconds if compile_seconds else None
    summary = {
        "program": receipt["program"],
        "size": receipt["size"],
        "device": receipt["device"],
        "revision": receipt["revision"],
        "nova_file": receipt.get("nova_file"),
        "trace_seconds": receipt.get("trace_seconds"),
        "lower_seconds": receipt.get("lower_seconds"),
        "compile_seconds": compile_seconds,
        "stablehlo_bytes": receipt.get("stablehlo_bytes"),
        "executable_bytes": receipt.get("executable_bytes"),
        "dump_count": len(chosen),
        "distinct_passes": len(grouped),
        "attributed_seconds": round(attributed, 4),
        "seconds_before_first_dump": round(before_first, 4)
        if before_first is not None
        else None,
        "seconds_after_last_dump": round(after_last, 4)
        if after_last is not None
        else None,
        "coverage_of_compile": round(coverage, 4) if coverage is not None else None,
        "instructions_first": chosen[0]["instructions"],
        "instructions_peak": max(
            (row["instructions"] or 0)
            for row in chosen
            if row["instructions"] is not None
        ),
        "table": table,
        "sequence": [
            {
                "index": index,
                "pass": row["pass"],
                "start": round(row["start"], 4),
                "seconds": round(row["seconds"], 4),
                "instructions": row["instructions"],
                "bytes": row["bytes"],
            }
            for index, row in enumerate(chosen)
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "ATTRIBUTION "
        + json.dumps(
            {
                key: summary[key]
                for key in (
                    "program",
                    "size",
                    "compile_seconds",
                    "attributed_seconds",
                    "coverage_of_compile",
                    "seconds_before_first_dump",
                    "seconds_after_last_dump",
                    "distinct_passes",
                    "dump_count",
                    "instructions_first",
                    "instructions_peak",
                )
            }
        ),
        flush=True,
    )
    for entry in table[:12]:
        print(
            "PASS %8.3f s %5.1f%% %4d dumps %s"
            % (
                entry["seconds"],
                100 * entry["share_of_attributed"],
                entry["dumps"],
                entry["pass"],
            ),
            flush=True,
        )

    if coverage is None:
        print("ATTRIBUTION-FAIL the receipt records no compile wall", flush=True)
        raise SystemExit(2)
    if coverage > 1.0 or attributed > compile_seconds * TOLERANCE:
        print(
            "ATTRIBUTION-FAIL %.3f s attributed against a %.3f s compile wall "
            "(coverage %.3f): the dump chain is longer than the compile it "
            "describes; the dumps are not in write order."
            % (attributed, compile_seconds, coverage),
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
