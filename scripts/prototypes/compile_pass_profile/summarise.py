"""Re-derive per-pass time in write order and check it against the compile wall.

The receipt keeps one row per dump with its write time and instruction count, so
this stage re-analyses without the dumps. Every quantity it emits comes from the
same reordered rows: the per-pass table, the per-dump gaps and the time before
the first dump and after the last. Those three must add up to the compile wall
the profiling process measured, and this stage refuses a receipt where they do
not.
"""

import argparse
import json
from pathlib import Path

TOLERANCE = 0.1


def summarise(receipt):
    sequence = sorted(receipt["sequence"], key=lambda row: row["start"])
    span = sequence[-1]["start"] - sequence[0]["start"]
    grouped = {}
    gaps = []
    for index, row in enumerate(sequence):
        end = (
            sequence[index + 1]["start"] if index + 1 < len(sequence) else row["start"]
        )
        seconds = max(0.0, end - row["start"])
        gaps.append(
            {
                "index": index,
                "pass": row["pass"],
                "start": row["start"],
                "seconds": round(seconds, 4),
                "instructions": row["instructions"],
                "bytes": row["bytes"],
            }
        )
        entry = grouped.setdefault(
            row["pass"], {"seconds": 0.0, "dumps": 0, "instructions": []}
        )
        entry["seconds"] += seconds
        entry["dumps"] += 1
        if row["instructions"] is not None:
            entry["instructions"].append(row["instructions"])
    for index, gap in enumerate(gaps):
        following = gaps[index + 1] if index + 1 < len(gaps) else None
        gap["instructions_after"] = following["instructions"] if following else None
        gap["next_pass"] = following["pass"] if following else None
    last_index = {gap["pass"]: index for index, gap in enumerate(gaps)}
    table = [
        {
            "pass": label,
            "seconds": round(entry["seconds"], 3),
            "share_of_attributed": round(entry["seconds"] / span, 4) if span else 0.0,
            "dumps": entry["dumps"],
            "instructions_entry": entry["instructions"][0]
            if entry["instructions"]
            else None,
            "instructions_exit": entry["instructions"][-1]
            if entry["instructions"]
            else None,
            "instructions_after": gaps[last_index[label]]["instructions_after"],
            "next_pass": gaps[last_index[label]]["next_pass"],
        }
        for label, entry in sorted(
            grouped.items(), key=lambda item: -item[1]["seconds"]
        )
    ]
    return {
        "program": receipt["program"],
        "size": receipt["size"],
        "device": receipt["device"],
        "revision": receipt["revision"],
        "nova_file": receipt.get("nova_file"),
        "trace_seconds": receipt.get("trace_seconds"),
        "lower_seconds": receipt.get("lower_seconds"),
        "compile_seconds": receipt.get("compile_seconds"),
        "stablehlo_bytes": receipt.get("stablehlo_bytes"),
        "executable_bytes": receipt.get("executable_bytes"),
        "dump_count": len(sequence),
        "distinct_passes": len(grouped),
        "attributed_seconds": round(sum(gap["seconds"] for gap in gaps), 4),
        "chain_span_seconds": round(span, 4),
        "seconds_before_first_dump": receipt.get("seconds_before_first_dump"),
        "seconds_after_last_dump": receipt.get("seconds_after_last_dump"),
        "coverage_of_compile": round(span / receipt["compile_seconds"], 4)
        if receipt.get("compile_seconds")
        else None,
        "instructions_first": sequence[0]["instructions"],
        "instructions_peak": max(
            (row["instructions"] or 0)
            for row in sequence
            if row["instructions"] is not None
        ),
        "table": table,
        "gaps": gaps,
        "largest_gaps": sorted(gaps, key=lambda gap: -gap["seconds"])[:12],
    }


def reconcile(profile):
    wall = profile["compile_seconds"]
    total = (
        (profile["seconds_before_first_dump"] or 0.0)
        + profile["attributed_seconds"]
        + (profile["seconds_after_last_dump"] or 0.0)
    )
    residual = total - wall
    profile["reconciled_seconds"] = round(total, 4)
    profile["reconciliation_residual_seconds"] = round(residual, 4)
    if abs(residual) > TOLERANCE:
        raise SystemExit(
            "RECONCILE-FAIL %s-%s: %.3f s before, %.3f s attributed, %.3f s after "
            "sum to %.3f s against a %.3f s compile wall (residual %+.3f s, "
            "tolerance %.1f s)"
            % (
                profile["program"],
                profile["size"],
                profile["seconds_before_first_dump"] or 0.0,
                profile["attributed_seconds"],
                profile["seconds_after_last_dump"] or 0.0,
                total,
                wall,
                residual,
                TOLERANCE,
            )
        )
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    for path in args.receipt:
        receipt = json.loads(path.read_text())
        profile = reconcile(summarise(receipt))
        profiles[receipt["program"] + "-" + str(receipt["size"])] = profile
        print(
            "%s compile=%.2fs attributed=%.2fs before=%.2fs after=%.2fs "
            "residual=%+.3fs dumps=%d"
            % (
                receipt["program"],
                profile["compile_seconds"],
                profile["attributed_seconds"],
                profile["seconds_before_first_dump"] or 0.0,
                profile["seconds_after_last_dump"] or 0.0,
                profile["reconciliation_residual_seconds"],
                profile["dump_count"],
            )
        )
    args.output.write_text(json.dumps(profiles, indent=2) + "\n")


if __name__ == "__main__":
    main()
