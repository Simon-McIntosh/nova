"""Re-derive per-pass time in write order and emit the profile the figure reads.

XLA restarts its dump index within each pipeline, so the order that matters is
the order the files were written. Each receipt keeps one row per dump with its
write time and instruction count, so this stage re-analyses without the dumps.
"""

import argparse
import json
from pathlib import Path


def summarise(receipt):
    sequence = sorted(receipt["sequence"], key=lambda row: row["start"])
    span = sequence[-1]["start"] - sequence[0]["start"]
    grouped = {}
    for index, row in enumerate(sequence):
        end = sequence[index + 1]["start"] if index + 1 < len(sequence) else row["start"]
        seconds = max(0.0, end - row["start"])
        entry = grouped.setdefault(
            row["pass"], {"seconds": 0.0, "dumps": 0, "instructions": []}
        )
        entry["seconds"] += seconds
        entry["dumps"] += 1
        if row["instructions"] is not None:
            entry["instructions"].append(row["instructions"])
    table = [
        {
            "pass": name,
            "seconds": round(entry["seconds"], 3),
            "share_of_attributed": round(entry["seconds"] / span, 4) if span else 0.0,
            "dumps": entry["dumps"],
            "instructions_first": entry["instructions"][0]
            if entry["instructions"]
            else None,
            "instructions_last": entry["instructions"][-1]
            if entry["instructions"]
            else None,
        }
        for name, entry in sorted(
            grouped.items(), key=lambda item: -item[1]["seconds"]
        )
    ]
    peaks = sorted(
        (
            {
                "start": row["start"],
                "seconds": row["seconds"],
                "pass": row["pass"],
                "instructions": row["instructions"],
                "bytes": row["bytes"],
            }
            for row in sequence
        ),
        key=lambda row: -row["seconds"],
    )
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
        "attributed_seconds": round(span, 3),
        "coverage_of_compile": round(span / receipt["compile_seconds"], 4)
        if receipt.get("compile_seconds")
        else None,
        "seconds_before_first_dump": receipt.get("seconds_before_first_dump"),
        "seconds_after_last_dump": receipt.get("seconds_after_last_dump"),
        "instructions_first": sequence[0]["instructions"],
        "instructions_last": sequence[-1]["instructions"],
        "instructions_peak": max((row["instructions"] or 0) for row in sequence),
        "table": table,
        "gaps": peaks[:12],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    for path in args.receipt:
        receipt = json.loads(path.read_text())
        profiles[receipt["program"] + "-" + str(receipt["size"])] = summarise(receipt)
    args.output.write_text(json.dumps(profiles, indent=2) + "\n")
    for key, profile in profiles.items():
        print(
            "%s compile=%.2fs attributed=%.2fs coverage=%.3f dumps=%d"
            % (
                key,
                profile["compile_seconds"],
                profile["attributed_seconds"],
                profile["coverage_of_compile"],
                profile["dump_count"],
            )
        )
        for entry in profile["table"][:10]:
            print(
                "   %8.3fs %5.1f%% %s"
                % (
                    entry["seconds"],
                    100 * entry["share_of_attributed"],
                    entry["pass"],
                )
            )


if __name__ == "__main__":
    main()