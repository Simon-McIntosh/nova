"""Validate durable support receipts independently of scheduler exit status."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd()} COMMAND={sys.executable} "
        f"{Path(__file__).resolve()} {args.run}",
        flush=True,
    )
    failures, coverage = [], []
    for case in ("diverted", "limited"):
        for requested in (550, 2000, 5000):
            cells = (
                3500
                if requested == 5000
                and (args.run / f"{case}-5000-binder.json").exists()
                else requested
            )
            for arm in ("legacy", "exact"):
                path = args.run / "rows" / f"{case}-{cells}-{arm}.json"
                row = json.loads(path.read_text()) if path.exists() else {}
                valid = row.get("completed") is True
                if valid:
                    for key in (
                        "map_relative_sup",
                        "map_relative_rms",
                        "booked_current_normalised_a",
                        "analytic_current_a",
                    ):
                        valid &= math.isfinite(row[key]) and row[key] > 0
                    size = row.get("serialized_executable_bytes")
                    valid &= (isinstance(size, int) and size > 0) or (
                        size is None
                        and row.get("serialization_status") == "unavailable"
                        and "size must be smaller than 2GiB"
                        in row.get("serialization_refusal", "")
                    )
                if not valid:
                    failures.append(f"missing-core-receipt::{case}-{cells}-{arm}")
                coverage.append(
                    dict(
                        case=case, cells=cells, arm=arm, completed=valid, path=str(path)
                    )
                )
    coarse = json.loads((args.run / "rows/diverted-550-legacy.json").read_text())
    exact = json.loads((args.run / "rows/diverted-550-exact.json").read_text())
    shifted = json.loads((args.run / "rows/diverted-550-shifted.json").read_text())
    guard = (args.run.parent / "fraction-guard-committed.log").read_text()
    discrepancies = list(
        map(float, re.findall(r"READ_POLYGON_FRACTION_ERROR=([0-9.e+-]+)", guard))
    )
    controls = {
        "legacy": coarse["positive_control_relative_delta"] <= 1e-9,
        "shifted": shifted["map_relative_sup"] > exact["map_relative_sup"],
        "fraction": len(discrepancies) == 2
        and discrepancies[0] < 2e-5
        and discrepancies[1] > 0.49
        and "GUARD_REFUSAL=read polygons lost fragment membership:" in guard
        and "GUARD_COMPLETE=passed" in guard,
    }
    failures.extend(f"control::{key}" for key, passed in controls.items() if not passed)
    refusals = [
        json.loads(path.read_text())
        for path in sorted((args.run / "rows").glob("*-read-refusal.json"))
    ]
    serialization = []
    for path in sorted((args.run / "logs").glob("*.log")):
        log = path.read_text()
        case = cells = arm = None
        observed = set()
        for line in log.splitlines():
            header = re.search(r" CASE=(\w+) CELLS=(\d+) ARMS=", line)
            if header:
                case, cells = header.groups()
            booking = re.search(
                r"STAGE_DONE (legacy|exact|read)-booking seconds=", line
            )
            if booking:
                arm = booking.group(1)
            refusal = re.search(r"size must be smaller than 2GiB: ([0-9]+)", line)
            if refusal and case and cells and arm:
                size = int(refusal.group(1))
                key = case, cells, arm, size
                if key not in observed:
                    observed.add(key)
                    serialization.append(
                        dict(
                            log=str(path),
                            case=case,
                            cells=int(cells),
                            arm=arm,
                            serializer_reported_proto_bytes=size,
                        )
                    )
    result = dict(
        revision=revision,
        completed=True,
        expected_core_rows=len(coverage),
        completed_core_rows=sum(row["completed"] for row in coverage),
        coverage=coverage,
        controls=controls,
        read_attempt_refusals=len(refusals),
        distinct_read_refusals=len({row["signature"] for row in refusals}),
        serialization_refusals=serialization,
        failures=failures,
        exit_status=int(bool(failures)),
    )
    (args.run / "receipt-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    for failure in failures:
        print("FAILED " + failure)
    print(
        f"RECEIPT_AUDIT core={result['completed_core_rows']}/{len(coverage)} "
        f"controls={sum(controls.values())}/{len(controls)} "
        f"read_refusals={len(refusals)} distinct={result['distinct_read_refusals']}"
    )
    print(f"EXIT={result['exit_status']}")
    return result["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
