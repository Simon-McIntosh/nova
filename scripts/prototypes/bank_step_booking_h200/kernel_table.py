"""Rank the fused kernels inside the booking tangent by device time.

The stage walk is the one ``bank_step_speed.parse_trace`` already performs: it
is called first and its own launch count for this stage is compared with the
count this module derives, so the ranking cannot drift from the summary the
profile logs carry.
"""

import argparse
import gzip
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

from scripts.prototypes.bank_step_speed import parse_trace

STAGE = "stage:local_booking_tangent"


def main(trace: Path, limit: int):
    captured = io.StringIO()
    with redirect_stdout(captured):
        parse_trace.main(trace)
    summary = captured.getvalue()
    print(summary.strip())
    walk = next(line for line in summary.splitlines() if f"name={STAGE[6:]}" in line)
    walking_launches = int(walk.split("launches=")[1].split()[0])

    kernels = tangent_kernels(load(trace))
    assert len(kernels) == walking_launches, (len(kernels), walking_launches)

    families = defaultdict(lambda: [0, 0.0])
    for event in kernels:
        name = event["name"]
        for suffix in (" (op)", "_kernel"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        entry = families[name]
        entry[0] += 1
        entry[1] += event["dur"] / 1000.0
    total = sum(value[1] for value in families.values())
    for name, (count, milliseconds) in sorted(
        families.items(), key=lambda item: -item[1][1]
    )[:limit]:
        share = 100.0 * milliseconds / total if total else 0.0
        print(
            f"KERNEL {name[:96]!r} launches={count} "
            f"device_ms={milliseconds:.4f} share={share:.1f}%"
        )


def open_trace(trace: Path):
    return gzip.open(trace, "rt")


def tangent_kernels(events):
    """The fused kernels recorded inside the booking tangent annotation."""
    devices = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "/device:GPU" in event.get("args", {}).get("name", "")
    }
    stage = next(
        event
        for event in events
        if event.get("ph") == "X" and event.get("args", {}).get("long_name") == STAGE
    )
    first, last = stage["ts"], stage["ts"] + stage["dur"]
    return [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") in devices
        and first <= event["ts"] <= last
        and not event.get("name", "").startswith(("Memcpy", "Memset"))
    ]


def load(trace: Path):
    with open_trace(trace) as stream:
        return json.load(stream)["traceEvents"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    arguments = parser.parse_args()
    main(arguments.trace, arguments.limit)
