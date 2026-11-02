"""Table the largest kernels inside one traced stage annotation."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

STAGE = "stage:local_booking_tangent"


def main(trace: Path, limit: int):
    with gzip.open(trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
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
        if event.get("ph") == "X"
        and event.get("args", {}).get("long_name") == STAGE
    )
    first, last = stage["ts"], stage["ts"] + stage["dur"]
    mine = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") in devices
        and first <= event["ts"] <= last
    ]
    kernels = [
        event
        for event in mine
        if not event.get("name", "").startswith(("Memcpy", "dur", "Memset"))
    ]
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
    print(f"STAGE_DEVICE name=local_booking_tangent device_ms={total:.6f} launches={len(kernels)}")
    ranked = sorted(families.items(), key=lambda item: -item[1][0 + 1])
    for name, (count, milliseconds) in ranked[:limit]:
        share = 100.0 * milliseconds / total if total else 0.0
        print(f"KERNEL {name[:96]!r} launches={count} device_ms={milliseconds:.4f} share={share:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    arguments = parser.parse_args()
    main(arguments.trace, arguments.limit)