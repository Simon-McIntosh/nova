"""Attribute a warmed booking trace to GPU kernel families."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def family(name):
    if "radix_sort" in name:
        return "segmented radix sort"
    if "dgemm" in name or "gemm" in name:
        return "matrix multiply"
    if "reduce" in name:
        return "reductions"
    if name.startswith(("Memcpy", "Memset")):
        return "copy and set"
    return "other kernels"


def main(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    devices = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "/device:GPU" in event.get("args", {}).get("name", "")
    }
    stages = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("args", {}).get("long_name") == "stage:local_booking_tangent"
    ]
    assert len(devices) == 1 and len(stages) == 1
    stage = stages[0]
    gpu = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") in devices
        and stage["ts"] <= event["ts"] <= stage["ts"] + stage["dur"]
    ]
    assert gpu
    groups = defaultdict(lambda: [0, 0.0])
    for event in gpu:
        group = groups[family(event["name"])]
        group[0] += 1
        group[1] += event["dur"] / 1000
    print("POSITIVE_CONTROL", "gpu_events", len(gpu), "stage_ms", stage["dur"] / 1000)
    for name, (count, duration) in sorted(groups.items(), key=lambda item: -item[1][1]):
        print("KERNEL_FAMILY", name, "count", count, "device_ms", round(duration, 6))
    for event in sorted(gpu, key=lambda item: -item["dur"])[:8]:
        print(
            "KERNEL_TOP",
            "duration_ms",
            round(event["dur"] / 1000, 6),
            "name",
            event["name"],
            "long_name",
            event.get("args", {}).get("long_name", ""),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    main(parser.parse_args().trace)
