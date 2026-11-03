"""Summarise device activity within one warm Newton call."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re


def main(path: Path, arm_log: Path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    trace_rows = [
        json.loads(line.partition(" ")[2])
        for line in arm_log.read_text().splitlines()
        if line.startswith("TRACE ")
    ]
    assert len(trace_rows) == 1
    measured_wall = trace_rows[0]["step_wall_ms"]
    devices = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and "/device:GPU" in event.get("args", {}).get("name", "")
    }
    spans = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("args", {}).get("long_name") == "whole_warm_step"
    ]
    if spans:
        assert len(spans) == 1
        span = spans[0]
        start, end = span["ts"], span["ts"] + span["dur"]
    else:
        start = min(event["ts"] for event in events if event.get("ph") == "X")
        end = max(
            event["ts"] + event["dur"] for event in events if event.get("ph") == "X"
        )
    device_events = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("pid") in devices
            and start <= event["ts"] < end
        ),
        key=lambda event: event["ts"],
    )
    kernels = [
        event
        for event in device_events
        if not event.get("name", "").startswith(("Memcpy", "Memset"))
    ]
    assert kernels, "trace has no GPU kernels; cannot infer idle time"
    intervals = []
    for event in device_events:
        left, right = event["ts"], event["ts"] + event["dur"]
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    busy = sum(right - left for left, right in intervals)
    gaps = []
    for earlier, later in zip(intervals, intervals[1:], strict=False):
        gap = later[0] - earlier[1]
        if gap > 0:
            gaps.append((gap, earlier, later))
    gaps.sort(key=lambda row: row[0], reverse=True)
    top_kernels = sorted(kernels, key=lambda event: event["dur"], reverse=True)[:3]
    host = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") not in devices
    ]

    def host_after(start_time, end_time):
        following = [event for event in host if start_time <= event["ts"] < end_time]
        if not following:
            return "none in gap"
        event = min(following, key=lambda item: item["ts"])
        return event.get("args", {}).get("long_name", event.get("name", "unknown"))

    def adjacent_kernel(timestamp, direction):
        if direction == "before":
            eligible = [event for event in kernels if event["ts"] < timestamp]
            return (
                max(eligible, key=lambda item: item["ts"])["name"]
                if eligible
                else "none"
            )
        eligible = [event for event in kernels if event["ts"] >= timestamp]
        return (
            min(eligible, key=lambda item: item["ts"])["name"] if eligible else "none"
        )

    copies = [event for event in device_events if event["name"] == "MemcpyD2H"]
    copy_bytes = sum(
        int(match.group(1))
        for event in copies
        if (
            match := re.search(
                r"size:(\d+)", event.get("args", {}).get("memcpy_details", "")
            )
        )
    )

    print(
        "TIMELINE "
        + json.dumps(
            {
                "wall_ms": measured_wall,
                "gpu_busy_ms": busy / 1000,
                "busy_fraction": busy / (1000 * measured_wall),
                "kernels": len(kernels),
                "device_to_host_copies": len(copies),
                "device_to_host_bytes": copy_bytes,
                "device_to_host_hlo": dict(
                    Counter(
                        event.get("args", {}).get("hlo_op", "unknown")
                        for event in copies
                    )
                ),
                "gpu_devices": sorted(map(str, devices)),
                "window": "annotation" if spans else "single-step trace",
                "idle_gaps": [
                    {
                        "ms": gap / 1000,
                        "preceding_kernel": adjacent_kernel(earlier[1], "before"),
                        "following_kernel": adjacent_kernel(later[0], "after"),
                        "following_host_call": host_after(earlier[1], later[0]),
                    }
                    for gap, earlier, later in gaps[:5]
                ],
                "largest_kernels": [
                    {"name": event["name"], "ms": event["dur"] / 1000}
                    for event in top_kernels
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--arm-log", type=Path, required=True)
    args = parser.parse_args()
    main(args.trace, args.arm_log)
