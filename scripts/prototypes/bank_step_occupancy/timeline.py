"""Summarise device activity within one warm Newton call."""

import argparse
import gzip
import json
from pathlib import Path


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
    kernels = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("pid") in devices
            and not event.get("name", "").startswith(("Memcpy", "Memset"))
            and start <= event["ts"] < end
        ),
        key=lambda event: event["ts"],
    )
    assert kernels, "trace has no GPU kernels; cannot infer idle time"
    intervals = []
    for event in kernels:
        left, right = event["ts"], event["ts"] + event["dur"]
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    busy = sum(right - left for left, right in intervals)
    gaps = []
    for earlier, later in zip(kernels, kernels[1:], strict=False):
        gap = later["ts"] - (earlier["ts"] + earlier["dur"])
        if gap > 0:
            gaps.append((gap, earlier, later))
    gaps.sort(key=lambda row: row[0], reverse=True)
    top_kernels = sorted(kernels, key=lambda event: event["dur"], reverse=True)[:3]
    host = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") not in devices
    ]

    def host_after(timestamp):
        following = [event for event in host if timestamp <= event["ts"] < end]
        if not following:
            return "none inside compiled dispatch"
        event = min(following, key=lambda item: item["ts"])
        return event.get("args", {}).get("long_name", event.get("name", "unknown"))

    print(
        "TIMELINE "
        + json.dumps(
            {
                "wall_ms": measured_wall,
                "gpu_busy_ms": busy / 1000,
                "busy_fraction": busy / (1000 * measured_wall),
                "kernels": len(kernels),
                "gpu_devices": sorted(map(str, devices)),
                "window": "annotation" if spans else "single-step trace",
                "idle_gaps": [
                    {
                        "ms": gap / 1000,
                        "preceding_kernel": earlier["name"],
                        "following_kernel": later["name"],
                        "following_host_call": host_after(
                            earlier["ts"] + earlier["dur"]
                        ),
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
