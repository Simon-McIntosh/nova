"""Count GPU activity inside each warm stage annotation."""

import argparse
import gzip
import json
from pathlib import Path


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
        and event.get("args", {}).get("long_name", "").startswith("stage:")
    ]
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") in devices
    ]
    kernels = [
        event
        for event in gpu_events
        if not event.get("name", "").startswith(("Memcpy", "Memset"))
    ]
    print(
        f"TRACE_EVENTS total={len(events)} gpu_events={len(gpu_events)} "
        f"kernels={len(kernels)} stages={len(stages)}"
    )
    assert stages and kernels
    for stage in stages:
        first = stage["ts"]
        last = first + stage["dur"]
        owned_gpu = [item for item in gpu_events if first <= item["ts"] <= last]
        owned_kernels = [item for item in kernels if first <= item["ts"] <= last]
        print(
            f"STAGE_DEVICE name={stage['args']['long_name'][6:]} "
            f"wall_ms={stage['dur'] / 1000:.6f} "
            f"device_ms={sum(item['dur'] for item in owned_gpu) / 1000:.6f} "
            f"launches={len(owned_kernels)} "
            f"copies={len(owned_gpu) - len(owned_kernels)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    main(parser.parse_args().trace)
