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
        if event.get("ph") == "X" and event.get("name", "").startswith("stage:")
    ]
    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") in devices
    ]
    print(f"TRACE_EVENTS total={len(events)} gpu={len(kernels)} stages={len(stages)}")
    assert stages and kernels
    for stage in stages:
        first = stage["ts"]
        last = first + stage["dur"]
        owned = [item for item in kernels if first <= item["ts"] <= last]
        print(
            f"STAGE_DEVICE name={stage['name'][6:]} "
            f"wall_ms={stage['dur'] / 1000:.6f} "
            f"device_ms={sum(item['dur'] for item in owned) / 1000:.6f} "
            f"launches={len(owned)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    main(parser.parse_args().trace)
