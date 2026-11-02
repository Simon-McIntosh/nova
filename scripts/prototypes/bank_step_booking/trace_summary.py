"""Count launches in a trace that contains one warmed device invocation."""

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
    gpu = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("pid") in devices
    ]
    kernels = [
        event
        for event in gpu
        if not event.get("name", "").startswith(("Memcpy", "Memset"))
    ]
    print(
        f"TRACE_EVENTS total={len(events)} gpu_events={len(gpu)} "
        f"kernels={len(kernels)} devices={len(devices)}"
    )
    assert len(devices) == 1 and kernels
    print(
        f"DEVICE_DURATION_MS={sum(event['dur'] for event in gpu) / 1000:.6f} "
        f"KERNEL_LAUNCHES={len(kernels)} COPIES={len(gpu) - len(kernels)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    main(parser.parse_args().trace)
