"""Assert that a profiled Newton call actually stays on the device."""

import gzip
import json
import os
from pathlib import Path


def test_compiled_newton_keeps_loop_predicates_on_device():
    profile = Path(os.environ["FIXED_POINT_PROFILE"])
    paths = list(profile.glob("plugins/profile/*/*.trace.json.gz"))
    assert len(paths) == 1, "name one call's profile directory"
    events = json.loads(gzip.decompress(paths[0].read_bytes()))["traceEvents"]
    gpu = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name"
        and "GPU" in event.get("args", {}).get("name", "")
    }
    device = [
        event for event in events if event.get("ph") == "X" and event.get("pid") in gpu
    ]
    assert device, "a missing device trace cannot establish absence of transfers"
    copies = [event for event in device if event.get("name") == "MemcpyD2H"]
    assert not copies, f"device loop copied to host: {copies}"
