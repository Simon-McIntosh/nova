"""Count GPU kernel events and host CUDA graph launches in profiler traces."""

import argparse
import gzip
import json
from pathlib import Path


def count_trace(directory: Path, module: str):
    traces = list(directory.rglob("*.trace.json.gz"))
    if len(traces) != 1:
        raise ValueError(
            f"expected one profiler trace under {directory}, got {len(traces)}"
        )
    with gzip.open(traces[0], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    processes = {
        item["pid"]: item["args"]["name"]
        for item in events
        if item.get("ph") == "M" and item.get("name") == "process_name"
    }
    gpu_ids = {
        pid for pid, name in processes.items() if name.startswith("/device:GPU:")
    }
    cpu_ids = {pid for pid, name in processes.items() if name == "/host:CPU"}
    if not gpu_ids or not cpu_ids:
        raise ValueError("profiler contains no named GPU and CPU processes")
    gpu = [
        item for item in events if item.get("ph") == "X" and item.get("pid") in gpu_ids
    ]
    host = [
        item for item in events if item.get("ph") == "X" and item.get("pid") in cpu_ids
    ]
    matching = [
        item for item in gpu if item.get("args", {}).get("hlo_module") == module
    ]
    kernels = [item for item in matching if "kernel_details" in item.get("args", {})]
    copies = [item for item in matching if item["name"].startswith("Memcpy")]
    graph_launches = [item for item in host if item["name"].startswith("cuGraphLaunch")]
    if not matching or not kernels or not graph_launches:
        raise ValueError(
            "trace did not observe the known map, a GPU kernel, and a host graph launch"
        )
    return {
        "trace_path": str(traces[0]),
        "gpu_events": len(matching),
        "gpu_kernel_events": len(kernels),
        "gpu_copies": len(copies),
        "host_graph_launches": len(graph_launches),
        "positive_module": module,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("limited", type=Path)
    parser.add_argument("diverted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = {
        "limited": count_trace(args.limited, "jit_mapped"),
        "diverted": count_trace(args.diverted, "jit_mapped"),
    }
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("RESULT " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
