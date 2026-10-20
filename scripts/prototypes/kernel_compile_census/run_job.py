"""Run the census as isolated rows within one scheduler allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROGRAMS = (
    "harmonic_multiply",
    "harmonic_moments",
    "harmonic_pole_moments",
    "packed_analytic_moments",
    "point_jet",
    "production_point_batch",
    "prototype_point_batch",
    "topology_read",
    "certificate_map",
)
KERNELS = frozenset(PROGRAMS[:5])


def resident_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def run_one(worker, output, name, size, limit):
    prefix = f"{name}-{size}"
    receipt = output / f"{prefix}.json"
    log = output / f"{prefix}.log"
    command = [
        sys.executable,
        "-u",
        str(worker),
        "--program",
        name,
        "--size",
        str(size),
        "--output",
        str(receipt),
    ]
    environment = dict(os.environ)
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    started = time.monotonic()
    samples = []
    with log.open("w") as stream:
        stream.write(f"# command: {json.dumps(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment
        )
        while process.poll() is None and time.monotonic() - started < limit:
            samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "resident_kib": resident_kib(process.pid),
                }
            )
            time.sleep(10)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stream.write(f"EXIT={process.returncode}\n")
    if receipt.exists():
        record = json.loads(receipt.read_text())
    else:
        record = {"program": name, "size": size, "phase": "no-receipt"}
    record["sampled_peak_resident_kib"] = max(
        (sample["resident_kib"] for sample in samples), default=0
    )
    record["sample_interval_seconds"] = 10
    record["samples"] = samples
    record["process_exit"] = process.returncode
    record["elapsed_seconds"] = time.monotonic() - started
    record["log"] = str(log)
    receipt.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(
        "CENSUS_ROW "
        + json.dumps(
            {
                key: record.get(key)
                for key in (
                    "program",
                    "size",
                    "phase",
                    "process_exit",
                    "elapsed_seconds",
                    "sampled_peak_resident_kib",
                    "error",
                )
            }
        ),
        flush=True,
    )
    return process.returncode == 0 and record.get("phase") == "compiled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, default=10800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in PROGRAMS:
        for size in (4, 64) if name in KERNELS else (132, 550):
            passed = (
                run_one(args.worker, args.output, name, size, args.row_limit) and passed
            )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
