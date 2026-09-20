"""Run a driver under a sampler that records its process mapping count until it dies.

A CPU compile that runs out of memory *mappings* aborts inside LLVM with
``Unable to allocate section memory!`` and leaves no trace of how close the
process came to the kernel's ``vm.max_map_count``.  This probe supplies that
trace: it starts the named driver as a child, then samples the child's own
``/proc`` entry once per interval, writing each sample to disk as it is taken so
an abort loses the last sample rather than the series.

The sampled quantity is the line count of ``/proc/<pid>/maps``, which is the
number of virtual memory areas the kernel holds for the process and the quantity
``vm.max_map_count`` bounds.  ``VmRSS`` and ``VmSize`` are read from
``/proc/<pid>/status`` at the same instant, so a section-memory abort can be read
against resident memory rather than against the requested allocation.

The sampler runs in this probe's own process, outside the driver, so it keeps
sampling while the driver holds the GIL through a long compile.  The driver is
started with no shell, so the pid recorded here is the payload's own.

Usage::

    python -m benchmarks.compile_abort_probe \
        --samples out/jsonl --summary out/json --interval 1.0 \
        -- /path/to/python -m benchmarks.some_driver --flag value

Everything after ``--`` is the driver command, executed verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


STATUS_KEYS = ("VmRSS", "VmSize", "VmPeak", "VmHWM")


def _read_status(pid: int) -> dict[str, int]:
    """Return the byte-valued Vm* fields of one process's /proc status."""
    reading: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as status:
            for line in status:
                name, _, rest = line.partition(":")
                if name in STATUS_KEYS:
                    reading[name] = int(rest.split()[0])
    except FileNotFoundError, ProcessLookupError, PermissionError:
        return {}
    return reading


def _count_maps(pid: int) -> int | None:
    """Count the virtual memory areas the kernel holds for one process."""
    count = 0
    try:
        with open(f"/proc/{pid}/maps", "rb") as maps:
            for _ in maps:
                count += 1
    except FileNotFoundError, ProcessLookupError, PermissionError:
        return None
    return count


def _read_map_limit() -> int | None:
    try:
        with open("/proc/sys/vm/max_map_count") as limit:
            return int(limit.read().strip())
    except OSError:
        return None


def sample_child(
    child: subprocess.Popen, interval: float, samples: Path
) -> dict[str, object]:
    """Poll one child's /proc entry until it exits; write every sample as taken.

    Liveness comes from the child handle rather than from ``/proc``: an exited
    child stops being sampled as soon as it is reaped, whereas an unreaped one
    stays visible as a zombie whose empty ``/proc/<pid>/maps`` reads as zero
    mappings rather than as an absent process, and a loop keyed on that reading
    never terminates.
    """
    pid = child.pid
    peak_maps = 0
    peak_sample: dict[str, object] | None = None
    last_sample: dict[str, object] | None = None
    count = 0
    started = time.monotonic()
    with open(samples, "w") as sink:
        while child.poll() is None:
            count += 1
            maps = _count_maps(pid)
            reading = _read_status(pid)
            if maps is None and not reading:
                break
            record: dict[str, object] = {
                "n": count,
                "t": round(time.monotonic() - started, 3),
                "maps": maps,
                **reading,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            last_sample = record
            if maps is not None and maps >= peak_maps:
                peak_maps = maps
                peak_sample = record
            time.sleep(interval)
    return {
        "samples_written": count,
        "peak_maps": peak_maps,
        "peak_sample": peak_sample,
        "last_sample": last_sample,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("no driver command given after --")

    args.samples.parent.mkdir(parents=True, exist_ok=True)
    map_limit = _read_map_limit()

    started = time.time()
    child = subprocess.Popen(command)
    print(f"PROBE_CHILD pid={child.pid} command={' '.join(command)}", flush=True)
    try:
        sampling = sample_child(child, args.interval, args.samples)
    except KeyboardInterrupt:
        child.kill()
        raise
    exit_code = child.wait()
    wall = time.time() - started

    summary: dict[str, object] = {
        "command": command,
        "pid": child.pid,
        "exit_code": exit_code,
        "wall_seconds": round(wall, 3),
        "interval_seconds": args.interval,
        "max_map_count": map_limit,
        "samples_path": str(args.samples),
        **sampling,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "PROBE_SUMMARY "
        f"exit={exit_code} peak_maps={summary['peak_maps']} "
        f"limit={map_limit} samples={summary['samples_written']} "
        f"wall={summary['wall_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
