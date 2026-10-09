"""Run decisive support rows before optional read rows in one allocation."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep


def stop_owned_tree(process):
    """Enumerate the measurement's descendants before signalling those PIDs."""
    listing = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,etime=,args="], text=True
    )
    records = {}
    for line in listing.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            records[int(fields[0])] = (int(fields[1]), line.strip())
    owned = {process.pid}
    while True:
        added = {pid for pid, (parent, _) in records.items() if parent in owned}
        if added <= owned:
            break
        owned |= added
    print(
        "STOP_RADIUS measurement process and enumerated descendants only; "
        "other jobs and services excluded.",
        flush=True,
    )
    for pid in sorted(owned):
        print("STOP_PID " + records.get(pid, (0, str(pid)))[1], flush=True)
    for pid in sorted(owned, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for pid in sorted(owned, reverse=True):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        print(f"STOP_SURVIVOR pid={pid} signal=SIGKILL", flush=True)
        os.kill(pid, signal.SIGKILL)
    process.wait()


def run_row(args, kind, cells, arms):
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-u",
        str(root / "scripts/prototypes/support/measure.py"),
        "--kind",
        kind,
        "--cells",
        str(cells),
        "--out",
        str(args.output / "rows"),
        "--cache",
        str(args.cache),
        "--control",
        str(args.control),
        "--oracle-fraction-bound",
        "1e-4",
        "--arms",
        *arms,
    ]
    label = "-".join(arms)
    log = (
        args.output
        / "logs"
        / f"{kind}-{cells}-{label}-{os.environ['SLURM_JOB_ID']}.log"
    )
    machine_started = None
    machine_done = False
    position = 0
    timed_out = False
    with log.open("w") as stream:
        stream.write(
            f"REVISION={os.environ['NOVA_MEASUREMENT_REVISION']} TREE={root} COMMAND="
            + " ".join(command)
            + "\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        with log.open() as reader:
            while process.poll() is None:
                reader.seek(position)
                for line in reader:
                    if line.startswith("STAGE_START machine"):
                        machine_started = monotonic()
                        machine_done = False
                    if line.startswith("STAGE_DONE machine"):
                        machine_done = True
                position = reader.tell()
                if (
                    machine_started is not None
                    and not machine_done
                    and monotonic() - machine_started >= 3600
                ):
                    timed_out = True
                    stop_owned_tree(process)
                    break
                sleep(2)
        code = 124 if timed_out else process.returncode
        stream.write(f"EXIT={code}\n")
    print(
        f"ROW case={kind} cells={cells} arms={label} EXIT={code} log={log}", flush=True
    )
    if timed_out:
        binder = dict(
            case=kind,
            requested_cells=cells,
            stage="machine",
            wall_seconds=monotonic() - machine_started,
            budget_seconds=3600,
            log=str(log),
        )
        (args.output / f"{kind}-{cells}-binder.json").write_text(
            json.dumps(binder, indent=2) + "\n"
        )
        print("MACHINE_BINDER " + json.dumps(binder), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("control", type=Path)
    args = parser.parse_args()
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    (args.output / "rows").mkdir(exist_ok=True)
    failures = []
    for kind, cells, arms in (
        ("diverted", 5000, ("exact",)),
        ("limited", 550, ("legacy", "exact")),
        ("limited", 5000, ("legacy", "exact")),
    ):
        code = run_row(args, kind, cells, arms)
        if code:
            failures.append((kind, cells, code))
    print("MEASUREMENT_FAILURES=" + json.dumps(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
