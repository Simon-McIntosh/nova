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
    decisive_failures = []
    for kind in ("diverted", "limited"):
        code = run_row(args, kind, 2000, ("legacy", "exact"))
        if code:
            decisive_failures.append((kind, 2000, code))
    if run_row(args, "diverted", 550, ("shifted",)):
        decisive_failures.append(("diverted", 550, "shifted"))
    largest = {}
    for kind in ("diverted", "limited"):
        cells = 5000
        code = run_row(args, kind, cells, ("legacy", "exact"))
        if code == 124:
            cells = 3500
            code = run_row(args, kind, cells, ("legacy", "exact"))
        largest[kind] = cells
        if code:
            decisive_failures.append((kind, cells, code))
    # The accepted diverted coarse pair is supplied as an input receipt.
    # The limited coarse pair completes the other case's comparison ladder.
    if run_row(args, "limited", 550, ("legacy", "exact")):
        decisive_failures.append(("limited", 550, "legacy-exact"))
    print("DECISIVE_FAILURES=" + json.dumps(decisive_failures), flush=True)
    refusals = set()
    for cells in (550, 2000, 5000):
        for kind in ("diverted", "limited"):
            actual = largest[kind] if cells == 5000 else cells
            code = run_row(args, kind, actual, ("read",))
            if code:
                path = args.output / "rows" / f"{kind}-{actual}-read-refusal.json"
                record = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"signature": f"process-exit:{code}"}
                )
                refusals.add(record["signature"])
                print("READ_REFUSAL " + json.dumps(record), flush=True)
                if len(refusals) >= 3:
                    print(
                        "READ_STOP distinct_refusals=3; decisive receipts retained",
                        flush=True,
                    )
                    return 1 if decisive_failures else 0
    print(
        f"LADDER_COMPLETE decisive_failures={len(decisive_failures)} "
        f"distinct_read_refusals={len(refusals)}",
        flush=True,
    )
    return 1 if decisive_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
