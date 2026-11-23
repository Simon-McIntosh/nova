"""Persist all remaining lowerings, then isolate native and eager Hessian rows."""

import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import sys
import time


GIB = 1024**3
VARIANTS = ("current", "plain", "without_samples", "masked_samples")


def cap_address_space():
    # Leave room for the coordinator and file buffers in the 240 GiB job.
    resource.setrlimit(resource.RLIMIT_AS, (220 * GIB, 220 * GIB))


def descendants(root):
    result, pending = [], [root]
    while pending:
        pid = pending.pop()
        base = Path(f"/proc/{pid}")
        try:
            command = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            status = (base / "status").read_text()
            rss = next(
                (
                    int(line.split()[1])
                    for line in status.splitlines()
                    if line.startswith("VmRSS:")
                ),
                0,
            )
            children = (base / "task" / str(pid) / "children").read_text().split()
        except FileNotFoundError, ProcessLookupError:
            continue
        result.append((pid, command, rss))
        pending.extend(map(int, children))
    return result


def stop_child(process, reason):
    members = descendants(process.pid)
    selected = [row for row in members if row[0] != process.pid] or members
    print(
        "CHILD_LIMIT "
        + json.dumps(
            {
                "reason": reason,
                "members": selected,
                "survivors": (
                    "The allocation driver and other jobs are outside this "
                    "row's child tree; the time wrapper reaps its child."
                ),
            }
        ),
        flush=True,
    )
    subprocess.run(
        [
            "ps",
            "-o",
            "pid,ppid,etime,args",
            "-p",
            ",".join(str(row[0]) for row in selected),
        ],
        check=False,
    )
    for pid, _command, _rss in reversed(selected):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        for pid, command, rss in descendants(process.pid):
            if pid == process.pid:
                continue
            print(f"CHILD_KILL pid={pid} rss_kib={rss} command={command}", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print(f"CHILD_KILL time_wrapper_pid={process.pid}", flush=True)
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("measure.py").resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    deadline = time.monotonic() + 5 * 3600 + 10 * 60
    records = []

    def run(variant, order, action, timeout, extra=()):
        stem = f"{action}-derivative-{variant}-{order}"
        remaining = deadline - time.monotonic()
        if remaining < 120:
            records.append({"stem": stem, "state": "unstarted-time-budget"})
            return False
        command = [
            sys.executable,
            str(script),
            "--variant",
            variant,
            "--order",
            str(order),
            "--output",
            str(args.output / f"{stem}.json"),
        ]
        if action == "lower":
            command.append("--lower-only")
        elif action == "eager":
            command += ["--eager", "--adaptive-eager"]
        else:
            command += ["--compile-instruction-limit", "10000000"]
        command += list(extra)
        timing = args.output / f"{stem}.time"
        wrapped = ["/usr/bin/time", "-v", "-o", str(timing), *command]
        log = args.output / f"{stem}.log"
        print(f"START {stem}", flush=True)
        started, peak = time.monotonic(), 0
        reason = None
        with log.open("w") as stream:
            stream.write(
                f"revision={revision} tree={Path.cwd().resolve()} "
                f"command={shlex.join(wrapped)}\n"
            )
            stream.flush()
            process = subprocess.Popen(
                wrapped,
                stdout=stream,
                stderr=subprocess.STDOUT,
                preexec_fn=cap_address_space,
            )
            while process.poll() is None:
                members = descendants(process.pid)
                peak = max(peak, sum(row[2] for row in members))
                if peak > 200 * GIB // 1024:
                    reason = "row RSS exceeds 200 GiB; preserve the job"
                elif time.monotonic() - started > min(timeout, remaining - 90):
                    reason = "row wall budget exhausted"
                if reason:
                    stop_child(process, reason)
                    break
                time.sleep(1)
            code = process.wait()
            stream.write(f"\nEXIT={code}\n")
        records.append(
            {
                "stem": stem,
                "variant": variant,
                "order": order,
                "action": action,
                "exit": code,
                "log": str(log),
                "time_log": str(timing),
                "sampled_tree_peak_rss_kib": peak,
                "seconds": time.monotonic() - started,
                "limit_reason": reason,
            }
        )
        (args.output / "index.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END " + json.dumps(records[-1]), flush=True)
        return code == 0

    # No native executable is attempted until every remaining row has lowered.
    for order in (2, 3, 4):
        for variant in VARIANTS:
            run(variant, order, "lower", 900)
    complete_lowerings = sum(
        row.get("action") == "lower" and row.get("exit") == 0 for row in records
    )
    if complete_lowerings != 12:
        print(
            f"LOWERING_INCOMPLETE={complete_lowerings}/12; "
            "native compilation not started",
            flush=True,
        )
        return 1
    # Each native arm has an independent address-space and RSS budget.
    for variant in VARIANTS[1:]:
        run(variant, 2, "compile", 1800)
    run("current", 2, "eager", 3600)
    # An eager variant still supplies accuracy if its whole-program compile fails.
    for variant in VARIANTS[1:]:
        array = args.output / f"compile-derivative-{variant}-2.npz"
        if not array.exists():
            run(variant, 2, "eager", 1800)
    parent_rss = sum(row[2] for row in descendants(os.getpid()))
    if deadline - time.monotonic() > 1200 and parent_rss < 20 * GIB // 1024:
        run("current", 2, "compile", 1200)
    else:
        print(
            "CURRENT_COMPILE_SKIPPED remaining wall reserved for durable output",
            flush=True,
        )
    complete_lowerings = sum(
        row.get("action") == "lower" and row.get("exit") == 0 for row in records
    )
    print(f"CENSUS_LOWERINGS_COMPLETE={complete_lowerings}/12", flush=True)
    print("CENSUS_ROWS_FINISHED", flush=True)
    return 0 if complete_lowerings == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
