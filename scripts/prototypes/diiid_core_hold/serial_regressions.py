"""Run incomplete files serially with a per-command resident-memory receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from continue_regressions import SCOPED, equivalent, readable
from regressions import ROOT, extract, gate


def peak_rss(path):
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text()
    )
    return int(match[1]) if match else None


def timed_gate(tree, revision, path, output, scratch):
    log = output / (path.replace("/", "-") + ".log")
    memory = log.with_suffix(".rss.txt")
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(memory),
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        path,
    ]
    literal = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
    env = dict(
        os.environ,
        PYTHONPATH=".",
        JAX_PLATFORMS="cpu",
        TMPDIR=str(scratch),
        PYTEST_TIMEOUT="3600",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="1",
    )
    with log.open("x") as stream:
        stream.write(f"# revision: {revision}; tree: {tree}; command: {literal}\n")
        stream.flush()
        probe = [
            sys.executable,
            "-c",
            "from nova.equilibrium import solve_request; "
            "import os; print('IMPORTED_MODULE=' + solve_request.__file__); "
            "print('RESOLVED_CWD=' + os.getcwd())",
        ]
        check = subprocess.run(
            probe,
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result = (
            check
            if check.returncode
            else subprocess.run(
                command,
                cwd=tree,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        )
        stream.write(f"\nEXIT={result.returncode}\n")
    text = log.read_text()
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
    completed = bool(
        re.search(r"\d+ (?:passed|failed|skipped|errors?).* in [\d.]+s", text)
    )
    if result.returncode and not failures:
        failures = [path + "::unattributed_process_abort"]
    return dict(
        revision=revision,
        command=literal,
        exit_status=result.returncode,
        log_path=str(log),
        completed=completed,
        failure_count=len(failures),
        failure_ids=failures,
        peak_rss_kib=peak_rss(memory) if memory.exists() else None,
        peak_rss_log=str(memory),
        memory_measure="GNU time maximum resident set size of the pytest command",
        oom_requires_scheduler_confirmation=result.returncode in (-9, 137),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "repaired"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(args.previous.read_text())
    files = json.loads(args.files.read_text())
    retained = {}
    for key, record in previous.items():
        if key.startswith(args.arm + ":tests/") and readable(record):
            equivalent(args.revision, record["revision"])
            retained[key] = record
    pending = [
        path
        for path in files
        if path not in SCOPED and args.arm + ":" + path not in retained
    ]
    if args.file_only and len(pending) > 1:
        raise ValueError("an isolated memory run must name at most one incomplete file")
    plan = dict(
        arm=args.arm,
        revision=args.revision,
        pending=pending,
        retained=list(retained),
        max_concurrent_pytest=1,
        isolated_file=args.file_only,
    )
    (args.output / "continuation-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    results = dict(retained)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("SERIAL_PLAN " + json.dumps(plan), flush=True)
    control = args.output / "rss-control.txt"
    sample = subprocess.run(
        [
            "/usr/bin/time",
            "-v",
            "-o",
            str(control),
            sys.executable,
            "-c",
            "payload=bytearray(16*1024*1024); print(len(payload))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert sample.stdout.strip() == "16777216" and peak_rss(control) >= 16 * 1024
    print("RSS_CONTROL_KIB=" + str(peak_rss(control)), flush=True)
    with tempfile.TemporaryDirectory(
        prefix="nova-serial-regression-", dir=os.environ["TMPDIR"]
    ) as scratch:
        tree = Path(scratch) / "tree"
        extract(tree, args.revision, ROOT)
        if not args.file_only:
            key = args.arm + ":suite"
            scoped = previous.get(key)
            if readable(scoped) and scoped["revision"] == args.revision:
                results[key] = scoped
            else:
                paths = [path for path in SCOPED if (tree / path).is_file()]
                results[key] = gate(
                    tree, args.revision, paths, args.output / "suite.log", Path(scratch)
                )
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
        for path in pending:
            folder = Path(scratch) / path.replace("/", "-")
            folder.mkdir()
            key = args.arm + ":" + path
            results[key] = timed_gate(tree, args.revision, path, args.output, folder)
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print("REGRESSION " + key + " " + json.dumps(results[key]), flush=True)


if __name__ == "__main__":
    main()
