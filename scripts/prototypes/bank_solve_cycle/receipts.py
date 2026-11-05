"""Assemble suite receipts from completed per-file logs without rerunning tests."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def assemble(run, names, revision, destination):
    records = []
    failure_ids = set()
    commands = []
    statuses = []
    for name in names:
        path = run / (name + ".log")
        text = path.read_text()
        lines = text.splitlines()
        if not lines[-1].startswith("EXIT="):
            raise ValueError(f"incomplete log: {path}")
        if not re.search(r"^=+ .* in .* =+$", text, re.MULTILINE):
            raise ValueError(f"no pytest summary: {path}")
        statuses.append(int(lines[-1].split("=", 1)[1]))
        header = lines[0]
        tree = header.split(" tree: ", 1)[1].split(" command: ", 1)[0]
        commands.append(
            "PYTHONPATH=" + shlex.quote(tree) + " " + header.split(" command: ", 1)[1]
        )
        for match in re.finditer(r"^FAILED\s+(\S+)", text, re.MULTILINE):
            identity = match[1]
            position = identity.find("tests/")
            failure_ids.add(identity[position:] if position >= 0 else identity)
        records.append(f"# included: {path}\n{text}")
    status = max(statuses)
    output = run / destination
    command = "\n".join(commands)
    output.write_text(
        f"# revision: {revision} command: {json.dumps(command)}\n"
        + "\n".join(records)
        + f"\nSUITE completed=true failure_count={len(failure_ids)}\nEXIT={status}\n"
    )
    return dict(
        revision=revision,
        command=command,
        exit_status=status,
        log_path=str(output),
        completed=True,
        failure_count=len(failure_ids),
        failure_ids=sorted(failure_ids),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = args.run / "manifest.md"
    lines = manifest.read_text().splitlines()
    fields = dict(line.split(":", 1) for line in lines if ":" in line)
    base = fields["orientation_base_sha"].strip()
    head_log = args.run / "contract-suite-head-test_bank_solve_cycle.log"
    revision = head_log.read_text().splitlines()[0].split()[2]
    baseline = assemble(
        args.run,
        [
            "cycle-seed-base-test_bank_solve_cycle",
            "stall-base-test_bank_solve_cycle",
            "cpu-suite-base-test_bank_solve_driver",
            "cpu-suite-base-test_support_read",
        ],
        base,
        "baseline-suite.log",
    )
    after = assemble(
        args.run,
        [
            "contract-suite-head-test_bank_solve_cycle",
            "contract-suite-head-test_bank_solve_driver",
            "contract-suite-head-test_support_read",
        ],
        revision,
        "after-suite.log",
    )
    commands = after["command"].splitlines()
    tests = commands[0]
    updates = {
        "status": "in-progress",
        "wait_condition": "",
        "wait_probe": "",
        "wait_terminal": "",
        "resume_brief": "",
        "checkpoint": (
            "All declared numerical jobs finished; suite receipts assembled. "
            "Final evidence refresh and manifest audit remain."
        ),
        "commits": json.dumps(
            subprocess.check_output(
                ["git", "rev-list", "--reverse", base + "..HEAD"], text=True
            ).split()
        ),
        "changed_paths": json.dumps(
            subprocess.check_output(
                ["git", "diff", "--name-only", base, "HEAD"], text=True
            ).splitlines()
        ),
        "tests": tests,
        "test_logs": json.dumps(
            [
                str(p)
                for p in args.run.glob("*.log")
                if p.name not in ("stderr.log", "supervisor.stderr.log")
            ]
        ),
        "baseline_suite": json.dumps(baseline),
        "after_suite": json.dumps(after),
        "measurement_module": "",
        "measurement_cwd": fields["orientation_worktree"].strip(),
        "negative_control_note": (
            "The controlled cycle fails at base on six steps exceeding two; "
            "head accepts in one. The real CPU restart also passes at base "
            "and is only a nonregression check."
        ),
        "blockers": "none"
        if after["exit_status"] == 0
        else "Head suite failures: " + ", ".join(after["failure_ids"]),
    }
    # Read the module path actually emitted by the base process.
    base_log = (args.run / "cycle-seed-base-test_bank_solve_cycle.log").read_text()
    match = re.search(r" module (\S+) cwd ", base_log)
    if match:
        updates["measurement_module"] = match[1]
    else:
        raise ValueError("base measurement did not print its imported module")
    output = []
    present = set()
    for line in lines:
        key = line.split(":", 1)[0]
        present.add(key)
        output.append(key + ": " + updates[key] if key in updates else line)
    output.extend(
        key + ": " + value for key, value in updates.items() if key not in present
    )
    manifest.write_text("\n".join(output) + "\n")
    print("BASELINE", json.dumps(baseline), flush=True)
    print("AFTER", json.dumps(after), flush=True)


if __name__ == "__main__":
    main()
