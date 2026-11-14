"""Read sacct for every named allocation and wait until all are terminal."""

import argparse
import json
from pathlib import Path
import subprocess

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("jobs", help="Comma-separated allocation ids")
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    wanted = args.jobs.split(",")
    if args.inventory and args.inventory.exists():
        wanted += [
            item["job_id"] for item in json.loads(args.inventory.read_text())["jobs"]
        ]
    wanted = list(dict.fromkeys(wanted))
    assert wanted and all(job.isdigit() for job in wanted)
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join(wanted), "-o", "JobIDRaw,State%30", "-P"],
        text=True,
        capture_output=True,
        check=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in wanted:
            states[fields[0]] = fields[1].split()[0]
    for job in wanted:
        print(f"JOB {job} {states.get(job, 'UNKNOWN')}")
    done = set(states) == set(wanted) and all(
        state in TERMINAL for state in states.values()
    )
    print("ALL_TERMINAL" if done else "PENDING")


if __name__ == "__main__":
    main()
