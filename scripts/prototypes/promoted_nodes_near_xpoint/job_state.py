"""Report a scheduler job's state, printing TERMINAL once it has ended.

A wait block matches its probe's output against the states it names.  The
scheduler prints one state per job while it lives and a different name for each
way a job can end, so this collapses those endings to one word and reports the
live state unchanged otherwise.
"""

import subprocess
import sys

ENDED = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def main() -> int:
    job = sys.argv[1]
    live = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if live:
        print(live[0])
        return 0
    ended = subprocess.run(
        ["sacct", "-n", "-X", "-o", "State", "-j", job],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if any(state.split()[0] in ENDED for state in ended):
        print("TERMINAL")
    elif ended:
        print(ended[0].split()[0])
    else:
        print("UNKNOWN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
