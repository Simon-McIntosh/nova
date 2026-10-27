"""Summarize the per-file logs of the importing-test arm.

Every log is one test module run in a fresh pytest process. A log is read as
complete only when it carries pytest's own summary line, so a process that
died before its summary is reported as incomplete rather than as passing.
"""

from __future__ import annotations

import pathlib
import re
import sys

COUNTS = re.compile(r"(\d+) (passed|failed|errors?|skipped|xpassed|xfailed)")


def read(path: pathlib.Path) -> dict:
    text = path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    summary = ""
    for line in reversed(lines):
        if COUNTS.search(line) and (" in " in line or "==" in line):
            summary = line.strip("= ").strip()
            break
    failures = sorted(
        {
            line.split(" ", 1)[1].strip()
            for line in lines
            if line.startswith(("FAILED ", "ERROR "))
        }
    )
    return {
        "file": path.name.removesuffix(".log"),
        "summary": summary,
        "complete": bool(summary),
        "failures": failures,
    }


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1])
    # a kept copy of a killed run ends .oom.log; counting it would report one
    # module twice and read a killed copy as a second unfinished file
    rows = [read(path) for path in sorted(root.glob("*.log")) if ".oom." not in path.name]
    for row in rows:
        state = row["summary"] or "INCOMPLETE"
        print(f"{row['file']}: {state}")
        for failure in row["failures"]:
            print(f"  FAILED {failure}")
    incomplete = [row["file"] for row in rows if not row["complete"]]
    failures = [f for row in rows for f in row["failures"]]
    done = len(rows) - len(incomplete)
    print(f"files={len(rows)} complete={done} failures={len(failures)}")
    if incomplete:
        print("incomplete: " + ", ".join(incomplete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
