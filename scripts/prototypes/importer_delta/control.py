"""Check the comparison against three known inputs, and write the log of that check.

    control.py <base receipts file> <compare.py> <log file>

Each control writes a pair of receipt files into a scratch directory and hands
them to compare.py exactly as the delta run does, so the control exercises the
same instrument rather than a second implementation of it. The controls are:

1. the base receipts compared with themselves relabelled as head, which must
   report no added id;
2. one base failure id removed from the base copy, which must report exactly
   that id as added;
3. one id planted in the head copy, which must report exactly that id.

A control 2 reporting nothing would mean the instrument cannot see a difference;
a control 3 reporting the planted id proves the pairing reaches the rows at all.
The log is rewritten on each run and ends with a bare exit line.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def relabel(rows, arm):
    out = []
    for row in rows:
        row = dict(row)
        row["arm"] = arm
        out.append(row)
    return out


def write(path, rows):
    path.write_text(json.dumps(rows))
    return str(path)


def added_ids(compare, base_rows, head_rows, work, tag):
    base = write(work / (tag + "-base.json"), base_rows)
    head = write(work / (tag + "-head.json"), head_rows)
    proc = subprocess.run(
        [sys.executable, compare, base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    ids = []
    for row in json.loads(proc.stdout):
        ids.extend(row["added"])
    return sorted(ids)


def main():
    base_rows_path = Path(sys.argv[1])
    compare = sys.argv[2]
    log = Path(sys.argv[3])
    base = json.loads(base_rows_path.read_text())

    target = None
    for row in base:
        if row["failure_ids"]:
            target = row
            break
    if target is None:
        raise SystemExit("no base receipt row carries a failure id to control with")
    removed = target["failure_ids"][0]
    planted = target["file"] + "::test_planted_new_failure"

    work = Path(tempfile.mkdtemp(prefix="importer-delta-control-"))
    base_drop = json.loads(json.dumps(base))
    for row in base_drop:
        if row["file"] == target["file"]:
            row["failure_ids"] = [i for i in row["failure_ids"] if i != removed]
    head_extra = relabel(base, "head")
    for row in head_extra:
        if row["file"] == target["file"]:
            row["failure_ids"] = list(row["failure_ids"]) + [planted]

    first = added_ids(compare, base, relabel(base, "head"), work, "self")
    second = added_ids(compare, base_drop, relabel(base, "head"), work, "drop")
    third = added_ids(compare, base, head_extra, work, "extra")
    ok = first == [] and second == [removed] and third == [planted]

    lines = [
        "# command="
        + sys.executable
        + " "
        + str(Path(__file__).resolve())
        + " "
        + str(base_rows_path)
        + " "
        + compare
        + " "
        + str(log),
        "# revision="
        + base[0].get("revision", "unrecorded")
        + ", "
        + str(len(base))
        + " base receipt rows",
        "# each control writes a pair of receipt files and hands them to the same "
        "compare.py the delta run uses",
        "# control 1: base receipts compared with themselves as head; expect nothing",
        "ADDED_TOTAL=" + str(len(first)),
        "# control 2: one base id removed from the base copy; expect exactly that id",
        "# planted=" + removed,
        "ADDED_TOTAL=" + str(len(second)),
    ]
    lines.extend("ADDED " + item for item in second)
    lines.extend(
        [
            "# control 3: one id planted in the head copy; expect exactly that id",
            "# planted=" + planted,
            "ADDED_TOTAL=" + str(len(third)),
        ]
    )
    lines.extend("ADDED " + item for item in third)
    lines.append("MISMATCHES=" + ("0" if ok else "1"))
    text = "\n".join(lines) + "\nEXIT=" + ("0" if ok else "1") + "\n"
    log.write_text(text)
    print(text, end="")
    return 0 if ok else 1


raise SystemExit(main())
