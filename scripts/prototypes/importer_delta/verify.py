"""Decide whether the gated device's table answers for every file it declares.

    verify.py <delta table>...

A table is gated when its config says so; a table that is not gated is
information. The verdict reads the gated tables only:

* every file the gated device declares must carry a completed head arm at the
  revision the table records, and a completed base arm too unless the head is
  the only arm the file has;
* no gated row may add a failure id;
* a gated table with no rows at all fails, because a table that measured
  nothing is not a clean one.

An information table is printed and never enters the verdict: a slow file the
card measured once is evidence a reader should see beside the gated rows, not a
row the gate can pass or fail.
"""

import json
import sys


def rows_in(path):
    return json.loads(open(path).read())


def has_pair(row):
    return (
        row.get("state") == "measured"
        and not row.get("head_only")
        and bool(row.get("base_completed"))
        and bool(row.get("head_completed"))
    )


def head_alone(row):
    return (
        row.get("state") == "measured"
        and bool(row.get("head_only"))
        and bool(row.get("head_completed"))
    )


def main():
    paths = sys.argv[1:]
    problems = []
    gated_rows = 0
    gated_files = 0
    for path in paths:
        table = rows_in(path)
        device = table["device"]
        rows = table["rows"]
        gated = bool(table.get("gated", True))
        kind = "gate" if gated else "information"
        pairs = 0
        alone = 0
        added = 0
        for row in rows:
            if has_pair(row):
                pairs += 1
            elif head_alone(row):
                alone += 1
            if row.get("added"):
                added += len(row["added"])
            if gated:
                if has_pair(row) or head_alone(row):
                    pass
                else:
                    state = row.get("state")
                    if state == "measured":
                        state = "head-not-completed"
                    print("UNMEASURED %s %s %s" % (device, row["file"], state))
                    problems.append("%s:%s" % (device, row["file"]))
                for item in row.get("added") or []:
                    print("ADDED %s %s %s" % (device, row["file"], item))
            else:
                print("INFO %s %s %s" % (device, row["file"], row.get("state")))
        if gated:
            if not rows:
                problems.append(device + ":no-rows")
            gated_rows += len(rows)
            gated_files += pairs + alone
        print(
            "DEVICE=%s %s ROWS=%d PAIRS=%d HEAD_ONLY=%d ADDED=%d"
            % (device, kind, len(rows), pairs, alone, added)
        )
        if gated and added:
            problems.append("%s:added=%d" % (device, added))
    ok = not problems
    print(
        "GATE=%s gated_files=%d gated_rows=%d problems=%d"
        % ("PASS" if ok else "FAIL", gated_files, gated_rows, len(problems))
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())