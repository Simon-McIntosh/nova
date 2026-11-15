"""Decide whether a set of device tables answers for every file the base set names.

    verify.py <base-rows files, comma separated> <delta table>...

Prints one line per device, one line per file carrying no completed pair, one
line per added failure id, and a verdict. A file carries a completed pair when
a base arm and a head arm for that file both reached a summary line on the same
device. The verdict passes only when every declared file carries a pair and no
row adds an id. It reads the rows, never the totals a table records about
itself: a table reporting a clean total over an unmeasured row is exactly the
failure this check exists to stop.
"""

import json
import sys


def rows_in(path):
    data = json.loads(open(path).read())
    return data["rows"] if isinstance(data, dict) else data


def declared_files(paths):
    files = {}
    for path in paths.split(","):
        for row in rows_in(path):
            files.setdefault(row["file"], row.get("device", "cpu"))
    return files


def main():
    declared = declared_files(sys.argv[1])
    covered = {}
    added = 0
    empties = []
    for path in sys.argv[2:]:
        data = json.loads(open(path).read())
        rows = data["rows"]
        device = data["device"]
        if not rows:
            empties.append(device)
            continue
        pairs = 0
        for row in rows:
            if has_pair(row):
                pairs += 1
                covered.setdefault(row["file"], set()).add(device)
            else:
                state = row.get("state")
                if state == "measured":
                    state = "head-not-completed"
                if not row.get("head_only"):
                    print("UNMEASURED %s %s %s" % (device, row["file"], state))
            for item in row.get("added") or []:
                print("ADDED %s %s %s" % (device, row["file"], item))
            added += len(row.get("added") or [])
        print(
            "DEVICE=%s ROWS=%d PAIRS=%d ADDED=%d"
            % (
                device,
                len(rows),
                pairs,
                sum(len(row.get("added") or []) for row in rows),
            )
        )
    return finish(declared, covered, added, empties)


def has_pair(row):
    return (
        row.get("state") == "measured"
        and not row.get("head_only")
        and bool(row.get("base_completed"))
        and bool(row.get("head_completed"))
    )


def finish(declared, covered, added, empties):
    missing = sorted(name for name in declared if name not in covered)
    for name in missing:
        print("MISSING " + name)
    ok = not missing and added == 0 and not empties
    print(
        "GATE=%s files=%d pairs=%d added=%d missing=%d empty=%d"
        % (
            "PASS" if ok else "FAIL",
            len(declared),
            len(covered),
            added,
            len(missing),
            len(empties),
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
