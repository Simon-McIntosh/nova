"""Report the failure ids a head arm adds over a base arm, per file and device.

    compare.py <base receipts, comma separated> <head receipts, comma separated>

Reads receipt rows carrying arm, device, file and failure_ids, pairs each head
row with the base row for the same device and file, and prints one row per pair
with the ids present at head and absent from the base arm on that device. The
total is printed to stderr as ADDED_TOTAL so a caller can gate on it without
parsing the rows. A head file with no base arm is reported with head_only set,
which is not an added id: there is nothing to add to.
"""

import json
import sys


def load(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(open(path).read()))
    return rows


def file_of(row):
    return row["file"]


def index(rows, arm):
    table = {}
    for row in rows:
        row = dict(row)
        row.setdefault("arm", arm)
        if row["arm"] != arm:
            continue
        table[(row.get("device", "cpu"), file_of(row))] = row
    return table


def delta(base_rows, head_rows):
    base = index(base_rows, "base")
    head = index(head_rows, "head")
    out = []
    for key, row in sorted(head.items()):
        base_row = base.get(key, {})
        base_ids = set(base_row.get("failure_ids") or [])
        head_ids = set(row.get("failure_ids") or [])
        out.append(
            {
                "device": key[0],
                "file": key[1],
                "base_count": len(base_ids),
                "head_count": len(head_ids),
                "added": sorted(head_ids - base_ids),
                "removed": sorted(base_ids - head_ids),
                "head_only": key not in base,
                "base_completed": base_row.get("completed"),
                "head_completed": row.get("completed"),
            }
        )
    return out


def main():
    base_rows = load(sys.argv[1].split(","))
    head_rows = load(sys.argv[2].split(","))
    rows = delta(base_rows, head_rows)
    print(json.dumps(rows, indent=2))
    total = sum(len(row["added"]) for row in rows)
    print("ADDED_TOTAL=" + str(total), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
