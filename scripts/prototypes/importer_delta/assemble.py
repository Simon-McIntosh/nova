"""Build one device's failure-id delta table from the recorded arm receipts.

    assemble.py --config <device config> [--base-rows <receipts file>] --out <file>

A device config names the device, the receipt file that holds its arms (resolved
against the config's own directory when relative) and the test files the device
declares. Both arms of a row are read from receipts for the same device, so the
comparison never crosses devices: the base arm comes from the device's own
receipt file when it carries one, and otherwise from the recorded base receipts
the --base-rows file names. A declared file with a base arm and no head arm is
written as head-pending, and one with neither arm as not-run. Neither carries an
added id, because a row states no verdict until both arms of one device have
completed.
"""

import argparse
import json
from pathlib import Path


def rows_in(path):
    data = json.loads(Path(path).read_text())
    return data["rows"] if isinstance(data, dict) else data


def by_file(rows):
    table = {}
    for row in rows:
        table[row["file"]] = row
    return table


def measure(head, base):
    base = base or {}
    head_ids = set(head["failure_ids"] or [])
    base_ids = set(base.get("failure_ids") or [])
    return {
        "file": head["file"],
        "base_count": len(base_ids),
        "head_count": len(head_ids),
        "added": sorted(head_ids - base_ids),
        "removed": sorted(base_ids - head_ids),
        "head_only": not base,
        "base_completed": bool(base.get("completed")) if base else None,
        "head_completed": bool(head.get("completed")),
        "base_seconds": base.get("seconds"),
        "head_seconds": head.get("seconds"),
        "state": "measured",
    }


def pending(file, base):
    return {
        "file": file,
        "base_count": len(base["failure_ids"] or []),
        "head_count": None,
        "added": [],
        "removed": [],
        "head_only": False,
        "base_completed": bool(base["completed"]),
        "head_completed": None,
        "base_seconds": base["seconds"],
        "head_seconds": None,
        "state": "head-pending",
    }


def absent(file):
    return {
        "file": file,
        "base_count": None,
        "head_count": None,
        "added": [],
        "removed": [],
        "head_only": False,
        "base_completed": None,
        "head_completed": None,
        "base_seconds": None,
        "head_seconds": None,
        "state": "not-run",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-rows")
    parser.add_argument("--receipts-dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    device = config["device"]
    receipts = Path(config["receipts"])
    if not receipts.is_absolute():
        root = Path(args.receipts_dir) if args.receipts_dir else config_path.parent
        receipts = (root / receipts).resolve()

    arms = [dict(row, arm=row.get("arm", "head")) for row in rows_in(receipts)]
    heads = by_file([row for row in arms if row["arm"] == "head"])
    recorded = by_file([row for row in arms if row["arm"] == "base"])

    base = {}
    if args.base_rows:
        for row in rows_in(args.base_rows):
            if row.get("device", device) == device:
                base[row["file"]] = row
    base.update(recorded)

    declared = set(config["files"])
    result = []
    for file in sorted(declared | set(heads)):
        head = heads.get(file)
        if head is not None:
            result.append(measure(head, base.get(file)))
        elif file in base:
            result.append(pending(file, base[file]))
        else:
            result.append(absent(file))

    head_only = set(config.get("head_only") or [])
    unmeasured = []
    for row in result:
        if row["state"] == "measured":
            if not row["head_completed"]:
                unmeasured.append({"file": row["file"], "state": "head-not-completed"})
        else:
            unmeasured.append({"file": row["file"], "state": row["state"]})
    missing = [
        row["file"]
        for row in result
        if row["state"] == "measured"
        and row["head_only"]
        and row["file"] not in head_only
    ]

    payload = {
        "device": device,
        "receipts": str(receipts),
        "base_rows": str(args.base_rows) if args.base_rows else None,
        "rows": result,
        "unmeasured": unmeasured,
        "added_total": sum(len(row["added"]) for row in result),
        "missing_base_files": missing,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    verdicts = len(result) - len(unmeasured)
    print(
        "device="
        + device
        + " rows=%d verdicts=%d unmeasured=%d added=%d"
        % (len(result), verdicts, len(unmeasured), payload["added_total"])
    )
    return 0


raise SystemExit(main())
