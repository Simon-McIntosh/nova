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

An arm counts for the revision the config declares and no other. A head row
recorded at some earlier revision measures that revision's code, so pairing a
head row with a base row without checking is how an earlier head's result gets
read as this head's. Such a row is written as head-other-revision and names the
revision it measured.
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


def facts(row):
    if not row:
        return {}
    return {
        "completed": bool(row.get("completed")),
        "seconds": row.get("seconds"),
        "summary": row.get("summary_line"),
        "peak_kb": row.get("peak_rss_kb"),
        "exit_status": row.get("exit_status"),
        "revision": row.get("revision"),
    }


def detail(prefix, item):
    return {
        prefix + "_completed": item.get("completed"),
        prefix + "_seconds": item.get("seconds"),
        prefix + "_summary": item.get("summary"),
        prefix + "_peak_kb": item.get("peak_kb"),
        prefix + "_exit_status": item.get("exit_status"),
        prefix + "_revision": item.get("revision"),
    }


def measure(head, base):
    base = base or {}
    head_ids = set(head["failure_ids"] or [])
    base_ids = set(base.get("failure_ids") or [])
    row = {
        "file": head["file"],
        "base_count": len(base_ids),
        "head_count": len(head_ids),
        "added": sorted(head_ids - base_ids),
        "removed": sorted(base_ids - head_ids),
        "head_only": not base,
        "state": "measured",
    }
    row.update(detail("base", facts(base) if base else {}))
    row.update(detail("head", facts(head)))
    return row


def pending(file, base, state="head-pending", head_revision=None):
    row = {
        "file": file,
        "base_count": len(base["failure_ids"] or []),
        "head_count": None,
        "added": [],
        "removed": [],
        "head_only": False,
        "state": state,
        "measured_head_revision": head_revision,
    }
    row.update(detail("base", facts(base)))
    row.update(detail("head", {}))
    return row


def at_revision(row, revision):
    return not revision or row.get("revision") in (None, revision)


def absent(file):
    row = {
        "file": file,
        "base_count": None,
        "head_count": None,
        "added": [],
        "removed": [],
        "head_only": False,
        "state": "not-run",
    }
    row.update(detail("base", {}))
    row.update(detail("head", {}))
    return row


def pairs(row):
    return (
        row["state"] == "measured"
        and not row["head_only"]
        and bool(row["base_completed"])
        and bool(row["head_completed"])
    )


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
    names = config["receipts"]
    if isinstance(names, str):
        names = [names]
    root = Path(args.receipts_dir) if args.receipts_dir else config_path.parent
    receipts = [(root / name).resolve() for name in names]

    head_revision = config.get("head_revision")
    base_revision = config.get("base_revision")

    arms = []
    for path in receipts:
        arms.extend(dict(row, arm=row.get("arm", "head")) for row in rows_in(path))
    heads = by_file([row for row in arms if row["arm"] == "head" and at_revision(row, head_revision)])
    elsewhere = by_file(
        [row for row in arms if row["arm"] == "head" and not at_revision(row, head_revision)]
    )
    recorded = by_file(
        [row for row in arms if row["arm"] == "base" and at_revision(row, base_revision)]
    )

    base = {}
    if args.base_rows:
        for path in args.base_rows.split(","):
            for row in rows_in(path):
                if row.get("device", device) == device:
                    base[row["file"]] = row
    base.update(recorded)

    declared = set(config["files"])
    result = []
    for file in sorted(declared | set(heads) | set(elsewhere)):
        head = heads.get(file)
        if head is not None:
            result.append(measure(head, base.get(file)))
        elif file in base and file in elsewhere:
            result.append(
                pending(
                    file,
                    base[file],
                    state="head-other-revision",
                    head_revision=elsewhere[file].get("revision"),
                )
            )
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
        "receipts": [str(path) for path in receipts],
        "base_rows": args.base_rows,
        "rows": result,
        "unmeasured": unmeasured,
        "added_total": sum(len(row["added"]) for row in result),
        "missing_base_files": missing,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    verdicts = sum(1 for row in result if pairs(row))
    print(
        "device="
        + device
        + " rows=%d verdicts=%d unmeasured=%d added=%d"
        % (len(result), verdicts, len(unmeasured), payload["added_total"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
