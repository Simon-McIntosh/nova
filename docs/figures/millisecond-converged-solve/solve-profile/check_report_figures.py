#!/usr/bin/env python3
"""Check that every figure this report references is present and readable.

The report names its figure artifacts by path or basename. This scans
report.md for any token ending in .svg or .json.gz, resolves each by basename
against this directory, and requires it to exist. Each compressed receipt is
additionally decompressed and parsed as JSON, so a truncated or corrupt
archive fails here rather than silently shipping. Exits nonzero on the first
class of defect found.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = re.compile(r"[\w./-]+\.(?:svg|json\.gz)")


def main() -> int:
    names = sorted({m.rsplit("/", 1)[-1] for m in REFERENCE.findall((HERE / "report.md").read_text())})
    if not names:
        print("NO REFERENCES: report.md names no .svg or .json.gz figure")
        return 1
    missing = [n for n in names if not (HERE / n).is_file()]
    if missing:
        print(f"MISSING {len(missing)} of {len(names)}: {missing}")
        return 1
    for name in names:
        path = HERE / name
        if name.endswith(".json.gz"):
            with gzip.open(path, "rb") as fh:
                json.load(fh)
            print(f"ok  {name} (parses as JSON, {path.stat().st_size} B on disk)")
        else:
            print(f"ok  {name} ({path.stat().st_size} B)")
    print(f"RESOLVED {len(names)} of {len(names)} references")
    return 0


if __name__ == "__main__":
    sys.exit(main())