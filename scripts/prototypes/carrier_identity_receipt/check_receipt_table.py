"""Check that the receipt table lists every key its record carries.

The table's caption states how many recorded figures it covers, so the claim
is read against the record rather than against the table: every key of the
record must appear as a row, and the stated count must equal the number of
keys. A key shown through more than one aspect (a partition as its groups and
its digest) may occupy several rows, so rows may outnumber keys; a key with no
row is the omission this check exists to catch.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

CAPTION_COUNT = re.compile(r"N\s*=\s*(\d+)")
ROW_LABEL = re.compile(r"<tr><td><code>([a-z_]+)</code>")


def main(argv: list[str]) -> int:
    fragment = pathlib.Path(argv[1]).read_text()
    record = json.loads(pathlib.Path(argv[2]).read_text())
    table = fragment.split("<table>", 1)[1].split("</table>", 1)[0]
    labelled = ROW_LABEL.findall(table)
    named = set(labelled)
    keys = set(record)
    caption = fragment.split("<table>", 1)[0].rsplit("<p ", 1)[-1]
    stated = CAPTION_COUNT.search(caption)
    status = 0
    missing = sorted(keys - named)
    if missing:
        print(f"MISSING {len(missing)} key(s) with no row: {', '.join(missing)}")
        status = 1
    extra = sorted(named - keys)
    if extra:
        print(f"EXTRA {len(extra)} row label(s) not in the record: {', '.join(extra)}")
        status = 1
    if stated is None:
        print("CAPTION states no N")
        status = 1
    elif int(stated.group(1)) != len(keys):
        print(f"CAPTION states N = {stated.group(1)}, record carries {len(keys)} keys")
        status = 1
    print(
        f"keys={len(keys)} named={len(named)} rows={len(labelled)} "
        f"stated={stated.group(1) if stated else 'none'}"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))