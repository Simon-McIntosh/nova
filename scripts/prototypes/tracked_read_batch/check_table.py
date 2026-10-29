"""Check the batch table against the record it claims to cover.

The fragment's table is checked by the shared receipt-table checker rather
than by a second copy of it: that checker reads the record's keys, the table's
row labels and the caption's stated count, and reports any key with no row.
Here it is pointed at the batch measurement record and the fragment carrying
the per-slice table, so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from scripts.prototypes.carrier_identity_receipt import check_receipt_table

ROOT = pathlib.Path(__file__).resolve().parents[3]
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/sub-millisecond-forward-path/tracked-read-batch.html"
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=pathlib.Path)
    parser.add_argument("--fragment", type=pathlib.Path, default=FRAGMENT)
    args = parser.parse_args(argv[1:])
    if args.record is None:
        run = pathlib.Path(
            "/home/ITER/mcintos/.config/reckon/crew/runs"
            "/r-20261029T090616166280-tracked-read-batch/batch_rows.json"
        )
        args.record = run
    return check_receipt_table.main(
        ["check_table", str(args.fragment), str(args.record)]
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
