"""Check the record's profile-content table against the receipts the solves were handed.

The table quotes each frame's last prescribed knot, and its caption says those
digits are read from the row's own recorded source-inputs.json. That claim is
worth nothing unless something compares the two, so this reads the printed cells
back out of the fragment and compares them with the receipts at the precision
each cell is printed to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

CASES = {
    "137": ("diverted",),
    "146": ("diverted", "limited"),
    "179": ("limited",),
}
ROW = re.compile(r"<tr><td>(?P<frame>137|146|179)</td>(?P<cells>(?:<td>.*?</td>)+)")
CELL = re.compile(r"<td>(.*?)</td>", re.DOTALL)
TAGS = re.compile(r"<[^>]+>")
EDGE = re.compile(r"[^/]*/\s*(?P<value>.*)")


def printed(text):
    """The number a cell prints, and how many decimals it prints it to."""

    text = text.replace("−", "-").strip()
    return float(text), len(text.split(".")[1]) if "." in text else 0


def table_cells(fragment):
    """The last of each pair the profile-content table prints, by frame."""

    body = fragment.split("<caption>Prescribed profile content", 1)[1]
    body = body.split("</table>", 1)[0]
    cells = {}
    for match in ROW.findall(body):
        frame, row = match[0], match[1]
        columns = [TAGS.sub("", cell) for cell in CELL.findall(row)]
        # columns: p' integral, removed share, p'(1) pair, FF' integral,
        # removed share, FF'(1) pair
        cells[frame] = {
            "p_prime": EDGE.match(columns[2]).group("value"),
            "ff_prime": EDGE.match(columns[5]).group("value"),
        }
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    args = parser.parse_args()
    cells = table_cells(args.fragment.read_text())
    problems = []
    for frame, branches in CASES.items():
        for branch in branches:
            inputs = json.loads(
                (
                    args.cohort
                    / "edge"
                    / f"frame-{frame}-{branch}"
                    / "source-inputs.json"
                ).read_text()
            )
            for key in ("p_prime", "ff_prime"):
                shown, decimals = printed(cells[frame][key])
                recorded = float(inputs[key]["values"][-1])
                agrees = round(recorded, decimals) == shown
                print(
                    f"frame {frame} {branch} {key}: table {shown} recorded "
                    f"{recorded:.9f} {'agrees' if agrees else 'DISAGREES'} "
                    f"at {decimals} decimals"
                )
                if not agrees:
                    problems.append(f"frame {frame} {branch} {key}")
    if problems:
        for name in problems:
            print(f"FAIL {name} prints a value its receipt does not record")
        return 1
    print(f"PASS {len(cells)} rows and both quantities match the receipts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
