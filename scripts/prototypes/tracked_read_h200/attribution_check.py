"""Check that every number in the accuracy table was produced by a recorded run.

The table exists to attribute a state's error to a stage of the read, and a
cell with no run behind it is an argument rather than a measurement. This
script reads the cells out of the fragment's table, reads the ``ACCURACY_ROW``
records out of the named logs, and reports each cell as found or missing.

A cell is found when a recorded row of the same kind and state carries the
value at the table's quoted precision. Usage::

    python attribution_check.py <fragment.html> <log> [<log> ...]
"""

import json
import re
import sys
from pathlib import Path

COLUMNS = {
    "axis": "axis_error_m",
    "x-point": "saddle_error_m",
    "boundary flux / span": "boundary_error",
    "substituted": "analytic_field_axis_error_m",
    "interpolant": "interpolant_axis_error_m",
}
RELATIVE_TOLERANCE = 0.02


def table_rows(fragment):
    """The accuracy table's body rows, as (kind, state, [cells])."""
    body = fragment.split("<table>", 1)[1].split("</table>", 1)[0]
    for row in re.findall(r"<tr><td>(.*?)</tr>", body, flags=re.S):
        cells = [re.sub(r"<.*?>", "", cell).strip() for cell in row.split("<td>")]
        kind, state = cells[0].split(",")
        yield kind.strip(), state.strip(), cells[1:]


def recorded(log_paths):
    """The recorded accuracy rows, keyed by (kind, state)."""
    rows = {}
    for path in log_paths:
        for line in Path(path).read_text().splitlines():
            if "ACCURACY_ROW" in line:
                row = json.loads(line.split("ACCURACY_ROW", 1)[1])
            elif line.startswith("{"):
                row = json.loads(line)
            else:
                continue
            if "kind" in row and "state" in row:
                rows[(row["kind"], row["state"])] = row
    return rows


def main(argv):
    fragment = Path(argv[1]).read_text()
    rows = recorded(argv[2:])
    failures = 0
    for kind, state, cells in table_rows(fragment):
        row = rows.get((kind, state))
        for (title, key), cell in zip(COLUMNS.items(), cells):
            if cell in {"", "—"}:
                continue
            quoted = float(cell)
            value = None if row is None else row.get(key)
            if value is None:
                print(f"ATTRIBUTION missing {kind} {state} {title} {cell}")
                failures += 1
                continue
            relative = abs(value - quoted) / max(abs(value), 1e-30)
            if relative > RELATIVE_TOLERANCE:
                print(
                    f"ATTRIBUTION differs {kind} {state} {title} "
                    f"quoted={cell} recorded={value:.6g}"
                )
                failures += 1
            else:
                print(
                    f"ATTRIBUTION found {kind} {state} {title} "
                    f"{cell} recorded={value:.6g}"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
