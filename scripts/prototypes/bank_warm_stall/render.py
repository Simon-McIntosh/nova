"""Render the reduced record as the fragment's HTML rows.

The boundary class is an integer in the record and a word in the table, so the
mapping lives here rather than in the writer of the table: a renderer that
prints the raw index cannot reproduce the fragment it is meant to feed.
"""

import argparse
import json
from pathlib import Path

CLASS = {0: "limited", 1: "diverted"}


def number(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def count(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}g}"


def decision(record):
    return (
        f"{CLASS.get(record['boundary_class'], record['boundary_class'])}, "
        f"cut {record['cut_cells']}, "
        f"normal {record['normal_cells']}, "
        f"frag {record['fragments']}, "
        f"saddle {record['admitted_saddles']}"
    )


def seed_reads(record):
    """The seed's own read is what the first update's cumulative count adds to."""
    first = record["table"][0]
    return first["cumulative_reads"] - len(first["trials"])


def rows(record):
    out = [
        "<tr><td>seed</td><td>—</td><td>—</td><td>—</td>"
        f"<td>{number(record['seed_residual'])}</td><td>—</td>"
        f"<td>—</td><td>—</td><td>—</td><td>{seed_reads(record)}</td>"
        f"<td>{decision(record['seed_decision'])}</td></tr>"
    ]
    for row in record["table"]:
        trials = "; ".join(
            f"{number(trial['scale'])}→{number(trial['residual'])}"
            + ("" if trial["valid"] else " (invalid)")
            for trial in row["trials"]
        )
        out.append(
            "<tr>"
            f"<td>{row['update'] + 1}</td>"
            f"<td>{number(row['incumbent'])}</td>"
            f"<td>{number(row['accepted_scale'])}</td>"
            f"<td>{number(row['accepted_residual'])}</td>"
            f"<td>{len(row['trials'])}</td>"
            f"<td>{trials}</td>"
            f"<td>{'yes' if row['refactored'] else 'no'}</td>"
            f"<td>{count(row['krylov_condition'])}</td>"
            f"<td>{count(row['krylov_iterations'])}</td>"
            f"<td>{row['cumulative_reads']}</td>"
            f"<td>{decision(row['decision'])}</td>"
            "</tr>"
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arm", default="warm")
    args = parser.parse_args()
    record = json.loads(args.report.read_text())[args.arm]
    for line in rows(record):
        print("      " + line)


if __name__ == "__main__":
    main()
