"""Render the observed table as the fragment's HTML rows."""

import argparse
import json
from pathlib import Path


def number(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.2e}"
        return f"{value:.{digits}g}"
    return str(value)


def decision(record):
    return (
        f"class {record['boundary_class']}, cut {record['cut_cells']}, "
        f"normal {record['normal_cells']}, "
        f"frag {record['fragments']}, "
        f"saddle {record['admitted_saddles']}, valid {int(record['valid'])}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arm", default="warm")
    args = parser.parse_args()
    record = json.loads(args.report.read_text())[args.arm]
    print(f"<!-- lane {record['lane']} of {args.arm} -->")
    for row in record["table"]:
        if row["update"] < 0:
            print(
                "<tr><td>seed</td><td>—</td><td>—</td><td>—</td><td>"
                + number(row["residual"])
                + "</td><td>—</td><td>—</td><td>—</td><td>—</td><td>"
                + decision(row["decision"])
                + "</td></tr>"
            )
            continue
        trials = "; ".join(
            f"{number(trial['scale'])}→{number(trial['residual'])}"
            + ("" if trial["valid"] else " (invalid)")
            for trial in row["trials"]
        )
        print(
            "<tr>"
            f"<td>{row['update'] + 1}</td>"
            f"<td>{number(row['incumbent'])}</td>"
            f"<td>{number(row['accepted_scale'])}</td>"
            f"<td>{number(row['residual'])}</td>"
            f"<td>{len(row['trials'])}</td>"
            f"<td>{trials}</td>"
            f"<td>{'yes' if row['refactored'] else 'no'}</td>"
            f"<td>{number(row['krylov_condition'])}</td>"
            f"<td>{number(row['krylov_iterations'])}</td>"
            f"<td>{number(row['reads'])}</td>"
            f"<td>{decision(row['accepted_decision'])}</td>"
            "</tr>"
        )


if __name__ == "__main__":
    main()
