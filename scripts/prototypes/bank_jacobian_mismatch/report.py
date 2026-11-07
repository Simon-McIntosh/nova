"""Render the three evidence tables from a mismatch receipt."""

import argparse
import json
import statistics
from pathlib import Path

CLASSES = ("cut", "normal", "whole", "axis", "saddle", "random")
LABELS = {
    "cut": "cut",
    "normal": "normal form",
    "whole": "whole",
    "axis": "axis cell",
    "saddle": "saddle cell",
    "random": "random",
}


def median(state, klass, key):
    values = [
        row[key]
        for row in state["directions"]
        if row["class"] == klass and key in row and row[key] == row[key]
    ]
    return statistics.median(values) if values else float("nan")


def cell(value):
    return "—" if value != value else f"{value:.3g}"


def table_one(receipt):
    header = "".join(f"<th>{LABELS[name]}</th>" for name in CLASSES)
    rows = []
    for state in receipt["states"]:
        for index, eps in enumerate(receipt["steps"]):
            cells = "".join(
                f"<td>{cell(median(state, name, f'analytic-full:h{eps:g}:rel'))}</td>"
                for name in CLASSES
            )
            label = state["label"] if index == 0 else ""
            span = (
                f'<td rowspan="{len(receipt["steps"])}">{label}</td>'
                if index == 0
                else ""
            )
            rows.append(f"<tr>{span}<td>{eps:g}</td>{cells}</tr>")
    return (
        "<table><thead><tr><th>state</th><th>relative step</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def table_two(receipt):
    header = (
        "<tr><th>state</th><th>direction class</th><th>Jacobian as assembled</th>"
        "<th>booking to CD</th><th>booking to CD, control gradient</th>"
        "<th>geometry to CD</th><th>read refresh share</th></tr>"
    )
    rows = []
    for state in receipt["states"]:
        for name in CLASSES:
            values = [
                median(state, name, f"analytic-full:h1e-06:rel"),
                median(state, name, f"booking-to-cd:h1e-06:rel"),
                median(state, name, f"booking-to-cd-control:h1e-06:rel"),
                median(state, name, f"geometry-to-cd:h1e-06:rel"),
                median(state, name, f"refresh:h1e-06:rel"),
            ]
            row = (
                f"<tr><td>{state['label']}</td><td>{LABELS[name]}</td>"
                + "".join(f"<td>{cell(value)}</td>" for value in values)
                + "</tr>"
            )
            rows.append(row)
    return "<table><thead>" + header + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def table_three(receipt):
    header = (
        "<tr><th>state</th><th>direction class</th><th>step</th>"
        "<th>cut</th><th>normal form</th><th>fragments</th><th>ownership</th>"
        "<th>saddle displacement</th></tr>"
    )
    rows = []
    for state in receipt["states"]:
        for klass in CLASSES:
            changed = [d for d in state["decisions"] if d["class"] == klass]
            if not changed:
                continue
            for index, entry in enumerate(changed):
                plus, minus = entry["plus"], entry["minus"]
                values = [
                    max(plus["cut"], minus["cut"]),
                    max(plus["normal"], minus["normal"]),
                    max(plus["fragments"], minus["fragments"]),
                    max(plus["ownership"] and 1 or 0, minus["ownership"] and 1 or 0),
                    max(plus["saddle_move"], minus["saddle_move"]),
                ]
                label = f"<td>{state['label']}</td><td>{LABELS[klass]}</td>" if index == 0 else "<td></td><td></td>"
                rows.append(
                    f"<tr>{label}<td>{entry['step']:g}</td>"
                    + "".join(f"<td>{cell(value) if isinstance(value, float) else value}</td>" for value in values)
                    + "</tr>"
                )
    return "<table><thead>" + header + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    print("TABLE_ONE")
    print(table_one(receipt))
    print("TABLE_TWO")
    print(table_two(receipt))
    print("TABLE_THREE")
    print(table_three(receipt))
    print("GRADIENT")
    print(json.dumps(receipt["gradient_check"], indent=2))
    for state in receipt["states"]:
        print("STATE", state["label"], state["base_residual_max"], state["class_counts"])


if __name__ == "__main__":
    main()