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


def compare(record):
    """Every warm-against-cold figure the fragment quotes, from the record alone.

    Divergence is relative to the warm arm, (cold - warm) / warm, so a positive
    value means the cold path is the larger residual at that update.
    """
    warm, cold = record["warm"], record["cold"]
    a, b = warm["residual_path"], cold["residual_path"]
    relative = [(y - x) / x * 100.0 for x, y in zip(a, b)]
    axis = [
        max(abs(p[0] - q[0]), abs(p[1] - q[1]))
        for p, q in (
            (row["decision"]["axis"], other["decision"]["axis"])
            for row, other in zip(warm["table"], cold["table"])
        )
    ]
    peak = max(range(len(relative)), key=lambda i: abs(relative[i]))
    return {
        "seed_warm": warm["seed_residual"],
        "seed_cold": cold["seed_residual"],
        "agreement_until": next(
            i for i in range(len(relative)) if abs(relative[i]) > 10.0
        )
        - 1,
        "agreement_percent": max(
            abs(x)
            for x in relative[
                : next(i for i in range(len(relative)) if abs(relative[i]) > 10.0)
            ]
        ),
        "diverge_at": f"{relative[11]:.0f}",
        "peak_update": peak,
        "peak_percent": abs(relative[peak]),
        "warm_rise_from": a[11],
        "warm_rise_to": max(a[11:]),
        "warm_peak_update": 11 + a[11:].index(max(a[11:])),
        "warm_terminal": warm["terminal_residual"],
        "cold_peak": max(b),
        "cold_peak_update": b.index(max(b)),
        "cold_terminal": cold["terminal_residual"],
        "warm_converged": sum(warm["converged"]),
        "cold_converged": sum(cold["converged"]),
        "lanes": len(warm["converged"]),
        "updates": len(relative) - 1,
        "axis_first_ten": max(axis[:10]),
        "axis_all": max(axis),
    }


def summary(record):
    figures = compare(record)
    lines = [
        f"seed residual            warm {number(figures['seed_warm'])}  "
        f"cold {number(figures['seed_cold'])}",
        f"agreement                updates 0-{figures['agreement_until']} within "
        f"{figures['agreement_percent']:.1f}% (relative to warm)",
        f"divergence               {figures['diverge_at']}% at update 11, peak "
        f"{figures['peak_percent']:.0f}% at update {figures['peak_update']}",
        f"warm path                {number(figures['warm_rise_from'])} at update 11 "
        f"to {number(figures['warm_rise_to'])} at update "
        f"{figures['warm_peak_update']}, "
        f"terminal {number(figures['warm_terminal'])}",
        f"cold path                peak {number(figures['cold_peak'])} at update "
        f"{figures['cold_peak_update']}, terminal "
        f"{number(figures['cold_terminal'])}",
        f"converged                {figures['warm_converged']} of {figures['lanes']} "
        f"warm, {figures['cold_converged']} of {figures['lanes']} cold, in "
        f"{figures['updates']} updates",
        f"axis separation          {figures['axis_first_ten']:.1e} over the first ten "
        f"updates, {figures['axis_all']:.1e} over all {figures['updates']}",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arm", default="warm")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    record = json.loads(args.report.read_text())
    if args.summary:
        for line in summary(record):
            print(line)
        return
    for line in rows(record[args.arm]):
        print("      " + line)


if __name__ == "__main__":
    main()
