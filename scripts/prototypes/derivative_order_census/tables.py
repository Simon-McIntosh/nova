"""Render receipt-derived HTML tables; no kernel or compilation is imported."""

import argparse
from html import escape
import json
from pathlib import Path

NAMES = {
    "current": "Current",
    "plain": "Plain autodiff",
    "without_samples": "Without samples",
    "masked_samples": "Masked samples",
}


def integer(value):
    return "Unavailable" if value is None else f"{value:,}"


def number(value):
    return "Unavailable" if value is None else f"{value:.9g}"


def table(identity, headers, rows):
    head = "".join(f"<th>{escape(cell)}</th>" for cell in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<table id="{identity}">\n<thead><tr>{head}</tr></thead>\n'
        f"<tbody>\n{body}\n</tbody>\n</table>"
    )


def render(summary):
    cost, native, accuracy = [], [], []
    for receipt in summary["rows"]:
        outcome = receipt.get("native_outcome")
        if outcome:
            state = "RSS limit" if "RSS" in outcome["limit_reason"] else "Wall limit"
            native.append(
                [
                    NAMES[receipt["variant"]],
                    f"{outcome['seconds']:.3f}",
                    f"{outcome['time_peak_rss_kib'] / 2**20:.3f}",
                    state,
                    str(outcome["exit"]),
                    "Unavailable",
                ]
            )
        else:
            state = "Complete" if receipt["phase"] == "complete" else "Lowering only"
        cost.append(
            [
                str(receipt["order"]),
                NAMES[receipt["variant"]],
                *(
                    integer(receipt[key])
                    for key in (
                        "stablehlo_bytes",
                        "hlo_instructions",
                        "packed_kernel_ops",
                        "packed_kernel_instances",
                        "temp_bytes",
                    )
                ),
                state,
            ]
        )
    for row in summary["comparisons"]:
        if row["state"] != "compared":
            continue
        accuracy.append(
            [
                str(row["order"]),
                NAMES[row["variant"]],
                str(row["generic_targets"]),
                str(row["corner_targets"]),
                number(row["generic_max_relative"]),
                number(row["corner_max_relative"]),
                f"{row['finite_values']} / {row['total_values']}",
                row["candidate_mode"],
            ]
        )
    return {
        "cost-table": table(
            "cost-table",
            [
                "Order",
                "Rule",
                "StableHLO bytes",
                "HLO instructions",
                "Attributed packed ops",
                "Retained packed scopes",
                "CPU temp bytes",
                "Outcome",
            ],
            cost,
        ),
        "accuracy-table": table(
            "accuracy-table",
            [
                "Order",
                "Rule",
                "Generic targets",
                "Corner-height targets",
                "Max relative difference, generic",
                "Max relative difference, corner",
                "Finite scalar entries",
                "Evaluation",
            ],
            accuracy,
        ),
        "native-table": table(
            "native-table",
            [
                "Rule",
                "Child elapsed [s]",
                "Peak RSS [GiB]",
                "Stop",
                "Child exit",
                "CPU temp bytes",
            ],
            native,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.write_text("\n\n".join(render(summary).values()) + "\n")
    compared = sum(row["state"] == "compared" for row in summary["comparisons"])
    native = sum(row.get("native_outcome") is not None for row in summary["rows"])
    print(
        f"TABLES cost_rows={len(summary['rows'])} accuracy_rows={compared} "
        f"native_rows={native}"
    )


if __name__ == "__main__":
    main()
