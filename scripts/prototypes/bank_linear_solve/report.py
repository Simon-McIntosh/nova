"""Render the archived linear-route timing and precision measurements."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked

LABELS = {
    "present": "Double LU",
    "refined-0": "Single LU, no refinement",
    "refined-1": "Single LU + 1 refinement",
    "refined-2": "Single LU + 2 refinements",
    "refined-3": "Single LU + 3 refinements",
    "gmres": "Block GMRES",
    "gmres-strict": "GMRES, both coordinate bounds",
    "stacked": "Stacked double LU",
}


def table(rows, precision=False):
    headings = (
        [
            "State",
            "Arm",
            "Lanes",
            "Update error",
            "Original condition estimate",
            "Balanced condition estimate",
            "Estimated error bound",
            "Meets precision",
        ]
        if precision
        else [
            "State",
            "Arm",
            "Lanes",
            "Warm wall [ms]",
            "Iterations / refinements",
            "Kernel launches",
        ]
    )
    lines = [
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headings)
        + "</tr></thead><tbody>"
    ]
    for row in rows:
        cells = [row["state"], LABELS[row["arm"]], str(row["lanes"])]
        if precision:
            cells += [
                f"{max(row[key]):.3e}"
                for key in (
                    "errors",
                    "condition_original",
                    "condition_balanced",
                    "bound",
                )
            ]
            cells += ["PASS" if row["passed"] else "MISS"]
        else:
            count = row["count"]
            cells += [
                f"{row['median_ms']:.3f}",
                f"{min(count)}–{max(count)}",
                str(row["kernels"]),
            ]
        lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "\n".join(lines) + "\n</tbody></table>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.receipt.read_text())
    assert len(rows) == 32 and all(len(r["samples_ms"]) >= 10 for r in rows)
    assert all(r["kernels"] > 0 for r in rows)
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True, sharex=True)
    for ax, state in zip(axes, ("trial", "converged"), strict=True):
        for index, arm in enumerate(LABELS):
            for lanes, marker, offset in ((1, "o", -0.12), (16, "s", 0.12)):
                row = next(
                    r
                    for r in rows
                    if (r["state"], r["arm"], r["lanes"]) == (state, arm, lanes)
                )
                colour = "#929292" if row["passed"] else "#c43c39"
                ax.plot(
                    row["median_ms"],
                    index + offset,
                    marker=marker,
                    color=colour,
                    markerfacecolor="white" if lanes == 1 else colour,
                    markersize=8,
                    linestyle="none",
                    zorder=3,
                )
        ax.set_xscale("log")
        ax.set_xlabel(f"{state.capitalize()} wall [ms]")
        ax.set_yticks(np.arange(len(LABELS)), list(LABELS.values()))
        ax.set_ylim(len(LABELS) - 0.5, -0.7)
        ax.axvline(1, color="#929292", linestyle=":", linewidth=1.2, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    largest = max(r["median_ms"] for r in rows)
    smallest = min(r["median_ms"] for r in rows)
    axes[0].set_xlim(min(0.5, smallest * 0.6), largest * 1.6)
    axes[0].set_xticks(
        [
            v
            for v in (0.1, 1, 10, 100, 1000)
            if min(0.5, smallest * 0.6) < v < largest * 1.6
        ]
    )
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color="#929292",
            linestyle="none",
            markerfacecolor="white" if lanes == 1 else "#929292",
            markersize=8,
            label=f"{lanes} lane" if lanes == 1 else "16 lanes",
        )
        for lanes, marker in ((1, "o"), (16, "s"))
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="s",
            color="#c43c39",
            linestyle="none",
            markersize=8,
            label="Precision miss",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.64, 0.96),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.14, top=0.85, wspace=0.12)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(line.rstrip() for line in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.preview, dpi=100)
    args.tables.write_text(table(rows) + "\n" + table(rows, precision=True))
    print(
        "REPORT "
        + json.dumps(
            {
                "rows": len(rows),
                "precision_passes": sum(r["passed"] for r in rows),
                "fastest": sorted(
                    [r for r in rows if r["passed"] and r["lanes"] == 16],
                    key=lambda r: r["median_ms"],
                )[0]["arm"],
            }
        )
    )


if __name__ == "__main__":
    main()
