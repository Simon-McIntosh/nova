"""Plot measured WEST probe magnitudes against unknown count for both bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from scripts.iron_linear_magnetisation.figure import save_figure  # noqa: E402
from scripts.iron_graded_refinement.figure import GRADED as LINEAR  # noqa: E402

CONSTANT = "#929292"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.receipts.glob("*.json"):
        row = json.loads(path.read_text())
        if path.stem == row.get("mesh"):
            rows.append(row)
    rows = [
        r
        for r in rows
        if "constant" in r and "mesh" in r and not r["constant"].get("missing")
    ]
    assert sum(bool(r.get("completed")) for r in rows) >= 3, (
        "the three required WEST meshes must complete"
    )
    plt.style.use(
        Path(__file__).parents[1] / "magnetisation_high_permeability/data-ink.mplstyle"
    )
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "west-linear-probes"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.2), dpi=100)
    for probe, axis in enumerate(axes):
        for basis, colour, factor in [("constant", CONSTANT, 2), ("linear", LINEAR, 6)]:
            for uniform, marker in [(True, "o"), (False, "s")]:
                subset = sorted(
                    (
                        r
                        for r in rows
                        if r["mesh"].startswith("uniform") == uniform
                        and (basis == "constant" or r.get("completed"))
                    ),
                    key=lambda r: r["cells"],
                )
                if not subset:
                    continue
                x = [factor * r["cells"] for r in subset]
                y = [
                    1e6
                    * (r["constant"] if basis == "constant" else r["full_drive"])[
                        "magnitude_T"
                    ][probe]
                    for r in subset
                ]
                axis.plot(
                    x,
                    y,
                    color=colour,
                    marker=marker,
                    linestyle="-" if uniform else "none",
                    linewidth=3 if basis == "linear" else 2.6,
                    markerfacecolor=colour if uniform else "white",
                    markeredgewidth=1.8,
                    markersize=8,
                    label=f"{basis}, {'uniform' if uniform else 'graded'}",
                    zorder=5 if basis == "linear" else 3,
                )
        axis.text(
            0.05, 0.94, f"+{[1, 2, 4][probe]} m", transform=axis.transAxes, va="top"
        )
        axis.set_xscale("log")
        axis.set_xlim(2500, 100000)
        axis.xaxis.set_major_locator(FixedLocator([3000, 10000, 30000, 90000]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.set_xlabel("magnetic unknowns")
        axis.set_ylabel("probe |B| [µT]")
        axis.margins(y=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.84, wspace=0.32)
    checked = save_figure(fig, args.output, args.png)
    print(f"checked {len(checked)} text artists; wrote {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
