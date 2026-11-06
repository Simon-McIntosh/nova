"""Table saved perturbation receipts and draw their stage sensitivity."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_read.plot import save_checked


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    receipts = {
        arm: json.loads((args.run / f"{arm}.json").read_text())
        for arm in ("gpu", "cpu")
    }
    rows, summaries = [], {}
    for arm, receipt in receipts.items():
        norm = np.asarray([p["norm"] for p in receipt["perturbations"]])
        assert norm.shape == (64, len(receipt["lanes"]))
        stats = dict(
            median=np.median(norm, axis=0),
            minimum=norm.min(axis=0),
            maximum=norm.max(axis=0),
            spread=np.ptp(norm, axis=0),
            above=np.sum(norm > 1e-10, axis=0),
        )
        summaries[arm] = {k: v.tolist() for k, v in stats.items()}
        for i, lane in enumerate(receipt["lanes"]):
            rows.append(
                [
                    arm,
                    lane,
                    f"{receipt['baseline_norm'][i]:.4e}",
                    f"{stats['median'][i]:.4e}",
                    f"{stats['minimum'][i]:.4e}",
                    f"{stats['maximum'][i]:.4e}",
                    f"{stats['spread'][i]:.4e}",
                    f"{stats['median'][i] / 1e-10:.3f}",
                    int(stats["above"][i]),
                ]
            )
    (args.run / "lane-table.html").write_text(
        table(
            [
                "Device",
                "Lane",
                "At saved state",
                "Median",
                "Minimum",
                "Maximum",
                "Max − min",
                "Median / certificate",
                "Above / 64",
            ],
            rows,
        )
    )
    arrays = np.load(args.run / "gpu-residuals.npz")
    rows, chart = [], []
    for name, row in receipts["gpu"]["stages"].items():
        spread = np.max(np.ptp(arrays[name], axis=0), axis=-1)
        delta = np.max(np.abs(arrays[name] - arrays["baseline"]), axis=-1)
        summary = dict(
            median_change=float(np.median(delta)),
            maximum_change=float(delta.max()),
            median_spread=float(np.median(spread)),
            maximum_spread=float(spread.max()),
        )
        summaries[name] = summary
        rows.append([name.replace("_", " "), *[f"{summary[k]:.4e}" for k in summary]])
        if spread.max() > 0:
            chart.append((name.replace("_", " "), np.median(spread), spread.max()))
    (args.run / "stage-table.html").write_text(
        table(
            [
                "Injected output",
                "Median |Δresidual|∞",
                "Maximum |Δresidual|∞",
                "Median lane spread",
                "Maximum lane spread",
            ],
            rows,
        )
    )
    (args.run / "summary.json").write_text(json.dumps(summaries, indent=2))
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for y, (name, median, maximum) in enumerate(chart):
        ax.hlines(y, median, maximum, linewidth=2.6, color="#929292", zorder=2)
        ax.plot(
            median,
            y,
            "o",
            color="#929292",
            markersize=7,
            zorder=3,
            label="median lane" if y == 0 else None,
        )
        ax.plot(
            maximum,
            y,
            "o",
            color=InkStyle().flux_color,
            markersize=7,
            zorder=4,
            label="largest lane" if y == 0 else None,
        )
    ax.axvline(
        1e-10,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="certificate",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_yticks(range(len(chart)), [c[0] for c in chart])
    ax.invert_yaxis()
    ax.set_xlabel("Residual spread across 64 one-ulp injections")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.14, top=0.88)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(x.rstrip() for x in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.run / "stage-spread.png", dpi=100)
    plt.close(fig)
    print("SUMMARY", json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
