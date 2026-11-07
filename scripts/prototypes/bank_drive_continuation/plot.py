"""Reads the batch makes against the drive step, direct against continuation.

Reads the measurement's own JSONL so the figure and the record beside it are
drawn from the same numbers. Reads are the batch's fresh evaluations at the
slice, seed read and terminal certificate included. A continuation's reads are
summed over its sub-steps, because the solve result it ends on carries only the
final sub-step's count.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.media.ink import DEFAULT_INK, trace_axes

CANVAS = (14.0, 7.2)
STYLE = {
    "direct": {"color": "#1f6feb", "marker": "s", "linestyle": "--", "zorder": 3},
    "continuation": {"color": "#2e8b57", "marker": "o",
                     "linestyle": "-", "zorder": 2},
}
LABELS = {"direct": "direct solve", "continuation": "drive continuation"}


def style():
    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    plt.rcParams.update({"axes.labelsize": 16, "legend.fontsize": 14})
    return DEFAULT_INK.variant(label_fontsize=14.0)


def total_reads(row):
    substeps = row.get("substep_rows") or []
    if substeps:
        return sum(substep["reads"] for substep in substeps)
    return row["reads_batch"]


def read(paths):
    rows = {name: {} for name in STYLE}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.startswith("ARM "):
                    continue
                row = json.loads(line[4:])
                key = float(row["step"])
                slot = rows[row["arm"]]
                if key in slot and total_reads(slot[key]) >= total_reads(row):
                    continue
                slot[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", type=float, default=1.0)
    args = parser.parse_args()
    rows = read(args.measure)
    ink = style()
    figure, axes = plt.subplots(figsize=CANVAS, dpi=100)
    trace_axes(axes, ink)
    for name, spec in STYLE.items():
        points = sorted(rows[name].items())
        if not points:
            continue
        steps = [step for step, _ in points]
        reads = [total_reads(row) for _, row in points]
        settled = [
            row["converged"] == row["lanes"] and row.get("certified") is not False
            for _, row in points
        ]
        axes.plot(
            steps,
            reads,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.6,
            zorder=spec["zorder"],
            label=LABELS[name],
        )
        for (step, value, done) in zip(steps, reads, settled):
            axes.plot(
                step,
                value,
                linestyle="none",
                marker=spec["marker"],
                markersize=9,
                markeredgewidth=1.6,
                markeredgecolor=spec["color"],
                markerfacecolor=spec["color"] if done else "white",
                zorder=spec["zorder"] + 2,
            )
        offset = 12 if name == "direct" else -20
        for step, value in zip(steps, reads):
            axes.annotate(
                str(value),
                (step, value),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=14,
                color=spec["color"],
            )
    axes.set_xscale("log")
    axes.set_yscale("log")
    ticks = sorted({step for slot in rows.values() for step in slot})
    axes.set_xticks(ticks)
    axes.set_xlim(ticks[0] / 2.0, ticks[-1] * 2.0)
    axes.set_ylim(2.0, 4000.0)
    axes.set_xlabel("requested drive step (relative amplitude)")
    axes.set_ylabel("batch reads per slice")
    axes.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(args.out, format="svg")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()