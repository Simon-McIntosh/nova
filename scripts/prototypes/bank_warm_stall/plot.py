"""Lane 0's fresh residual against update, warm-started against cold.

Reads the reduced record committed beside the instrument, so the figure and the
table it sits beside are drawn from the same numbers. The two arms coincide for
the first ten updates, so they are separated by marker and dash as well as by
colour; the cold arm is drawn over the warm one for the same reason.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.media.ink import DEFAULT_INK, trace_axes

CANVAS = (14.0, 7.2)
STYLE = {
    "warm": {"color": "#2f6fd0", "marker": "o", "linestyle": "-", "zorder": 2},
    "cold": {"color": "#d97b16", "marker": "s", "linestyle": "--", "zorder": 3},
}
LABELS = {"warm": "warm start", "cold": "cold start"}


def style():
    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    plt.rcParams.update({"axes.labelsize": 16, "legend.fontsize": 14})
    return DEFAULT_INK.variant(label_fontsize=14.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    record = json.load(open(args.report))
    ink = style()
    figure, axes = plt.subplots(figsize=CANVAS, dpi=100)
    trace_axes(axes, ink)
    low, high = math.inf, 0.0
    for arm in ("warm", "cold"):
        path = record[arm]["residual_path"]
        low, high = min(low, min(path)), max(high, max(path))
        axes.plot(
            range(len(path)),
            path,
            markersize=5,
            markevery=3,
            linewidth=3.0 if arm == "warm" else 2.6,
            label=LABELS[arm],
            **STYLE[arm],
        )
    axes.set_xlabel("update")
    axes.set_ylabel("fresh residual [-]")
    axes.set_xlim(-1.2, len(record["warm"]["residual_path"]) - 0.3)
    axes.set_ylim(low * 0.85, high * 1.12)
    axes.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    decades = math.log10(low / args.tolerance)
    axes.annotate(
        f"tolerance {args.tolerance:.0e}\n{decades:.1f} decades below this range",
        xy=(0.985, 0.06),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=ink.label_fontsize,
        color="#6b6b6b",
    )
    figure.tight_layout()
    figure.savefig(args.out, format="svg")
    figure.savefig(args.out.rsplit(".", 1)[0] + ".png", format="png", dpi=140)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
