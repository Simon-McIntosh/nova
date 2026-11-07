"""One lane's fresh residual against update, warm-started against cold."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.media.ink import DEFAULT_INK, trace_axes

CANVAS = (14.0, 7.2)
COLOURS = {"warm": "#2f6fd0", "cold": "#d97b16"}
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
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    record = json.load(open(args.report))
    ink = style()
    figure, axes = plt.subplots(figsize=CANVAS, dpi=100)
    trace_axes(axes, ink)
    for arm in ("cold", "warm"):
        rows = record[arm]["table"]
        xs = [row["update"] for row in rows]
        ys = [row["residual"] for row in rows]
        axes.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            markevery=3,
            linewidth=3.0 if arm == "warm" else 2.6,
            color=COLOURS[arm],
            label=LABELS[arm],
            zorder=3 if arm == "warm" else 2,
        )
    axes.axhline(
        args.tolerance, color="#929292", linewidth=1.2, linestyle=":", zorder=1
    )
    axes.annotate(
        "tolerance",
        xy=(axes.get_xlim()[1], args.tolerance),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=ink.label_fontsize,
        color="#6b6b6b",
    )
    axes.set_xlabel("update")
    axes.set_ylabel("fresh residual")
    axes.set_yscale("log")
    axes.set_xlim(
        -1.2, max(len(record[arm]["table"]) for arm in ("warm", "cold")) - 0.3
    )
    axes.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    figure.tight_layout()
    figure.savefig(args.out, format="svg")
    figure.savefig(args.out.rsplit(".", 1)[0] + ".png", format="png", dpi=140)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
