"""Draw the whole-solve wall per slice against bank width, base against head.

Reads the ``RESULT`` lines the timing driver appended to its ``solve-<width>``
logs, one row per process, and writes one SVG. Wall per slice is the figure's
subject because the section's target is stated per slice: 1 ms.

usage: plot.py <output-dir> <svg-path>
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_MS = 1.0
BASE = "#1a1a1a"
HEAD = "#2e8b57"
GREY = "#929292"


def rows(directory):
    """Collect one row per (arm, matrix-free, width, process)."""
    collected = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("solve-") and name.endswith(".log")):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("RESULT "):
                    continue
                payload = json.loads(line[len("RESULT ") :])
                collected.append(
                    {
                        "width": payload["width"],
                        "matrix_free": payload["matrix_free"],
                        "per_slice_ms": payload["per_slice_ms"],
                        "wall_ms": payload["wall_ms"],
                    }
                )
    return collected


def series(collected):
    """Base and head walls per slice, one entry per width, median over passes."""
    grouped = {}
    for row in collected:
        arm = "head" if row["matrix_free"] else "base"
        grouped.setdefault((arm, row["width"]), []).append(row["per_slice_ms"])
    out = {}
    for (arm, width), values in grouped.items():
        values.sort()
        out[(arm, width)] = values[len(values) // 2]
    return out


def draw(collected, path):
    data = series(collected)
    widths = sorted({width for _, width in data})

    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 7.2), layout="constrained")

    axis.axhline(
        TARGET_MS,
        color=GREY,
        linewidth=1.2,
        linestyle=":",
        zorder=1,
    )
    axis.annotate(
        "1 ms target",
        xy=(widths[0], TARGET_MS),
        xytext=(0, 4),
        textcoords="offset points",
        color=GREY,
        fontsize=14,
        ha="left",
        va="bottom",
    )

    styles = {
        "base": dict(color=BASE, linewidth=2.6, linestyle="--", marker="o", zorder=2),
        "head": dict(color=HEAD, linewidth=3.0, linestyle="-", marker="s", zorder=3),
    }
    for arm in ("base", "head"):
        xs = [width for width in widths if (arm, width) in data]
        ys = [data[(arm, width)] for width in xs]
        if not xs:
            continue
        axis.plot(xs, ys, label=arm, **styles[arm])
        axis.annotate(
            arm,
            xy=(xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=styles[arm]["color"],
            fontsize=14,
            va="center",
            ha="left",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(widths)
    axis.set_xticklabels([str(width) for width in widths])
    axis.minorticks_off()
    axis.set_xlabel("bank width [slices]")
    axis.set_ylabel("whole-solve wall per slice [ms]")
    axis.set_xlim(widths[0] * 0.85, widths[-1] * 1.25)

    figure.savefig(path)
    plt.close(figure)


def main():
    directory, path = sys.argv[1], sys.argv[2]
    collected = rows(directory)
    if not collected:
        raise SystemExit(f"no RESULT lines under {directory}")
    draw(collected, path)
    print(f"# wrote {path} from {len(collected)} rows", flush=True)


if __name__ == "__main__":
    main()
