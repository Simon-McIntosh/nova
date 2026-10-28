"""Plot milliseconds per Newton-step stage against the 1 ms target.

Reads the job receipt, so the figure cannot drift from the measurement it
states. The x axis is logarithmic because the stages span 1.6 ms to 168 ms,
about two decades against a 1 ms target.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TARGET = "data-ink"
CM = 1 / 2.54
STAGES = (
    ("topology_read_ms", "topology read"),
    ("booking_ms", "booking"),
    ("map_ms", "map"),
    ("linear_solve_ms", "linear solve"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    step = (receipt.get("steps") or {}).get("1", {})
    rows = [(label, step[key]) for key, label in STAGES if key in step]
    batch = (receipt.get("steps") or {}).get("16", {})
    if "map_per_solve_ms" in batch:
        rows.append(("map, width 16 per slice", batch["map_per_solve_ms"]))

    plt.style.use(TARGET)
    figure, axes = plt.subplots(figsize=(35.56 * CM, 18.29 * CM))
    values = np.array([value for _, value in rows])
    positions = np.arange(len(rows))[::-1]
    subject = ["#3b6ea5" if "width 16" not in label else "#929292" for label, _ in rows]
    axes.barh(positions, values, height=0.55, color=subject, zorder=3)
    axes.set_yticks(positions, [label for label, _ in rows])
    axes.set_xscale("log")
    axes.set_xlabel("wall per stage [ms]")
    axes.axvline(1.0, color="#b03030", linewidth=1.2, linestyle=":", zorder=4)
    axes.annotate(
        "1 ms target",
        xy=(1.0, positions.max() + 0.55),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#b03030",
        fontsize=11,
    )
    for position, value in zip(positions, values):
        axes.annotate(
            f"{value:.1f}",
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
            color="#222222",
        )
    axes.set_xlim(right=values.max() * 3)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(False)
    axes.set_title("")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format=args.output.suffix.lstrip("."))
    print(f"WROTE {args.output} rows={[(label, round(value, 3)) for label, value in rows]}")


if __name__ == "__main__":
    main()