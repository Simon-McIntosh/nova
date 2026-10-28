"""Plot milliseconds per Newton-step stage against the 1 ms target.

Reads the job receipt, so the figure cannot drift from the measurement it
states. The stages are drawn as dots with their values, not as bars: a bar's
length is read as a magnitude, and across three decades a log bar overstates the
small stages and clips the large one, while a dot carries no length to misread
and the number is written beside it. The axis is logarithmic only so the stages
spread apart; every value is labelled. The batch arm is not a stage of one step and sits below a gap, in grey.
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
    ("jacobian_block_ms", "Jacobian block"),
    ("linear_solve_ms", "linear solve"),
)
SUBJECT = "#3b6ea5"
BATCH = "#929292"
LIMIT = "#b03030"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    step = (receipt.get("steps") or {}).get("1", {})
    rows = [(label, step[key], SUBJECT) for key, label in STAGES if key in step]
    batch = (receipt.get("steps") or {}).get("16", {})
    separated = "map_per_solve_ms" in batch
    if separated:
        rows.append(("map at width 16, per slice", batch["map_per_solve_ms"], BATCH))

    plt.style.use(TARGET)
    figure, axes = plt.subplots(figsize=(35.56 * CM, 18.29 * CM))
    positions = []
    cursor = len(rows) - 1
    for index in range(len(rows)):
        if separated and index == len(rows) - 1:
            cursor -= 0.9
        positions.append(cursor)
        cursor -= 1
    values = np.array([value for _, value, _ in rows])

    for position, (label, value, colour) in zip(positions, rows):
        axes.plot([value], [position], marker="o", markersize=9, color=colour, zorder=3)
        axes.annotate(
            f"{value:.1f}",
            xy=(value, position),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=14,
            color="#222222",
        )

    axes.set_yticks(positions, [label for label, _, _ in rows])
    axes.set_xscale("log")
    axes.set_xlim(0.8, values.max() * 2.2)
    axes.set_ylim(min(positions) - 0.8, max(positions) + 0.8)
    axes.set_xlabel("wall per stage [ms]", fontsize=16)
    axes.tick_params(labelsize=14)
    axes.axvline(1.0, color=LIMIT, linewidth=1.2, linestyle=":", zorder=2)
    axes.annotate(
        "1 ms target",
        xy=(1.0, max(positions) + 0.7),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=LIMIT,
        fontsize=14,
    )
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(False)
    axes.set_title("")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format=args.output.suffix.lstrip("."))
    print(
        f"WROTE {args.output} rows={[(label, round(value, 3)) for label, value, _ in rows]}"
    )


if __name__ == "__main__":
    main()
