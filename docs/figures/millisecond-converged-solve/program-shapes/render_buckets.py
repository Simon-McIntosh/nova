"""Render the bucket-padding and program-count figure for the compile-cost evidence.

Reads the census receipt and writes one PNG beside it; no solver code runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "program-shape-census.json"
OUTPUT = HERE / "program-shape-buckets.png"


def main() -> int:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    waste = receipt["combined"]["padding_waste"]
    axes = sorted(
        waste,
        key=lambda name: -max(waste[name]["padding_ratio"].values()),
    )
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    height = 0.9 / len(axes)
    for index, name in enumerate(axes):
        ratios = sorted(
            waste[name]["padding_ratio"].items(), key=lambda kv: float(kv[0])
        )
        left.barh(
            [index + offset * height for offset, _ in enumerate(ratios)],
            [ratio for _, ratio in ratios],
            height=0.8 * height,
            label=name if index == 0 else None,
        )
    left.set_yticks([index + 0.45 / len(axes) for index in range(len(axes))])
    left.set_yticklabels(axes, fontsize=8)
    left.axvline(1.0, color="black", linewidth=0.8, linestyle="dashed")
    left.set_xlabel("padding ratio: bucketed floor over realised capacity", fontsize=8)
    left.set_title("padding waste per capacity axis", fontsize=9, loc="left")
    left.tick_params(labelsize=8)

    bank_programs = receipt["bank"]["program_count_today"]
    certificate_programs = receipt["certificate"]["distinct_capacity_vectors_today"]
    today = [bank_programs, certificate_programs]
    design = [1, receipt["certificate"]["distinct_bucketed_vectors"]]
    positions = [0, 1]
    width = 0.36
    right.bar(
        [p - width / 2 for p in positions],
        today,
        width,
        label=f"today ({sum(today)} programs)",
        color="#3b6ea5",
    )
    right.bar(
        [p + width / 2 for p in positions],
        design,
        width,
        label=f"one per bucket ({sum(design)})",
        color="#a5563b",
    )
    for p, value in zip(positions, today):
        right.text(p - width / 2, value + 0.15, str(value), ha="center", fontsize=8)
    for p, value in zip(positions, design):
        right.text(p + width / 2, value + 0.15, str(value), ha="center", fontsize=8)
    right.set_xticks(positions)
    right.set_xticklabels(["bank\n(12 arms)", "certificate\n(4 cases)"], fontsize=8)
    right.set_ylabel("compiled programs", fontsize=8)
    right.set_title("program count", fontsize=9, loc="left")
    right.tick_params(labelsize=8)
    right.legend(fontsize=7)
    right.set_ylim(0, max(today + design) * 1.35)

    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=160)
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
