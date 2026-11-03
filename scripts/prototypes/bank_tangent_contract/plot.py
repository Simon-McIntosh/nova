"""Compare warm booking wall and device storage on one H200."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T095720496569-bank-tangent-contract"
)
OUT = Path(
    "docs/figures/sub-millisecond-forward-path/bank-tangent-contract/tangent-wall-memory.svg"
)
COLORS = {"base": "#929292", "head": "#2e8b57"}


def record(arm, label):
    for line in (RUN / f"h200-{arm}.log").read_text().splitlines():
        if line.startswith(label + " "):
            return json.loads(line[len(label) + 1 :])
    raise ValueError(f"missing {label} receipt for {arm}")


def main():
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    categories = (("booking tangent", "whole step"), ("one slice", "16 lanes"))
    series = {
        "base": (
            (record("base", "TANGENT")["wall_ms"], record("base", "STEP")["wall_ms"]),
            (
                record("base", "TANGENT")["memory"]["temp_size_in_bytes"] / 1e9,
                record("base", "BATCH_COMPILED")["memory"]["temp_size_in_bytes"] / 1e9,
            ),
        ),
        "head": (
            (record("head", "TANGENT")["wall_ms"], record("head", "STEP")["wall_ms"]),
            (
                record("head", "TANGENT")["memory"]["temp_size_in_bytes"] / 1e9,
                record("head", "BATCH_COMPILED")["memory"]["temp_size_in_bytes"] / 1e9,
            ),
        ),
    }
    for panel, axis in enumerate(axes):
        for arm, offset in (("base", -0.13), ("head", 0.13)):
            values = series[arm][panel]
            x = np.arange(2) + offset
            axis.scatter(x, values, s=140, color=COLORS[arm], label=arm, zorder=3)
            for position, value in zip(x, values, strict=True):
                axis.text(
                    position,
                    value + (1.1 if panel == 0 else 1.6),
                    f"{value:.1f}",
                    color=COLORS[arm],
                    ha="center",
                    va="bottom",
                )
        axis.set_xticks((0, 1), categories[panel])
        axis.set_xlim(-0.38, 1.38)
        axis.set_ylim(0, 70 if panel else 54)
        axis.set_ylabel("wall [ms]" if panel == 0 else "temporary device memory [GB]")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.82, wspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg")
    svg = OUT.read_text()
    svg = re.sub(
        r'width="[^"]+" height="[^"]+"', 'width="31.1cm" height="16cm"', svg, count=1
    )
    OUT.write_text(svg)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
