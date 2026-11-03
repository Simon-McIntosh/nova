"""Draw the occupied and idle parts of a single warm Newton call."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def row(path: Path):
    records = [
        json.loads(line.partition(" ")[2])
        for line in path.read_text().splitlines()
        if line.startswith("TIMELINE ")
    ]
    assert len(records) == 1
    return records[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bucketed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = [("Base", row(args.base)), ("Bucketed", row(args.bucketed))]
    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 7.2))
    busy_color = "#2e8b57"
    idle_color = "#929292"
    for position, (name, values) in enumerate(data):
        busy = values["gpu_busy_ms"]
        idle = values["wall_ms"] - busy
        assert idle >= 0
        axis.barh(
            position,
            busy,
            height=0.44,
            color=busy_color,
            label="GPU busy" if position == 0 else None,
        )
        axis.barh(
            position,
            idle,
            left=busy,
            height=0.44,
            color=idle_color,
            label="GPU idle" if position == 0 else None,
        )
        axis.text(
            values["wall_ms"] + 0.35,
            position,
            f"{values['wall_ms']:.2f} ms",
            va="center",
            fontsize=14,
        )
    axis.set_yticks([0, 1], [name for name, _ in data])
    axis.invert_yaxis()
    axis.set_xlim(0, max(values["wall_ms"] for _, values in data) * 1.18)
    axis.set_xlabel("Whole warm Newton step [ms]")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.82)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
