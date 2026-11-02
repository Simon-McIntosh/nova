"""Draw the booking tangent wall read straight from the stage profile logs.

Every bar is the median of the three synchronized calls the profile program
printed, so the figure cannot drift from the logs it cites.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE = "local_booking_tangent"
COLOUR_BASE = "#929292"
COLOUR_KERNEL = "#b06a2a"
COLOUR_BOTH = "#2e8b57"
ROWS = (
    ("base", COLOUR_BASE),
    ("shared kernel only", COLOUR_KERNEL),
    ("shared kernel + cut buckets", COLOUR_BOTH),
)
DEVICES = (
    ("H200, reserved card", ("base-h200", "head-h200k", "head-h200")),
    ("P100 on titan", ("base-p100", "head-p100k", "head-p100")),
)


def stage_median(log: Path) -> float:
    for line in log.read_text().splitlines():
        if not line.startswith("STAGE {"):
            continue
        event = json.loads(line[len("STAGE ") :])
        if event["name"] == STAGE:
            return event["median_ms"]
    raise ValueError(f"no {STAGE} stage line in {log}")


def main(run_dir: Path, path: Path):
    measured = {}
    for title, tags in DEVICES:
        measured[title] = [stage_median(run_dir / f"{tag}-profile.log") for tag in tags]
    fig, panels = plt.subplots(1, 2, figsize=(14.0, 7.2))
    for axis, (title, values) in zip(panels, measured.items(), strict=True):
        bars = axis.barh(
            [2, 1, 0],
            values,
            height=0.55,
            color=[colour for _, colour in ROWS],
            zorder=3,
        )
        axis.set_yticks([2, 1, 0])
        axis.set_yticklabels([name for name, _ in ROWS])
        axis.set_xlabel("warm booking tangent [ms]")
        axis.set_title(title, loc="left")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=12,
            )
        axis.set_xlim(0, max(values) * 1.18)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    print("WROTE", path)
    for title, values in measured.items():
        print("READ", title, [round(value, 3) for value in values])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.path)
