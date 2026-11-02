"""Draw the booking tangent wall for the base and both attempted arithmetic cuts.

Two devices are shown because the change regressed one and improved the other;
a single axis would hide that.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "base"
KERNEL = "shared kernel"
BOTH = "shared kernel\n+ cut buckets"
COLOUR_BASE = "#929292"
COLOUR_KERNEL = "#b06a2c"
COLOUR_BOTH = "#2e8b57"

DATA = {
    "H200, reserved card": (27.85, 40.76, 31.87),
    "P100 on titan": (580.32, 592.86, 508.26),
}


def main(path: Path):
    fig, panels = plt.subplots(1, 2, figsize=(14.0, 7.2))
    for axis, (title, values) in zip(panels, DATA.items(), strict=True):
        bars = axis.barh(
            [2, 1, 0],
            values,
            height=0.55,
            color=(COLOUR_BASE, COLOUR_KERNEL, COLOUR_BOTH),
            zorder=3,
        )
        axis.set_yticks([2, 1, 0])
        axis.set_yticklabels([BASE, "shared kernel only", "shared kernel\n+ cut buckets"])
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
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print("WROTE", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    main(parser.parse_args().path)