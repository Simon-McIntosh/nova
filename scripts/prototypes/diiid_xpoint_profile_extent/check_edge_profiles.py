"""Check that the edge-profile figure states each column's magnitude.

The figure compares three frames, so a reader has to be able to read a height
as a number and to compare that number between the panels above and below it.
Both properties are invisible to the writer's own clipping check: a panel whose
tick labels are suppressed clips nothing and overlaps nothing while carrying no
magnitude at all, and panels that autoscale independently each clip nothing
while the same height means a different value in each. This check therefore
builds the figure through the drawing module with its save step intercepted and
reads the axes themselves.

Two properties, one per column of the grid:

- every panel carries at least ``MINIMUM_LABELS`` printed y tick labels;
- the three panels of a column share one vertical scale, so a height is
  comparable between frames.

The two columns are not required to share a scale: p' and FF' are different
quantities in different units. ``--module`` points the check at another copy of
the drawing module, which is how the check is falsified against a figure that
carries the defect.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MINIMUM_LABELS = 2
PANELS = 6


def load(path):
    """Import the drawing module under a private name, without running it."""

    spec = importlib.util.spec_from_file_location("edge_profile_render_under_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw(module, tables):
    """Build the edge-profile figure and return the axes it was drawn on."""

    captured = {}

    def intercept(figure, axes, notes, svg, png):
        captured["axes"] = list(axes)

    module.save = intercept
    module.edge_profiles(tables, Path("."), Path("."))
    return captured["axes"]


def printed(ax):
    """The y tick labels a reader can actually see on this panel."""

    return [
        label.get_text()
        for label in ax.get_yticklabels()
        if label.get_text().strip() and label.get_visible()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=Path,
        default=Path(__file__).with_name("render.py"),
        help="the drawing module to check",
    )
    parser.add_argument("--tables", type=Path, required=True, help="arm-tables.json")
    args = parser.parse_args()

    axes = draw(load(args.module), json.loads(args.tables.read_text()))
    failures = []
    if len(axes) != PANELS:
        print(f"FAIL drew {len(axes)} panels, expected {PANELS}")
        return 1

    for index, ax in enumerate(axes):
        labels = printed(ax)
        row, column = index // 2 + 1, index % 2 + 1
        if len(labels) < MINIMUM_LABELS:
            failures.append(
                f"row {row} column {column}: {len(labels)} printed y tick "
                f"labels {labels}, need {MINIMUM_LABELS}"
            )
        else:
            print(f"PASS row {row} column {column}: {labels}")

    for column in (0, 1):
        panels = axes[column::2]
        scales = {tuple(round(float(v), 12) for v in ax.get_ylim()) for ax in panels}
        if len(scales) != 1:
            failures.append(
                f"column {column + 1}: {len(scales)} distinct vertical scales "
                f"{sorted(scales)}"
            )
        else:
            (scale,) = scales
            print(f"PASS column {column + 1}: one scale {scale} over 3 frames")

    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print(f"PASS {PANELS} panels, two columns each on one scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())