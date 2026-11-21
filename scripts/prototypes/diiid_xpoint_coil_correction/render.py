"""Draw the paired lower-X zooms for the measured and corrected current arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.prototypes.diiid_xpoint_profile_extent import render  # noqa: E402

render.ARMS = ("base", "corrected")
render.NAMES = {"base": "measured currents", "corrected": "fitted currents"}
ROWS = ((137, "diverted"), (179, "limited"), (146, "diverted"), (146, "limited"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    for frame, branch in ROWS:
        render.xpoint_panels(args.input, frame, branch, args.output, args.previews)


if __name__ == "__main__":
    main()
