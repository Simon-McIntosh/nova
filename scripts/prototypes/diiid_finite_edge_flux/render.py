"""Draw the base and finite-edge terminal flux panels for every gate row.

The panel painter is the committed lower-X instrument, which draws solved and
EFIT contours on shared levels with the wall and both null sets; it reads its
pair of arms from module state, so this driver sets the arms this node solved.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "extent_render",
    ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/render.py",
)
extent_render = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extent_render
SPEC.loader.exec_module(extent_render)

ROWS = (
    (137, "diverted"),
    (146, "diverted"),
    (146, "limited"),
    (179, "limited"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    extent_render.ARMS = ("base", "head")
    extent_render.NAMES = {"base": "base tables", "head": "EFIT finite edge"}
    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    args.output.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    for frame, branch in ROWS:
        extent_render.xpoint_panels(
            args.input, frame, branch, args.output, args.previews
        )
        print(f"PANELS frame-{frame}-{branch}", flush=True)


if __name__ == "__main__":
    main()
