"""Check that the EFIT reference markers read above the solved ones.

On a converged row the solved axis and EFIT's axis sit within a millimetre of
each other, so two marks of equal size at one zorder leave only the last one
drawn visible and the panel reads as a single null. The drawing code puts the
reference marker above the solved one and makes it larger, so the solved mark
shows inside the hollow reference mark; this renders the 137D row and asserts
that ordering on every panel rather than trusting the picture.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location("marker_render", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--module", type=Path, default=None)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    module = load(args.module or Path(__file__).with_name("render.py"))
    drawn = []
    original = module.draw_nulls

    def record(axes, **kwargs):
        style = kwargs["style"]
        drawn.append(
            dict(
                marker=style.axis_marker,
                zorder=style.zorder_markers,
                size=style.axis_markersize,
                axis=kwargs.get("magnetic_axis"),
                x_point=kwargs.get("x_points")[0],
            )
        )
        return original(axes, **kwargs)

    module.draw_nulls = record
    args.scratch.mkdir(parents=True, exist_ok=True)
    module.xpoint_panels(args.input, 137, "diverted", args.scratch, args.scratch)
    problems = []
    for pair in zip(drawn[0::2], drawn[1::2], strict=True):
        reference, solved = pair
        separation = 1000 * np.linalg.norm(
            np.asarray(reference["axis"]) - np.asarray(solved["axis"])
        )
        print(
            f"panel: EFIT axis {reference['marker']} zorder {reference['zorder']} "
            f"size {reference['size']}, solved axis {solved['marker']} zorder "
            f"{solved['zorder']} size {solved['size']}, axes "
            f"{separation:.3f} mm apart"
        )
        if reference["zorder"] <= solved["zorder"]:
            problems.append("the EFIT marker is not drawn above the solved one")
        if reference["size"] <= solved["size"]:
            problems.append("the EFIT marker is not larger than the solved one")
    if problems:
        for problem in sorted(set(problems)):
            print(f"FAIL {problem}")
        return 1
    print(f"PASS {len(drawn) // 2} panels, the EFIT markers read above the solved ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
