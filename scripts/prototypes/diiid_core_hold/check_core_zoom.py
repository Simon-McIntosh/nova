"""Check that the saved core zoom separates its maxima and saddles."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from scripts.prototypes.diiid_xpoint_offset.style import SOLVED


SVG = "{http://www.w3.org/2000/svg}"
LINK = "{http://www.w3.org/1999/xlink}href"


def check_core_zoom(path: Path):
    """Require all seven core markers and visible space between their strokes."""
    root = ET.parse(path).getroot()
    panel = root.find(f".//{SVG}g[@id='axes_2']")
    if panel is None:
        raise ValueError("base core zoom is absent")
    definitions = {p.get("id"): p for p in root.iter(SVG + "path") if p.get("id")}
    markers = []
    for use in panel.iter(SVG + "use"):
        style = use.get("style", "")
        if f"fill: {SOLVED}" not in style:
            continue
        definition = definitions[use.get(LINK).removeprefix("#")]
        coordinates = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", definition.get("d")
        )
        vertices = list(
            zip(
                map(float, coordinates[::2]), map(float, coordinates[1::2]), strict=True
            )
        )
        distinct = len(set(vertices))
        kind = {3: "maximum", 12: "saddle"}.get(distinct)
        if kind is None:
            raise ValueError(f"unrecognized filled core marker: {distinct} vertices")
        diameter = max(
            max(p[i] for p in vertices) - min(p[i] for p in vertices) for i in (0, 1)
        )
        # Include a conservative stroke allowance for both marker shapes.
        markers.append(
            dict(
                kind=kind,
                x=float(use.get("x")),
                y=float(use.get("y")),
                diameter_pt=diameter + 1.2,
            )
        )
    counts = {
        kind: sum(p["kind"] == kind for p in markers) for kind in ("maximum", "saddle")
    }
    if counts != {"maximum": 4, "saddle": 3}:
        raise ValueError(f"expected four maxima and three saddles, observed {counts}")
    pairs = []
    for a, b in itertools.combinations(markers, 2):
        distance = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        pairs.append(
            dict(
                kinds=[a["kind"], b["kind"]],
                distance_pt=distance,
                separation_in_marker_diameters=distance
                / max(a["diameter_pt"], b["diameter_pt"]),
            )
        )
    closest = min(pairs, key=lambda p: p["separation_in_marker_diameters"])
    result = dict(counts=counts, closest_pair=closest, minimum_required_diameters=1.25)
    print("CORE_ZOOM " + json.dumps(result), flush=True)
    if closest["separation_in_marker_diameters"] < 1.25:
        raise ValueError(
            "core critical-point markers overlap or lack visible separation"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    check_core_zoom(args.figure)


if __name__ == "__main__":
    main()
