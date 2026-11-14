"""Check that refused-cell circles remain visible around coincident cutoff stars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SVG = "{http://www.w3.org/2000/svg}"
HREF = "{http://www.w3.org/1999/xlink}href"
REFUSED = "#bf3446"


def expected_marks(folder):
    spatial = json.loads((folder / "spatial.json").read_text())
    last_inner = len(spatial["trips"]) - 1
    refused = {
        tuple(point)
        for proposal in spatial["proposals"]
        if proposal["inner"] == last_inner and not proposal["accepted"]
        for point in proposal["comparisons"][0]["changed_rz_m"]
    }
    cutoff = json.loads((folder / "cutoff-jumps.json").read_text())[
        "smallest_crossing_pair"
    ]
    crossing = {tuple(point) for point in cutoff["crossing_rz_m"]} if cutoff else set()
    return refused, crossing


def marker_instances(root):
    elements = {element.get("id"): element for element in root.iter()}
    circles, stars = [], []
    for mark in elements["axes_2"].iter(SVG + "use"):
        style = dict(
            entry.strip().split(": ", 1)
            for entry in mark.get("style", "").split(";")
            if ": " in entry
        )
        if style.get("stroke") != REFUSED:
            continue
        path = elements[mark.get(HREF)[1:]].get("d")
        coordinates = list(
            map(float, re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", path))
        )
        points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
        width = float(style.get("stroke-width", 1))
        radius = (
            max(abs(value) for value in coordinates)
            if "C" in path
            else max(math.hypot(*point) for point in points)
        )
        record = {
            "xy": (float(mark.get("x")), float(mark.get("y"))),
            "inner": radius - width / 2,
            "outer": radius + width / 2,
        }
        (circles if "C" in path else stars).append(record)
    return circles, stars


def check(figures, cohort, case):
    folder = cohort / "edge" / case if case.startswith("frame-") else cohort / case
    refused, cutoff = expected_marks(folder)
    root = ET.parse(figures / f"{case}.svg").getroot()
    circles, stars = marker_instances(root)
    errors = []
    if len(circles) != len(refused):
        errors.append(f"{len(circles)} circle marks for {len(refused)} refused cells")
    if len(stars) != len(cutoff):
        errors.append(f"{len(stars)} star marks for {len(cutoff)} cutoff cells")
    footer = f"{len(refused)} cells change in terminal refused proposals"
    if footer not in [element.text for element in root.iter(SVG + "text")]:
        errors.append("footer does not match the receipt's refused-cell count")
    gaps = [
        circle["inner"] - star["outer"]
        for circle in circles
        for star in stars
        if math.dist(circle["xy"], star["xy"]) < 1e-6
    ]
    if len(gaps) != len(refused & cutoff):
        errors.append(
            "coincident rendered markers do not match the receipt intersection"
        )
    for gap in gaps:
        if gap < 1.0:
            errors.append(
                f"cutoff star obscures its refused-cell circle: gap={gap:.3f} pt"
            )
    return {
        "case": case,
        "refused_cells": len(refused),
        "circle_marks": len(circles),
        "cutoff_cells": len(cutoff),
        "coincident_cells": len(gaps),
        "circle_star_clearance_pt": gaps,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--only", nargs="+", required=True)
    args = parser.parse_args()
    rows = [check(args.figures, args.input, case) for case in args.only]
    for row in rows:
        print("MARKER_VISIBILITY " + json.dumps(row))
    failed = sum(bool(row["errors"]) for row in rows)
    print(f"MARKER_SUMMARY cases={len(rows)} failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
