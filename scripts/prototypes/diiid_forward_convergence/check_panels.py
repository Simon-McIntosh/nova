"""Read a rendered panel back and check the two things its legend claims.

The panel says it draws a vessel and marks the label axis.  Both claims are
checkable in the drawing rather than in the code that produced it: the wall
polyline is the set of path vertices drawn in the wall ink, and the label-axis
marker is an open triangle of marker size sitting inside the axes rather than in
the legend.  A rectangle passes the first test only if it is measured, not
assumed, so the vertex count and the number of diagonal segments are reported.
"""

import argparse
import re
from pathlib import Path

PATH = re.compile(r'<path[^>]*d="([^"]+)"[^>]*>', re.S)
STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.S)
PATH_RULES = re.compile(r"(#?[a-z]*\d*)\s*\{([^}]*)\}")


def _wall_stroke(document: str, colour: str) -> list[str]:
    """Return the path data of every element stroked in ``colour``."""

    styles = {}
    for block in STYLE_BLOCK.findall(document):
        for name, body in PATH_RULES.findall(block):
            styles[name.strip()] = body
    found = []
    for match in re.finditer(r"<path([^>]*)>", document, re.S):
        attributes = match.group(1)
        classes = re.findall(r'class="([^"]+)"', attributes)
        stroke = re.findall(r"stroke:\s*(#[0-9a-fA-F]{6})", attributes)
        for name in classes:
            for token in re.split(r"[ ;]+", name):
                body = styles.get(token, "")
                stroke += re.findall(r"stroke:\s*(#[0-9a-fA-F]{6})", body)
        if colour.lower() in {value.lower() for value in stroke}:
            data = re.findall(r'd="([^"]+)"', attributes)
            found.extend(data)
    return found


def _vertices(data: str) -> list[tuple[float, float]]:
    numbers = [float(value) for value in re.findall(r"-?\d+\.?\d*", data)]
    return list(zip(numbers[0::2], numbers[1::2]))


def analyse(svg: Path, colour: str) -> dict:
    document = svg.read_text()
    paths = _wall_stroke(document, colour)
    longest = max(paths, key=lambda data: len(_vertices(data))) if paths else ""
    vertices = _vertices(longest)
    diagonal = 0
    for (r0, z0), (r1, z1) in zip(vertices, vertices[1:]):
        if abs(r1 - r0) > 1e-9 and abs(z1 - z0) > 1e-9:
            diagonal += 1
    # A legend key is drawn outside the axes; a plotted marker is inside a
    # clipped group, so this counts marks the reader can find on the data.
    groups = re.findall(r'<g clip-path="url\([^)]*\)"[^>]*>(.*?)</g>', document, re.S)
    markers = sum(len(re.findall(r"<use ", group)) for group in groups)
    return {
        "svg": str(svg),
        "wall_paths": len(paths),
        "wall_vertices": len(vertices),
        "wall_diagonal_segments": diagonal,
        "markers_in_axes": markers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("svgs", type=Path, nargs="+")
    parser.add_argument("--wall-colour", default="#000000")
    parser.add_argument("--min-vertices", type=int, default=40)
    parser.add_argument("--min-markers", type=int, default=3)
    arguments = parser.parse_args()
    failures = []
    for svg in arguments.svgs:
        result = analyse(svg, arguments.wall_colour)
        vessel = result["wall_vertices"] >= arguments.min_vertices
        shaped = result["wall_diagonal_segments"] > 0
        marked = result["markers_in_axes"] >= arguments.min_markers
        print(
            f"{svg.name:34s} wall_vertices={result['wall_vertices']:3d} "
            f"diagonal_segments={result['wall_diagonal_segments']:3d} "
            f"markers_in_axes={result['markers_in_axes']:2d} "
            f"vessel={'yes' if vessel and shaped else 'NO (rectangle)'}"
            f" markers={'yes' if marked else 'MISSING'}"
        )
        if not (vessel and shaped and marked):
            failures.append(svg.name)
    if failures:
        print(f"REFUSED {failures}")
        return 1
    print("PANEL WALL CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
