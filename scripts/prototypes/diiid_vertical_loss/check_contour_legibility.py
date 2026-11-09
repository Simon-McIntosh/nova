"""Refuse a panel whose contour lines have fused into a block of colour.

A contour set drawn wider than the gap between neighbouring levels fills the
space between them, and a filled band carries no level information: the reader
cannot tell one contour from the next.  A drawn line is thin in the direction
across it and long along it; a fused band is thick in both, so a pixel whose
horizontal and vertical runs are both longer than one line is inside a block.

Run it on the rasterised panel, since the fusion is a property of what the
reader sees rather than of the level array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

SOLVED_FLUX = np.array([0x33, 0x66, 0xCC], dtype=float)
TOLERANCE = 0.18
"""Fractional RGB distance from the solved-flux colour that still counts as it."""

FUSED_RUN = 9
"""Run length [px] above which a line is wide enough to be a fused band.

A single solved contour is 3.0 pt, which is 4.6 px at 110 dpi; a band two lines
wide is 9 px, so the threshold sits just above one line and well below two.
"""

BLOCK_PIXELS = 100
"""Fused pixels above which a panel carries a block rather than a few corners.

A handful of thick pixels is where two contours cross or a line turns sharply,
which is read as two lines meeting rather than as a filled region.  The defect
this check exists for fills thousands.
"""


def _runs(mask: np.ndarray) -> np.ndarray:
    """Return, at every pixel, the length of the run through it along axis 1."""

    runs = np.zeros(mask.shape, dtype=int)
    for row in range(mask.shape[0]):
        line = mask[row]
        if not line.any():
            continue
        edges = np.diff(np.concatenate(([0], line.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends, strict=False):
            runs[row, start:end] = end - start
    return runs


def measure(png: Path) -> dict:
    image = mpimg.imread(png)[..., :3]
    if image.dtype != np.uint8:
        image = image * 255.0
    mask = np.linalg.norm(image - SOLVED_FLUX, axis=2) < TOLERANCE * 255.0
    across = _runs(mask)
    along = _runs(mask.T).T
    thickness = np.minimum(across, along)
    fused = thickness > FUSED_RUN
    rows = np.flatnonzero(fused.any(axis=1))
    return {
        "png": str(png),
        "solved_pixels": int(mask.sum()),
        "fused_pixels": int(fused.sum()),
        "fused_rows": [int(rows.min()), int(rows.max())] if len(rows) else [],
        "max_thickness": int(thickness.max()) if mask.any() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()
    records = [measure(path) for path in arguments.png]
    for record in records:
        print(json.dumps(record))
    if arguments.json is not None:
        arguments.json.write_text(json.dumps({"panels": records}, indent=2) + "\n")
    fused = [record for record in records if record["fused_pixels"] > BLOCK_PIXELS]
    if fused:
        names = ", ".join(Path(record["png"]).name for record in fused)
        raise SystemExit(f"CONTOURS FUSED: {names}")
    print("CONTOURS LEGIBLE")


if __name__ == "__main__":
    main()
