"""Census one grid cell's geometry from a carrier stage dump.

The Krylov trace reports its first non-finite derivative at one grid cell.  The
claim that the cell is ordinary rather than degenerate is a measurement, so it
is recorded from the stage arrays: section, vertex count, area, area rank,
centre, and the flags the carrier build sets.

Usage: cell_census.py <stage-dir> <cell-index> <out.json>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    stage_dir = Path(sys.argv[1]).resolve()
    cell = int(sys.argv[2])
    out_path = Path(sys.argv[3]).resolve()

    polygons = np.load(stage_dir / "s2_polygons.npz")
    coilset = np.load(stage_dir / "s1_coilset.npz", allow_pickle=True)
    stencil = np.load(stage_dir / "s5_stencil.npz", allow_pickle=True)

    area = np.asarray(polygons["area"], dtype=np.float64)
    offsets = np.asarray(polygons["polygon_offsets"], dtype=np.int64)
    vertices = np.asarray(polygons["polygon_vertices"], dtype=np.float64)
    sections = np.asarray(coilset["sections"], dtype=str)
    centres = np.asarray(coilset["centres"], dtype=np.float64)
    regular = np.asarray(stencil["regular"], dtype=bool)

    order = np.argsort(-area)
    rank = int(np.flatnonzero(order == cell)[0]) + 1
    cell_vertices = vertices[offsets[cell] : offsets[cell + 1]]

    report = {
        "cell_index": cell,
        "section": str(sections[cell]),
        "centre_rz_m": [float(value) for value in centres[cell]],
        "authored_vertex_count": int(len(cell_vertices)),
        "authored_polygon_is_hexagon": bool(len(cell_vertices) == 6),
        "area_m2": float(area[cell]),
        "area_sum_m2": float(area.sum()),
        "area_rank_descending": rank,
        "area_median_m2": float(np.median(area)),
        "area_ratio_to_median": float(area[cell] / np.median(area)),
        "area_z_score": float((area[cell] - area.mean()) / area.std()),
        "regular_flag": bool(regular[cell]),
        "stencil_row": [int(value) for value in stencil["stencil"][cell]],
        "ring_condition": float(stencil["ring_condition"][cell]),
        "grid_cells": int(len(area)),
    }

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
