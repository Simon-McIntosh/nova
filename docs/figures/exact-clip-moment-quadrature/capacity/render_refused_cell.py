"""Render the refused single-null cell and its row.

Left panel: the 110-cell diverted single-null layout, every atomic cell drawn as
an unfilled outline, with the cell the derived bound refuses highlighted.
Right panel: that cell's own clipped polygon at the raised capacity -- the two
traced level runs (interior vertices) against the straight chain (vertices on
the cell boundary).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import poloidal_axes

HERE = Path(__file__).resolve().parent
REPORT = json.loads((HERE / "refused-cell-diverted-single-null.json").read_text())
OUT = HERE / "diverted-single-null-refused-cell.png"

refused = REPORT["refused_cells"][0]
cell = refused["cell_index"]


def closed(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, points[:1]))


figure, (row_axes, cell_axes) = plt.subplots(1, 2, figsize=(11.0, 5.0))
poloidal_axes(row_axes)
poloidal_axes(cell_axes)

for index, outline in enumerate(REPORT["cell_outlines"]):
    points = np.asarray(outline)
    edge = "#c25f5f" if index == cell else "#6f7b8a"
    width = 0.9 if index == cell else 0.4
    row_axes.plot(
        closed(points)[:, 0], closed(points)[:, 1], color=edge, linewidth=width
    )
row_axes.set_title(
    "diverted single-null, 110 cells: bound 162 = 128 + 34; one cell refused",
    fontsize=9,
)

polygon = np.asarray(refused["polygon"])
interior = np.ones(len(polygon), dtype=bool)
interior[refused["boundary_vertex_indices"]] = False
left, bottom, right, top = refused["cell_bounds"]
cell_axes.plot(
    closed(np.asarray([[left, bottom], [right, bottom], [right, top], [left, top]]))[
        :, 0
    ],
    closed(np.asarray([[left, bottom], [right, bottom], [right, top], [left, top]]))[
        :, 1
    ],
    color="#4a5568",
    linewidth=0.8,
)
cell_axes.plot(
    closed(polygon)[:, 0], closed(polygon)[:, 1], color="#2f4f6f", linewidth=0.7
)
cell_axes.plot(polygon[interior, 0], polygon[interior, 1], ".", ms=1.6, color="#1f77b4")
cell_axes.plot(
    polygon[~interior, 0], polygon[~interior, 1], "o", ms=4.0, color="#c25f5f"
)
cell_axes.set_title(
    "refused cell at capacity 4096: 260 live = 2 x 128 arc samples + 4 straight",
    fontsize=9,
)

figure.tight_layout()
figure.savefig(OUT, dpi=170)
print("REFUSED_CELL_FIGURE", OUT)
