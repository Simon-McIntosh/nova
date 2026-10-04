"""Draw two measured terminal fields on common poloidal contour levels."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--receipt-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
DRIVER = Path(__file__).with_name("measure.py")
spec = importlib.util.spec_from_file_location("gate_frame_instrument", DRIVER)
assert spec and spec.loader
instrument = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = instrument
spec.loader.exec_module(instrument)

slug = "d3d-shot-0003ff34e7-parquet-frame-44"
receipts = [
    json.loads((arguments.receipt_root / label / f"{slug}.json").read_text())
    for label in ("original", "current")
]
arrays = [instrument._load_terminal_flux_artifact(receipt) for receipt in receipts]
assert arrays[0]["terminal_grid_wb"].shape == arrays[1]["terminal_grid_wb"].shape
for field in (
    "radius_m",
    "height_m",
    "wall_coordinate_m",
    "wall_unit_offsets",
    "wall_unit_closed",
    "wall_unit_kinds",
):
    assert np.array_equal(arrays[0][field], arrays[1][field]), field
units = instrument._wall_units_from_arrays(arrays[0])
levels = poloidal.contour_levels(
    np.concatenate([a["terminal_grid_wb"].ravel() for a in arrays]), count=15
)
fig, ax = plt.subplots(figsize=(14, 8))
poloidal_axes(ax)
colors = ("#3b6ea8", "#a65e2e")
names = ("Artifact solver", "Current solver")
styles = ("dashed", "solid")
for receipt, data, color, name, style in zip(
    receipts, arrays, colors, names, styles, strict=True
):
    contours = poloidal.draw_flux_contours(
        ax,
        data["radius_m"],
        data["height_m"],
        data["terminal_grid_wb"].T,
        levels,
        color=color,
        linewidth=3.0 if style == "solid" else 2.6,
        wall=units,
    )
    contours.set_linestyle(style)
    segments = [
        segment
        for collection in contours.allsegs
        for segment in collection
        if len(segment) > 1
    ]
    assert segments, name
    segment = max(segments, key=len)
    point = segment[
        int(np.argmax(segment[:, 0]) if style == "dashed" else np.argmin(segment[:, 0]))
    ]
    ax.annotate(
        name,
        xy=point,
        xytext=(20 if style == "dashed" else -20, 0),
        textcoords="offset points",
        ha="left" if style == "dashed" else "right",
        va="center",
        color=color,
        fontsize=20,
    )
    instrument._draw_topology(
        ax,
        receipt["terminal_flux_artifact"]["terminal_topology"],
        units,
        DEFAULT_INK.variant(
            axis_color=color,
            xpoint_color=color,
            axis_markersize=11 if style == "dashed" else 7,
            xpoint_markersize=13 if style == "dashed" else 9,
        ),
    )
poloidal.draw_wall(ax, units=units, linewidth=2.6)
assert not ax.axison
fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
target = arguments.output
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, dpi=100)
plt.close(fig)
print(
    "FIGURE",
    json.dumps(
        {
            "path": str(target),
            "levels": levels.tolist(),
            "residuals": [
                receipt["main"]["with_exit"]["terminal_residual"]
                for receipt in receipts
            ],
            "converged": [
                receipt["main"]["with_exit"]["converged"] for receipt in receipts
            ],
            "null_sets": 2,
            "wall_units": len(units),
            "axis_off": True,
        },
        sort_keys=True,
    ),
)
