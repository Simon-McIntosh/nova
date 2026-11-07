"""Render the frame 144 limited seed with its classified critical points.

The panel shows the seed flux as unfilled contours inside the DIII-D limiter,
the six critical points the local-index census kept, and the EFIT label axis.
The refused flux maximum is marked apart from the other extrema because it is
the point the axis identification rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
for extra in (
    REPOSITORY,
    REPOSITORY / "scripts" / "prototypes" / "diiid_axis_identification",
    REPOSITORY / "scripts" / "prototypes" / "diiid_forward_convergence",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import measure_branches  # noqa: E402
import benchmarks.diiid_forward_gs_match as bench  # noqa: E402
from nova.media import poloidal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    held = np.load(arguments.cache)
    radius, height, seed = held["radius"], held["height"], held["seed"]
    expansion = float(bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    prepared = measure_branches._prepare_frame(144, expansion)
    limiter = np.asarray(prepared["contour"], dtype=float)
    label_axis = np.asarray(prepared["axis"], dtype=float)

    record = json.loads(arguments.census.read_text())
    kept = record["kept"]
    extrema = [p for p in kept if p["local_index"] > 0]
    saddles = [p for p in kept if p["local_index"] < 0]
    rejected = min(
        extrema,
        key=lambda p: abs(p["r"] - label_axis[0]) + abs(p["z"] - label_axis[1]),
    )
    other_extrema = [p for p in extrema if p is not rejected]

    if "data-ink" in plt.style.available:
        plt.style.use("data-ink")
    figure, axes = plt.subplots(figsize=(14, 7.2))
    levels = poloidal.contour_levels(np.asarray(seed).T, 12)
    poloidal.draw_flux_contours(axes, radius, height, np.asarray(seed).T, levels)
    poloidal.draw_wall(axes, radius=limiter[:, 0], height=limiter[:, 1])
    poloidal.draw_nulls(
        axes,
        magnetic_axis=(rejected["r"], rejected["z"]),
        x_points=np.asarray([(p["r"], p["z"]) for p in saddles]),
        other_x_points=np.asarray([(p["r"], p["z"]) for p in other_extrema]),
        contain=limiter,
    )
    axes.plot(
        label_axis[0],
        label_axis[1],
        marker="+",
        markersize=14,
        color="#2e8b57",
        linestyle="none",
        zorder=20,
    )
    axes.annotate(
        "rejected axis",
        (rejected["r"], rejected["z"]),
        textcoords="offset points",
        xytext=(14, 14),
        color="#2e8b57",
    )
    axes.annotate(
        "EFIT axis",
        (label_axis[0], label_axis[1]),
        textcoords="offset points",
        xytext=(14, -18),
        color="#2e8b57",
    )
    axes.set_aspect("equal")
    axes.set_axis_off()
    figure.savefig(arguments.output)
    print("wrote", arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
