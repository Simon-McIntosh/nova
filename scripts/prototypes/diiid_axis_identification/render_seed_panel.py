"""Render the frame 144 limited seed with its classified critical points.

The panel shows the seed flux as unfilled contours inside the DIII-D limiter,
the six critical points the local-index census kept, and the EFIT label axis.
The flux maximum nearest the label axis is marked apart from the census
because it is the point the axis identification rejected.
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
from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402

FEATURE = "#2e8b57"


def _handle(marker: str, filled: bool, color: str, label: str):
    return plt.Line2D(
        [],
        [],
        marker=marker,
        linestyle="none",
        color=color,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markersize=7.5,
        markeredgewidth=1.2,
        label=label,
    )


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
    label_axis = np.asarray(prepared["axis"], dtype=float).reshape(-1)[:2]

    record = json.loads(arguments.census.read_text())
    kept = record["kept"]
    extrema = [p for p in kept if p["local_index"] > 0]
    saddles = [p for p in kept if p["local_index"] < 0]
    rejected = min(
        extrema,
        key=lambda p: float(np.hypot(p["r"] - label_axis[0], p["z"] - label_axis[1])),
    )
    other_extrema = [p for p in extrema if p is not rejected]

    if "data-ink" in plt.style.available:
        plt.style.use("data-ink")
    figure = plt.figure(figsize=(14.0, 7.2))
    axes = figure.add_axes([0.02, 0.03, 0.46, 0.94])
    poloidal_axes(axes)

    field = np.asarray(seed).T
    levels = poloidal.contour_levels(field, 12)
    poloidal.draw_flux_contours(axes, radius, height, field, levels)
    poloidal.draw_wall(axes, radius=limiter[:, 0], height=limiter[:, 1])

    tally = poloidal.draw_nulls(
        axes,
        magnetic_axis=(rejected["r"], rejected["z"]),
        x_points=np.asarray([(p["r"], p["z"]) for p in saddles]),
        other_x_points=np.asarray([(p["r"], p["z"]) for p in other_extrema]),
        contain=limiter,
    )
    axes.plot(
        rejected["r"],
        rejected["z"],
        marker="o",
        markersize=14,
        markerfacecolor="none",
        markeredgecolor=FEATURE,
        markeredgewidth=1.6,
        linestyle="none",
        zorder=25,
    )
    axes.plot(
        label_axis[0],
        label_axis[1],
        marker="+",
        markersize=13,
        color=FEATURE,
        linestyle="none",
        zorder=25,
    )
    axes.annotate(
        "rejected axis",
        (rejected["r"], rejected["z"]),
        textcoords="offset points",
        xytext=(22, 26),
        color=FEATURE,
        fontsize=14,
        arrowprops={"arrowstyle": "-", "color": FEATURE, "linewidth": 1.0},
        zorder=26,
    )
    axes.annotate(
        "EFIT axis",
        (label_axis[0], label_axis[1]),
        textcoords="offset points",
        xytext=(-96, -22),
        color=FEATURE,
        fontsize=14,
        arrowprops={"arrowstyle": "-", "color": FEATURE, "linewidth": 1.0},
        zorder=26,
    )

    margin = 0.18
    axes.set_xlim(
        float(limiter[:, 0].min()) - margin, float(limiter[:, 0].max()) + margin
    )
    axes.set_ylim(
        float(limiter[:, 1].min()) - margin, float(limiter[:, 1].max()) + margin
    )

    handles = [
        _handle("^", True, DEFAULT_INK.axis_color, "O-point (the rejected axis)"),
        _handle("X", True, DEFAULT_INK.xpoint_color, "saddle"),
        _handle("X", False, DEFAULT_INK.xpoint_color, "other extremum"),
        plt.Line2D(
            [],
            [],
            marker="+",
            linestyle="none",
            color=FEATURE,
            markersize=11,
            label="EFIT label axis",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.50, 0.80),
        frameon=False,
        fontsize=14,
        handletextpad=0.8,
        labelspacing=0.9,
    )

    lines = ["kept critical points, local index"]
    for entry in kept:
        kind = "saddle" if entry["local_index"] < 0 else "extremum"
        star = " *" if entry is rejected else "  "
        lines.append(
            "%s %-8s R=%.4f  Z=%+.4f  index %+d"
            % (star, kind, entry["r"], entry["z"], entry["local_index"])
        )
    lines.append("")
    lines.append("kept index sum    %+d" % record["kept_index_sum"])
    lines.append("boundary winding  %.4f" % record["boundary_winding"])
    lines.append("difference        %+d" % record["winding_minus_index_sum"])
    figure.text(
        0.50,
        0.34,
        "\n".join(lines),
        family="monospace",
        fontsize=12,
        va="top",
        color="#333333",
    )
    figure.savefig(arguments.output, format="svg")
    print("null tally:", tally)
    print("wrote", arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
